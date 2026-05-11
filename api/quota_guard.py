"""
api/quota_guard.py — Cikgu AI Kimia
=====================================
Per-user request throttling and OCR quota protection.

No Redis, no external services — pure in-memory with asyncio.
On Render free tier with a single process, this is sufficient.

Architecture:
  UserCooldown     — per-user minimum gap between text requests
  OCRQuota         — per-user daily OCR photo limit + global daily cap
  AsyncOCRQueue    — semaphore-backed queue for concurrent OCR jobs

Usage:
    guard = get_quota_guard()

    # Text message
    ok, wait = guard.check_text_cooldown(user_id)
    if not ok:
        await update.message.reply_text(f"Sila tunggu {wait:.0f}s.")
        return

    # Photo/OCR
    ok, reason = guard.check_ocr_allowed(user_id)
    if not ok:
        await update.message.reply_text(reason)
        return

    async with guard.ocr_slot():
        result = await extract_vision(...)

    guard.record_ocr_used(user_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional, Tuple

logger = logging.getLogger("cikgu.quota_guard")

# ── Config (all overrideable via env) ──────────────────────────────────────
import os

# Text message cooldown: minimum seconds between messages per user
TEXT_COOLDOWN_SECONDS    = float(os.environ.get("QUOTA_TEXT_COOLDOWN",  "2.0"))

# Per-user OCR (photo) limits
OCR_USER_DAILY_LIMIT     = int(os.environ.get("QUOTA_OCR_USER_DAILY",   "20"))
OCR_USER_HOURLY_LIMIT    = int(os.environ.get("QUOTA_OCR_USER_HOURLY",  "10"))

# Global OCR daily limit (protects Groq Vision RPD budget)
# Groq Vision = 950 effective RPD (see groq_client.py)
# Reserve 100 for other uses → 850 available for photo OCR
OCR_GLOBAL_DAILY_LIMIT   = int(os.environ.get("QUOTA_OCR_GLOBAL_DAILY", "800"))

# Max concurrent OCR jobs (each Groq vision call takes 3–10s)
OCR_MAX_CONCURRENT       = int(os.environ.get("QUOTA_OCR_CONCURRENT",   "3"))

# Message queue: max queued messages per user before we drop
MESSAGE_QUEUE_MAX        = int(os.environ.get("QUOTA_MSG_QUEUE_MAX",    "3"))


# ── Internal state stores ──────────────────────────────────────────────────

class _UserCooldownStore:
    """Track last request timestamp per user."""

    def __init__(self):
        self._last: dict[int, float] = {}

    def check(self, user_id: int, cooldown_sec: float) -> Tuple[bool, float]:
        """
        Returns (allowed, wait_seconds).
        allowed=True if user can send now.
        """
        now  = time.monotonic()
        last = self._last.get(user_id, 0.0)
        gap  = now - last
        if gap >= cooldown_sec:
            return True, 0.0
        return False, round(cooldown_sec - gap, 1)

    def record(self, user_id: int):
        self._last[user_id] = time.monotonic()


class _OCRUsageStore:
    """
    Per-user and global daily/hourly OCR usage counters.
    Resets at UTC midnight (daily) or after 3600s (hourly window).
    """

    def __init__(self):
        # { user_id: [(timestamp, 1), ...] }  — rolling window approach
        self._events: dict[int, list[float]] = defaultdict(list)
        self._global_events: list[float]     = []

    def _now(self) -> float:
        return time.time()

    def _trim(self, events: list[float], window_sec: float) -> list[float]:
        cutoff = self._now() - window_sec
        return [t for t in events if t > cutoff]

    def count_user_hourly(self, user_id: int) -> int:
        self._events[user_id] = self._trim(self._events[user_id], 3600)
        return len(self._events[user_id])

    def count_user_daily(self, user_id: int) -> int:
        self._events[user_id] = self._trim(self._events[user_id], 86400)
        return len(self._events[user_id])

    def count_global_daily(self) -> int:
        self._global_events = self._trim(self._global_events, 86400)
        return len(self._global_events)

    def record(self, user_id: int):
        now = self._now()
        self._events[user_id].append(now)
        self._global_events.append(now)

    def check_allowed(
        self,
        user_id: int,
        user_daily_limit: int,
        user_hourly_limit: int,
        global_daily_limit: int,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason_if_denied).
        """
        global_count = self.count_global_daily()
        if global_count >= global_daily_limit:
            logger.warning(f"[quota] Global OCR daily limit hit: {global_count}")
            return False, "global_daily_limit"

        user_hourly = self.count_user_hourly(user_id)
        if user_hourly >= user_hourly_limit:
            return False, f"user_hourly_limit:{user_hourly}/{user_hourly_limit}"

        user_daily = self.count_user_daily(user_id)
        if user_daily >= user_daily_limit:
            return False, f"user_daily_limit:{user_daily}/{user_daily_limit}"

        return True, ""


class _MessageQueueGuard:
    """
    Track per-user pending message count to prevent queue flooding.
    Uses a simple in-memory counter, not a real queue.
    """

    def __init__(self):
        self._pending: dict[int, int] = defaultdict(int)

    def increment(self, user_id: int) -> int:
        self._pending[user_id] += 1
        return self._pending[user_id]

    def decrement(self, user_id: int):
        if self._pending[user_id] > 0:
            self._pending[user_id] -= 1

    def is_full(self, user_id: int) -> bool:
        return self._pending[user_id] >= MESSAGE_QUEUE_MAX


# ── Main quota guard ───────────────────────────────────────────────────────

class QuotaGuard:

    def __init__(self):
        self._text_cooldown  = _UserCooldownStore()
        self._ocr_usage      = _OCRUsageStore()
        self._msg_queue      = _MessageQueueGuard()
        self._ocr_semaphore  = asyncio.Semaphore(OCR_MAX_CONCURRENT)

    # ── Text message throttle ──────────────────────────────────────────

    def check_text_cooldown(self, user_id: int) -> Tuple[bool, float]:
        """
        Returns (allowed, wait_seconds).
        Call this at the start of handle_message().
        """
        return self._text_cooldown.check(user_id, TEXT_COOLDOWN_SECONDS)

    def record_text_request(self, user_id: int):
        self._text_cooldown.record(user_id)

    # ── Message queue guard ────────────────────────────────────────────

    def check_queue_full(self, user_id: int) -> bool:
        """Returns True if user has too many pending messages."""
        return self._msg_queue.is_full(user_id)

    def enter_processing(self, user_id: int) -> int:
        return self._msg_queue.increment(user_id)

    def exit_processing(self, user_id: int):
        self._msg_queue.decrement(user_id)

    # ── OCR / Vision throttle ──────────────────────────────────────────

    def check_ocr_allowed(self, user_id: int) -> Tuple[bool, str]:
        """
        Returns (allowed, denial_reason_or_empty_string).
        Call this before triggering vision pipeline.
        """
        return self._ocr_usage.check_allowed(
            user_id,
            user_daily_limit=OCR_USER_DAILY_LIMIT,
            user_hourly_limit=OCR_USER_HOURLY_LIMIT,
            global_daily_limit=OCR_GLOBAL_DAILY_LIMIT,
        )

    def record_ocr_used(self, user_id: int):
        """Call this after a successful OCR request (Groq or local)."""
        self._ocr_usage.record(user_id)
        hourly = self._ocr_usage.count_user_hourly(user_id)
        daily  = self._ocr_usage.count_global_daily()
        logger.info(f"[quota] OCR user={user_id} hourly={hourly} global_daily={daily}")

    @asynccontextmanager
    async def ocr_slot(self):
        """
        Async context manager that limits concurrent OCR jobs.
        Usage:
            async with guard.ocr_slot():
                result = await do_ocr(...)
        """
        async with self._ocr_semaphore:
            yield

    # ── Status / diagnostics ───────────────────────────────────────────

    def status(self) -> dict:
        """Return quota status for /api/health endpoint."""
        return {
            "ocr_global_daily": {
                "used":  self._ocr_usage.count_global_daily(),
                "limit": OCR_GLOBAL_DAILY_LIMIT,
            },
            "ocr_concurrency_max": OCR_MAX_CONCURRENT,
            "text_cooldown_sec":   TEXT_COOLDOWN_SECONDS,
            "ocr_user_daily_limit": OCR_USER_DAILY_LIMIT,
            "ocr_user_hourly_limit": OCR_USER_HOURLY_LIMIT,
        }


# ── User-facing denial messages ────────────────────────────────────────────

def denial_message(reason: str, lang: str = "BM") -> str:
    if lang == "BM":
        messages = {
            "text_cooldown":
                "⏳ Sila tunggu sebentar sebelum menghantar soalan lain.",
            "queue_full":
                "⏳ Cikgu AI sedang sibuk. Sila hantar soalan semula dalam sebentar.",
            "user_hourly_limit":
                "⚠️ Had gambar sejam telah dicapai. Cuba lagi dalam satu jam.",
            "user_daily_limit":
                "⚠️ Had gambar harian telah dicapai. Cuba lagi esok.",
            "global_daily_limit":
                "⚠️ Sistem sedang sibuk. Sila taip soalan dalam teks untuk sekarang.",
        }
    else:
        messages = {
            "text_cooldown":
                "⏳ Please wait a moment before sending another question.",
            "queue_full":
                "⏳ Cikgu AI is busy. Please resend in a moment.",
            "user_hourly_limit":
                "⚠️ Hourly photo limit reached. Try again in an hour.",
            "user_daily_limit":
                "⚠️ Daily photo limit reached. Try again tomorrow.",
            "global_daily_limit":
                "⚠️ System is busy. Please type your question as text for now.",
        }
    # Match by prefix since reason may include "user_hourly_limit:5/10"
    for key, msg in messages.items():
        if reason.startswith(key):
            return msg
    return messages.get("global_daily_limit", "⚠️ Limit reached. Try again later.")


# ── Singleton ──────────────────────────────────────────────────────────────

_guard: Optional[QuotaGuard] = None


def get_quota_guard() -> QuotaGuard:
    global _guard
    if _guard is None:
        _guard = QuotaGuard()
    return _guard
