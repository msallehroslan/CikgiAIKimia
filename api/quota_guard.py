"""
api/quota_guard.py — Cikgu AI Kimia  [PRODUCTION HARDENING v4.0]
=================================================================
REPLACES: existing api/quota_guard.py  (adds per-user text cooldown fix,
          message queue guard, and monitoring hooks)

Per-user request throttling and OCR quota protection.
No Redis, no external services — pure asyncio in-memory.

WHY THIS IS ENOUGH:
  Render free tier runs a SINGLE Python process.
  asyncio is single-threaded. No race conditions on in-memory state.
  On process restart quotas reset — acceptable: Groq's own 429 is
  the hard floor if we over-reset.

PROTECTION LAYERS:
  1. UserCooldown      — minimum gap between messages per user (spam)
  2. MessageQueueGuard — max N requests pending per user (flood)
  3. OCRUsage          — per-user hourly + daily OCR limit
  4. GlobalOCR         — system-wide daily OCR cap (protects Groq RPD)
  5. AsyncOCRSemaphore — max N concurrent OCR jobs (concurrent cap)

HOW TO INTEGRATE IN telegram_bot.py:

    guard = get_quota_guard()

    # TEXT message handler:
    allowed, wait = guard.check_text_cooldown(user_id)
    if not allowed:
        await msg.reply_text(denial_message("text_cooldown"), parse_mode=HTML)
        return
    if guard.check_queue_full(user_id):
        await msg.reply_text(denial_message("queue_full"), parse_mode=HTML)
        return
    guard.record_text_request(user_id)
    guard.enter_processing(user_id)
    try:
        ... process ...
    finally:
        guard.exit_processing(user_id)

    # PHOTO / OCR handler:
    allowed, reason = guard.check_ocr_allowed(user_id)
    if not allowed:
        await msg.reply_text(denial_message(reason), parse_mode=HTML)
        return
    async with guard.ocr_slot():
        result = await extract_question_from_image(image_bytes)
    guard.record_ocr_used(user_id)

ENV VARS (all optional — sensible defaults for Render free tier):
  QUOTA_TEXT_COOLDOWN    = seconds between text messages (default: 2.0)
  QUOTA_OCR_USER_DAILY   = per-user daily OCR limit (default: 20)
  QUOTA_OCR_USER_HOURLY  = per-user hourly OCR limit (default: 10)
  QUOTA_OCR_GLOBAL_DAILY = system daily OCR cap (default: 800)
  QUOTA_OCR_CONCURRENT   = max parallel OCR jobs (default: 3)
  QUOTA_MSG_QUEUE_MAX    = max pending messages per user (default: 3)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("cikgu.quota_guard")

# ── Config ───────────────────────────────────────────────────────────────────
TEXT_COOLDOWN_SECONDS  = float(os.environ.get("QUOTA_TEXT_COOLDOWN",  "2.0"))
OCR_USER_DAILY_LIMIT   = int(os.environ.get("QUOTA_OCR_USER_DAILY",   "20"))
OCR_USER_HOURLY_LIMIT  = int(os.environ.get("QUOTA_OCR_USER_HOURLY",  "10"))
# Groq Vision = 950 effective RPD; reserve 150 for retries → 800 for OCR
OCR_GLOBAL_DAILY_LIMIT = int(os.environ.get("QUOTA_OCR_GLOBAL_DAILY", "800"))
OCR_MAX_CONCURRENT     = int(os.environ.get("QUOTA_OCR_CONCURRENT",   "3"))
MESSAGE_QUEUE_MAX      = int(os.environ.get("QUOTA_MSG_QUEUE_MAX",    "3"))


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL STATE STORES
# ═══════════════════════════════════════════════════════════════════════════════

class _UserCooldownStore:
    """Track last-request timestamp per user."""

    def __init__(self) -> None:
        self._last: Dict[int, float] = {}

    def check(self, user_id: int, cooldown_sec: float) -> Tuple[bool, float]:
        """
        Returns (allowed, wait_seconds).
        allowed=True if user can send right now.
        """
        now  = time.monotonic()
        last = self._last.get(user_id, 0.0)
        gap  = now - last
        if gap >= cooldown_sec:
            return True, 0.0
        return False, round(cooldown_sec - gap, 1)

    def record(self, user_id: int) -> None:
        self._last[user_id] = time.monotonic()


class _OCRUsageStore:
    """
    Per-user and global rolling-window event counters.

    Uses time.time() timestamps in a list per user.
    _trim() removes events outside the window on each read.
    This is O(n) per check — acceptable for low-traffic free tier.
    For high traffic, replace with a deque + periodic GC.
    """

    def __init__(self) -> None:
        self._events: Dict[int, List[float]] = defaultdict(list)
        self._global: List[float]            = []

    def _now(self) -> float:
        return time.time()

    @staticmethod
    def _trim(events: List[float], window_sec: float, now: float) -> List[float]:
        cutoff = now - window_sec
        return [t for t in events if t > cutoff]

    def count_user_hourly(self, user_id: int) -> int:
        now = self._now()
        self._events[user_id] = self._trim(self._events[user_id], 3600, now)
        return len(self._events[user_id])

    def count_user_daily(self, user_id: int) -> int:
        now = self._now()
        self._events[user_id] = self._trim(self._events[user_id], 86400, now)
        return len(self._events[user_id])

    def count_global_daily(self) -> int:
        now = self._now()
        self._global = self._trim(self._global, 86400, now)
        return len(self._global)

    def record(self, user_id: int) -> None:
        now = self._now()
        self._events[user_id].append(now)
        self._global.append(now)

    def check_allowed(
        self,
        user_id: int,
        user_daily_limit: int,
        user_hourly_limit: int,
        global_daily_limit: int,
    ) -> Tuple[bool, str]:
        """Returns (allowed, denial_reason_or_empty_string)."""
        global_count = self.count_global_daily()
        if global_count >= global_daily_limit:
            logger.warning(
                f"[quota] global_daily_limit_hit "
                f"count={global_count} limit={global_daily_limit}"
            )
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
    A user with MESSAGE_QUEUE_MAX requests already in-flight gets rejected.
    """

    def __init__(self) -> None:
        self._pending: Dict[int, int] = defaultdict(int)

    def increment(self, user_id: int) -> int:
        self._pending[user_id] += 1
        return self._pending[user_id]

    def decrement(self, user_id: int) -> None:
        if self._pending[user_id] > 0:
            self._pending[user_id] -= 1

    def is_full(self, user_id: int) -> bool:
        return self._pending[user_id] >= MESSAGE_QUEUE_MAX

    def pending_count(self, user_id: int) -> int:
        return self._pending[user_id]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QUOTA GUARD
# ═══════════════════════════════════════════════════════════════════════════════

class QuotaGuard:

    def __init__(self) -> None:
        self._text_cooldown = _UserCooldownStore()
        self._ocr_usage     = _OCRUsageStore()
        self._msg_queue     = _MessageQueueGuard()
        self._ocr_semaphore = asyncio.Semaphore(OCR_MAX_CONCURRENT)

    # ── Text message throttle ─────────────────────────────────────────────────

    def check_text_cooldown(self, user_id: int) -> Tuple[bool, float]:
        """Returns (allowed, wait_seconds). Call at start of text handler."""
        return self._text_cooldown.check(user_id, TEXT_COOLDOWN_SECONDS)

    def record_text_request(self, user_id: int) -> None:
        """Call after cooldown check passes, before processing."""
        self._text_cooldown.record(user_id)

    # ── Message queue guard ───────────────────────────────────────────────────

    def check_queue_full(self, user_id: int) -> bool:
        """Returns True if user already has too many pending requests."""
        return self._msg_queue.is_full(user_id)

    def enter_processing(self, user_id: int) -> int:
        """Call when starting to process a message. Returns pending count."""
        return self._msg_queue.increment(user_id)

    def exit_processing(self, user_id: int) -> None:
        """Call in finally block after processing completes."""
        self._msg_queue.decrement(user_id)

    # ── OCR / Vision throttle ─────────────────────────────────────────────────

    def check_ocr_allowed(self, user_id: int) -> Tuple[bool, str]:
        """
        Returns (allowed, denial_reason_or_empty).
        Call before triggering vision pipeline.
        """
        return self._ocr_usage.check_allowed(
            user_id,
            user_daily_limit=OCR_USER_DAILY_LIMIT,
            user_hourly_limit=OCR_USER_HOURLY_LIMIT,
            global_daily_limit=OCR_GLOBAL_DAILY_LIMIT,
        )

    def record_ocr_used(self, user_id: int) -> None:
        """Call after a successful OCR request (Groq Vision or local OCR)."""
        self._ocr_usage.record(user_id)
        hourly = self._ocr_usage.count_user_hourly(user_id)
        daily  = self._ocr_usage.count_global_daily()
        logger.info(
            f"[quota] ocr_recorded "
            f"user_id={user_id} user_hourly={hourly} global_daily={daily}"
        )

    @asynccontextmanager
    async def ocr_slot(self):
        """
        Async context manager that limits concurrent OCR jobs.
        Prevents N simultaneous Groq Vision calls from hammering the API.

        Usage:
            async with guard.ocr_slot():
                result = await extract_question_from_image(...)
        """
        async with self._ocr_semaphore:
            yield

    # ── Status / diagnostics ──────────────────────────────────────────────────

    def status(self) -> dict:
        """Return quota status for /api/health endpoint."""
        return {
            "ocr_global_daily": {
                "used":  self._ocr_usage.count_global_daily(),
                "limit": OCR_GLOBAL_DAILY_LIMIT,
                "remaining": max(
                    0, OCR_GLOBAL_DAILY_LIMIT - self._ocr_usage.count_global_daily()
                ),
            },
            "config": {
                "text_cooldown_sec":    TEXT_COOLDOWN_SECONDS,
                "ocr_user_daily_limit": OCR_USER_DAILY_LIMIT,
                "ocr_user_hourly_limit":OCR_USER_HOURLY_LIMIT,
                "ocr_concurrent_max":   OCR_MAX_CONCURRENT,
                "msg_queue_max":        MESSAGE_QUEUE_MAX,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# USER-FACING DENIAL MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════

def denial_message(reason: str, lang: str = "BM") -> str:
    """Return HTML-safe user-facing message for quota denial reasons."""
    if lang == "BM":
        messages = {
            "text_cooldown":
                "⏳ Sila tunggu sebentar sebelum menghantar soalan lain.",
            "queue_full":
                "⏳ Cikgu AI sedang sibuk memproses soalan anda yang lain. "
                "Sila cuba sebentar lagi.",
            "user_hourly_limit":
                "⚠️ Had gambar sejam telah dicapai (10 gambar/jam). "
                "Cuba lagi dalam satu jam.",
            "user_daily_limit":
                "⚠️ Had gambar harian telah dicapai (20 gambar/hari). "
                "Cuba lagi esok.",
            "global_daily_limit":
                "⚠️ Sistem sedang sibuk. Sila taip soalan anda dalam teks "
                "untuk sekarang.",
        }
    else:
        messages = {
            "text_cooldown":
                "⏳ Please wait a moment before sending another question.",
            "queue_full":
                "⏳ Cikgu AI is busy. Please try again in a moment.",
            "user_hourly_limit":
                "⚠️ Hourly photo limit reached (10 photos/hour). "
                "Try again in an hour.",
            "user_daily_limit":
                "⚠️ Daily photo limit reached (20 photos/day). "
                "Try again tomorrow.",
            "global_daily_limit":
                "⚠️ System is busy. Please type your question as text for now.",
        }
    # Match by prefix (reason may include "user_hourly_limit:5/10")
    for key, msg in messages.items():
        if reason.startswith(key):
            return msg
    return messages.get("global_daily_limit", "⚠️ Limit reached. Try again later.")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_guard: Optional[QuotaGuard] = None


def get_quota_guard() -> QuotaGuard:
    """Return or create the singleton QuotaGuard instance."""
    global _guard
    if _guard is None:
        _guard = QuotaGuard()
    return _guard
