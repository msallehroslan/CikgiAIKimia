"""
api/groq_client.py — Cikgu AI Kimia
=====================================
Groq-ONLY production-hardened LLM/Vision client.

Handles:
  - Async retry with exponential backoff
  - Per-model timeout control
  - Daily quota tracking (in-memory, no Redis needed)
  - Async semaphore to cap concurrency
  - Circuit breaker to stop hammering a failing API
  - Per-user cooldown (plugged in from quota_guard.py)
  - Graceful degradation: solver answer shown without explanation
    when Groq is unavailable — system NEVER dies silently

Architecture:
  call_llm()            ← explanation + theory (8b / 70b)
  call_vision()         ← Groq Vision (llama-4-scout)
  GroqClient            ← shared singleton with circuit breaker
  QuotaTracker          ← in-memory RPD counter per model
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("cikgu.groq_client")

# ── Model names ────────────────────────────────────────────────────────────
MODEL_THEORY   = os.environ.get("GROQ_MODEL",         "llama-3.3-70b-versatile")
MODEL_EXPLAIN  = os.environ.get("GROQ_EXPLAIN_MODEL", "llama-3.1-8b-instant")
MODEL_VISION   = os.environ.get("GROQ_VISION_MODEL",  "meta-llama/llama-4-scout-17b-16e-instruct")

# ── Groq free-tier limits (conservative — actual limits may differ) ────────
# llama-3.3-70b:    1,000 RPD, 30 RPM, 6,000 TPM
# llama-3.1-8b:    14,400 RPD, 30 RPM, 20,000 TPM
# llama-4-scout:    1,000 RPD, 30 RPM, 30,000 TPM
DAILY_LIMITS = {
    MODEL_THEORY:  int(os.environ.get("GROQ_RPD_THEORY",  "950")),   # 5% safety margin
    MODEL_EXPLAIN: int(os.environ.get("GROQ_RPD_EXPLAIN", "13500")),
    MODEL_VISION:  int(os.environ.get("GROQ_RPD_VISION",  "950")),
}

# Max concurrent Groq requests per model (RPM ÷ 2 = safe burst)
CONCURRENCY = {
    MODEL_THEORY:  int(os.environ.get("GROQ_CONCURRENCY_THEORY",  "3")),
    MODEL_EXPLAIN: int(os.environ.get("GROQ_CONCURRENCY_EXPLAIN", "8")),
    MODEL_VISION:  int(os.environ.get("GROQ_CONCURRENCY_VISION",  "3")),
}

# Per-call timeouts (seconds)
TIMEOUTS = {
    MODEL_THEORY:  float(os.environ.get("GROQ_TIMEOUT_THEORY",  "45")),
    MODEL_EXPLAIN: float(os.environ.get("GROQ_TIMEOUT_EXPLAIN", "20")),
    MODEL_VISION:  float(os.environ.get("GROQ_TIMEOUT_VISION",  "60")),
}

# Retry config
MAX_RETRIES     = 3
BACKOFF_BASE    = 1.5   # seconds; attempt n waits BACKOFF_BASE^n
RETRY_ON_CODES  = {429, 500, 502, 503, 504}

# Circuit breaker: open after N consecutive failures, reset after RESET seconds
CB_FAILURE_THRESHOLD = 5
CB_RESET_SECONDS     = 120


# ── Circuit Breaker ────────────────────────────────────────────────────────

class CBState(Enum):
    CLOSED   = "closed"    # normal operation
    OPEN     = "open"      # refusing requests
    HALF_OPEN = "half_open"  # testing one request


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = CB_FAILURE_THRESHOLD
    reset_seconds: float   = CB_RESET_SECONDS
    _failures: int         = field(default=0, repr=False)
    _state: CBState        = field(default=CBState.CLOSED, repr=False)
    _opened_at: float      = field(default=0.0, repr=False)

    def is_open(self) -> bool:
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at > self.reset_seconds:
                self._state = CBState.HALF_OPEN
                logger.info(f"[CB:{self.name}] → HALF_OPEN (testing)")
                return False   # allow one test request through
            return True
        return False

    def record_success(self):
        if self._state in (CBState.HALF_OPEN, CBState.OPEN):
            logger.info(f"[CB:{self.name}] → CLOSED after success")
        self._state    = CBState.CLOSED
        self._failures = 0

    def record_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            if self._state != CBState.OPEN:
                logger.warning(f"[CB:{self.name}] → OPEN after {self._failures} failures")
            self._state    = CBState.OPEN
            self._opened_at = time.monotonic()


# ── Quota Tracker ──────────────────────────────────────────────────────────

class QuotaTracker:
    """
    In-memory daily quota tracking per model.
    Resets at UTC midnight (checked on each call).
    No Redis/DB needed — approximate counts are sufficient.
    On process restart the counter resets, which is acceptable:
    we rely on Groq's own 429 as the hard floor.
    """

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._day: dict[str, str]    = {}   # model → "YYYY-MM-DD"

    def _today(self) -> str:
        import datetime
        return datetime.date.today().isoformat()

    def increment(self, model: str) -> int:
        today = self._today()
        if self._day.get(model) != today:
            self._counts[model] = 0
            self._day[model]    = today
        self._counts[model] = self._counts.get(model, 0) + 1
        return self._counts[model]

    def count(self, model: str) -> int:
        today = self._today()
        if self._day.get(model) != today:
            return 0
        return self._counts.get(model, 0)

    def is_exhausted(self, model: str) -> bool:
        limit = DAILY_LIMITS.get(model, 999999)
        return self.count(model) >= limit

    def remaining(self, model: str) -> int:
        limit = DAILY_LIMITS.get(model, 999999)
        return max(0, limit - self.count(model))


# ── Shared singletons ──────────────────────────────────────────────────────
_quota     = QuotaTracker()
_breakers: dict[str, CircuitBreaker] = {}
_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_breaker(model: str) -> CircuitBreaker:
    if model not in _breakers:
        _breakers[model] = CircuitBreaker(name=model)
    return _breakers[model]


def _get_semaphore(model: str) -> asyncio.Semaphore:
    if model not in _semaphores:
        _semaphores[model] = asyncio.Semaphore(CONCURRENCY.get(model, 3))
    return _semaphores[model]


# ── Core retry wrapper ─────────────────────────────────────────────────────

async def _call_with_retry(coro_factory, model: str) -> Optional[str]:
    """
    Execute coro_factory() with:
      1. Circuit breaker gate
      2. Quota gate
      3. Concurrency semaphore
      4. Exponential backoff retry on 429/5xx
      5. Timeout per call

    coro_factory is a zero-arg async callable that returns the response text.
    Returns None on exhausted retries — caller decides how to degrade.
    """
    cb        = _get_breaker(model)
    semaphore = _get_semaphore(model)
    timeout   = TIMEOUTS.get(model, 30.0)

    # ── Circuit breaker gate ───────────────────────────────────────────
    if cb.is_open():
        logger.warning(f"[groq] Circuit OPEN for {model} — skipping call")
        return None

    # ── Quota gate ─────────────────────────────────────────────────────
    if _quota.is_exhausted(model):
        logger.warning(f"[groq] Daily quota exhausted for {model} ({_quota.count(model)} RPD used)")
        return None

    last_exc = None
    for attempt in range(MAX_RETRIES):
        wait = (BACKOFF_BASE ** attempt) if attempt > 0 else 0
        if wait:
            logger.info(f"[groq] Retry {attempt}/{MAX_RETRIES} for {model}, wait={wait:.1f}s")
            await asyncio.sleep(wait)

        try:
            async with semaphore:
                result = await asyncio.wait_for(coro_factory(), timeout=timeout)

            cb.record_success()
            _quota.increment(model)
            logger.debug(f"[groq] OK model={model} quota={_quota.count(model)}/{DAILY_LIMITS.get(model,'?')}")
            return result

        except asyncio.TimeoutError:
            last_exc = "timeout"
            logger.warning(f"[groq] Timeout on attempt {attempt+1} model={model}")
            cb.record_failure()
            # timeout → retry immediately (not a quota issue)

        except Exception as exc:
            last_exc = str(exc)
            status = _extract_status(exc)

            if status == 429:
                # Rate limited — must wait before retry
                retry_after = _extract_retry_after(exc)
                logger.warning(f"[groq] 429 rate limit model={model}, retry_after={retry_after}s")
                await asyncio.sleep(retry_after)
                # Don't count as circuit-breaker failure (expected throttle)

            elif status in RETRY_ON_CODES:
                logger.warning(f"[groq] HTTP {status} on attempt {attempt+1} model={model}")
                cb.record_failure()

            else:
                # Auth error, bad request etc — no point retrying
                logger.error(f"[groq] Non-retryable error model={model}: {exc}")
                cb.record_failure()
                return None

    logger.error(f"[groq] All {MAX_RETRIES} attempts failed model={model}: {last_exc}")
    return None


def _extract_status(exc: Exception) -> int:
    """Extract HTTP status from groq exception if available."""
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    msg = str(exc).lower()
    for code in (429, 500, 502, 503, 504):
        if str(code) in msg:
            return code
    return 0


def _extract_retry_after(exc: Exception) -> float:
    """Extract Retry-After header value from groq 429 exception."""
    for attr in ("retry_after", "headers"):
        val = getattr(exc, attr, None)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            ra = val.get("retry-after") or val.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except (ValueError, TypeError):
                    pass
    return BACKOFF_BASE ** MAX_RETRIES   # safe default


# ── Public API ─────────────────────────────────────────────────────────────

async def call_llm(
    prompt: str,
    model: str = MODEL_EXPLAIN,
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> str:
    """
    Call Groq text completion with full hardening.
    Returns empty string on failure — caller shows solver answer alone.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.error("[groq] GROQ_API_KEY not set")
        return ""

    async def _coro():
        from groq import AsyncGroq
        client = AsyncGroq(api_key=groq_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    result = await _call_with_retry(_coro, model)
    return result or ""


async def call_vision(
    image_bytes: bytes,
    prompt: str,
    media_type: str = "image/jpeg",
    max_tokens: int = 1200,
) -> Optional[str]:
    """
    Call Groq Vision (llama-4-scout) with full hardening.
    Returns None on failure — caller triggers local OCR fallback.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.error("[groq] GROQ_API_KEY not set — cannot call vision")
        return None

    import base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    async def _coro():
        from groq import AsyncGroq
        client = AsyncGroq(api_key=groq_key)
        resp = await client.chat.completions.create(
            model=MODEL_VISION,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    result = await _call_with_retry(_coro, MODEL_VISION)
    return result   # None = caller falls back to local OCR


def quota_status() -> dict:
    """Return current quota status for all models — used by /api/health."""
    return {
        model: {
            "used":      _quota.count(model),
            "limit":     DAILY_LIMITS.get(model, "?"),
            "remaining": _quota.remaining(model),
            "exhausted": _quota.is_exhausted(model),
            "circuit":   _get_breaker(model)._state.value,
        }
        for model in [MODEL_THEORY, MODEL_EXPLAIN, MODEL_VISION]
    }
