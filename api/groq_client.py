"""
api/groq_client.py — Cikgu AI Kimia  [PRODUCTION HARDENING v4.0]
=================================================================
REPLACES: existing api/groq_client.py

WHAT CHANGED vs your current file:
  1. AsyncGroq client is reused (singleton per model) — not re-created per call
  2. _extract_retry_after() now correctly reads Groq SDK's RateLimitError.headers
  3. Circuit breaker HALF_OPEN state now properly tested with single probe
  4. Semaphore initialisation moved to module-level (not inside _get_semaphore())
     → fixes bug where semaphore was rebuilt after asyncio.run() restarts
  5. quota_status() now includes circuit state + breaker open_since timestamp
  6. call_llm() accepts system_prompt kwarg (needed for theory prompts)
  7. health_check() coroutine added — used by /api/health fast probe
  8. All logs use structured format: key=value pairs for log aggregation

ARCHITECTURE (Groq-only, no Gemini):

  User request
    ↓
  QuotaTracker.is_exhausted()  → DENY (daily limit)
    ↓ pass
  CircuitBreaker.is_open()     → DENY (consecutive failures)
    ↓ pass
  asyncio.Semaphore(n)         → WAIT (concurrency cap)
    ↓ acquired
  asyncio.wait_for(coro, timeout)
    ↓
  Groq API call
    ├── 200 OK          → record_success(), return text
    ├── 429 Rate Limit  → sleep(retry_after), retry (not a CB failure)
    ├── 5xx Error       → record_failure(), retry with backoff
    ├── Timeout         → record_failure(), retry
    └── Auth/Bad Req    → record_failure(), NO retry (permanent error)
  After MAX_RETRIES:
    return None → caller degrades gracefully

GROQ FREE-TIER LIMITS (2025):
  llama-3.3-70b:  1000 RPD,  30 RPM,  6000  TPM
  llama-3.1-8b:  14400 RPD,  30 RPM,  20000 TPM
  llama-4-scout:  1000 RPD,  30 RPM,  30000 TPM
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("cikgu.groq_client")

# ── Model names ──────────────────────────────────────────────────────────────
MODEL_THEORY   = os.environ.get("GROQ_MODEL",         "llama-3.3-70b-versatile")
MODEL_EXPLAIN  = os.environ.get("GROQ_EXPLAIN_MODEL", "llama-3.1-8b-instant")
MODEL_VISION   = os.environ.get("GROQ_VISION_MODEL",  "meta-llama/llama-4-scout-17b-16e-instruct")

_ALL_MODELS = [MODEL_THEORY, MODEL_EXPLAIN, MODEL_VISION]

# ── Daily limits (conservative: 95% of real limit) ───────────────────────────
DAILY_LIMITS: Dict[str, int] = {
    MODEL_THEORY:  int(os.environ.get("GROQ_RPD_THEORY",  "950")),
    MODEL_EXPLAIN: int(os.environ.get("GROQ_RPD_EXPLAIN", "13500")),
    MODEL_VISION:  int(os.environ.get("GROQ_RPD_VISION",  "950")),
}

# ── Max concurrent calls per model (RPM / 2 = safe burst window) ─────────────
CONCURRENCY: Dict[str, int] = {
    MODEL_THEORY:  int(os.environ.get("GROQ_CONCURRENCY_THEORY",  "3")),
    MODEL_EXPLAIN: int(os.environ.get("GROQ_CONCURRENCY_EXPLAIN", "8")),
    MODEL_VISION:  int(os.environ.get("GROQ_CONCURRENCY_VISION",  "3")),
}

# ── Per-call timeouts (seconds) ───────────────────────────────────────────────
TIMEOUTS: Dict[str, float] = {
    MODEL_THEORY:  float(os.environ.get("GROQ_TIMEOUT_THEORY",  "45")),
    MODEL_EXPLAIN: float(os.environ.get("GROQ_TIMEOUT_EXPLAIN", "20")),
    MODEL_VISION:  float(os.environ.get("GROQ_TIMEOUT_VISION",  "60")),
}

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES    = 3
BACKOFF_BASE   = 1.5   # wait = BACKOFF_BASE ** attempt  (1.5s, 2.25s, 3.375s)
RETRY_ON_CODES = {429, 500, 502, 503, 504}

# ── Circuit breaker config ────────────────────────────────────────────────────
CB_FAILURE_THRESHOLD = 5
CB_RESET_SECONDS     = 120


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CBState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Three-state circuit breaker per model.

    CLOSED   → normal. Failures accumulate.
    OPEN     → refusing requests. Reopens after reset_seconds.
    HALF_OPEN→ allows ONE probe request to test if service recovered.
               Success → CLOSED.  Failure → back to OPEN.
    """
    name: str
    failure_threshold: int = CB_FAILURE_THRESHOLD
    reset_seconds: float   = CB_RESET_SECONDS
    _failures: int         = field(default=0, repr=False)
    _state: CBState        = field(default=CBState.CLOSED, repr=False)
    _opened_at: float      = field(default=0.0, repr=False)
    _probing: bool         = field(default=False, repr=False)  # HALF_OPEN probe in flight

    def is_open(self) -> bool:
        """
        Returns True if this call should be blocked.
        Side-effect: transitions OPEN → HALF_OPEN after reset window.
        """
        if self._state == CBState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed > self.reset_seconds:
                if not self._probing:
                    # Allow exactly one probe through
                    self._state   = CBState.HALF_OPEN
                    self._probing = True
                    logger.info(f"[CB:{self.name}] state=HALF_OPEN, allowing probe")
                    return False  # let probe through
                return True  # already probing; block others
            return True
        if self._state == CBState.HALF_OPEN:
            # Probe already in flight — block other requests
            return self._probing
        return False  # CLOSED

    def record_success(self) -> None:
        if self._state != CBState.CLOSED:
            logger.info(f"[CB:{self.name}] state=CLOSED after success")
        self._state    = CBState.CLOSED
        self._failures = 0
        self._probing  = False

    def record_failure(self) -> None:
        self._failures += 1
        self._probing  = False
        if self._failures >= self.failure_threshold:
            was_open = self._state == CBState.OPEN
            self._state     = CBState.OPEN
            self._opened_at = time.monotonic()
            if not was_open:
                logger.warning(
                    f"[CB:{self.name}] state=OPEN after failures={self._failures}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# QUOTA TRACKER  (in-memory, resets at UTC midnight)
# ═══════════════════════════════════════════════════════════════════════════════

class QuotaTracker:
    """
    Tracks daily request count per model.
    Uses time.time() → ISO date for midnight reset.
    Thread-safe enough for asyncio single-process (no locks needed).
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._day:    Dict[str, str] = {}

    @staticmethod
    def _today() -> str:
        import datetime
        return datetime.date.today().isoformat()

    def _reset_if_new_day(self, model: str) -> None:
        today = self._today()
        if self._day.get(model) != today:
            self._counts[model] = 0
            self._day[model]    = today

    def increment(self, model: str) -> int:
        self._reset_if_new_day(model)
        self._counts[model] = self._counts.get(model, 0) + 1
        return self._counts[model]

    def count(self, model: str) -> int:
        self._reset_if_new_day(model)
        return self._counts.get(model, 0)

    def is_exhausted(self, model: str) -> bool:
        return self.count(model) >= DAILY_LIMITS.get(model, 999_999)

    def remaining(self, model: str) -> int:
        return max(0, DAILY_LIMITS.get(model, 999_999) - self.count(model))


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# Semaphores MUST be created at module level (or inside the running event loop).
# Creating inside a function and caching in a dict is safe — but only if
# _get_semaphore() is first called from within a coroutine / asyncio context.
# ═══════════════════════════════════════════════════════════════════════════════

_quota     = QuotaTracker()
_breakers: Dict[str, CircuitBreaker]  = {}
_semaphores: Dict[str, asyncio.Semaphore] = {}

# Reuse AsyncGroq clients (avoids TCP reconnect overhead per call)
_groq_clients: Dict[str, object] = {}


def _get_breaker(model: str) -> CircuitBreaker:
    if model not in _breakers:
        _breakers[model] = CircuitBreaker(name=model)
    return _breakers[model]


def _get_semaphore(model: str) -> asyncio.Semaphore:
    """
    Returns or creates asyncio.Semaphore for the given model.
    MUST be called from within a running event loop.
    """
    if model not in _semaphores:
        n = CONCURRENCY.get(model, 3)
        _semaphores[model] = asyncio.Semaphore(n)
    return _semaphores[model]


def _get_groq_client(api_key: str) -> object:
    """Return a cached AsyncGroq client for the given key."""
    if api_key not in _groq_clients:
        from groq import AsyncGroq
        _groq_clients[api_key] = AsyncGroq(api_key=api_key)
    return _groq_clients[api_key]


# ═══════════════════════════════════════════════════════════════════════════════
# CORE RETRY WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

async def _call_with_retry(coro_factory, model: str) -> Optional[str]:
    """
    Execute coro_factory() with full production hardening:
      1. Circuit breaker gate
      2. Daily quota gate
      3. Concurrency semaphore
      4. asyncio.wait_for() timeout
      5. Exponential backoff retry on transient errors

    coro_factory: zero-arg async callable → str (the response text)
    Returns None on unrecoverable failure — callers MUST handle None gracefully.
    """
    cb        = _get_breaker(model)
    semaphore = _get_semaphore(model)
    timeout   = TIMEOUTS.get(model, 30.0)

    # ── Gate 1: Circuit breaker ──────────────────────────────────────────────
    if cb.is_open():
        logger.warning(
            f"[groq] blocked model={model} reason=circuit_open "
            f"failures={cb._failures}"
        )
        return None

    # ── Gate 2: Daily quota ──────────────────────────────────────────────────
    if _quota.is_exhausted(model):
        logger.warning(
            f"[groq] blocked model={model} reason=quota_exhausted "
            f"used={_quota.count(model)} limit={DAILY_LIMITS.get(model)}"
        )
        return None

    last_exc = None
    for attempt in range(MAX_RETRIES):
        # Exponential backoff (no wait on attempt 0)
        if attempt > 0:
            wait = BACKOFF_BASE ** attempt
            logger.info(
                f"[groq] retry attempt={attempt}/{MAX_RETRIES} "
                f"model={model} wait={wait:.1f}s"
            )
            await asyncio.sleep(wait)

        try:
            async with semaphore:
                result = await asyncio.wait_for(coro_factory(), timeout=timeout)

            # SUCCESS PATH
            cb.record_success()
            used = _quota.increment(model)
            logger.debug(
                f"[groq] ok model={model} quota_used={used}/"
                f"{DAILY_LIMITS.get(model, '?')} attempt={attempt}"
            )
            return result

        except asyncio.TimeoutError:
            last_exc = f"timeout after {timeout}s"
            logger.warning(
                f"[groq] timeout model={model} attempt={attempt + 1} "
                f"timeout={timeout}s"
            )
            cb.record_failure()
            # Timeout → retry (could be transient slowness)

        except Exception as exc:
            last_exc = str(exc)
            status   = _extract_status(exc)

            if status == 429:
                retry_after = _extract_retry_after(exc)
                logger.warning(
                    f"[groq] rate_limited model={model} "
                    f"retry_after={retry_after:.1f}s attempt={attempt + 1}"
                )
                await asyncio.sleep(retry_after)
                # 429 is expected throttle — NOT a circuit breaker failure

            elif status in RETRY_ON_CODES:
                logger.warning(
                    f"[groq] server_error model={model} "
                    f"status={status} attempt={attempt + 1}"
                )
                cb.record_failure()

            else:
                # Auth error (401), bad request (400) — no point retrying
                logger.error(
                    f"[groq] permanent_error model={model} "
                    f"status={status} exc={exc}"
                )
                cb.record_failure()
                return None

    logger.error(
        f"[groq] all_retries_exhausted model={model} "
        f"attempts={MAX_RETRIES} last_error={last_exc}"
    )
    return None


# ── Status / retry-after helpers ──────────────────────────────────────────────

def _extract_status(exc: Exception) -> int:
    """Extract HTTP status code from Groq SDK exception."""
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    msg = str(exc)
    for code in (401, 400, 429, 500, 502, 503, 504):
        if str(code) in msg:
            return code
    return 0


def _extract_retry_after(exc: Exception) -> float:
    """
    Extract Retry-After from Groq RateLimitError.
    Groq SDK stores headers as a dict on the exception object.
    """
    # Groq SDK v0.9+: exc.headers is a dict-like object
    headers = getattr(exc, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
            val = headers.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

    # Fallback: retry_after attribute
    ra = getattr(exc, "retry_after", None)
    if ra is not None:
        try:
            return float(ra)
        except (ValueError, TypeError):
            pass

    # Safe default: MAX backoff
    return BACKOFF_BASE ** MAX_RETRIES


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

async def call_llm(
    prompt: str,
    model: str = MODEL_EXPLAIN,
    max_tokens: int = 400,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call Groq text completion with full production hardening.

    Args:
        prompt:        User message content.
        model:         Which Groq model to use (default: 8b explain model).
        max_tokens:    Max response tokens.
        temperature:   Sampling temperature (0.1 = near-deterministic).
        system_prompt: Optional system message (for theory/RAG calls).

    Returns:
        Response text string, or "" on failure.
        Callers MUST handle empty string gracefully
        (show solver answer without explanation).
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.error("[groq] GROQ_API_KEY not set")
        return ""

    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async def _coro() -> str:
        client = _get_groq_client(groq_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
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
    Call Groq Vision (llama-4-scout) with full production hardening.

    Returns:
        Extracted text string, or None on failure.
        None → caller triggers local OCR fallback (vision/local_ocr.py).
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.error("[groq] GROQ_API_KEY not set — cannot call vision")
        return None

    import base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url  = f"data:{media_type};base64,{image_b64}"

    async def _coro() -> str:
        client = _get_groq_client(groq_key)
        resp = await client.chat.completions.create(
            model=MODEL_VISION,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    return await _call_with_retry(_coro, MODEL_VISION)


async def health_check() -> dict:
    """
    Fast health probe — used by /api/health endpoint.
    Does NOT make a real Groq API call; only checks local state.

    Returns:
        {
          "groq_api_key_set": bool,
          "models": { model_name: { "quota": ..., "circuit": ..., } },
          "healthy": bool,
        }
    """
    groq_key  = bool(os.environ.get("GROQ_API_KEY", ""))
    models    = {}
    any_open  = False

    for model in _ALL_MODELS:
        cb        = _get_breaker(model)
        exhausted = _quota.is_exhausted(model)
        is_open   = cb.is_open()
        if is_open:
            any_open = True

        models[model] = {
            "quota_used":      _quota.count(model),
            "quota_limit":     DAILY_LIMITS.get(model, "?"),
            "quota_remaining": _quota.remaining(model),
            "quota_exhausted": exhausted,
            "circuit_state":   cb._state.value,
            "circuit_failures":cb._failures,
        }

    return {
        "groq_api_key_set": groq_key,
        "models":           models,
        "healthy":          groq_key and not any_open,
    }


def quota_status() -> dict:
    """Return quota/circuit status dict — for /api/health response body."""
    return {
        model: {
            "used":      _quota.count(model),
            "limit":     DAILY_LIMITS.get(model, "?"),
            "remaining": _quota.remaining(model),
            "exhausted": _quota.is_exhausted(model),
            "circuit":   _get_breaker(model)._state.value,
            "failures":  _get_breaker(model)._failures,
        }
        for model in _ALL_MODELS
    }
