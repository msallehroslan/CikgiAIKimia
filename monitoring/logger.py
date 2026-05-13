"""
monitoring/logger.py — Cikgu AI Kimia  [PRODUCTION HARDENING v4.0]
===================================================================
NEW FILE — structured JSON logging + request tracing + metrics.

LOGGING ARCHITECTURE:
  All cikgu.* loggers emit structured JSON to stdout.
  Render.com captures stdout → searchable in Render log viewer.
  Each log line is a JSON object with standard fields.

LOG SCHEMA:
  {
    "ts":       "2025-01-15T10:30:00.123Z",    // ISO timestamp
    "level":    "INFO",                         // DEBUG/INFO/WARNING/ERROR
    "logger":   "cikgu.main",                  // module logger name
    "trace_id": "abc123",                       // request trace ID
    "user_id":  12345678,                       // Telegram user ID (int)
    "event":    "solver_ok",                    // machine-readable event
    "msg":      "Solver completed ...",         // human-readable message
    // + event-specific extra fields:
    "task":     "calorimetry",
    "duration_ms": 234.5,
    "model":    "llama-3.1-8b-instant",
    "quota_used": 42,
  }

USAGE:
    from monitoring.logger import get_logger, RequestTrace

    logger = get_logger("cikgu.main")

    # Start a traced request
    with RequestTrace(user_id=12345) as trace:
        logger.info("Request started", extra=trace.context(event="request_start"))
        ...
        logger.info("Done", extra=trace.context(event="request_done", task="calorimetry"))

    # Or standalone:
    logger.info(
        "OCR complete",
        extra={"event": "ocr_ok", "user_id": uid, "provider": "groq_vision", "chars": 450}
    )

STANDARD EVENTS:
  request_start    request_done     request_error
  solver_ok        solver_error     solver_validation_fail
  ocr_start        ocr_ok           ocr_fallback         ocr_fail
  groq_ok          groq_retry       groq_timeout         groq_quota
  groq_circuit_open
  rag_ok           rag_no_context
  quota_denied     quota_ok
  cache_hit        cache_miss
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# JSON LOG FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    Each line is valid JSON — grep-able, parseable by log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base fields
        obj: Dict[str, Any] = {
            "ts":     datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Merge extra fields (set via logger.info("msg", extra={...}))
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info", "exc_text",
                "stack_info", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "message", "taskName",
            ):
                obj[key] = val

        # Exception info
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)

        try:
            return json.dumps(obj, default=str, ensure_ascii=False)
        except Exception:
            return json.dumps({"ts": obj["ts"], "level": "ERROR",
                                "msg": "log serialisation failed"})


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_configured = False


def _configure_logging() -> None:
    """Configure root cikgu.* logger with JSON formatter."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    # Apply to all cikgu.* loggers
    root = logging.getLogger("cikgu")
    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    root.addHandler(handler)
    root.propagate = False

    # Also configure the top-level app logger
    app_logger = logging.getLogger("cikgu_ai_kimia")
    app_logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    app_logger.addHandler(handler)
    app_logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Call once at module level."""
    _configure_logging()
    return logging.getLogger(name)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST TRACE CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class RequestTrace:
    """
    Context manager that generates a unique trace_id for each request.
    Use as `with RequestTrace(user_id=uid) as trace:` to automatically
    attach trace_id + user_id to all log events in the request.

    Usage:
        with RequestTrace(user_id=12345678) as trace:
            logger.info("processing", extra=trace.context(
                event="solver_start", task="calorimetry"
            ))
    """

    def __init__(
        self,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.trace_id   = uuid.uuid4().hex[:12]
        self.user_id    = user_id
        self.session_id = session_id
        self._start     = time.monotonic()

    def __enter__(self) -> "RequestTrace":
        return self

    def __exit__(self, *_) -> None:
        pass  # Context exit; caller logs completion if needed

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000

    def context(self, **extra: Any) -> Dict[str, Any]:
        """
        Return dict suitable for logger extra= kwarg.
        Merges trace_id, user_id, session_id with caller's fields.
        """
        ctx: Dict[str, Any] = {"trace_id": self.trace_id}
        if self.user_id is not None:
            ctx["user_id"] = self.user_id
        if self.session_id is not None:
            ctx["session_id"] = self.session_id
        ctx.update(extra)
        return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURED EVENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# These are thin wrappers that ensure consistent field names across the codebase.
# All events visible in Render logs can be grepped/filtered by "event" field.

_logger = get_logger("cikgu.events")


def log_ocr_start(user_id: int, trace_id: str, provider: str, image_bytes: int) -> None:
    _logger.info("OCR started", extra={
        "event": "ocr_start", "trace_id": trace_id,
        "user_id": user_id, "provider": provider, "image_bytes": image_bytes,
    })


def log_ocr_result(
    user_id: int, trace_id: str, provider: str,
    success: bool, chars: int, confidence: float, duration_ms: float,
) -> None:
    event = "ocr_ok" if success else "ocr_fail"
    _logger.info("OCR result", extra={
        "event": event, "trace_id": trace_id, "user_id": user_id,
        "provider": provider, "success": success,
        "chars": chars, "confidence": confidence, "duration_ms": round(duration_ms, 1),
    })


def log_ocr_fallback(user_id: int, trace_id: str, reason: str) -> None:
    _logger.warning("OCR fallback triggered", extra={
        "event": "ocr_fallback", "trace_id": trace_id,
        "user_id": user_id, "reason": reason,
    })


def log_solver_result(
    user_id: int, trace_id: str, task: str,
    success: bool, duration_ms: float, error: Optional[str] = None,
) -> None:
    event = "solver_ok" if success else "solver_error"
    extra: Dict[str, Any] = {
        "event": event, "trace_id": trace_id, "user_id": user_id,
        "task": task, "success": success, "duration_ms": round(duration_ms, 1),
    }
    if error:
        extra["error"] = error
    _logger.info("Solver result", extra=extra)


def log_solver_validation(
    user_id: int, trace_id: str, task: str,
    stage: str, has_critical: bool, summary: str,
) -> None:
    """stage: "pre_solve" | "post_solve" """
    level = logging.WARNING if has_critical else logging.INFO
    _logger.log(level, "Validation result", extra={
        "event": "solver_validation", "trace_id": trace_id,
        "user_id": user_id, "task": task, "stage": stage,
        "has_critical": has_critical, "summary": summary,
    })


def log_groq_call(
    trace_id: str, model: str, event: str,
    attempt: Optional[int] = None,
    duration_ms: Optional[float] = None,
    quota_used: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """event: "groq_ok" | "groq_retry" | "groq_timeout" | "groq_quota" | "groq_circuit_open" """
    extra: Dict[str, Any] = {
        "event": event, "trace_id": trace_id, "model": model,
    }
    if attempt is not None:
        extra["attempt"] = attempt
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 1)
    if quota_used is not None:
        extra["quota_used"] = quota_used
    if error:
        extra["error"] = error
    level = logging.WARNING if event != "groq_ok" else logging.DEBUG
    _logger.log(level, f"Groq call: {event}", extra=extra)


def log_rag_result(
    trace_id: str, query: str, results_count: int,
    has_context: bool, duration_ms: float,
) -> None:
    event = "rag_ok" if has_context else "rag_no_context"
    _logger.info("RAG retrieval", extra={
        "event": event, "trace_id": trace_id,
        "query_len": len(query), "results": results_count,
        "has_context": has_context, "duration_ms": round(duration_ms, 1),
    })


def log_quota_check(
    user_id: int, trace_id: str,
    check_type: str, allowed: bool, reason: Optional[str] = None,
) -> None:
    event = "quota_ok" if allowed else "quota_denied"
    extra: Dict[str, Any] = {
        "event": event, "trace_id": trace_id,
        "user_id": user_id, "check_type": check_type, "allowed": allowed,
    }
    if reason:
        extra["reason"] = reason
    level = logging.WARNING if not allowed else logging.DEBUG
    _logger.log(level, "Quota check", extra=extra)


def log_request_done(
    user_id: int, trace_id: str, task: str,
    total_ms: float, from_cache: bool = False,
) -> None:
    _logger.info("Request complete", extra={
        "event": "request_done", "trace_id": trace_id,
        "user_id": user_id, "task": task,
        "total_ms": round(total_ms, 1), "from_cache": from_cache,
    })


def log_request_error(
    user_id: int, trace_id: str, error: str, total_ms: float,
) -> None:
    _logger.error("Request failed", extra={
        "event": "request_error", "trace_id": trace_id,
        "user_id": user_id, "error": error, "total_ms": round(total_ms, 1),
    })
