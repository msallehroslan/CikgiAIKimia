"""
solver/dispatcher.py — Cikgu AI Kimia  [FIXED v4.1]
=====================================================
FIXED: removed broken import aliases that don't exist in solver_engine.py.

solver_engine.py already has its own complete solve_by_task().
This file is a thin wrapper that main.py imports as:
    from dispatcher import solve_by_task

The wrapper re-exports solver_engine.solve_by_task directly.
All task logic lives in solver_engine.py — DO NOT duplicate here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("cikgu.dispatcher")

# solver_engine already contains a complete solve_by_task() function.
# Re-export it directly — no need for a separate dispatch table.
from solver_engine import solve_by_task

__all__ = ["solve_by_task"]
