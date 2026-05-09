"""
memory.py — Cikgu AI Kimia Smart Memory System
===============================================
Two-layer memory architecture:

LAYER 1: SHARED CACHE (qa_cache collection)
  - Caches Q&A pairs shared across ALL students
  - Same question = instant answer, no API call
  - Bot gets smarter with every question asked
  - Reduces Groq API calls by up to 70%

LAYER 2: PERSONAL MEMORY (sessions collection)
  - Per-student conversation history
  - Remembers "soalan saya tadi..."
  - Context-aware follow-up answers
  - Survives bot restarts

Firestore Collections:
  qa_cache/
    {question_hash}/
      question, answer, answer_type, task,
      hit_count, created_at, updated_at
  sessions/
    {session_id}/
      user_name, updated_at
      messages/
        {msg_id}/
          role, content, timestamp

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import os
import time
import json
import hashlib
from typing import List, Dict, Optional, Tuple

import firebase_admin
from firebase_admin import credentials, firestore

# ---------------------------------------------------------------------------
# FIREBASE INIT
# ---------------------------------------------------------------------------

_db = None

def _get_db():
    global _db
    if _db is None:
        if not firebase_admin._apps:
            # Read from your Render env var name
            cred_json = (
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "") or
                os.environ.get("FIREBASE_CREDENTIALS", "") or
                os.environ.get("FIREBASE_CREDENTIALS_PATH", "")
            )
            if cred_json:
                try:
                    cred_dict = json.loads(cred_json)
                    cred = credentials.Certificate(cred_dict)
                except Exception:
                    cred = credentials.Certificate(cred_json)
            else:
                raise ValueError(
                    "No Firebase credentials found. "
                    "Set GOOGLE_APPLICATION_CREDENTIALS_JSON in Render environment."
                )
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
    return _db


def init_db():
    """Initialize Firebase connection."""
    try:
        _get_db()
        print("[memory] ✅ Firebase Firestore connected")
        return True
    except Exception as e:
        print(f"[memory] ❌ Firebase error: {e}")
        return False


# ---------------------------------------------------------------------------
# LAYER 1: SHARED Q&A CACHE
# ---------------------------------------------------------------------------

def _hash_question(question: str) -> str:
    """Create a normalized hash for question matching."""
    normalized = question.lower().strip()
    # Remove extra spaces
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def get_cached_answer(question: str) -> Optional[Dict]:
    """
    Check if this question has been answered before.
    Returns cached answer dict or None.

    Used for SHARED cache — all students benefit.
    """
    try:
        db = _get_db()
        q_hash = _hash_question(question)

        doc = db.collection("qa_cache").document(q_hash).get()
        if doc.exists:
            data = doc.to_dict()
            # Update hit count and last accessed
            doc.reference.update({
                "hit_count": firestore.Increment(1),
                "last_accessed": time.time(),
            })
            print(f"[cache] ✅ Cache hit! ({data.get('hit_count', 0)+1} hits) — {question[:50]}")
            return data
        return None

    except Exception as e:
        print(f"[cache] get_cached_answer error: {e}")
        return None


def save_to_cache(
    question: str,
    answer: str,
    answer_type: str,
    task: str = "",
    language: str = "BM",
):
    """
    Save Q&A pair to shared cache.
    All students will benefit from this cached answer.
    """
    try:
        db = _get_db()
        q_hash = _hash_question(question)
        now = time.time()

        db.collection("qa_cache").document(q_hash).set({
            "question": question,
            "question_normalized": question.lower().strip(),
            "answer": answer,
            "answer_type": answer_type,
            "task": task,
            "language": language,
            "hit_count": 0,
            "created_at": now,
            "updated_at": now,
            "last_accessed": now,
        })
        print(f"[cache] 💾 Saved to cache — {question[:50]}")

    except Exception as e:
        print(f"[cache] save_to_cache error: {e}")


def should_cache(answer_type: str, task: str) -> bool:
    """
    Decide if this answer should be cached.
    Cache: calculations + theory
    Don't cache: personal/session questions
    """
    # Don't cache fallback answers
    if answer_type == "fallback":
        return False

    # Don't cache personal questions
    personal_tasks = {"unknown"}
    if task in personal_tasks:
        return False

    return True


def get_cache_stats() -> Dict:
    """Get shared cache statistics."""
    try:
        db = _get_db()
        docs = db.collection("qa_cache").stream()
        items = [d.to_dict() for d in docs]

        total = len(items)
        total_hits = sum(d.get("hit_count", 0) for d in items)
        top_questions = sorted(items, key=lambda x: x.get("hit_count", 0), reverse=True)[:5]

        return {
            "total_cached": total,
            "total_cache_hits": total_hits,
            "api_calls_saved": total_hits,
            "top_questions": [
                {
                    "question": q.get("question", "")[:60],
                    "hits": q.get("hit_count", 0),
                    "answer_type": q.get("answer_type", ""),
                }
                for q in top_questions
            ],
        }
    except Exception as e:
        print(f"[cache] stats error: {e}")
        return {}


# ---------------------------------------------------------------------------
# LAYER 2: PERSONAL MEMORY (per student)
# ---------------------------------------------------------------------------

def save_message(
    session_id: str,
    role: str,
    content: str,
    user_name: str = "",
):
    """Save a message to personal session history."""
    try:
        db = _get_db()
        now = time.time()

        # Update session metadata
        session_ref = db.collection("sessions").document(session_id)
        session_ref.set({
            "updated_at": now,
            "user_name": user_name or session_id,
        }, merge=True)

        # Add message to subcollection
        session_ref.collection("messages").add({
            "role": role,
            "content": content[:2000],  # limit content size
            "timestamp": now,
        })

        # Keep only last 20 messages per session
        _trim_messages(session_id, keep=20)

    except Exception as e:
        print(f"[memory] save_message error: {e}")


def get_history(session_id: str, last_n: int = 6) -> List[Dict]:
    """Get last N messages for a session (personal memory)."""
    try:
        db = _get_db()
        messages = (
            db.collection("sessions")
            .document(session_id)
            .collection("messages")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(last_n)
            .stream()
        )

        result = [
            {
                "role": m.to_dict()["role"],
                "content": m.to_dict()["content"],
            }
            for m in messages
        ]
        return list(reversed(result))  # chronological order

    except Exception as e:
        print(f"[memory] get_history error: {e}")
        return []


def build_history_context(session_id: str, last_n: int = 4, lang: str = "BM") -> str:
    """
    Build conversation history string for LLM context.
    Enables 'soalan saya tadi...' to work correctly.
    """
    history = get_history(session_id, last_n)
    if not history:
        return ""

    if lang == "EN":
        lines = ["\n\nPREVIOUS CONVERSATION:"]
        for msg in history:
            role = "Student" if msg["role"] == "user" else "Cikgu AI"
            content = msg["content"][:200]
            lines.append(f"{role}: {content}")
        lines.append("(Use above history if the new question refers to previous conversation)")
    else:
        lines = ["\n\nSEJARAH PERBUALAN SEBELUM INI:"]
        for msg in history:
            role = "Pelajar" if msg["role"] == "user" else "Cikgu AI"
            content = msg["content"][:200]
            lines.append(f"{role}: {content}")
        lines.append("(Gunakan sejarah di atas jika soalan baru berkaitan dengan perbualan sebelum ini)")

    return "\n".join(lines)


def is_personal_question(question: str) -> bool:
    """
    Detect if question refers to previous conversation.
    These should use personal memory, not shared cache.
    """
    personal_keywords = [
        # BM
        "tadi", "sebelum", "soalan saya", "yang saya tanya",
        "jawapan tadi", "maksud tadi", "itu tadi", "saya faham",
        "boleh ulang", "terangkan lagi", "lebih lanjut",
        # EN
        "previous", "before", "just asked", "my question",
        "repeat", "explain more", "what you said",
        "earlier", "last question", "again",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in personal_keywords)


def clear_session(session_id: str):
    """Delete all messages for a session."""
    try:
        db = _get_db()
        messages = (
            db.collection("sessions")
            .document(session_id)
            .collection("messages")
            .stream()
        )
        for msg in messages:
            msg.reference.delete()

        db.collection("sessions").document(session_id).delete()
        print(f"[memory] Session {session_id} cleared")

    except Exception as e:
        print(f"[memory] clear_session error: {e}")


# ---------------------------------------------------------------------------
# SMART LOOKUP: Cache + Personal Memory Combined
# ---------------------------------------------------------------------------

def smart_lookup(
    question: str,
    session_id: str,
    lang: str = "BM",
) -> Tuple[Optional[Dict], str]:
    """
    Combined smart lookup:
    1. If personal question → get history context (don't use cache)
    2. If general question → check shared cache first

    Returns:
        (cached_result, history_context)
        cached_result: None if not found in cache
        history_context: string to inject into LLM prompt
    """
    history_context = ""

    # Always get personal history for context
    if session_id:
        history_context = build_history_context(session_id, last_n=4, lang=lang)

    # Personal questions — skip cache, use history
    if is_personal_question(question):
        print(f"[memory] 👤 Personal question detected — using history")
        return None, history_context

    # General questions — check shared cache
    cached = get_cached_answer(question)
    if cached:
        return cached, history_context

    return None, history_context


# ---------------------------------------------------------------------------
# ADMIN / STATS
# ---------------------------------------------------------------------------

def get_all_stats() -> Dict:
    """Get full system stats for admin dashboard."""
    try:
        db = _get_db()

        # Session count
        sessions = list(db.collection("sessions").stream())
        session_count = len(sessions)

        # Cache stats
        cache_stats = get_cache_stats()

        return {
            "total_students": session_count,
            "cache": cache_stats,
        }
    except Exception as e:
        return {"error": str(e)}


def cleanup_old_sessions(days: int = 7):
    """Delete sessions not updated in X days."""
    try:
        db = _get_db()
        cutoff = time.time() - (days * 86400)
        old = (
            db.collection("sessions")
            .where("updated_at", "<", cutoff)
            .stream()
        )
        count = 0
        for session in old:
            clear_session(session.id)
            count += 1
        if count:
            print(f"[memory] Cleaned {count} old sessions")
        return count
    except Exception as e:
        print(f"[memory] cleanup error: {e}")
        return 0


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def _trim_messages(session_id: str, keep: int = 20):
    """Keep only the most recent messages."""
    try:
        db = _get_db()
        all_msgs = list(
            db.collection("sessions")
            .document(session_id)
            .collection("messages")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
        )
        if len(all_msgs) > keep:
            for msg in all_msgs[keep:]:
                msg.reference.delete()
    except Exception as e:
        print(f"[memory] trim error: {e}")
