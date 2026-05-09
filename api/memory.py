"""
memory.py — Cikgu AI Kimia (Firebase Firestore)
================================================
Persistent conversation memory using Firebase Firestore.
- Free tier: 50k reads/day, 20k writes/day
- Survives Render restarts
- Real-time, scalable
"""

import os
import time
from typing import List, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore

# ── Init Firebase ────────────────────────────────────────────────────────────
_db = None

def _get_db():
    global _db
    if _db is None:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
    return _db


def init_db():
    """Initialize Firebase connection."""
    try:
        _get_db()
        print("[memory] Firebase Firestore connected")
    except Exception as e:
        print(f"[memory] Firebase error: {e}")


# ── Core Functions ────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, user_name: str = ""):
    """Save a message to Firestore."""
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
            "content": content[:2000],
            "timestamp": now,
        })

        # Keep only last 20 messages
        _trim_messages(session_id, keep=20)

    except Exception as e:
        print(f"[memory] save_message error: {e}")


def get_history(session_id: str, last_n: int = 6) -> List[Dict]:
    """Get last N messages for a session."""
    try:
        db = _get_db()
        messages = db.collection("sessions")\
            .document(session_id)\
            .collection("messages")\
            .order_by("timestamp", direction=firestore.Query.DESCENDING)\
            .limit(last_n)\
            .stream()

        result = [{"role": m.to_dict()["role"],
                   "content": m.to_dict()["content"]}
                  for m in messages]
        return list(reversed(result))  # chronological order

    except Exception as e:
        print(f"[memory] get_history error: {e}")
        return []


def clear_session(session_id: str):
    """Delete all messages for a session."""
    try:
        db = _get_db()
        # Delete all messages in subcollection
        messages = db.collection("sessions")\
            .document(session_id)\
            .collection("messages")\
            .stream()
        for msg in messages:
            msg.reference.delete()

        # Delete session document
        db.collection("sessions").document(session_id).delete()
        print(f"[memory] Session {session_id} cleared")

    except Exception as e:
        print(f"[memory] clear_session error: {e}")


def build_history_prompt(session_id: str, last_n: int = 6) -> str:
    """Build history context string for LLM prompt."""
    history = get_history(session_id, last_n)
    if not history:
        return ""

    lines = ["\n\nSEJARAH PERBUALAN SEBELUM INI:"]
    for msg in history:
        role = "Pelajar" if msg["role"] == "user" else "Cikgu AI"
        content = msg["content"][:300]
        lines.append(f"{role}: {content}")
    lines.append("\n(Gunakan sejarah di atas jika soalan baru berkaitan dengan perbualan sebelum ini)")
    return "\n".join(lines)


def all_sessions_stats() -> List[Dict]:
    """Admin: list all sessions."""
    try:
        db = _get_db()
        sessions = db.collection("sessions").order_by(
            "updated_at", direction=firestore.Query.DESCENDING
        ).limit(50).stream()
        return [{"session_id": s.id, **s.to_dict()} for s in sessions]
    except Exception as e:
        print(f"[memory] all_sessions_stats error: {e}")
        return []


def get_session_summary(session_id: str) -> Optional[Dict]:
    """Get session metadata."""
    try:
        db = _get_db()
        doc = db.collection("sessions").document(session_id).get()
        if doc.exists:
            return {"session_id": session_id, **doc.to_dict()}
        return None
    except Exception as e:
        return None


def cleanup_old_sessions():
    """Delete sessions not updated in 7 days."""
    try:
        db = _get_db()
        cutoff = time.time() - (7 * 86400)
        old = db.collection("sessions")\
            .where("updated_at", "<", cutoff)\
            .stream()
        count = 0
        for session in old:
            clear_session(session.id)
            count += 1
        if count:
            print(f"[memory] Cleaned {count} old sessions")
    except Exception as e:
        print(f"[memory] cleanup error: {e}")


# ── Helper ────────────────────────────────────────────────────────────────────

def _trim_messages(session_id: str, keep: int = 20):
    """Keep only the most recent `keep` messages."""
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


# Auto-init
init_db()