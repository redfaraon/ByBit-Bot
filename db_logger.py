import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parent / "assets" / "bot_data.db"
_LOCK = threading.RLock()
_SCHEMA_CREATED = False


def _ensure_dir() -> None:
    try:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _connect() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(str(_DB_PATH), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_decisions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stage TEXT,
            tokens INTEGER,
            token_limit INTEGER,
            token_soft_limit INTEGER,
            duration REAL,
            response_action TEXT,
            response_reason TEXT,
            low_confidence INTEGER,
            auto_low_confidence INTEGER,
            needs TEXT,
            context_counts TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT,
            symbol TEXT,
            context TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS protection_checks (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            reason TEXT,
            has_stop INTEGER,
            has_take INTEGER,
            needs_protection INTEGER
        )
        """
    )
    conn.commit()


def initialize() -> None:
    global _SCHEMA_CREATED
    with _LOCK:
        if _SCHEMA_CREATED:
            return
        try:
            conn = _connect()
            _create_schema(conn)
            _SCHEMA_CREATED = True
        finally:
            try:
                conn.close()
            except Exception:
                pass


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "null"


def log_ai_decision(entry: dict[str, Any]) -> None:
    if not entry or "symbol" not in entry:
        return
    ts = entry.get("timestamp") or datetime.utcnow().isoformat()
    symbol = str(entry.get("symbol", "")).strip()
    stage = entry.get("stage")
    tokens = entry.get("tokens")
    token_limit = entry.get("token_limit")
    token_soft_limit = entry.get("token_soft_limit")
    duration = entry.get("duration_sec")
    response_action = entry.get("response_action")
    response_reason = entry.get("response_reason")
    low_confidence = 1 if entry.get("low_confidence") else 0
    auto_low_confidence = 1 if entry.get("auto_low_confidence") else 0
    needs = _safe_json(entry.get("needs") or [])
    context_counts = _safe_json(entry.get("context_counts") or {})
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ai_decisions (
                timestamp,
                symbol,
                stage,
                tokens,
                token_limit,
                token_soft_limit,
                duration,
                response_action,
                response_reason,
                low_confidence,
                auto_low_confidence,
                needs,
                context_counts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                symbol,
                stage,
                tokens,
                token_limit,
                token_soft_limit,
                duration,
                response_action,
                response_reason,
                low_confidence,
                auto_low_confidence,
                needs,
                context_counts,
            ),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def log_error_event(message: str, severity: str = "INFO", symbol: str | None = None, context: str | None = None) -> None:
    ts = datetime.utcnow().isoformat()
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO errors (timestamp, severity, message, symbol, context)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, severity, message, symbol, context),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def log_protection_check(
    symbol: str, reason: str, has_stop: bool, has_take: bool, needs_protection: bool
) -> None:
    ts = datetime.utcnow().isoformat()
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO protection_checks (
                timestamp,
                symbol,
                reason,
                has_stop,
                has_take,
                needs_protection
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                symbol,
                reason,
                1 if has_stop else 0,
                1 if has_take else 0,
                1 if needs_protection else 0,
            ),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
