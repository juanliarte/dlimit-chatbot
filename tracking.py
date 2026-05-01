"""
tracking.py
===========
Persistencia simple de conversaciones del chatbot Dlimit en SQLite.

La BBDD se crea en DB_PATH (por defecto /var/data/conversations.db, donde
Render monta el disco persistente). Si DB_PATH no es accesible, el módulo
degrada con gracia (log + return) sin romper el chatbot.

Uso desde chatbot_server.py:

    from tracking import init_db, log_conversation, update_lead_info, export_csv

    init_db()  # llamar una vez al arrancar la app

    conv_id = log_conversation(
        session_id="abc",
        ip="1.2.3.4",
        language="es",
        question="hola",
        answer="...",
        sources_count=3,
        tokens_in=100,
        tokens_out=200,
        response_time_ms=2500,
        had_email=True,
    )

    # Más tarde, cuando se conozca info del lead:
    update_lead_info(conv_id, was_lead=True, lead_email="x@y.com",
                     lead_urgency="HIGH", lead_sector="hostelería")
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import threading
import time
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "/var/data/conversations.db")

# Lock para evitar problemas de concurrencia en escrituras
_lock = threading.Lock()

# Versión del schema (incrementar si cambia la tabla)
SCHEMA_VERSION = 1


def _connect():
    """Abre conexión SQLite con timeout razonable."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> bool:
    """Crea la tabla conversations si no existe. Return True si OK."""
    try:
        # Crear directorio padre si no existe (defensivo)
        parent = os.path.dirname(DB_PATH)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass

        with _connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    ip TEXT,
                    country TEXT,
                    language TEXT,
                    question TEXT NOT NULL,
                    answer TEXT,
                    sources_count INTEGER,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    response_time_ms INTEGER,
                    had_email INTEGER DEFAULT 0,
                    was_lead INTEGER,
                    lead_email TEXT,
                    lead_urgency TEXT,
                    lead_sector TEXT,
                    lead_quantity TEXT,
                    schema_version INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversations(ts_utc);
                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conv_was_lead ON conversations(was_lead);
            """)
        print(f"[tracking] init_db OK · DB_PATH={DB_PATH}", flush=True)
        return True
    except Exception as e:
        print(f"[tracking] init_db FAILED: {e}", flush=True)
        return False


def log_conversation(
    *,
    session_id: Optional[str] = None,
    ip: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    question: str,
    answer: Optional[str] = None,
    sources_count: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    response_time_ms: int = 0,
    had_email: bool = False,
) -> Optional[int]:
    """Inserta una fila y devuelve el id. None si falla."""
    try:
        with _lock:
            with _connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO conversations (
                        session_id, ip, country, language, question, answer,
                        sources_count, tokens_in, tokens_out, response_time_ms,
                        had_email, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, ip, country, language,
                        (question or "")[:8000],
                        (answer or "")[:16000],
                        sources_count, tokens_in, tokens_out,
                        response_time_ms,
                        1 if had_email else 0,
                        SCHEMA_VERSION,
                    ),
                )
                return cur.lastrowid
    except Exception as e:
        print(f"[tracking] log_conversation FAILED: {e}", flush=True)
        return None


def update_lead_info(
    conv_id: int,
    *,
    was_lead: Optional[bool] = None,
    lead_email: Optional[str] = None,
    lead_urgency: Optional[str] = None,
    lead_sector: Optional[str] = None,
    lead_quantity: Optional[str] = None,
) -> bool:
    """Actualiza la fila con la info del lead detectada de forma asíncrona."""
    if conv_id is None:
        return False
    try:
        fields = []
        values = []
        if was_lead is not None:
            fields.append("was_lead = ?")
            values.append(1 if was_lead else 0)
        if lead_email is not None:
            fields.append("lead_email = ?")
            values.append(lead_email[:200])
        if lead_urgency is not None:
            fields.append("lead_urgency = ?")
            values.append(lead_urgency[:20])
        if lead_sector is not None:
            fields.append("lead_sector = ?")
            values.append(lead_sector[:100])
        if lead_quantity is not None:
            fields.append("lead_quantity = ?")
            values.append(str(lead_quantity)[:50])
        if not fields:
            return False
        values.append(conv_id)

        with _lock:
            with _connect() as conn:
                conn.execute(
                    f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?",
                    values,
                )
        return True
    except Exception as e:
        print(f"[tracking] update_lead_info FAILED: {e}", flush=True)
        return False


def export_csv() -> str:
    """Devuelve un string CSV con todas las filas de la tabla."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY id ASC"
            ).fetchall()
        if not rows:
            return ""
        cols = rows[0].keys()
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([r[c] for c in cols])
        return out.getvalue()
    except Exception as e:
        print(f"[tracking] export_csv FAILED: {e}", flush=True)
        return ""


def stats() -> dict:
    """Devuelve un dict con métricas básicas para /admin/stats."""
    try:
        with _connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            leads = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE was_lead = 1"
            ).fetchone()[0]
            with_email = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE had_email = 1"
            ).fetchone()[0]
            avg_ms = conn.execute(
                "SELECT ROUND(AVG(response_time_ms)) FROM conversations WHERE response_time_ms > 0"
            ).fetchone()[0] or 0
            tokens_in_sum = conn.execute(
                "SELECT COALESCE(SUM(tokens_in), 0) FROM conversations"
            ).fetchone()[0]
            tokens_out_sum = conn.execute(
                "SELECT COALESCE(SUM(tokens_out), 0) FROM conversations"
            ).fetchone()[0]
            by_lang = {
                r[0] or "(unknown)": r[1]
                for r in conn.execute(
                    "SELECT language, COUNT(*) FROM conversations "
                    "GROUP BY language ORDER BY 2 DESC LIMIT 10"
                ).fetchall()
            }
            by_urgency = {
                r[0] or "(none)": r[1]
                for r in conn.execute(
                    "SELECT lead_urgency, COUNT(*) FROM conversations "
                    "WHERE was_lead = 1 GROUP BY lead_urgency"
                ).fetchall()
            }
        return {
            "total_conversations": total,
            "total_leads": leads,
            "lead_conversion_pct": round(leads * 100.0 / total, 2) if total else 0,
            "had_email_count": with_email,
            "avg_response_time_ms": int(avg_ms),
            "tokens_in_sum": tokens_in_sum,
            "tokens_out_sum": tokens_out_sum,
            "by_language": by_lang,
            "by_lead_urgency": by_urgency,
            "db_path": DB_PATH,
            "schema_version": SCHEMA_VERSION,
        }
    except Exception as e:
        print(f"[tracking] stats FAILED: {e}", flush=True)
        return {"error": str(e)}
