"""The only SQL in the service: reads clients and features, writes scores and their audit trail (schema `crew`).

The connection pool is created on first use from the DATABASE_URL environment variable (Supabase Session
pooler, user crew_app). Without it every call raises DatabaseNotConfigured; nothing falls back to a default.
Supavisor does not keep prepared statements across clients, so they are disabled (prepare_threshold=None).
Timestamps are returned as ISO 8601 text, the same form the API and the templates show.
"""
import json
import os
import threading

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

DATABASE_URL_ENV = "DATABASE_URL"
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 4  # per gunicorn worker; the Supabase free pooler has a small connection budget
MAX_PAGE_SIZE = 500
SOURCES = ("api", "ui")

_pool = None
_pool_lock = threading.Lock()


class DatabaseNotConfigured(RuntimeError):
    """DATABASE_URL is not set: endpoints that need the database answer 503."""


def is_configured() -> bool:
    return bool(os.environ.get(DATABASE_URL_ENV))


def _get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            url = os.environ.get(DATABASE_URL_ENV)
            if not url:
                raise DatabaseNotConfigured(f"{DATABASE_URL_ENV} is not set: the service has no database")
            _pool = ConnectionPool(
                url,
                min_size=POOL_MIN_SIZE,
                max_size=POOL_MAX_SIZE,
                kwargs={"prepare_threshold": None, "row_factory": dict_row},
                check=ConnectionPool.check_connection,  # the pooler may drop idle connections
                name="crew",
                open=True,
            )
    return _pool


def _iso(value):
    return None if value is None else value.isoformat()


def _jsonb(value) -> Jsonb:
    return Jsonb(value, dumps=lambda obj: json.dumps(obj, allow_nan=False))  # NaN is not JSON: fail, never store it


def _check_page(limit: int, offset: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be an integer from 1 to {MAX_PAGE_SIZE}, got {limit!r}")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError(f"offset must be a non-negative integer, got {offset!r}")


def list_clients(limit: int, offset: int) -> list[dict]:
    """One row per client with its latest score (score fields None when never scored), riskiest first."""
    _check_page(limit, offset)
    with _get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT c.sk_id_curr, c.target, s.pd_raw, s.pd_calibrated, s.above_statistical, s.above_business,
                   s.model_version, s.scored_at
            FROM crew.clients c
            LEFT JOIN LATERAL (
                SELECT * FROM crew.scores x WHERE x.sk_id_curr = c.sk_id_curr
                ORDER BY x.scored_at DESC, x.score_id DESC LIMIT 1
            ) s ON true
            ORDER BY s.pd_raw DESC NULLS LAST, c.sk_id_curr
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        ).fetchall()
    return [{**row, "scored_at": _iso(row["scored_at"])} for row in rows]


def count_clients() -> int:
    with _get_pool().connection() as conn:
        return conn.execute("SELECT count(*) AS n FROM crew.clients").fetchone()["n"]


def get_client(sk_id_curr: int) -> dict | None:
    with _get_pool().connection() as conn:
        row = conn.execute(
            "SELECT sk_id_curr, target, loaded_at FROM crew.clients WHERE sk_id_curr = %s", (sk_id_curr,)
        ).fetchone()
    return None if row is None else {**row, "loaded_at": _iso(row["loaded_at"])}


def get_features(sk_id_curr: int) -> dict | None:
    """The stored feature record in the API input format, or None. More than one feature set is an error."""
    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT feature_set, features FROM crew.client_features WHERE sk_id_curr = %s", (sk_id_curr,)
        ).fetchall()
    if len(rows) > 1:
        sets = sorted(row["feature_set"] for row in rows)
        raise RuntimeError(f"client {sk_id_curr} has {len(rows)} feature sets {sets}; exactly one is expected")
    return rows[0]["features"] if rows else None


def latest_score(sk_id_curr: int) -> dict | None:
    """The stored score_record() result of the latest score, plus scored_at."""
    with _get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT result, scored_at FROM crew.scores WHERE sk_id_curr = %s
            ORDER BY scored_at DESC, score_id DESC LIMIT 1
            """,
            (sk_id_curr,),
        ).fetchone()
    return None if row is None else {**row["result"], "scored_at": _iso(row["scored_at"])}


def save_score(sk_id_curr: int, result: dict, source: str) -> dict:
    """Insert the score and its audit row in one transaction (EU AI Act Art. 12 logging). Returns score_id, scored_at."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    flags = result["flags"]
    with _get_pool().connection() as conn, conn.transaction():
        saved = conn.execute(
            """
            INSERT INTO crew.scores (sk_id_curr, model_version, bundle_sha256, pd_raw, pd_calibrated,
                                     above_statistical, above_business, result)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING score_id, scored_at
            """,
            (sk_id_curr, result["model_version"], result["bundle_sha256"], result["pd_raw"], result["pd_calibrated"],
             flags["above_statistical"], flags["above_business"], _jsonb(result)),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO crew.score_audit_log (score_id, sk_id_curr, event, source, model_version, bundle_sha256)
            VALUES (%s, %s, 'scored', %s, %s, %s)
            """,
            (saved["score_id"], sk_id_curr, source, result["model_version"], result["bundle_sha256"]),
        )
    return {"score_id": saved["score_id"], "scored_at": _iso(saved["scored_at"])}
