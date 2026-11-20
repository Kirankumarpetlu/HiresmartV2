import atexit
from pathlib import Path
from threading import Lock

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None
_lock = Lock()
_registered = False


def pool_object() -> ConnectionPool:
    global _pool, _registered
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ConnectionPool(
                    settings.database_url.get_secret_value(), min_size=1, max_size=8,
                    kwargs={"row_factory": dict_row, "connect_timeout": 10}, configure=register_vector,
                    open=False, timeout=10,
                )
                if not _registered:
                    atexit.register(_pool.close)
                    _registered = True
    return _pool


def get_pool() -> ConnectionPool:
    pool = pool_object()
    if pool.closed:
        with _lock:
            if pool.closed:
                pool.open()
                pool.wait(timeout=15)
    return pool


def fetch(sql: str, params=None) -> list[dict]:
    with get_pool().connection() as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql: str, params=None) -> None:
    with get_pool().connection() as conn:
        conn.execute(sql, params)


def check_dimensions() -> None:
    rows = fetch("""
        SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod) AS type
        FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname IN ('resume_chunks', 'job_descriptions')
          AND a.attname = 'embedding' AND NOT a.attisdropped
    """)
    if len(rows) != 2 or any(r["type"] != f"vector({settings.embed_dim})" for r in rows):
        raise RuntimeError(f"EMBED_DIM={settings.embed_dim} does not match database vector columns; apply the schema/dimension migration first")


def init_database() -> None:
    import psycopg

    sql = (Path(__file__).resolve().parent.parent / "db" / "init.sql").read_text()
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=10) as conn:
        conn.execute(sql)
    check_dimensions()
