from contextlib import contextmanager
import sqlite3
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str((_ROOT / "db" / "db.sqlite3").resolve())
DB_SQL_PATH = str((_ROOT / "db" / "db.sql").resolve())

can_get_conn_without_init = False
is_initialized = False

@contextmanager
def get_db_connection():
    if not is_initialized and not can_get_conn_without_init:
        raise Exception("Database not initialized")
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    global can_get_conn_without_init
    global is_initialized
    if is_initialized:
        raise Warning("Database already initialized")
    # ensure database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    can_get_conn_without_init = True
    with get_db_connection() as conn:
        can_get_conn_without_init = False
        # read schema from SQL file (UTF-8)
        try:
            with open(DB_SQL_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found: {DB_SQL_PATH}")
        conn.executescript(schema_sql)
        is_initialized = True
    if not is_initialized:
        raise Exception("Failed to initialize database")


def clear_db():
    """Deletes the SQLite database file if it exists."""
    global is_initialized
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    is_initialized = False
