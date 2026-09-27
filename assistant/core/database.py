"""SQLite storage foundation for the Virtual Assistant.

Uses Python's standard-library :mod:`sqlite3`. No ORM and no third-party
driver, which keeps the dependency count where it is.

Design notes
------------
* **Nothing happens on import.** Constructing a :class:`Database` only
  records a path. The file is created the first time a connection is
  actually opened, so importing this module is always free and safe.
* **Idempotent initialisation.** :meth:`Database.initialize` uses
  ``CREATE TABLE IF NOT EXISTS`` and records the schema version, so it can
  be run on every startup without error or data loss.
* **Safe cleanup.** :class:`Database` is a context manager and
  :meth:`Database.close` is safe to call more than once.
* **Ready for a second reader.** Write-ahead logging and a busy timeout are
  enabled so the future read-only dashboard can read while the assistant
  writes, instead of hitting "database is locked".

Only the two tables the foundation needs are created here. Notes, tasks and
preferences are added by later steps, each with its own migration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Sequence

from assistant.logging_config import get_logger

logger = get_logger(__name__)

#: Current schema version. Bump this when a later step adds a table, and
#: add the matching migration to ``_MIGRATIONS``.
SCHEMA_VERSION = 2

#: Version 1: the foundation. One bookkeeping table and one table for the
#: command history that ``AppContext.log_command`` already produces.
_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT    NOT NULL DEFAULT (datetime('now')),
    tool_name   TEXT,
    utterance   TEXT,
    response    TEXT
);

CREATE INDEX IF NOT EXISTS idx_interactions_time ON interactions (occurred_at);
CREATE INDEX IF NOT EXISTS idx_interactions_tool ON interactions (tool_name);
"""

#: Version 2: user notes. Additive only, so it applies cleanly to a database
#: that already holds command history.
_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    body        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notes_created ON notes (created_at);
"""

#: Applied in order by :meth:`Database.initialize`. Each entry is a
#: ``(version, sql)`` pair; already-applied versions are skipped, so an
#: existing database upgrades in place without touching its data.
_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, _SCHEMA_V1),
    (2, _SCHEMA_V2),
)


class Database:
    """A lazily opened connection to the assistant's SQLite database.

    Args:
        path: location of the database file. Parent directories are created
            on first connection. Pass ``":memory:"`` for a throwaway
            in-memory database, which is what the tests use.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn: sqlite3.Connection | None = None

    @property
    def path(self) -> str:
        """The database file path. Never contains a secret."""
        return self._path

    @property
    def is_open(self) -> bool:
        """True when a connection is currently held."""
        return self._conn is not None

    def connect(self) -> sqlite3.Connection:
        """Open the connection, or return the existing one.

        The first call creates the parent directory and the database file.
        Foreign keys, write-ahead logging and a busy timeout are applied.

        Returns:
            The live :class:`sqlite3.Connection`.

        Raises:
            sqlite3.Error: if the file cannot be opened.
        """
        if self._conn is not None:
            return self._conn

        if self._path != ":memory:":
            parent = Path(self._path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Concurrency and safety defaults chosen for a desktop app.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        self._conn = conn
        logger.debug("Opened SQLite database at %s", self._path)
        return conn

    def close(self) -> None:
        """Close the connection. Safe to call when already closed."""
        if self._conn is None:
            return
        try:
            self._conn.commit()
        except sqlite3.Error as exc:
            # An uncommittable database must not stop the app shutting down.
            logger.warning("Could not commit before closing the database: %s", exc)
        try:
            self._conn.close()
        finally:
            self._conn = None
            logger.debug("Closed SQLite database at %s", self._path)

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def initialize(self) -> int:
        """Create the schema if it is not already present.

        Safe to call on every startup: each step uses
        ``CREATE ... IF NOT EXISTS`` and a version is recorded only once.
        Already-applied migrations are skipped.

        Returns:
            The schema version now in effect.

        Raises:
            sqlite3.Error: if the schema cannot be created.
        """
        conn = self.connect()
        applied = self.applied_versions()

        for version, sql in _MIGRATIONS:
            if version in applied:
                logger.debug("Migration %s already applied; skipping.", version)
                continue
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (version,)
            )
            conn.commit()
            logger.info("Applied database migration %s", version)

        current = self.schema_version()
        if current != SCHEMA_VERSION:
            # The file is newer than this code understands.
            logger.warning(
                "Database schema version %s is newer than the expected %s.",
                current,
                SCHEMA_VERSION,
            )
        return current

    def schema_version(self) -> int:
        """Return the highest applied schema version, or 0 if empty."""
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT MAX(version) AS version FROM schema_version"
            ).fetchone()
        except sqlite3.Error:
            # Table not there yet, so nothing has been applied.
            return 0
        return int(row["version"]) if row and row["version"] is not None else 0

    def applied_versions(self) -> set[int]:
        """Return every schema version already recorded in the file."""
        conn = self.connect()
        try:
            rows = conn.execute("SELECT version FROM schema_version").fetchall()
        except sqlite3.Error:
            return set()
        return {int(row["version"]) for row in rows}

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Run one statement and commit it."""
        conn = self.connect()
        cursor = conn.execute(sql, tuple(params))
        conn.commit()
        return cursor

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        """Run one statement for many parameter sets, then commit."""
        conn = self.connect()
        conn.executemany(sql, [tuple(row) for row in rows])
        conn.commit()

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a SELECT and return every row."""
        return self.connect().execute(sql, tuple(params)).fetchall()

    def query_one(
        self, sql: str, params: Sequence[Any] = ()
    ) -> sqlite3.Row | None:
        """Run a SELECT and return the first row, or None."""
        return self.connect().execute(sql, tuple(params)).fetchone()

    def table_names(self) -> list[str]:
        """Return the table names present in the database file."""
        rows = self.query_all(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
        return [row["name"] for row in rows]

    def __repr__(self) -> str:
        # Only the path and connection state. No connection internals and
        # no row data, so logging a Database can never leak content.
        return f"Database(path={self._path!r}, open={self.is_open})"
