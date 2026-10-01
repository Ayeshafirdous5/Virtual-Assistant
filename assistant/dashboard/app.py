"""The dashboard application: read-only database access and the ASGI app.

This module is the only place in the dashboard that touches SQLite, and it
touches it in one specific way: **read-only, always**. There is no code path
from an HTTP request to an ``INSERT``, ``UPDATE`` or ``DELETE``, and that is a
property of the connection rather than a promise made in a docstring.

How read-only is enforced
-------------------------
The connection is opened with the SQLite URI form ``file:<path>?mode=ro``.
SQLite itself then rejects any attempt to write, so a bug in a template, a
route or a future refactor cannot corrupt the assistant's history. On top of
that, :class:`ReadOnlyDatabase` deliberately exposes **only** the two read
methods :class:`~assistant.analytics.service.AnalyticsService` needs.
``execute`` and ``executemany`` are absent, so there is nothing to call by
mistake.

Nothing is opened at import time. :func:`create_app` builds the app and the
template environment; the database is opened per request by
:func:`open_readonly` and closed in a ``finally`` block, so importing this
package cannot create or touch a file.

Secrets
-------
:class:`Settings` holds exactly one path and nothing else. No API key, no
``.env`` content and no credential is read, stored, rendered or served. The
health endpoint deliberately reports no path, since a liveness probe has no
use for one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from assistant.logging_config import get_logger

logger = get_logger(__name__)

#: Loopback only. The dashboard shows a user's own command history, and this
#: sprint has no authentication, so it must not be reachable from the network.
DEFAULT_HOST = "127.0.0.1"

#: Deliberately non-standard, so it cannot collide with another local service.
DEFAULT_PORT = 8765

#: The templates that ship with this package.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: The stylesheet and script that ship with this package.
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ReadOnlyDatabase:
    """A minimal, read-only stand-in for :class:`~assistant.core.database.Database`.

    Exposes exactly the two query methods
    :class:`~assistant.analytics.service.AnalyticsService` calls, plus
    ``close``. There is deliberately no ``execute`` and no ``executemany``, so
    a caller cannot write even if it wanted to.

    Args:
        connection: an already-open ``sqlite3`` connection in ``mode=ro``.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        """Run a SELECT and return every row."""
        return self._connection.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """Run a SELECT and return the first row, or ``None``."""
        return self._connection.execute(sql, tuple(params)).fetchone()

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        self._connection.close()

    @property
    def read_only(self) -> bool:
        """True. The connection cannot write, so this reports a fact."""
        return True

    def __repr__(self) -> str:
        # No path: this object may be logged, and a path is not something a
        # log line needs to carry.
        return "ReadOnlyDatabase(read_only=True)"


def open_readonly(path: str | Path) -> ReadOnlyDatabase:
    """Open a database that cannot be written to.

    Uses the ``file:...?mode=ro`` URI form, which makes SQLite itself refuse
    writes. The file is not created if it is missing: opening a missing
    database read-only is an error, which is the correct signal, and it stops
    the dashboard from silently creating an empty database beside the real
    one.

    Args:
        path: location of the assistant's SQLite file.

    Returns:
        A :class:`ReadOnlyDatabase` the caller must close.

    Raises:
        sqlite3.Error: if the file is missing or cannot be opened read-only.
    """
    target = Path(path)
    uri = f"file:{target.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    # sqlite3.Row lets the analytics layer index rows by column name, exactly
    # as the real Database class does, so no adapter code is needed.
    connection.row_factory = sqlite3.Row
    return ReadOnlyDatabase(connection)


@dataclass(frozen=True)
class Settings:
    """The dashboard's entire configuration.

    One path, and nothing else. Holding the database path here rather than
    reading an environment variable inside a route keeps "what does the
    dashboard know" answerable by reading one short class.
    """

    db_path: Path

    @property
    def exists(self) -> bool:
        """True when the configured database file is present."""
        return self.db_path.exists()


def create_app(db_path: str | Path | None = None) -> FastAPI:
    """Build the dashboard application.

    Args:
        db_path: the assistant's database. When omitted, the project's
            configured path is used, so ``python -m assistant.dashboard.app``
            needs no arguments.

    Returns:
        A configured :class:`~fastapi.FastAPI` app.

    Building the app opens no connection and reads no data, so it is safe to
    call in a test with no database present.
    """
    if db_path is None:
        from assistant.config import get_config

        db_path = get_config().db_path

    settings = Settings(db_path=Path(db_path))

    app = FastAPI(
        title="Virtual Assistant Analytics",
        description="A read-only view of the assistant's own command history.",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.dashboard = settings
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # The directory ships with the package, so it normally always exists. It
    # is mounted only when present, so a missing static folder degrades to an
    # unstyled page rather than a 500 on every request.
    if STATIC_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
        )
    else:  # pragma: no cover - only when the package is installed incomplete
        logger.warning("Static directory missing at %s", STATIC_DIR)

    from assistant.dashboard.routes import router

    app.include_router(router)

    if not settings.exists:
        # Worth saying once at startup: the dashboard will render an empty
        # state, and this is the reason for it.
        logger.warning(
            "No analytics database at the configured location yet. "
            "Start the assistant once, then reload."
        )
    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dashboard with uvicorn.

    Args:
        argv: unused. Accepted so the signature matches the project's other
            ``main`` entry points, and so a future command line can use it.

    Returns:
        A process exit code. Returning rather than calling ``sys.exit`` keeps
        this importable and testable.
    """
    import uvicorn

    logger.info(
        "Starting the read-only analytics dashboard on http://%s:%d",
        DEFAULT_HOST,
        DEFAULT_PORT,
    )
    uvicorn.run(
        create_app(),
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        log_level="info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

