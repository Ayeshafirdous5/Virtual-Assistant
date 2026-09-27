"""Shared pytest fixtures.

The most important guarantee here: **no test touches the real database.**
Every fixture that needs storage points at ``tmp_path``, and the ``config``
fixture rewrites ``db_path`` before anything can open it. Tests also never
open a sound card, a browser or a socket.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from assistant.config import get_config
from assistant.core.context import AppContext
from assistant.core.database import Database
from assistant.speech.text import TextListener, TextSpeaker

# Absolute path to the project root, so imports work however pytest is run.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A throwaway database path inside pytest's temporary directory."""
    return tmp_path / "test.db"


@pytest.fixture
def config(db_path: Path):
    """Real configuration with the database path pointed at a temp file.

    Every other setting is the real one, so tests exercise production
    defaults while never opening the user's database.
    """
    return dataclasses.replace(
        get_config(),
        db_path=db_path,
        log_file=db_path.parent / "test.log",
        log_dir=db_path.parent,
    )


@pytest.fixture
def ctx(config) -> AppContext:
    """A context with no speech and no database attached yet."""
    return AppContext(config=config)


@pytest.fixture
def db(db_path: Path):
    """An initialised database, always closed when the test finishes."""
    database = Database(db_path)
    database.initialize()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def memory_db():
    """A migrated in-memory database, always closed afterwards."""
    database = Database(":memory:")
    database.initialize()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def ctx_with_db(ctx: AppContext, db) -> AppContext:
    """A context with the temporary database attached."""
    ctx.db = db
    return ctx


@pytest.fixture
def run_text_session(config, db):
    """Drive :func:`assistant.app.run` with scripted text input.

    Returns a callable taking a list of replies. Speech is stubbed, so no
    sound is produced and the transcript is captured for assertions.
    """

    def _run(replies, database=None):
        from assistant.app import run
        from assistant.core.router import Router
        from assistant.tools import build_default_router

        spoken: list[str] = []
        ctx = AppContext(config=config)
        ctx.listener = TextListener(responses=list(replies))

        class RecordingSpeaker:
            def speak(self, text: str) -> None:
                spoken.append(text)

        ctx.speaker = RecordingSpeaker()
        if database is not None:
            ctx.db = database

        code = run(ctx, build_default_router())
        return code, spoken

    return _run


class ExplodingSpeaker:
    """A Speaker that fails the test if anything tries to use it.

    Used to prove that a tool never speaks for itself.
    """

    def speak(self, text: str) -> None:
        raise AssertionError("A tool must not call ctx.speaker.speak()")

    def listen(self) -> str:
        raise AssertionError("A tool must not call ctx.listener.listen()")
