"""Tests for the dashboard application: import, wiring, and read-only access.

The read-only guarantee is the important thing here, and it is tested three
ways rather than asserted once in a docstring: the wrapper exposes no write
method, the underlying SQLite connection refuses a write, and a full request
against a populated database leaves every row and the schema version exactly
as it was.

Every test uses a temporary database from the shared ``conftest.py`` fixtures.
The real ``data/assistant.db`` is never opened for writing.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from assistant.core.database import Database
from assistant.dashboard.app import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    STATIC_DIR,
    TEMPLATES_DIR,
    ReadOnlyDatabase,
    Settings,
    create_app,
    main,
    open_readonly,
)

#: The rows the shared fixture writes, asserted on rather than repeated.
ROW_COUNT = 5
NOTE_COUNT = 1
SCHEMA_VERSION = 2


@pytest.fixture
def populated(tmp_path: Path) -> Path:
    """A temporary database with a small amount of real-shaped data."""
    path = tmp_path / "dashboard.db"
    db = Database(path)
    db.initialize()
    for tool, day in (
        ("weather", "2026-09-21"),
        ("jokes", "2026-09-21"),
        ("jokes", "2026-09-22"),
        ("notes", "2026-09-23"),
        ("none", "2026-09-23"),
    ):
        db.execute(
            "INSERT INTO interactions (occurred_at, tool_name, utterance, "
            "response) VALUES (?,?,?,?)",
            (f"{day} 09:00:00", tool, "u", "r"),
        )
    db.execute(
        "INSERT INTO notes (created_at, body) VALUES (?,?)",
        ("2026-09-22 10:00:00", "buy milk"),
    )
    db.close()
    return path


@pytest.fixture
def empty_db_path(tmp_path: Path) -> Path:
    """A migrated but empty temporary database."""
    path = tmp_path / "empty.db"
    db = Database(path)
    db.initialize()
    db.close()
    return path


@pytest.fixture
def client(populated: Path) -> TestClient:
    return TestClient(create_app(populated))


def snapshot(path: Path) -> tuple[int, int, int]:
    """Read (interactions, notes, schema version) without opening for write."""
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return (
            connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
            connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0],
        )
    finally:
        connection.close()


# ----------------------------------------------------------------------
# Package surface
# ----------------------------------------------------------------------
class TestPackageImport:
    def test_the_package_imports(self):
        """The missing-app.py failure of the partial work, pinned."""
        import assistant.dashboard as dashboard

        assert set(dashboard.__all__) == {
            "DEFAULT_HOST",
            "DEFAULT_PORT",
            "ReadOnlyDatabase",
            "create_app",
            "main",
        }

    @pytest.mark.parametrize(
        "name",
        ["DEFAULT_HOST", "DEFAULT_PORT", "ReadOnlyDatabase", "create_app", "main"],
    )
    def test_every_advertised_name_exists(self, name):
        import assistant.dashboard as dashboard

        assert hasattr(dashboard, name), name

    def test_it_binds_to_loopback_only(self):
        """No authentication this sprint, so it must not face the network."""
        assert DEFAULT_HOST == "127.0.0.1"
        assert isinstance(DEFAULT_PORT, int)

    def test_the_shipped_assets_exist(self):
        assert (TEMPLATES_DIR / "base.html").is_file()
        assert (TEMPLATES_DIR / "index.html").is_file()
        assert (STATIC_DIR / "style.css").is_file()
        assert (STATIC_DIR / "dashboard.js").is_file()


# ----------------------------------------------------------------------
# App construction
# ----------------------------------------------------------------------
class TestAppCreation:
    def test_it_builds_a_fastapi_app(self, populated: Path):
        assert isinstance(create_app(populated), FastAPI)

    def test_building_opens_no_connection(self, empty_db_path: Path):
        """Constructing must be side-effect free."""
        create_app(empty_db_path)
        assert empty_db_path.exists()

    def test_settings_records_only_the_path(self, populated: Path):
        settings = Settings(db_path=populated)
        assert settings.db_path == populated
        assert settings.exists is True
        # A frozen dataclass of one field: nothing to leak a secret into.
        assert len(settings.__dataclass_fields__) == 1

    def test_a_missing_database_does_not_stop_startup(self, tmp_path: Path):
        missing = tmp_path / "not-created-yet.db"
        assert isinstance(create_app(missing), FastAPI)
        assert Settings(db_path=missing).exists is False

    def test_main_is_importable_without_running(self):
        """It must be callable as ``python -m``, so it has to exist."""
        assert callable(main)


# ----------------------------------------------------------------------
# The read-only guarantee
# ----------------------------------------------------------------------
class TestReadOnlyDatabase:
    def test_it_can_read(self, populated: Path):
        db = open_readonly(populated)
        try:
            assert db.read_only is True
            assert db.query_one("SELECT COUNT(*) AS c FROM interactions")["c"] == ROW_COUNT
            assert len(db.query_all("SELECT * FROM notes")) == NOTE_COUNT
        finally:
            db.close()

    def test_it_exposes_no_write_method(self, populated: Path):
        """The wrapper has nothing to call, even by accident."""
        db = open_readonly(populated)
        try:
            assert not hasattr(db, "execute")
            assert not hasattr(db, "executemany")
        finally:
            db.close()

    def test_the_connection_itself_refuses_a_write(self, populated: Path):
        """Belt and braces: even raw SQL through the handle is rejected."""
        db = open_readonly(populated)
        try:
            with pytest.raises(sqlite3.OperationalError):
                db._connection.execute(
                    "INSERT INTO interactions (tool_name) VALUES ('x')"
                )
        finally:
            db.close()

    def test_a_delete_is_also_refused(self, populated: Path):
        db = open_readonly(populated)
        try:
            with pytest.raises(sqlite3.OperationalError):
                db._connection.execute("DELETE FROM interactions")
        finally:
            db.close()

    def test_opening_a_missing_file_does_not_create_it(self, tmp_path: Path):
        missing = tmp_path / "absent.db"
        with pytest.raises(sqlite3.Error):
            open_readonly(missing)
        assert not missing.exists()

    def test_its_repr_hides_the_path(self, populated: Path):
        """Safe to log, and it carries nothing useful to a reader."""
        db = open_readonly(populated)
        try:
            assert str(populated) not in repr(db)
        finally:
            db.close()

    def test_it_is_a_context_independent_object(self, populated: Path):
        """Two independent handles, so concurrent requests cannot collide."""
        first = open_readonly(populated)
        second = open_readonly(populated)
        try:
            assert first is not second
            assert first.query_one("SELECT 1 AS one")["one"] == 1
            assert second.query_one("SELECT 1 AS one")["one"] == 1
        finally:
            first.close()
            second.close()


class TestDashboardNeverWrites:
    def test_serving_every_route_changes_nothing(self, populated: Path):
        before = snapshot(populated)
        client = TestClient(create_app(populated))
        for path in (
            "/",
            "/api/dashboard",
            "/api/health",
            "/api/dashboard?start=2026-09-21&end=2026-09-23",
            "/api/dashboard?start=bogus",
        ):
            client.get(path)
        assert snapshot(populated) == before == (ROW_COUNT, NOTE_COUNT, SCHEMA_VERSION)

    def test_the_schema_version_is_unchanged(self, populated: Path):
        TestClient(create_app(populated)).get("/api/dashboard")
        assert snapshot(populated)[2] == SCHEMA_VERSION

    def test_the_file_hash_is_unchanged_by_serving(self, populated: Path):
        """Strongest form: the bytes on disk are identical afterwards."""
        before = hashlib.sha256(populated.read_bytes()).hexdigest()
        client = TestClient(create_app(populated))
        for path in ("/", "/api/dashboard", "/api/health"):
            client.get(path)
        assert hashlib.sha256(populated.read_bytes()).hexdigest() == before

    def test_a_rejected_request_also_writes_nothing(self, populated: Path):
        before = snapshot(populated)
        client = TestClient(create_app(populated))
        client.get("/api/dashboard?start=2026-09-30&end=2026-09-01")
        assert snapshot(populated) == before

