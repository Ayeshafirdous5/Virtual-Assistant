"""Database lifecycle, migrations, and graceful failure handling."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from assistant.core.database import SCHEMA_VERSION, Database


class TestInitialisation:
    def test_import_creates_nothing(self, db_path: Path):
        # Constructing must not touch the disk.
        Database(db_path)
        assert not db_path.exists()

    def test_initialize_creates_file_and_parents(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "t.db"
        d = Database(nested)
        try:
            d.initialize()
            assert nested.exists()
        finally:
            d.close()

    def test_initialize_creates_expected_tables(self, db):
        tables = db.table_names()
        assert "schema_version" in tables
        assert "interactions" in tables
        assert "notes" in tables

    def test_pragmas_applied(self, db):
        conn = db.connect()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_memory_database(self, memory_db):
        assert "notes" in memory_db.table_names()
        assert ":memory:" in memory_db.path


class TestConnectionLifecycle:
    def test_connect_is_idempotent(self, memory_db):
        first = memory_db.connect()
        assert memory_db.connect() is first

    def test_close_releases(self, memory_db):
        memory_db.connect()
        assert memory_db.is_open is True
        memory_db.close()
        assert memory_db.is_open is False

    def test_close_twice_is_safe(self, memory_db):
        memory_db.close()
        memory_db.close()
        assert memory_db.is_open is False

    def test_can_reconnect_after_close(self, memory_db):
        memory_db.connect()
        memory_db.close()
        memory_db.connect()
        assert memory_db.is_open is True

    def test_context_manager(self, db_path: Path):
        with Database(db_path) as d:
            d.initialize()
            assert d.is_open is True
        assert d.is_open is False

    def test_repr_hides_internals(self, memory_db):
        text = repr(memory_db)
        assert ":memory:" in text
        assert "sqlite3" not in text


class TestMigrations:
    def test_fresh_database_gets_latest_version(self, memory_db):
        assert memory_db.schema_version() == SCHEMA_VERSION

    def test_schema_version_constant_is_two(self):
        assert SCHEMA_VERSION == 2

    def test_all_versions_recorded_once(self, memory_db):
        assert memory_db.applied_versions() == {1, 2}

    def test_initialize_is_idempotent(self, memory_db):
        for _ in range(3):
            memory_db.initialize()
        assert memory_db.schema_version() == SCHEMA_VERSION
        rows = memory_db.query_all("SELECT version FROM schema_version")
        assert len(rows) == 2

    def test_v1_database_upgrades_to_v2_in_place(self, db_path: Path):
        """Simulate a real Phase 3 Step 1 database and upgrade it."""
        from assistant.core.database import _MIGRATIONS

        d = Database(db_path)
        d.connect()
        try:
            # Apply only migration 1, exactly as the old code did.
            d.connect().executescript(_MIGRATIONS[0][1])
            d.execute("INSERT INTO schema_version (version) VALUES (1)")
            d.execute(
                "INSERT INTO interactions (tool_name, utterance, response) "
                "VALUES (?,?,?)",
                ("jokes", "tell me a joke", "setup | punchline"),
            )
            assert d.schema_version() == 1
            assert "notes" not in d.table_names()
            before = [r["id"] for r in d.query_all("SELECT id FROM interactions")]

            # Now upgrade in place.
            assert d.initialize() == 2
            assert "notes" in d.table_names()

            after = [r["id"] for r in d.query_all("SELECT id FROM interactions")]
            assert after == before, "existing interactions must survive"
            row = d.query_one("SELECT utterance FROM interactions")
            assert row["utterance"] == "tell me a joke"
        finally:
            d.close()

    def test_upgrade_is_idempotent_on_existing_file(self, db_path: Path):
        d = Database(db_path)
        try:
            d.initialize()
            d.execute("INSERT INTO notes (body) VALUES ('keep me')")
            d.initialize()
            d.initialize()
            assert d.schema_version() == 2
            assert d.query_one("SELECT COUNT(*) AS n FROM notes")["n"] == 1
        finally:
            d.close()

    def test_newer_schema_is_preserved(self, memory_db):
        memory_db.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (99,)
        )
        memory_db.initialize()
        assert 99 in memory_db.applied_versions()


class TestQueryHelpers:
    def test_execute_and_query_one(self, memory_db):
        memory_db.execute("INSERT INTO notes (body) VALUES ('a')")
        assert memory_db.query_one("SELECT body FROM notes")["body"] == "a"

    def test_query_one_returns_none_when_absent(self, memory_db):
        assert memory_db.query_one("SELECT body FROM notes WHERE id = 999") is None

    def test_query_all(self, memory_db):
        memory_db.executemany(
            "INSERT INTO notes (body) VALUES (?)", [("a",), ("b",)]
        )
        assert len(memory_db.query_all("SELECT body FROM notes")) == 2

    def test_bad_sql_raises(self, memory_db):
        with pytest.raises(sqlite3.Error):
            memory_db.query_all("SELECT * FROM table_that_does_not_exist")


class TestGracefulFailure:
    def test_unopenable_path_raises_at_initialize(self, tmp_path: Path):
        # A directory where a file is expected.
        d = Database(tmp_path)
        with pytest.raises(sqlite3.Error):
            d.initialize()

    def test_closed_database_reopens_lazily(self, memory_db):
        memory_db.close()
        memory_db.query_all("SELECT 1")
        assert memory_db.is_open is True
