"""Repository reads and note CRUD."""

from __future__ import annotations

import pytest

from assistant.storage import (
    Note,
    NoteRepository,
    count_by_tool,
    recent_interactions,
    total_interactions,
)


def record(db, tool="jokes", utterance="a joke", response="a punchline"):
    db.execute(
        "INSERT INTO interactions (tool_name, utterance, response) VALUES (?,?,?)",
        (tool, utterance, response),
    )


class TestInteractionReads:
    def test_empty_database(self, memory_db):
        assert total_interactions(memory_db) == 0
        assert recent_interactions(memory_db) == []
        assert count_by_tool(memory_db) == []

    def test_recent_is_newest_first(self, memory_db):
        record(memory_db, utterance="first")
        record(memory_db, utterance="second")
        items = recent_interactions(memory_db)
        assert [i.utterance for i in items] == ["second", "first"]

    def test_recent_respects_limit(self, memory_db):
        for i in range(5):
            record(memory_db, utterance=f"u{i}")
        assert len(recent_interactions(memory_db, limit=2)) == 2

    def test_recent_filters_by_tool(self, memory_db):
        record(memory_db, tool="jokes")
        record(memory_db, tool="facts")
        items = recent_interactions(memory_db, tool_name="jokes")
        assert len(items) == 1
        assert items[0].tool_name == "jokes"

    def test_count_by_tool_orders_by_count(self, memory_db):
        record(memory_db, tool="facts")
        record(memory_db, tool="jokes")
        record(memory_db, tool="jokes")
        counts = {c.tool_name: c.count for c in count_by_tool(memory_db)}
        assert counts == {"jokes": 2, "facts": 1}
        assert count_by_tool(memory_db)[0].tool_name == "jokes"

    def test_null_tool_becomes_unmatched(self, memory_db):
        record(memory_db, tool=None)
        items = recent_interactions(memory_db)
        assert items[0].tool_name == "unmatched"
        assert items[0].is_matched is False

    def test_matched_flag(self, memory_db):
        record(memory_db, tool="jokes")
        assert recent_interactions(memory_db)[0].is_matched is True

    def test_total_counts_rows(self, memory_db):
        for _ in range(3):
            record(memory_db)
        assert total_interactions(memory_db) == 3


class TestNoteRepository:
    def test_add_returns_note(self, memory_db):
        note = NoteRepository(memory_db).add("buy milk")
        assert isinstance(note, Note)
        assert note.body == "buy milk"
        assert note.id > 0

    def test_add_normalises_whitespace(self, memory_db):
        note = NoteRepository(memory_db).add("  call   the   dentist  ")
        assert note.body == "call the dentist"

    def test_add_truncates_long_body(self, memory_db):
        note = NoteRepository(memory_db).add("x" * 900)
        assert len(note.body) == 500

    def test_add_rejects_empty(self, memory_db):
        with pytest.raises(ValueError):
            NoteRepository(memory_db).add("   ")

    def test_list_is_newest_first(self, memory_db):
        repo = NoteRepository(memory_db)
        repo.add("first")
        repo.add("second")
        assert [n.body for n in repo.list_notes()] == ["second", "first"]

    def test_list_respects_limit(self, memory_db):
        repo = NoteRepository(memory_db)
        for i in range(4):
            repo.add(f"n{i}")
        assert len(repo.list_notes(limit=2)) == 2

    def test_list_all(self, memory_db):
        repo = NoteRepository(memory_db)
        for i in range(3):
            repo.add(f"n{i}")
        assert len(repo.list_notes()) == 3

    def test_get(self, memory_db):
        repo = NoteRepository(memory_db)
        note = repo.add("find me")
        assert repo.get(note.id).body == "find me"

    def test_get_missing_returns_none(self, memory_db):
        assert NoteRepository(memory_db).get(999) is None

    def test_delete_existing(self, memory_db):
        repo = NoteRepository(memory_db)
        note = repo.add("bye")
        assert repo.delete(note.id) is True
        assert repo.get(note.id) is None
        assert repo.count() == 0

    def test_delete_missing_returns_false(self, memory_db):
        assert NoteRepository(memory_db).delete(999) is False

    def test_count(self, memory_db):
        repo = NoteRepository(memory_db)
        assert repo.count() == 0
        repo.add("a")
        repo.add("b")
        assert repo.count() == 2
