"""Read and write access for the ``notes`` table.

A small class rather than loose functions, because notes are the first
feature that both reads and writes, and grouping them keeps the SQL in one
place. The tool layer never sees a query.

Nothing here catches database errors: the caller decides what a failure
means. :class:`~assistant.tools.notes.NoteTool` turns any of them into a
sentence, so a broken database can never stop the assistant.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

#: Refuse to store an unbounded note, so one command cannot fill the file.
MAX_BODY_LENGTH = 500

_COLUMNS = "id, created_at, body"


@dataclass(frozen=True)
class Note:
    """One stored note."""

    id: int
    created_at: str
    body: str


def _to_note(row: sqlite3.Row) -> Note:
    """Convert a database row into a :class:`Note`."""
    return Note(
        id=int(row["id"]),
        created_at=str(row["created_at"] or ""),
        body=str(row["body"] or ""),
    )


class NoteRepository:
    """Create, read and delete notes for one database.

    Args:
        db: an open :class:`~assistant.core.database.Database`.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    def add(self, body: str) -> Note:
        """Store a new note and return it with its assigned id.

        Args:
            body: the note text. Whitespace is trimmed and truncated to
                :data:`MAX_BODY_LENGTH`.

        Returns:
            The stored :class:`Note`.

        Raises:
            ValueError: if ``body`` is empty once trimmed.
            sqlite3.Error: if the insert fails.
        """
        text = " ".join((body or "").split())[:MAX_BODY_LENGTH]
        if not text:
            raise ValueError("A note cannot be empty.")

        cursor = self._db.execute(
            "INSERT INTO notes (body) VALUES (?)", (text,)
        )
        return Note(id=int(cursor.lastrowid or 0), created_at="", body=text)

    def list_notes(self, limit: int | None = None) -> list[Note]:
        """Return notes, newest first.

        Args:
            limit: maximum number of notes to return. ``None`` returns all.

        Returns:
            A list of :class:`Note`, newest first. Empty when there are none.

        Raises:
            sqlite3.Error: if the query fails.
        """
        sql = f"SELECT {_COLUMNS} FROM notes ORDER BY id DESC"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        return [_to_note(row) for row in self._db.query_all(sql, params)]

    def get(self, note_id: int) -> Note | None:
        """Return one note by id, or None when it does not exist."""
        row = self._db.query_one(
            f"SELECT {_COLUMNS} FROM notes WHERE id = ?", (note_id,)
        )
        return _to_note(row) if row else None

    def delete(self, note_id: int) -> bool:
        """Delete one note by id.

        Returns:
            True if a note was removed, False if the id did not exist.

        Raises:
            sqlite3.Error: if the delete fails.
        """
        cursor = self._db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return bool(cursor.rowcount)

    def count(self) -> int:
        """Return how many notes are stored."""
        row = self._db.query_one("SELECT COUNT(*) AS total FROM notes")
        return int(row["total"]) if row else 0
