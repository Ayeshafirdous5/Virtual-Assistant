"""Storage layer: typed reads over the SQLite tables.

Repositories are plain functions plus small dataclasses. There is no ORM and
no query builder, because the schema is three tables and an abstraction
layer would cost more than it saves.

Adding a module here is the only thing needed to grow the storage layer.
"""

from assistant.storage.notes import Note, NoteRepository
from assistant.storage.repositories import (
    Interaction,
    count_by_tool,
    recent_interactions,
    total_interactions,
)

__all__ = [
    "Interaction",
    "Note",
    "NoteRepository",
    "count_by_tool",
    "recent_interactions",
    "total_interactions",
]
