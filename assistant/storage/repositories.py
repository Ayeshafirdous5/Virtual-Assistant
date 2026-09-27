"""Typed read helpers for the ``interactions`` table.

Plain functions over :class:`~assistant.core.database.Database`. Each one
returns a dataclass or a plain value so callers never touch raw
``sqlite3.Row`` objects, and none of them format text for speech. Turning
rows into sentences is the caller's job, which keeps this layer reusable by
a future dashboard or stats command.

Only the reads the assistant actually needs are here. Date-range filtering,
full-text search and aggregation beyond a simple count-by-tool are
deliberately absent until something needs them.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

# Used when a row has no tool name, so callers get a string not an None.
UNKNOWN_TOOL = "unmatched"

_COLUMNS = "id, occurred_at, tool_name, utterance, response"


@dataclass(frozen=True)
class Interaction:
    """One recorded command."""

    id: int
    occurred_at: str
    tool_name: str
    utterance: str
    response: str

    @property
    def is_matched(self) -> bool:
        """True when a tool actually handled the command."""
        return self.tool_name not in ("", UNKNOWN_TOOL)


@dataclass(frozen=True)
class ToolCount:
    """How many commands one tool handled."""

    tool_name: str
    count: int


def _to_interaction(row: sqlite3.Row) -> Interaction:
    """Convert a database row into an :class:`Interaction`."""
    return Interaction(
        id=int(row["id"]),
        occurred_at=str(row["occurred_at"] or ""),
        tool_name=str(row["tool_name"] or UNKNOWN_TOOL),
        utterance=str(row["utterance"] or ""),
        response=str(row["response"] or ""),
    )


def recent_interactions(
    db: Any, limit: int = 5, tool_name: str | None = None
) -> list[Interaction]:
    """Return the most recent commands, newest first.

    Args:
        db: an open :class:`~assistant.core.database.Database`.
        limit: maximum number of rows to return.
        tool_name: when given, only that tool's commands are returned.

    Returns:
        Up to ``limit`` interactions, newest first. Empty when there is no
        history yet.

    Raises:
        sqlite3.Error: if the query fails. Callers that must not fail should
            catch it, as the history tool does.
    """
    sql = f"SELECT {_COLUMNS} FROM interactions"
    params: list[Any] = []

    if tool_name:
        sql += " WHERE tool_name = ?"
        params.append(tool_name)

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, int(limit)))

    return [_to_interaction(row) for row in db.query_all(sql, params)]


def count_by_tool(db: Any) -> list[ToolCount]:
    """Return how many commands each tool handled, most used first.

    Commands that matched no tool are grouped as ``"unmatched"``.

    Returns:
        Counts ordered by descending count, then by name for a stable order.

    Raises:
        sqlite3.Error: if the query fails.
    """
    rows = db.query_all(
        "SELECT COALESCE(NULLIF(tool_name, ''), ?) AS name, COUNT(*) AS total "
        "FROM interactions GROUP BY name ORDER BY total DESC, name ASC",
        (UNKNOWN_TOOL,),
    )
    return [
        ToolCount(tool_name=str(row["name"]), count=int(row["total"]))
        for row in rows
    ]


def total_interactions(db: Any) -> int:
    """Return the total number of recorded commands."""
    row = db.query_one("SELECT COUNT(*) AS total FROM interactions")
    return int(row["total"]) if row else 0
