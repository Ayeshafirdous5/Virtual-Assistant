"""Reusable, deterministic fixtures for building analytics test data.

The analytics tests all need the same shapes: interactions spread over several
days, a mix of matched and unmatched rows, notes on several days, and an empty
database. Building those rows inline in every module means every test file
reinvents the same helpers and, worse, invents slightly different timestamps.

This module is that shared vocabulary. Everything here writes **explicit**
timestamps rather than using ``datetime('now')``, so a test asserting on a
day, a week or a date range produces the same result whenever the suite runs.
No assertion anywhere in the analytics tests should depend on the wall clock.

Nothing here opens the real database. Each helper takes the
``memory_db``/``db`` fixture from ``conftest.py``, which is always a temporary
in-memory or ``tmp_path`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

#: A Monday. Chosen deliberately: ISO week boundaries are cleanest when the
#: week starts here, so week-bucketing tests read clearly.
ANCHOR = date(2026, 9, 21)


def stamp(day: date, hour: int = 12, minute: int = 0) -> str:
    """Return a stored timestamp for ``day`` at the given time.

    Returns a ``"YYYY-MM-DD HH:MM:SS"`` string in exactly the format the
    ``interactions.occurred_at`` and ``notes.created_at`` columns use.
    """
    return f"{day.isoformat()} {hour:02d}:{minute:02d}:00"


def day_at(offset: int, hour: int = 12) -> str:
    """Return a timestamp ``offset`` days after :data:`ANCHOR`.

    ``offset`` may be negative, which is how a test places data *before* the
    anchor and proves a range's lower bound is honoured.
    """
    return stamp(ANCHOR + timedelta(days=offset), hour)


def add_interaction(
    db,
    tool: str = "weather",
    day: date | None = None,
    hour: int = 12,
) -> None:
    """Insert one interaction with an explicit timestamp.

    Pass ``"none"`` or ``"unmatched"`` as ``tool`` to record a command that
    matched nothing.
    """
    db.execute(
        "INSERT INTO interactions (occurred_at, tool_name, utterance, response) "
        "VALUES (?,?,?,?)",
        (stamp(day or ANCHOR, hour), tool, "an utterance", "a response"),
    )


def add_interactions(
    db, tool: str, count: int, day: date | None = None
) -> None:
    """Insert ``count`` interactions on one day, at distinct hours."""
    target = day or ANCHOR
    for index in range(count):
        add_interaction(db, tool, day=target, hour=8 + index % 12)


def add_note(
    db, body: str = "buy milk", day: date | None = None, hour: int = 12
) -> None:
    """Insert one note with an explicit timestamp."""
    db.execute(
        "INSERT INTO notes (created_at, body) VALUES (?,?)",
        (stamp(day or ANCHOR, hour), body),
    )


@dataclass
class Dataset:
    """A database plus the helpers needed to work with it.

    Returned by :func:`analytics_fixture` so a test can write ``data.day(-1)``
    instead of recomputing dates inline, which keeps the interesting part of a
    test about the assertion rather than the calendar.
    """

    db: object

    def interaction(
        self, tool: str = "weather", day: date | None = None, hour: int = 12
    ) -> "Dataset":
        """Add one interaction and return ``self`` so calls can chain."""
        add_interaction(self.db, tool, day, hour)
        return self

    def interactions(
        self, tool: str, count: int, day: date | None = None
    ) -> "Dataset":
        """Add ``count`` interactions on one day and return ``self``."""
        add_interactions(self.db, tool, count, day)
        return self

    def note(
        self, body: str = "buy milk", day: date | None = None, hour: int = 12
    ) -> "Dataset":
        """Add one note and return ``self``."""
        add_note(self.db, body, day, hour)
        return self

    def day(self, offset: int) -> date:
        """Return the date ``offset`` days after :data:`ANCHOR`."""
        return ANCHOR + timedelta(days=offset)

    def when(self, offset: int, hour: int = 12) -> str:
        """Return a stored timestamp ``offset`` days after :data:`ANCHOR`."""
        return day_at(offset, hour)


def analytics_fixture(db) -> Dataset:
    """Return a :class:`Dataset` wrapping a temporary database."""
    return Dataset(db)


# ----------------------------------------------------------------------
# Ready-made shapes, for the cases nearly every new test needs
# ----------------------------------------------------------------------
def empty(db) -> Dataset:
    """A database with nothing in it at all."""
    return analytics_fixture(db)


def two_tools_over_three_days(db) -> Dataset:
    """Two tools across three days, plus one unmatched row.

    The standard "there is real activity" shape::

        day 0   weather x2, jokes x1
        day 1   jokes x2
        day 2   weather x1, none x1

    That is 7 interactions over 3 days: 6 matched and 1 unmatched, with
    ``jokes`` and ``weather`` tied on 3 each and the alphabetical rule
    resolving it to ``jokes``. All three days fall in ISO week 2026-W39.
    """
    data = analytics_fixture(db)
    data.interactions("weather", 2, day=data.day(0))
    data.interactions("jokes", 1, day=data.day(0))
    data.interactions("jokes", 2, day=data.day(1))
    data.interactions("weather", 1, day=data.day(2))
    data.interaction("none", day=data.day(2))
    return data


def tie_between_two_tools(db) -> Dataset:
    """Two tools with identical counts, for tie-break tests.

    Each appears exactly three times, so the winner must come from the
    alphabetical rule rather than from the count.
    """
    data = analytics_fixture(db)
    data.interactions("weather", 3, day=data.day(0))
    data.interactions("jokes", 3, day=data.day(1))
    return data


def notes_over_three_days(db) -> Dataset:
    """Three notes on three different days, with known body lengths.

    The bodies are 8, 18 and 4 characters, giving a total of 30 and a mean of
    ``10.0`` -- a value a test can state exactly rather than recompute. The
    longest is ``"a longer note here"``.
    """
    data = analytics_fixture(db)
    data.note("buy milk", day=data.day(0))
    data.note("a longer note here", day=data.day(1))
    data.note("milk", day=data.day(2))
    return data


def only_unmatched(db) -> Dataset:
    """Interactions that matched nothing at all, using both sentinels."""
    data = analytics_fixture(db)
    data.interaction("none")
    data.interaction("none")
    data.interaction("unmatched")
    return data

