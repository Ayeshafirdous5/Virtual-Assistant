"""Read-only analytics over the SQLite history and notes tables.

This is the first layer that *aggregates* rather than reads a page of rows.
:mod:`assistant.storage` answers "what happened recently"; this module answers
"how much has happened, and when". Both read the same two tables, and this
module adds no new data of its own.

Design rules
------------
* **No SQL outside this package.** Tools call typed methods here and never see
  a query string, exactly as :mod:`assistant.storage` already guarantees.
* **No invented metrics.** Every value returned is a fact the current schema
  can support. Where the schema cannot answer a question, the answer is
  documented as a limitation rather than approximated -- see
  :data:`NOT_A_TOOL` and the notes section of
  :class:`AnalyticsService`.
* **Deterministic ordering.** Counts tie-break on name, and time series are
  returned in chronological order, so the same database always produces the
  same report and a test can assert on it directly.
* **Errors propagate.** This layer does not catch database errors, for the
  same reason :mod:`assistant.storage.notes` does not: the caller decides
  what a failure means. :class:`~assistant.tools.analytics.AnalyticsTool`
  turns one into a spoken sentence.
* **Read-only.** Nothing here writes, so calling a method can never change
  the user's history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from assistant.storage.notes import NoteRepository
from assistant.storage.repositories import ToolCount, count_by_tool

#: Values that mean "no tool handled this", and so are not a tool.
#:
#: Two different sentinels exist in real data and both are honoured here.
#: :mod:`assistant.app` records an unresolved command as the literal string
#: ``"none"``, while :mod:`assistant.storage.repositories` maps a ``NULL`` or
#: empty ``tool_name`` to ``"unmatched"``. Counting either as a tool would
#: inflate :meth:`AnalyticsService.unique_tools_used` and could make an
#: unmatched command look like the most-used feature.
NOT_A_TOOL = frozenset({"", "none", "unmatched"})

#: Longest note body the notes repository will store. Mirrored here so a
#: caller can report the limit without importing the storage layer.
MAX_NOTE_LENGTH = 500


@dataclass(frozen=True)
class TimeBucket:
    """One period of activity.

    Attributes:
        period: the bucket label, an ISO date for daily counts
            (``"2026-09-26"``) or an ISO week for weekly counts
            (``"2026-W38"``).
        count: how many interactions fall in that period.
    """

    period: str
    count: int

    def __repr__(self) -> str:
        return f"TimeBucket(period={self.period!r}, count={self.count})"


@dataclass(frozen=True)
class AnalyticsSummary:
    """The headline figures, in one immutable object.

    Every field is a measured fact. ``None`` means "no data to report",
    which is different from zero: zero means "measured, and it was nothing".
    """

    #: How many interactions have been recorded, matched or not.
    total_interactions: int
    #: How many distinct real tools have handled a command.
    unique_tools: int
    #: The most-used tool name, or ``None`` when nothing is recorded.
    most_used_tool: str | None
    #: How many times :attr:`most_used_tool` ran. Zero when there is none.
    most_used_count: int
    #: Earliest recorded timestamp, or ``None`` when nothing is recorded.
    first_interaction: str | None
    #: Latest recorded timestamp, or ``None`` when nothing is recorded.
    last_interaction: str | None
    #: How many notes are stored.
    total_notes: int
    #: How many distinct days have at least one interaction.
    active_days: int

    @property
    def has_activity(self) -> bool:
        """True when at least one interaction has been recorded."""
        return self.total_interactions > 0

    def __repr__(self) -> str:
        return (
            "AnalyticsSummary("
            f"total_interactions={self.total_interactions}, "
            f"unique_tools={self.unique_tools}, "
            f"most_used_tool={self.most_used_tool!r}, "
            f"most_used_count={self.most_used_count}, "
            f"first_interaction={self.first_interaction!r}, "
            f"last_interaction={self.last_interaction!r}, "
            f"total_notes={self.total_notes}, "
            f"active_days={self.active_days})"
        )


class AnalyticsService:
    """Typed analytics over one database.

    Args:
        db: an initialised
            :class:`~assistant.core.database.Database`. Typed as ``Any`` to
            match :mod:`assistant.storage`, which does the same so the
            storage layer never has to import the database module.

    Every method is read-only. An empty database is not an error: each one
    returns a zero, an empty list, or ``None``, so a caller never needs a
    special case for "no data yet".

    Raises:
        sqlite3.Error: if a query fails, including when the schema has not
            been created. Nothing is caught here by design; the tool layer
            decides how to report a failure.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # A. Interaction totals
    # ------------------------------------------------------------------
    def total_interactions(self) -> int:
        """Return how many interactions are recorded, matched or not.

        This counts rows in ``interactions``, which includes commands that
        matched no tool. It is the honest "how much have you used me" number;
        use :meth:`command_counts` to see what actually did the work.
        """
        row = self._db.query_one("SELECT COUNT(*) AS total FROM interactions")
        return int(row["total"]) if row else 0

    def first_interaction_time(self) -> str | None:
        """Return the earliest recorded timestamp, or ``None`` if empty.

        The stored format is ``"YYYY-MM-DD HH:MM:SS"``, so a plain string
        comparison is also a chronological one and no date parsing is needed
        for the ordering itself.
        """
        return self._time_extreme("MIN")

    def last_interaction_time(self) -> str | None:
        """Return the latest recorded timestamp, or ``None`` if empty."""
        return self._time_extreme("MAX")

    def _time_extreme(self, function: str) -> str | None:
        """Run ``MIN`` or ``MAX`` over ``occurred_at`` and normalise the result.

        ``function`` is never taken from a caller: it is only ever the two
        literals passed above, so the string cannot become an injection
        point. SQLite returns one row with ``NULL`` for an empty table, which
        is why the row itself exists but the value can still be ``None``.
        """
        row = self._db.query_one(
            f"SELECT {function}(occurred_at) AS value FROM interactions"
        )
        if row is None:
            return None
        value = row["value"]
        return str(value) if value else None

    # ------------------------------------------------------------------
    # B. Tool usage
    # ------------------------------------------------------------------
    def command_counts(self) -> list[ToolCount]:
        """Return per-tool counts, most used first, ties broken by name.

        Unresolved commands are grouped under one of the
        :data:`NOT_A_TOOL` sentinels and are **kept** here rather than
        dropped, because they are real rows and hiding them would make the
        counts fail to sum to :meth:`total_interactions`. Use
        :meth:`matched_command_counts` for the tool-only view.
        """
        return count_by_tool(self._db)

    def matched_command_counts(self) -> list[ToolCount]:
        """Return per-tool counts with unresolved commands removed.

        This is the list to show a user who asked what they use, because
        ``none`` is not a feature. Ordering is inherited from
        :meth:`command_counts`, so it stays most-used-first then alphabetical.
        """
        return [
            count
            for count in self.command_counts()
            if count.tool_name not in NOT_A_TOOL
        ]

    def unique_tools_used(self) -> int:
        """Return how many distinct real tools have handled a command.

        Unresolved commands do not count: they are not tools. So a database
        holding only unmatched commands reports ``0`` unique tools, which is
        the truthful answer, not a bug.
        """
        return len(self.matched_command_counts())

    def most_used_command(self) -> ToolCount | None:
        """Return the most-used tool, or ``None`` when nothing is recorded.

        Ties are resolved by :meth:`command_counts`, which already orders by
        descending count then ascending name, so a tie always yields the same
        answer for the same data rather than whichever row SQLite happens to
        emit first. Unresolved commands are excluded: reporting ``none`` as
        the most-used feature would be nonsense.
        """
        matched = self.matched_command_counts()
        return matched[0] if matched else None

    def unmatched_count(self) -> int:
        """Return how many recorded interactions matched no tool."""
        return sum(
            count.count
            for count in self.command_counts()
            if count.tool_name in NOT_A_TOOL
        )

    # ------------------------------------------------------------------
    # C. Time-based activity
    # ------------------------------------------------------------------
    def daily_interaction_counts(
        self, limit: int | None = None
    ) -> list[TimeBucket]:
        """Return interactions per calendar day, oldest day first.

        Args:
            limit: when given, only the most recent ``limit`` days are
                returned, still in chronological order.

        Days with no activity are **not** included. They are not missing
        data, and inserting explicit zeros would imply the assistant knows
        about days it was never run, which it does not.
        """
        rows = self._grouped_counts("date(occurred_at)", limit)
        return [
            TimeBucket(period=str(row["period"]), count=int(row["total"]))
            for row in rows
        ]

    def weekly_interaction_counts(
        self, limit: int | None = None
    ) -> list[TimeBucket]:
        """Return interactions per ISO week, oldest week first.

        Weeks are ISO-8601 (``2026-W38``) and are computed in Python rather
        than with ``strftime('%V')``, which returns NULL on the SQLite
        3.45.3 this project runs against. Doing it in Python also makes the
        week boundary testable without depending on the local clock.

        Empty weeks are omitted, for the same reason as in
        :meth:`daily_interaction_counts`.
        """
        rows = self._db.query_all(
            "SELECT occurred_at FROM interactions "
            "WHERE occurred_at IS NOT NULL AND occurred_at != ''"
        )

        totals: dict[str, int] = {}
        for row in rows:
            label = _iso_week_label(str(row["occurred_at"]))
            if label is not None:
                totals[label] = totals.get(label, 0) + 1

        ordered = sorted(totals.items())
        if limit is not None:
            ordered = ordered[-max(1, int(limit)) :]
        return [
            TimeBucket(period=label, count=count) for label, count in ordered
        ]

    def _grouped_counts(
        self, expression: str, limit: int | None
    ) -> list[dict[str, Any]]:
        """Group interactions by a date expression, chronologically.

        ``expression`` is always a literal defined in this module, never
        caller input, so the interpolation is safe. Ordering is ascending on
        the expression, which is chronological for the stored date format.
        """
        sql = (
            f"SELECT {expression} AS period, COUNT(*) AS total "
            f"FROM interactions WHERE {expression} IS NOT NULL "
            "GROUP BY period ORDER BY period ASC"
        )
        rows = [dict(row) for row in self._db.query_all(sql)]
        if limit is not None:
            rows = rows[-max(1, int(limit)) :]
        return rows

    def busiest_day(self) -> TimeBucket | None:
        """Return the single busiest day, or ``None`` when empty.

        Ties are broken by the later date, so repeated calls on unchanged
        data always agree.
        """
        days = self.daily_interaction_counts()
        if not days:
            return None
        return max(days, key=lambda bucket: (bucket.count, bucket.period))

    def active_days(self) -> int:
        """Return how many distinct days have at least one interaction."""
        row = self._db.query_one(
            "SELECT COUNT(DISTINCT date(occurred_at)) AS total FROM interactions"
        )
        return int(row["total"]) if row else 0

    # ------------------------------------------------------------------
    # D. Summary
    # ------------------------------------------------------------------
    def summary(self) -> AnalyticsSummary:
        """Return the headline figures in one object.

        Fields are read from the same committed data, so the numbers within
        one summary are mutually consistent.
        """
        top = self.most_used_command()
        return AnalyticsSummary(
            total_interactions=self.total_interactions(),
            unique_tools=self.unique_tools_used(),
            most_used_tool=top.tool_name if top else None,
            most_used_count=top.count if top else 0,
            first_interaction=self.first_interaction_time(),
            last_interaction=self.last_interaction_time(),
            total_notes=self.total_notes(),
            active_days=self.active_days(),
        )

    # ------------------------------------------------------------------
    # E. Notes analytics
    # ------------------------------------------------------------------
    def total_notes(self) -> int:
        """Return how many notes are stored."""
        return NoteRepository(self._db).count()

    def active_note_count(self) -> int:
        """Return how many notes are currently stored.

        The ``notes`` table has no status, archive or completion column, so
        "active" can only mean "present". A future schema that adds one would
        need this method to filter on it; today returning the total is the
        only honest answer, and the name is kept because it is the question
        callers actually ask.

        Documented limitation: notes cannot be marked done or archived.
        """
        return self.total_notes()

    def first_note_time(self) -> str | None:
        """Return the earliest note timestamp, or None when there are none."""
        return self._note_time_extreme("MIN")

    def last_note_time(self) -> str | None:
        """Return the latest note timestamp, or None when there are none."""
        return self._note_time_extreme("MAX")

    def _note_time_extreme(self, function: str) -> str | None:
        """Run ``MIN`` or ``MAX`` over ``notes.created_at``.

        As with :meth:`_time_extreme`, ``function`` is only ever one of the
        two literals passed by the callers above.
        """
        row = self._db.query_one(
            f"SELECT {function}(created_at) AS value FROM notes"
        )
        if row is None:
            return None
        value = row["value"]
        return str(value) if value else None

    def daily_note_counts(self, limit: int | None = None) -> list[TimeBucket]:
        """Return notes created per calendar day, oldest day first.

        Supported by the existing ``created_at`` column, so no schema change
        was needed. Days with no note are omitted, as elsewhere.
        """
        sql = (
            "SELECT date(created_at) AS period, COUNT(*) AS total "
            "FROM notes WHERE date(created_at) IS NOT NULL "
            "GROUP BY period ORDER BY period ASC"
        )
        rows = [dict(row) for row in self._db.query_all(sql)]
        if limit is not None:
            rows = rows[-max(1, int(limit)) :]
        return [
            TimeBucket(period=str(row["period"]), count=int(row["total"]))
            for row in rows
        ]

    def longest_note_length(self) -> int:
        """Return the length of the longest stored note body.

        Reported because the repository truncates at
        :data:`~assistant.storage.notes.MAX_BODY_LENGTH`, so a very long note
        is silently shortened. This makes that visible without changing it.
        """
        row = self._db.query_one("SELECT MAX(LENGTH(body)) AS longest FROM notes")
        return int(row["longest"]) if row and row["longest"] else 0

    def __repr__(self) -> str:
        # The Database repr carries only a path and an open flag, so the
        # service can be logged without leaking any row content.
        return f"AnalyticsService(db={self._db!r})"


def _iso_week_label(timestamp: str) -> str | None:
    """Return the ``YYYY-Www`` ISO week label for a stored timestamp.

    Returns ``None`` for anything unparseable rather than raising, so one
    malformed row cannot break a whole report. Only the date part is read,
    which keeps the result independent of the process timezone: the stored
    value is already UTC text, and reinterpreting it locally would move an
    interaction to a different day near midnight.
    """
    text = timestamp.strip()
    if len(text) < 10:
        return None
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError:
        return None
    iso_year, iso_week, _ = parsed.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"




