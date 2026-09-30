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
from datetime import date, timedelta
from typing import Any

from assistant.storage.notes import NoteRepository
from assistant.storage.repositories import ToolCount, UNKNOWN_TOOL

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
class DateRange:
    """An inclusive span of calendar days, or "all time".

    Used as the optional filter on the analytics methods. Three ways to build
    one, all named so no caller has to remember which end is inclusive:

    * :meth:`all` -- no filter at all, which is the default everywhere.
    * :meth:`last_days` -- the ``n`` days ending on a given day.
    * :meth:`between` -- two explicit dates.

    Both bounds are **inclusive**, matching how a person says "on the 26th".
    An open bound is allowed (a range may start before the data begins), and
    ``None`` on both sides means all time.

    Attributes:
        start: first day included, or ``None`` for no lower bound.
        end: last day included, or ``None`` for no upper bound.
    """

    start: date | None = None
    end: date | None = None

    @classmethod
    def all(cls) -> "DateRange":
        """Return a range that filters nothing."""
        return cls()

    @classmethod
    def between(cls, start: date, end: date) -> "DateRange":
        """Return the inclusive range ``start`` to ``end``.

        Raises:
            ValueError: if ``end`` falls before ``start``. That is a caller
                mistake, not a data condition, and silently swapping the two
                would return a confident wrong answer.
        """
        if end < start:
            raise ValueError(
                f"The end date {end} falls before the start date {start}."
            )
        return cls(start=start, end=end)

    @classmethod
    def last_days(cls, days: int, on: date) -> "DateRange":
        """Return the ``days`` days ending on (and including) ``on``.

        ``on`` is a required argument rather than defaulting to today so that
        every result is reproducible. A report that changes when the test
        suite happens to run is not a testable report.

        Args:
            days: how many days to cover. Must be at least 1.
            on: the last day included.

        Raises:
            ValueError: if ``days`` is less than 1.
        """
        if days < 1:
            raise ValueError("A date range must cover at least one day.")
        return cls(start=on - timedelta(days=days - 1), end=on)

    @property
    def is_unbounded(self) -> bool:
        """True when this range applies no filter."""
        return self.start is None and self.end is None

    def contains(self, day: date) -> bool:
        """True when ``day`` falls inside this range."""
        if self.start is not None and day < self.start:
            return False
        if self.end is not None and day > self.end:
            return False
        return True

    def days(self) -> list[date]:
        """Return every day in the range, in order.

        Returns an empty list for an unbounded range, because an open-ended
        range has no enumerable length. That is the honest answer and it stops
        a caller from accidentally looping over the year 1 to 9999.
        """
        if self.is_unbounded:
            return []
        assert self.start is not None and self.end is not None
        span = (self.end - self.start).days
        return [self.start + timedelta(days=offset) for offset in range(span + 1)]

    def sql_clause(self) -> tuple[str, list[Any]]:
        """Return a ``WHERE`` fragment and its parameters for this range.

        Filtering happens on ``date(occurred_at)`` rather than the raw
        timestamp so that a day is included whole, from 00:00:00 to 23:59:59,
        without the caller having to think about clock times. Both bounds use
        ``>=``/``<=`` because the range is inclusive.

        An unbounded range returns an empty fragment, which composes directly
        into a query with no ``WHERE`` at all.

        The fragment is built from fixed strings only; the bounds are always
        passed as bound parameters, so a date can never become part of the
        SQL.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if self.start is not None:
            clauses.append("date(occurred_at) >= ?")
            params.append(self.start.isoformat())
        if self.end is not None:
            clauses.append("date(occurred_at) <= ?")
            params.append(self.end.isoformat())
        if not clauses:
            return "", []
        return " WHERE " + " AND ".join(clauses), params

    def __str__(self) -> str:
        if self.is_unbounded:
            return "all time"
        if self.start == self.end:
            return str(self.start)
        return f"{self.start} to {self.end}"


@dataclass(frozen=True)
class ToolUsage:
    """One tool's share of the recorded usage.

    Attributes:
        tool_name: the tool.
        count: how many times it handled a command.
        percentage: its share of **all** interactions in the same range, as a
            percentage rounded to one decimal place. Unresolved commands are
            part of that denominator, so the percentages of real tools add up
            to less than 100 when some commands matched nothing -- which is
            the honest reading. See :attr:`is_unmatched`.
    """

    tool_name: str
    count: int
    percentage: float

    @property
    def is_unmatched(self) -> bool:
        """True when this row is an unresolved command, not a real tool."""
        return self.tool_name in NOT_A_TOOL

    def __repr__(self) -> str:
        return (
            f"ToolUsage(tool_name={self.tool_name!r}, count={self.count}, "
            f"percentage={self.percentage})"
        )


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
    #: Interactions that matched a real tool. Added in Sprint 2.
    matched_total: int = 0
    #: Interactions that matched nothing. Added in Sprint 2.
    unmatched_total: int = 0
    #: Share of interactions that matched a tool, 0-100, one decimal. Sprint 2.
    match_rate: float = 0.0

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
            f"active_days={self.active_days}, "
            f"matched_total={self.matched_total}, "
            f"unmatched_total={self.unmatched_total}, "
            f"match_rate={self.match_rate})"
        )


@dataclass(frozen=True)
class NotesSummary:
    """Everything the notes table can honestly say about itself.

    The ``notes`` table has no status column, so there is no notion of an
    active, archived or completed note. :attr:`total` is therefore also the
    number of currently stored notes, and this class does not invent a second
    meaning for it.

    Attributes:
        total: how many notes are stored.
        first_created: earliest ``created_at``, or ``None``.
        last_created: latest ``created_at``, or ``None``.
        average_length: mean character length, one decimal, or ``0.0``.
        longest_length: length of the longest note, or ``0``.
        days_with_notes: how many distinct days have at least one note.
    """

    total: int
    first_created: str | None
    last_created: str | None
    average_length: float
    longest_length: int
    days_with_notes: int

    @property
    def has_notes(self) -> bool:
        """True when at least one note is stored."""
        return self.total > 0

    def __repr__(self) -> str:
        return (
            "NotesSummary("
            f"total={self.total}, "
            f"first_created={self.first_created!r}, "
            f"last_created={self.last_created!r}, "
            f"average_length={self.average_length}, "
            f"longest_length={self.longest_length}, "
            f"days_with_notes={self.days_with_notes})"
        )


@dataclass(frozen=True)
class DashboardReport:
    """One read of everything a dashboard needs, in a single call.

    Grouped the way a dashboard groups things: headline numbers, a ranked
    tool list, an activity trend, and the notes figures. Every field is a
    measured fact from the same committed data.

    Attributes:
        summary: the headline KPIs, as :class:`AnalyticsSummary`.
        top_tools: ranked tool usage, most used first.
        daily: per-day interaction counts, chronological.
        weekly: per-week interaction counts, chronological.
        notes: the notes figures, as :class:`NotesSummary`.
        busiest_day: the single busiest day, or ``None``.
        busiest_week: the single busiest week, or ``None``.
        range_used: the range the report covers, as a readable string.
    """

    summary: AnalyticsSummary
    top_tools: tuple[ToolUsage, ...]
    daily: tuple[TimeBucket, ...]
    weekly: tuple[TimeBucket, ...]
    notes: NotesSummary
    busiest_day: TimeBucket | None
    busiest_week: TimeBucket | None
    range_used: str


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
    def total_interactions(
        self, date_range: "DateRange | None" = None
    ) -> int:
        """Return how many interactions are recorded, matched or not.

        This counts rows in ``interactions``, which includes commands that
        matched no tool. It is the honest "how much have you used me" number;
        use :meth:`command_counts` to see what actually did the work.

        Args:
            date_range: restrict to an inclusive span of days. ``None`` (the
                default) means all time, which is what the tool uses.
        """
        clause, params = self._range_filter(date_range)
        row = self._db.query_one(
            f"SELECT COUNT(*) AS total FROM interactions{clause}", params
        )
        return int(row["total"]) if row else 0

    def first_interaction_time(
        self, date_range: "DateRange | None" = None
    ) -> str | None:
        """Return the earliest recorded timestamp, or ``None`` if empty.

        The stored format is ``"YYYY-MM-DD HH:MM:SS"``, so a plain string
        comparison is also a chronological one and no date parsing is needed
        for the ordering itself.
        """
        return self._time_extreme("MIN", date_range)

    def last_interaction_time(
        self, date_range: "DateRange | None" = None
    ) -> str | None:
        """Return the latest recorded timestamp, or ``None`` if empty."""
        return self._time_extreme("MAX", date_range)

    def _time_extreme(self, function: str, date_range: "DateRange | None" = None) -> str | None:
        """Run ``MIN`` or ``MAX`` over ``occurred_at`` and normalise the result.

        ``function`` is never taken from a caller: it is only ever the two
        literals passed above, so the string cannot become an injection
        point. SQLite returns one row with ``NULL`` for an empty table, which
        is why the row itself exists but the value can still be ``None``.
        """
        clause, params = self._range_filter(date_range)
        row = self._db.query_one(
            f"SELECT {function}(occurred_at) AS value FROM interactions{clause}",
            params,
        )
        if row is None:
            return None
        value = row["value"]
        return str(value) if value else None

    # ------------------------------------------------------------------
    # Date-range plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _range_filter(
        date_range: "DateRange | None",
    ) -> tuple[str, list[Any]]:
        """Return a ``WHERE`` fragment and params for an optional range.

        ``None`` means all time and produces an empty fragment, so every
        ranged method has the same shape: filter, then query. Keeping this in
        one place is what stops each method from reinventing, and eventually
        disagreeing about, what a date range means.
        """
        if date_range is None:
            return "", []
        return date_range.sql_clause()

    def _total_in(self, date_range: "DateRange | None") -> int:
        """Count interactions inside a range, for percentage denominators."""
        clause, params = self._range_filter(date_range)
        row = self._db.query_one(
            f"SELECT COUNT(*) AS total FROM interactions{clause}", params
        )
        return int(row["total"]) if row else 0

    # ------------------------------------------------------------------
    # B. Tool usage
    # ------------------------------------------------------------------
    def command_counts(
        self, date_range: "DateRange | None" = None
    ) -> list[ToolCount]:
        """Return per-tool counts, most used first, ties broken by name.

        Unresolved commands are grouped under one of the
        :data:`NOT_A_TOOL` sentinels and are **kept** here rather than
        dropped, because they are real rows and hiding them would make the
        counts fail to sum to :meth:`total_interactions`. Use
        :meth:`matched_command_counts` for the tool-only view.

        The query is written here rather than delegated to
        :func:`~assistant.storage.repositories.count_by_tool`, because that
        helper takes no date range. The ordering is identical, so the two
        agree on all-time data.
        """
        clause, params = self._range_filter(date_range)
        rows = self._db.query_all(
            "SELECT COALESCE(NULLIF(tool_name, ''), ?) AS name, "
            f"COUNT(*) AS total FROM interactions{clause} "
            "GROUP BY name ORDER BY total DESC, name ASC",
            [UNKNOWN_TOOL] + params,
        )
        return [
            ToolCount(tool_name=str(row["name"]), count=int(row["total"]))
            for row in rows
        ]

    def matched_command_counts(
        self, date_range: "DateRange | None" = None
    ) -> list[ToolCount]:
        """Return per-tool counts with unresolved commands removed.

        This is the list to show a user who asked what they use, because
        ``none`` is not a feature. Ordering is inherited from
        :meth:`command_counts`, so it stays most-used-first then alphabetical.
        """
        return [
            count
            for count in self.command_counts(date_range)
            if count.tool_name not in NOT_A_TOOL
        ]

    def unique_tools_used(
        self, date_range: "DateRange | None" = None
    ) -> int:
        """Return how many distinct real tools have handled a command.

        Unresolved commands do not count: they are not tools. So a database
        holding only unmatched commands reports ``0`` unique tools, which is
        the truthful answer, not a bug.
        """
        return len(self.matched_command_counts(date_range))

    def most_used_command(
        self, date_range: "DateRange | None" = None
    ) -> ToolCount | None:
        """Return the most-used tool, or ``None`` when nothing is recorded.

        Ties are resolved by :meth:`command_counts`, which already orders by
        descending count then ascending name, so a tie always yields the same
        answer for the same data rather than whichever row SQLite happens to
        emit first. Unresolved commands are excluded: reporting ``none`` as
        the most-used feature would be nonsense.
        """
        matched = self.matched_command_counts(date_range)
        return matched[0] if matched else None

    def top_tools(
        self, limit: int = 5, date_range: "DateRange | None" = None
    ) -> list[ToolCount]:
        """Return the ``limit`` most-used real tools, most used first.

        Unresolved commands are excluded, since a dashboard ranking "none"
        as the top tool would be misleading. Ties keep the alphabetical order
        from :meth:`command_counts`, so the cut is deterministic: two tools
        tied at the boundary always yield the same one.
        """
        if limit < 1:
            raise ValueError("top_tools needs a limit of at least 1.")
        return self.matched_command_counts(date_range)[:limit]

    def tool_usage(
        self, date_range: "DateRange | None" = None
    ) -> list[ToolUsage]:
        """Return per-tool counts with their share of all interactions.

        The percentage denominator is :meth:`total_interactions` for the same
        range, so unresolved commands count in the denominator but appear as
        their own :class:`ToolUsage` row flagged :attr:`ToolUsage.is_unmatched`
        rather than being hidden. Real tools therefore add up to less than
        100 when some commands matched nothing, which is the honest reading.

        Returns an empty list when there is no activity at all, because a
        percentage of nothing is not zero, it is undefined.
        """
        total = self.total_interactions(date_range)
        if total == 0:
            return []
        return [
            ToolUsage(
                tool_name=count.tool_name,
                count=count.count,
                percentage=round(count.count * 100.0 / total, 1),
            )
            for count in self.command_counts(date_range)
        ]

    def matched_total(
        self, date_range: "DateRange | None" = None
    ) -> int:
        """Return how many interactions matched a real tool."""
        return sum(count.count for count in self.matched_command_counts(date_range))

    def unmatched_count(
        self, date_range: "DateRange | None" = None
    ) -> int:
        """Return how many recorded interactions matched no tool."""
        return sum(
            count.count
            for count in self.command_counts(date_range)
            if count.tool_name in NOT_A_TOOL
        )

    def match_rate(self, date_range: "DateRange | None" = None) -> float:
        """Return the percentage of interactions that matched a tool.

        ``0.0`` when nothing has been recorded: no activity is not a 0% match
        rate, but reporting 100% for an empty database would be worse, and
        there is nothing to measure either way.
        """
        total = self.total_interactions(date_range)
        if total == 0:
            return 0.0
        return round(self.matched_total(date_range) * 100.0 / total, 1)

    # ------------------------------------------------------------------
    # C. Time-based activity
    # ------------------------------------------------------------------
    def daily_interaction_counts(
        self,
        limit: int | None = None,
        date_range: "DateRange | None" = None,
        fill: bool = False,
    ) -> list[TimeBucket]:
        """Return interactions per calendar day, oldest day first.

        Args:
            limit: when given, only the most recent ``limit`` days are
                returned, still in chronological order.
            date_range: restrict to an inclusive span of days.
            fill: when ``True``, days inside a bounded ``date_range`` that saw
                no activity are included with a count of ``0``.

        Days with no activity are **not** included by default. They are not
        missing data, and inserting explicit zeros would imply the assistant
        knows about days it was never run, which it does not. The exception is
        an explicitly requested range: there the caller has already declared
        the window, so zero-filling is the useful reading and a chart needs
        it. Filling is only possible for a bounded range, because an unbounded
        one has no days to enumerate.
        """
        rows = self._grouped_counts("date(occurred_at)", limit, date_range)
        buckets = [
            TimeBucket(period=str(row["period"]), count=int(row["total"]))
            for row in rows
        ]
        if not fill or date_range is None or date_range.is_unbounded:
            return buckets
        return _zero_fill_days(buckets, date_range.days())

    def weekly_interaction_counts(
        self,
        limit: int | None = None,
        date_range: "DateRange | None" = None,
    ) -> list[TimeBucket]:
        """Return interactions per ISO week, oldest week first.

        Weeks are ISO-8601 (``2026-W38``) and are computed in Python rather
        than with ``strftime('%V')``, which returns NULL on the SQLite
        3.45.3 this project runs against. Doing it in Python also makes the
        week boundary testable without depending on the local clock.

        Empty weeks are omitted, for the same reason as in
        :meth:`daily_interaction_counts`. Weeks cannot be zero-filled: a range
        of days can be enumerated, but a set of ISO weeks would need its
        boundaries worked out, and a caller that needs a chart window can use
        the daily series instead.
        """
        # Conditions are collected into one list, and the range contributes
        # conditions rather than a finished WHERE clause, so clause order and
        # parameter order are built together and cannot drift apart.
        conditions = ["occurred_at IS NOT NULL", "occurred_at != ''"]
        params: list[Any] = []
        if date_range is not None:
            if date_range.start is not None:
                conditions.append("date(occurred_at) >= ?")
                params.append(date_range.start.isoformat())
            if date_range.end is not None:
                conditions.append("date(occurred_at) <= ?")
                params.append(date_range.end.isoformat())

        rows = self._db.query_all(
            "SELECT occurred_at FROM interactions WHERE "
            + " AND ".join(conditions),
            params,
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
        self,
        expression: str,
        limit: int | None,
        date_range: "DateRange | None" = None,
    ) -> list[dict[str, Any]]:
        """Group interactions by a date expression, chronologically.

        ``expression`` is always a literal defined in this module, never
        caller input, so the interpolation is safe. Ordering is ascending on
        the expression, which is chronological for the stored date format.
        """
        # Conditions are collected into one list so the clause order and the
        # parameter order are built together and cannot drift apart. The
        # range contributes conditions, not a finished WHERE clause, which
        # is why they can be combined here at all.
        conditions = [f"{expression} IS NOT NULL"]
        params: list[Any] = []
        if date_range is not None:
            if date_range.start is not None:
                conditions.append(f"date(occurred_at) >= ?")
                params.append(date_range.start.isoformat())
            if date_range.end is not None:
                conditions.append("date(occurred_at) <= ?")
                params.append(date_range.end.isoformat())

        sql = (
            f"SELECT {expression} AS period, COUNT(*) AS total "
            "FROM interactions WHERE "
            + " AND ".join(conditions)
            + " GROUP BY period ORDER BY period ASC"
        )
        rows = [dict(row) for row in self._db.query_all(sql, params)]
        if limit is not None:
            rows = rows[-max(1, int(limit)) :]
        return rows

    def busiest_day(
        self, date_range: "DateRange | None" = None
    ) -> TimeBucket | None:
        """Return the single busiest day, or ``None`` when empty.

        Ties are broken by the later date, so repeated calls on unchanged
        data always agree.
        """
        days = self.daily_interaction_counts(date_range=date_range)
        if not days:
            return None
        return max(days, key=lambda bucket: (bucket.count, bucket.period))

    def busiest_week(
        self, date_range: "DateRange | None" = None
    ) -> TimeBucket | None:
        """Return the single busiest ISO week, or ``None`` when empty.

        Ties are broken by the later week, matching :meth:`busiest_day`.
        """
        weeks = self.weekly_interaction_counts(date_range=date_range)
        if not weeks:
            return None
        return max(weeks, key=lambda bucket: (bucket.count, bucket.period))

    def active_days(
        self, date_range: "DateRange | None" = None
    ) -> int:
        """Return how many distinct days have at least one interaction."""
        clause, params = self._range_filter(date_range)
        row = self._db.query_one(
            "SELECT COUNT(DISTINCT date(occurred_at)) AS total "
            f"FROM interactions{clause}",
            params,
        )
        return int(row["total"]) if row else 0

    # ------------------------------------------------------------------
    # D. Summary
    # ------------------------------------------------------------------
    def summary(
        self, date_range: "DateRange | None" = None
    ) -> AnalyticsSummary:
        """Return the headline figures in one object.

        Fields are read from the same committed data, so the numbers within
        one summary are mutually consistent. The notes figures are **not**
        filtered by ``date_range``: notes are a separate table with their own
        timestamps and no meaningful relationship to an interaction window, so
        filtering them by the same range would imply a link that does not
        exist. Use :meth:`notes_summary` when notes are the subject.
        """
        top = self.most_used_command(date_range)
        total = self.total_interactions(date_range)
        return AnalyticsSummary(
            total_interactions=total,
            unique_tools=self.unique_tools_used(date_range),
            most_used_tool=top.tool_name if top else None,
            most_used_count=top.count if top else 0,
            first_interaction=self.first_interaction_time(date_range),
            last_interaction=self.last_interaction_time(date_range),
            total_notes=self.total_notes(),
            active_days=self.active_days(date_range),
            matched_total=self.matched_total(date_range),
            unmatched_total=self.unmatched_count(date_range),
            match_rate=self.match_rate(date_range),
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

    def average_note_length(self) -> float:
        """Return the mean character length of stored notes, to one decimal.

        ``0.0`` when there are no notes: an average over nothing is undefined,
        and reporting ``0.0`` matches "there are no notes to average".
        """
        row = self._db.query_one("SELECT AVG(LENGTH(body)) AS mean FROM notes")
        if row is None or row["mean"] is None:
            return 0.0
        return round(float(row["mean"]), 1)

    def notes_per_day(self) -> float:
        """Return the average number of notes added per active day.

        The denominator is the number of days that actually received a note,
        not the span between the first and last note. Averaging over days the
        assistant never ran would understate the real habit, which is what a
        person means by "how many notes do I write a day".

        ``0.0`` when there are no notes.
        """
        total = self.total_notes()
        days = self.note_days()
        if total == 0 or days == 0:
            return 0.0
        return round(total / days, 2)

    def note_days(self) -> int:
        """Return how many distinct days have at least one note."""
        row = self._db.query_one(
            "SELECT COUNT(DISTINCT date(created_at)) AS total FROM notes"
        )
        return int(row["total"]) if row else 0

    def longest_note(self) -> str | None:
        """Return the body of the longest note, or ``None`` when there are none.

        Returns the text rather than the length, because a caller showing a
        dashboard usually wants to display the note itself. Ties are broken by
        the lowest id, so the same note comes back every time.
        """
        row = self._db.query_one(
            "SELECT body FROM notes ORDER BY LENGTH(body) DESC, id ASC LIMIT 1"
        )
        if row is None:
            return None
        body = row["body"]
        return str(body) if body else None

    def notes_summary(self) -> NotesSummary:
        """Return every honest note figure in one object.

        Note the absence of an "active" field: the schema has no status
        column, so :attr:`NotesSummary.total` is the count of currently
        stored notes and nothing more is claimed about them.
        """
        return NotesSummary(
            total=self.total_notes(),
            first_created=self.first_note_time(),
            last_created=self.last_note_time(),
            average_length=self.average_note_length(),
            longest_length=self.longest_note_length(),
            days_with_notes=self.note_days(),
        )

    def dashboard(
        self,
        date_range: "DateRange | None" = None,
        top_limit: int = 5,
    ) -> DashboardReport:
        """Return everything a dashboard needs, in one read.

        Args:
            date_range: restrict the interaction figures to an inclusive span
                of days. ``None`` means all time.
            top_limit: how many tools to rank in ``top_tools``.

        The interaction figures honour ``date_range``; the notes figures do
        not, for the reason given on :meth:`summary`. That asymmetry is
        deliberate and is recorded in :attr:`DashboardReport.range_used` so a
        future reader is not misled into thinking the notes figure is
        windowed too.
        """
        return DashboardReport(
            summary=self.summary(date_range),
            top_tools=tuple(self.top_tools(top_limit, date_range)),
            daily=tuple(self.daily_interaction_counts(date_range=date_range)),
            weekly=tuple(self.weekly_interaction_counts(date_range=date_range)),
            notes=self.notes_summary(),
            busiest_day=self.busiest_day(date_range),
            busiest_week=self.busiest_week(date_range),
            range_used=str(date_range) if date_range else "all time",
        )

    def __repr__(self) -> str:
        # The Database repr carries only a path and an open flag, so the
        # service can be logged without leaking any row content.
        return f"AnalyticsService(db={self._db!r})"


def _zero_fill_days(
    buckets: list[TimeBucket], days: list[date]
) -> list[TimeBucket]:
    """Return one bucket per day in ``days``, zero-filling the gaps.

    Only meaningful for an explicitly requested range, where the caller has
    already declared the window and a chart needs every point in it. Days
    outside the observed data are included with ``0``; days inside the data
    keep their measured count. The result is chronological, and a day that is
    not in ``days`` at all (which should be impossible when the range and the
    query agree) is dropped rather than silently appended.
    """
    measured = {bucket.period: bucket.count for bucket in buckets}
    wanted = [day.isoformat() for day in days]
    return [
        TimeBucket(period=period, count=measured.get(period, 0))
        for period in wanted
    ]


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




