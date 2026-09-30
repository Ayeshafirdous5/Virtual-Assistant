"""Tests for the read-only analytics service.

Every test runs against a temporary in-memory or file-backed database built
by the shared ``conftest.py`` fixtures, so the real ``data/assistant.db`` is
never opened, let alone written. The service performs no writes at all, which
:func:`TestTheServiceNeverWrites` proves rather than assumes.

Timestamps are inserted explicitly rather than relying on ``datetime('now')``,
so day and week bucketing is deterministic and does not depend on when the
suite runs.
"""

from __future__ import annotations

import sqlite3

import pytest

from assistant.analytics import (
    NOT_A_TOOL,
    AnalyticsService,
    AnalyticsSummary,
    TimeBucket,
)
from assistant.analytics.service import _iso_week_label


@pytest.fixture
def service(memory_db):
    """An analytics service over an empty migrated database."""
    return AnalyticsService(memory_db)


def record(db, tool="jokes", when="2026-09-26 10:00:00", utterance="a joke"):
    """Insert one interaction with an explicit timestamp."""
    db.execute(
        "INSERT INTO interactions (occurred_at, tool_name, utterance, response) "
        "VALUES (?,?,?,?)",
        (when, tool, utterance, "an answer"),
    )


# ----------------------------------------------------------------------
# Empty database
# ----------------------------------------------------------------------
class TestEmptyDatabase:
    """No data is not an error, and never a crash."""

    def test_totals_are_zero(self, service):
        assert service.total_interactions() == 0
        assert service.unique_tools_used() == 0
        assert service.active_days() == 0
        assert service.total_notes() == 0
        assert service.active_note_count() == 0
        assert service.unmatched_count() == 0

    def test_timestamps_are_none(self, service):
        assert service.first_interaction_time() is None
        assert service.last_interaction_time() is None
        assert service.first_note_time() is None
        assert service.last_note_time() is None

    def test_collections_are_empty(self, service):
        assert service.command_counts() == []
        assert service.matched_command_counts() == []
        assert service.daily_interaction_counts() == []
        assert service.weekly_interaction_counts() == []
        assert service.daily_note_counts() == []
        assert service.longest_note_length() == 0

    def test_no_most_used_and_no_busiest_day(self, service):
        assert service.most_used_command() is None
        assert service.busiest_day() is None

    def test_summary_is_empty_but_consistent(self, service):
        summary = service.summary()
        assert summary.total_interactions == 0
        assert summary.unique_tools == 0
        assert summary.most_used_tool is None
        assert summary.most_used_count == 0
        assert summary.first_interaction is None
        assert summary.last_interaction is None
        assert summary.total_notes == 0


# ----------------------------------------------------------------------
# One interaction
# ----------------------------------------------------------------------
class TestOneInteraction:
    def test_totals(self, service, memory_db):
        record(memory_db)
        assert service.total_interactions() == 1
        assert service.unique_tools_used() == 1
        assert service.active_days() == 1

    def test_first_and_last_are_the_same_row(self, service, memory_db):
        record(memory_db, when="2026-09-26 10:30:00")
        assert service.first_interaction_time() == "2026-09-26 10:30:00"
        assert service.last_interaction_time() == "2026-09-26 10:30:00"

    def test_most_used(self, service, memory_db):
        record(memory_db, tool="weather")
        top = service.most_used_command()
        assert top is not None
        assert top.tool_name == "weather"
        assert top.count == 1

    def test_single_day_bucket(self, service, memory_db):
        record(memory_db, when="2026-09-26 10:00:00")
        assert service.daily_interaction_counts() == [TimeBucket("2026-09-26", 1)]

    def test_single_week_bucket(self, service, memory_db):
        record(memory_db, when="2026-09-26 10:00:00")
        assert service.weekly_interaction_counts() == [TimeBucket("2026-W39", 1)]


# ----------------------------------------------------------------------
# Multiple interactions and repeated tools
# ----------------------------------------------------------------------
class TestMultipleInteractions:
    def test_counts_sum_to_the_total(self, service, memory_db):
        for _ in range(3):
            record(memory_db, tool="jokes")
        record(memory_db, tool="weather")
        record(memory_db, tool="notes")
        assert sum(c.count for c in service.command_counts()) == 5
        assert service.total_interactions() == 5

    def test_repeated_tool_is_aggregated(self, service, memory_db):
        for _ in range(4):
            record(memory_db, tool="notes")
        counts = service.command_counts()
        assert len(counts) == 1
        assert counts[0].tool_name == "notes"
        assert counts[0].count == 4

    def test_most_used_is_the_highest_count(self, service, memory_db):
        record(memory_db, tool="weather")
        for _ in range(3):
            record(memory_db, tool="notes")
        top = service.most_used_command()
        assert top.tool_name == "notes"
        assert top.count == 3

    def test_first_and_last_span_the_whole_range(self, service, memory_db):
        record(memory_db, when="2026-09-26 08:00:00")
        record(memory_db, when="2026-09-27 19:45:00")
        record(memory_db, when="2026-09-26 12:00:00")
        assert service.first_interaction_time() == "2026-09-26 08:00:00"
        assert service.last_interaction_time() == "2026-09-27 19:45:00"

    def test_insertion_order_does_not_affect_first_or_last(
        self, service, memory_db
    ):
        """Out-of-order inserts must not confuse MIN/MAX."""
        record(memory_db, when="2026-09-27 09:00:00")
        record(memory_db, when="2026-09-25 09:00:00")
        record(memory_db, when="2026-09-26 09:00:00")
        assert service.first_interaction_time() == "2026-09-25 09:00:00"
        assert service.last_interaction_time() == "2026-09-27 09:00:00"


# ----------------------------------------------------------------------
# Unique tool counting and the unmatched sentinels
# ----------------------------------------------------------------------
class TestUniqueTools:
    def test_distinct_tools_are_counted_once_each(self, service, memory_db):
        for tool in ("weather", "jokes", "notes"):
            record(memory_db, tool=tool)
        assert service.unique_tools_used() == 3

    def test_repeats_do_not_inflate_the_unique_count(self, service, memory_db):
        for _ in range(5):
            record(memory_db, tool="weather")
        for _ in range(3):
            record(memory_db, tool="jokes")
        assert service.unique_tools_used() == 2

    def test_the_apps_none_sentinel_is_not_a_tool(self, service, memory_db):
        """The app records unresolved commands as the literal "none"."""
        for _ in range(3):
            record(memory_db, tool="none")
        record(memory_db, tool="weather")
        assert service.unique_tools_used() == 1
        assert service.unmatched_count() == 3
        assert service.most_used_command().tool_name == "weather"

    def test_the_storages_unmatched_sentinel_is_not_a_tool(
        self, service, memory_db
    ):
        record(memory_db, tool=None)
        record(memory_db, tool="")
        assert service.unique_tools_used() == 0
        assert service.unmatched_count() == 2
        assert service.most_used_command() is None

    def test_unmatched_is_still_counted_in_the_total(self, service, memory_db):
        """It is a real row, so hiding it would understate usage."""
        record(memory_db, tool="none")
        record(memory_db, tool="weather")
        assert service.total_interactions() == 2
        assert service.unique_tools_used() == 1

    def test_command_counts_still_include_unmatched(self, service, memory_db):
        record(memory_db, tool="none")
        record(memory_db, tool="weather")
        names = {c.tool_name for c in service.command_counts()}
        assert "none" in names
        assert "none" not in {c.tool_name for c in service.matched_command_counts()}

    def test_the_sentinel_set_is_exactly_the_three_real_values(self):
        assert NOT_A_TOOL == {"", "none", "unmatched"}


# ----------------------------------------------------------------------
# Deterministic ordering and tie behaviour
# ----------------------------------------------------------------------
class TestOrderingAndTies:
    def test_counts_are_ordered_by_count_then_name(self, service, memory_db):
        for _ in range(2):
            record(memory_db, tool="zebra")
        for _ in range(2):
            record(memory_db, tool="alpha")
        record(memory_db, tool="beta")
        pairs = [(c.tool_name, c.count) for c in service.command_counts()]
        assert pairs == [("alpha", 2), ("zebra", 2), ("beta", 1)]

    def test_a_tie_resolves_alphabetically_every_time(self, service, memory_db):
        """The same tie must always give the same winner."""
        for _ in range(2):
            record(memory_db, tool="weather")
        for _ in range(2):
            record(memory_db, tool="jokes")
        winners = {service.most_used_command().tool_name for _ in range(5)}
        assert winners == {"jokes"}

    def test_repeated_calls_return_identical_results(self, service, memory_db):
        for tool in ("weather", "notes", "jokes", "filler"):
            record(memory_db, tool=tool)
        first = service.command_counts()
        second = service.command_counts()
        assert first == second
        assert service.summary() == service.summary()

    def test_daily_counts_are_chronological(self, service, memory_db):
        record(memory_db, when="2026-09-27 09:00:00")
        record(memory_db, when="2026-09-25 09:00:00")
        record(memory_db, when="2026-09-26 09:00:00")
        assert [b.period for b in service.daily_interaction_counts()] == [
            "2026-09-25",
            "2026-09-26",
            "2026-09-27",
        ]

    def test_busiest_day_picks_the_highest_count(self, service, memory_db):
        record(memory_db, when="2026-09-25 09:00:00")
        record(memory_db, when="2026-09-26 09:00:00")
        record(memory_db, when="2026-09-26 10:00:00")
        assert service.busiest_day() == TimeBucket("2026-09-26", 2)

    def test_busiest_day_breaks_a_tie_towards_the_later_date(
        self, service, memory_db
    ):
        record(memory_db, when="2026-09-25 09:00:00")
        record(memory_db, when="2026-09-26 09:00:00")
        assert service.busiest_day().period == "2026-09-26"

    def test_empty_days_are_not_invented(self, service, memory_db):
        """A gap is not a zero. The assistant was not running."""
        record(memory_db, when="2026-09-25 09:00:00")
        record(memory_db, when="2026-09-27 09:00:00")
        assert [b.period for b in service.daily_interaction_counts()] == [
            "2026-09-25",
            "2026-09-27",
        ]


# ----------------------------------------------------------------------
# Daily and weekly bucketing
# ----------------------------------------------------------------------
class TestDailyCounts:
    def test_interactions_group_by_day(self, service, memory_db):
        for hour in ("08", "09", "10"):
            record(memory_db, when=f"2026-09-26 {hour}:00:00")
        record(memory_db, when="2026-09-27 08:00:00")
        assert service.daily_interaction_counts() == [
            TimeBucket("2026-09-26", 3),
            TimeBucket("2026-09-27", 1),
        ]

    def test_limit_keeps_the_most_recent_days_in_order(
        self, service, memory_db
    ):
        for day in ("25", "26", "27", "28"):
            record(memory_db, when=f"2026-09-{day} 09:00:00")
        assert service.daily_interaction_counts(limit=2) == [
            TimeBucket("2026-09-27", 1),
            TimeBucket("2026-09-28", 1),
        ]

    def test_active_days_counts_distinct_days(self, service, memory_db):
        for _ in range(4):
            record(memory_db, when="2026-09-26 09:00:00")
        record(memory_db, when="2026-09-27 09:00:00")
        assert service.active_days() == 2


class TestWeeklyCounts:
    def test_a_week_is_grouped(self, service, memory_db):
        # 2026-09-21 is a Monday; 2026-09-27 the Sunday of the same week.
        for day in ("21", "23", "27"):
            record(memory_db, when=f"2026-09-{day} 09:00:00")
        record(memory_db, when="2026-09-28 09:00:00")
        assert service.weekly_interaction_counts() == [
            TimeBucket("2026-W39", 3),
            TimeBucket("2026-W40", 1),
        ]

    def test_limit_keeps_the_most_recent_weeks(self, service, memory_db):
        for day in ("21", "28", "29"):
            record(memory_db, when=f"2026-09-{day} 09:00:00")
        assert service.weekly_interaction_counts(limit=1) == [
            TimeBucket("2026-W40", 2)
        ]

    def test_week_totals_match_the_daily_totals(self, service, memory_db):
        """The two views must not disagree about how much happened."""
        for day in ("21", "22", "30"):
            record(memory_db, when=f"2026-09-{day} 09:00:00")
        assert sum(b.count for b in service.weekly_interaction_counts()) == 3
        assert sum(b.count for b in service.daily_interaction_counts()) == 3


class TestIsoWeekLabel:
    """The week label is computed in Python, so it is pinned directly."""

    @pytest.mark.parametrize(
        "timestamp,expected",
        [
            ("2026-09-26 10:00:00", "2026-W39"),   # Saturday
            ("2026-09-21 00:00:00", "2026-W39"),   # Monday of the same week
            ("2026-09-27 23:59:59", "2026-W39"),   # Sunday of the same week
            ("2026-09-28 00:00:00", "2026-W40"),   # next Monday
            ("2026-01-01 00:00:00", "2026-W01"),   # belongs to last year
            ("2027-01-04 00:00:00", "2027-W01"),
        ],
    )
    def test_labels(self, timestamp, expected):
        assert _iso_week_label(timestamp) == expected

    def test_year_boundary_uses_the_iso_year(self):
        """2027-01-01 is a Friday, so ISO calls it week 53 of 2026."""
        assert _iso_week_label("2027-01-01 12:00:00") == "2026-W53"

    @pytest.mark.parametrize(
        "bad", ["", "not-a-date", "2026-13-45 00:00:00", "26-09-26", "   "]
    )
    def test_unparseable_values_return_none(self, bad):
        assert _iso_week_label(bad) is None

    def test_one_bad_row_does_not_break_the_whole_report(
        self, service, memory_db
    ):
        record(memory_db, when="2026-09-26 09:00:00")
        memory_db.execute(
            "INSERT INTO interactions (occurred_at, tool_name) VALUES (?,?)",
            ("rubbish", "jokes"),
        )
        assert service.weekly_interaction_counts() == [
            TimeBucket("2026-W39", 1)
        ]


# ----------------------------------------------------------------------
# Summary correctness
# ----------------------------------------------------------------------
class TestSummary:
    def test_summary_matches_the_individual_methods(self, service, memory_db):
        for tool in ("weather", "notes", "notes"):
            record(memory_db, tool=tool)
        record(memory_db, tool="none", when="2026-09-27 09:00:00")

        summary = service.summary()
        assert summary.total_interactions == service.total_interactions() == 4
        assert summary.unique_tools == service.unique_tools_used() == 2
        assert summary.most_used_tool == "notes"
        assert summary.most_used_count == 2
        assert summary.first_interaction == "2026-09-26 10:00:00"
        assert summary.last_interaction == "2026-09-27 09:00:00"
        assert summary.total_notes == 0
        assert summary.active_days == 2
        assert summary.has_activity is True

    def test_summary_counts_unmatched_in_the_total_only(
        self, service, memory_db
    ):
        """Three unmatched rows are usage, but not a tool."""
        for _ in range(3):
            record(memory_db, tool="none")
        summary = service.summary()
        assert summary.total_interactions == 3
        assert summary.unique_tools == 0
        assert summary.most_used_tool is None
        assert summary.most_used_count == 0
        assert summary.has_activity is True

    def test_summary_is_immutable(self, service):
        with pytest.raises(Exception):
            service.summary().total_interactions = 99

    def test_time_bucket_is_immutable_and_compares_by_value(self):
        assert TimeBucket("2026-09-26", 3) == TimeBucket("2026-09-26", 3)
        assert TimeBucket("2026-09-26", 3) != TimeBucket("2026-09-26", 4)
        with pytest.raises(Exception):
            TimeBucket("2026-09-26", 3).count = 4

    def test_summary_repr_is_informative(self, service, memory_db):
        record(memory_db, tool="weather")
        text = repr(service.summary())
        assert "total_interactions=1" in text
        assert "most_used_tool='weather'" in text


# ----------------------------------------------------------------------
# Notes analytics
# ----------------------------------------------------------------------
class TestNotesAnalytics:
    def add_note(self, db, body="buy milk", when="2026-09-26 10:00:00"):
        db.execute(
            "INSERT INTO notes (created_at, body) VALUES (?,?)", (when, body)
        )

    def test_total_notes(self, service, memory_db):
        for index in range(3):
            self.add_note(memory_db, body=f"note {index}")
        assert service.total_notes() == 3

    def test_active_equals_total_because_there_is_no_status(
        self, service, memory_db
    ):
        """Documented schema limitation, pinned so it cannot be assumed away."""
        for index in range(2):
            self.add_note(memory_db, body=f"note {index}")
        assert service.active_note_count() == service.total_notes() == 2

    def test_note_timestamps(self, service, memory_db):
        self.add_note(memory_db, when="2026-09-25 08:00:00")
        self.add_note(memory_db, when="2026-09-27 08:00:00")
        assert service.first_note_time() == "2026-09-25 08:00:00"
        assert service.last_note_time() == "2026-09-27 08:00:00"

    def test_notes_created_over_time(self, service, memory_db):
        self.add_note(memory_db, when="2026-09-26 08:00:00")
        self.add_note(memory_db, when="2026-09-26 09:00:00")
        self.add_note(memory_db, when="2026-09-27 08:00:00")
        assert service.daily_note_counts() == [
            TimeBucket("2026-09-26", 2),
            TimeBucket("2026-09-27", 1),
        ]

    def test_daily_note_counts_respects_limit(self, service, memory_db):
        for day in ("25", "26", "27"):
            self.add_note(memory_db, when=f"2026-09-{day} 08:00:00")
        assert service.daily_note_counts(limit=1) == [
            TimeBucket("2026-09-27", 1)
        ]

    def test_longest_note_length(self, service, memory_db):
        self.add_note(memory_db, body="short")
        longer = "a much longer note body"
        self.add_note(memory_db, body=longer)
        assert service.longest_note_length() == len(longer)

    def test_notes_and_interactions_are_independent(self, service, memory_db):
        """A note is not an interaction, and vice versa."""
        self.add_note(memory_db)
        record(memory_db)
        assert service.total_notes() == 1
        assert service.total_interactions() == 1

    def test_notes_work_with_the_real_repository(self, service, memory_db):
        from assistant.storage.notes import NoteRepository

        NoteRepository(memory_db).add("via the repository")
        assert service.total_notes() == 1
        assert service.active_note_count() == 1


# ----------------------------------------------------------------------
# The read-only guarantee
# ----------------------------------------------------------------------
class TestTheServiceNeverWrites:
    """Proved, not assumed: the service holds no write path at all."""

    def test_the_module_issues_no_write_statements(self):
        import inspect

        import assistant.analytics.service as module

        source = inspect.getsource(module).upper()
        for statement in (
            "INSERT INTO",
            "UPDATE ",
            "DELETE FROM",
            "CREATE TABLE",
            "DROP TABLE",
        ):
            assert statement not in source, statement

    def test_reading_does_not_change_any_row(self, service, memory_db):
        for _ in range(3):
            record(memory_db, tool="notes")
        memory_db.execute(
            "INSERT INTO notes (body) VALUES ('a note')"
        )
        before = (
            memory_db.query_one("SELECT COUNT(*) AS c FROM interactions")["c"],
            memory_db.query_one("SELECT COUNT(*) AS c FROM notes")["c"],
        )
        service.summary()
        service.command_counts()
        service.daily_interaction_counts()
        service.weekly_interaction_counts()
        service.daily_note_counts()
        service.unmatched_count()
        service.busiest_day()
        service.longest_note_length()
        after = (
            memory_db.query_one("SELECT COUNT(*) AS c FROM interactions")["c"],
            memory_db.query_one("SELECT COUNT(*) AS c FROM notes")["c"],
        )
        assert before == after == (3, 1)

    def test_no_schema_change_was_required(self, memory_db):
        """The sprint reused the existing tables at their existing version."""
        from assistant.core.database import SCHEMA_VERSION

        assert memory_db.schema_version() == SCHEMA_VERSION == 2
        assert "interactions" in memory_db.table_names()
        assert "notes" in memory_db.table_names()


# ----------------------------------------------------------------------
# Degraded behaviour
# ----------------------------------------------------------------------
class TestDegradedBehaviour:
    """A missing schema raises, and the tool layer turns that into a sentence."""

    def test_an_uninitialised_database_raises(self):
        from assistant.core.database import Database

        database = Database(":memory:")  # deliberately never initialised
        try:
            service = AnalyticsService(database)
            with pytest.raises(sqlite3.Error):
                service.summary()
        finally:
            database.close()

    def test_a_closed_database_raises_rather_than_returning_zeros(
        self, memory_db
    ):
        """Silently returning 0 for a broken database would be a lie."""
        service = AnalyticsService(memory_db)
        memory_db.close()
        with pytest.raises(sqlite3.Error):
            service.total_interactions()

    def test_a_missing_table_does_not_crash_the_service_object(
        self, memory_db
    ):
        from assistant.core.database import Database

        database = Database(":memory:")
        try:
            service = AnalyticsService(database)
            # The object is still usable; only the queries fail.
            assert "AnalyticsService(" in repr(service)
            with pytest.raises(sqlite3.Error):
                service.daily_note_counts()
        finally:
            database.close()


