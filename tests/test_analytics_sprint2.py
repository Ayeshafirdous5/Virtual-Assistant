"""Sprint 2 tests: date ranges, rankings, time series and notes analytics.

Everything here is built from :mod:`tests.analytics_fixtures`, which writes
explicit timestamps, so no assertion in this file depends on the current date.
The two date-dependent paths in the tool (``today`` and ``recent``) are the
only places the clock is read, and both are covered by pinning the rows to
``date.today()`` at the moment the test runs rather than to a hard-coded day.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from assistant.analytics import (
    AnalyticsService,
    DateRange,
    DashboardReport,
    NotesSummary,
    ToolUsage,
)
from tests.analytics_fixtures import (
    ANCHOR,
    Dataset,
    analytics_fixture,
    empty,
    notes_over_three_days,
    only_unmatched,
    tie_between_two_tools,
    two_tools_over_three_days,
)


@pytest.fixture
def service(memory_db):
    """An analytics service over an empty migrated database."""
    return AnalyticsService(memory_db)


# ----------------------------------------------------------------------
# The DateRange value object
# ----------------------------------------------------------------------
class TestDateRange:
    def test_all_is_unbounded(self):
        assert DateRange.all().is_unbounded is True
        assert str(DateRange.all()) == "all time"

    def test_default_construction_is_all_time(self):
        assert DateRange().is_unbounded is True

    def test_between_is_inclusive_at_both_ends(self):
        span = DateRange.between(date(2026, 9, 21), date(2026, 9, 23))
        assert span.contains(date(2026, 9, 21))
        assert span.contains(date(2026, 9, 23))
        assert not span.contains(date(2026, 9, 20))
        assert not span.contains(date(2026, 9, 24))

    def test_between_rejects_a_reversed_range(self):
        with pytest.raises(ValueError):
            DateRange.between(date(2026, 9, 23), date(2026, 9, 21))

    def test_a_single_day_range_is_allowed(self):
        one = DateRange.between(date(2026, 9, 21), date(2026, 9, 21))
        assert one.days() == [date(2026, 9, 21)]
        assert str(one) == "2026-09-21"

    def test_last_days_counts_inclusively(self):
        span = DateRange.last_days(3, date(2026, 9, 23))
        assert span.start == date(2026, 9, 21)
        assert span.end == date(2026, 9, 23)
        assert len(span.days()) == 3

    def test_last_days_of_one_is_a_single_day(self):
        span = DateRange.last_days(1, date(2026, 9, 23))
        assert span.days() == [date(2026, 9, 23)]

    def test_last_days_rejects_zero(self):
        with pytest.raises(ValueError):
            DateRange.last_days(0, date(2026, 9, 23))

    def test_days_of_an_unbounded_range_is_empty(self):
        """An open range has no enumerable length, and says so."""
        assert DateRange.all().days() == []

    def test_days_are_chronological_and_complete(self):
        span = DateRange.between(date(2026, 9, 21), date(2026, 9, 25))
        assert [d.day for d in span.days()] == [21, 22, 23, 24, 25]

    def test_open_ended_ranges_are_allowed(self):
        assert DateRange(start=date(2026, 9, 21)).contains(date(2030, 1, 1))
        assert DateRange(end=date(2026, 9, 21)).contains(date(2000, 1, 1))

    def test_sql_clause_is_empty_when_unbounded(self):
        assert DateRange.all().sql_clause() == ("", [])

    def test_sql_clause_binds_the_dates_as_parameters(self):
        """A date must never be interpolated into SQL."""
        clause, params = DateRange.between(
            date(2026, 9, 21), date(2026, 9, 23)
        ).sql_clause()
        assert clause.startswith(" WHERE ")
        assert "?" in clause
        assert params == ["2026-09-21", "2026-09-23"]

    def test_str_reads_as_a_sentence(self):
        assert str(DateRange.between(date(2026, 9, 21), date(2026, 9, 23))) == (
            "2026-09-21 to 2026-09-23"
        )

    def test_it_is_immutable(self):
        with pytest.raises(Exception):
            DateRange(start=date(2026, 9, 21)).start = date(2026, 1, 1)


# ----------------------------------------------------------------------
# Date-range filtering against real rows
# ----------------------------------------------------------------------
class TestRangeFiltering:
    def test_all_time_is_the_default(self, memory_db):
        two_tools_over_three_days(memory_db)
        service = AnalyticsService(memory_db)
        assert service.total_interactions() == 7
        assert service.total_interactions(DateRange.all()) == 7
        assert service.total_interactions(None) == 7

    def test_an_explicit_range_excludes_earlier_rows(
        self, memory_db, service
    ):
        two_tools_over_three_days(memory_db)
        span = DateRange.between(ANCHOR, ANCHOR + timedelta(days=1))
        assert service.total_interactions(span) == 5

    def test_a_single_day_range(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        span = DateRange.between(ANCHOR, ANCHOR)
        assert service.total_interactions(span) == 3

    def test_a_range_before_the_data_returns_zero(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        old = DateRange.between(date(2020, 1, 1), date(2020, 1, 31))
        assert service.total_interactions(old) == 0

    def test_a_range_after_the_data_returns_zero(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        future = DateRange.between(date(2030, 1, 1), date(2030, 1, 31))
        assert service.total_interactions(future) == 0

    def test_last_days_window(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        # The last two days contain day 1 and day 2.
        window = DateRange.last_days(2, ANCHOR + timedelta(days=2))
        assert service.total_interactions(window) == 4

    def test_first_and_last_are_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        span = DateRange.between(
            ANCHOR + timedelta(days=1), ANCHOR + timedelta(days=2)
        )
        assert service.first_interaction_time(span).startswith("2026-09-22")
        assert service.last_interaction_time(span).startswith("2026-09-23")

    def test_command_counts_are_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        span = DateRange.between(ANCHOR, ANCHOR)
        counts = {c.tool_name: c.count for c in service.command_counts(span)}
        assert counts == {"weather": 2, "jokes": 1}

    def test_most_used_is_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        # Day 0 favours weather; day 1 favours jokes.
        assert (
            service.most_used_command(DateRange.between(ANCHOR, ANCHOR))
            .tool_name
            == "weather"
        )
        assert (
            service.most_used_command(
                DateRange.between(ANCHOR + timedelta(days=1), ANCHOR + timedelta(days=1))
            )
            .tool_name
            == "jokes"
        )

    def test_active_days_is_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        assert service.active_days() == 3
        assert service.active_days(DateRange.between(ANCHOR, ANCHOR)) == 1

    def test_weekly_counts_are_filtered(self, memory_db, service):
        from assistant.analytics import TimeBucket

        two_tools_over_three_days(memory_db)
        # All three fixture days fall in one week, so narrowing to the first
        # two days narrows that week's total from 7 to 5.
        span = DateRange.between(ANCHOR, ANCHOR + timedelta(days=1))
        assert service.weekly_interaction_counts(date_range=span) == [
            TimeBucket("2026-W39", 5)
        ]
        assert service.weekly_interaction_counts() == [TimeBucket("2026-W39", 7)]


# ----------------------------------------------------------------------
# Rankings, percentages and ties
# ----------------------------------------------------------------------
class TestTopTools:
    def test_top_tools_are_ranked(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        assert [t.tool_name for t in service.top_tools()] == ["jokes", "weather"]

    def test_top_tools_respects_the_limit(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        assert len(service.top_tools(1)) == 1

    def test_top_tools_excludes_unmatched(self, memory_db, service):
        only_unmatched(memory_db)
        assert service.top_tools() == []

    def test_top_tools_never_returns_none(self, memory_db, service):
        empty(memory_db)
        assert service.top_tools() == []

    def test_top_tools_rejects_a_zero_limit(self, service):
        with pytest.raises(ValueError):
            service.top_tools(0)

    def test_a_tie_breaks_alphabetically_and_deterministically(
        self, memory_db, service
    ):
        tie_between_two_tools(memory_db)
        winners = {service.top_tools(1)[0].tool_name for _ in range(5)}
        assert winners == {"jokes"}

    def test_top_tools_is_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        # Day 0 holds weather x2 and jokes x1, so both appear; day 1 holds
        # jokes x2 alone, so only jokes appears there.
        assert [t.tool_name for t in service.top_tools(5, DateRange.between(ANCHOR, ANCHOR))] == [
            "weather",
            "jokes",
        ]
        day1 = DateRange.between(ANCHOR + timedelta(days=1), ANCHOR + timedelta(days=1))
        assert [t.tool_name for t in service.top_tools(5, day1)] == ["jokes"]


class TestUsagePercentages:
    def test_percentages_add_up_to_about_one_hundred(self, memory_db, service):
        """Rounded to one decimal, so the total may land on 100.1 or 99.9.

        Each row is rounded independently, so their sum is not guaranteed to
        be exactly 100. Asserting exact equality here would be asserting an
        arithmetic accident rather than the property that matters: the rows
        together account for the whole total.
        """
        two_tools_over_three_days(memory_db)
        usage = service.tool_usage()
        assert sum(u.count for u in usage) == service.total_interactions()
        assert abs(sum(u.percentage for u in usage) - 100.0) < 0.5

    def test_percentage_of_the_total(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        by_name = {u.tool_name: u for u in service.tool_usage()}
        assert by_name["jokes"].count == 3
        assert by_name["jokes"].percentage == 42.9

    def test_unmatched_is_reported_not_hidden(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        by_name = {u.tool_name: u for u in service.tool_usage()}
        assert by_name["none"].is_unmatched is True
        assert by_name["jokes"].is_unmatched is False

    def test_usage_is_empty_when_nothing_is_recorded(self, memory_db, service):
        empty(memory_db)
        assert service.tool_usage() == []

    def test_usage_is_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        span = DateRange.between(ANCHOR, ANCHOR)
        usage = service.tool_usage(span)
        assert sum(u.count for u in usage) == 3
        assert abs(sum(u.percentage for u in usage) - 100.0) < 0.5
        assert {u.tool_name for u in usage} == {"weather", "jokes"}

    def test_a_full_database_gives_a_hundred_percent_match(self, memory_db):
        data = analytics_fixture(memory_db)
        data.interactions("weather", 3)
        service = AnalyticsService(memory_db)
        assert service.match_rate() == 100.0
        assert service.unmatched_count() == 0

    def test_no_activity_is_zero_not_one_hundred(self, memory_db, service):
        empty(memory_db)
        assert service.match_rate() == 0.0

    def test_tool_usage_is_immutable(self):
        with pytest.raises(Exception):
            ToolUsage("weather", 1, 50.0).count = 2


# ----------------------------------------------------------------------
# Matched versus unmatched
# ----------------------------------------------------------------------
class TestMatchedVersusUnmatched:
    def test_the_two_totals_add_up_to_the_total(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        total = service.total_interactions()
        assert (
            service.matched_total() + service.unmatched_count() == total == 7
        )

    def test_both_sentinels_count_as_unmatched(self, memory_db, service):
        only_unmatched(memory_db)
        assert service.unmatched_count() == 3
        assert service.matched_total() == 0

    def test_unique_tools_ignores_both_sentinels(self, memory_db, service):
        only_unmatched(memory_db)
        assert service.unique_tools_used() == 0

    def test_most_used_ignores_unmatched(self, memory_db, service):
        only_unmatched(memory_db)
        assert service.most_used_command() is None

    def test_the_summary_carries_the_split(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        summary = service.summary()
        assert summary.matched_total == 6
        assert summary.unmatched_total == 1
        assert summary.match_rate == 85.7

    def test_the_split_is_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        # Day 0 has no unmatched rows at all.
        summary = service.summary(DateRange.between(ANCHOR, ANCHOR))
        assert summary.unmatched_total == 0
        assert summary.match_rate == 100.0


# ----------------------------------------------------------------------
# Time series for a dashboard
# ----------------------------------------------------------------------
class TestTimeSeries:
    def test_daily_series_is_chronological(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        assert [b.period for b in service.daily_interaction_counts()] == [
            "2026-09-21",
            "2026-09-22",
            "2026-09-23",
        ]

    def test_gaps_are_omitted_by_default(self, memory_db, service):
        data = analytics_fixture(memory_db)
        data.interaction("weather", day=data.day(0))
        data.interaction("weather", day=data.day(5))
        periods = [b.period for b in service.daily_interaction_counts()]
        assert periods == ["2026-09-21", "2026-09-26"]

    def test_a_requested_range_can_zero_fill(self, memory_db, service):
        """Only an explicit range triggers filling, because a chart needs it."""
        data = analytics_fixture(memory_db)
        data.interaction("weather", day=data.day(0))
        data.interaction("weather", day=data.day(3))
        span = DateRange.between(data.day(0), data.day(3))
        filled = service.daily_interaction_counts(date_range=span, fill=True)
        assert [(b.period, b.count) for b in filled] == [
            ("2026-09-21", 1),
            ("2026-09-22", 0),
            ("2026-09-23", 0),
            ("2026-09-24", 1),
        ]

    def test_filling_is_ignored_without_a_range(self, memory_db, service):
        data = analytics_fixture(memory_db)
        data.interaction("weather", day=data.day(0))
        assert len(service.daily_interaction_counts(fill=True)) == 1

    def test_filling_an_empty_range_yields_all_zeros(self, memory_db, service):
        span = DateRange.between(date(2020, 1, 1), date(2020, 1, 3))
        filled = service.daily_interaction_counts(date_range=span, fill=True)
        assert [b.count for b in filled] == [0, 0, 0]

    def test_weekly_series_is_chronological(self, memory_db, service):
        data = analytics_fixture(memory_db)
        data.interactions("weather", 2, day=data.day(0))     # W39
        data.interactions("weather", 3, day=data.day(8))     # W40
        assert [b.period for b in service.weekly_interaction_counts()] == [
            "2026-W39",
            "2026-W40",
        ]

    def test_empty_weeks_are_never_zero_filled(self, memory_db, service):
        """Documented: weeks have no enumeration, so they are never filled."""
        data = analytics_fixture(memory_db)
        data.interaction("weather", day=data.day(0))
        assert len(service.weekly_interaction_counts()) == 1

    def test_the_two_series_agree_on_the_total(self, memory_db, service):
        data = analytics_fixture(memory_db)
        for offset in (0, 8, 16):
            data.interaction("weather", day=data.day(offset))
        assert (
            sum(b.count for b in service.daily_interaction_counts())
            == sum(b.count for b in service.weekly_interaction_counts())
            == service.total_interactions()
        )

    def test_busiest_day(self, memory_db, service):
        data = analytics_fixture(memory_db)
        data.interaction("weather", day=data.day(0))
        data.interactions("weather", 3, day=data.day(1))
        assert service.busiest_day().period == "2026-09-22"
        assert service.busiest_day().count == 3

    def test_busiest_week(self, memory_db, service):
        data = analytics_fixture(memory_db)
        data.interactions("weather", 2, day=data.day(0))
        data.interactions("weather", 5, day=data.day(8))
        assert service.busiest_week() == service.weekly_interaction_counts()[1]
        assert service.busiest_week().count == 5

    def test_busiest_is_none_when_empty(self, memory_db, service):
        empty(memory_db)
        assert service.busiest_day() is None
        assert service.busiest_week() is None

    def test_busiest_day_is_filtered(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        # Day 0 has 3; the whole dataset also peaks at 3, so narrow to day 1
        # (2 interactions) to prove the filter is applied.
        span = DateRange.between(ANCHOR + timedelta(days=1), ANCHOR + timedelta(days=1))
        assert service.busiest_day(span).count == 2

    def test_a_one_bad_timestamp_does_not_break_the_week_series(
        self, memory_db, service
    ):
        memory_db.execute(
            "INSERT INTO interactions (occurred_at, tool_name) VALUES (?,?)",
            ("rubbish", "jokes"),
        )
        data = analytics_fixture(memory_db)
        data.interaction("weather", day=data.day(0))
        assert sum(b.count for b in service.weekly_interaction_counts()) == 1


# ----------------------------------------------------------------------
# Notes analytics
# ----------------------------------------------------------------------
class TestNotesAnalyticsExpansion:
    def test_notes_per_day(self, memory_db, service):
        notes_over_three_days(memory_db)
        assert service.total_notes() == 3
        assert service.note_days() == 3
        assert service.notes_per_day() == 1.0

    def test_notes_per_day_with_several_on_one_day(self, memory_db, service):
        data = analytics_fixture(memory_db)
        data.note("one", day=data.day(0))
        data.note("two", day=data.day(0))
        assert service.notes_per_day() == 2.0

    def test_average_note_length(self, memory_db, service):
        # Bodies of 8, 18 and 4 characters: a mean of exactly 10.0.
        notes_over_three_days(memory_db)
        assert service.average_note_length() == 10.0

    def test_longest_note_returns_the_text(self, memory_db, service):
        notes_over_three_days(memory_db)
        assert service.longest_note() == "a longer note here"
        assert service.longest_note_length() == 18

    def test_longest_note_breaks_a_tie_by_lowest_id(self, memory_db, service):
        data = analytics_fixture(memory_db)
        data.note("aaaa", day=data.day(0))
        data.note("bbbb", day=data.day(1))
        # Both are 4 characters, so the earlier row must win every time.
        assert {service.longest_note() for _ in range(5)} == {"aaaa"}

    def test_notes_figures_are_zero_when_empty(self, memory_db, service):
        empty(memory_db)
        assert service.average_note_length() == 0.0
        assert service.notes_per_day() == 0.0
        assert service.longest_note() is None
        assert service.note_days() == 0

    def test_notes_summary_holds_every_figure(self, memory_db, service):
        notes_over_three_days(memory_db)
        assert service.notes_summary() == NotesSummary(
            total=3,
            first_created="2026-09-21 12:00:00",
            last_created="2026-09-23 12:00:00",
            average_length=10.0,
            longest_length=18,
            days_with_notes=3,
        )

    def test_notes_summary_on_an_empty_database(self, memory_db, service):
        empty(memory_db)
        summary = service.notes_summary()
        assert summary.total == 0
        assert summary.has_notes is False
        assert summary.first_created is None

    def test_notes_are_not_filtered_by_an_interaction_range(
        self, memory_db, service
    ):
        """Documented asymmetry: a range filters interactions, not notes.

        Notes have their own timestamps and no meaningful link to an
        interaction window, so filtering them by the same range would imply a
        relationship the data does not contain.
        """
        notes_over_three_days(memory_db)
        two_tools_over_three_days(memory_db)
        summary = service.summary(DateRange.between(ANCHOR, ANCHOR))
        assert summary.total_interactions == 3
        assert summary.total_notes == 3

    def test_notes_summary_is_immutable(self):
        with pytest.raises(Exception):
            NotesSummary(1, None, None, 1.0, 1, 1).total = 2


# ----------------------------------------------------------------------
# Dashboard report
# ----------------------------------------------------------------------
class TestDashboardReport:
    def test_it_carries_every_group(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        notes_over_three_days(memory_db)
        report = service.dashboard()
        assert isinstance(report, DashboardReport)
        assert report.summary.total_interactions == 7
        assert [t.tool_name for t in report.top_tools] == ["jokes", "weather"]
        assert len(report.daily) == 3
        assert len(report.weekly) == 1
        assert report.notes.total == 3
        assert report.busiest_day.count == 3
        assert report.busiest_week.count == 7
        assert report.range_used == "all time"

    def test_the_series_are_immutable_tuples(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        report = service.dashboard()
        assert isinstance(report.daily, tuple)
        assert isinstance(report.weekly, tuple)
        assert isinstance(report.top_tools, tuple)

    def test_it_records_the_range_it_used(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        report = service.dashboard(DateRange.between(ANCHOR, ANCHOR))
        assert report.range_used == "2026-09-21"
        assert report.summary.total_interactions == 3

    def test_the_top_limit_is_honoured(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        assert len(service.dashboard(top_limit=1).top_tools) == 1

    def test_an_empty_database_gives_an_empty_report(self, memory_db, service):
        empty(memory_db)
        report = service.dashboard()
        assert report.summary.total_interactions == 0
        assert report.top_tools == ()
        assert report.daily == ()
        assert report.notes.total == 0
        assert report.busiest_day is None
        assert report.busiest_week is None

    def test_it_is_deterministic(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        assert service.dashboard() == service.dashboard()


# ----------------------------------------------------------------------
# Read-only, still
# ----------------------------------------------------------------------
class TestStillReadOnly:
    def test_no_write_statements_anywhere_in_the_module(self):
        import inspect

        import assistant.analytics.service as module

        source = inspect.getsource(module).upper()
        for statement in ("INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE"):
            assert statement not in source, statement

    def test_a_full_dashboard_changes_nothing(self, memory_db):
        two_tools_over_three_days(memory_db)
        notes_over_three_days(memory_db)
        before = (
            memory_db.query_one("SELECT COUNT(*) AS c FROM interactions")["c"],
            memory_db.query_one("SELECT COUNT(*) AS c FROM notes")["c"],
            memory_db.schema_version(),
        )
        service = AnalyticsService(memory_db)
        service.dashboard()
        service.dashboard(DateRange.all())
        service.tool_usage()
        service.top_tools(3)
        service.notes_summary()
        after = (
            memory_db.query_one("SELECT COUNT(*) AS c FROM interactions")["c"],
            memory_db.query_one("SELECT COUNT(*) AS c FROM notes")["c"],
            memory_db.schema_version(),
        )
        assert before == after == (7, 3, 2)

    def test_the_schema_version_is_untouched(self, memory_db):
        """Sprint 2 added metrics, not tables, so the version must not move."""
        from assistant.core.database import SCHEMA_VERSION

        two_tools_over_three_days(memory_db)
        AnalyticsService(memory_db).dashboard()
        assert memory_db.schema_version() == SCHEMA_VERSION == 2

    def test_a_range_with_no_data_does_not_raise(self, memory_db, service):
        empty(memory_db)
        span = DateRange.between(date(1999, 1, 1), date(1999, 12, 31))
        assert service.dashboard(span).summary.total_interactions == 0

    def test_an_uninitialised_database_still_raises(self):
        from assistant.core.database import Database

        database = Database(":memory:")
        try:
            with pytest.raises(sqlite3.Error):
                AnalyticsService(database).dashboard()
        finally:
            database.close()




    def test_the_summary_respects_the_range(self, memory_db, service):
        two_tools_over_three_days(memory_db)
        summary = service.summary(DateRange.between(ANCHOR, ANCHOR))
        assert summary.total_interactions == 3
        assert summary.active_days == 1
        assert summary.most_used_tool == "weather"
        assert summary.most_used_count == 2

