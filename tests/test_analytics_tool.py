"""Tests for the user-facing analytics tool.

Runs against temporary databases from ``conftest.py``, so the real
``data/assistant.db`` is never touched. The point of this file is that every
number the tool speaks can be traced back to a real row: the expected lines
below are written out in full, so a wording change or a wrong figure is a test
failure rather than something a user notices.
"""

from __future__ import annotations

import pytest

from assistant.analytics import AnalyticsSummary
from assistant.core.context import AppContext
from assistant.core.database import Database
from assistant.tools import build_default_router
from assistant.tools.analytics import (
    NO_ACTIVITY,
    NO_MATCHED_ACTIVITY,
    TOP_DAYS,
    AnalyticsTool,
    _readable_timestamp,
)


@pytest.fixture
def tool():
    return AnalyticsTool()


@pytest.fixture
def live(ctx: AppContext, db: Database) -> AppContext:
    """A context with the temporary database attached."""
    ctx.db = db
    return ctx


def record(db, tool_name="weather", when="2026-09-26 10:00:00"):
    db.execute(
        "INSERT INTO interactions (occurred_at, tool_name, utterance, response) "
        "VALUES (?,?,?,?)",
        (when, tool_name, "a command", "an answer"),
    )


def add_note(db, body="buy milk", when="2026-09-26 10:00:00"):
    db.execute(
        "INSERT INTO notes (created_at, body) VALUES (?,?)", (when, body)
    )


def spoken(tool, ctx) -> str:
    """Run the tool and join its lines into one string for asserting on."""
    return " ".join(tool.execute(ctx))


# ----------------------------------------------------------------------
# Empty and degraded cases
# ----------------------------------------------------------------------
class TestNoData:
    def test_without_a_database(self, tool, ctx):
        lines = tool.execute(ctx)
        assert len(lines) == 1
        assert "no usage data" in lines[0]

    def test_empty_database_says_so(self, tool, live):
        assert tool.execute(live) == [NO_ACTIVITY]

    def test_only_unmatched_commands(self, tool, live, db):
        """Real usage that matched nothing is not a tool to report."""
        record(db, tool_name="none")
        record(db, tool_name="none")
        assert tool.execute(live) == [NO_MATCHED_ACTIVITY]

    def test_a_closed_database_reopens_onto_the_same_data(self, tool, live, db):
        """Closing is not breakage: the file is still there and still has rows.

        The ``db`` fixture is a real file, so a reconnect finds the same
        schema and the same rows. The tool therefore reports the usage
        rather than an error, which is the correct answer.
        """
        record(db, tool_name="weather")
        db.close()
        text = spoken(tool, live)
        assert "1 command across 1 tool" in text
        assert "could not work out my usage" not in text

    def test_a_closed_database_with_no_rows_reads_as_empty(
        self, tool, live, db
    ):
        db.close()
        assert tool.execute(live) == [NO_ACTIVITY]

    def test_an_uninitialised_database_becomes_a_sentence(self, tool, ctx):
        database = Database(":memory:")
        try:
            ctx.db = database
            lines = tool.execute(ctx)
            assert "could not work out my usage" in lines[0]
        finally:
            database.close()


# ----------------------------------------------------------------------
# Formatting, driven by real rows
# ----------------------------------------------------------------------
class TestFormatting:
    def test_a_single_command(self, tool, live, db):
        record(db, tool_name="weather", when="2026-09-26 10:00:00")
        text = spoken(tool, live)
        assert "1 command" in text
        assert "across 1 tool" in text
        assert "over 1 day" in text
        assert "You use weather most, 1 times." in text
        assert "You have not saved any notes." in text

    def test_pluralisation_with_many(self, tool, live, db):
        for _ in range(5):
            record(db, tool_name="notes")
        record(db, tool_name="weather", when="2026-09-27 10:00:00")
        text = spoken(tool, live)
        assert "6 commands" in text
        assert "across 2 tools" in text
        assert "over 2 days" in text

    def test_the_most_used_tool_is_named(self, tool, live, db):
        for _ in range(3):
            record(db, tool_name="jokes")
        record(db, tool_name="facts")
        assert "You use jokes most, 3 times." in spoken(tool, live)

    def test_unmatched_commands_are_reported_not_hidden(self, tool, live, db):
        record(db, tool_name="weather")
        record(db, tool_name="none")
        record(db, tool_name="none")
        text = spoken(tool, live)
        assert "2 commands did not match anything I recognise." in text
        # Still counted in the total, because they are real usage, but they
        # are not counted as a tool.
        assert "3 commands across 1 tool" in text

    def test_one_unmatched_command_is_singular(self, tool, live, db):
        record(db, tool_name="weather")
        record(db, tool_name="none")
        assert "1 command did not match" in spoken(tool, live)

    def test_first_and_last_dates(self, tool, live, db):
        record(db, when="2026-09-25 08:00:00")
        record(db, when="2026-09-27 19:45:00")
        text = spoken(tool, live)
        assert "Your first command was on 2026-09-25, the latest on 2026-09-27." in text

    def test_a_single_day_is_reported_once(self, tool, live, db):
        record(db, when="2026-09-26 08:00:00")
        record(db, when="2026-09-26 19:00:00")
        text = spoken(tool, live)
        assert "All of that happened on 2026-09-26." in text

    def test_the_daily_breakdown_names_the_busiest_day_first(
        self, tool, live, db
    ):
        """Busiest first; ties then run from the most recent day backwards.

        The order is (count, date) sorted ascending and then reversed, which
        is what makes the busiest day land first and gives a total order, so
        the same data always produces the same sentence.
        """
        record(db, when="2026-09-25 08:00:00")
        for _ in range(3):
            record(db, when="2026-09-26 08:00:00")
        record(db, when="2026-09-27 08:00:00")
        text = spoken(tool, live)
        assert "By day: 2026-09-26 with 3, 2026-09-27 with 1, 2026-09-25 with 1." in text

    def test_the_daily_breakdown_is_deterministic(self, tool, live, db):
        for when in ("2026-09-25", "2026-09-26", "2026-09-27"):
            record(db, when=f"{when} 08:00:00")
        first = spoken(tool, live)
        assert all(spoken(tool, live) == first for _ in range(3))

    def test_the_daily_breakdown_is_capped(self, tool, live, db):
        for day in range(1, 10):
            record(db, when=f"2026-09-{day:02d} 08:00:00")
        text = spoken(tool, live)
        breakdown = text.split("By day:")[1]
        assert breakdown.count("2026-09-") == TOP_DAYS
        # The cap keeps the most recent days, not the oldest.
        assert "2026-09-09" in breakdown
        assert "2026-09-01" not in breakdown


# ----------------------------------------------------------------------
# Notes reporting
# ----------------------------------------------------------------------
class TestNotesReporting:
    def test_no_notes(self, tool, live, db):
        record(db)
        assert "You have not saved any notes." in spoken(tool, live)

    def test_one_note_is_singular(self, tool, live, db):
        record(db)
        add_note(db)
        text = spoken(tool, live)
        assert "You have 1 note. They were all added on 2026-09-26." in text

    def test_several_notes_over_several_days(self, tool, live, db):
        record(db)
        add_note(db, when="2026-09-25 08:00:00")
        add_note(db, when="2026-09-27 08:00:00")
        text = spoken(tool, live)
        assert "You have 2 notes." in text
        assert "The first was on 2026-09-25 and the latest on 2026-09-27." in text

    def test_notes_and_interactions_are_counted_separately(self, tool, live, db):
        for _ in range(4):
            record(db)
        for _ in range(2):
            add_note(db)
        text = spoken(tool, live)
        assert "4 commands" in text
        assert "You have 2 notes." in text


# ----------------------------------------------------------------------
# Realistic combined output
# ----------------------------------------------------------------------
class TestRealisticOutput:
    def test_a_full_session_reads_correctly(self, tool, live, db):
        """The whole summary, checked end to end against real rows."""
        record(db, tool_name="weather", when="2026-09-25 08:00:00")
        record(db, tool_name="jokes", when="2026-09-25 09:00:00")
        record(db, tool_name="notes", when="2026-09-25 10:00:00")
        record(db, tool_name="none", when="2026-09-25 11:00:00")
        record(db, tool_name="notes", when="2026-09-27 12:00:00")
        add_note(db, body="buy milk", when="2026-09-25 10:00:00")

        lines = tool.execute(live)
        text = " ".join(lines)
        assert "5 commands across 3 tools over 2 days." in text
        assert "You use notes most, 2 times." in text
        assert "1 command did not match anything I recognise." in text
        assert "Your first command was on 2026-09-25, the latest on 2026-09-27." in text
        assert "By day: 2026-09-25 with 4, 2026-09-27 with 1." in text
        assert "You have 1 note." in text
        # Every line is a non-empty string the app can speak.
        assert all(isinstance(line, str) and line.strip() for line in lines)

    def test_every_number_is_traceable_to_a_row(self, tool, live, db):
        """The guard against a fabricated figure appearing by accident."""
        for _ in range(3):
            record(db, tool_name="news")
        add_note(db)
        text = spoken(tool, live)
        # 3 interactions, 1 tool, 3 uses of news, 1 note: all real.
        assert "3 commands across 1 tool" in text
        assert "You use news most, 3 times." in text
        assert "You have 1 note." in text
        assert "4 commands" not in text
        assert "2 notes" not in text


# ----------------------------------------------------------------------
# Registration and routing
# ----------------------------------------------------------------------
class TestRegistration:
    def test_it_is_registered(self):
        assert "analytics" in build_default_router()

    def test_it_appears_in_the_help_text(self):
        assert "analytics" in build_default_router().help_text()

    @pytest.mark.parametrize(
        "utterance",
        [
            "show my analytics",
            "show analytics",
            "usage statistics",
            "how many commands have I used",
            "what do I use most",
            "my activity summary",
        ],
    )
    def test_each_declared_phrase_routes_here(self, utterance, live):
        from assistant.core.router import Match

        match = build_default_router().find_match(utterance)
        assert isinstance(match, Match), utterance
        assert match.tool_name == "analytics", utterance

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("history", "history"),
            ("what did I ask", "history"),
            ("note buy milk", "notes"),
            ("list notes", "notes"),
            ("tell me the weather", "weather"),
            ("play a song", "youtube"),
            ("exit", "system"),
            ("quit", "system"),
        ],
    )
    def test_no_other_tool_lost_a_command(self, utterance, expected, live):
        """A new tool must not steal a phrase another tool already owned."""
        from assistant.core.router import NoMatch

        match = build_default_router().find_match(utterance)
        if isinstance(match, NoMatch):
            pytest.fail(f"{utterance!r} no longer routes anywhere")
        assert match.tool_name == expected, utterance

    @pytest.mark.parametrize("utterance", ["summary", "statistics", "analytics"])
    def test_ambiguous_single_words_are_not_claimed(self, utterance, live):
        """Deliberately unregistered: too generic to grab safely."""
        from assistant.core.router import NoMatch

        match = build_default_router().find_match(utterance)
        assert isinstance(match, NoMatch), f"{utterance!r} was claimed"


# ----------------------------------------------------------------------
# Tool contract
# ----------------------------------------------------------------------
class TestToolContract:
    def test_name_and_description(self, tool):
        assert tool.name == "analytics"
        assert tool.description

    def test_patterns_are_lowercase_and_non_empty(self, tool):
        patterns = tool.patterns()
        assert patterns
        assert all(p == p.lower() and p.strip() for p in patterns)
        assert len(set(patterns)) == len(patterns)

    def test_no_pattern_is_a_single_generic_word(self, tool):
        """A one-word pattern would claim far more than it should."""
        for pattern in tool.patterns():
            assert len(pattern.split()) >= 2, pattern

    def test_extract_slots_defaults_to_empty(self, tool, live):
        assert tool.extract_slots("show analytics", live) == {}

    def test_handle_runs_the_whole_path(self, tool, live, db):
        record(db, tool_name="weather")
        result = tool.handle("show analytics", live)
        assert "1 command across 1 tool" in " ".join(result)

    def test_it_never_touches_speech(self, tool, live, db):
        """A tool must not call the speaker; the app does that."""
        from tests.conftest import ExplodingSpeaker

        record(db)
        live.speaker = ExplodingSpeaker()
        live.listener = ExplodingSpeaker()
        tool.execute(live)  # must not raise

    def test_repr(self, tool):
        assert repr(tool) == "<Tool analytics>"


# ----------------------------------------------------------------------
# Timestamp formatting
# ----------------------------------------------------------------------
class TestReadableTimestamp:
    def test_the_date_part_is_kept(self):
        assert _readable_timestamp("2026-09-26 10:29:24") == "2026-09-26"

    def test_none_stays_none(self):
        assert _readable_timestamp(None) is None
        assert _readable_timestamp("") is None

    def test_an_unexpected_shape_is_returned_unchanged(self):
        """Better to show the raw value than to invent a date."""
        assert _readable_timestamp("not-a-timestamp") == "not-a-timestamp"
        assert _readable_timestamp("2026-09-26") == "2026-09-26"

