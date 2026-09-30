"""Tests for the user-facing analytics tool.

Runs against temporary databases from ``conftest.py``, so the real
``data/assistant.db`` is never touched. The point of this file is that every
number the tool speaks can be traced back to a real row: the expected lines
below are written out in full, so a wording change or a wrong figure is a test
failure rather than something a user notices.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from assistant.analytics import AnalyticsSummary
from assistant.core.context import AppContext
from assistant.core.database import Database
from assistant.tools import build_default_router
from assistant.tools.analytics import (
    MODE_MOST_USED,
    MODE_RECENT,
    MODE_SUMMARY,
    MODE_TODAY,
    NO_ACTIVITY,
    NO_MATCHED_ACTIVITY,
    RECENT_DAYS,
    TOP_DAYS,
    AnalyticsTool,
    _MODE_PHRASES,
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


def spoken_with(tool, ctx, slots) -> str:
    """Run the tool with explicit slots and join the lines."""
    return " ".join(tool.execute(ctx, slots))


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
        assert "You use weather most, 1 time." in text
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

    def test_extract_slots_defaults_to_the_summary_mode(self, tool, live):
        """Sprint 2: the tool now always reports which report it will give.

        The default is the summary, so an utterance that matched the tool but
        named no mode still produces the full report rather than nothing.
        """
        assert tool.extract_slots("show analytics", live) == {"mode": "summary"}
        assert tool.extract_slots("", live) == {"mode": "summary"}

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
# Sprint 2: reporting modes
# ----------------------------------------------------------------------
class TestModeDetection:
    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("show analytics", MODE_SUMMARY),
            ("show my analytics", MODE_SUMMARY),
            ("usage statistics", MODE_SUMMARY),
            ("my activity summary", MODE_SUMMARY),
            ("usage summary", MODE_SUMMARY),
            ("what do i use most", MODE_MOST_USED),
            ("what do i use most today", MODE_MOST_USED),
            ("most used command", MODE_MOST_USED),
            ("my most used command", MODE_MOST_USED),
            ("which command do i use most", MODE_MOST_USED),
            ("today's activity", MODE_TODAY),
            ("todays activity", MODE_TODAY),
            ("what did i do today", MODE_TODAY),
            ("recent activity", MODE_RECENT),
            ("last 7 days", MODE_RECENT),
            ("this week", MODE_SUMMARY),  # never a mode: see the tool docstring
        ],
    )
    def test_the_mode_is_read_from_the_utterance(
        self, tool, live, utterance, expected
    ):
        assert tool.extract_slots(utterance, live)["mode"] == expected

    def test_a_longer_phrase_beats_the_shorter_one_it_contains(
        self, tool, live
    ):
        """"...most today" must not be read as a today question."""
        assert (
            tool.extract_slots("what do i use most today", live)["mode"]
            == MODE_MOST_USED
        )

    def test_an_unrecognised_utterance_falls_back_to_the_summary(
        self, tool, live
    ):
        assert tool.extract_slots("something else entirely", live)["mode"] == (
            MODE_SUMMARY
        )

    def test_detection_is_case_insensitive(self, tool, live):
        assert (
            tool.extract_slots("SHOW MY ANALYTICS", live)["mode"] == MODE_SUMMARY
        )
        assert tool.extract_slots("RECENT ACTIVITY", live)["mode"] == MODE_RECENT

    def test_every_mode_phrase_has_two_or_more_words(self):
        """No single generic word may select a mode."""
        for phrase, _mode in _MODE_PHRASES:
            assert len(phrase.split()) >= 2, phrase

    def test_every_declared_pattern_is_unique(self, tool):
        patterns = tool.patterns()
        assert len(patterns) == len(set(patterns))

    def test_the_pattern_list_includes_the_mode_phrases(self, tool):
        patterns = set(tool.patterns())
        for phrase, _mode in _MODE_PHRASES:
            assert phrase in patterns, phrase


class TestNoCollisionsWithOtherTools:
    """Sprint 2 regression: a time window is not a safe pattern.

    ``this week`` was briefly registered and it broke a genuine weather
    question. As an 8-character pattern it outranked the weather alias
    ``is it going to rain`` when the two appeared in the same sentence, and
    ``is it going to rain this week`` became ambiguous instead of reaching
    the weather tool.

    The general rule this encodes: a phrase may only claim an utterance if it
    is specific to analytics. A bare time window is not, so none is
    registered.
    """

    def test_the_weather_question_resolves_again(self, live):
        from assistant.app import build_runtime_lexicon, resolve

        router = build_default_router()
        lexicon = build_runtime_lexicon(router)
        tool = resolve(live, router, "is it going to rain this week", lexicon)
        assert tool is not None
        assert tool.name == "weather", "the analytics tool stole a weather query"

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("is it going to rain", "weather"),
            ("is it going to rain this week", "weather"),
            ("will it rain this week", "weather"),
            ("show me this week's weather", "weather"),
            ("what is the weather", "weather"),
        ],
    )
    def test_weather_phrases_are_unaffected(self, live, utterance, expected):
        from assistant.app import build_runtime_lexicon, resolve

        router = build_default_router()
        lexicon = build_runtime_lexicon(router)
        tool = resolve(live, router, utterance, lexicon)
        assert tool is not None, utterance
        assert tool.name == expected, utterance

    @pytest.mark.parametrize("phrase", ["this week", "last week", "today"])
    def test_no_bare_time_window_is_a_pattern(self, tool, phrase):
        """The specific rule that prevents the regression recurring."""
        assert phrase not in tool.patterns()

    def test_no_pattern_is_contained_in_another_tools_alias(self, tool):
        """No analytics phrase may be a substring of a rival's vocabulary.

        This is a general safety net, not the check that caught ``this week``.
        That phrase is not a substring of any single alias -- it collided only
        once it appeared *alongside* one in a longer sentence -- so the routing
        tests above are what actually guard it. This check covers the simpler
        case where one tool's vocabulary swallows another's outright.
        """
        from assistant.app import COMMON_ALIASES

        analytics_patterns = set(tool.patterns())
        for intent, aliases in COMMON_ALIASES.items():
            if intent == "analytics":
                continue
            for alias in aliases:
                for pattern in analytics_patterns:
                    assert pattern not in alias, (
                        f"analytics pattern {pattern!r} sits inside the "
                        f"{intent} alias {alias!r}"
                    )

    def test_the_other_tools_still_own_their_commands(self, live):
        from assistant.core.router import NoMatch

        router = build_default_router()
        for utterance, expected in (
            ("history", "history"),
            ("note buy milk", "notes"),
            ("play a song", "youtube"),
            ("tell me a joke", "jokes"),
            ("exit", "system"),
        ):
            match = router.find_match(utterance)
            assert not isinstance(match, NoMatch), utterance
            assert match.tool_name == expected, utterance


class TestPluralisation:
    """Spoken output must not say "1 times".

    This was wrong in Sprint 1 and every mode that reports a count shared the
    same mistake. The rule now lives in one helper, so it is pinned there and
    checked in the output of each mode.
    """

    def test_the_helper_agrees_with_the_count(self):
        from assistant.tools.analytics import _times

        assert _times(0) == "times"
        assert _times(1) == "time"
        assert _times(2) == "times"
        assert _times(11) == "times"

    @pytest.mark.parametrize("mode", [MODE_SUMMARY, MODE_MOST_USED])
    def test_no_mode_ever_says_one_times(self, tool, live, db, mode):
        record(db, tool_name="weather")
        record(db, tool_name="weather")
        record(db, tool_name="notes")
        text = spoken_with(tool, live, {"mode": mode})
        assert "1 times" not in text

    def test_the_today_mode_says_time_for_one(self, tool, live, db):
        today = date.today()
        record(db, tool_name="weather", when=f"{today} 09:00:00")
        text = spoken_with(tool, live, {"mode": MODE_TODAY})
        assert "1 time." in text
        assert "1 times" not in text

    def test_the_recent_mode_says_time_for_one(self, tool, live, db):
        record(db, tool_name="weather", when=f"{date.today()} 09:00:00")
        text = spoken_with(tool, live, {"mode": MODE_RECENT})
        assert "1 time." in text


class TestMostUsedMode:
    def test_it_ranks_the_tools(self, tool, live, db):
        for _ in range(3):
            record(db, tool_name="jokes")
        record(db, tool_name="weather")
        lines = tool.execute(live, {"mode": MODE_MOST_USED})
        assert lines[0] == "Here is what you use most:"
        assert "jokes, 3 times." in lines
        assert "weather, 1 time." in lines

    def test_it_never_lists_unmatched(self, tool, live, db):
        for _ in range(5):
            record(db, tool_name="none")
        record(db, tool_name="weather")
        text = spoken_with(tool, live, {"mode": MODE_MOST_USED})
        assert "none" not in text
        assert "weather" in text

    def test_it_says_so_when_nothing_matched(self, tool, live, db):
        record(db, tool_name="none")
        assert tool.execute(live, {"mode": MODE_MOST_USED}) == [
            NO_MATCHED_ACTIVITY
        ]

    def test_it_works_when_only_unmatched_rows_exist(self, tool, live, db):
        record(db, tool_name="none")
        assert tool.execute(live, {"mode": MODE_MOST_USED}) == [
            NO_MATCHED_ACTIVITY
        ]


class TestTodayMode:
    """The only clock-dependent mode, so its rows are pinned to the clock.

    Rather than freeze time, the fixture writes rows dated ``date.today()``.
    That keeps the test correct whenever it runs and proves the window is
    today's rather than a hard-coded day.
    """

    def test_it_reports_todays_commands(self, tool, live, db):
        today = date.today()
        record(db, tool_name="weather", when=f"{today} 09:00:00")
        record(db, tool_name="jokes", when=f"{today} 10:00:00")
        record(db, tool_name="jokes", when=f"{today} 11:00:00")
        text = spoken_with(tool, live, {"mode": MODE_TODAY})
        assert f"Today, {today.isoformat()}, you have made 3 commands." in text
        assert "The most used was jokes, 2 times." in text

    def test_it_excludes_older_rows(self, tool, live, db):
        today = date.today()
        record(db, tool_name="weather", when=f"{today} 09:00:00")
        old = today - timedelta(days=30)
        for _ in range(5):
            record(db, tool_name="jokes", when=f"{old} 09:00:00")
        text = spoken_with(tool, live, {"mode": MODE_TODAY})
        assert "you have made 1 command." in text

    def test_it_says_so_when_there_is_none_today(self, tool, live, db):
        old = date.today() - timedelta(days=30)
        for _ in range(3):
            record(db, tool_name="weather", when=f"{old} 09:00:00")
        lines = tool.execute(live, {"mode": MODE_TODAY})
        assert len(lines) == 1
        assert "You have not used me today" in lines[0]

    def test_a_single_command_today_is_singular(self, tool, live, db):
        today = date.today()
        record(db, tool_name="weather", when=f"{today} 09:00:00")
        text = spoken_with(tool, live, {"mode": MODE_TODAY})
        assert "you have made 1 command." in text


class TestRecentMode:
    def test_it_covers_the_last_seven_days(self, tool, live, db):
        today = date.today()
        for offset in range(RECENT_DAYS):
            record(db, tool_name="weather", when=f"{today - timedelta(days=offset)} 09:00:00")
        text = spoken_with(tool, live, {"mode": MODE_RECENT})
        assert f"In the last {RECENT_DAYS} days you have made {RECENT_DAYS} commands." in text

    def test_it_excludes_older_rows(self, tool, live, db):
        today = date.today()
        record(db, tool_name="weather", when=f"{today} 09:00:00")
        stale = today - timedelta(days=RECENT_DAYS + 5)
        for _ in range(9):
            record(db, tool_name="jokes", when=f"{stale} 09:00:00")
        text = spoken_with(tool, live, {"mode": MODE_RECENT})
        assert "you have made 1 command." in text

    def test_it_says_so_when_the_window_is_empty(self, tool, live, db):
        stale = date.today() - timedelta(days=60)
        for _ in range(3):
            record(db, tool_name="weather", when=f"{stale} 09:00:00")
        lines = tool.execute(live, {"mode": MODE_RECENT})
        assert len(lines) == 1
        assert f"You have not used me in the last {RECENT_DAYS} days." in lines[0]

    def test_it_names_the_top_tool_in_the_window(self, tool, live, db):
        today = date.today()
        for _ in range(3):
            record(db, tool_name="jokes", when=f"{today} 09:00:00")
        record(db, tool_name="weather", when=f"{today} 10:00:00")
        assert "You use jokes most, 3 times." in spoken_with(
            tool, live, {"mode": MODE_RECENT}
        )


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

