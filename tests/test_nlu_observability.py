"""Tests for NLU observability: the ``--debug-nlu`` flag and logging.

Two guarantees are checked here.

First, **observability changes nothing**. Every command that resolved a
tool before this step must resolve the same tool now, and the diagnostics
must never reach the speaker or the listener.

Second, the diagnostics must be *useful and stable*: the same command must
always produce the same block, and the block must show the facts a later
tuning pass needs, notably the weak candidates that explain a rejection.
"""

from __future__ import annotations

import pytest

from assistant.app import (
    STATUS_AMBIGUOUS,
    STATUS_FALLBACK,
    STATUS_NO_MATCH,
    STATUS_RESOLVED,
    Resolution,
    build_runtime_lexicon,
    format_nlu_debug,
    parse_args,
    resolve,
    resolve_detail,
    run,
    _ranked_candidates,
)
from assistant.core.context import AppContext
from assistant.tools import build_default_router


class ScriptedListener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def listen(self):
        self.calls += 1
        return self.responses.pop(0) if self.responses else ""


class RecordingSpeaker:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


@pytest.fixture
def router():
    return build_default_router()


@pytest.fixture
def lexicon(router):
    return build_runtime_lexicon(router)


def make_ctx(config, db=None, replies=()):
    ctx = AppContext(config=config)
    ctx.speaker = RecordingSpeaker()
    ctx.listener = ScriptedListener(list(replies))
    if db is not None:
        ctx.db = db
    return ctx


@pytest.fixture
def live(config, db):
    return make_ctx(config, db)


def block(ctx, router, lexicon, utterance, replies=()):
    """Resolve one utterance and return its debug block as a list of lines."""
    fresh = make_ctx(ctx.config, replies=replies)
    resolution = resolve_detail(fresh, router, utterance, lexicon)
    text = format_nlu_debug(
        utterance, resolution, _ranked_candidates(utterance, lexicon)
    )
    return text.splitlines()


# ----------------------------------------------------------------------
# The flag itself
# ----------------------------------------------------------------------
class TestDebugFlag:
    def test_disabled_by_default(self):
        assert parse_args([]).debug_nlu is False
        assert parse_args(["--text"]).debug_nlu is False

    def test_flag_enables_it(self):
        assert parse_args(["--debug-nlu"]).debug_nlu is True

    def test_text_flag_still_works_alongside_it(self):
        args = parse_args(["--text", "--debug-nlu"])
        assert args.text is True
        assert args.debug_nlu is True

    def test_log_level_still_works_alongside_it(self):
        args = parse_args(["--debug-nlu", "--log-level", "debug"])
        assert args.log_level == "debug"
        assert args.debug_nlu is True


class TestDebugIsOffUnlessAsked:
    def test_nothing_printed_in_normal_mode(self, config, db, capsys):
        ctx = make_ctx(config, db, ["exit"])
        run(ctx, build_default_router())
        assert "[NLU]" not in capsys.readouterr().out

    def test_nothing_printed_when_debug_is_false(self, config, db, capsys):
        ctx = make_ctx(config, db, ["exit"])
        run(ctx, build_default_router(), debug_nlu=False)
        assert "[NLU]" not in capsys.readouterr().out

    def test_printed_when_debug_is_true(self, config, db, capsys):
        ctx = make_ctx(config, db, ["exit"])
        run(ctx, build_default_router(), debug_nlu=True)
        out = capsys.readouterr().out
        assert "[NLU]" in out
        assert "input: exit" in out
        assert "intent: system" in out


# ----------------------------------------------------------------------
# What the block shows
# ----------------------------------------------------------------------
class TestBlockContents:
    def test_starts_with_the_marker(self, live, router, lexicon):
        assert block(live, router, lexicon, "exit")[0] == "[NLU]"

    def test_shows_the_input(self, live, router, lexicon):
        assert "input: wether" in block(live, router, lexicon, "wether")

    def test_exact_intent_prints(self, live, router, lexicon):
        lines = block(live, router, lexicon, "what is the weather")
        joined = "\n".join(lines)
        assert "status: resolved" in joined
        assert "intent: weather" in joined
        assert "score: 1.000" in joined
        assert "confidence: clear" in joined
        assert "method: exact" in joined
        assert "resolved_tool: weather" in joined

    def test_fuzzy_intent_prints_score_and_method(self, live, router, lexicon):
        joined = "\n".join(block(live, router, lexicon, "wether"))
        assert "intent: weather" in joined
        assert "method: fuzzy" in joined
        assert "score: 0.923" in joined
        assert "resolved_tool: weather" in joined

    def test_ambiguous_prints_alternatives(self, live, router, lexicon):
        joined = "\n".join(
            block(live, router, lexicon, "information about the news")
        )
        assert "status: ambiguous" in joined
        assert "intent: information" in joined
        assert "confidence: ambiguous" in joined
        assert "alternatives:" in joined
        assert "- news (1.000)" in joined

    def test_ambiguous_prints_the_full_ranking(self, live, router, lexicon):
        joined = "\n".join(
            block(live, router, lexicon, "information about the news")
        )
        assert "candidates:" in joined
        assert "1. information  1.000  exact" in joined
        assert "2. news         1.000  exact" in joined

    def test_no_match_says_so(self, live, router, lexicon):
        joined = "\n".join(block(live, router, lexicon, "quite good"))
        assert "status: no-match" in joined
        assert "intent: none" in joined
        assert "confidence: none" in joined
        assert "resolved_tool: none" in joined

    def test_rejected_candidate_is_visible(self, live, router, lexicon):
        # The whole point of the ranking: "play" inside "replay" was seen
        # and then refused, which is otherwise invisible.
        joined = "\n".join(block(live, router, lexicon, "replay my song"))
        assert "youtube" in joined
        assert "containment" in joined
        assert "resolved_tool: none" in joined

    def test_nothing_internal_is_dumped(self, live, router, lexicon):
        joined = "\n".join(
            block(live, router, lexicon, "information about the news")
        )
        for leak in ("ParsedIntent(", "Lexicon(", "re.compile", "Candidate(", "0x"):
            assert leak not in joined

    def test_one_block_per_command(self, config, db, capsys):
        ctx = make_ctx(config, db, ["exit", "note hello there"])
        ctx.listener = ScriptedListener(["note hello there", "exit"])
        run(ctx, build_default_router(), debug_nlu=True)
        out = capsys.readouterr().out
        assert out.count("[NLU]") == 2


# ----------------------------------------------------------------------
# The whole point: nothing changed
# ----------------------------------------------------------------------
class TestNoBehaviourChange:
    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("what is the weather", "weather"),
            ("wether", "weather"),
            ("tell me the news", "news"),
            ("joke", "jokes"),
            ("note buy milk", "notes"),
            ("history", "history"),
            ("play lofi beats", "youtube"),
            ("exit", "system"),
        ],
    )
    def test_still_resolves_the_same_tool(
        self, live, router, lexicon, utterance, expected
    ):
        tool = resolve(live, router, utterance, lexicon)
        assert tool is not None
        assert tool.name == expected

    @pytest.mark.parametrize(
        "utterance",
        [
            "quite good",
            "replay my song",
            "playback speed",
            "i noted that down",
            "goodbye is in the dictionary",
            "I don't want to exit",
            "do not play music",
        ],
    )
    def test_false_positives_stay_blocked(
        self, live, router, lexicon, utterance
    ):
        assert resolve(live, router, utterance, lexicon) is None

    def test_ambiguous_stays_ambiguous(self, live, router, lexicon):
        resolution = resolve_detail(
            live, router, "information about the news", lexicon
        )
        assert resolution.status == STATUS_AMBIGUOUS
        assert resolution.parsed.confidence == "ambiguous"

    def test_debug_does_not_change_the_tool(self, config, db, capsys):
        quiet = make_ctx(config, db, ["note quiet run", "exit"])
        run(quiet, build_default_router())
        quiet_spoken = list(quiet.speaker.spoken)

        loud = make_ctx(config, db, ["note loud run", "exit"])
        run(loud, build_default_router(), debug_nlu=True)
        capsys.readouterr()

        # Same number of utterances spoken: debugging only prints.
        assert len(quiet_spoken) == len(loud.speaker.spoken)

    def test_debug_is_not_spoken(self, config, db):
        ctx = make_ctx(config, db, ["exit"])
        run(ctx, build_default_router(), debug_nlu=True)
        for line in ctx.speaker.spoken:
            assert "[NLU]" not in line
            assert "intent:" not in line
            assert "candidates:" not in line

    def test_debug_output_never_reaches_the_speaker(self, config, db):
        ctx = make_ctx(config, db, ["what is the weather", "exit"])
        run(ctx, build_default_router(), debug_nlu=True)
        joined = " ".join(ctx.speaker.spoken)
        # The tool's real answer is still what gets spoken...
        assert "Temperature in Hyderabad" in joined
        # ...and none of the diagnostic fields leak into it.
        assert "score:" not in joined
        assert "method:" not in joined
        assert "trigger:" not in joined

    def test_debug_never_consumes_a_listener_turn(self, config, db):
        # Two commands in, two commands out: the flag must not add a
        # listen() of its own.
        ctx = make_ctx(config, db, ["history", "exit"])
        run(ctx, build_default_router(), debug_nlu=True)
        assert ctx.listener.calls == 2


# ----------------------------------------------------------------------
# Fallback visibility
# ----------------------------------------------------------------------
def _break_parser(monkeypatch):
    import assistant.app as app

    def boom(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(app, "parse", boom)


class TestFallbackIsVisible:
    def test_parser_exception_is_reported(
        self, live, router, lexicon, monkeypatch
    ):
        import assistant.app as app

        _break_parser(monkeypatch)
        resolution = app.resolve_detail(live, router, "tell me a joke", lexicon)
        assert resolution.status == STATUS_FALLBACK
        assert "exception" in resolution.reason
        assert resolution.tool_name == "jokes"

    def test_fallback_block_says_why(self, config, router, lexicon, monkeypatch):
        import assistant.app as app

        _break_parser(monkeypatch)
        resolution = app.resolve_detail(
            make_ctx(config), router, "tell me a joke", lexicon
        )
        joined = format_nlu_debug("tell me a joke", resolution)
        assert "status: fallback" in joined
        assert "reason: parser exception" in joined
        assert "resolved_tool: jokes" in joined

    def test_unregistered_intent_is_reported(
        self, live, router, lexicon, monkeypatch
    ):
        import assistant.app as app
        from assistant.nlu import ParsedIntent

        monkeypatch.setattr(
            app,
            "parse",
            lambda *a, **k: ParsedIntent(
                name="ghost_tool",
                score=1.0,
                trigger="x",
                method="exact",
                confidence="clear",
                entities={},
                alternatives=(),
            ),
        )
        resolution = app.resolve_detail(live, router, "tell me a joke", lexicon)
        assert resolution.status == STATUS_FALLBACK
        assert resolution.reason == "unregistered intent"

    def test_fallback_still_works_in_normal_mode(
        self, live, router, lexicon, monkeypatch
    ):
        import assistant.app as app

        _break_parser(monkeypatch)
        assert app.resolve(live, router, "tell me a joke", lexicon).name == "jokes"

    def test_no_match_does_not_report_a_fallback(self, live, router, lexicon):
        # The Step 5 safety behaviour must be reported accurately: a
        # parser None is a decision, not a fallback.
        resolution = resolve_detail(live, router, "quite good", lexicon)
        assert resolution.status == STATUS_NO_MATCH
        assert resolution.reason == ""

    def test_debug_does_not_reenable_the_unsafe_fallback(
        self, live, router, lexicon
    ):
        # Turning diagnostics on must never bring the legacy router back
        # for a confident "not a command".
        resolution = resolve_detail(live, router, "quite good", lexicon)
        assert resolution.status == STATUS_NO_MATCH
        assert resolution.tool is None


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
class TestCommandLogging:
    def test_old_log_line_is_unchanged_without_metadata(self, ctx, caplog):
        with caplog.at_level("INFO", logger="assistant"):
            ctx.log_command("jokes", "tell me a joke", "setup | punchline")
        assert any(
            r.getMessage()
            == "command=jokes utterance='tell me a joke' response='setup | punchline'"
            for r in caplog.records
        )

    def test_metadata_is_appended_when_supplied(self, ctx, caplog):
        with caplog.at_level("INFO", logger="assistant"):
            ctx.log_command(
                "weather",
                "wether",
                "sunny",
                nlu={
                    "status": "resolved",
                    "intent": "weather",
                    "score": 0.923,
                    "confidence": "clear",
                    "method": "fuzzy",
                    "trigger": "weather",
                },
            )
        message = [r for r in caplog.records if "nlu=" in r.getMessage()][0].getMessage()
        assert message.startswith(
            "command=weather utterance='wether' response='sunny' nlu="
        )
        for part in ("intent=weather", "method=fuzzy", "confidence=clear"):
            assert part in message

    def test_existing_fields_keep_their_meaning(self, ctx, caplog):
        with caplog.at_level("INFO", logger="assistant"):
            ctx.log_command("notes", "note milk", "Saved", nlu={"intent": "notes"})
        message = caplog.records[-1].getMessage()
        assert "command=notes" in message
        assert "utterance='note milk'" in message
        assert "response='Saved'" in message

    def test_metadata_is_never_stored(self, ctx_with_db):
        ctx_with_db.log_command(
            "weather", "wether", "sunny", nlu={"intent": "weather"}
        )
        row = ctx_with_db.db.query_one(
            "SELECT tool_name, utterance, response FROM interactions"
        )
        assert row["tool_name"] == "weather"
        assert row["utterance"] == "wether"
        assert row["response"] == "sunny"

    def test_schema_is_unchanged(self, ctx_with_db):
        columns = [
            row["name"]
            for row in ctx_with_db.db.query_all("PRAGMA table_info(interactions)")
        ]
        assert columns == ["id", "occurred_at", "tool_name", "utterance", "response"]

    def test_no_metadata_for_a_bare_no_match(self, live, router, lexicon):
        assert resolve_detail(live, router, "quite good", lexicon).metadata() is None

    def test_metadata_is_present_for_a_real_decision(self, live, router, lexicon):
        data = resolve_detail(live, router, "wether", lexicon).metadata()
        assert data["intent"] == "weather"
        assert data["method"] == "fuzzy"
        assert data["confidence"] == "clear"

    def test_metadata_alternatives_are_rendered(self, live, router, lexicon):
        resolution = resolve_detail(
            live, router, "information about the news", lexicon
        )
        assert resolution.metadata()["alternatives"] == ["news(1.0)"]


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
class TestDeterminism:
    @pytest.mark.parametrize(
        "utterance",
        [
            "wether",
            "information about the news",
            "quite good",
            "replay my song",
            "exit",
            "note buy milk",
        ],
    )
    def test_same_block_every_time(self, live, router, lexicon, utterance):
        first = "\n".join(block(live, router, lexicon, utterance))
        second = "\n".join(block(live, router, lexicon, utterance))
        assert first == second

    def test_metadata_key_order_is_fixed(self, live, router, lexicon):
        resolution = resolve_detail(live, router, "wether", lexicon)
        assert list(resolution.metadata()) == [
            "status",
            "intent",
            "score",
            "confidence",
            "method",
            "trigger",
        ]


class TestTheNamedExamples:
    def test_wether(self, live, router, lexicon):
        joined = "\n".join(block(live, router, lexicon, "wether"))
        assert "resolved_tool: weather" in joined
        assert "method: fuzzy" in joined

    def test_information_about_the_news(self, live, router, lexicon):
        joined = "\n".join(
            block(live, router, lexicon, "information about the news")
        )
        assert "confidence: ambiguous" in joined
        assert "- news (1.000)" in joined

    def test_quite_good(self, live, router, lexicon):
        assert "resolved_tool: none" in "\n".join(
            block(live, router, lexicon, "quite good")
        )

    def test_replay_my_song(self, live, router, lexicon):
        assert "resolved_tool: none" in "\n".join(
            block(live, router, lexicon, "replay my song")
        )

    def test_exit(self, live, router, lexicon):
        assert "resolved_tool: system" in "\n".join(
            block(live, router, lexicon, "exit")
        )
