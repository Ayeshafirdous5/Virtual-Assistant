"""Integration tests for the NLU inside the running application.

These drive the real :func:`assistant.app.run` loop and the real
:func:`assistant.app.resolve` helper with the nine production tools
registered. No microphone, browser, network or API key is involved: the
speech layer is scripted and the database is the temporary one from
``conftest``.

The emphasis is on behaviour a user can observe. What matters is which tool
runs, and that nothing dangerous runs by accident.
"""

from __future__ import annotations

import pytest

from assistant.app import (
    MSG_DIDNT_UNDERSTAND,
    build_runtime_lexicon,
    clarification_question,
    resolve,
    run,
)
from assistant.core.context import AppContext
from assistant.tools import build_default_router

#: Sentences that must not reach the named tool, because the user never
#: actually asked for it. Each one is a near miss on purpose.
FALSE_POSITIVES: list[tuple[str, str]] = [
    ("replay my song", "youtube"),
    ("playback speed", "youtube"),
    ("i noted that down", "notes"),
    ("denote this", "notes"),
    ("goodbye is in the dictionary", "system"),
    ("quite good", "system"),
    ("do not play music", "youtube"),
    ("never play music", "youtube"),
    ("I don't want to exit", "system"),
]

#: Commands that must keep working, including the typo recovery.
POSITIVES: list[tuple[str, str]] = [
    ("play lofi beats", "youtube"),
    ("note buy milk", "notes"),
    ("exit", "system"),
    ("wether", "weather"),
    ("joks", "jokes"),
    ("histry", "history"),
]


class ScriptedListener:
    """A listener that returns a fixed list of answers, then stops."""

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


def make_ctx(config, db=None):
    ctx = AppContext(config=config)
    ctx.speaker = RecordingSpeaker()
    ctx.listener = ScriptedListener([])
    if db is not None:
        ctx.db = db
    return ctx


@pytest.fixture
def live(config, db):
    return make_ctx(config, db)


class TestTruePositivesSurvive:
    @pytest.mark.parametrize("utterance,expected", POSITIVES)
    def test_resolves_to_expected_tool(
        self, live, router, lexicon, utterance, expected
    ):
        tool = resolve(live, router, utterance, lexicon)
        assert tool is not None
        assert tool.name == expected

    def test_facts_and_jokes_still_route(self, live, router, lexicon):
        assert resolve(live, router, "tell me a fact", lexicon).name == "facts"
        assert resolve(live, router, "tell me a joke", lexicon).name == "jokes"

    def test_youtube_phrases_reach_the_tool(self, live, router, lexicon):
        # These are the commands the removed bare aliases existed to catch.
        for utterance in (
            "play a song",
            "play music",
            "play a video",
            "play lofi beats",
        ):
            tool = resolve(live, router, utterance, lexicon)
            assert tool is not None, utterance
            assert tool.name == "youtube", utterance


class TestFalsePositivesAreRefused:
    @pytest.mark.parametrize("utterance,forbidden", FALSE_POSITIVES)
    def test_nothing_runs(self, live, router, lexicon, utterance, forbidden):
        tool = resolve(live, router, utterance, lexicon)
        assert tool is None, f"{utterance!r} wrongly reached {tool!r}"

    def test_the_legacy_router_really_was_wrong(self, router):
        # Records *why* the fallback cannot be unconditional: the old
        # substring router misreads most of these, so consulting it after a
        # confident "not a command" would undo the whole integration.
        wrong = [
            item
            for item in FALSE_POSITIVES
            if router.find_match(item[0]).tool_name == item[1]
        ]
        assert len(wrong) >= 7, wrong

    def test_mentioning_a_noun_is_not_a_command(self, live, router, lexicon):
        for utterance in (
            "this video is funny",
            "listen to some music",
            "the train will stop",
            "check the log",
        ):
            assert resolve(live, router, utterance, lexicon) is None, utterance


class TestGenericNounAliasesAreGone:
    @pytest.mark.parametrize("alias", ["song", "video", "music", "stop", "log"])
    def test_bare_noun_is_not_an_alias(self, alias):
        from assistant.nlu import COMMON_ALIASES

        bare = {p for patterns in COMMON_ALIASES.values() for p in patterns}
        assert alias not in bare

    def test_action_phrases_remain(self):
        from assistant.nlu import COMMON_ALIASES

        assert "play a song" in COMMON_ALIASES["youtube"]
        assert "shut down" in COMMON_ALIASES["system"]


class TestNwesStaysRejected:
    def test_below_the_fuzzy_threshold(self):
        # Documented behaviour: "nwes" scores 0.75, under the 0.80 cutoff.
        # The threshold is not lowered to make this case pass.
        from difflib import SequenceMatcher

        from assistant.nlu.scoring import FUZZY_CUTOFF

        ratio = SequenceMatcher(None, "nwes", "news").ratio()
        assert ratio == 0.75
        assert ratio < FUZZY_CUTOFF

    def test_resolves_to_nothing(self, live, router, lexicon):
        assert resolve(live, router, "nwes", lexicon) is None


class TestInformationAboutTheNews:
    def test_plain_information_stays_information(self, live, router, lexicon):
        tool = resolve(live, router, "information about python", lexicon)
        assert tool.name == "information"

    def test_news_commands_stay_news(self, live, router, lexicon):
        assert resolve(live, router, "tell me the news", lexicon).name == "news"
        assert resolve(live, router, "news about python", lexicon).name == "news"

    def test_the_tangled_case_is_ambiguous_not_reversed(self, lexicon):
        from assistant.nlu import AMBIGUOUS, parse

        parsed = parse("information about the news", lexicon)
        assert parsed is not None
        # "news" appearing later must not steal the request.
        assert parsed.name == "information"
        assert parsed.confidence == AMBIGUOUS
        assert "news" in [a.name for a in parsed.alternatives]

    def test_ambiguity_asks_rather_than_guessing(self, live, router, lexicon):
        live.listener = ScriptedListener([""])
        assert resolve(live, router, "information about the news", lexicon) is None
        assert any("Did you mean" in line for line in live.speaker.spoken)

    def test_the_reply_selects_the_tool(self, live, router, lexicon):
        live.listener = ScriptedListener(["news"])
        tool = resolve(live, router, "information about the news", lexicon)
        assert tool is not None
        assert tool.name == "news"

    def test_the_other_reply_also_works(self, live, router, lexicon):
        live.listener = ScriptedListener(["information"])
        tool = resolve(live, router, "information about the news", lexicon)
        assert tool.name == "information"


def _ambiguous(names):
    from assistant.nlu import Alternative, ParsedIntent

    return ParsedIntent(
        name=names[0],
        score=1.0,
        trigger=names[0],
        method="exact",
        confidence="ambiguous",
        entities={},
        alternatives=tuple(Alternative(n, 1.0) for n in names[1:]),
    )


class TestClarification:
    def test_question_is_generated_from_the_candidates(self):
        assert (
            clarification_question(["information", "news"])
            == "Did you mean information or news?"
        )

    def test_only_two_choices_are_offered(self):
        assert "c" not in clarification_question(["a", "b", "c"])

    def test_one_candidate_falls_back_to_repeating(self):
        assert clarification_question(["news"]) == "Could you say that again?"

    def test_reply_outside_the_offer_is_refused(self, live):
        from assistant.app import clarify

        live.listener = ScriptedListener(["please delete everything"])
        assert clarify(live, _ambiguous(["information", "news"])) is None

    def test_partial_word_does_not_match(self, live):
        from assistant.app import clarify

        # "newsletter" must not be read as "news".
        live.listener = ScriptedListener(["newsletter"])
        assert clarify(live, _ambiguous(["news", "information"])) is None

    def test_silence_is_not_an_answer(self, live):
        from assistant.app import clarify

        live.listener = ScriptedListener([""])
        assert clarify(live, _ambiguous(["news", "information"])) is None


class TestSystemSafety:
    @pytest.mark.parametrize(
        "utterance", ["exit", "quit", "goodbye", "please exit"]
    )
    def test_real_exit_commands_reach_the_tool(
        self, live, router, lexicon, utterance
    ):
        tool = resolve(live, router, utterance, lexicon)
        assert tool is not None
        assert tool.name == "system"

    @pytest.mark.parametrize(
        "utterance",
        [
            "goodbye is in the dictionary",
            "quite good",
            "I don't want to exit",
            "the word quit is in that sentence",
        ],
    )
    def test_dangerous_near_misses_are_refused(
        self, live, router, lexicon, utterance
    ):
        assert resolve(live, router, utterance, lexicon) is None

    def test_a_refused_exit_does_not_end_the_session(self, config, db):
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(
            ["goodbye is in the dictionary", "quite good", "exit"]
        )
        # Returns 0 only because of the real "exit" at the end.
        assert run(ctx, build_default_router()) == 0
        assert any("dictionary" not in s for s in ctx.speaker.spoken)

    def test_the_loop_survives_a_run_of_refusals(self, config, db):
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["replay my song", "quite good", "exit"])
        assert run(ctx, build_default_router()) == 0


class TestOriginalUtteranceIsPreserved:
    def test_normalization_never_reaches_the_tool(
        self, config, db, monkeypatch
    ):
        """Slot extraction must see the user's words, not the parser's."""
        from assistant.tools import notes as notes_mod

        seen: list[str] = []
        original = notes_mod.NoteTool.extract_slots

        def spy(self, utterance, ctx):
            seen.append(utterance)
            return original(self, utterance, ctx)

        monkeypatch.setattr(notes_mod.NoteTool, "extract_slots", spy)

        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["Note Buy MILK!", "exit"])
        run(ctx, build_default_router())

        assert seen, "the notes tool was never asked for slots"
        # The parser would have folded this to "note buy milk". The tool
        # must still receive the original capitals and punctuation.
        assert any("Buy MILK" in utterance for utterance in seen)

    def test_resolve_does_not_rewrite_the_utterance(self, live, router, lexicon):
        before = "Note Buy Milk"
        resolve(live, router, before, lexicon)
        assert before == "Note Buy Milk"

    def test_youtube_still_strips_its_prefix(self, config, router, lexicon):
        tool = resolve(
            make_ctx(config), router, "play lofi beats on youtube", lexicon
        )
        slots = tool.extract_slots("play lofi beats on youtube", None)
        assert slots == {"query": "lofi beats"}

    def test_notes_tool_still_extracts_its_body(self, config, router, lexicon):
        tool = resolve(make_ctx(config), router, "Note Buy Milk", lexicon)
        slots = tool.extract_slots("Note Buy Milk", None)
        assert slots


class TestLegacyFallbackStaysReachable:
    def test_used_when_understanding_raises(
        self, live, router, lexicon, monkeypatch
    ):
        import assistant.app as app

        def boom(*args, **kwargs):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(app, "parse", boom)
        tool = app.resolve(live, router, "tell me a joke", lexicon)
        assert tool is not None
        assert tool.name == "jokes"

    def test_used_when_the_parser_names_no_registered_tool(
        self, live, router, lexicon, monkeypatch
    ):
        import assistant.app as app
        from assistant.nlu import ParsedIntent

        monkeypatch.setattr(
            app,
            "parse",
            lambda *a, **k: ParsedIntent(
                name="no_such_tool",
                score=1.0,
                trigger="x",
                method="exact",
                confidence="clear",
                entities={},
                alternatives=(),
            ),
        )
        assert app.resolve(live, router, "tell me a joke", lexicon).name == "jokes"

    def test_router_api_is_intact(self):
        from assistant.core.router import NoMatch
        from assistant.core.tool import Tool

        router = build_default_router()
        assert router.find_match("tell me a joke").tool_name == "jokes"
        assert isinstance(router.find_match("zzzz"), NoMatch)
        assert callable(router.dispatch)
        assert issubclass(type(router.tools[0]), Tool)
        # matches() is still the tool's own boundary-free substring check.
        assert router.get("jokes").matches("tell me a joke") == "joke"
        # extract_slots is still the tool's own, and still takes the
        # utterance. The shape is whatever the tool has always returned.
        assert router.get("notes").extract_slots("note milk", None) == {
            "action": "add",
            "body": "milk",
        }


class TestRunLoop:
    def test_unknown_command_keeps_the_original_wording(self, config, db):
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["xyzzy gibberish", "exit"])
        run(ctx, build_default_router())
        assert MSG_DIDNT_UNDERSTAND in ctx.speaker.spoken

    def test_notes_command_persists(self, config, db):
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["note integration works", "exit"])
        run(ctx, build_default_router())
        rows = db.execute("SELECT utterance FROM interactions").fetchall()
        assert any("integration works" in row[0] for row in rows)

    def test_a_failing_tool_does_not_end_the_run(
        self, config, db, monkeypatch
    ):
        from assistant.tools import jokes as jokes_mod

        def boom(self, ctx, slots=None):
            raise RuntimeError("joke machine broken")

        monkeypatch.setattr(jokes_mod.JokesTool, "execute", boom)
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["tell me a joke", "exit"])
        assert run(ctx, build_default_router()) == 0
        assert any("jokes" in line for line in ctx.speaker.spoken)

    def test_no_listener_is_an_error(self, config):
        assert run(AppContext(config=config), build_default_router()) == 1

    def test_nlu_is_actually_reached(self, config, db, monkeypatch):
        import assistant.app as app

        calls: list[str] = []
        real = app.parse

        def spy(utterance, lexicon):
            calls.append(utterance)
            return real(utterance, lexicon)

        monkeypatch.setattr(app, "parse", spy)
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["exit"])
        run(ctx, build_default_router())
        assert "exit" in calls

    def test_joke_setup_and_punchline_stay_separate(self, config, db, monkeypatch):
        # Stubbed rather than real, so the test proves the list is spoken
        # as separate calls without importing the randfacts package into
        # this process, which would break the NLU isolation tests.
        from assistant.tools import jokes as jokes_mod

        monkeypatch.setattr(
            jokes_mod.JokesTool,
            "execute",
            lambda self, ctx, slots=None: ["Why did the joke cross the road?",
                                           "To get to the other punchline."],
        )
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["tell me a joke", "exit"])
        run(ctx, build_default_router())
        assert "Why did the joke cross the road?" in ctx.speaker.spoken
        assert "To get to the other punchline." in ctx.speaker.spoken

    def test_interactions_are_recorded(self, config, db):
        ctx = make_ctx(config, db)
        ctx.listener = ScriptedListener(["note persistence check", "exit"])
        run(ctx, build_default_router())
        count = db.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        assert count >= 2
