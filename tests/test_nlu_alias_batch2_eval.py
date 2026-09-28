"""Step 14: the evaluation of the remaining alias candidates, as tests.

**No production code changed in this step.** Every candidate failed the
evaluation, so the correct change is to record why, not to add a
vocabulary entry that would misroute real sentences.

What was measured
-----------------
Twelve alias-shaped gaps were left after Batch 1. Each was tested with
its intended intent, at least three natural variants, at least two
nearby non-command sentences, and at least one command that could
collide. Eighteen distinct alias forms were tried across three rounds,
including reformulations aimed at the same gaps. Seventeen were
rejected and one was found redundant:

============================  =========  ==========================
form tried                    intent     why it was rejected
============================  =========  ==========================
"something funny"             jokes      *"it was something funny"*
"amuse me"                    jokes      *"he will amuse me later"*
"celsius"                     weather    *"it was 30 celsius"*
"degrees outside"             weather    *"he has degrees outside"*
"wind speed"                  weather    *"the wind speed was high"*
"gossip"                      news       *"she spread gossip"*
"trivia"                      facts      *"trivia night was fun"*
"did you know"                facts      *"did you know she left"*
"google"                      info       *"he works at google"*
"define"                      info       *"hard to define"*
"search the web"              info       *"I will search the web"*
"power off"                   system     **six** sentence endings
"how warm is it"              weather    *"he asked how warm is it"*
"how many degrees"            weather    *"he has how many degrees"*
"is it warm"                  weather    *"he said is it warm"*
"who is"                      info       *"who is in the kitchen"*
"what is"                     info       *"what is the time"*
"temperature outside"         weather    redundant, see :data:`REDUNDANT`
============================  =========  ==========================

The one that mattered most
---------------------------
``"power off"`` looked clean on the first attempt. It was **not**. The
``system`` intent is held to a higher bar and the trigger must sit in
the closing words of an utterance, which protected it from *leading*
mentions and hid six failures behind a three-sentence sample:

    "my phone will power off"      -> would have quit the assistant
    "it will power off"            -> would have quit the assistant
    "the heater will power off"    -> would have quit the assistant
    "the device will power off"    -> would have quit the assistant
    "it should power off"          -> would have quit the assistant
    "he said it will power off"    -> would have quit the assistant

"Power off" is naturally a *trailing* phrase in English, which is exactly
where the system guard is most permissive. This is why the negatives for
every candidate here are deliberately more numerous than the brief asked
for.

What these tests do
-------------------
They **pin the rejections**. An alias that is not in the table cannot
silently arrive in a later step, and the false-positive sentences are
asserted to fall through *today*, so a future change that introduces one
shows up as a failure with the sentence that caused it.
"""

from __future__ import annotations

import pytest

from assistant.nlu.lexicon import COMMON_ALIASES
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.nlu.scoring import score_intents

#: Every alias form evaluated in Step 14, with the intent it was tried
#: against and the sentence that ruled it out.
REJECTED: tuple[tuple[str, str, str], ...] = (
    ("something funny", "jokes", "it was something funny"),
    ("amuse me", "jokes", "he will amuse me later"),
    ("celsius", "weather", "it was 30 celsius yesterday"),
    ("degrees outside", "weather", "he has degrees outside law"),
    ("wind speed", "weather", "the wind speed was high"),
    ("gossip", "news", "she spread gossip"),
    ("trivia", "facts", "that is a fun trivia game"),
    ("did you know", "facts", "did you know she left"),
    ("google", "information", "he works at google"),
    ("define", "information", "the problem is hard to define"),
    ("search the web", "information", "I will search the web later"),
    ("power off", "system", "my phone will power off"),
    # Reformulations of the same gaps, tried in the second round.
    ("how warm is it", "weather", "he asked how warm is it"),
    ("how many degrees", "weather", "he has how many degrees"),
    ("is it warm", "weather", "he said is it warm"),
    ("who is", "information", "who is in the kitchen"),
    ("what is", "information", "what is the time"),
)

#: One reformulation turned out to be **redundant** rather than unsafe,
#: which is a different verdict and worth keeping separate. Every sentence
#: containing "temperature outside" already reaches the weather tool
#: through its own single-word pattern, so the alias could add capability
#: and could not add a false positive. It is recorded as unnecessary
#: rather than rejected.
REDUNDANT: tuple[tuple[str, str], ...] = (("temperature outside", "weather"),)

#: The six sentence endings that made "power off" unsafe, kept separate
#: because this is the finding that matters most in the whole step.
POWER_OFF_ENDINGS: tuple[str, ...] = (
    "my phone will power off",
    "it will power off",
    "the heater will power off",
    "the device will power off",
    "it should power off",
    "he said it will power off",
)

#: The twelve gaps Batch 1 left behind, mapped to the intent each names.
REMAINING_GAPS: tuple[tuple[str, str], ...] = (
    ("something funny", "jokes"),
    ("amuse me", "jokes"),
    ("power off", "system"),
    ("celsius", "weather"),
    ("degrees outside", "weather"),
    ("wind speed", "weather"),
    ("gossip", "news"),
    ("trivia", "facts"),
    ("did you know", "facts"),
    ("google python", "information"),
    ("define gravity", "information"),
    ("search the web for python", "information"),
)


@pytest.fixture(scope="module")
def lexicon():
    from assistant.app import build_runtime_lexicon
    from assistant.tools import build_default_router

    return build_runtime_lexicon(build_default_router())


def boundary_hits(utterance: str, lexicon) -> list[tuple[str, str, float]]:
    """Every intent that reached the action boundary, at any stage."""
    return [
        (c.intent, c.method, c.score)
        for c in score_intents(normalize(utterance), lexicon)
        if c.method in ("exact", "phrase") and c.score >= 0.65
    ]


class TestBatchTwoIsEmpty:
    """The finding, stated as an assertion so it cannot drift."""

    def test_no_alias_was_added_in_step_fourteen(self):
        """The table is exactly what Batch 1 left."""
        assert COMMON_ALIASES["weather"] == (
            "forecast",
            "how hot",
            "how cold",
            "will it rain",
            "is it going to rain",
        )
        assert COMMON_ALIASES["jokes"] == ("make me laugh",)
        assert COMMON_ALIASES["news"] == ("current affairs", "what happened today")
        assert COMMON_ALIASES["information"] == (
            "look up",
            "search for",
            "tell me about",
        )
        assert COMMON_ALIASES["system"] == (
            "stop the assistant",
            "stop listening",
            "shut down",
            "that is all",
        )

    @pytest.mark.parametrize("alias,intent,why", REJECTED)
    def test_the_rejected_alias_is_absent(self, alias, intent, why):
        assert alias not in COMMON_ALIASES.get(intent, ()), alias

    def test_every_form_tried_was_distinct(self):
        assert len(REJECTED) == 17
        assert len({alias for alias, _, _ in REJECTED}) == 17
        assert len(REMAINING_GAPS) == 12


class TestRejectionsStillHold:
    """Each false-positive sentence must fall through today.

    If one of these ever starts matching, the alias that caused it has
    arrived, and the failure names the sentence rather than a count.
    """

    @pytest.mark.parametrize("alias,intent,utterance", REJECTED)
    def test_the_reasoning_sentence_still_falls_through(
        self, alias, intent, utterance, lexicon
    ):
        matched = boundary_hits(utterance, lexicon)
        assert not any(intent == m[0] for m in matched), (utterance, matched)

    @pytest.mark.parametrize("alias,intent", REDUNDANT)
    def test_a_redundant_alias_would_add_nothing(self, alias, intent, lexicon):
        """Every sentence holding the phrase already reaches the tool.

        This is why it is not in the rejected list: it cannot misroute
        anything, and it cannot help either, because the tool's own
        single-word pattern already fires first.
        """
        for utterance in (
            "he felt the temperature outside",
            "the temperature outside is nice",
        ):
            parsed = parse(utterance, lexicon)
            assert parsed is not None, utterance
            assert parsed.name == intent, utterance
            assert parsed.trigger == "temperature", parsed.trigger
        assert alias not in COMMON_ALIASES[intent]

    @pytest.mark.parametrize("utterance", POWER_OFF_ENDINGS)
    def test_a_device_mention_does_not_quit_the_assistant(
        self, utterance, lexicon
    ):
        """The finding that made "power off" a rejection.

        The system guard requires the trigger in the closing words, so
        these are exactly the sentences it lets through, and exactly why
        a trailing phrase is the wrong shape for a terminating tool.
        """
        matched = boundary_hits(utterance, lexicon)
        assert not any("system" == m[0] for m in matched), (utterance, matched)
        assert parse(utterance, lexicon) is None, utterance

    @pytest.mark.parametrize("utterance", POWER_OFF_ENDINGS)
    def test_no_device_mention_reaches_a_tool(self, utterance, lexicon):
        """End to end: none of these may terminate the application."""
        from assistant.app import resolve_detail
        from assistant.core.context import AppContext
        from assistant.tools import build_default_router

        class Silent:
            def speak(self, text: str) -> None:
                pass

            def listen(self) -> str:
                return ""

        ctx = AppContext()
        ctx.speaker = Silent()
        ctx.listener = Silent()
        resolution = resolve_detail(ctx, build_default_router(), utterance, lexicon)
        assert resolution.tool is None, utterance
        assert resolution.status != "resolved", utterance


class TestTheGapsAreStillOpen:
    """The honest position: these remain known gaps, deliberately.

    Recorded so the count of unfinished work stays visible rather than
    quietly becoming zero. Closing any of them needs a different layer,
    not a longer alias list.
    """

    @pytest.mark.parametrize("utterance,intent", REMAINING_GAPS)
    def test_a_rejected_gap_is_still_a_gap(self, utterance, intent, lexicon):
        assert parse(utterance, lexicon) is None, utterance

    def test_the_lexer_reached_its_limit(self):
        """The generalisation this step reached, stated once.

        Every remaining gap is a case where the natural wording also
        occurs in a sentence that is not a command. Adding the wording
        would trade a missed request for a wrong action, and the project's
        own documentation rejected "song" and "stop" for exactly this.
        """
        assert len(REJECTED) == 17
        assert not any(
            alias in COMMON_ALIASES.get(intent, ())
            for alias, intent, _ in REJECTED
        )

