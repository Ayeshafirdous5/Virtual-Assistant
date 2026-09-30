"""Phase 5 Step 22 -- the want frame, ``you ... want to <trigger>``.

Step 21 measured ten utterances that still reached :class:`SystemTool`
and should not have. Eight of them shared one structure::

    do you want to quit          you want to exit
    do you want to exit          do you really want to quit
    do you want to shut down     do you want to goodbye
    do you want to bye

That is a question, or a remark, about whether the *assistant* should
stop. It carries the "want" of the "quit" token, and "you" as the one
pronoun the subject rule cannot refuse, because "you" is normally the
correct addressee for a command.

Step 22 closed exactly that structure, in
:func:`assistant.nlu.scoring._passes_system_guard`, and nothing else.
The two conversational cases were explicitly left for a later step, and
these tests fail if that boundary ever moves silently.

======  ==================================================================
Group   What it pins
======  ==================================================================
A       the eight preference questions, refused
B       fourteen genuine orders, still honoured
C       nearby "want" wording, still safe
D       the two conversational cases, still measured and still unsafe
E       the Step 17 subject family, still refused
F       the Step 20 definition family, still refused
======  ==================================================================
"""

from __future__ import annotations

import pytest

from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.nlu.scoring import (
    DEFINITION_MARKERS,
    SELF_OTHER_SUBJECTS,
    SYSTEM_MIN_SCORE,
    SYSTEM_TAIL_TOKENS,
    _is_want_frame_about_you,
)
from tests.test_nlu_system_preference_questions import (
    DANGEROUS_FAMILY,
    RECORDED_DANGEROUS,
    RECORDED_REMAINING_DANGERS,
    _lexicon,
)
from tests.nlu_system_preference_question_corpus import (
    LOAD_BEARING_COMMANDS,
    PREFERENCE_QUESTIONS,
)

#: The eight cases Step 22 was asked to block, quoted rather than derived
#: so a change cannot quietly shorten the list and pass by omission.
PREFERENCE_CASES: tuple[str, ...] = tuple(
    case.utterance
    for case in PREFERENCE_QUESTIONS
    if case.utterance in DANGEROUS_FAMILY
)

#: The fourteen orders from the step's own group B.
GENUINE_COMMANDS: tuple[str, ...] = (
    "exit",
    "quit",
    "goodbye",
    "bye",
    "please exit",
    "goodbye assistant",
    "shut down",
    "you can quit now",
    "can you exit",
    "would you quit now",
    "will you quit",
    "can you shut down",
    "would you exit now",
    "should you quit now",
)

#: Wording near the family that must not be caught with it.
NEARBY_SAFE: tuple[str, ...] = (
    "I want to quit smoking",
    "I want to exit early",
    "do you want me to quit",
    "I want information about Hyderabad",
    "I want a joke",
)

#: The Step 17 subject family, quoted from its own test module.
STEP17_FAMILY: tuple[str, ...] = (
    "I should quit smoking",
    "I want to quit smoking",
    "she decided to quit smoking",
    "he plans to quit smoking",
    "I need to quit smoking",
    "he quit smoking",
)

#: The Step 20 definition family, likewise.
STEP20_FAMILY: tuple[str, ...] = (
    "what does quit mean",
    "what does exit mean",
    "what does goodbye mean",
    "what is the meaning of quit",
    "define exit",
    "explain quit",
)

#: The polite orders, quoted because each is addressed to the assistant
#: and none of them contains a want frame.
POLITE_ORDERS: tuple[str, ...] = (
    "can you exit",
    "would you quit now",
    "will you quit",
    "can you shut down",
    "would you exit now",
    "should you quit now",
    "you can quit now",
    "you may exit",
)


def _context():
    """A context whose speaker and listener do nothing."""
    from assistant.core.context import AppContext

    class Silent:
        def speak(self, text: str) -> None:
            pass

        def listen(self) -> str:
            return ""

    ctx = AppContext()
    ctx.speaker = Silent()
    ctx.listener = Silent()
    return ctx


def runs_system_tool(utterance: str) -> bool:
    """Whether this utterance really reaches :class:`SystemTool`.

    Measured through the real resolver rather than the Step 21 corpus,
    so it works for any utterance. Using the corpus here would tie every
    assertion to the case happening to be in the corpus, and a missing
    case would surface as ``StopIteration`` rather than as a failed
    expectation.
    """
    from assistant.app import resolve_detail
    from assistant.tools import build_default_router
    from assistant.tools.system import SystemTool

    resolution = resolve_detail(
        _context(), build_default_router(), utterance, _lexicon()
    )
    return isinstance(resolution.tool, SystemTool)


def _trigger_start(tokens: list[str]) -> int:
    """The index of the **start** of the matched system trigger.

    Measured on "shut" rather than "down", because that is what the guard
    sees and what a two-word trigger has to be judged from.
    """
    for word in ("quit", "exit", "goodbye", "bye", "shut"):
        if word in tokens:
            return tokens.index(word)
    raise AssertionError(f"no system trigger in {tokens!r}")


def test_the_group_has_the_eight_cases_the_step_named():
    assert len(PREFERENCE_CASES) == 8
    assert set(PREFERENCE_CASES) == set(DANGEROUS_FAMILY) - set(
        RECORDED_REMAINING_DANGERS
    )


# --------------------------------------------------------------------------
# A. The eight preference-frame cases
# --------------------------------------------------------------------------
class TestGroupAThePreferenceQuestionsAreRefused:
    @pytest.mark.parametrize("utterance", PREFERENCE_CASES)
    def test_no_longer_executes(self, utterance):
        assert not runs_system_tool(utterance), utterance

    @pytest.mark.parametrize("utterance", PREFERENCE_CASES)
    def test_is_not_a_system_command(self, utterance):
        """It must not even be classified as one."""
        parsed = parse(utterance, _lexicon())
        assert parsed is None or parsed.name != "system", utterance

    @pytest.mark.parametrize("utterance", PREFERENCE_CASES)
    def test_is_caught_by_the_want_frame_condition(self, utterance):
        """Each case is refused by this condition, not by some other one.

        Worth pinning one by one: a future change that blocked these for
        an unrelated reason would hide a regression in the rule that
        exists to keep "you can quit now" working.
        """
        tokens = normalize(utterance).tokens
        start = _trigger_start(tokens)
        assert _is_want_frame_about_you(tokens, start), utterance

    def test_the_phrase_trigger_is_caught_from_its_start(self):
        """``index`` is the start of the trigger, not its last token.

        "shut down" ends in "down", whose preceding token is "shut". A
        condition reading the final token would inspect the wrong word
        entirely, so the start index is the only reading that agrees
        with how the guard is called.
        """
        tokens = normalize("do you want to shut down").tokens
        start = tokens.index("shut")
        assert tokens[start - 1] == "to"
        assert tokens[start + 1] == "down"
        assert _is_want_frame_about_you(tokens, start)
        # Shifted by one, the condition inspects "shut" and finds a
        # different neighbour, so the two readings really do differ.
        assert tokens[start + 1] != "to"
        assert not _is_want_frame_about_you(tokens, start + 1)


# --------------------------------------------------------------------------
# B. Genuine orders
# --------------------------------------------------------------------------
class TestGroupBGenuineOrdersStillRun:
    @pytest.mark.parametrize("utterance", GENUINE_COMMANDS)
    def test_still_executes(self, utterance):
        assert runs_system_tool(utterance), utterance

    @pytest.mark.parametrize("utterance", LOAD_BEARING_COMMANDS)
    def test_every_load_bearing_command_still_runs(self, utterance):
        """The Step 21 list, which a fix must not shorten to pass."""
        assert runs_system_tool(utterance), utterance

    @pytest.mark.parametrize("utterance", POLITE_ORDERS)
    def test_polite_address_to_the_assistant_is_untouched(self, utterance):
        """"you" as addressee must not be refused on its own.

        Each is a real order and none contains a want frame, which is
        precisely why they survive: the condition needs both marks, and
        either one alone is refused as evidence.
        """
        assert runs_system_tool(utterance), utterance

    def test_a_want_word_without_the_frame_is_still_a_command(self):
        """``you want quit`` runs, because the frame needs its "to".

        Pinned because a future simplification to bare "want" would pass
        every group A test while breaking this one.
        """
        assert runs_system_tool("you want quit")


# --------------------------------------------------------------------------
# C. Nearby wording that must not be caught
# --------------------------------------------------------------------------
class TestGroupCNearbyWantWordingIsSafe:
    @pytest.mark.parametrize("utterance", NEARBY_SAFE)
    def test_does_not_execute(self, utterance):
        assert not runs_system_tool(utterance), utterance

    @pytest.mark.parametrize(
        "utterance", ["I want information about Hyderabad", "I want a joke"]
    )
    def test_non_system_intents_are_untouched(self, utterance):
        """These are not merely refused, they are still answered.

        The guard is scoped to the system intent, so an unrelated request
        must still reach its own tool rather than fall through.
        """
        expected = "information" if "information" in utterance else "jokes"
        assert parse(utterance, _lexicon()).name == expected, utterance

    def test_the_first_person_cases_belong_to_the_subject_rule(self):
        """"I want to quit" is refused before the want frame is reached.

        The new condition never sees these, and the assertion proves it,
        so a future removal of the subject rule cannot be masked by the
        want frame quietly standing in for it.
        """
        tokens = normalize("I want to quit smoking").tokens
        assert not _is_want_frame_about_you(tokens, tokens.index("quit"))

    def test_you_alone_is_never_enough(self):
        tokens = normalize("can you exit").tokens
        assert not _is_want_frame_about_you(tokens, tokens.index("exit"))

    def test_a_want_frame_without_you_is_never_enough(self):
        tokens = normalize("I want to quit").tokens
        assert not _is_want_frame_about_you(tokens, tokens.index("quit"))


# --------------------------------------------------------------------------
# D. The two conversational cases, still measured and still unsafe
# --------------------------------------------------------------------------
class TestGroupDTheRemainingTwoAreStillMeasured:
    @pytest.mark.parametrize("utterance", RECORDED_REMAINING_DANGERS)
    def test_still_executes_and_is_still_a_defect(self, utterance):
        """Step 22 was told not to fix these, and did not.

        This test **records a known fault**. It is expected to fail the
        day a later step fixes it, and that failure is the signal to
        move :data:`RECORDED_DANGEROUS` deliberately rather than
        silently.
        """
        assert runs_system_tool(utterance), utterance

    def test_the_boundary_between_this_step_and_the_next_has_not_moved(self):
        assert len(DANGEROUS_FAMILY) == 10
        assert len(PREFERENCE_CASES) == 8
        assert set(DANGEROUS_FAMILY) - set(PREFERENCE_CASES) == set(
            RECORDED_REMAINING_DANGERS
        )

    def test_the_step21_count_moved_by_exactly_the_eight(self):
        assert RECORDED_DANGEROUS == 2


# --------------------------------------------------------------------------
# E and F. The earlier families must not come back
# --------------------------------------------------------------------------
class TestGroupEStep17SubjectFamily:
    @pytest.mark.parametrize("utterance", STEP17_FAMILY)
    def test_still_refused(self, utterance):
        assert not runs_system_tool(utterance), utterance


class TestGroupFStep20DefinitionFamily:
    @pytest.mark.parametrize("utterance", STEP20_FAMILY)
    def test_still_refused(self, utterance):
        assert not runs_system_tool(utterance), utterance


# --------------------------------------------------------------------------
# What this change was not allowed to touch
# --------------------------------------------------------------------------
def _mentions_want_frame(module_name: str) -> bool:
    """Whether a module has been given a want-frame concept of its own."""
    import importlib
    import inspect

    from assistant.nlu import scoring

    module = importlib.import_module(f"assistant.nlu.{module_name}")
    name = scoring._is_want_frame_about_you.__name__
    return name in inspect.getsource(module)


class TestNothingElseWasChanged:
    def test_you_is_not_in_the_subject_list(self):
        """Adding it would refuse every polite order in group B."""
        assert "you" not in SELF_OTHER_SUBJECTS
        for word in ("i", "me", "we", "they", "she", "he"):
            assert word in SELF_OTHER_SUBJECTS, word

    def test_to_was_not_added_as_a_safety_marker(self):
        """A bare "to" rule would break "time to quit" and "ready to quit"."""
        assert "to" not in DEFINITION_MARKERS
        for utterance in ("time to quit", "ready to quit", "try to quit"):
            assert parse(utterance, _lexicon()) is not None, utterance

    def test_thresholds_and_tail_are_unchanged(self):
        assert SYSTEM_MIN_SCORE == 0.95
        assert SYSTEM_TAIL_TOKENS == 2

    def test_the_definition_markers_are_unchanged(self):
        assert DEFINITION_MARKERS == frozenset(
            {"does", "do", "of", "define", "explain"}
        )

    @pytest.mark.parametrize("module", ["parser", "lexicon", "framing"])
    def test_the_guard_is_the_only_layer_that_moved(self, module):
        """The fix lives in the scorer and nowhere else.

        The parser has no want-frame concept and must not grow one, or
        the narrow owner this step chose would quietly stop being the
        only one.
        """
        assert not _mentions_want_frame(module), module


# --------------------------------------------------------------------------
# The one known cost, pinned so it cannot be forgotten
# --------------------------------------------------------------------------
class TestKnownCost:
    def test_a_conditional_wrapper_is_refused(self):
        """"if you want to quit now" is a real order, and it is refused.

        The condition cannot tell a conditional wrapper from a question
        without parsing the clause, and clause parsing is out of scope
        for a scorer. Refusing costs the user one retry; allowing it
        would cost a wrong answer, so the asymmetry is deliberate. Pinned
        so the trade stays visible rather than latent.
        """
        assert not runs_system_tool("if you want to quit now")
        assert runs_system_tool("you want quit")
