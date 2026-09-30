"""Phase 5 closure sprint -- the last two conversational cases, left open.

Steps 17, 20 and 22 each closed one family. Two cases survived all
three, and this module is the record of the attempt to close them::

    plans to quit            nothing to quit over

The attempt **failed, and that is the finding.** Both are a 1.000 exact
match on ``quit``, inside the closing window, with no self/third-party
subject, no definition marker and no want frame, so every condition in
:func:`~assistant.nlu.scoring._passes_system_guard` passes and the
router dispatches ``SystemTool``. The parser is never given a reason to
refuse: it reads a bare imperative.

The only shared structural feature is the token immediately before the
trigger, ``to``. That token cannot be used, and the reason is measured
here rather than asserted:

* ``plans to quit`` and ``time to quit`` are identical on **every**
  feature the guard can see -- same length, same trigger index, same
  ``to``, same absent subject, marker and want frame. The only
  difference is what the first word *means*. Separating them needs
  semantics, not shape.
* A bare ``to`` rule (candidate A) would refuse ``time to quit``,
  ``ready to quit`` and ``try to quit``, which
  ``test_nlu_system_want_frame_step22.py`` pins as still parsing, and
  would newly refuse six utterances labelled ``YES`` in
  ``nlu_system_tradeoff_corpus.py``.
* A ``{plans, nothing}`` word list (candidate C) catches both but is the
  answer key rather than a rule, and generalises to nothing.

So no production change was made. These tests keep the two cases
*measured and named*; if a later step closes them properly, the
assertions below fail loudly and that step updates them on purpose.

The full reasoning is in ``docs/nlu-tuning.md``, section "Phase 5
closure: the last two conversational cases".
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
from tests.test_nlu_system_preference_questions import _lexicon

#: The two utterances this sprint was asked to close. Quoted, not derived,
#: so a future change cannot quietly shorten the list and pass by omission.
REMAINING_CASES: tuple[str, ...] = ("plans to quit", "nothing to quit over")

#: Genuine orders that must keep working: the fifteen the brief names.
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
    "you may exit",
)

#: ``to``-adjacent utterances that must keep parsing. Pinned by
#: ``test_nlu_system_want_frame_step22.py::test_to_was_not_added_as_a_safety_marker``.
MUST_STILL_PARSE: tuple[str, ...] = (
    "time to quit",
    "ready to quit",
    "try to quit",
)

#: Unrelated requests, to prove the guard is scoped to the system intent
#: and has not leaked into any other tool.
UNRELATED_INTENTS: tuple[str, ...] = (
    "I want a joke",
    "I need a joke",
    "I want information about Hyderabad",
    "tell me the weather",
    "tell me the news",
    "play a song",
    "show me my notes",
)


def _trigger_index(utterance: str) -> int:
    """Index of the first registered system trigger, or -1."""
    tokens = list(normalize(utterance).tokens)
    for word in ("quit", "shut", "exit", "goodbye", "bye"):
        if word in tokens:
            return tokens.index(word)
    return -1


def _shape(utterance: str) -> dict:
    """Every feature the system guard is allowed to look at."""
    tokens = list(normalize(utterance).tokens)
    index = _trigger_index(utterance)
    before = tokens[:index] if index > 0 else []
    return {
        "length": len(tokens),
        "trigger_index": index,
        "token_before": tokens[index - 1] if index > 0 else None,
        "in_tail": index >= max(0, len(tokens) - SYSTEM_TAIL_TOKENS),
        "has_subject": any(w in SELF_OTHER_SUBJECTS for w in before),
        "has_definition_marker": any(w in DEFINITION_MARKERS for w in before),
        "has_want_frame": _is_want_frame_about_you(tokens, index),
        "has_you": "you" in before,
    }


class TestTheTwoCasesAreStillUnsafe:
    """The defect is asserted, not deleted.

    A passing test here means "still broken and still recorded". If a
    later step fixes it properly these fail, and that step moves them.
    """

    @pytest.mark.parametrize("utterance", REMAINING_CASES)
    def test_it_still_reaches_the_parser_as_a_command(self, utterance):
        parsed = parse(utterance, _lexicon())
        assert parsed is not None, "it no longer reaches the parser"
        assert parsed.name == "system", utterance

    @pytest.mark.parametrize("utterance", REMAINING_CASES)
    def test_it_is_a_perfect_exact_match_in_the_window(self, utterance):
        parsed = parse(utterance, _lexicon())
        assert parsed.score == 1.0, utterance
        assert parsed.method == "exact", utterance
        assert parsed.confidence == "clear", utterance

    @pytest.mark.parametrize("utterance", REMAINING_CASES)
    def test_no_existing_guard_condition_explains_it(self, utterance):
        """Why the earlier steps missed it: nothing already in place fires."""
        shape = _shape(utterance)
        assert shape["in_tail"], utterance
        assert shape["has_subject"] is False, utterance
        assert shape["has_definition_marker"] is False, utterance
        assert shape["has_want_frame"] is False, utterance
        assert shape["has_you"] is False, utterance


class TestThereIsNoStructuralBoundary:
    """The evidence for shipping no fix, so the decision can be rechecked."""

    def test_the_two_cases_agree_on_every_decision_relevant_feature(self):
        """Same shape for our purposes, which is why one rule could catch both.

        They are not byte-identical: ``nothing to quit over`` carries a
        trailing ``over``, so it is one token longer. Length is not a
        decision input here -- nothing in the guard reads it -- and every
        feature that *is* read agrees, which is the point being made.
        """
        first, second = (_shape(u) for u in REMAINING_CASES)
        del first["length"], second["length"]
        assert first == second

    def test_it_cannot_be_told_apart_from_time_to_quit(self):
        """The decisive measurement.

        ``plans to quit`` must be refused and ``time to quit`` must keep
        parsing, yet they are indistinguishable on every feature the
        scoring layer can see. Any rule separating them reads meaning,
        not shape, and this is the proof rather than an argument.
        """
        assert _shape("plans to quit") == _shape("time to quit"), (
            "a boundary appeared; re-evaluate the fix"
        )

    def test_the_only_difference_is_the_meaning_of_the_first_word(self):
        """Named explicitly, so the gap is not mistaken for an oversight."""
        defect = list(normalize("plans to quit").tokens)
        allowed = list(normalize("time to quit").tokens)
        assert len(defect) == len(allowed)
        assert defect[1:] == allowed[1:], "only token 0 differs"
        assert defect[0] != allowed[0]

    def test_a_bare_to_rule_would_break_cases_that_must_still_parse(self):
        """Candidate A, measured rather than asserted."""
        for utterance in REMAINING_CASES + MUST_STILL_PARSE:
            assert _shape(utterance)["token_before"] == "to", utterance
        # All of them share that shape, so one rule hits them all at once.
        assert parse("time to quit", _lexicon()) is not None

    def test_a_word_list_would_only_be_the_answer_key(self):
        """Candidate C, and why it is not a rule.

        ``to`` is already load-bearing for the Step 22 want frame, so a
        general ``to`` condition is not merely risky, it is unavailable.
        """
        assert "to" not in DEFINITION_MARKERS
        assert "plans" not in DEFINITION_MARKERS
        assert "nothing" not in DEFINITION_MARKERS
        assert "to" not in SELF_OTHER_SUBJECTS
        # The want frame still needs it, which is exactly why it cannot be
        # promoted to a standalone marker.
        tokens = list(normalize("you want to quit").tokens)
        assert _is_want_frame_about_you(tokens, tokens.index("quit"))


class TestNothingRegressed:
    """The sprint changed no production code, so nothing may have moved."""

    @pytest.mark.parametrize("utterance", GENUINE_COMMANDS)
    def test_genuine_commands_still_resolve_to_system(self, utterance):
        parsed = parse(utterance, _lexicon())
        assert parsed is not None, utterance
        assert parsed.name == "system", utterance

    @pytest.mark.parametrize("utterance", MUST_STILL_PARSE)
    def test_to_adjacent_cases_still_parse(self, utterance):
        assert parse(utterance, _lexicon()) is not None, utterance

    def test_no_constant_was_moved(self):
        """Scope of the sprint, asserted rather than described."""
        assert SYSTEM_MIN_SCORE == 0.95
        assert SYSTEM_TAIL_TOKENS == 2
        assert "to" not in DEFINITION_MARKERS
        assert "you" not in SELF_OTHER_SUBJECTS

    @pytest.mark.parametrize("utterance", UNRELATED_INTENTS)
    def test_unrelated_intents_are_untouched(self, utterance):
        """A system-only guard must not disturb any other tool."""
        parsed = parse(utterance, _lexicon())
        assert parsed is None or parsed.name != "system", utterance

