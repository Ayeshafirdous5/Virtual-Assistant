"""Measurement of the cost the Step 17 system-safety boundary imposes.

Step 17 made the system guard refuse any trigger preceded by a first- or
third-person subject. That stopped seven sentences from terminating the
assistant, and it also stopped some sentences the user meant as an exit
request. This module measures the second effect without changing anything.

Severity
--------
``FALSE_EXECUTION`` is the severe outcome: the user said something
ordinary and the assistant quit. ``FALSE_BLOCK`` is the mild one: the user
asked to leave, was not understood, and can simply say "exit" instead.
Every ordinary sentence in group E is therefore asserted safe, and no
label is ever weakened to make a number look better.

Undecided cases
---------------
Some sentences genuinely do not say whether the user wants advice or wants
an exit. *"can I quit"* and *"I should quit smoking"* share a subject and
a trigger word, and only the object distinguishes them. Those cases carry
``expected == undecided`` and are reported separately, never scored.
"""

from __future__ import annotations

import pytest

from assistant.app import build_runtime_lexicon, resolve_detail
from assistant.core.context import AppContext
from assistant.core.router import NoMatch
from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.nlu.scoring import score_intents
from assistant.tools import build_default_router
from tests.nlu_system_tradeoff_corpus import (
    CATEGORIES,
    CORPUS,
    NO,
    UNDECIDED,
    YES,
    TradeoffCase,
    by_category,
)

# Result classes.
CORRECT_COMMAND = "CORRECT_COMMAND"
CORRECT_BLOCK = "CORRECT_BLOCK"
FALSE_BLOCK = "FALSE_BLOCK"
FALSE_EXECUTION = "FALSE_EXECUTION"
AMBIGUOUS = "AMBIGUOUS"
UNSUPPORTED_BUT_SAFE = "UNSUPPORTED_BUT_SAFE"


class Silent:
    def speak(self, text: str) -> None:
        pass

    def listen(self) -> str:
        return ""


@pytest.fixture
def router():
    return build_default_router()


@pytest.fixture
def lexicon(router):
    return build_runtime_lexicon(router)


class Result:
    """The full pipeline trace and the resulting classification."""

    def __init__(self, case: TradeoffCase, router, lexicon) -> None:
        self.case = case
        normalized = normalize(case.utterance)
        candidates = score_intents(normalized, lexicon)
        parsed = parse(case.utterance, lexicon)

        ctx = AppContext()
        ctx.speaker = Silent()
        ctx.listener = Silent()
        resolution = resolve_detail(ctx, router, case.utterance, lexicon)
        match = router.find_match(case.utterance)

        self.normalized = normalized.text
        self.tokens = normalized.tokens
        self.candidates = [(c.intent, c.trigger, round(c.score, 3), c.method)
                           for c in candidates]
        self.parsed_name = parsed.name if parsed else None
        self.score = round(parsed.score, 3) if parsed else 0.0
        self.confidence = parsed.confidence if parsed else "-"
        self.framing = resolution.framing or "-"
        self.status = resolution.status
        self.tool = resolution.tool_name or "none"
        self.executes = resolution.tool_name == "system"
        self.legacy = "" if isinstance(match, NoMatch) else match.tool_name
        # Why did nothing run? Either the guard removed a candidate that the
        # lexicon clearly recognises, or the sentence never named a trigger.
        # The two are different problems, so they are recorded apart.
        self.guard_blocked = self._guard_removed_candidate(normalized, lexicon)

    def _guard_removed_candidate(self, normalized, lexicon) -> bool:
        if self.parsed_name is not None:
            return False
        return any(
            entry.regex.search(normalized.text)
            for entry in lexicon.entries_for("system")
        )

    @property
    def classification(self) -> str:
        """Score the observed behaviour against what the user meant."""
        if self.case.expected == UNDECIDED:
            return AMBIGUOUS
        if self.executes:
            return CORRECT_COMMAND if self.case.expected == YES else FALSE_EXECUTION
        if self.case.expected == NO:
            return CORRECT_BLOCK
        # Expected a command, none ran.
        if self.parsed_name is None:
            return UNSUPPORTED_BUT_SAFE
        return FALSE_BLOCK

    def row(self) -> str:
        return (
            f"  {self.case.utterance!r:<34} {self.case.expected:<9} "
            f"parsed={str(self.parsed_name):<8} {self.framing:<8} "
            f"status={self.status:<9} tool={self.tool:<7} "
            f"exec={str(self.executes):<5} {self.classification}"
        )


def results(router, lexicon) -> list[Result]:
    return [Result(case, router, lexicon) for case in CORPUS]


def by_class(items: list[Result], name: str) -> list[Result]:
    return [r for r in items if r.classification == name]


#: Totals recorded when this corpus was first measured. Pinned so a change
#: to the guard has to be reflected here deliberately.
RECORDED_CORRECT_COMMAND = 16
RECORDED_CORRECT_BLOCK = 11
RECORDED_FALSE_BLOCK = 0
RECORDED_FALSE_EXECUTION = 0
RECORDED_AMBIGUOUS = 7
RECORDED_UNSUPPORTED_BUT_SAFE = 12
#: Of those 12, how many the guard removed versus never matched a trigger.
RECORDED_GUARD_BLOCKED = 11
RECORDED_NEVER_A_TRIGGER = 1


def _full_report(router, lexicon) -> str:
    items = results(router, lexicon)
    lines = ["system command trade-off (Step 18)", ""]

    lines.append("  totals:")
    for name in (CORRECT_COMMAND, CORRECT_BLOCK, FALSE_BLOCK,
                 FALSE_EXECUTION, AMBIGUOUS, UNSUPPORTED_BUT_SAFE):
        lines.append(f"    {name:<24} {len(by_class(items, name))}")

    lines.append("")
    lines.append("  by category (executed / scored):")
    for category in CATEGORIES:
        subset = by_category(category)
        scored = [c for c in subset if c.expected != UNDECIDED]
        executed = sum(1 for c in scored if c.utterance and _executes(c, router, lexicon))
        lines.append(
            f"    {category:<30} {executed:>2} executed / {len(scored):>2} scored"
        )

    lines.append("")
    lines.append("  per case:")
    for r in items:
        lines.append(r.row())

    for name in (FALSE_EXECUTION, FALSE_BLOCK):
        group = by_class(items, name)
        if group:
            lines.append("")
            lines.append(f"  {name} ({len(group)}):")
            for r in group:
                lines.append(
                    f"    - {r.case.utterance!r} [{r.case.category}] {r.case.reason}"
                )
    return "\n".join(lines)


def _executes(case: TradeoffCase, router, lexicon) -> bool:
    return Result(case, router, lexicon).executes


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_size(self):
        assert len(CORPUS) >= 40

    def test_no_duplicates(self):
        texts = [c.utterance for c in CORPUS]
        assert len(texts) == len(set(texts))

    def test_every_category_present(self):
        assert {c.category for c in CORPUS} == set(CATEGORIES)

    def test_every_case_explained(self):
        for case in CORPUS:
            assert case.reason, case.utterance
            assert case.form, case.utterance
            assert case.expected in (YES, NO, UNDECIDED)

    def test_all_seven_step_17_dangers_are_present_and_labelled_no(self):
        dangers = [
            "I should quit smoking", "I want to quit smoking",
            "she decided to quit smoking", "he plans to quit smoking",
            "I need to quit smoking", "he quit smoking", "I want to exit early",
        ]
        by_text = {c.utterance: c for c in CORPUS}
        for danger in dangers:
            assert danger in by_text, danger
            assert by_text[danger].expected == NO, danger


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------
class TestMeasurement:
    def test_no_ordinary_sentence_can_terminate_the_assistant(self, router, lexicon):
        """The severe failure. Any failure here is a FALSE_EXECUTION."""
        ordinary = [c for c in CORPUS if c.expected == NO]
        bad = [r for r in results(router, lexicon)
               if r.case in ordinary and r.executes]
        assert not bad, [r.case.utterance for r in bad]

    def test_the_seven_step_17_dangers_stay_blocked(self, router, lexicon):
        dangers = {
            "I should quit smoking", "I want to quit smoking",
            "she decided to quit smoking", "he plans to quit smoking",
            "I need to quit smoking", "he quit smoking", "I want to exit early",
        }
        for r in results(router, lexicon):
            if r.case.utterance in dangers:
                assert not r.executes, r.case.utterance

    def test_every_direct_command_still_works(self, router, lexicon):
        for case in CORPUS:
            if case.category != "A-direct-command":
                continue
            r = Result(case, router, lexicon)
            assert r.executes, r.case.utterance

    def test_no_false_execution_at_all(self, router, lexicon):
        assert by_class(results(router, lexicon), FALSE_EXECUTION) == []

    def test_every_case_is_classified(self, router, lexicon):
        allowed = {CORRECT_COMMAND, CORRECT_BLOCK, FALSE_BLOCK,
                   FALSE_EXECUTION, AMBIGUOUS, UNSUPPORTED_BUT_SAFE}
        for r in results(router, lexicon):
            assert r.classification in allowed, r.case.utterance

    def test_undecided_cases_are_never_scored(self, router, lexicon):
        for r in results(router, lexicon):
            if r.case.expected == UNDECIDED:
                assert r.classification == AMBIGUOUS

    def test_totals_add_up(self, router, lexicon):
        items = results(router, lexicon)
        assert sum(len(by_class(items, n)) for n in (
            CORRECT_COMMAND, CORRECT_BLOCK, FALSE_BLOCK,
            FALSE_EXECUTION, AMBIGUOUS, UNSUPPORTED_BUT_SAFE,
        )) == len(items)

    def test_recorded_totals_are_accurate(self, router, lexicon):
        items = results(router, lexicon)
        assert len(by_class(items, CORRECT_COMMAND)) == RECORDED_CORRECT_COMMAND
        assert len(by_class(items, CORRECT_BLOCK)) == RECORDED_CORRECT_BLOCK
        assert len(by_class(items, FALSE_BLOCK)) == RECORDED_FALSE_BLOCK
        assert len(by_class(items, FALSE_EXECUTION)) == RECORDED_FALSE_EXECUTION
        assert len(by_class(items, AMBIGUOUS)) == RECORDED_AMBIGUOUS
        assert len(by_class(items, UNSUPPORTED_BUT_SAFE)) == RECORDED_UNSUPPORTED_BUT_SAFE

    def test_the_focus_sentences_are_traced(self, router, lexicon):
        focus = {
            "can I quit", "should I exit", "I want you to exit", "can you exit",
            "should we quit", "I should quit smoking",
        }
        for r in results(router, lexicon):
            if r.case.utterance in focus:
                assert r.normalized
                assert r.status

    def test_the_report_is_printed(self, router, lexicon, capsys):
        report = _full_report(router, lexicon)
        with capsys.disabled():
            print(report)

    def test_the_unmet_requests_split_into_two_causes(self, router, lexicon):
        """A guard refusal and a missing trigger are different problems."""
        unmet = by_class(results(router, lexicon), UNSUPPORTED_BUT_SAFE)
        guarded = [r for r in unmet if r.guard_blocked]
        absent = [r for r in unmet if not r.guard_blocked]
        assert len(guarded) == RECORDED_GUARD_BLOCKED
        assert len(absent) == RECORDED_NEVER_A_TRIGGER
        for r in absent:
            assert not r.candidates, r.case.utterance
