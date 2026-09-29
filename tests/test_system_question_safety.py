"""Step 19: why questions about a command word still run that command.

Step 18 found three sentences that terminate the assistant even though the
user was only asking about the word:

    "what does quit mean"    "what does exit mean"    "do you want to quit"

This module traces them and measures the surrounding family. It changes
nothing. The cases are recorded as they behave, so a later fix has to
update this file deliberately rather than drift.

Why the question is interesting
-------------------------------
Every layer behaved correctly *in isolation* and the answer is still wrong:

* the scorer matched a real, registered trigger on a real word boundary;
* the guard passed it, because all three of its conditions held;
* the parser was confident and clear;
* **framing was consulted and actively said ``request``**, because a leading
  interrogative is one of framing's request cues.

That last point is the crux. Framing's rule is right in general -- "what is
the weather" *is* a request -- and wrong here, because the trigger is the
object of a question rather than the action being asked for. Framing cannot
own this fix without breaking legitimate questions, so the guard can.

Safety categories
-----------------
Every case is labelled with what the user **meant**, and the observed
outcome is classified against that label. ``DANGEROUS_EXECUTION`` is the
severe one and is never relaxed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from assistant.app import build_runtime_lexicon, resolve_detail
from assistant.core.context import AppContext
from assistant.core.router import NoMatch
from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.nlu.scoring import (
    SELF_OTHER_SUBJECTS,
    SYSTEM_TAIL_TOKENS,
    score_intents,
)
from assistant.tools import build_default_router

#: The user means "do this".
YES = "yes"
#: The user is asking about, or describing, not asking for.
NO = "no"
#: Genuinely undecidable: the same English reads both ways.
UNDECIDED = "undecided"

# Result classes.
SAFE_COMMAND = "SAFE_COMMAND"
SAFE_BLOCK = "SAFE_BLOCK"
DANGEROUS_EXECUTION = "DANGEROUS_EXECUTION"
AMBIGUOUS_BUT_SAFE = "AMBIGUOUS_BUT_SAFE"
UNSUPPORTED = "UNSUPPORTED"

GROUP_MEANING = "A-meaning-question"
GROUP_PERSON = "B-about-quitting"
GROUP_COMMAND = "C-genuine-command"
GROUP_ORDINARY = "D-ordinary-sentence"

GROUPS = (GROUP_MEANING, GROUP_PERSON, GROUP_COMMAND, GROUP_ORDINARY)


@dataclass(frozen=True)
class QuestionCase:
    """One utterance and the safety outcome it should have."""

    utterance: str
    expected: str
    group: str
    reason: str = ""


# ----------------------------------------------------------------------
# A. Asking what a command word means. Never an action request.
# ----------------------------------------------------------------------
MEANING_QUESTIONS: list[QuestionCase] = [
    QuestionCase("what does quit mean", NO, GROUP_MEANING,
                 "the Step 18 finding: asks for a definition"),
    QuestionCase("what does exit mean", NO, GROUP_MEANING,
                 "the Step 18 finding: asks for a definition"),
    QuestionCase("do you want to quit", NO, GROUP_MEANING,
                 "the Step 18 finding: asks the assistant's preference"),
    QuestionCase("what does goodbye mean", NO, GROUP_MEANING,
                 "asks for a definition"),
    QuestionCase("what does shut down mean", NO, GROUP_MEANING,
                 "asks for a definition"),
    QuestionCase("what is the meaning of quit", NO, GROUP_MEANING,
                 "asks for a definition"),
    QuestionCase("define exit", NO, GROUP_MEANING,
                 "an imperative about a word, not an action"),
    QuestionCase("explain quit", NO, GROUP_MEANING,
                 "an imperative about a word, not an action"),
]

# ----------------------------------------------------------------------
# B. Questions about whether someone should, or does, quit.
# ----------------------------------------------------------------------
PERSON_QUESTIONS: list[QuestionCase] = [
    QuestionCase("should I quit", UNDECIDED, GROUP_PERSON,
                 "may mean 'may I' or 'advise me'"),
    QuestionCase("should you quit", UNDECIDED, GROUP_PERSON,
                 "may mean 'advise me' or 'you should quit'"),
    QuestionCase("can I quit", UNDECIDED, GROUP_PERSON,
                 "a permission question"),
    QuestionCase("can you quit", YES, GROUP_PERSON,
                 "a request addressed to the assistant"),
    QuestionCase("will you quit", YES, GROUP_PERSON,
                 "a request addressed to the assistant"),
    QuestionCase("why did he quit", NO, GROUP_PERSON, "about a third party"),
    QuestionCase("did she quit", NO, GROUP_PERSON, "about a third party"),
    QuestionCase("what does quitting mean", NO, GROUP_PERSON,
                 "asks for a definition"),
]

# ----------------------------------------------------------------------
# C. Genuine commands. Every one must keep working.
# ----------------------------------------------------------------------
GENUINE_COMMANDS: list[QuestionCase] = [
    QuestionCase("exit", YES, GROUP_COMMAND, "bare command"),
    QuestionCase("quit", YES, GROUP_COMMAND, "bare command"),
    QuestionCase("goodbye", YES, GROUP_COMMAND, "bare command"),
    QuestionCase("bye", YES, GROUP_COMMAND, "bare command"),
    QuestionCase("please exit", YES, GROUP_COMMAND, "politeness, no subject"),
    QuestionCase("goodbye assistant", YES, GROUP_COMMAND,
                 "addresses the assistant, no subject before the trigger"),
    QuestionCase("shut down", YES, GROUP_COMMAND, "two-word alias"),
    QuestionCase("you can quit now", YES, GROUP_COMMAND,
                 "second person, the addressee"),
    QuestionCase("can you exit", YES, GROUP_COMMAND,
                 "a question addressed to the assistant"),
    QuestionCase("would you quit now", YES, GROUP_COMMAND,
                 "a polite question addressed to the assistant"),
]

# ----------------------------------------------------------------------
# D. Ordinary sentences.
# ----------------------------------------------------------------------
ORDINARY_SENTENCES: list[QuestionCase] = [
    QuestionCase("I should quit smoking", NO, GROUP_ORDINARY,
                 "the original Step 15 finding"),
    QuestionCase("I want to quit smoking", NO, GROUP_ORDINARY,
                 "a personal decision"),
    QuestionCase("he quit smoking", NO, GROUP_ORDINARY, "a third party"),
    QuestionCase("the device will power off", NO, GROUP_ORDINARY,
                 "no system trigger"),
    QuestionCase("there is an exit sign", NO, GROUP_ORDINARY, "a sign"),
]

CORPUS: list[QuestionCase] = (
    MEANING_QUESTIONS + PERSON_QUESTIONS + GENUINE_COMMANDS + ORDINARY_SENTENCES
)


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


class Trace:
    """The full pipeline trace for one utterance."""

    def __init__(self, case: QuestionCase, router, lexicon) -> None:
        self.case = case
        utterance = case.utterance
        normalized = normalize(utterance)
        parsed = parse(utterance, lexicon)
        candidates = score_intents(normalized, lexicon)

        ctx = AppContext()
        ctx.speaker = Silent()
        ctx.listener = Silent()
        resolution = resolve_detail(ctx, router, utterance, lexicon)
        match = router.find_match(utterance)

        self.normalized = normalized.text
        self.tokens = normalized.tokens
        self.candidates = [(c.intent, c.trigger, round(c.score, 3), c.method)
                           for c in candidates]
        self.parsed_name = parsed.name if parsed else None
        self.score = round(parsed.score, 3) if parsed else 0.0
        self.confidence = parsed.confidence if parsed else "-"
        # What framing did, and whether it was even reached.
        self.framing_consulted = parsed is not None
        self.framing = resolution.framing or "-"
        self.status = resolution.status
        self.tool = resolution.tool_name or "none"
        self.executes = resolution.tool_name == "system"
        self.legacy = "" if isinstance(match, NoMatch) else match.tool_name

        trigger = parsed.trigger if parsed else None
        self.trigger_index = (
            normalized.tokens.index(trigger)
            if trigger and trigger in normalized.tokens
            else -1
        )
        self.in_window = (
            self.trigger_index >= 0
            and self.trigger_index
            >= max(0, len(normalized.tokens) - SYSTEM_TAIL_TOKENS)
        )
        self.subject_before = [
            t for t in normalized.tokens[: self.trigger_index]
            if t in SELF_OTHER_SUBJECTS
        ] if self.trigger_index > 0 else []

    @property
    def classification(self) -> str:
        if self.case.expected == UNDECIDED:
            return AMBIGUOUS_BUT_SAFE
        if self.executes:
            return SAFE_COMMAND if self.case.expected == YES else DANGEROUS_EXECUTION
        return SAFE_BLOCK if self.case.expected == NO else UNSUPPORTED

    def row(self) -> str:
        return (
            f"  {self.case.utterance!r:<30} {self.case.expected:<9} "
            f"idx={self.trigger_index:<2} win={str(self.in_window):<5} "
            f"subj={str(self.subject_before or '-'):<8} "
            f"framing={self.framing:<8} exec={str(self.executes):<5} "
            f"{self.classification}"
        )


def traces(router, lexicon) -> list[Trace]:
    return [Trace(case, router, lexicon) for case in CORPUS]


def by_class(items, name: str) -> list[Trace]:
    return [t for t in items if t.classification == name]


#: Measured when this corpus was first run. Pinned.
#:
#: **Step 20 moved these deliberately.** The apposition guard in
#: :mod:`assistant.nlu.scoring` closed six of the seven dangerous
#: executions, so six findings moved from :data:`DANGEROUS_EXECUTION` to
#: :data:`SAFE_BLOCK`:
#:
#:     what does quit mean          define exit
#:     what does exit mean          explain quit
#:     what does goodbye mean       what is the meaning of quit
#:
#: Only "do you want to quit" is still dangerous, and Step 20 was
#: explicitly told to leave it alone. It is measured here and in
#: ``tests/test_nlu_system_safety_step20.py``, not quietly dropped.
RECORDED_SAFE_COMMAND = 12
RECORDED_SAFE_BLOCK = 15
RECORDED_DANGEROUS = 1
RECORDED_AMBIGUOUS = 3
RECORDED_UNSUPPORTED = 0

#: The one case deliberately left open by Step 20.
RECORDED_REMAINING_DANGER = "do you want to quit"


def _report(router, lexicon) -> str:
    items = traces(router, lexicon)
    lines = ["system question safety (Step 19)", "", "  totals:"]
    for name in (SAFE_COMMAND, SAFE_BLOCK, DANGEROUS_EXECUTION,
                 AMBIGUOUS_BUT_SAFE, UNSUPPORTED):
        lines.append(f"    {name:<22} {len(by_class(items, name))}")
    lines += ["", "  per case:"]
    lines += [t.row() for t in items]
    for name in (DANGEROUS_EXECUTION, UNSUPPORTED):
        group = by_class(items, name)
        if group:
            lines += ["", f"  {name} ({len(group)}):"]
            for t in group:
                lines.append(f"    - {t.case.utterance!r} [{t.case.group}] {t.case.reason}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_size(self):
        assert len(CORPUS) >= 30

    def test_no_duplicates(self):
        texts = [c.utterance for c in CORPUS]
        assert len(texts) == len(set(texts))

    def test_every_group_present(self):
        assert {c.group for c in CORPUS} == set(GROUPS)

    def test_every_case_explained(self):
        for case in CORPUS:
            assert case.reason, case.utterance
            assert case.expected in (YES, NO, UNDECIDED)

    def test_the_three_step_18_findings_are_present_and_labelled_no(self):
        findings = {"what does quit mean", "what does exit mean",
                    "do you want to quit"}
        by_text = {c.utterance: c for c in CORPUS}
        for finding in findings:
            assert finding in by_text, finding
            assert by_text[finding].expected == NO, finding


# ----------------------------------------------------------------------
# The three findings, traced stage by stage
# ----------------------------------------------------------------------
class TestStep18Findings:
    """Kept because the Step 18 findings are what motivated Step 20.

    Two of the three no longer execute. The trace is kept rather than
    deleted, because "the finding no longer reproduces" is the evidence
    that the fix worked, and the third case is the one Step 20 was told
    to leave alone.
    """

    #: Closed by Step 20: an apposition marker now sits before the trigger.
    CLOSED_BY_STEP_20 = ("what does quit mean", "what does exit mean")

    #: Still open, deliberately measured rather than silently dropped.
    STILL_OPEN = ("do you want to quit",)

    @pytest.mark.parametrize("utterance", CLOSED_BY_STEP_20)
    def test_the_closed_findings_no_longer_execute(self, router, lexicon, utterance):
        trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, "finding"),
                      router, lexicon)
        assert not trace.executes, "the fix was undone"
        assert trace.classification == SAFE_BLOCK

    @pytest.mark.parametrize("utterance", STILL_OPEN)
    def test_the_open_finding_still_executes(self, router, lexicon, utterance):
        """Step 20's instruction was to leave this one alone."""
        trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, "finding"),
                      router, lexicon)
        assert trace.executes, "the finding no longer reproduces"
        assert trace.classification == DANGEROUS_EXECUTION
        assert utterance == RECORDED_REMAINING_DANGER

    def test_the_guard_no_longer_passes_the_closed_findings(self, router, lexicon):
        """The apposition condition is what now stops them.

        Score, window and subject all still hold, so they cannot be
        asserted through the trace: the guard filters the candidate out
        and no system candidate survives to be scored. The marker's
        presence in the token stream is asserted instead, which is the
        actual cause.
        """
        from assistant.nlu.normalize import normalize
        from assistant.nlu.scoring import DEFINITION_MARKERS

        for utterance in self.CLOSED_BY_STEP_20:
            trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, ""),
                          router, lexicon)
            # The guard dropped the candidate, so the trace has no trigger
            # to report. The marker's place in the token stream is asserted
            # instead, because that is the actual cause.
            tokens = normalize(utterance).tokens
            at = tokens.index("quit" if "quit" in tokens else "exit")
            assert at > 0, utterance
            assert tokens[at - 1] in DEFINITION_MARKERS, utterance
            assert trace.score == 0.0, utterance
            assert not trace.executes, utterance

    def test_the_open_finding_still_passes_every_condition(self, router, lexicon):
        """Why it survived: the token before "quit" is "to", not a marker."""
        for utterance in self.STILL_OPEN:
            trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, ""),
                          router, lexicon)
            assert trace.score >= 0.95
            assert trace.in_window, utterance
            assert not trace.subject_before, utterance
            assert trace.executes, utterance

    def test_framing_is_consulted_only_for_the_open_finding(self, router, lexicon):
        """The crux, updated: framing is now reached for one case, not three."""
        for utterance in self.CLOSED_BY_STEP_20:
            trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, ""),
                          router, lexicon)
            assert not trace.framing_consulted, utterance
        for utterance in self.STILL_OPEN:
            trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, ""),
                          router, lexicon)
            assert trace.framing_consulted, utterance
            assert trace.framing == framing.REQUEST, utterance

    def test_framing_is_not_reached_when_the_guard_fires(self, router, lexicon):
        """By contrast, a guarded case never gets as far as framing."""
        trace = Trace(QuestionCase("I should quit smoking", NO, GROUP_ORDINARY, ""),
                      router, lexicon)
        assert not trace.executes
        assert trace.parsed_name is None
        assert not trace.framing_consulted


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------
class TestMeasurement:
    def test_every_genuine_command_still_works(self, router, lexicon):
        for case in GENUINE_COMMANDS:
            trace = Trace(case, router, lexicon)
            assert trace.executes, case.utterance
            assert trace.classification == SAFE_COMMAND, case.utterance

    def test_no_ordinary_sentence_terminates(self, router, lexicon):
        for case in ORDINARY_SENTENCES:
            assert not Trace(case, router, lexicon).executes, case.utterance

    def test_the_dangerous_execution_count_is_pinned(self, router, lexicon):
        """The severe class. Recorded, not asserted away."""
        items = traces(router, lexicon)
        assert len(by_class(items, DANGEROUS_EXECUTION)) == RECORDED_DANGEROUS
        assert not by_class(items, UNSUPPORTED), [
            t.case.utterance for t in by_class(items, UNSUPPORTED)
        ]

    def test_recorded_totals_are_accurate(self, router, lexicon):
        items = traces(router, lexicon)
        assert len(by_class(items, SAFE_COMMAND)) == RECORDED_SAFE_COMMAND
        assert len(by_class(items, SAFE_BLOCK)) == RECORDED_SAFE_BLOCK
        assert len(by_class(items, DANGEROUS_EXECUTION)) == RECORDED_DANGEROUS
        assert len(by_class(items, AMBIGUOUS_BUT_SAFE)) == RECORDED_AMBIGUOUS

    def test_totals_add_up(self, router, lexicon):
        items = traces(router, lexicon)
        assert sum(len(by_class(items, n)) for n in (
            SAFE_COMMAND, SAFE_BLOCK, DANGEROUS_EXECUTION,
            AMBIGUOUS_BUT_SAFE, UNSUPPORTED,
        )) == len(items)

    def test_undecided_cases_are_never_scored(self, router, lexicon):
        for t in traces(router, lexicon):
            if t.case.expected == UNDECIDED:
                assert t.classification == AMBIGUOUS_BUT_SAFE

    def test_every_case_is_classified(self, router, lexicon):
        allowed = {SAFE_COMMAND, SAFE_BLOCK, DANGEROUS_EXECUTION,
                   AMBIGUOUS_BUT_SAFE, UNSUPPORTED}
        for t in traces(router, lexicon):
            assert t.classification in allowed, t.case.utterance

    def test_the_trace_is_complete(self, router, lexicon):
        for t in traces(router, lexicon):
            assert t.normalized
            assert isinstance(t.tokens, tuple)
            assert t.confidence
            assert t.status
            assert t.tool

    def test_the_report_is_printed(self, router, lexicon, capsys):
        report = _report(router, lexicon)
        with capsys.disabled():
            print(report)


# ----------------------------------------------------------------------
# Structural observations that shape the proposed boundary
# ----------------------------------------------------------------------
class TestStructuralFindings:
    def test_third_party_questions_are_already_safe(self, router, lexicon):
        """Step 17's subject check already covers these."""
        for utterance in ("why did he quit", "did she quit", "should I quit",
                          "can I quit"):
            trace = Trace(QuestionCase(utterance, NO, GROUP_PERSON, ""),
                          router, lexicon)
            assert not trace.executes, utterance

    def test_the_meaning_family_is_no_longer_covered_by_accident(
        self, router, lexicon
    ):
        """Asking a word's meaning was the open family. Step 20 closed it.

        These six were *deliberately* safe rather than safe by shape, and
        the test is retained in its inverted form so the family cannot
        reopen without someone noticing.
        """
        for utterance in ("what does quit mean", "what does exit mean",
                          "what does goodbye mean",
                          "what is the meaning of quit", "define exit",
                          "explain quit"):
            trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, ""),
                          router, lexicon)
            assert not trace.executes, utterance
            assert trace.classification == SAFE_BLOCK, utterance

    def test_two_meaning_questions_are_safe_for_structural_reasons(
        self, router, lexicon
    ):
        """Not by intent, by accident of shape. Both are worth recording.

        "what does shut down mean" is safe because "shut down" is a
        two-word trigger, so it occupies two tokens and pushes its start
        outside the closing window. "what does quitting mean" is safe
        because "quitting" is not a registered trigger at all. Neither is a
        defence anyone chose.
        """
        for utterance in ("what does shut down mean", "what does quitting mean"):
            trace = Trace(QuestionCase(utterance, NO, GROUP_MEANING, ""),
                          router, lexicon)
            assert not trace.executes, utterance

    def test_the_guard_cannot_separate_them_on_its_own(self, router, lexicon):
        """The finding that forced a fourth condition, and it still holds.

        Every genuine command and the one remaining dangerous question is
        a 1.000 exact match with no subject before the trigger, inside the
        closing window. The first three conditions say nothing about the
        difference, which is why the guard needed a fourth.

        Two-word triggers are excluded because their start index does not
        behave the same way: "shut down" is a safe command that the window
        would not describe at all.
        """
        checked = 0
        for trace in traces(router, lexicon):
            if trace.classification not in (SAFE_COMMAND, DANGEROUS_EXECUTION):
                continue
            if trace.trigger_index < 0:
                continue  # a multi-word trigger, handled separately
            checked += 1
            assert trace.score == 1.0, trace.case.utterance
            assert trace.in_window, trace.case.utterance
            assert not trace.subject_before, trace.case.utterance
        assert checked >= 12, checked

    def test_a_definition_frame_before_the_trigger_is_the_usable_signal(
        self, router, lexicon
    ):
        """The signal Step 19 identified, now carrying no work left to do.

        "does", "do", "of", "define" and "explain" sit immediately before
        the trigger, making it the thing being defined rather than the
        thing being done. No genuine command in this corpus has one, which
        is what made the boundary safe to add.
        """
        frames = {"does", "do", "of", "define", "explain"}
        dangerous = by_class(traces(router, lexicon), DANGEROUS_EXECUTION)
        with_frame = [
            t for t in dangerous
            if t.tokens[: t.trigger_index]
            and t.tokens[: t.trigger_index][-1] in frames
        ]
        # None of the six that used to be here still is. The one survivor
        # has no frame, which is precisely why it is still dangerous.
        assert len(with_frame) == 0, [t.case.utterance for t in with_frame]
        exception = [t for t in dangerous if t not in with_frame]
        assert [t.case.utterance for t in exception] == ["do you want to quit"]

        for case in GENUINE_COMMANDS:
            trace = Trace(case, router, lexicon)
            before = trace.tokens[: max(0, trace.trigger_index)]
            assert not (before and before[-1] in frames), case.utterance

    def test_second_person_commands_are_unaffected(self, router, lexicon):
        """The fix must not touch these; they are the load-bearing cases."""
        for utterance in ("can you exit", "would you quit now", "can you quit",
                          "will you quit", "you can quit now"):
            trace = Trace(QuestionCase(utterance, YES, GROUP_COMMAND, ""),
                          router, lexicon)
            assert trace.executes, utterance
            assert not trace.subject_before, utterance

    def test_the_guard_is_the_only_system_aware_layer(self, router, lexicon):
        """Scoping: the guard is reached for system and nothing else."""
        assert SELF_OTHER_SUBJECTS
        assert SYSTEM_TAIL_TOKENS == 2
