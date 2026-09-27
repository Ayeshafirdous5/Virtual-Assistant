"""Evidence-based measurement of the NLU decision thresholds.

This file does two jobs, and deliberately nothing else.

First it **measures** the current configuration against
:mod:`tests.nlu_tuning_corpus` and records the result, so the numbers in
``docs/nlu-tuning.md`` can be reproduced at any time.

Second it **sweeps** ``MIN_CONFIDENCE`` and ``AMBIGUITY_MARGIN`` across a
range of candidate values and reports what each one would do, so any
decision to change a constant rests on numbers rather than on taste.

The constants are never modified. Every sweep patches the module attribute
for the duration of one evaluation and restores it afterwards, so the
running assistant is unaffected and the suite proves the shipped values
still behave correctly.

``FUZZY_CUTOFF`` is not swept. Its value is a recorded decision, covered
here only by a guard test.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from assistant.app import build_runtime_lexicon
from assistant.nlu import parser as parser_mod
from assistant.nlu.parser import AMBIGUOUS, parse
from assistant.nlu.scoring import FUZZY_CUTOFF
from assistant.tools import build_default_router
from tests.nlu_tuning_corpus import (
    AMBIGUOUS as EXPECT_AMBIGUOUS,
    CORPUS,
    GROUP_KNOWN_GOOD,
    GROUP_NEAR_TIE,
    GROUP_SAFETY,
    NO_MATCH,
    label,
)

# The values the shipped code uses today.
CURRENT_MIN_CONFIDENCE = parser_mod.MIN_CONFIDENCE
CURRENT_AMBIGUITY_MARGIN = parser_mod.AMBIGUITY_MARGIN

#: Candidate values swept for MIN_CONFIDENCE: the suggested band, widened
#: far enough on both sides to find where behaviour actually changes, so
#: the shipped value can be seen to sit on a plateau rather than a cliff.
MIN_CONFIDENCE_SWEEP = (
    0.50, 0.55, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.75, 0.80, 0.85, 0.90,
)

#: Candidate values swept for AMBIGUITY_MARGIN, likewise widened past the
#: suggested band so the edges of the decision are visible.
AMBIGUITY_MARGIN_SWEEP = (
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.10, 0.15, 0.20,
)

# Outcome labels.
CORRECT = "correct"
FALSE_POSITIVE = "false_positive"
MISSED = "missed"
WRONG_INTENT = "wrong_intent"
UNNECESSARY_AMBIGUITY = "unnecessary_ambiguity"
MISSED_AMBIGUITY = "missed_ambiguity"

#: The real runtime vocabulary, built from the real registered tools.
LEXICON = build_runtime_lexicon(build_default_router())


@contextlib.contextmanager
def thresholds(min_confidence: float, ambiguity_margin: float):
    """Temporarily apply candidate threshold values.

    Both constants are read from the module namespace at call time, so
    patching the attributes changes the behaviour of :func:`parse` for the
    duration and nothing else.
    """
    original_min = parser_mod.MIN_CONFIDENCE
    original_margin = parser_mod.AMBIGUITY_MARGIN
    parser_mod.MIN_CONFIDENCE = min_confidence
    parser_mod.AMBIGUITY_MARGIN = ambiguity_margin
    try:
        yield
    finally:
        parser_mod.MIN_CONFIDENCE = original_min
        parser_mod.AMBIGUITY_MARGIN = original_margin


@dataclass(frozen=True)
class Observation:
    """What the NLU actually did with one utterance."""

    case: object
    actual: str | None
    score: float
    method: str
    confidence: str
    outcome: str

    def describe(self) -> str:
        """One readable line, used in assertion messages."""
        return (
            f"{self.case.utterance!r}: expected {label(self.case.expected)}, "
            f"got {label(self.actual)} (score={self.score:.3f} "
            f"method={self.method} confidence={self.confidence}) "
            f"-> {self.outcome}"
        )


@dataclass
class Summary:
    """Counts for one threshold combination."""

    min_confidence: float
    ambiguity_margin: float
    observations: list = field(default_factory=list)

    def count(self, outcome: str) -> int:
        return sum(1 for o in self.observations if o.outcome == outcome)

    @property
    def correct(self) -> int:
        return self.count(CORRECT)

    @property
    def false_positives(self) -> int:
        return self.count(FALSE_POSITIVE)

    @property
    def missed(self) -> int:
        return self.count(MISSED)

    @property
    def wrong_intent(self) -> int:
        return self.count(WRONG_INTENT)

    @property
    def unnecessary_ambiguity(self) -> int:
        return self.count(UNNECESSARY_AMBIGUITY)

    @property
    def missed_ambiguity(self) -> int:
        return self.count(MISSED_AMBIGUITY)

    @property
    def errors(self) -> int:
        return len(self.observations) - self.correct

    def failures(self) -> list:
        return [o for o in self.observations if o.outcome != CORRECT]


def observe(case) -> Observation:
    """Run the NLU on one case and classify the outcome."""
    parsed = parse(case.utterance, LEXICON)

    actual = parsed.name if parsed is not None else None
    score = parsed.score if parsed is not None else 0.0
    method = parsed.method if parsed is not None else "-"
    confidence = parsed.confidence if parsed is not None else "-"
    is_ambiguous = parsed is not None and parsed.confidence == AMBIGUOUS

    if case.expected is NO_MATCH:
        outcome = CORRECT if parsed is None else FALSE_POSITIVE
    elif case.expected is EXPECT_AMBIGUOUS:
        if parsed is None:
            outcome = MISSED
        elif is_ambiguous:
            outcome = CORRECT
        else:
            outcome = MISSED_AMBIGUITY
    else:
        if parsed is None:
            outcome = MISSED
        elif is_ambiguous:
            outcome = UNNECESSARY_AMBIGUITY
        elif actual != case.expected:
            outcome = WRONG_INTENT
        else:
            outcome = CORRECT

    return Observation(case, actual, score, method, confidence, outcome)


def evaluate(
    min_confidence: float = CURRENT_MIN_CONFIDENCE,
    ambiguity_margin: float = CURRENT_AMBIGUITY_MARGIN,
    corpus=CORPUS,
) -> Summary:
    """Evaluate the whole corpus at one pair of threshold values."""
    with thresholds(min_confidence, ambiguity_margin):
        summary = Summary(min_confidence, ambiguity_margin)
        summary.observations = [observe(case) for case in corpus]
        return summary


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_size_is_in_the_expected_range(self):
        assert 60 <= len(CORPUS) <= 100

    def test_has_no_duplicate_utterances(self):
        texts = [case.utterance for case in CORPUS]
        assert len(texts) == len(set(texts))

    def test_covers_every_group(self):
        groups = {case.group for case in CORPUS}
        assert "A-known-good" in groups
        assert "B-safety" in groups
        assert "C-natural" in groups

    def test_covers_every_registered_tool(self):
        expected = {t.name for t in build_default_router().tools}
        covered = {c.expected for c in CORPUS if c.expected not in (None, "ambiguous")}
        assert expected <= covered

    def test_includes_the_safety_classes(self):
        texts = {case.utterance for case in CORPUS}
        for required in (
            "i noted that down",
            "replay my song",
            "playback speed",
            "goodbye is in the dictionary",
            "quite good",
            "I don't want to exit",
            "never play music",
            "do not play music",
        ):
            assert required in texts, required

    def test_includes_the_nwes_case(self):
        assert "nwes" in {case.utterance for case in CORPUS}

    def test_includes_the_ambiguity_case(self):
        texts = {c.utterance for c in CORPUS if c.expected is EXPECT_AMBIGUOUS}
        assert "information about the news" in texts

    def test_every_non_obvious_case_is_explained(self):
        """A surprising label must say why; an obvious one need not.

        Safety cases, near-ties and unresolved labels are the ones where a
        reader needs the reasoning. A plain no-match such as "what should
        I cook tonight" needs none.
        """
        for case in CORPUS:
            surprising = case.group in (GROUP_SAFETY, GROUP_NEAR_TIE)
            if surprising:
                assert case.note, case.utterance


# ----------------------------------------------------------------------
# The shipped configuration
# ----------------------------------------------------------------------
class TestCurrentConfiguration:
    def test_no_safety_case_reaches_a_tool(self):
        """The real safety guarantee: no near miss may run a tool.

        Scoped to the safety group. The natural group deliberately contains
        a few known limitations, which are recorded separately rather than
        being hidden or relabelled.
        """
        summary = evaluate()
        offenders = [
            o
            for o in summary.observations
            if o.outcome == FALSE_POSITIVE and o.case.group == GROUP_SAFETY
        ]
        assert not offenders, "\n".join(o.describe() for o in offenders)

    def test_every_known_good_command_still_resolves(self):
        """The 25 backward-compatibility commands must all still work."""
        summary = evaluate()
        broken = [
            o
            for o in summary.observations
            if o.outcome == MISSED and o.case.group == GROUP_KNOWN_GOOD
        ]
        assert not broken, "\n".join(o.describe() for o in broken)

    def test_no_command_resolves_to_the_wrong_tool(self):
        """No case is routed to an intent the corpus did not expect."""
        summary = evaluate()
        confused = [o for o in summary.observations if o.outcome == WRONG_INTENT]
        assert not confused, "\n".join(o.describe() for o in confused)

    def test_the_ambiguity_case_is_ambiguous(self):
        summary = evaluate()
        assert not summary.missed_ambiguity

    def test_nwes_stays_rejected(self):
        summary = evaluate()
        observation = next(
            o for o in summary.observations if o.case.utterance == "nwes"
        )
        assert observation.actual is None
        assert observation.outcome == CORRECT

    def test_fuzzy_cutoff_is_unchanged(self):
        # Step 7 does not tune this. It is asserted so a future change has
        # to be deliberate.
        assert FUZZY_CUTOFF == 0.80

    def test_nwes_is_still_below_the_cutoff(self):
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, "nwes", "news").ratio()
        assert round(ratio, 3) == 0.750
        assert ratio < FUZZY_CUTOFF

    def test_measurement_is_deterministic(self):
        first = evaluate()
        second = evaluate()
        assert [o.outcome for o in first.observations] == [
            o.outcome for o in second.observations
        ]

    def test_sweeping_restores_the_shipped_values(self):
        evaluate(0.90, 0.20)
        assert parser_mod.MIN_CONFIDENCE == CURRENT_MIN_CONFIDENCE
        assert parser_mod.AMBIGUITY_MARGIN == CURRENT_AMBIGUITY_MARGIN


# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------
def _table(values, **kwargs):
    """Render a sweep as fixed-width text, for the tuning record."""
    rows = []
    for value in values:
        summary = evaluate(min_confidence=value, **kwargs) if "min_confidence" in kwargs else evaluate(ambiguity_margin=value, **kwargs)
        rows.append((value, summary))
    return rows


class TestMinConfidenceSweep:
    def test_shipped_value_is_optimal(self):
        """No other value scores better than the one already shipped."""
        results = {v: evaluate(min_confidence=v).errors for v in MIN_CONFIDENCE_SWEEP}
        best = min(results.values())
        assert results[CURRENT_MIN_CONFIDENCE] == best, results

    def test_shipped_value_is_not_a_cliff_but_a_plateau(self):
        """A wide band of values behaves identically, so 0.65 is not fragile."""
        plateau = [
            v for v in MIN_CONFIDENCE_SWEEP
            if evaluate(min_confidence=v).errors
            == evaluate(min_confidence=CURRENT_MIN_CONFIDENCE).errors
        ]
        assert len(plateau) >= 10, plateau

    def test_containment_must_stay_below_the_floor(self):
        """Dropping to 0.50 lets the legacy containment stage become actionable."""
        low = evaluate(min_confidence=0.50)
        shipped = evaluate(min_confidence=CURRENT_MIN_CONFIDENCE)
        assert low.false_positives > shipped.false_positives

    def test_raising_the_floor_never_helps(self):
        """Higher floors only lose commands; they never remove a false positive."""
        for value in (0.70, 0.80, 0.90):
            assert evaluate(min_confidence=value).missed >= evaluate().missed


class TestAmbiguityMarginSweep:
    def test_shipped_value_is_optimal(self):
        results = {v: evaluate(ambiguity_margin=v).errors for v in AMBIGUITY_MARGIN_SWEEP}
        best = min(results.values())
        assert results[CURRENT_AMBIGUITY_MARGIN] == best, results

    def test_lowering_the_margin_loses_the_phrase_ties(self):
        """0.050 is exactly EXACT_SCORE - PHRASE_SCORE; below it they go unasked."""
        from assistant.nlu.scoring import PHRASE_SCORE

        assert round(1.0 - PHRASE_SCORE, 4) == 0.05
        lowered = evaluate(ambiguity_margin=0.04)
        assert lowered.missed_ambiguity == 2

    def test_raising_the_margin_over_asks(self):
        """A wide margin turns a decisive exact-over-fuzzy win into a question."""
        raised = evaluate(ambiguity_margin=0.10)
        assert raised.unnecessary_ambiguity > evaluate().unnecessary_ambiguity

    def test_the_optimal_window_is_bounded(self):
        """Values at or above the smallest non-zero gap, and below it, both lose."""
        for worse in (0.04, 0.10, 0.20):
            assert evaluate(ambiguity_margin=worse).errors > evaluate().errors


class TestShippedConstants:
    """The measured decision, recorded so it cannot drift unnoticed."""

    def test_min_confidence_is_unchanged_at_065(self):
        assert parser_mod.MIN_CONFIDENCE == 0.65

    def test_ambiguity_margin_is_unchanged_at_005(self):
        assert parser_mod.AMBIGUITY_MARGIN == 0.05

    def test_fuzzy_cutoff_is_unchanged_at_080(self):
        assert FUZZY_CUTOFF == 0.80

    def test_known_gap_structure_is_documented_in_the_record(self):
        """The exact-versus-phrase gap is what the margin is sized against."""
        from assistant.nlu.scoring import PHRASE_SCORE

        assert round(1.0 - PHRASE_SCORE, 4) == CURRENT_AMBIGUITY_MARGIN


class TestRemainingFailuresAreNotThresholdFixable:
    """The residual errors are architectural, and that is the finding."""

    def test_they_all_sit_at_the_extremes(self):
        """Either a perfect 1.000 match, or nothing at all, or the margin trade-off."""
        for observation in evaluate().failures():
            assert observation.score in (0.0, 1.0), observation.describe()
            assert observation.outcome in (
                FALSE_POSITIVE,
                MISSED,
                UNNECESSARY_AMBIGUITY,
            ), observation.describe()

    def test_known_limitations_are_still_reported(self):
        failures = {o.case.utterance: o.outcome for o in evaluate().failures()}
        assert failures["quit the assistant"] == MISSED
        assert failures["stop the assistant"] == MISSED
        assert failures["the weather is nice today I guess"] == FALSE_POSITIVE
        assert failures["I want to log this information"] == FALSE_POSITIVE
