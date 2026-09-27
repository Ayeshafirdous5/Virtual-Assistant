"""The public entry point to the Natural Language Understanding layer.

:func:`parse` composes the three layers built so far and turns their output
into one small, immutable result:

``normalize``  ->  ``score_intents``  ->  :class:`ParsedIntent`

It adds no second scoring algorithm. Exact, phrase, fuzzy and containment
matching, the negation guard and the ``system`` safety rule all still belong
to :mod:`assistant.nlu.scoring`; this module only consumes what they return,
decides how confident the answer is, and lifts out a few generic entities.

Independence
------------
The parser imports nothing from the application. It never touches
``assistant.tools``, the router, the context, speech or the database, never
constructs a tool and never executes one. It is the last inert layer, so it
can be exercised in full before anything is wired together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from assistant.nlu.lexicon import Lexicon
from assistant.nlu.normalize import Normalized, normalize
from assistant.nlu.scoring import Candidate, SCORE_PRECISION, score_intents

#: Two candidates this close together are treated as a genuine choice rather
#: than a decision. The comparison is **inclusive**: a difference of exactly
#: 0.05 counts as ambiguous. That follows the convention used everywhere else
#: in the NLU layer, where a threshold is stated as "at least" and tested
#: with ``>=``, so "within 0.05" means ``<= 0.05`` here too.
AMBIGUITY_MARGIN = 0.05

#: The parser is the point where a suggestion becomes an action, so a floor
#: is applied here. The legacy containment stage scores 0.50 and exists to
#: preserve old behaviour for diagnostics, not to be acted on blindly: it is
#: how "replay" and "playback" would reach the YouTube tool. Anything at or
#: above this score was matched with a real word or a near-miss.
MIN_CONFIDENCE = 0.65

#: Confidence labels.
CLEAR = "clear"
AMBIGUOUS = "ambiguous"

#: Prepositions that introduce the object of a request.
TAIL_PREPOSITIONS = frozenset({"about", "on", "for", "regarding"})

#: Quoted text in the raw utterance. Quotation marks are punctuation, so
#: normalisation erases them; this is the one entity that has to be read
#: before the text is folded.
_QUOTED = re.compile(r"[\"']\s?([^\"']{2,}?)\s?[\"']")


@dataclass(frozen=True)
class Alternative:
    """A runner-up intent, reduced to what a caller needs.

    A small immutable type rather than the internal
    :class:`~assistant.nlu.scoring.Candidate`, so the parser's public result
    does not expose the scoring layer's internals.
    """

    name: str
    score: float


@dataclass(frozen=True)
class ParsedIntent:
    """What the assistant believes the user asked for.

    Attributes:
        name: the winning intent, matching a tool's name.
        score: confidence between 0 and 1.
        trigger: the lexicon word that produced the match.
        method: the scoring stage that won, for example ``"exact"``.
        confidence: :data:`CLEAR` or :data:`AMBIGUOUS`.
        entities: generic values lifted from the utterance. A key is absent
            when the entity was not present; nothing is invented.
        alternatives: the remaining intents, best first. Empty when the
            winner was the only candidate.
    """

    name: str
    score: float
    trigger: str
    method: str
    confidence: str
    entities: dict[str, object] = field(default_factory=dict)
    alternatives: tuple[Alternative, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        """True when the runner-up was close enough to be worth asking about."""
        return self.confidence == AMBIGUOUS

    @property
    def is_clear(self) -> bool:
        """True when the winner was decisive."""
        return self.confidence == CLEAR


def _tail(tokens: tuple[str, ...]) -> str | None:
    """Return the words after the first preposition, if there are any."""
    for index, token in enumerate(tokens):
        if token in TAIL_PREPOSITIONS:
            remainder = " ".join(tokens[index + 1 :]).strip()
            return remainder or None
    return None


def _quoted(raw: str) -> str | None:
    """Return text enclosed in straight or curly quotation marks."""
    match = _QUOTED.search(raw or "")
    return match.group(1).strip() if match else None


def extract_entities(normalized: Normalized, raw: str) -> dict[str, object]:
    """Lift generic values out of an utterance.

    Only things no tool could reasonably own are extracted. Anything
    tool-specific, such as which note to delete, stays in the tool's own
    ``extract_slots`` and is never duplicated here.

    Args:
        normalized: the normalised utterance, used for numbers and the tail.
        raw: the original utterance, used only for quoted text, which
            normalisation removes.

    Returns:
        A dict holding only the entities actually present.
    """
    entities: dict[str, object] = {}

    if normalized.numbers:
        entities["number"] = normalized.numbers[0]

    tail = _tail(normalized.tokens)
    if tail:
        entities["tail"] = tail

    quoted = _quoted(raw)
    if quoted:
        entities["quoted"] = quoted

    return entities


def _decide_confidence(
    best: Candidate, runner_up: Candidate | None
) -> str:
    """Label the result :data:`CLEAR` or :data:`AMBIGUOUS`.

    Ordering comes straight from
    :func:`~assistant.nlu.scoring.rank_candidates`, so the comparison follows
    the ranking the scoring layer already applied and no new tie-breaking
    policy is introduced here.

    The gap is rounded before it is compared. Without that, ``1.0 - 0.95``
    evaluates to ``0.050000000000000044`` in binary floating point, which is
    greater than 0.05, so the inclusive boundary would quietly reject the
    exact tie it exists to catch. Rounding to the same precision the scoring
    layer already uses keeps the rule honest.
    """
    if runner_up is None:
        return CLEAR
    gap = round(best.score - runner_up.score, SCORE_PRECISION)
    return AMBIGUOUS if gap <= AMBIGUITY_MARGIN else CLEAR


def _to_alternatives(candidates: list[Candidate]) -> tuple[Alternative, ...]:
    """Reduce the runners-up to small immutable alternatives."""
    return tuple(
        Alternative(name=item.intent, score=item.score) for item in candidates
    )


def parse(utterance: str | None, lexicon: Lexicon) -> ParsedIntent | None:
    """Interpret one utterance.

    Runs the full pipeline: normalise the text, score it against the
    lexicon, drop anything too weak to act on, decide how confident the
    answer is, and lift out a few generic entities.

    Args:
        utterance: the raw utterance, exactly as the listener produced it.
            It is never modified or passed on; the parser only reports.
        lexicon: the vocabulary to match against.

    Returns:
        A :class:`ParsedIntent`, or ``None`` when nothing matched strongly
        enough to act on. No placeholder intent is ever invented.
    """
    normalized = normalize(utterance)
    candidates = score_intents(normalized, lexicon)

    # The action boundary: a weak legacy match is not something to act on.
    strong = [item for item in candidates if item.score >= MIN_CONFIDENCE]
    if not strong:
        return None

    best = strong[0]
    runner_up = strong[1] if len(strong) > 1 else None

    return ParsedIntent(
        name=best.intent,
        score=best.score,
        trigger=best.trigger,
        method=best.method,
        confidence=_decide_confidence(best, runner_up),
        entities=extract_entities(normalized, utterance or ""),
        alternatives=_to_alternatives(strong[1:]),
    )
