"""Intent scoring for the Natural Language Understanding layer.

Takes a :class:`~assistant.nlu.normalize.Normalized` utterance and a
:class:`~assistant.nlu.lexicon.Lexicon`, and returns ranked
:class:`Candidate` objects. It answers one question: *how strongly does this
utterance suggest each intent, and why?*

It is deliberately inert. Nothing here imports the tools, the router, the
speech stack or the database, constructs a ``Tool``, executes anything, or
extracts slots. It reads data and produces numbers.

Stages
------
The first stage that produces a candidate for a given intent wins; lower
stages only fill gaps.

===== ================ ====== =========================================
Stage Method           Score  Notes
===== ================ ====== =========================================
1     ``exact``        1.00   single-word trigger on a word boundary
2     ``phrase``       0.95   multi-word trigger, exact, contiguous
3     ``fuzzy``        0.65+  ``difflib`` on tokens of 4+ characters
4     ``containment``  0.50   legacy substring behaviour, last resort
===== ================ ====== =========================================

Two guards shape the result. A **leading trigger** earns a small structural
bonus, and the ``system`` intent, which can terminate the assistant, is held
to a much higher bar; see :func:`_passes_system_guard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from assistant.nlu.lexicon import Lexicon
from assistant.nlu.normalize import Normalized

#: Confidence of a single-word exact trigger.
EXACT_SCORE = 1.00
#: Confidence of a multi-word phrase trigger.
PHRASE_SCORE = 0.95
#: Confidence of the legacy containment fallback.
CONTAINMENT_SCORE = 0.50

#: Minimum fuzzy score after the length penalty is applied.
FUZZY_CUTOFF = 0.80
#: Shortest token considered for fuzzy matching. Below this, almost every
#: short word is within a few edits of many others and noise wins.
FUZZY_MIN_TOKEN = 4
#: Multiplier that softens a fuzzy hit when the token is shorter than the
#: trigger it is compared with.
LENGTH_PENALTY_FACTOR = 1.2
#: How much longer than a trigger a token may be and still count as a
#: misspelling of it. A typo is normally the same size or shorter, so this
#: rejects "replay" as a candidate match for "play" while still allowing a
#: genuine one-character slip in either direction.
FUZZY_LENGTH_TOLERANCE = 1.4
#: Longest tail still treated as a plausible inflection, in characters. See
#: :func:`is_inflected_form`.
MAX_INFLECTION_SUFFIX = 3

#: Structural bonus when a trigger starts the utterance.
LEADING_BONUS = 0.05
#: Hard ceiling, so a bonus can never exceed a perfect exact match.
MAX_SCORE = 1.00

#: The exit intent needs near certainty, because a false positive here ends
#: the program. Fuzzy and containment candidates can never reach it.
SYSTEM_INTENT = "system"
SYSTEM_MIN_SCORE = 0.95
#: An exit word must also sit at the very end of the utterance, so that
#: "goodbye is in the dictionary" is not read as a command to quit.
SYSTEM_TAIL_TOKENS = 2

#: Words that name the **speaker** or a **third party**.
#:
#: A system trigger preceded by one of these is almost never an order to the
#: assistant. "I should quit smoking", "she decided to quit smoking" and
#: "he plans to quit smoking" all use a command word as the object of a
#: personal decision, and all of them used to terminate the application.
#:
#: Second person is deliberately **absent**. "you can quit now" and "goodbye
#: assistant" address the assistant, which is exactly the case the guard is
#: meant to allow, so "you" must never block.
#:
#: This is scoped to the ``system`` intent alone. A sentence such as "I need
#: a joke" is untouched, because the guard is never consulted for it.
SELF_OTHER_SUBJECTS = frozenset(
    {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
    }
)

#: Words that put a following noun into **apposition**: it is being
#: named or explained, not acted on.
#:
#:     "what does quit mean"          "does" before "quit"
#:     "what is the meaning of quit"   "of"   before "quit"
#:     "define exit"                   "define" before "exit"
#:     "explain quit"                  "explain" before "quit"
#:
#: All four used to terminate the application. The subject rule above
#: cannot catch them, because none of them names a speaker or a third
#: party: "what does quit mean" names nobody at all, and the subject
#: window sees only "what does".
#:
#: Scoped deliberately. This is **not** a general question rule: a
#: trigger is refused only when one of these words sits immediately
#: before it, and only for a **single-word** trigger. Two consequences
#: are load-bearing:
#:
#: * A multi-word trigger such as "shut down" is exempt, so
#:   "please shut down" keeps working. Were the check applied to the
#:   trigger's last token instead of its start, "down" would be treated
#:   as the trigger and the guard would read the wrong neighbour.
#: * A bare "do" is in the set because "what do you mean by quit" and
#:   "what does quit mean" are the same shape, but "do you want to
#:   quit" is **not** caught: the token before "quit" there is "to".
#:   That sentence is a separate safety problem and is left for a
#:   later step rather than half-fixed here.
DEFINITION_MARKERS = frozenset({"does", "do", "of", "define", "explain"})

#: Inflected forms of "want", used only to recognise a **want frame**.
#:
#: The frame this exists to catch is "you ... want to <system trigger>":
#:
#:     "do you want to quit"       a question about the assistant
#:     "you want to exit"          a statement about the assistant
#:
#: A want frame alone decides nothing. "I want to quit smoking" and
#: "do you want me to quit" both contain one, and both are already
#: refused by the subject rule because a real subject precedes the
#: trigger. What is unsafe is specifically **"you"** in that frame,
#: because "you" is the one pronoun the subject rule cannot use.
#:
#: The word ``to`` is load-bearing and is *not* being used as a safety
#: marker on its own. It is read together with a want form, so the
#: structure that matters is "want to" directly before the trigger:
#:
#:     "you want to quit"    "want to" before "quit"   -> refused
#:     "you want quit"       "want" before "quit"      -> allowed
#:
#: Reading "to" alone would break "time to quit", "ready to quit" and
#: "try to quit" while fixing "plans to quit"; Step 21 measured that,
#: and :data:`WANT_FORMS` plus the adjacency requirement is what keeps
#: the two apart.
WANT_FORMS = frozenset({"want", "wants"})

#: Words that cancel a nearby trigger.
NEGATIONS = frozenset({"not", "no", "never", "dont", "stop", "nor"})
#: How far back to look for a negation. Kept tight on purpose: every shape
#: the guard exists for ("do not play", "never play", "stop playing") puts the
#: negation one or two tokens ahead of the trigger. Widening the window to
#: catch "do not ask me about the weather" would also swallow "I have no idea,
#: tell me the news", where the "no" belongs to an earlier thought. The
#: immediate window keeps the guard from over-blocking genuine requests.
NEGATION_WINDOW = 3
#: Conjunctions that end a clause, so a negation in an earlier clause does
#: not cancel a trigger in a later one.
CLAUSE_BREAKS = frozenset(
    {"and", "but", "or", "so", "then", "also", "plus", "however", "because"}
)

#: Scores are rounded before comparison, so float noise cannot make two
#: otherwise identical candidates sort differently between runs.
SCORE_PRECISION = 4


@dataclass(frozen=True)
class Candidate:
    """One scored intent, with the evidence that produced it.

    Attributes:
        intent: the intent this candidate would activate.
        score: confidence between 0 and 1, rounded for stable comparison.
        trigger: the lexicon word that fired.
        method: which stage produced it, for example ``"exact"``.
    """

    intent: str
    score: float
    trigger: str
    method: str

    @property
    def sort_key(self) -> tuple[float, int, str, str]:
        """Ordering used by :func:`rank_candidates`.

        Higher score first, then the longer trigger, then lexical order.
        Depends on nothing but plain values, so it never varies between runs.
        """
        return (-self.score, -len(self.trigger), self.intent, self.trigger)


def _round(value: float) -> float:
    """Round a score so comparisons are stable across runs."""
    return round(float(value), SCORE_PRECISION)


def _token_index(text: str, char_pos: int) -> int:
    """Convert a character offset in ``text`` into a token index.

    ``text`` is always ``" ".join(tokens)``, so counting the separators before
    ``char_pos`` gives the index directly.
    """
    return text.count(" ", 0, char_pos)


def _match_index(entry, text: str) -> int:
    """Return the token index where ``entry`` first matches, or -1."""
    found = entry.regex.search(text)
    if found is None:
        return -1
    return _token_index(text, found.start())


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Sort candidates best first, deterministically.

    Args:
        candidates: the candidates to order.

    Returns:
        A new list ordered by score, then trigger length, then intent, then
        trigger. The input list is not modified.
    """
    return sorted(candidates, key=lambda item: item.sort_key)


def is_negated(text: str, index: int) -> bool:
    """Report whether the trigger at ``index`` sits inside a negation.

    The rule is a **local clause window**, not a whole-sentence scan. Walking
    back at most :data:`NEGATION_WINDOW` tokens covers every shape the guard
    exists for, "do not play music", "never play", "stop playing", each of
    which puts the negation one or two tokens ahead of the trigger, and the
    walk stops at a clause break so an earlier clause cannot cancel a later
    one.

    That tightness is deliberate. A whole-sentence scan would suppress a
    genuine request: in "I have no idea, tell me the news" the "no" belongs to
    a different thought, and cancelling the news request would be worse than
    missing a negation. The cost of the local window is the mirror image, a
    distant negation is missed, so in "do not ask me about the weather" the
    weather trigger still surfaces. That case needs a real scope decision
    over the sentence, which belongs to the parser rather than to a
    substring heuristic, so the trade is recorded here instead of hidden.

    Args:
        text: the normalised text.
        index: token index of the trigger.

    Returns:
        True when a negation governs this trigger.
    """
    if index <= 0:
        return False

    tokens = text.split()
    start = max(0, index - NEGATION_WINDOW)
    for position in range(index - 1, start - 1, -1):
        token = tokens[position]
        if token in CLAUSE_BREAKS:
            return False
        if token in NEGATIONS:
            return True
    return False


def _leading_bonus(index: int) -> float:
    """Structural bonus for a trigger that starts the utterance."""
    return LEADING_BONUS if index == 0 else 0.0


def is_inflected_form(token: str, pattern: str) -> bool:
    """Report whether ``token`` is ``pattern`` with only a suffix appended.

    This is the rule that separates a misspelling from an inflected word, and
    it rests on one observation about how the two are produced.

    A **typo** is damage *to the inside* of a word: a letter is substituted,
    dropped or transposed. ``wether`` is ``weather`` with ``a`` swapped for
    ``e``; ``joks`` is ``jokes`` with the final letter lost. In both cases the
    token and the trigger differ at some interior position, and the token is
    the same size as the trigger or shorter.

    An **inflection** leaves the trigger intact and adds an ending to the end.
    ``noted`` is ``note`` plus ``d``, ``played`` is ``play`` plus ``ed``,
    ``notes`` is ``note`` plus ``s``. The token *starts with* the whole
    trigger, so every letter the user said is present and correct.

    That makes ``token.startswith(pattern)`` the discriminator, and it needs
    no list of endings, no stemmer and no dictionary: it follows from where
    the difference sits. The suffix is capped at
    :data:`MAX_INFLECTION_SUFFIX` characters to keep the rule narrow, so a
    much longer word that merely happens to begin with the trigger is not
    silently discarded here; it simply falls below the fuzzy cutoff on its
    own.

    The cost is a genuine but rare typo that adds a character at the end,
    ``weathers`` for ``weather``. Losing that is the right trade: such a
    mis-key is unlikely, while ``noted`` is an ordinary English word that
    happens to contain a command.

    Args:
        token: the word heard.
        pattern: the trigger being compared.

    Returns:
        True when the token looks like an inflected form of the trigger
        rather than a misspelling of it.
    """
    if not token.startswith(pattern):
        return False
    extra = len(token) - len(pattern)
    return 0 < extra <= MAX_INFLECTION_SUFFIX


def _is_want_frame_about_you(tokens: list[str], index: int) -> bool:
    """True when a want frame about the assistant governs this trigger.

    The structure, measured in Step 21 and quoted in
    :data:`WANT_FORMS`, is::

        ... you ... want to <trigger>

    Two marks are required and either alone is refused as evidence:

    * a want form, somewhere in the words before the trigger, and
    * the token immediately before the trigger is ``to``,

    together with ``you`` somewhere before the trigger. The second mark
    is what separates a want *frame* from the ordinary imperative, so
    "you want quit" is still allowed and only "you want to quit" is
    not.

    ``index`` is the **start** of the matched trigger, so a two-word
    trigger such as "shut down" is measured against the words before
    "shut", and "do you want to shut down" is refused.
    """
    if index <= 1 or tokens[index - 1] != "to":
        return False
    before = tokens[:index]
    return "you" in before and any(token in WANT_FORMS for token in before)


def _passes_system_guard(text: str, candidate: Candidate, index: int) -> bool:
    """Extra safety for the exit intent.

    Five conditions, all required. The score must be near perfect, which
    rules out every fuzzy and containment candidate; no word naming the
    speaker or a third party may appear before the trigger; no
    definition word may sit immediately before a single-word trigger; no
    want frame about the assistant may govern the trigger; and the
    trigger must sit in the closing words of the utterance.

    The **subject** condition is the one that stops a command word being read
    as an order. The window alone was sized against a single leading case,
    ``"goodbye is in the dictionary"``, and it cannot tell a trigger that
    genuinely closes the sentence from one that merely sits near the end:

        "i should quit smoking"    4 tokens, "quit" at index 2
        "i want to exit early"     4 tokens, "exit" at index 3

    Both satisfied the window and both terminated the assistant, because
    the trigger was followed by an object, not because anyone was being
    told to quit. Requiring that nobody is named before the trigger is the
    structural distinction between an order and a statement, and it is why
    the two conditions are kept together: the window still blocks the
    leading mentions it was written for, and the subject check blocks the
    trailing ones it was not.

    The **apposition** condition catches a shape neither of the others can
    see, because it names nobody at all:

        "what does quit mean"          "does" before "quit"
        "what is the meaning of quit"   "of"   before "quit"
        "define exit"                   "define" before "exit"

    ``index`` is the start of the matched trigger, so a multi-word
    trigger is measured from its first word. That is what keeps "shut
    down" safe: the neighbours of ``shut`` and of ``down`` are different,
    and only the former is consulted.

    The **want frame** condition closes the last family Step 21 found.
    It is placed after the subject rule rather than instead of it, so
    "I want to quit smoking" and "do you want me to quit" are still
    refused by the rule that was written for them, and the new
    condition only ever handles the case where "you" is the subject.

    One known cost, accepted rather than engineered away: "if you want
    to quit now" is a genuine order and is refused, because the
    condition cannot tell a conditional wrapper from a question without
    parsing the clause. Refusing it costs the user one retry; allowing
    it would cost a wrong answer, so the asymmetry is deliberate.
    """
    if candidate.intent != SYSTEM_INTENT:
        return True
    if candidate.score < SYSTEM_MIN_SCORE:
        return False

    tokens = text.split()
    if index > 0 and any(token in SELF_OTHER_SUBJECTS for token in tokens[:index]):
        return False

    # A definition or an explanation puts the trigger in apposition.
    # Multi-word triggers are exempt, so "please shut down" is untouched.
    if " " not in candidate.trigger and index > 0:
        if tokens[index - 1] in DEFINITION_MARKERS:
            return False

    # A want frame about the assistant: "do you want to quit". "you" is
    # the one pronoun the subject rule cannot use, so a question about
    # whether the assistant should stop is refused here instead.
    if _is_want_frame_about_you(tokens, index):
        return False

    tail_start = max(0, len(tokens) - SYSTEM_TAIL_TOKENS)
    return index >= tail_start


def _exact_candidates(
    normalized: Normalized, lexicon: Lexicon
) -> list[Candidate]:
    """Stages 1 and 2: boundary-exact triggers, split by shape.

    A single-word trigger is unambiguous and scores :data:`EXACT_SCORE`. A
    multi-word phrase is nearly as certain and scores :data:`PHRASE_SCORE`;
    the split keeps the two stages from overlapping, so the phrase stage
    still has work to do.
    """
    text = normalized.text
    if not text:
        return []

    found: list[Candidate] = []
    for entry in lexicon.exact_matches(text):
        index = _match_index(entry, text)
        if index < 0 or is_negated(text, index):
            continue
        is_phrase = " " in entry.pattern
        base = PHRASE_SCORE if is_phrase else EXACT_SCORE
        found.append(
            Candidate(
                intent=entry.intent,
                score=_round(min(MAX_SCORE, base + _leading_bonus(index))),
                trigger=entry.pattern,
                method="phrase" if is_phrase else "exact",
            )
        )
    return found


def _fuzzy_candidates(
    normalized: Normalized, lexicon: Lexicon
) -> list[Candidate]:
    """Stage 3: near-miss tokens, compared with :class:`SequenceMatcher`.

    ``difflib`` is used directly rather than through
    :func:`difflib.get_close_matches`, because the formula needs the raw
    ratio, and because that helper takes ``cutoff`` as its fourth positional
    argument, which is easy to pass in the wrong place.

    Four guards matter here. A token identical to a trigger is skipped,
    because that is stage 1's job and a self-match would score a perfect
    1.00 for a word the boundary matcher already rejected. An intent is
    skipped entirely when the token is an inflected form of any of its
    triggers, so "noted" is not read as a misspelling of "note" or "notes".
    The negation guard is applied, since a fuzzy hit can otherwise smuggle a
    cancelled command through. And one winner per intent is kept, so the
    result cannot depend on iteration order.
    """
    if not normalized.tokens:
        return []

    triggers = {entry.pattern for entry in lexicon.entries}
    patterns_by_intent: dict[str, set[str]] = {}
    for entry in lexicon.entries:
        patterns_by_intent.setdefault(entry.intent, set()).add(entry.pattern)
    best: dict[str, Candidate] = {}

    for index, token in enumerate(normalized.tokens):
        if len(token) < FUZZY_MIN_TOKEN or token in triggers:
            continue
        if is_negated(normalized.text, index):
            continue
        # The inflection guard is applied per intent, not per pair. An intent
        # often carries several word forms: "note" and "notes" both belong to
        # the notes tool. Checking pairs one at a time lets the plural slip
        # through, because "noted" extends "note" so that pair is refused,
        # yet "noted" is one letter from "notes" and fuzzy-matches it. Once
        # the token is recognised as an inflected form of any word belonging
        # to an intent, it cannot be a misspelling of any other word in that
        # same intent, so the whole intent is skipped.
        blocked = {
            intent
            for intent, patterns in patterns_by_intent.items()
            if any(is_inflected_form(token, pattern) for pattern in patterns)
        }

        for entry in lexicon.entries:
            if entry.intent in blocked:
                continue
            # No lower length pre-filter: a typo is usually *shorter* than
            # the word it mistypes, and the length penalty already discounts
            # that case. The upper bound is the one that matters, keeping a
            # longer unrelated word such as "replay" from being read as a
            # misspelling of "play".
            if len(token) > len(entry.pattern) * FUZZY_LENGTH_TOLERANCE:
                continue
            ratio = SequenceMatcher(None, token, entry.pattern).ratio()
            if ratio < FUZZY_CUTOFF:
                continue
            penalty = min(
                1.0, (len(token) / len(entry.pattern)) * LENGTH_PENALTY_FACTOR
            )
            score = _round(ratio * penalty)
            if score < FUZZY_CUTOFF:
                continue
            current = best.get(entry.intent)
            if current is None or score > current.score:
                best[entry.intent] = Candidate(
                    intent=entry.intent,
                    score=score,
                    trigger=entry.pattern,
                    method="fuzzy",
                )
    return list(best.values())


def _containment_candidates(
    normalized: Normalized, lexicon: Lexicon
) -> list[Candidate]:
    """Stage 4: the legacy substring behaviour, as a last resort.

    This reproduces the original router's containment rule on purpose, so the
    NLU can stand in for it without losing commands it used to accept. It
    scores lowest, so a boundary-aware candidate always wins when one exists.

    The negation guard applies here too. Without it a cancelled command
    would be suppressed by the exact and fuzzy stages and then quietly
    reinstated by the legacy one, which is precisely the bug the guard
    exists to stop.
    """
    text = normalized.text
    if not text:
        return []

    found: list[Candidate] = []
    for entry in lexicon.entries:
        if not entry.pattern or entry.pattern not in text:
            continue
        if is_negated(text, _token_index(text, text.find(entry.pattern))):
            continue
        found.append(
            Candidate(
                intent=entry.intent,
                score=CONTAINMENT_SCORE,
                trigger=entry.pattern,
                method="containment",
            )
        )
    return found


def _index_of(text: str, lexicon: Lexicon, candidate: Candidate) -> int:
    """Find where a candidate's trigger sits, or -1 when it is not located."""
    for entry in lexicon.entries:
        if entry.intent == candidate.intent and entry.pattern == candidate.trigger:
            return _match_index(entry, text)
    return -1


def score_intents(
    normalized: Normalized, lexicon: Lexicon
) -> list[Candidate]:
    """Score every intent the utterance might mean, best first.

    Runs the stages in order and keeps only the strongest candidate per
    intent, so a later stage can never lower a score an earlier one earned.

    Args:
        normalized: the utterance from
            :func:`~assistant.nlu.normalize.normalize`.
        lexicon: the vocabulary to match against.

    Returns:
        Candidates ordered by :func:`rank_candidates`. Empty when nothing
        matched. The ``system`` guard is applied last, to every candidate.
    """
    strongest: dict[str, Candidate] = {}

    for batch in (
        _exact_candidates(normalized, lexicon),
        _fuzzy_candidates(normalized, lexicon),
        _containment_candidates(normalized, lexicon),
    ):
        for candidate in batch:
            current = strongest.get(candidate.intent)
            if current is None or candidate.score > current.score:
                strongest[candidate.intent] = candidate

    text = normalized.text
    kept = [
        candidate
        for candidate in strongest.values()
        if _passes_system_guard(
            text, candidate, _index_of(text, lexicon, candidate)
        )
    ]
    return rank_candidates(kept)
