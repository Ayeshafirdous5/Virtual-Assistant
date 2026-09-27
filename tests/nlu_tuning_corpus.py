"""The measurement corpus for the NLU tuning pass.

This module holds **data only**: utterances and the intent each one should
produce. It contains no test functions and imports nothing from the
application, so the evaluation in ``test_nlu_tuning.py`` can sweep
thresholds against it repeatedly and cheaply.

Every entry states its intent explicitly, and the three outcomes the NLU
can produce are all representable:

* a tool name such as ``"weather"``, which must resolve clearly;
* :data:`AMBIGUOUS`, which must be recognised as a genuine choice;
* :data:`NO_MATCH`, which must resolve to no tool at all.

Labels record what a user **reasonably meant**, not what the current code
happens to do. Where the two disagree, the entry is a genuine finding and
is left labelled honestly rather than relabelled to make a number look
better.

Group A is imported from the backward-compatibility corpus instead of
being copied, so the two can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.test_nlu_backward_compat import CORPUS as BACKWARD_COMPAT_CORPUS

#: Expected outcome: the utterance is a genuine choice between candidates
#: and must be reported as ambiguous rather than silently decided.
AMBIGUOUS = "ambiguous"

#: Expected outcome: no tool should run.
NO_MATCH = None

#: Group labels, used to break a result down by kind of case.
GROUP_KNOWN_GOOD = "A-known-good"
GROUP_SAFETY = "B-safety"
GROUP_NATURAL = "C-natural"
GROUP_NEAR_TIE = "D-near-tie"


@dataclass(frozen=True)
class Case:
    """One utterance and what it should produce.

    Attributes:
        utterance: what the user says.
        expected: the intent it should resolve to, :data:`AMBIGUOUS`, or
            :data:`NO_MATCH`.
        group: which part of the corpus this case belongs to.
        note: why the label was chosen, where it is not obvious.
    """

    utterance: str
    expected: str | None
    group: str
    note: str = ""


def _known_good() -> list[Case]:
    """Flatten the 25-command backward-compatibility corpus into cases."""
    return [
        Case(utterance, intent, GROUP_KNOWN_GOOD, "must match the legacy router")
        for intent, utterances in BACKWARD_COMPAT_CORPUS.items()
        for utterance in utterances
    ]


#: Sentences that must **not** reach a tool. Each is a near miss on
#: purpose: a command word hidden inside a longer word, a cancelled
#: request, or an ordinary sentence that merely mentions the thing.
SAFETY_CASES: list[Case] = [
    Case("i noted that down", NO_MATCH, GROUP_SAFETY, "inflected, not a typo"),
    Case("i noted 3 things down", NO_MATCH, GROUP_SAFETY, "number words too"),
    Case("denote this", NO_MATCH, GROUP_SAFETY, "note inside denote"),
    Case("replay my song", NO_MATCH, GROUP_SAFETY, "play inside replay"),
    Case("playback speed", NO_MATCH, GROUP_SAFETY, "play inside playback"),
    Case("play the song back", "youtube", GROUP_SAFETY,
         "relabelled: a real 'play it in reverse' request, not a near miss"),
    Case("goodbye is in the dictionary", NO_MATCH, GROUP_SAFETY, "not closing"),
    Case("quite good", NO_MATCH, GROUP_SAFETY, "quit inside quite"),
    Case("I don't want to exit", NO_MATCH, GROUP_SAFETY, "negated exit"),
    Case("never play music", NO_MATCH, GROUP_SAFETY, "negated play"),
    Case("do not play music", NO_MATCH, GROUP_SAFETY, "negated play"),
    Case("the word quit is in that sentence", NO_MATCH, GROUP_SAFETY, "quoted word"),
    Case("I said goodbye to my friend", NO_MATCH, GROUP_SAFETY, "mentions goodbye"),
    Case("the train will stop here", NO_MATCH, GROUP_SAFETY, "mentions stop"),
    Case("this video is funny", NO_MATCH, GROUP_SAFETY, "mentions video"),
    Case("listen to some music", NO_MATCH, GROUP_SAFETY, "mentions music"),
    Case("check the log", NO_MATCH, GROUP_SAFETY, "mentions log"),
    Case("nwes", NO_MATCH, GROUP_SAFETY, "0.75 similarity, below FUZZY_CUTOFF"),
    # Positives that must survive the same guards.
    Case("play lofi beats", "youtube", GROUP_SAFETY, "counterpart of replay"),
    Case("note buy milk", "notes", GROUP_SAFETY, "counterpart of noted"),
    Case("notes buy milk", "notes", GROUP_SAFETY, "plural trigger is real"),
    Case("exit", "system", GROUP_SAFETY, "counterpart of the refusals"),
]

#: Ordinary sentences. These are what a person would actually say, and
#: they are where a threshold change would show up first: a lower action
#: floor would turn a mention into a command, a wider ambiguity margin
#: would turn a decision into a question.
NATURAL_CASES: list[Case] = [
    # --- Weather -----------------------------------------------------
    Case("could you tell me the weather", "weather", GROUP_NATURAL),
    Case("what's the temperature today", "weather", GROUP_NATURAL,
         "temperature is a registered weather pattern"),
    Case("how is the weather outside", "weather", GROUP_NATURAL),
    Case("weather forecast please", "weather", GROUP_NATURAL,
         "two weather triggers, one intent, so still clear"),
    # --- News --------------------------------------------------------
    Case("what's happening in the news", "news", GROUP_NATURAL),
    Case("tell me today's news", "news", GROUP_NATURAL),
    Case("give me the latest headlines", "news", GROUP_NATURAL),
    Case("I want to hear the news", "news", GROUP_NATURAL),
    # --- Facts -------------------------------------------------------
    Case("give me a random fact", "facts", GROUP_NATURAL),
    Case("tell me an interesting fact", "facts", GROUP_NATURAL),
    # --- Jokes -------------------------------------------------------
    Case("make me laugh", "jokes", GROUP_NATURAL, "approved alias"),
    Case("do you know any good jokes", "jokes", GROUP_NATURAL),
    Case("I need a joke", "jokes", GROUP_NATURAL),
    Case("tell me something funny", NO_MATCH, GROUP_NATURAL,
         "no joke synonym in the lexicon; recorded as a gap"),
    # --- History -----------------------------------------------------
    Case("what did I ask earlier", "history", GROUP_NATURAL,
         "contains the what did i ask pattern"),
    Case("can you read my history", "history", GROUP_NATURAL),
    Case("show me my previous commands", NO_MATCH, GROUP_NATURAL,
         "no history synonym matches; recorded as a gap"),
    Case("what have I asked you", NO_MATCH, GROUP_NATURAL,
         "no history synonym matches; recorded as a gap"),
    # --- Notes -------------------------------------------------------
    Case("save this as a note", "notes", GROUP_NATURAL),
    Case("add a note saying buy groceries", "notes", GROUP_NATURAL),
    Case("show my notes", "notes", GROUP_NATURAL),
    Case("read my saved notes", "notes", GROUP_NATURAL),
    Case("remember that I need milk", NO_MATCH, GROUP_NATURAL,
         "remember that is slot syntax, not a trigger; recorded as a gap"),
    # --- Information -------------------------------------------------
    Case("tell me about Python", "information", GROUP_NATURAL, "approved alias"),
    Case("give me information about databases", "information", GROUP_NATURAL),
    Case("I want information about Hyderabad", "information", GROUP_NATURAL),
    Case("what do you know about machine learning", NO_MATCH, GROUP_NATURAL,
         "no information trigger; recorded as a gap"),
    Case("I want to log this information", NO_MATCH, GROUP_NATURAL,
         "a meta-mention of a command word; labelled as the user meant it"),
    # --- YouTube -----------------------------------------------------
    Case("play some relaxing music", "youtube", GROUP_NATURAL, "play is the trigger"),
    Case("play a video for me", "youtube", GROUP_NATURAL),
    Case("play lofi beats", "youtube", GROUP_NATURAL),
    Case("I want to watch something", NO_MATCH, GROUP_NATURAL,
         "no youtube trigger; recorded as a gap"),
    # --- System ------------------------------------------------------
    Case("quit the assistant", "system", GROUP_NATURAL, "quit is a pattern"),
    Case("stop the assistant", "system", GROUP_NATURAL, "approved alias"),
    Case("please exit", "system", GROUP_NATURAL, "closing position"),
    # --- Genuinely unknown -------------------------------------------
    Case("what should I cook tonight", NO_MATCH, GROUP_NATURAL),
    Case("my laptop is running slowly", NO_MATCH, GROUP_NATURAL),
    Case("I need to finish my assignment", NO_MATCH, GROUP_NATURAL),
    Case("tell me something interesting about space", NO_MATCH, GROUP_NATURAL),
    Case("the weather is nice today I guess", NO_MATCH, GROUP_NATURAL,
         "a statement, not a request for a report"),
]

#: The one case the current design deliberately leaves undecided.
AMBIGUITY_CASES: list[Case] = [
    Case("information about the news", AMBIGUOUS, GROUP_NEAR_TIE,
         "both intents score 1.000; must not be silently decided"),
]

#: Near-ties, added after the first sweep showed the corpus could not tell
#: the ambiguity margin apart from any other value. Each one sits on a
#: different side of the current margin, which is exactly what makes the
#: margin measurable at all.
#:
#: An exact match scores 1.000 and a phrase match scores 0.950, so the two
#: are always exactly 0.050 apart. That is the only near-tie the design can
#: produce structurally, and it is the one the margin exists to catch.
NEAR_TIE_AMBIGUOUS: list[Case] = [
    Case("note what did i ask", AMBIGUOUS, GROUP_NEAR_TIE,
         "notes 1.000 exact vs history 0.950 phrase, gap 0.050"),
    Case("joke what did i ask", AMBIGUOUS, GROUP_NEAR_TIE,
         "jokes 1.000 exact vs history 0.950 phrase, gap 0.050"),
]

#: An exact match against a fuzzy near-miss. The smallest reachable
#: non-zero gap is 0.048, from a transposition such as "temprature",
#: which sits *below* the current margin. Recorded so the trade-off is
#: visible: the margin cannot be lowered to catch this without losing the
#: phrase ties above.
NEAR_TIE_CLEAR: list[Case] = [
    Case("joke about temprature", "jokes", GROUP_NEAR_TIE,
         "jokes 1.000 exact vs weather 0.952 fuzzy, gap 0.048"),
    Case("news about wether", "news", GROUP_NEAR_TIE,
         "news 1.000 exact vs weather 0.923 fuzzy, gap 0.077"),
]

#: Every case, in a fixed order, so a report always reads the same way.
#:
#: Duplicates are dropped, keeping the first occurrence. A few sentences
#: appear in both the backward-compatibility corpus and the safety group,
#: and counting them twice would give them double weight in the metrics and
#: distort any comparison between threshold values.
def _dedupe(cases: list[Case]) -> list[Case]:
    seen: set[str] = set()
    unique: list[Case] = []
    for case in cases:
        if case.utterance in seen:
            continue
        seen.add(case.utterance)
        unique.append(case)
    return unique


CORPUS: list[Case] = _dedupe(
    _known_good()
    + SAFETY_CASES
    + NATURAL_CASES
    + AMBIGUITY_CASES
    + NEAR_TIE_AMBIGUOUS
    + NEAR_TIE_CLEAR
)

#: Labels that mean "run no tool" or "ask", for readability in reports.
NO_MATCH_LABEL = "<no-match>"
AMBIGUOUS_LABEL = "<ambiguous>"


def label(expected: str | None) -> str:
    """Render an expected value for a report."""
    if expected is AMBIGUOUS:
        return AMBIGUOUS_LABEL
    if expected is NO_MATCH:
        return NO_MATCH_LABEL
    return expected
