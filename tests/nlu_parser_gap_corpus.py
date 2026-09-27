"""A consolidated corpus of parser-level gaps.

This step is **measurement and diagnosis only**. No production code is
changed, and nothing here proposes an edit.

Why this corpus exists
----------------------
Eleven framing steps measured the framing layer and kept hitting the
same wall from the other side. Two of those corpora still report cases
the **parser** never routes to framing at all:

    tests/nlu_framing_verb_forms_corpus   5 parser no-matches
    tests.nlu_framing_telling_corpus       5 parser no-matches

Five of those overlap, so there are **six distinct sentences**. They
were never framing bugs and never could have been: the parser returns
``None`` before framing is consulted, so no framing rule could have
reached them. This corpus gathers those six, adds cases beside them for
contrast, and records what the parser actually does with each one.

What the six have in common
---------------------------
Nothing to do with grammar. Every one of them is missing a **trigger
word**. "I liked you telling that story" fails because no word in it
names an action or a topic the lexicon knows: not "telling", and not
"story". The two control cases in :data:`VERB_FORM_GAPS` show this
directly, because the same verb form with a real trigger in the sentence
does parse:

    "I liked you telling that story"   -> no candidate at all
    "I heard you telling a joke"       -> jokes 1.00, exact

So "telling" and "tells" are not the problem. **A sentence that names
nothing the assistant can act on is the problem**, and that is by
design.

The distinction this corpus is built around
-------------------------------------------
A parser gap is not a defect until you know which of three things it is:

* a **missing alias**, where a real synonym is simply absent;
* **intentional safety behaviour**, where a candidate was produced and
  then deliberately held below the action boundary;
* **genuinely unsupported wording**, where nothing scored at all and
  adding support would mean guessing at what the user meant.

Only the first is a candidate for improvement, and even then only if it
does not raise false positives. :data:`SAFE_TO_IMPROVE` and
:data:`SHOULD_STAY_UNSUPPORTED` record that judgement as data, so it can
be argued with rather than re-derived.

Framing is a downstream layer
-----------------------------
``framing`` is consulted **only after** a clear parse. A ``NO_MATCH`` or
an ``AMBIGUOUS`` result never reaches it. :attr:`ParserGapCase.framing_relevant`
records that, because "the framing verdict was neutral" and "framing was
never asked" are completely different findings and have been confused
before.

Labelling
---------
Labels record what a human would reasonably expect the **parser** to
return, never what it currently returns. Nothing here is tuned to make
the suite green.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The parser returned an intent, confidently.
MATCH = "match"

#: The parser returned an intent, but two were within the margin.
AMBIGUOUS = "ambiguous"

#: The parser returned nothing. Framing is never consulted.
NO_MATCH = "no-match"

#: Inflected and narrative verb forms that name no action.
CATEGORY_VERB_FORM = "verb-form-gap"

#: A person plainly asking for a tool, in wording the lexicon lacks.
CATEGORY_PHRASING = "natural-phrasing-gap"

#: The tool exists; the words for it are simply not in the vocabulary.
CATEGORY_ALIAS = "alias-gap"

#: Ordinary conversation. These must keep returning nothing.
CATEGORY_NON_COMMAND = "non-command"

#: Questions that must keep resolving, and must not be broken later.
CATEGORY_QUESTION = "question"

#: A real command sitting behind a narrative clause.
CATEGORY_EMBEDDED = "embedded-request"

#: Near-misses the fuzzy safeguards are designed to refuse.
CATEGORY_FUZZY = "fuzzy-boundary"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_VERB_FORM,
    CATEGORY_PHRASING,
    CATEGORY_ALIAS,
    CATEGORY_NON_COMMAND,
    CATEGORY_QUESTION,
    CATEGORY_EMBEDDED,
    CATEGORY_FUZZY,
)

#: Cause labels, assigned from the parser's own evidence rather than
#: guessed. :data:`CAUSES` documents what each one means.
CAUSE_NONE = "correct-behaviour"
CAUSE_MISSING_ALIAS = "missing-lexicon-alias"
CAUSE_CONTAINMENT_FLOOR = "intentional-safety-lower-containment-below-action-boundary"
CAUSE_NO_TRIGGER = "genuinely-unsupported-wording"
CAUSE_TYPO_APPEND = "fuzzy-length-tolerance-rejects-appended-character"

CAUSES: dict[str, str] = {
    CAUSE_NONE: "The parser returned what a person would expect.",
    CAUSE_MISSING_ALIAS: (
        "The tool exists and the sentence is a clear request for it, but "
        "the wording is not in the lexicon. A synonym would fix it."
    ),
    CAUSE_CONTAINMENT_FLOOR: (
        "A candidate was produced by the containment stage and then held "
        "below the action boundary on purpose. This is the safeguard "
        "working, not a gap."
    ),
    CAUSE_NO_TRIGGER: (
        "Nothing scored at all: no word in the sentence names an action or "
        "a topic the lexicon knows. Supporting it would mean guessing."
    ),
    CAUSE_TYPO_APPEND: (
        "The token is longer than any plausible typo of the trigger, so "
        "the fuzzy stage declines it. Documented in the scoring module."
    ),
}

#: Causes that are safe to address without guessing at user intent.
SAFE_TO_IMPROVE: frozenset[str] = frozenset({CAUSE_MISSING_ALIAS})

#: Causes that must keep returning nothing, because addressing them would
#: mean guessing or would undo a safety safeguard.
SHOULD_STAY_UNSUPPORTED: frozenset[str] = frozenset(
    {CAUSE_NO_TRIGGER, CAUSE_CONTAINMENT_FLOOR, CAUSE_TYPO_APPEND}
)


@dataclass(frozen=True)
class ParserGapCase:
    """One utterance and what the parser is expected to make of it.

    Attributes:
        utterance: what the user says.
        intent: the intent a clear match should name, or the tool the
            sentence is *about* when the expected outcome is a no-match.
        expected: :data:`MATCH`, :data:`AMBIGUOUS`, or :data:`NO_MATCH`.
        category: which group this case belongs to.
        cause: the diagnosis, from :data:`CAUSES`.
        note: why the label was chosen, and how close the call is.
        source: the corpus this case came from, or ``""`` when it is new.
    """

    utterance: str
    intent: str
    expected: str
    category: str
    cause: str
    note: str = ""
    source: str = ""

    @property
    def framing_relevant(self) -> bool:
        """True when framing is consulted at all for this outcome.

        Framing runs only after a **clear** parse. A no-match returns
        before it, and so does an ambiguous result, which asks the user
        instead. This is the single most important column in the corpus:
        it is what stops a parser gap from being counted as a framing one.
        """
        return self.expected == MATCH

    @property
    def is_gap(self) -> bool:
        """True when the parser does not do what this case expects."""
        return self.cause != CAUSE_NONE

    def describe(self) -> str:
        return (
            f"{self.utterance!r} -> {self.expected} {self.intent} "
            f"({self.category}; {self.cause})"
        )


# ----------------------------------------------------------------------
# A. Verb-form gaps. The eight sentences the framing corpora recorded as
#    parser no-matches, plus two controls that prove the verb form is
#    not the cause.
# ----------------------------------------------------------------------
VERB_FORM_GAPS: list[ParserGapCase] = [
    ParserGapCase("they kept telling me the same thing", "news", NO_MATCH,
                  CATEGORY_VERB_FORM, CAUSE_NO_TRIGGER,
                  "no topic word; 'the same thing' names nothing",
                  "verb-form corpus / telling corpus"),
    ParserGapCase("I liked you telling that story", "jokes", NO_MATCH,
                  CATEGORY_VERB_FORM, CAUSE_NO_TRIGGER,
                  "'story' is not a trigger; the jokes trigger is 'joke'",
                  "verb-form corpus / telling corpus"),
    ParserGapCase("keep telling me more", "jokes", NO_MATCH,
                  CATEGORY_VERB_FORM, CAUSE_NO_TRIGGER,
                  "the clearest genuine request in the whole study, and it "
                  "names no topic at all",
                  "verb-form corpus / telling corpus"),
    ParserGapCase("stop telling me that", "news", NO_MATCH,
                  CATEGORY_VERB_FORM, CAUSE_NO_TRIGGER,
                  "'that' names nothing; the negation guard does not apply",
                  "verb-form corpus / telling corpus"),
    ParserGapCase("I remember you telling me that, explain it again",
                  "information", NO_MATCH, CATEGORY_VERB_FORM, CAUSE_NO_TRIGGER,
                  "'explain' is not a trigger and 'that' is not a topic",
                  "verb-form corpus / telling corpus"),
    ParserGapCase("continue telling me the story", "jokes", NO_MATCH,
                  CATEGORY_VERB_FORM, CAUSE_NO_TRIGGER,
                  "'continue' is not a trigger and 'story' is not a topic",
                  "telling corpus"),
    # Controls. Same verb form, but a real trigger is present, so these
    # parse. This is the evidence that the verb form is not the problem.
    ParserGapCase("my friend tells me jokes", "jokes", MATCH,
                  CATEGORY_VERB_FORM, CAUSE_NONE,
                  "control: 'jokes' is an exact trigger, so it matches"),
    ParserGapCase("I heard you telling a joke", "jokes", MATCH,
                  CATEGORY_VERB_FORM, CAUSE_NONE,
                  "control: 'joke' is an exact trigger, so it matches"),
]

# ----------------------------------------------------------------------
# B. Natural phrasing. A person plainly asking for a tool, in words the
#    lexicon does not carry.
# ----------------------------------------------------------------------
PHRASING_GAPS: list[ParserGapCase] = [
    ParserGapCase("is it going to rain", "weather", NO_MATCH,
                  CATEGORY_PHRASING, CAUSE_MISSING_ALIAS,
                  "a natural weather question; no rain or forecast wording"),
    ParserGapCase("will it rain tomorrow", "weather", NO_MATCH,
                  CATEGORY_PHRASING, CAUSE_MISSING_ALIAS,
                  "same gap, phrased with a modal"),
    ParserGapCase("do I need an umbrella", "weather", NO_MATCH,
                  CATEGORY_PHRASING, CAUSE_NO_TRIGGER,
                  "an indirect request; nothing names the weather"),
    ParserGapCase("something funny", "jokes", NO_MATCH,
                  CATEGORY_PHRASING, CAUSE_MISSING_ALIAS,
                  "'funny' is not carried; the alias table has 'laugh'"),
    ParserGapCase("amuse me", "jokes", NO_MATCH,
                  CATEGORY_PHRASING, CAUSE_MISSING_ALIAS,
                  "a synonym for the joke tool that is not in the table"),
    ParserGapCase("put on a song", "youtube", NO_MATCH,
                  CATEGORY_PHRASING, CAUSE_MISSING_ALIAS,
                  "'put on' is a common way to ask for music"),
    ParserGapCase("power off", "system", NO_MATCH,
                  CATEGORY_PHRASING, CAUSE_MISSING_ALIAS,
                  "high-stakes intent; the guard also holds system high"),
]


# ----------------------------------------------------------------------
# C. Vocabulary and alias gaps. The tool exists and is reachable; the
#    words for it are simply not in the lexicon.
# ----------------------------------------------------------------------
ALIAS_GAPS: list[ParserGapCase] = [
    ParserGapCase("celsius", "weather", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "a unit, not a weather trigger"),
    ParserGapCase("degrees outside", "weather", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "a natural way to ask for weather"),
    ParserGapCase("wind speed", "weather", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "a specific field nobody can ask for"),
    ParserGapCase("gossip", "news", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "light news, but not a known word"),
    ParserGapCase("trivia", "facts", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "the facts tool answers 'random fact'"),
    ParserGapCase("did you know", "facts", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "a natural opener; 'did you' is not "
                  "carried because it also opens every other question"),
    ParserGapCase("google python", "information", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "brand name instead of 'look up'"),
    ParserGapCase("define gravity", "information", NO_MATCH, CATEGORY_ALIAS,
                  CAUSE_MISSING_ALIAS, "'define' is a natural lookup verb"),
    ParserGapCase("search the web for python", "information", NO_MATCH,
                  CATEGORY_ALIAS, CAUSE_MISSING_ALIAS,
                  "'search' alone is a cue, but not this phrase"),
]

# ----------------------------------------------------------------------
# D. Ordinary conversation. These must keep returning nothing, and a
#    future parser change must not make them match.
# ----------------------------------------------------------------------
NON_COMMANDS: list[ParserGapCase] = [
    ParserGapCase("she was telling me the story", "jokes", NO_MATCH,
                  CATEGORY_NON_COMMAND, CAUSE_NO_TRIGGER,
                  "a narrative sentence with no topic word"),
    ParserGapCase("we discussed it yesterday", "news", NO_MATCH,
                  CATEGORY_NON_COMMAND, CAUSE_NO_TRIGGER,
                  "'it' names nothing, which is the point"),
    ParserGapCase("that was a strange thing", "facts", NO_MATCH,
                  CATEGORY_NON_COMMAND, CAUSE_NO_TRIGGER,
                  "a remark about an unspecified thing"),
    ParserGapCase("good morning", "facts", NO_MATCH,
                  CATEGORY_NON_COMMAND, CAUSE_NONE,
                  "a greeting; nothing should be looked up"),
    ParserGapCase("thanks", "news", NO_MATCH, CATEGORY_NON_COMMAND,
                  CAUSE_NONE, "politeness, not a command"),
    ParserGapCase("never mind", "news", NO_MATCH, CATEGORY_NON_COMMAND,
                  CAUSE_NONE, "a retraction, not a command"),
]


# ----------------------------------------------------------------------
# E. Questions. These must keep resolving, and must not be broken by
#    whatever the other groups turn into.
# ----------------------------------------------------------------------
QUESTIONS: list[ParserGapCase] = [
    ParserGapCase("what is the weather", "weather", MATCH, CATEGORY_QUESTION,
                  CAUSE_NONE, "the canonical weather question"),
    ParserGapCase("did you hear the news", "news", MATCH, CATEGORY_QUESTION,
                  CAUSE_NONE, "a question that must never be blocked"),
    ParserGapCase("can you tell me a joke", "jokes", MATCH, CATEGORY_QUESTION,
                  CAUSE_NONE, "asking the assistant to act"),
    ParserGapCase("what did I say", "history", MATCH, CATEGORY_QUESTION,
                  CAUSE_NONE, "the history alias, matched as a phrase"),
    ParserGapCase("who told you the news", "news", MATCH, CATEGORY_QUESTION,
                  CAUSE_NONE, "a question about a past event"),
    ParserGapCase("make me laugh again", "jokes", MATCH, CATEGORY_QUESTION,
                  CAUSE_NONE, "a phrase match, and a real request"),
]

# ----------------------------------------------------------------------
# F. A real command behind a narrative clause. The parser is not the
#    problem here; framing is, and framing has its own corpora.
# ----------------------------------------------------------------------
EMBEDDED_REQUESTS: list[ParserGapCase] = [
    ParserGapCase("we talked about the weather, check it", "weather", MATCH,
                  CATEGORY_EMBEDDED, CAUSE_NONE,
                  "the parser finds the trigger; framing lets the request win"),
    ParserGapCase("she told me a joke, tell me another", "jokes", MATCH,
                  CATEGORY_EMBEDDED, CAUSE_NONE,
                  "'tell' is a request cue and a clear match"),
    ParserGapCase("I heard you tell a joke, tell me another one", "jokes",
                  MATCH, CATEGORY_EMBEDDED, CAUSE_NONE,
                  "covered by the Step 7 reported-speech guard"),
    ParserGapCase("my friend told me a joke, tell me a better one", "jokes",
                  MATCH, CATEGORY_EMBEDDED, CAUSE_NONE,
                  "'tell' must win over 'told'"),
    ParserGapCase("I heard the news, tell me another joke", "jokes", AMBIGUOUS,
                  CATEGORY_EMBEDDED, CAUSE_NONE,
                  "news and jokes both score 1.00, so the parser asks "
                  "instead of guessing; framing is never consulted for an "
                  "ambiguous result, so this is correct behaviour"),
]

# ----------------------------------------------------------------------
# G. The fuzzy boundary. These must stay no-match, because the fuzzy
#    safeguards are deliberately refusing them.
# ----------------------------------------------------------------------
FUZZY_BOUNDARY: list[ParserGapCase] = [
    ParserGapCase("newss", "news", NO_MATCH, CATEGORY_FUZZY,
                  CAUSE_TYPO_APPEND, "an appended character, not a typo"),
    ParserGapCase("joke1", "jokes", NO_MATCH, CATEGORY_FUZZY,
                  CAUSE_CONTAINMENT_FLOOR,
                  "a digit appended; only containment scored it"),
    ParserGapCase("notess", "notes", NO_MATCH, CATEGORY_FUZZY,
                  CAUSE_TYPO_APPEND, "an appended character, not a typo"),
    ParserGapCase("replay the video", "youtube", NO_MATCH, CATEGORY_FUZZY,
                  CAUSE_CONTAINMENT_FLOOR,
                  "containment found 'play' inside 'replay' and was held "
                  "below the action boundary on purpose"),
    ParserGapCase("denote", "notes", NO_MATCH, CATEGORY_FUZZY,
                  CAUSE_CONTAINMENT_FLOOR,
                  "the substring case the app module calls out explicitly"),
    ParserGapCase("quite good", "news", NO_MATCH, CATEGORY_FUZZY,
                  CAUSE_NO_TRIGGER, "no trigger and nothing to contain"),
]

#: Every case, in a fixed order so a report always reads the same way.
CORPUS: list[ParserGapCase] = (
    VERB_FORM_GAPS
    + PHRASING_GAPS
    + ALIAS_GAPS
    + NON_COMMANDS
    + QUESTIONS
    + EMBEDDED_REQUESTS
    + FUZZY_BOUNDARY
)


def by_category(category: str) -> list[ParserGapCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]

