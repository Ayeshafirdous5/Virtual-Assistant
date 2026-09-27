"""The measurement corpus for the gerund ``telling``.

This step is **measurement only**. No production rule is changed here.

Why this study exists
---------------------
Three earlier steps walked the ``tell`` family one form at a time:

* Step 7 closed eight leaks caused by the bare ``tell`` request cue.
* Step 8 measured all four forms and found "tells" and "telling" sitting
  in *neither* :data:`REQUEST_CUES` nor :data:`NARRATIVE_VERBS`.
* Step 9 closed "tells", which had a clean structural mark.

``telling`` is the last one standing, and it is the hardest of the four,
because the gerund appears in **both** directions at once:

    "I remember you telling me a joke"   a report of past speech
    "keep telling me more"               an order to the assistant

There is no subject test that separates those, because the first has a
subject in front of the gerund and the second has an implicit one. Any
rule that blocks the first on structure alone will block the second too.
So the only honest way to decide is to measure how often each shape
actually appears, and to keep the two populations in separate
categories. That is what this corpus does.

What "measure" means here
-------------------------
A verdict is recorded for every case, and the report separates three
outcomes per case, because they are not the same problem:

framing mismatch
    The parser produced an intent, framing was consulted, and framing
    answered wrongly. **Only these are addressable by a change to
    ``framing.py``.**
parser no-match
    The sentence names no trigger word, so the parser returns ``None``
    and framing is never asked. This is a parser-layer gap. It is
    reported separately and is never counted as a framing fix.
another layer
    The NLU negation guard or the ambiguity check stopped it first.

A parser gap being counted as a framing fix would improve the accuracy
number for entirely the wrong reason, so the two are kept apart here
exactly as they were in the Step 8 verb-form study.

Deliberately left out
---------------------
The six parser no-match cases already recorded in
:mod:`tests.nlu_framing_verb_forms_corpus` are **not** repeated here.
They are a parser-layer problem, and mixing them into this study would
make it look larger than it is.

Labelling
---------
Labels record what a user **reasonably meant**, never what the code
returns. Where the two differ the case stays mislabelled on purpose.
Nothing here is tuned to make the suite green.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUEST = "request"
MENTION = "mention"
NEUTRAL = "neutral"

#: Reported speech, past or in progress.
CATEGORY_NARRATIVE = "narrative-telling"

#: The user is asking the assistant to keep going.
CATEGORY_REQUEST = "telling-request"

#: Questions carrying the gerund.
CATEGORY_QUESTION = "telling-question"

#: A remark, then a real request in the same breath.
CATEGORY_AFTER_NARRATIVE = "telling-then-request"

#: The gerunds one or two letters away. Measured, not assumed: each one
#: is labelled by what a user meant, and the report says what the code
#: actually does with it.
CATEGORY_GERUND_TRAP = "nearby-gerund"

#: Who is doing the telling, varied across every pronoun that can be.
CATEGORY_SUBJECT = "telling-subject"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_NARRATIVE,
    CATEGORY_REQUEST,
    CATEGORY_QUESTION,
    CATEGORY_AFTER_NARRATIVE,
    CATEGORY_GERUND_TRAP,
    CATEGORY_SUBJECT,
)


@dataclass(frozen=True)
class TellingCase:
    """One utterance and the verdict a person would expect.

    Attributes:
        utterance: what the user says.
        intent: the intent it is judged against. Framing is judged in
            isolation against this, so the parser's own behaviour is
            measured separately by the evaluation module.
        expected: :data:`REQUEST`, :data:`MENTION`, or :data:`NEUTRAL`.
        category: which group this case belongs to.
        note: why the label was chosen, and how close the call is.
    """

    utterance: str
    intent: str
    expected: str
    category: str
    note: str = ""

    @property
    def blocks(self) -> bool:
        """True when this verdict would stop the command running."""
        return self.expected == MENTION

    def describe(self) -> str:
        return f"{self.utterance!r} [{self.intent}] -> {self.expected}"


# ----------------------------------------------------------------------
# A. Reported speech. The gerund attaches to a past or continuing
#    event, and the user is describing it rather than ordering it.
# ----------------------------------------------------------------------
NARRATIVE_TELLING: list[TellingCase] = [
    TellingCase("I remember you telling me a joke", "jokes", MENTION,
                CATEGORY_NARRATIVE,
                "the one gap Step 9 left open; a report of past speech"),
    TellingCase("I remember him telling me the news", "news", MENTION,
                CATEGORY_NARRATIVE, "recalled speech from a third party"),
    TellingCase("she was telling me the story", "jokes", MENTION,
                CATEGORY_NARRATIVE, "'was' makes it a past continuous"),
    TellingCase("we talked about her telling me the news", "news", MENTION,
                CATEGORY_NARRATIVE, "a remark about a remark, twice over"),
    TellingCase("I recall you telling me that yesterday", "news", MENTION,
                CATEGORY_NARRATIVE, "recalled speech, marked by 'yesterday'"),
    TellingCase("I remember him telling that story", "jokes", MENTION,
                CATEGORY_NARRATIVE, "recalled speech"),
    TellingCase("they kept telling me the same thing", "news", MENTION,
                CATEGORY_NARRATIVE, "'kept' marks a repeated past event"),
    TellingCase("I liked you telling that story", "jokes", MENTION,
                CATEGORY_NARRATIVE, "a reaction to past speech"),
    TellingCase("I heard her telling the story", "jokes", MENTION,
                CATEGORY_NARRATIVE, "'heard' plus the gerund"),
    TellingCase("you were telling me a joke", "jokes", MENTION,
                CATEGORY_NARRATIVE, "'were' reports an interrupted event"),
]

# ----------------------------------------------------------------------
# B. The user asking the assistant to keep going. These are the cases a
#    "telling is a remark" rule would break, and they are the reason
#    this category exists in the corpus at all.
# ----------------------------------------------------------------------
TELLING_REQUESTS: list[TellingCase] = [
    TellingCase("keep telling me more", "jokes", REQUEST, CATEGORY_REQUEST,
                "an order to carry on; no subject to attach a report to"),
    TellingCase("tell me more", "jokes", REQUEST, CATEGORY_REQUEST,
                "the bare imperative, carried by the request cue"),
    TellingCase("can you keep telling me jokes?", "jokes", REQUEST,
                CATEGORY_REQUEST, "'can' opens a question"),
    TellingCase("continue telling me the story", "jokes", REQUEST,
                CATEGORY_REQUEST, "'continue' is an imperative about the action"),
    TellingCase("keep telling me more jokes", "jokes", REQUEST,
                CATEGORY_REQUEST, "the same order, with a plural object"),
    TellingCase("please keep telling me", "news", REQUEST, CATEGORY_REQUEST,
                "'please' is a request cue in its own right"),
    TellingCase("you're telling me something interesting", "facts", MENTION,
                CATEGORY_REQUEST,
                "listed as a request in the brief, but it asks for nothing: "
                "a remark about what the assistant is doing"),
]


# ----------------------------------------------------------------------
# C. Questions carrying the gerund. Two kinds are held apart, because
#    a rule that blocks one would have to block the other: asking what
#    somebody was saying, and asking the assistant to keep saying it.
# ----------------------------------------------------------------------
TELLING_QUESTIONS: list[TellingCase] = [
    # C1. Asking about an event.
    TellingCase("what are you telling me?", "news", REQUEST, CATEGORY_QUESTION,
                "asks what is being said; a question must not block"),
    TellingCase("what was he telling you?", "news", REQUEST, CATEGORY_QUESTION,
                "asks about a past event"),
    TellingCase("why were you telling me that?", "news", REQUEST,
                CATEGORY_QUESTION, "asks for a reason"),
    TellingCase("how long has he been telling you this?", "news", REQUEST,
                CATEGORY_QUESTION, "asks about a duration"),
    TellingCase("did you keep telling him the news?", "news", REQUEST,
                CATEGORY_QUESTION, "asks whether it happened"),
    # C2. Asking the assistant to act.
    TellingCase("will you keep telling me jokes?", "jokes", REQUEST,
                CATEGORY_QUESTION, "'will' asks for an action"),
    TellingCase("can you keep telling me the story?", "jokes", REQUEST,
                CATEGORY_QUESTION, "a modal question aimed at the assistant"),
]

# ----------------------------------------------------------------------
# D. A remark, then a real request. Whatever is done about the gerund,
#    the request half has to survive.
# ----------------------------------------------------------------------
TELLING_THEN_REQUEST: list[TellingCase] = [
    TellingCase("I remember you telling me that, explain it again", "facts",
                REQUEST, CATEGORY_AFTER_NARRATIVE,
                "'explain' is a request cue in its own right"),
    TellingCase("she was telling me the news, now tell me today's news", "news",
                REQUEST, CATEGORY_AFTER_NARRATIVE, "'tell' is the ask"),
    TellingCase("I remember you telling that joke, tell me another one", "jokes",
                REQUEST, CATEGORY_AFTER_NARRATIVE,
                "the bare cue in the second half"),
    TellingCase("you were telling me a joke, tell me another", "jokes", REQUEST,
                CATEGORY_AFTER_NARRATIVE, "an interrupted remark, then an order"),
    TellingCase("I remember you telling me the weather, check it", "weather",
                REQUEST, CATEGORY_AFTER_NARRATIVE, "'check' is the cue"),
]

# ----------------------------------------------------------------------
# E. The gerunds next door. These are measured rather than assumed: the
#    rule that is eventually written for "telling" must not be written by
#    analogy with these, because some are already handled and some are
#    not.
# ----------------------------------------------------------------------
NEARBY_GERUNDS: list[TellingCase] = [
    TellingCase("we were talking about the weather", "weather", MENTION,
                CATEGORY_GERUND_TRAP, "'talked' is a narrative verb"),
    TellingCase("we were discussing the news", "news", MENTION,
                CATEGORY_GERUND_TRAP, "'discussed' is a narrative verb"),
    TellingCase("he was explaining the joke to me", "jokes", MENTION,
                CATEGORY_GERUND_TRAP, "'explained' is a narrative verb"),
    TellingCase("she was mentioning the weather", "weather", MENTION,
                CATEGORY_GERUND_TRAP, "'mentioned' is a narrative verb"),
    TellingCase("he was saying the news was good", "news", MENTION,
                CATEGORY_GERUND_TRAP, "'said' is unlisted, but the copula is not"),
    TellingCase("he was hearing the news at noon", "weather", MENTION,
                CATEGORY_GERUND_TRAP,
                "the copula carries this one, not the gerund"),
]

# ----------------------------------------------------------------------
# F. Who is telling. Every subject that can carry the gerund, so a future
#    rule cannot quietly work for "you" and miss "they".
#
#    Each utterance here is distinct from the ones in section A on
#    purpose: they exist to vary the subject, not to repeat a case.
# ----------------------------------------------------------------------
TELLING_SUBJECTS: list[TellingCase] = [
    TellingCase("I remember her telling that story", "jokes", MENTION,
                CATEGORY_SUBJECT, "third person feminine, recalled"),
    TellingCase("I recall him telling me the news", "news", MENTION,
                CATEGORY_SUBJECT, "third person masculine, recalled"),
    TellingCase("they were telling me jokes", "jokes", MENTION,
                CATEGORY_SUBJECT, "third person plural, past continuous"),
    TellingCase("my friend was telling me about the weather", "weather", MENTION,
                CATEGORY_SUBJECT, "a noun subject rather than a pronoun"),
    TellingCase("we were telling each other jokes", "jokes", MENTION,
                CATEGORY_SUBJECT, "first person plural, mutual"),
    TellingCase("I remember you telling me the weather", "weather", MENTION,
                CATEGORY_SUBJECT, "second person, recalled"),
]


#: Every case, in a fixed order so a report always reads the same way.
#:
#: The first two categories are the ones that matter. Every rule written
#: for "telling" has to separate :data:`NARRATIVE_TELLING` from
#: :data:`TELLING_REQUESTS`, and those two groups together are the
#: decision the next step has to make.
CORPUS: list[TellingCase] = (
    NARRATIVE_TELLING
    + TELLING_REQUESTS
    + TELLING_QUESTIONS
    + TELLING_THEN_REQUEST
    + NEARBY_GERUNDS
    + TELLING_SUBJECTS
)

#: The two populations a "telling" rule has to tell apart. Kept as a
#: named pair so a test can assert on them directly rather than by
#: retyping the utterances.
DECIDING_PAIR = (NARRATIVE_TELLING, TELLING_REQUESTS)


def by_category(category: str) -> list[TellingCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]

