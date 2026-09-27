"""The measurement corpus for the contextual framing layer.

This module holds **data only**: short user utterances, the intent each is
judged against, and the verdict a person would expect. It contains no test
functions and imports nothing from the application, so the evaluation in
``test_nlu_framing_eval.py`` can re-run it cheaply and repeatedly.

Verdicts
--------
``REQUEST``, ``MENTION`` and ``NEUTRAL`` mirror
:mod:`assistant.nlu.framing`, but are written here as plain strings so this
file stays readable on its own. A test asserts the two sets match.

The distinction that actually matters is not the three-way label, it is
whether the sentence **blocks**:

* ``MENTION`` blocks the command.
* ``REQUEST`` and ``NEUTRAL`` both let it through.

So a bare request carrying no cue at all is *correctly* ``NEUTRAL``, and a
wrong ``REQUEST`` is the expensive error because it lets a remark run a
tool. :attr:`FramingCase.blocks` makes that explicit, and the evaluation
reports on it.

Labelling rule
--------------
Labels record what a user **reasonably meant**, never what the current code
happens to return. Where the two differ the case is a finding and is left
mislabelled on purpose so the evaluation can surface it. Nothing here is
tuned to make the suite look good.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUEST = "request"
MENTION = "mention"
NEUTRAL = "neutral"

# Categories, so a report can be read by kind of case.
CATEGORY_REQUEST = "clear-request"
CATEGORY_REMARK = "clear-remark"
CATEGORY_FRAGMENT = "fragment"
CATEGORY_COMPETING = "competing-action"
CATEGORY_CONVERSATIONAL = "conversational"
CATEGORY_QUESTION = "question-shaped"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_REQUEST,
    CATEGORY_REMARK,
    CATEGORY_FRAGMENT,
    CATEGORY_COMPETING,
    CATEGORY_CONVERSATIONAL,
    CATEGORY_QUESTION,
)


@dataclass(frozen=True)
class FramingCase:
    """One utterance and the verdict a person would expect.

    Attributes:
        utterance: what the user says.
        intent: the intent it is judged against, because some rules are
            intent specific: "log" only competes with ``information``.
        expected: :data:`REQUEST`, :data:`MENTION`, or :data:`NEUTRAL`.
        category: which group this case belongs to.
        note: why the label was chosen where it is not obvious.
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
# 1. Clear requests. Every one carries an explicit cue, so REQUEST.
# ----------------------------------------------------------------------
CLEAR_REQUESTS: list[FramingCase] = [
    FramingCase("what is the weather", "weather", REQUEST, CATEGORY_REQUEST,
                "opens with a question word"),
    FramingCase("tell me the weather", "weather", REQUEST, CATEGORY_REQUEST,
                "'tell' is a request verb"),
    FramingCase("check the weather", "weather", REQUEST, CATEGORY_REQUEST,
                "'check' is a request verb"),
    FramingCase("show me the logs", "history", REQUEST, CATEGORY_REQUEST,
                "'show' is a request verb"),
    FramingCase("tell me a joke", "jokes", REQUEST, CATEGORY_REQUEST),
    FramingCase("play some music", "youtube", REQUEST, CATEGORY_REQUEST,
                "'play' is a request verb"),
    FramingCase("get today's news", "news", REQUEST, CATEGORY_REQUEST,
                "'get' is a request verb"),
    FramingCase("please exit", "system", REQUEST, CATEGORY_REQUEST,
                "'please' is a politeness cue"),
]

# ----------------------------------------------------------------------
# 2. Clear remarks. A copula or a hedge makes the sentence a statement.
# ----------------------------------------------------------------------
CLEAR_REMARKS: list[FramingCase] = [
    FramingCase("the weather is nice today", "weather", MENTION,
                CATEGORY_REMARK, "declarative, copula 'is'"),
    FramingCase("I was talking about the weather", "weather", MENTION,
                CATEGORY_REMARK, "past tense narrative, copula 'was'"),
    FramingCase("that joke was funny", "jokes", MENTION, CATEGORY_REMARK,
                "declarative, copula 'was'"),
    FramingCase("I mentioned the logs earlier", "history", MENTION,
                CATEGORY_REMARK, "narrative about an earlier remark"),
    FramingCase("the news was interesting", "news", MENTION, CATEGORY_REMARK,
                "declarative, copula 'was'"),
]

# ----------------------------------------------------------------------
# 3. Fragments. Too short to imply either, so NEUTRAL is correct: the
#    layer should have no opinion and must not block.
# ----------------------------------------------------------------------
FRAGMENTS: list[FramingCase] = [
    FramingCase("weather today", "weather", NEUTRAL, CATEGORY_FRAGMENT,
                "no cue either way; must still run"),
    FramingCase("today's weather", "weather", NEUTRAL, CATEGORY_FRAGMENT,
                "no cue either way; must still run"),
    FramingCase("the news", "news", NEUTRAL, CATEGORY_FRAGMENT),
    FramingCase("some music", "youtube", NEUTRAL, CATEGORY_FRAGMENT),
    FramingCase("a joke", "jokes", NEUTRAL, CATEGORY_FRAGMENT),
]

# ----------------------------------------------------------------------
# 4. Sentences built around a competing action. The expected verdict
#    differs per case: some are a genuine ask, some are the known false
#    positive. "want" and "need" are deliberately not request cues.
# ----------------------------------------------------------------------
COMPETING_ACTION: list[FramingCase] = [
    FramingCase("I want to log this information", "information", MENTION,
                CATEGORY_COMPETING, "the known false positive"),
    FramingCase("I need to record this", "information", MENTION,
                CATEGORY_COMPETING, "a recording request, not a lookup"),
    FramingCase("save this information for me", "information", MENTION,
                CATEGORY_COMPETING, "'save' competes with a lookup"),
    FramingCase("tell me information about the weather", "information",
                REQUEST, CATEGORY_COMPETING,
                "'tell' overrides; this really is a lookup"),
    FramingCase("I want information about Hyderabad", "information", NEUTRAL,
                CATEGORY_COMPETING,
                "a real ask with no cue; neutral still runs it"),
]

# ----------------------------------------------------------------------
# 5. Ordinary conversation that happens to contain a command word.
# ----------------------------------------------------------------------
CONVERSATIONAL: list[FramingCase] = [
    FramingCase("the weather seems good today", "weather", MENTION,
                CATEGORY_CONVERSATIONAL, "'seems' is a copula cue"),
    FramingCase("I heard the news this morning", "news", MENTION,
                CATEGORY_CONVERSATIONAL, "past tense narrative"),
    FramingCase("my friend told me a joke", "jokes", MENTION,
                CATEGORY_CONVERSATIONAL, "narrative about someone else"),
    FramingCase("I was listening to music earlier", "youtube", MENTION,
                CATEGORY_CONVERSATIONAL, "copula 'was'"),
    FramingCase("we discussed the news yesterday", "news", MENTION,
                CATEGORY_CONVERSATIONAL, "narrative"),
]

# ----------------------------------------------------------------------
# 6. Question-shaped. These verify the positional question handling: a
#    leading copula asks, a mid-sentence copula remarks.
# ----------------------------------------------------------------------
QUESTION_SHAPED: list[FramingCase] = [
    FramingCase("is the weather good", "weather", REQUEST, CATEGORY_QUESTION,
                "leading copula is a question, not a statement"),
    FramingCase("is today's weather okay", "weather", REQUEST, CATEGORY_QUESTION),
    FramingCase("what do you think about the weather", "weather", REQUEST,
                CATEGORY_QUESTION,
                "'what' wins before the hedge 'think' is reached"),
    FramingCase("did you hear the news", "news", REQUEST, CATEGORY_QUESTION),
]

#: Every case, in a fixed order so a report always reads the same way.
CORPUS: list[FramingCase] = (
    CLEAR_REQUESTS
    + CLEAR_REMARKS
    + FRAGMENTS
    + COMPETING_ACTION
    + CONVERSATIONAL
    + QUESTION_SHAPED
)


def by_category(category: str) -> list[FramingCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]
