"""The adversarial corpus for the contextual framing layer.

This is the **second** framing corpus and it is deliberately hostile. The
first one (:mod:`tests.nlu_framing_corpus`) is saturated at 32 of 32, so
it can no longer detect anything. This one exists to try to break the
current rules.

How the cases are chosen
------------------------
Every case is built to sit near a decision boundary:

* a genuine request that also contains a word which normally marks a remark,
  so the two signals have to be weighed against each other;
* a remark that contains a word which normally marks a request;
* a sentence with no cue at all, where blocking would be a mistake.

That is where a small rule list is most likely to be wrong, so that is
where the cases are.

Two of the eight categories are **known-layering** cases, and the reason
is worth stating before any number is read:

``NEGATION_CASES`` and part of ``CONVERSATIONAL_CASES`` are situations the
framing layer is not the right place to solve. *"don't tell me a joke"* is
already stopped upstream, by the NLU negation guard, which makes the parser
return ``None`` before framing is ever consulted. Those cases are recorded
because a future change to framing should not be confused into thinking it
has to solve them, and because they show where the layers meet.

Labelling
---------
Labels record what a user **reasonably meant**, never what the current code
returns. A case the rules get wrong stays mislabelled on purpose so the
evaluation can surface it. Nothing here is tuned to make the suite green.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUEST = "request"
MENTION = "mention"
NEUTRAL = "neutral"

#: A request that arrives wrapped in conversational padding.
CATEGORY_EMBEDDED = "request-embedded"
#: A remark that trails directly behind a real request cue.
CATEGORY_AFTER_CUE = "mention-after-cue"
#: Plain conversation about the topic.
CATEGORY_CONVERSATIONAL = "conversational"
#: Negated requests. See the module docstring: handled upstream.
CATEGORY_NEGATION = "negation"
#: Too short to imply anything either way.
CATEGORY_FRAGMENT = "fragment"
#: Natural questions.
CATEGORY_QUESTION = "question"
#: Narrative verbs sitting next to a genuine request.
CATEGORY_NARRATIVE_NEAR = "narrative-near-request"
#: Words that look like a remark but are part of a real command.
CATEGORY_TRAP = "lexical-trap"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_EMBEDDED,
    CATEGORY_AFTER_CUE,
    CATEGORY_CONVERSATIONAL,
    CATEGORY_NEGATION,
    CATEGORY_FRAGMENT,
    CATEGORY_QUESTION,
    CATEGORY_NARRATIVE_NEAR,
    CATEGORY_TRAP,
)

#: Categories where the framing layer is not the intended line of defence.
KNOWN_LAYERING: frozenset[str] = frozenset({CATEGORY_NEGATION})


@dataclass(frozen=True)
class AdversarialCase:
    """One hostile utterance and the verdict a person would expect.

    Attributes:
        utterance: what the user says.
        intent: the intent it is judged against.
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
# 1. A request wrapped in conversational padding. A remark marker is
#    present, but a real request cue must win.
# ----------------------------------------------------------------------
EMBEDDED_REQUESTS: list[AdversarialCase] = [
    AdversarialCase("the weather was good, check it again", "weather", REQUEST,
                    CATEGORY_EMBEDDED, "'was' is a remark marker, 'check' must win"),
    AdversarialCase("I heard the news, tell me today's news", "news", REQUEST,
                    CATEGORY_EMBEDDED, "'heard' is narrative, 'tell' must win"),
    AdversarialCase("my friend told me a joke, tell me another one", "jokes",
                    REQUEST, CATEGORY_EMBEDDED, "'told' is narrative, 'tell' must win"),
    AdversarialCase("we discussed the weather, what's it like now", "weather",
                    REQUEST, CATEGORY_EMBEDDED, "'discussed' is narrative, question must win"),
    AdversarialCase("I was listening to music earlier, play it again", "youtube",
                    REQUEST, CATEGORY_EMBEDDED, "'was' is a remark, 'play' must win"),
    AdversarialCase("the news was long, show me the headlines", "news", REQUEST,
                    CATEGORY_EMBEDDED, "'was' is a remark, 'show' must win"),
]

# ----------------------------------------------------------------------
# 2. A remark trailing directly behind a genuine request cue. The cue
#    is for the first clause, not the trailing one.
# ----------------------------------------------------------------------
MENTIONS_AFTER_CUE: list[AdversarialCase] = [
    AdversarialCase("tell me the news I heard earlier", "news", REQUEST,
                    CATEGORY_AFTER_CUE, "'tell' addresses the assistant"),
    AdversarialCase("tell me a joke I mentioned yesterday", "jokes", REQUEST,
                    CATEGORY_AFTER_CUE, "'tell' addresses the assistant"),
    AdversarialCase("check the weather I was talking about", "weather", REQUEST,
                    CATEGORY_AFTER_CUE, "'check' addresses the assistant"),
    AdversarialCase("show me the logs we discussed", "history", REQUEST,
                    CATEGORY_AFTER_CUE, "'show' addresses the assistant"),
    AdversarialCase("play the music I heard on the radio", "youtube", REQUEST,
                    CATEGORY_AFTER_CUE, "'play' addresses the assistant"),
]

# ----------------------------------------------------------------------
# 3. Plain conversation. No request cue anywhere, so these should block.
# ----------------------------------------------------------------------
CONVERSATIONAL_MENTIONS: list[AdversarialCase] = [
    AdversarialCase("I heard you tell a joke", "jokes", MENTION,
                    CATEGORY_CONVERSATIONAL,
                    "the user is describing what the assistant did"),
    AdversarialCase("we talked about the weather", "weather", MENTION,
                    CATEGORY_CONVERSATIONAL, "a past conversation"),
    AdversarialCase("my friend mentioned the news", "news", MENTION,
                    CATEGORY_CONVERSATIONAL, "third-party narrative"),
    AdversarialCase("I discussed the logs with him", "history", MENTION,
                    CATEGORY_CONVERSATIONAL, "'discussed' is past"),
    AdversarialCase("she reported the weather to me", "weather", MENTION,
                    CATEGORY_CONVERSATIONAL, "third-party narrative"),
    AdversarialCase("he confirmed the joke was good", "jokes", MENTION,
                    CATEGORY_CONVERSATIONAL, "'confirmed' is past"),
]

# ----------------------------------------------------------------------
# 4. Negated requests. Recorded for completeness: the NLU negation guard
#    already returns None for these before framing is consulted.
# ----------------------------------------------------------------------
NEGATION_CASES: list[AdversarialCase] = [
    AdversarialCase("don't tell me a joke", "jokes", MENTION, CATEGORY_NEGATION,
                    "already stopped by the NLU negation guard upstream"),
    AdversarialCase("don't tell me the news", "news", MENTION, CATEGORY_NEGATION,
                    "already stopped upstream"),
    AdversarialCase("I didn't ask for the news", "news", MENTION,
                    CATEGORY_NEGATION, "already stopped upstream"),
    AdversarialCase("I wasn't talking about the weather", "weather", MENTION,
                    CATEGORY_NEGATION, "already stopped upstream"),
    AdversarialCase("I never discussed the weather", "weather", MENTION,
                    CATEGORY_NEGATION, "already stopped upstream"),
]

# ----------------------------------------------------------------------
# 5. Fragments. No cue at all, so NEUTRAL and the command must run.
# ----------------------------------------------------------------------
SHORT_FRAGMENTS: list[AdversarialCase] = [
    AdversarialCase("weather?", "weather", NEUTRAL, CATEGORY_FRAGMENT,
                    "punctuation only, no cue"),
    AdversarialCase("news?", "news", NEUTRAL, CATEGORY_FRAGMENT,
                    "punctuation only, no cue"),
    AdversarialCase("a joke", "jokes", NEUTRAL, CATEGORY_FRAGMENT,
                    "a bare noun phrase, no cue"),
    AdversarialCase("today's weather", "weather", NEUTRAL, CATEGORY_FRAGMENT,
                    "a possessive noun phrase, no cue"),
    AdversarialCase("some music", "youtube", NEUTRAL, CATEGORY_FRAGMENT,
                    "a bare quantifier phrase, no cue"),
    AdversarialCase("the logs?", "history", NEUTRAL, CATEGORY_FRAGMENT,
                    "punctuation only, no cue"),
]

# ----------------------------------------------------------------------
# 6. Natural questions. A leading interrogative must always win.
# ----------------------------------------------------------------------
NATURAL_QUESTIONS: list[AdversarialCase] = [
    AdversarialCase("what do you think about the weather?", "weather", REQUEST,
                    CATEGORY_QUESTION, "leading 'what'"),
    AdversarialCase("did you hear the news?", "news", REQUEST, CATEGORY_QUESTION,
                    "leading 'did'"),
    AdversarialCase("is the weather good today?", "weather", REQUEST,
                    CATEGORY_QUESTION, "leading copula is a question"),
    AdversarialCase("why was the news interesting?", "news", REQUEST,
                    CATEGORY_QUESTION, "leading 'why' beats the 'was' inside"),
    AdversarialCase("how was your day?", "news", REQUEST, CATEGORY_QUESTION,
                    "no real trigger, but a question must not be blocked"),
    AdversarialCase("can you tell me about the news?", "news", REQUEST,
                    CATEGORY_QUESTION, "leading 'can'"),
]

# ----------------------------------------------------------------------
# 7. Narrative verbs sitting next to a genuine request. The request
#    half must not be blocked by the narrative half.
# ----------------------------------------------------------------------
NARRATIVE_NEAR_REQUESTS: list[AdversarialCase] = [
    AdversarialCase("I mentioned the weather, what is it now", "weather", REQUEST,
                    CATEGORY_NARRATIVE_NEAR, "'what' must win over 'mentioned'"),
    AdversarialCase("we talked about the news, give me the latest", "news",
                    REQUEST, CATEGORY_NARRATIVE_NEAR, "'give' must win"),
    AdversarialCase("I reported the joke earlier, tell me another", "jokes",
                    REQUEST, CATEGORY_NARRATIVE_NEAR, "'tell' must win"),
    AdversarialCase("she explained the news, find me something else", "news",
                    REQUEST, CATEGORY_NARRATIVE_NEAR, "'find' must win"),
    AdversarialCase("he commented on the weather, check it now", "weather",
                    REQUEST, CATEGORY_NARRATIVE_NEAR, "'check' must win"),
    AdversarialCase("I mentioned the logs, read them back", "history", REQUEST,
                    CATEGORY_NARRATIVE_NEAR, "'read' must win"),
]

# ----------------------------------------------------------------------
# 8. Lexical traps. Words that look like a remark but belong to a
#    working command.
# ----------------------------------------------------------------------
LEXICAL_TRAPS: list[AdversarialCase] = [
    AdversarialCase("tell me the news", "news", REQUEST, CATEGORY_TRAP,
                    "'tell' is a request verb, not a narrative marker"),
    AdversarialCase("can you tell me a joke?", "jokes", REQUEST, CATEGORY_TRAP,
                    "'can' and 'tell' are both request cues"),
    AdversarialCase("what did I say earlier?", "history", REQUEST, CATEGORY_TRAP,
                    "'what did i say' is the history alias"),
    AdversarialCase("can you read my history?", "history", REQUEST, CATEGORY_TRAP,
                    "'read' is deliberately excluded from the narrative verbs"),
    AdversarialCase("tell me what you heard", "news", REQUEST, CATEGORY_TRAP,
                    "'tell' must beat the narrative 'heard'"),
    AdversarialCase("show me the news you reported", "news", REQUEST,
                    CATEGORY_TRAP, "'show' must beat the narrative 'reported'"),
]

#: Every case, in a fixed order so a report always reads the same way.
CORPUS: list[AdversarialCase] = (
    EMBEDDED_REQUESTS
    + MENTIONS_AFTER_CUE
    + CONVERSATIONAL_MENTIONS
    + NEGATION_CASES
    + SHORT_FRAGMENTS
    + NATURAL_QUESTIONS
    + NARRATIVE_NEAR_REQUESTS
    + LEXICAL_TRAPS
)


def by_category(category: str) -> list[AdversarialCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]
