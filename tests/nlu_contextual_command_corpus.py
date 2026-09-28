"""Contextual command phrasing: a measurement corpus.

This step is **measurement only**. No production code is changed and no
fix is proposed here.

The question
------------
Phase 5 Step 14 evaluated the twelve remaining alias-shaped gaps and
found that every ordinary alias form produced a false positive. The
question this corpus asks is the obvious follow-up:

    can **context or structure** tell the command from its twin, where
    a word list cannot?

The short answer this corpus records is **mostly yes, and not safely**.
The signal that separates them is positional: in every command tested
the trigger is sentence-initial, and in the conversational twin it is
mid-sentence. That separates 17 of 19 pairs. It fails on the two
question-shaped families, and it is not sufficient on its own, because
real users do not always lead with the verb. Both limits are measured
below rather than assumed.

Why position cannot become a rule
---------------------------------
A "must be sentence-initial" rule is the obvious implementation of the
signal, and it would break ordinary speech. Every one of these is a
genuine request with the phrase in the second slot:

    "ok amuse me"                "so define gravity"
    "today wind speed please"    "right gossip"

:data:`MID_SENTENCE_COMMANDS` exists to keep that cost visible.

Why some families are worse than others
---------------------------------------
:data:`DANGEROUS_CASES` are the ones where being wrong is not a missed
request. "the device will power off" and "the train will stop" are
ordinary sentences about ordinary things, and a rule that fires on them
closes the application. The system intent is already protected by a
higher score bar and a tail-position requirement, and this corpus shows
where that protection stops working.

Labelling
---------
:attr:`ContextCase.intent` is the tool the sentence is **about**,
whether or not it is a request. :attr:`ContextCase.outcome` says which
it is. :attr:`ContextCase.expected` is what the parser is **expected**
to return, and it is allowed to be
:data:`~tests.nlu_contextual_command_corpus.NO_MATCH` for a genuine
command that is not yet supported. Nothing here is tuned to be green.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The parser returned an intent, confidently.
MATCH = "match"

#: The parser returned nothing. Framing is never consulted.
NO_MATCH = "no-match"

#: The user is asking for the tool.
COMMAND = "command"

#: The user is talking, and the words only resemble a request.
NON_COMMAND = "non-command"

#: Ordinary, missing a request costs the user nothing serious.
MISS_IS_HARMLESS = "harmless-miss"

#: Being wrong here runs the wrong tool, or closes the application.
MISS_IS_DANGEROUS = "dangerous-false-positive"

CATEGORY_INFORMATION = "information"
CATEGORY_FACTS_JOKES = "facts-jokes"
CATEGORY_WEATHER = "weather"
CATEGORY_NEWS = "news"
CATEGORY_SYSTEM = "system-dangerous"
CATEGORY_SUPPORTED = "already-supported"
CATEGORY_MID_SENTENCE = "mid-sentence-command"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_INFORMATION,
    CATEGORY_FACTS_JOKES,
    CATEGORY_WEATHER,
    CATEGORY_NEWS,
    CATEGORY_SYSTEM,
    CATEGORY_SUPPORTED,
    CATEGORY_MID_SENTENCE,
)

#: How each case is classified after measurement.
SAFE_CONTEXTUAL = "safe-contextual-candidate"
UNSAFE = "unsafe"
ALREADY_SUPPORTED = "already-supported"
INTENTIONALLY_UNSUPPORTED = "intentionally-unsupported"
NEEDS_MORE_EVIDENCE = "needs-more-evidence"

CLASSIFICATIONS: tuple[str, ...] = (
    SAFE_CONTEXTUAL,
    UNSAFE,
    ALREADY_SUPPORTED,
    INTENTIONALLY_UNSUPPORTED,
    NEEDS_MORE_EVIDENCE,
)


@dataclass(frozen=True)
class ContextCase:
    """One utterance, what it is, and what the parser is expected to do.

    Attributes:
        utterance: what the user says.
        outcome: :data:`COMMAND` or :data:`NON_COMMAND`.
        intent: the tool the sentence is about.
        category: which family it belongs to.
        expected: what the parser should return.
        significance: what being wrong would cost.
        classification: the verdict this step reached.
        trigger: the phrase a contextual rule would look for.
        note: why the label and the verdict are what they are.
    """

    utterance: str
    outcome: str
    intent: str
    category: str
    expected: str
    significance: str
    classification: str
    trigger: str = ""
    note: str = ""

    @property
    def is_command(self) -> bool:
        return self.outcome == COMMAND

    @property
    def is_dangerous(self) -> bool:
        """True when matching this wrongly would be costly."""
        return self.significance == MISS_IS_DANGEROUS

# ----------------------------------------------------------------------
# A. Information. "define" and "search the web" separate cleanly by
#    position; "what is" does not, because its twin is a question too.
# ----------------------------------------------------------------------
INFORMATION_CASES: list[ContextCase] = [
    ContextCase("define gravity", COMMAND, "information", CATEGORY_INFORMATION,
                NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "define",
                "sentence-initial; the twin buries it at position five"),
    ContextCase("the problem is hard to define", NON_COMMAND, "information",
                CATEGORY_INFORMATION, NO_MATCH, MISS_IS_DANGEROUS, SAFE_CONTEXTUAL,
                "define", "if this matched it would open a browser"),
    ContextCase("search the web for python", COMMAND, "information",
                CATEGORY_INFORMATION, NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL,
                "search the web", "sentence-initial; the twin starts with 'I will'"),
    ContextCase("I will search the web later", NON_COMMAND, "information",
                CATEGORY_INFORMATION, NO_MATCH, MISS_IS_DANGEROUS, SAFE_CONTEXTUAL,
                "search the web", "a stated intention, not a request"),
    ContextCase("what is gravity", COMMAND, "information", CATEGORY_INFORMATION,
                NO_MATCH, MISS_IS_HARMLESS, NEEDS_MORE_EVIDENCE, "what is",
                "NOT positionally separable: the twin opens the same way"),
    ContextCase("what is the time", NON_COMMAND, "information",
                CATEGORY_INFORMATION, NO_MATCH, MISS_IS_DANGEROUS,
                NEEDS_MORE_EVIDENCE, "what is", "a question about something else"),
    ContextCase("google python", COMMAND, "information", CATEGORY_INFORMATION,
                NO_MATCH, MISS_IS_HARMLESS, UNSAFE, "google",
                "'he works at google' is an ordinary sentence"),
    ContextCase("he works at google", NON_COMMAND, "information",
                CATEGORY_INFORMATION, NO_MATCH, MISS_IS_DANGEROUS, UNSAFE,
                "google", "a proper noun, not a verb"),
]

# ----------------------------------------------------------------------
# B. Facts and jokes. "did you know" is the second family that position
#    cannot separate.
# ----------------------------------------------------------------------
FACTS_JOKES_CASES: list[ContextCase] = [
    ContextCase("something funny", COMMAND, "jokes", CATEGORY_FACTS_JOKES,
                NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "something funny",
                "sentence-initial; every twin has it mid-sentence"),
    ContextCase("it was something funny", NON_COMMAND, "jokes",
                CATEGORY_FACTS_JOKES, NO_MATCH, MISS_IS_DANGEROUS,
                SAFE_CONTEXTUAL, "something funny", "a remark about a past event"),
    ContextCase("amuse me", COMMAND, "jokes", CATEGORY_FACTS_JOKES, NO_MATCH,
                MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "amuse me",
                "an imperative; the twin has a third person subject"),
    ContextCase("he will amuse me later", NON_COMMAND, "jokes",
                CATEGORY_FACTS_JOKES, NO_MATCH, MISS_IS_DANGEROUS,
                SAFE_CONTEXTUAL, "amuse me", "a prediction about someone else"),
    ContextCase("crack me up", COMMAND, "jokes", CATEGORY_FACTS_JOKES, NO_MATCH,
                MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "crack me up",
                "an imperative, sentence-initial"),
    ContextCase("he will crack me up", NON_COMMAND, "jokes",
                CATEGORY_FACTS_JOKES, NO_MATCH, MISS_IS_DANGEROUS,
                SAFE_CONTEXTUAL, "crack me up", "a prediction, not a request"),
    ContextCase("did you know", COMMAND, "facts", CATEGORY_FACTS_JOKES, NO_MATCH,
                MISS_IS_HARMLESS, NEEDS_MORE_EVIDENCE, "did you know",
                "NOT positionally separable, and it opens every question"),
    ContextCase("did you know she left", NON_COMMAND, "facts",
                CATEGORY_FACTS_JOKES, NO_MATCH, MISS_IS_DANGEROUS,
                NEEDS_MORE_EVIDENCE, "did you know", "a question about a person"),
    ContextCase("tell me a fact", COMMAND, "facts", CATEGORY_FACTS_JOKES, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "fact",
                "already resolves through the facts pattern"),
    ContextCase("he told me a fact", NON_COMMAND, "facts", CATEGORY_FACTS_JOKES,
                MATCH, MISS_IS_DANGEROUS, ALREADY_SUPPORTED, "fact",
                "already matches today; a pre-existing leak, not a new one"),
]


# ----------------------------------------------------------------------
# C. Weather. Every family here separates cleanly by position, and none
#    of them is dangerous.
# ----------------------------------------------------------------------
WEATHER_CASES: list[ContextCase] = [
    ContextCase("celsius", COMMAND, "weather", CATEGORY_WEATHER, NO_MATCH,
                MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "celsius",
                "a bare unit, but only the request reads as a request"),
    ContextCase("it was 30 celsius yesterday", NON_COMMAND, "weather",
                CATEGORY_WEATHER, NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL,
                "celsius", "a statement about a measurement"),
    ContextCase("degrees outside", COMMAND, "weather", CATEGORY_WEATHER,
                NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "degrees outside",
                "sentence-initial; the twin has a subject before it"),
    ContextCase("he has degrees outside law", NON_COMMAND, "weather",
                CATEGORY_WEATHER, NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL,
                "degrees outside", "an unrelated meaning of the same words"),
    ContextCase("wind speed", COMMAND, "weather", CATEGORY_WEATHER, NO_MATCH,
                MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "wind speed",
                "sentence-initial; the twin has a determiner first"),
    ContextCase("the wind speed was high", NON_COMMAND, "weather",
                CATEGORY_WEATHER, NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL,
                "wind speed", "a statement about a measurement"),
    ContextCase("how warm is it", COMMAND, "weather", CATEGORY_WEATHER, NO_MATCH,
                MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "how warm is it",
                "matches the shape of the shipped 'how cold' alias"),
    ContextCase("he asked how warm is it", NON_COMMAND, "weather",
                CATEGORY_WEATHER, NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL,
                "how warm is it", "reported speech, not a question"),
    ContextCase("is it warm", COMMAND, "weather", CATEGORY_WEATHER, NO_MATCH,
                MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "is it warm",
                "sentence-initial; the twin has a reporting verb first"),
    ContextCase("he said is it warm", NON_COMMAND, "weather", CATEGORY_WEATHER,
                NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "is it warm",
                "quoted speech inside a statement"),
]

# ----------------------------------------------------------------------
# D. News. Both separate by position, and both are harmless.
# ----------------------------------------------------------------------
NEWS_CASES: list[ContextCase] = [
    ContextCase("gossip", COMMAND, "news", CATEGORY_NEWS, NO_MATCH,
                MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "gossip",
                "bare noun, but the twin has a verb before it"),
    ContextCase("she spread gossip", NON_COMMAND, "news", CATEGORY_NEWS,
                NO_MATCH, MISS_IS_HARMLESS, SAFE_CONTEXTUAL, "gossip",
                "a statement about someone talking"),
    ContextCase("in the news today", COMMAND, "news", CATEGORY_NEWS, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "in the news today",
                "already resolves through the 'news' tool pattern, so this "
                "is not a gap at all"),
    ContextCase("he was big in the news today", NON_COMMAND, "news",
                CATEGORY_NEWS, MATCH, MISS_IS_HARMLESS, ALREADY_SUPPORTED,
                "in the news today",
                "the parser matches, and framing then blocks it; the two "
                "layers together are safe here"),
]

# ----------------------------------------------------------------------
# E. System. The dangerous family. Position separates these too, and it
#    does not matter, because the twin sentences are ordinary English
#    about ordinary devices.
# ----------------------------------------------------------------------
SYSTEM_CASES: list[ContextCase] = [
    ContextCase("power off", COMMAND, "system", CATEGORY_SYSTEM, NO_MATCH,
                MISS_IS_HARMLESS, UNSAFE, "power off",
                "missing a real request costs nothing; it is the twin "
                "below that is expensive"),
    ContextCase("the device will power off", NON_COMMAND, "system",
                CATEGORY_SYSTEM, NO_MATCH, MISS_IS_DANGEROUS, UNSAFE,
                "power off", "would close the assistant if it matched"),
    ContextCase("stop", COMMAND, "system", CATEGORY_SYSTEM, NO_MATCH,
                MISS_IS_HARMLESS, UNSAFE, "stop",
                "the most overloaded word in English, and the twin below "
                "is the reason"),
    ContextCase("the train will stop", NON_COMMAND, "system", CATEGORY_SYSTEM,
                NO_MATCH, MISS_IS_DANGEROUS, UNSAFE, "stop",
                "would close the assistant if it matched"),
    ContextCase("close the assistant", COMMAND, "system", CATEGORY_SYSTEM,
                NO_MATCH, MISS_IS_HARMLESS, UNSAFE, "close the assistant",
                "already covered by 'stop the assistant'"),
    ContextCase("she will close the assistant later", NON_COMMAND, "system",
                CATEGORY_SYSTEM, NO_MATCH, MISS_IS_DANGEROUS, UNSAFE,
                "close the assistant", "a third-person statement"),
    ContextCase("quit", COMMAND, "system", CATEGORY_SYSTEM, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "quit",
                "already works, and the system guard keeps it at the end"),
    ContextCase("I should quit smoking", NON_COMMAND, "system", CATEGORY_SYSTEM,
                NO_MATCH, MISS_IS_DANGEROUS, UNSAFE, "quit",
                "**a live bug found by this step**: the parser returns "
                "system/clear and framing returns neutral, so this "
                "terminates the assistant today. Pre-existing, not caused "
                "by anything measured here"),
]


# ----------------------------------------------------------------------
# F. Commands that already work. Here the risk is the opposite: a
#    contextual rule must not disturb any of them.
# ----------------------------------------------------------------------
SUPPORTED_CASES: list[ContextCase] = [
    ContextCase("tell me a joke", COMMAND, "jokes", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "joke", "covered by Batch 1 era cues"),
    ContextCase("make me laugh", COMMAND, "jokes", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "make me laugh", "a shipped alias"),
    ContextCase("look up python", COMMAND, "information", CATEGORY_SUPPORTED,
                MATCH, MISS_IS_HARMLESS, ALREADY_SUPPORTED, "look up", "a shipped alias"),
    ContextCase("tell me about Python", COMMAND, "information", CATEGORY_SUPPORTED,
                MATCH, MISS_IS_HARMLESS, ALREADY_SUPPORTED, "tell me about", "a shipped alias"),
    ContextCase("search for gravity", COMMAND, "information", CATEGORY_SUPPORTED,
                MATCH, MISS_IS_HARMLESS, ALREADY_SUPPORTED, "search for", "a shipped alias"),
    ContextCase("what is the weather", COMMAND, "weather", CATEGORY_SUPPORTED,
                MATCH, MISS_IS_HARMLESS, ALREADY_SUPPORTED, "weather", "the tool pattern"),
    ContextCase("how cold is it", COMMAND, "weather", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "how cold", "a shipped alias"),
    ContextCase("top headlines", COMMAND, "news", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "headlines", "a tool pattern"),
    ContextCase("random fact", COMMAND, "facts", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "fact", "a tool pattern"),
    ContextCase("play a song", COMMAND, "youtube", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "play", "a shipped alias"),
    ContextCase("what did I say", COMMAND, "history", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "what did i say", "a shipped alias"),
    ContextCase("note buy milk", COMMAND, "notes", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "note", "a tool pattern"),
    ContextCase("exit", COMMAND, "system", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "exit", "a tool pattern"),
    ContextCase("put on a song", COMMAND, "youtube", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "put on a song", "a Batch 1 alias"),
    ContextCase("will it rain", COMMAND, "weather", CATEGORY_SUPPORTED, MATCH,
                MISS_IS_HARMLESS, ALREADY_SUPPORTED, "will it rain", "a Batch 1 alias"),
]

# ----------------------------------------------------------------------
# G. The cost of a positional rule. Every one of these is a genuine
#    request with the trigger in the second slot, so a "must be
#    sentence-initial" rule would break all of them.
# ----------------------------------------------------------------------
MID_SENTENCE_COMMANDS: list[ContextCase] = [
    ContextCase("ok amuse me", COMMAND, "jokes", CATEGORY_MID_SENTENCE,
                NO_MATCH, MISS_IS_HARMLESS, NEEDS_MORE_EVIDENCE, "amuse me",
                "a bare acknowledgement first"),
    ContextCase("so define gravity", COMMAND, "information", CATEGORY_MID_SENTENCE,
                NO_MATCH, MISS_IS_HARMLESS, NEEDS_MORE_EVIDENCE, "define",
                "a discourse marker first"),
    ContextCase("today wind speed please", COMMAND, "weather", CATEGORY_MID_SENTENCE,
                NO_MATCH, MISS_IS_HARMLESS, NEEDS_MORE_EVIDENCE, "wind speed",
                "an adverb first and a politeness marker last"),
    ContextCase("right gossip", COMMAND, "news", CATEGORY_MID_SENTENCE,
                NO_MATCH, MISS_IS_HARMLESS, NEEDS_MORE_EVIDENCE, "gossip",
                "a single-word filler first"),
    ContextCase("python define what is a class", COMMAND, "information",
                CATEGORY_MID_SENTENCE, NO_MATCH, MISS_IS_HARMLESS,
                NEEDS_MORE_EVIDENCE, "define", "the topic is spoken first"),
]

#: Every case, in a fixed order so a report always reads the same way.
CORPUS: list[ContextCase] = (
    INFORMATION_CASES
    + FACTS_JOKES_CASES
    + WEATHER_CASES
    + NEWS_CASES
    + SYSTEM_CASES
    + SUPPORTED_CASES
    + MID_SENTENCE_COMMANDS
)

#: The commands whose conversational twin would be dangerous to match.
#: Used by the report and the tests to keep the two risks apart.
DANGEROUS_CASES: tuple[str, ...] = tuple(
    case.utterance
    for case in CORPUS
    if case.significance == MISS_IS_DANGEROUS
)


def by_category(category: str) -> list[ContextCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]

