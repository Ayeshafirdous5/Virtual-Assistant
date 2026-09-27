"""Decide whether an utterance is a request or merely a mention.

Why this module exists
----------------------
The scoring stages match a trigger *word*. That is enough to recognise a
command, but it cannot tell a command from a sentence that happens to
contain the same word as a topic:

    "what is the weather"      a request for a report
    "the weather is nice today" a remark about the weather

Both contain ``weather`` and both score 1.000 by an exact match, so every
guard built on scoring alone is satisfied by both. This layer adds the one
piece of context scoring cannot supply: **is the user asking, or just
talking about it?**

Scope
-----
This is deliberately not natural language processing. It is a short list
of request cues, a short list of hedging and copula markers, and a small
number of intent-specific rules. No parser, no grammar, no model.

It is also deliberately **one-sided**. ``MENTION`` is only returned when
there is positive evidence of a remark, because a wrong ``REQUEST`` is the
costly error: it would stop a genuine command from working. Anything
unclear falls through to :data:`NEUTRAL`, which always means "proceed".

Independence
------------
Like the rest of :mod:`assistant.nlu`, this imports no tool, no router and
no application module. It reads a
:class:`~assistant.nlu.normalize.Normalized` and plain strings.
"""

from __future__ import annotations

from assistant.nlu.normalize import Normalized

#: The user is asking for something. Proceed.
REQUEST = "request"

#: The trigger word is being talked about, not asked for. Do not proceed.
MENTION = "mention"

#: No opinion. Proceed, because the default must never block a command.
NEUTRAL = "neutral"

#: Verbs and adverbs that ask for something. Position independent: a
#: request verb anywhere in the sentence is a request cue, because "please
#: could you", "can you" and "would you" all still ask.
#:
#: "want" and "need" are deliberately **absent**. They express a wish rather
#: than a request aimed at the assistant, and they are exactly the verbs that
#: introduce a competing action: "I want to log this information", "I need to
#: save this". Treating them as request cues would make that rule unreachable.
#: A sentence like "I need a joke" still works, because having no cue at all
#: is :data:`NEUTRAL`, and neutral always proceeds.
REQUEST_CUES = frozenset(
    {
        "tell", "give", "show", "check", "find", "fetch", "get", "search",
        "look", "read", "list", "make", "add", "create", "jot", "write",
        "remind", "play", "put", "set", "send", "report", "summary",
        "please", "pls",
    }
)

#: Words that open a question. Unlike a request verb these only count in
#: question position, which is what separates "is the weather good" from
#: "the weather is good".
INTERROGATIVE = frozenset(
    {
        "what", "how", "when", "where", "who", "whom", "whose", "why",
        "which", "is", "are", "was", "were", "do", "does", "did", "can",
        "could", "would", "will", "should", "has", "have", "had", "am",
    }
)

#: Words that mark speculation. A sentence that hedges is commenting, not
#: ordering.
HEDGE_CUES = frozenset(
    {"guess", "think", "suppose", "believe", "reckon", "apparently", "probably"}
)

#: Copulas. Present in a remark, absent in a request.
COPULA_CUES = frozenset({"is", "are", "was", "were", "seems", "sounded"})

#: Verbs that mean "write this down" rather than "look this up". When one
#: of these appears in a sentence that also mentions an information trigger,
#: the trigger is the object of the recording, not a lookup request.
RECORDING_VERBS = frozenset(
    {
        "log", "record", "save", "store", "file", "bookmark", "track",
        "print", "copy", "paste", "export", "archive", "scribble",
    }
)

#: Intent-specific rules, kept in one place so adding an intent is obvious.
#:
#: Each entry lists verbs that, when present, mean the trigger is being
#: mentioned rather than requested. Only intents with a genuine competing
#: action appear here; everything else relies on the general cues.
INTENT_RULES: dict[str, frozenset[str]] = {
    # "log this information" is a record request, not a Wikipedia lookup.
    "information": RECORDING_VERBS,
}

#: Openers that carry no meaning for framing and so are skipped when
#: looking for the first meaningful word.
FILLER_OPENERS = frozenset({"please", "pls", "so", "ok", "okay", "hey", "now"})

#: Past-tense verbs of speech, information transfer and discussion.
#:
#: Every entry is unambiguously past, so a sentence containing one is
#: reporting an event that already happened rather than asking for
#: something. This is the narrative guard:
#:
#:     "I mentioned the logs earlier"     mentioned
#:     "I heard the news this morning"    heard
#:     "my friend told me a joke"         told
#:     "we discussed the news yesterday"  discussed
#:
#: The list is deliberately closed and deliberately excludes near
#: neighbours that would cost real commands:
#:
#: * **"read"** is absent, because "can you read my history" is a genuine
#:   command.
#: * **"say"** is absent, because "what did I say" is the history alias.
#: * **"tell"** is a request cue and is handled much earlier.
#:
#: Past time expressions such as "earlier", "yesterday" and "this morning"
#: were evaluated as a signal on their own and **rejected**. A bare
#: "weather yesterday" is a fragment a user may well mean as a request, so
#: a time cue alone would misclassify it. Every one of the sentences the
#: narrative guard exists for already carries a narrative verb, so the
#: time expressions add nothing and cost a false block.
NARRATIVE_VERBS = frozenset(
    {
        "mentioned", "heard", "told", "discussed", "chatted", "talked",
        "reported", "announced", "explained", "confirmed", "commented",
    }
)


def _first_content_token(tokens: tuple[str, ...]) -> str:
    """Return the first word that is not a filler like "please" or "so"."""
    for token in tokens:
        if token not in FILLER_OPENERS:
            return token
    return ""


def _opens_a_question(tokens: tuple[str, ...]) -> bool:
    """True when the sentence starts as a question rather than a remark.

    "is the weather good" opens a question. "the weather is good" does
    not, even though both contain the same copula. The only difference is
    where the copula sits, which is why the check is positional.
    """
    return _first_content_token(tokens) in INTERROGATIVE


def assess(normalized: Normalized, intent: str) -> str:
    """Judge whether ``intent`` is being requested or merely mentioned.

    Args:
        normalized: the normalised utterance.
        intent: the intent the parser chose.

    Returns:
        :data:`REQUEST`, :data:`MENTION`, or :data:`NEUTRAL`. Only
        ``MENTION`` should stop the command; ``NEUTRAL`` must not.
    """
    tokens = normalized.tokens
    if not tokens:
        return NEUTRAL

    # 1. An explicit request cue, or a sentence that opens as a question, is
    #    the user asking. This is checked first so no later rule can talk a
    #    real command out of being run.
    if any(token in REQUEST_CUES for token in tokens):
        return REQUEST
    if _opens_a_question(tokens):
        return REQUEST

    # 2. Hedging marks a remark.
    if any(token in HEDGE_CUES for token in tokens):
        return MENTION

    # 3. A copula in a sentence that is not a question makes it a remark.
    #    "the weather is nice today" is a statement; "what is the weather"
    #    already returned above.
    if any(token in COPULA_CUES for token in tokens):
        return MENTION

    # 4. An intent-specific competing action, such as "log this information".
    competing = INTENT_RULES.get(intent)
    if competing and any(token in competing for token in tokens):
        return MENTION

    # 5. A past-tense verb of speech, transfer or discussion. Placed last on
    #    purpose: it can only turn a "no opinion" into a remark, so it can
    #    never override a request cue or a question above.
    if any(token in NARRATIVE_VERBS for token in tokens):
        return MENTION

    # 6. Nothing conclusive. Proceed: a bare command such as "weather" or
    #    "exit" carries no cue at all, and must keep working.
    return NEUTRAL
