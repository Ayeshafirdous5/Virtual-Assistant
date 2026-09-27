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

Questions
---------
A question is recognised by **inversion**: the verb comes before its
subject. A remark never does that, which is what lets the layer tell
"is the weather good" from "the weather is good" without reading either
one for keywords. Inversion is also positional, so a question counts
wherever it sits in the sentence, not only at the front:

    "what is the weather"                      opens as a question
    "we discussed the weather, what is it now"  asks at the end

Both are requests, and both outrank every remark rule below them.

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

#: :data:`INTERROGATIVE` split into the two halves a question is built
#: from. Together these two sets **partition** :data:`INTERROGATIVE`, so
#: the leading-question check and the trailing-question check can never
#: disagree about which words are interrogative at all.
#:
#: The test applied to them is inversion, not vocabulary: a question puts
#: the verb before its subject, and a remark does not.
WH_QUESTIONS = frozenset(
    {
        "what", "how", "when", "where", "who", "whom", "whose", "why",
        "which",
    }
)

#: The other half of :data:`INTERROGATIVE`: verbs that move ahead of a
#: subject to form a question. A declarative keeps the subject first.
INVERTING_AUXILIARIES = frozenset(
    {
        "is", "are", "was", "were", "am", "do", "does", "did", "can",
        "could", "will", "would", "should", "has", "have", "had",
    }
)

#: Subjects a question can be inverted onto. Deliberately personal
#: pronouns only. A determiner is not enough evidence, because "is the"
#: also opens remarks such as "the weather is nice" once the subject is
#: a noun phrase. A pronoun after an auxiliary is: "is it", "did you",
#: "was he", "can we".
INVERTED_SUBJECTS = frozenset({"i", "you", "he", "she", "it", "we", "they"})

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

# ----------------------------------------------------------------------
# The reported-"tell" guard
# ----------------------------------------------------------------------
#: ``tell`` is a request cue, so it is a request wherever it appears. That
#: is what carries "tell me a joke", and it is also what made these eight
#: sentences look like requests when they are only reporting speech:
#:
#:     I heard you tell a joke          I heard him tell a joke
#:     I heard her tell the news        I saw you tell a joke
#:     I saw him tell the news          I tell you a joke every morning
#:     you tell me that every day       he told me to tell you a joke
#:
#: Each of those has a **structural** mark that a genuine request never
#: has, and that is the only thing this guard looks at. It never counts a
#: bare "tell", and it never guesses from position alone.
PERCEPTION_VERBS = frozenset({"heard", "saw"})

#: Subjects that can be the one "telling" after a perception verb.
#:
#: Deliberately absent is **"me"**: "did you hear me tell the news?" is a
#: genuine question, and the question rule must be free to win. The
#: plural and reflexive subjects are unlisted because no case in the tell
#: corpus attests them, and an unattested subject is a guess.
REPORTED_SUBJECTS = frozenset({"you", "him", "her"})

#: Verbs that take a "to"-infinitive reporting a command that was given.
#: "told" is the member the corpus attests; "asked" takes the same
#: complement and is listed for the same reason.
REPORTING_VERBS = frozenset({"told", "asked"})

#: A subject in first position makes the sentence declarative, so "tell"
#: is no longer the imperative. This is what separates "you tell me that
#: every day" from "tell me the news".
DECLARATIVE_SUBJECTS = frozenset({"i", "you"})

#: Frequency words. A subject plus "tell" is only clear evidence when it
#: also describes a habit, which is what the two attested declarative
#: cases have in common. A bare "I tell you a joke" carries no such
#: evidence and is therefore left exactly as it is today.
HABITUAL_MARKERS = frozenset({"every", "always", "usually", "often", "sometimes"})


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
#: * **"tell"** is absent, because it is a request cue. It is also a
#:   reporting verb, so the request cue is *vetoed* where the structure
#:   says the sentence is reporting speech; see
#:   :func:`_reports_speech`.
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


def _has_trailing_question(tokens: tuple[str, ...]) -> bool:
    """True when a question clause begins part way through the sentence.

    "we discussed the weather, what's it like now" asks something, but it
    opens with a narrative clause, so :func:`_opens_a_question` never sees
    the question and the remark rules below would swallow the request. What
    identifies the question is not the question *word* but the inversion
    inside the clause:

        "what is it"    a question word, an auxiliary, then a subject
        "is it"         an auxiliary, then a subject

    No remark inverts, so "I mentioned what I heard" and "we discussed
    what was funny" keep returning :data:`MENTION` even though both carry a
    question word. That is the difference between this and a keyword
    search for "what", "is" or "did".

    Only positions after the first token are examined, so a leading
    question remains :func:`_opens_a_question`'s business and the
    behaviour of a sentence that opens as one is untouched.
    """
    for index in range(1, len(tokens) - 1):
        opener = tokens[index]
        following = tokens[index + 1]
        if opener in INVERTING_AUXILIARIES and following in INVERTED_SUBJECTS:
            return True
        if (
            opener in WH_QUESTIONS
            and following in INVERTING_AUXILIARIES
            and tokens[index + 2] in INVERTED_SUBJECTS
        ):
            return True
    return False


def _reports_speech(tokens: tuple[str, ...], index: int) -> bool:
    """True when the ``tell`` at ``index`` reports speech instead of asking.

    This is a **veto on one request cue**, never a rule of its own. It
    answers a single question: is this particular "tell" describing an
    event that already happened? If the answer is not clearly yes, this
    returns ``False`` and the cue counts exactly as it did before.

    Three structures, each taken from a case the tell corpus measures as
    wrong, and each absent from every case it measures as right:

    1. a perception verb and its subject, then "tell"
       ("I heard you tell a joke", "I saw him tell the news")
    2. a reporting verb, then a "to"-infinitive containing "tell"
       ("he told me to tell you a joke")
    3. a subject in first position, then "tell", then a frequency word
       ("I tell you a joke every morning")

    What is deliberately *not* here matters as much. A pronoun before
    "tell" proves nothing: "can you tell me a joke?" has one, and so does
    "what did she tell you?". Neither is reported speech, and both must
    keep working, so no rule of that shape can exist. That is why each
    branch needs a second, independent mark before it will fire.
    """
    if tokens[index] != "tell":
        return False

    # 1. Reported perception: "I heard you tell a joke".
    if (
        index >= 2
        and tokens[index - 2] in PERCEPTION_VERBS
        and tokens[index - 1] in REPORTED_SUBJECTS
    ):
        return True

    # 2. A command that was reported rather than given: "he told me to
    #    tell you a joke". The reporting verb has to be there; "I want to
    #    tell you a joke" has the same infinitive and is left alone.
    if (
        index >= 2
        and tokens[index - 1] == "to"
        and any(token in REPORTING_VERBS for token in tokens[: index - 1])
    ):
        return True

    # 3. A habitual statement rather than an order: "you tell me that
    #    every day". A first-position subject makes the clause
    #    declarative, and the frequency word is the second mark.
    if (
        index == 1
        and tokens[0] in DECLARATIVE_SUBJECTS
        and any(token in HABITUAL_MARKERS for token in tokens)
    ):
        return True

    return False


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

    # 1. An explicit request cue, or a question, is the user asking. This
    #    is checked first so no later rule can talk a real command out of
    #    being run. A question counts in two positions: opening the
    #    utterance, and opening a clause later on ("we discussed the
    #    weather, what is it now"). The remark rules below all sit after
    #    this one, which is what keeps an explicit ask from being talked
    #    out of running.
    #
    #    "tell" is the one cue a reported-speech construction can veto,
    #    because it is the one cue that is also an ordinary reporting verb.
    #    Every other cue, and the question rules, are untouched.
    if any(
        token in REQUEST_CUES and not _reports_speech(tokens, index)
        for index, token in enumerate(tokens)
    ):
        return REQUEST
    if _opens_a_question(tokens):
        return REQUEST
    if _has_trailing_question(tokens):
        return REQUEST

    # 1b. The vetoed "tell" is now the only evidence left, and it is
    #     positive evidence of a remark, so it decides here. Placed after
    #     the question rules on purpose: "did you hear me tell the news?"
    #     and "I heard you tell a joke, do you have another one?" are both
    #     still questions, and a question outranks a reporting verb.
    if any(
        token == "tell" and _reports_speech(tokens, index)
        for index, token in enumerate(tokens)
    ):
        return MENTION

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
