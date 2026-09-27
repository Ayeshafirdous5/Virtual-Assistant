"""The measurement corpus for ``tell`` in the contextual framing layer.

This step is **measurement only**. No production rule is changed here. The
question being asked is narrow and is worth stating before any number is
read:

    ``tell`` is a request cue, so it is a request wherever it appears.

That single fact is what carries most real commands ("tell me a joke")
and it is also what produces the one remaining live framing gap, because
the same word appears inside sentences that are only *reporting* speech
("I heard you tell a joke"). This corpus measures both halves so a
future change can be judged against numbers instead of intuition.

Why a third corpus
------------------
The first framing corpus (:mod:`tests.nlu_framing_corpus`) is saturated
at 32 of 32 and the second (:mod:`tests.nlu_framing_adversarial_corpus`)
is a broad sweep that has already served its purpose. Neither isolates
one word. This one is a single-word study, so it can be re-run and read
without the rest of the framing picture getting in the way.

What is deliberately left out
-----------------------------
**Negation.** "don't tell me a joke" is already stopped upstream by the
NLU negation guard, which returns ``None`` before framing is consulted.
Including it here would measure a layer that is not the one under study,
so it is left to the adversarial corpus, where it is recorded as a
known-layering case.

Labelling
---------
Labels record what a user **reasonably meant**, never what the current
code returns. A case the rules get wrong stays mislabelled on purpose so
the measurement can surface it. Nothing here is tuned to make the suite
green, and the evaluation in ``test_nlu_framing_tell.py`` reports
mismatches instead of failing on them.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUEST = "request"
MENTION = "mention"
NEUTRAL = "neutral"

#: The user is asking the assistant to say something. Proceed.
CATEGORY_REQUEST = "tell-request"

#: The user is reporting speech that already happened. Do not proceed.
CATEGORY_NARRATIVE = "tell-narrative"

#: "tell" with a speaker other than the user: first, second, third person.
CATEGORY_SUBJECT = "tell-subject"

#: A question containing "tell". Both asking for an action and asking
#: about a past event live here, because a question must not be blocked.
CATEGORY_QUESTION = "tell-question"

#: A narrative clause followed by a real request in the same breath.
CATEGORY_AFTER_NARRATIVE = "narrative-then-tell"

#: Words that sit one edit away from "tell" and from the verbs around it.
#: A future fix that reaches too far shows up here first.
CATEGORY_TRAP = "tell-lexical-trap"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_REQUEST,
    CATEGORY_NARRATIVE,
    CATEGORY_SUBJECT,
    CATEGORY_QUESTION,
    CATEGORY_AFTER_NARRATIVE,
    CATEGORY_TRAP,
)


@dataclass(frozen=True)
class TellCase:
    """One utterance and the verdict a person would expect.

    Attributes:
        utterance: what the user says.
        intent: the intent it is judged against, because one framing rule
            is intent specific and the rest are not.
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
# 1. Genuine requests. "tell" is aimed at the assistant and asks for
#    something to happen now.
# ----------------------------------------------------------------------
TELL_REQUESTS: list[TellCase] = [
    TellCase("tell me a joke", "jokes", REQUEST, CATEGORY_REQUEST,
             "'tell' is a request cue addressed to the assistant"),
    TellCase("tell me the news", "news", REQUEST, CATEGORY_REQUEST,
             "'tell' is a request cue"),
    TellCase("tell me about the weather", "weather", REQUEST, CATEGORY_REQUEST,
             "'tell' is a request cue"),
    TellCase("can you tell me the news?", "news", REQUEST, CATEGORY_REQUEST,
             "a question-shaped request; 'can' opens a question"),
    TellCase("please tell me the news", "news", REQUEST, CATEGORY_REQUEST,
             "'please' and 'tell' are both request cues"),
    TellCase("tell me what you heard", "news", REQUEST, CATEGORY_REQUEST,
             "asks for a report; 'what' opens a question"),
    TellCase("tell me about Python", "information", REQUEST, CATEGORY_REQUEST,
             "'tell' is a request cue"),
    TellCase("tell me a joke please", "jokes", REQUEST, CATEGORY_REQUEST,
             "the cue is position independent"),
    TellCase("tell me what's on the news", "news", REQUEST, CATEGORY_REQUEST,
             "the contraction expands to 'what is'"),
    TellCase("tell me again", "jokes", REQUEST, CATEGORY_REQUEST,
             "a bare continuation of an earlier request"),
]

# ----------------------------------------------------------------------
# 2. Narrative constructions. The user is describing speech that
#    already happened, including their own report of the assistant.
# ----------------------------------------------------------------------
TELL_NARRATIVES: list[TellCase] = [
    TellCase("I heard you tell a joke", "jokes", MENTION, CATEGORY_NARRATIVE,
             "describing what the assistant did; the one live gap"),
    TellCase("I heard him tell a joke", "jokes", MENTION, CATEGORY_NARRATIVE,
             "a third party retelling, not an ask"),
    TellCase("I heard her tell the news", "news", MENTION, CATEGORY_NARRATIVE,
             "a report of something already said"),
    TellCase("I saw you tell a joke", "jokes", MENTION, CATEGORY_NARRATIVE,
             "'saw' plus 'tell' is still an observation"),
    TellCase("my friend told me a joke", "jokes", MENTION, CATEGORY_NARRATIVE,
             "'told' is past, not a cue"),
    TellCase("you told me a joke earlier", "jokes", MENTION, CATEGORY_NARRATIVE,
             "second person, past"),
    TellCase("she told me the news yesterday", "news", MENTION, CATEGORY_NARRATIVE,
             "third person, past"),
    TellCase("we talked about what he told me", "jokes", MENTION,
             CATEGORY_NARRATIVE, "the joke is quoted, not requested"),
    TellCase("I mentioned the joke you told me", "jokes", MENTION,
             CATEGORY_NARRATIVE, "both verbs are past"),
    TellCase("you told me that yesterday", "news", MENTION, CATEGORY_NARRATIVE,
             "a reference to an earlier remark"),
]

# ----------------------------------------------------------------------
# 3. Who is doing the telling. First person, second person, third
#    person, past and present.
# ----------------------------------------------------------------------
TELL_SUBJECTS: list[TellCase] = [
    TellCase("I told you the news yesterday", "news", MENTION, CATEGORY_SUBJECT,
             "first person, past"),
    TellCase("I tell you a joke every morning", "jokes", MENTION,
             CATEGORY_SUBJECT,
             "first person, present: a habit, not an order"),
    TellCase("you told me the joke", "jokes", MENTION, CATEGORY_SUBJECT,
             "second person, past"),
    TellCase("you tell me that every day", "jokes", MENTION, CATEGORY_SUBJECT,
             "second person, present: an observation about behaviour"),
    TellCase("she told me about the weather", "weather", MENTION,
             CATEGORY_SUBJECT, "third person, past"),
    TellCase("he told me the news", "news", MENTION, CATEGORY_SUBJECT,
             "third person, past"),
    TellCase("my brother tells me jokes", "jokes", MENTION, CATEGORY_SUBJECT,
             "third person, present: 'tells' is not a request cue"),
    TellCase("I told you a joke", "jokes", MENTION, CATEGORY_SUBJECT,
             "first person, past"),
]

# ----------------------------------------------------------------------
# 4. Questions containing "tell". Two different questions are here on
#    purpose: asking the assistant to do something now, and asking about
#    something said in the past. Both are questions, and a question that
#    gets blocked is a command that stopped working, so both are
#    labelled REQUEST.
# ----------------------------------------------------------------------
TELL_QUESTIONS: list[TellCase] = [
    # 4a. Asking the assistant to perform an action.
    TellCase("can you tell me a joke?", "jokes", REQUEST, CATEGORY_QUESTION,
             "a question addressed to the assistant"),
    TellCase("would you tell me the news?", "news", REQUEST, CATEGORY_QUESTION,
             "'would' opens a question"),
    TellCase("could you tell me a joke, please?", "jokes", REQUEST,
             CATEGORY_QUESTION, "a modal question plus a cue"),
    # 4b. Asking about a past event. Still a question, still a request.
    TellCase("what did I tell you earlier?", "history", REQUEST,
             CATEGORY_QUESTION,
             "asks about a past event; a question is not a remark"),
    TellCase("what did she tell you?", "news", REQUEST, CATEGORY_QUESTION,
             "asks about a past event"),
    TellCase("did he tell you the news?", "news", REQUEST, CATEGORY_QUESTION,
             "'did' opens a question"),
    TellCase("when did I tell you that?", "history", REQUEST, CATEGORY_QUESTION,
             "asks about a past event"),
    TellCase("did I tell you about the weather?", "weather", REQUEST,
             CATEGORY_QUESTION, "asks whether it was said"),
    TellCase("who told you the news?", "news", REQUEST, CATEGORY_QUESTION,
             "a question, not a statement"),
    TellCase("have I told you about this?", "history", REQUEST,
             CATEGORY_QUESTION, "'have' opens a question"),
    TellCase("did you hear me tell the news?", "news", REQUEST, CATEGORY_QUESTION,
             "asks about a past event"),
]

# ----------------------------------------------------------------------
# 5. A narrative clause and then a real request. The remark half must
#    not swallow the request half.
# ----------------------------------------------------------------------
TELL_AFTER_NARRATIVE: list[TellCase] = [
    TellCase("I heard you tell a joke, tell me another one", "jokes", REQUEST,
             CATEGORY_AFTER_NARRATIVE, "the request half is explicit"),
    TellCase("she told me the news, now tell me today's news", "news", REQUEST,
             CATEGORY_AFTER_NARRATIVE, "'tell' is the ask"),
    TellCase("I heard you tell that story, can you tell it again?", "jokes",
             REQUEST, CATEGORY_AFTER_NARRATIVE, "a question plus a cue"),
    TellCase("you told me a joke earlier, tell me another", "jokes", REQUEST,
             CATEGORY_AFTER_NARRATIVE, "the narrative half is past"),
    TellCase("I heard him tell a joke, do you have another one?", "jokes",
             REQUEST, CATEGORY_AFTER_NARRATIVE,
             "no 'tell' in the second half; the question carries it"),
    TellCase("my friend told me a joke, tell me a better one", "jokes", REQUEST,
             CATEGORY_AFTER_NARRATIVE, "'tell' must win over 'told'"),
]

# ----------------------------------------------------------------------
# 6. Lexical traps. The verbs one edit away from "tell" and from the
#    verbs that surround it. A fix that reaches too far shows up here
#    first, so each case says which word is doing the work.
# ----------------------------------------------------------------------
TELL_TRAPS: list[TellCase] = [
    TellCase("I heard you telling a joke", "jokes", MENTION, CATEGORY_TRAP,
             "'telling' is not the request cue 'tell'"),
    TellCase("I remember you telling me a joke", "jokes", MENTION, CATEGORY_TRAP,
             "a past event, though no narrative verb marks it either"),
    TellCase("you told me that", "news", MENTION, CATEGORY_TRAP,
             "second person, past"),
    TellCase("he said the news was good", "news", MENTION, CATEGORY_TRAP,
             "'said' with a copula is a statement"),
    TellCase("I mentioned the news to him", "news", MENTION, CATEGORY_TRAP,
             "'mentioned' is past"),
    TellCase("she was telling me about the weather", "weather", MENTION,
             CATEGORY_TRAP, "'was' makes it a statement"),
    TellCase("I heard you say the weather is nice", "weather", MENTION,
             CATEGORY_TRAP, "a remark about what was said"),
    TellCase("you mentioned telling me a joke", "jokes", MENTION, CATEGORY_TRAP,
             "'mentioned' plus 'telling'"),
    TellCase("I saw him tell the news", "news", MENTION, CATEGORY_TRAP,
             "an observation, not a request"),
    TellCase("he told me to tell you a joke", "jokes", MENTION, CATEGORY_TRAP,
             "the 'tell' sits inside reported speech"),
    TellCase("I heard the news you told me about", "news", MENTION,
             CATEGORY_TRAP, "'told' is the verb; the news is its object"),
    TellCase("she told me the weather was cold", "weather", MENTION,
             CATEGORY_TRAP, "reported speech carrying a copula"),
]

#: Every case, in a fixed order so a report always reads the same way.
CORPUS: list[TellCase] = (
    TELL_REQUESTS
    + TELL_NARRATIVES
    + TELL_SUBJECTS
    + TELL_QUESTIONS
    + TELL_AFTER_NARRATIVE
    + TELL_TRAPS
)


def by_category(category: str) -> list[TellCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]

