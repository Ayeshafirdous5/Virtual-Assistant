"""The measurement corpus for the verb forms of ``tell``.

This step is **measurement only**. No production rule is changed here.

Why this study exists
---------------------
Step 7 closed the eight leaks caused by ``tell`` being an unconditional
request cue. Two leaks survived, and they are of a completely different
kind:

    "my brother tells me jokes"
    "I remember you telling me a joke"

Neither contains the token ``tell``. They contain **"tells"** and
**"telling"**, and those forms sit in *neither* :data:`REQUEST_CUES` nor
:data:`NARRATIVE_VERBS`. So framing has no opinion at all about them and
returns :data:`NEUTRAL`, which always proceeds. A future step therefore
has to decide whether "tells" and "telling" should join the narrative
guard, and this corpus is the evidence for that decision.

The four forms, and where each one currently stands
---------------------------------------------------
=========  ===================  =========================
form       in the production     how framing reads it
           lists
=========  ===================  =========================
tell       REQUEST_CUES          a request cue: a request
told       NARRATIVE_VERBS       a remark: blocked
tells      in neither            no opinion: neutral
telling    in neither            no opinion: neutral
=========  ===================  =========================

"tells" and "telling" are the gap. They are the present third person and
the gerund of a verb the layer already treats as a remark in the past,
so the inconsistency is real, but **every** rule change here has a cost:
"telling" appears inside genuine requests too, as in "keep telling me
more". That is the tension this corpus is built to measure rather than
guess at.

What is deliberately left out
-----------------------------
**Negation.** "don't tell me a joke" is stopped upstream by the NLU
negation guard in :mod:`assistant.nlu.scoring`, which drops the trigger
before the parser can return an intent. Measuring it here would report a
layer that never sees the sentence. It is recorded in the adversarial
corpus as a known-layering case, and this module reports the same
distinction for every case it does hold.

Labelling
---------
Labels record what a user **reasonably meant**, never what the code
returns. Where the two differ the case stays mislabelled on purpose so
the measurement can surface it. Nothing here is tuned to make the suite
green.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUEST = "request"
MENTION = "mention"
NEUTRAL = "neutral"

#: Present third person reporting speech. "my brother tells me jokes".
CATEGORY_TELLS = "narrative-tells"

#: The gerund, reporting speech in progress or recalled. "she was
#: telling me the news".
CATEGORY_TELLING = "narrative-telling"

#: Commands aimed at the assistant, including the ones that use a form
#: other than the bare imperative.
CATEGORY_REQUEST_FORM = "request-other-form"

#: Questions carrying one of the four forms.
CATEGORY_QUESTION = "verb-form-question"

#: The verbs one or two letters away, which a future rule could reach.
CATEGORY_TRAP = "nearby-verb-trap"

#: A remark, then a real request in the same breath.
CATEGORY_AFTER_NARRATIVE = "narrative-then-request"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_TELLS,
    CATEGORY_TELLING,
    CATEGORY_REQUEST_FORM,
    CATEGORY_QUESTION,
    CATEGORY_TRAP,
    CATEGORY_AFTER_NARRATIVE,
)

#: The four forms this corpus studies, and where the production lists
#: put them. Kept here as data so a test can assert the whole table at
#: once instead of one word at a time.
VERB_FORMS: dict[str, str] = {
    "tell": "request-cue",
    "told": "narrative-verb",
    "tells": "unlisted",
    "telling": "unlisted",
}


@dataclass(frozen=True)
class VerbFormCase:
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
# A. Present third person. "tells" is unlisted, so these are the
#    clearest test of what the layer does with an unlisted form.
# ----------------------------------------------------------------------
NARRATIVE_TELLS: list[VerbFormCase] = [
    VerbFormCase("my brother tells me jokes", "jokes", MENTION, CATEGORY_TELLS,
                 "a habit someone else has; 'tells' is unlisted"),
    VerbFormCase("she tells me the news", "news", MENTION, CATEGORY_TELLS,
                 "present third person reporting, not an ask"),
    VerbFormCase("he tells everyone the story", "jokes", MENTION, CATEGORY_TELLS,
                 "present third person; the object is the story"),
    VerbFormCase("my friend tells me about the weather", "weather", MENTION,
                 CATEGORY_TELLS, "'about' makes the weather a topic"),
    VerbFormCase("he tells me the weather is fine", "weather", MENTION,
                 CATEGORY_TELLS, "a reported statement, carried by the copula"),
    VerbFormCase("she tells me what she had for lunch", "jokes", MENTION,
                 CATEGORY_TELLS, "a report of earlier speech"),
    VerbFormCase("my dad tells me the same joke daily", "jokes", MENTION,
                 CATEGORY_TELLS, "a habit, carried by the frequency word"),
    VerbFormCase("the teacher tells us the weather", "weather", MENTION,
                 CATEGORY_TELLS, "third person with a plural object"),
]


# ----------------------------------------------------------------------
# B. The gerund. Reported speech in progress, or recalled. This form is
#    the hardest to place, because it also appears in real commands.
# ----------------------------------------------------------------------
NARRATIVE_TELLING: list[VerbFormCase] = [
    VerbFormCase("I remember you telling me a joke", "jokes", MENTION,
                 CATEGORY_TELLING, "recalling an earlier event"),
    VerbFormCase("she was telling me the news", "news", MENTION,
                 CATEGORY_TELLING, "'was' makes it a past continuous"),
    VerbFormCase("I remember him telling that story", "jokes", MENTION,
                 CATEGORY_TELLING, "recalled speech"),
    VerbFormCase("we were talking about her telling me the news", "news",
                 MENTION, CATEGORY_TELLING,
                 "a remark about a remark, twice over"),
    VerbFormCase("you were telling me a joke", "jokes", MENTION,
                 CATEGORY_TELLING, "'were' reports an interrupted event"),
    VerbFormCase("I heard her telling the story", "jokes", MENTION,
                 CATEGORY_TELLING, "'heard' plus the gerund"),
    VerbFormCase("they kept telling me the same thing", "news", MENTION,
                 CATEGORY_TELLING, "'kept' marks a repeated past event"),
    VerbFormCase("I liked you telling that story", "jokes", MENTION,
                 CATEGORY_TELLING, "a reaction to past speech"),
]

# ----------------------------------------------------------------------
# C. Commands aimed at the assistant that do not use the bare
#    imperative. Any rule for "tells" or "telling" has to spare these.
# ----------------------------------------------------------------------
REQUEST_OTHER_FORM: list[VerbFormCase] = [
    VerbFormCase("tell me a joke", "jokes", REQUEST, CATEGORY_REQUEST_FORM,
                 "the plainest form, and the one that must never break"),
    VerbFormCase("can you tell me the news?", "news", REQUEST,
                 CATEGORY_REQUEST_FORM, "'can' opens a question"),
    VerbFormCase("please tell me about the weather", "weather", REQUEST,
                 CATEGORY_REQUEST_FORM, "'please' plus the cue"),
    VerbFormCase("what are you telling me?", "news", REQUEST,
                 CATEGORY_REQUEST_FORM,
                 "a question using the gerund; a question must not block"),
    VerbFormCase("keep telling me more", "jokes", REQUEST, CATEGORY_REQUEST_FORM,
                 "the closest call here: the gerund, but clearly an order"),
    VerbFormCase("stop telling me that", "news", MENTION, CATEGORY_REQUEST_FORM,
                 "asking the assistant to stop, not to speak"),
    VerbFormCase("tell me the news please", "news", REQUEST,
                 CATEGORY_REQUEST_FORM, "the cue is position independent"),
    VerbFormCase("I want you to tell me the news", "news", REQUEST,
                 CATEGORY_REQUEST_FORM, "a stated wish, with the cue"),
]

# ----------------------------------------------------------------------
# D. Questions carrying one of the four forms. Three groups are kept
#    apart: asking the assistant to act, asking about a past event, and
#    questions that merely contain the word. All three are requests,
#    because a question that is blocked is a command that broke.
# ----------------------------------------------------------------------
VERB_FORM_QUESTIONS: list[VerbFormCase] = [
    # D1. Asking the assistant to act.
    VerbFormCase("will you tell me the weather?", "weather", REQUEST,
                 CATEGORY_QUESTION, "'will' opens a question"),
    VerbFormCase("could you tell me a joke?", "jokes", REQUEST,
                 CATEGORY_QUESTION, "a modal question"),
    VerbFormCase("why won't you tell me the news?", "news", REQUEST,
                 CATEGORY_QUESTION, "a complaint, still a question"),
    # D2. Asking about a past event.
    VerbFormCase("what did he tell you?", "news", REQUEST, CATEGORY_QUESTION,
                 "asks about a past event"),
    VerbFormCase("did she tell you the news?", "news", REQUEST,
                 CATEGORY_QUESTION, "asks whether it was said"),
    VerbFormCase("what is he telling me?", "weather", REQUEST,
                 CATEGORY_QUESTION, "asks about the current utterance"),
    # D3. Contains the word without asking the assistant to act.
    VerbFormCase("who tells you the weather?", "weather", REQUEST,
                 CATEGORY_QUESTION, "asks who, not an order"),
    VerbFormCase("is she telling me the truth about the news?", "news",
                 REQUEST, CATEGORY_QUESTION, "a yes/no question"),
    VerbFormCase("how long has he been telling you jokes?", "jokes", REQUEST,
                 CATEGORY_QUESTION, "asks about a duration"),
]


# ----------------------------------------------------------------------
# E. The verbs one or two letters away. A future rule for "tells" or
#    "telling" could easily reach for these, so each case is here to
#    show what a too-broad rule would cost.
# ----------------------------------------------------------------------
NEARBY_VERB_TRAPS: list[VerbFormCase] = [
    VerbFormCase("my friend told me a joke", "jokes", MENTION,
                 CATEGORY_TRAP, "'told' is already a narrative verb"),
    VerbFormCase("he said the news was good", "news", MENTION, CATEGORY_TRAP,
                 "'said' with a copula is a statement"),
    VerbFormCase("she mentioned the weather", "weather", MENTION,
                 CATEGORY_TRAP, "'mentioned' is a narrative verb"),
    VerbFormCase("he explained the joke to me", "jokes", MENTION,
                 CATEGORY_TRAP, "'explained' is a narrative verb"),
    VerbFormCase("I heard the news this morning", "news", MENTION,
                 CATEGORY_TRAP, "'heard' is a narrative verb"),
    VerbFormCase("we discussed the weather yesterday", "weather", MENTION,
                 CATEGORY_TRAP, "'discussed' is a narrative verb"),
    VerbFormCase("he was telling me about the weather", "weather", MENTION,
                 CATEGORY_TRAP, "'was' already catches this one"),
    VerbFormCase("they told me the news earlier", "news", MENTION,
                 CATEGORY_TRAP, "past tense, already correct"),
    VerbFormCase("the weather is nice today", "weather", MENTION,
                 CATEGORY_TRAP, "a plain remark, carried by the copula"),
    VerbFormCase("my notes are on the desk", "notes", MENTION, CATEGORY_TRAP,
                 "a remark, carried by the copula"),
]

# ----------------------------------------------------------------------
# F. A remark, then a real request in the same breath. The request half
#    must survive whatever is done about the narrative half.
# ----------------------------------------------------------------------
NARRATIVE_THEN_REQUEST: list[VerbFormCase] = [
    VerbFormCase("my brother tells me jokes, tell me one too", "jokes",
                 REQUEST, CATEGORY_AFTER_NARRATIVE,
                 "the second clause is a bare request"),
    VerbFormCase("I remember you telling me that, explain it again", "information",
                 REQUEST, CATEGORY_AFTER_NARRATIVE,
                 "'explain' is a request cue of its own"),
    VerbFormCase("she told me the news, now tell me today's news", "news",
                 REQUEST, CATEGORY_AFTER_NARRATIVE, "'tell' is the ask"),
    VerbFormCase("she was telling me the news, tell me the latest", "news",
                 REQUEST, CATEGORY_AFTER_NARRATIVE,
                 "the gerund half is a remark, the request half is not"),
    VerbFormCase("he tells me the weather, check it for me", "weather",
                 REQUEST, CATEGORY_AFTER_NARRATIVE, "'check' is the cue"),
    VerbFormCase("you were telling me a joke, tell me another", "jokes",
                 REQUEST, CATEGORY_AFTER_NARRATIVE,
                 "an interrupted remark, then a request"),
]

#: Every case, in a fixed order so a report always reads the same way.
CORPUS: list[VerbFormCase] = (
    NARRATIVE_TELLS
    + NARRATIVE_TELLING
    + REQUEST_OTHER_FORM
    + VERB_FORM_QUESTIONS
    + NEARBY_VERB_TRAPS
    + NARRATIVE_THEN_REQUEST
)


def by_category(category: str) -> list[VerbFormCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]

