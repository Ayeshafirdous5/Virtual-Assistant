"""The measurement corpus for preference questions about the assistant.

This step is **measurement only**. No production rule is changed and no
fix is proposed in the corpus itself.

The one sentence that is left
-----------------------------
Steps 17 and 20 closed fourteen dangerous executions between them and
left one::

    "do you want to quit"   ->  system / clear  ->  the app terminates

This corpus asks whether that sentence is **one** or a **family**, and
what separates the family from the commands that must keep working.

What the measurement found
--------------------------
It is a family, and it is larger than one verb. ``do you want to``
reaches every system trigger:

    do you want to quit          do you want to exit
    do you want to goodbye       do you want to shut down

and the same frame reaches the bare form as well: *"you want to quit"*
and *"you want to exit"*.

The separating structure is **which word is the subject**. Every member
of the family has ``you`` in the subject position of a want-frame, and
``you`` is the one pronoun
:data:`~assistant.nlu.scoring.SELF_OTHER_SUBJECTS` deliberately omits,
because *"you can quit now"* and *"can you exit"* address the assistant
and must work. Every other subject is already refused by that rule:

    "does she want to quit"     refused: "she"
    "do they want to quit"      refused: "they"
    "do you want me to quit"    refused: "me"

So the gap is narrow and it is a specific interaction, not a missing
word: ``you`` is right to be allowed as an **addressee** and wrong to be
allowed as a **subject**.

Why a narrow fix is not obvious
-------------------------------
The obvious one-token fix, adding ``to`` to the Step 20 apposition
markers, is **not safe**, and this corpus carries the evidence. Blocking
on ``to`` before the trigger would break working commands and fix other
bugs at the same time:

    "time to quit"        currently works   would break
    "ready to quit"       currently works   would break
    "try to quit"         currently works   would break
    "plans to quit"       currently WRONG  would be fixed
    "nothing to quit over" currently WRONG  would be fixed

The signal is not separable on the word alone. :data:`AMBIGUOUS_CASES`
holds the sentences that straddle it, and :data:`~assistant.nlu.framing`
is measured here too, because it is the obvious alternative owner and
this corpus shows it actively **approving** every member of the family.

Labelling
---------
Labels are recorded by measurement, not by intention, and the two are
separated on purpose. :attr:`PreferenceCase.intent_note` records what a
person might have meant; :attr:`PreferenceCase.classification` records
what the application actually did. A sentence that is harmless to mean
and fatal to execute is labelled :data:`DANGEROUS_EXECUTION`, never
:data:`AMBIGUOUS`.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The assistant was asked to do something and did.
SAFE_COMMAND = "safe-command"

#: Something was said, and nothing ran.
SAFE_BLOCK = "safe-block"

#: :class:`~assistant.tools.system.SystemTool` was reached. Always this,
#: whatever the natural-language reading, because the cost is the same.
DANGEROUS_EXECUTION = "dangerous-execution"

#: Two candidates tied, so the assistant asked instead of guessing.
AMBIGUOUS = "ambiguous"

#: Nothing matched, for better or worse.
UNSUPPORTED = "unsupported"

CLASSIFICATIONS: tuple[str, ...] = (
    SAFE_COMMAND,
    SAFE_BLOCK,
    DANGEROUS_EXECUTION,
    AMBIGUOUS,
    UNSUPPORTED,
)

#: Preference questions, the family Step 21 set out to size.
CATEGORY_PREFERENCE = "preference-question"

#: Questions aimed at the assistant that must keep working.
CATEGORY_ASSISTANT_DIRECTED = "assistant-directed"

#: The user asking themselves for permission.
CATEGORY_SELF_PERMISSION = "self-permission"

#: Unambiguous orders.
CATEGORY_DIRECT = "direct-command"

#: Ordinary conversation about quitting or leaving.
CATEGORY_CONVERSATIONAL = "conversational"

#: Questions about a word's meaning, closed in Step 20.
CATEGORY_SEMANTIC = "semantic"

#: The same frame with every pronoun.
CATEGORY_SUBJECT = "subject-variant"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_PREFERENCE,
    CATEGORY_ASSISTANT_DIRECTED,
    CATEGORY_SELF_PERMISSION,
    CATEGORY_DIRECT,
    CATEGORY_CONVERSATIONAL,
    CATEGORY_SEMANTIC,
    CATEGORY_SUBJECT,
)


@dataclass(frozen=True)
class PreferenceCase:
    """One utterance and everything the measurement records about it.

    Attributes:
        utterance: what the user says.
        category: which group it belongs to.
        note: why the case is here.
        intent_note: what a person might reasonably have meant, which is
            kept separate from what the application did.
    """

    utterance: str
    category: str
    note: str = ""
    intent_note: str = ""

    @property
    def describe(self) -> str:
        return f"{self.utterance!r} [{self.category}]"

    @property
    def should_execute(self) -> bool:
        """True when this sentence really is an order to the assistant.

        The expectation column. Reaching ``SystemTool`` is only dangerous
        when the sentence was never an order, so the two are combined in
        :func:`~tests.test_nlu_system_preference_questions.classify`.
        """
        return self.category in EXPECT_RUN_CATEGORIES

# ----------------------------------------------------------------------
# A. The family. "do you want to" reaches every system trigger, and the
#    bare declarative form does too.
# ----------------------------------------------------------------------
PREFERENCE_QUESTIONS: list[PreferenceCase] = [
    PreferenceCase("do you want to quit", CATEGORY_PREFERENCE,
                   "the one sentence left open by Step 20",
                   "asking whether the assistant should stop"),
    PreferenceCase("do you want to exit", CATEGORY_PREFERENCE,
                   "same frame, different trigger", "asking permission"),
    PreferenceCase("do you want to shut down", CATEGORY_PREFERENCE,
                   "reaches the multi-word trigger too", "asking permission"),
    PreferenceCase("do you want to goodbye", CATEGORY_PREFERENCE,
                   "ungrammatical, and it still fires", "almost nothing"),
    PreferenceCase("do you want to bye", CATEGORY_PREFERENCE,
                   "ungrammatical, and it still fires", "almost nothing"),
    PreferenceCase("you want to quit", CATEGORY_PREFERENCE,
                   "the declarative form, no question mark needed",
                   "stating a fact about the assistant"),
    PreferenceCase("you want to exit", CATEGORY_PREFERENCE,
                   "the declarative form again", "stating a fact"),
    PreferenceCase("do you really want to quit", CATEGORY_PREFERENCE,
                   "the intensifier does not change the outcome",
                   "still a question"),
]

# ----------------------------------------------------------------------
# B. Assistant-directed questions. The load-bearing cases: the guard
#    must keep allowing these.
# ----------------------------------------------------------------------
ASSISTANT_DIRECTED: list[PreferenceCase] = [
    PreferenceCase("can you exit", CATEGORY_ASSISTANT_DIRECTED,
                   "a modal in front of you", "a real order"),
    PreferenceCase("would you quit now", CATEGORY_ASSISTANT_DIRECTED,
                   "a modal in front of you", "a real order"),
    PreferenceCase("will you quit", CATEGORY_ASSISTANT_DIRECTED,
                   "a modal in front of you", "a real order"),
    PreferenceCase("can you shut down", CATEGORY_ASSISTANT_DIRECTED,
                   "a modal before a phrase trigger", "a real order"),
    PreferenceCase("would you exit now", CATEGORY_ASSISTANT_DIRECTED,
                   "a modal in front of you", "a real order"),
    PreferenceCase("should you quit now", CATEGORY_ASSISTANT_DIRECTED,
                   "a modal in front of you", "a real order"),
    PreferenceCase("you can quit now", CATEGORY_ASSISTANT_DIRECTED,
                   "you leads, and it still works", "a real order"),
    PreferenceCase("will you exit", CATEGORY_ASSISTANT_DIRECTED,
                   "a modal in front of you", "a real order"),
]


# ----------------------------------------------------------------------
# C. The user asking themselves. Already refused, and must stay refused.
# ----------------------------------------------------------------------
SELF_PERMISSION: list[PreferenceCase] = [
    PreferenceCase("can I quit", CATEGORY_SELF_PERMISSION,
                   "refused by the first-person subject rule", "asking themselves"),
    PreferenceCase("should I exit", CATEGORY_SELF_PERMISSION,
                   "refused by the first-person subject rule", "asking themselves"),
    PreferenceCase("may I quit", CATEGORY_SELF_PERMISSION,
                   "refused by the first-person subject rule", "asking themselves"),
    PreferenceCase("can I ask you to exit", CATEGORY_SELF_PERMISSION,
                   "refused; the first person governs the sentence", "asking themselves"),
    PreferenceCase("should we quit", CATEGORY_SELF_PERMISSION,
                   "refused by the plural subject rule", "asking themselves"),
    PreferenceCase("do you want me to quit", CATEGORY_SELF_PERMISSION,
                   "refused, because me precedes the trigger", "a real question"),
    PreferenceCase("do you want me to exit", CATEGORY_SELF_PERMISSION,
                   "refused, because me precedes the trigger", "a real question"),
    PreferenceCase("can I shut down", CATEGORY_SELF_PERMISSION,
                   "refused by the first-person subject rule", "asking themselves"),
]

# ----------------------------------------------------------------------
# D. Unambiguous orders.
# ----------------------------------------------------------------------
DIRECT_COMMANDS: list[PreferenceCase] = [
    PreferenceCase("exit", CATEGORY_DIRECT, "the shortest form", "an order"),
    PreferenceCase("quit", CATEGORY_DIRECT, "the shortest form", "an order"),
    PreferenceCase("goodbye", CATEGORY_DIRECT, "the shortest form", "an order"),
    PreferenceCase("bye", CATEGORY_DIRECT, "the shortest form", "an order"),
    PreferenceCase("please exit", CATEGORY_DIRECT, "a politeness marker", "an order"),
    PreferenceCase("goodbye assistant", CATEGORY_DIRECT, "an addressee", "an order"),
    PreferenceCase("shut down", CATEGORY_DIRECT, "a phrase trigger", "an order"),
    PreferenceCase("ok pls exit now", CATEGORY_DIRECT, "filler around the trigger",
                   "an order"),
]

# ----------------------------------------------------------------------
# E. Ordinary conversation. Two of these are already wrong, which is
#    recorded rather than hidden: they are the sentences a "to" rule
#    would happen to fix, and the reason it cannot be used alone.
# ----------------------------------------------------------------------
CONVERSATIONAL: list[PreferenceCase] = [
    PreferenceCase("I want to quit smoking", CATEGORY_CONVERSATIONAL,
                   "refused by the first-person subject rule",
                   "a personal decision"),
    PreferenceCase("I should quit smoking", CATEGORY_CONVERSATIONAL,
                   "refused by the first-person subject rule",
                   "a personal decision"),
    PreferenceCase("he quit smoking", CATEGORY_CONVERSATIONAL,
                   "refused by the third-person subject rule", "a report"),
    PreferenceCase("she wants to quit", CATEGORY_CONVERSATIONAL,
                   "no trigger: the verb is wants, not a system word", "a report"),
    PreferenceCase("the device will power off", CATEGORY_CONVERSATIONAL,
                   "no trigger reaches it", "a report about a device"),
    PreferenceCase("there is an exit sign", CATEGORY_CONVERSATIONAL,
                   "refused; the tail window blocks it", "a remark"),
    PreferenceCase("plans to quit", CATEGORY_CONVERSATIONAL,
                   "currently WRONG, and worth knowing", "a report about plans"),
    PreferenceCase("nothing to quit over", CATEGORY_CONVERSATIONAL,
                   "currently WRONG, and worth knowing", "a remark"),
]


# ----------------------------------------------------------------------
# F. Meaning questions, closed by Step 20. Included to show the
#    boundary is not simply "any question about a system word".
# ----------------------------------------------------------------------
SEMANTIC_QUESTIONS: list[PreferenceCase] = [
    PreferenceCase("what does quit mean", CATEGORY_SEMANTIC,
                   "closed by the apposition marker", "asking for a definition"),
    PreferenceCase("what does exit mean", CATEGORY_SEMANTIC,
                   "closed by the apposition marker", "asking for a definition"),
    PreferenceCase("why did he quit", CATEGORY_SEMANTIC,
                   "refused; he precedes the trigger", "asking for a reason"),
    PreferenceCase("did she quit", CATEGORY_SEMANTIC,
                   "refused; she precedes the trigger", "asking about a past event"),
    PreferenceCase("what is the meaning of exit", CATEGORY_SEMANTIC,
                   "closed by the apposition marker", "asking for a definition"),
    PreferenceCase("what does quit mean exactly", CATEGORY_SEMANTIC,
                   "still closed with a tail word added", "asking for a definition"),
]

# ----------------------------------------------------------------------
# G. Subject variants. Every pronoun, to show which one is the gap.
# ----------------------------------------------------------------------
SUBJECT_VARIANTS: list[PreferenceCase] = [
    PreferenceCase("I want to quit", CATEGORY_SUBJECT,
                   "refused: the first person is in the subject set", "myself"),
    PreferenceCase("we want to quit", CATEGORY_SUBJECT,
                   "refused: the first person plural is in the set", "myself"),
    PreferenceCase("he want to quit", CATEGORY_SUBJECT,
                   "refused: he is in the set", "someone else"),
    PreferenceCase("she want to quit", CATEGORY_SUBJECT,
                   "refused: she is in the set", "someone else"),
    PreferenceCase("they want to quit", CATEGORY_SUBJECT,
                   "refused: they is in the set", "someone else"),
    PreferenceCase("me want to quit", CATEGORY_SUBJECT,
                   "refused: me is in the set", "myself"),
    PreferenceCase("him want to quit", CATEGORY_SUBJECT,
                   "refused: him is in the set", "someone else"),
    PreferenceCase("her want to quit", CATEGORY_SUBJECT,
                   "refused: her is in the set", "someone else"),
    PreferenceCase("us want to quit", CATEGORY_SUBJECT,
                   "refused: us is in the set", "myself"),
    PreferenceCase("them want to quit", CATEGORY_SUBJECT,
                   "refused: them is in the set", "someone else"),
    PreferenceCase("does she want to quit", CATEGORY_SUBJECT,
                   "refused: she is in the set", "asking about someone else"),
    PreferenceCase("do they want to quit", CATEGORY_SUBJECT,
                   "refused: they is in the set", "asking about someone else"),
]

#: Every case, in a fixed order so a report always reads the same way.
CORPUS: list[PreferenceCase] = (
    PREFERENCE_QUESTIONS
    + ASSISTANT_DIRECTED
    + SELF_PERMISSION
    + DIRECT_COMMANDS
    + CONVERSATIONAL
    + SEMANTIC_QUESTIONS
    + SUBJECT_VARIANTS
)

#: The dangerous family, named by hand. Measured and checked against
#: the outcome below, so a future change cannot quietly shrink it.
DANGEROUS_FAMILY: tuple[str, ...] = (
    "do you want to quit",
    "do you want to exit",
    "do you want to shut down",
    "do you want to goodbye",
    "do you want to bye",
    "you want to quit",
    "you want to exit",
    "do you really want to quit",
    "plans to quit",
    "nothing to quit over",
)

#: Categories whose members are genuine orders. Everything else in the
#: corpus is something the user is **saying**, not doing.
#
#: This is the expectation column, and it is what makes the
#: classification meaningful. Reaching ``SystemTool`` is only dangerous
#: when the sentence was never an order, so :data:`DANGEROUS_EXECUTION`
#: is the conjunction of the two: the tool ran **and** the sentence was
#: not a command. Without this the label would call "exit" dangerous,
#: which would make the metric useless.
EXPECT_RUN_CATEGORIES: frozenset[str] = frozenset(
    {CATEGORY_ASSISTANT_DIRECTED, CATEGORY_DIRECT}
)

#: Commands that must survive any future fix. Quoted here rather than
#: derived, so a fix cannot quietly shorten the list to make itself pass.
LOAD_BEARING_COMMANDS: tuple[str, ...] = (
    "can you exit",
    "would you quit now",
    "will you quit",
    "can you shut down",
    "would you exit now",
    "should you quit now",
    "you can quit now",
    "will you exit",
    "exit",
    "quit",
    "goodbye",
    "bye",
    "please exit",
    "goodbye assistant",
    "shut down",
    "ok pls exit now",
)

#: Sentences that straddle a "to"-based rule: some currently work and
#: some are currently wrong, so the word alone cannot decide.
AMBIGUOUS_CASES: tuple[str, ...] = (
    "time to quit",
    "ready to quit",
    "try to quit",
    "attempt to exit",
    "about to quit",
    "plans to quit",
    "nothing to quit over",
    "go to exit",
)


def by_category(category: str) -> list[PreferenceCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]

