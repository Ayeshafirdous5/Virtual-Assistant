"""The Step 18 corpus: what the system-safety boundary costs.

Step 17 added a subject check to the system guard so that *"I should quit
smoking"* can no longer terminate the assistant. The check works by
refusing any system trigger that has a first- or third-person subject
before it. That is the right trade for safety, but it has a cost: every
first-person way of *asking* to leave is now refused along with every way
of *not* asking.

This corpus exists to measure that cost honestly rather than to argue for
or against it. It holds three kinds of sentence the guard cannot tell
apart by construction:

* a personal decision    "I should quit smoking"      should not exit
* a personal request     "I need to exit"              should exit
* a request for advice   "can I quit"                  genuinely unclear

Only the first is unambiguous, which is why cases carry an expected
outcome of :data:`UNDECIDED` wherever a reasonable person could answer
either way. Those are never counted as a pass or a fail.

Labelling
---------
``expected`` records what the **user meant**, not what the assistant
supports. A sentence the user plainly intended as an exit request is
``YES`` even when the assistant cannot express it, because that gap is
exactly the cost being measured. Nothing is relabelled to make a number
look better, and the seven dangerous Step 17 sentences are all ``NO``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The user genuinely wants the assistant to stop.
YES = "yes"
#: The user is talking about the word, not asking for it.
NO = "no"
#: A reasonable person could answer either way. Never scored.
UNDECIDED = "undecided"

# Sentence form, which is what the guard actually keys on.
FORM_INSTRUCTION = "instruction"
FORM_QUESTION = "question"
FORM_PERMISSION = "permission"
FORM_STATEMENT = "statement"

# Categories.
CAT_DIRECT = "A-direct-command"
CAT_FIRST_PERSON = "B-first-third-person"
CAT_ASSISTANT_QUESTION = "C-assistant-question"
CAT_PERMISSION = "D-permission-request"
CAT_ORDINARY = "E-ordinary-sentence"
CAT_AMBIGUOUS = "F-ambiguous-conversational"

CATEGORIES: tuple[str, ...] = (
    CAT_DIRECT,
    CAT_FIRST_PERSON,
    CAT_ASSISTANT_QUESTION,
    CAT_PERMISSION,
    CAT_ORDINARY,
    CAT_AMBIGUOUS,
)


@dataclass(frozen=True)
class TradeoffCase:
    """One system-related utterance and what the user meant by it.

    Attributes:
        utterance: what the user says.
        expected: :data:`YES`, :data:`NO`, or :data:`UNDECIDED`.
        category: which group this case belongs to.
        form: the grammatical shape, which is what the guard keys on.
        reason: why the label was chosen.
    """

    utterance: str
    expected: str
    category: str
    form: str
    reason: str = ""


# ----------------------------------------------------------------------
# A. Direct commands. The eight Step 17 was required to preserve.
# ----------------------------------------------------------------------
DIRECT_COMMANDS: list[TradeoffCase] = [
    TradeoffCase("exit", YES, CAT_DIRECT, FORM_INSTRUCTION, "bare command"),
    TradeoffCase("quit", YES, CAT_DIRECT, FORM_INSTRUCTION, "bare command"),
    TradeoffCase("goodbye", YES, CAT_DIRECT, FORM_INSTRUCTION, "bare command"),
    TradeoffCase("bye", YES, CAT_DIRECT, FORM_INSTRUCTION, "bare command"),
    TradeoffCase("please exit", YES, CAT_DIRECT, FORM_INSTRUCTION,
                 "politeness prefix, no subject"),
    TradeoffCase("goodbye assistant", YES, CAT_DIRECT, FORM_INSTRUCTION,
                 "addresses the assistant, no subject before the trigger"),
    TradeoffCase("shut down", YES, CAT_DIRECT, FORM_INSTRUCTION,
                 "two-word alias, no subject"),
    TradeoffCase("you can quit now", YES, CAT_DIRECT, FORM_INSTRUCTION,
                 "second person is the addressee, not a subject"),
]

# ----------------------------------------------------------------------
# B. First- and third-person forms. The user does mean to leave, but the
#    sentence names the speaker, which is exactly what Step 17 refuses.
# ----------------------------------------------------------------------
FIRST_PERSON_FORMS: list[TradeoffCase] = [
    TradeoffCase("I need to exit", YES, CAT_FIRST_PERSON, FORM_STATEMENT,
                 "a real exit request wearing a first-person subject"),
    TradeoffCase("let me quit", YES, CAT_FIRST_PERSON, FORM_INSTRUCTION,
                 "an imperative, but the subject is the speaker"),
    TradeoffCase("can I quit", YES, CAT_FIRST_PERSON, FORM_QUESTION,
                 "a permission question the user means as a request"),
    TradeoffCase("should I exit", YES, CAT_FIRST_PERSON, FORM_QUESTION,
                 "asking permission, meaning to leave"),
    TradeoffCase("I will quit now", YES, CAT_FIRST_PERSON, FORM_STATEMENT,
                 "a declaration of intent to leave"),
    TradeoffCase("us exit now", YES, CAT_FIRST_PERSON, FORM_INSTRUCTION,
                 "first person plural"),
    TradeoffCase("I want to exit", YES, CAT_FIRST_PERSON, FORM_STATEMENT,
                 "plain first-person request"),
    TradeoffCase("I must quit", YES, CAT_FIRST_PERSON, FORM_STATEMENT,
                 "first-person necessity"),
]

# ----------------------------------------------------------------------
# C. Questions aimed at the assistant. The subject is the assistant, so
#    the guard should let them through.
# ----------------------------------------------------------------------
ASSISTANT_QUESTIONS: list[TradeoffCase] = [
    TradeoffCase("can you exit", YES, CAT_ASSISTANT_QUESTION, FORM_QUESTION,
                 "asking the assistant to leave"),
    TradeoffCase("should you quit now", YES, CAT_ASSISTANT_QUESTION,
                 FORM_QUESTION, "asking the assistant to leave"),
    TradeoffCase("would you quit now", YES, CAT_ASSISTANT_QUESTION,
                 FORM_QUESTION, "polite request to the assistant"),
    TradeoffCase("are you ready to exit", YES, CAT_ASSISTANT_QUESTION,
                 FORM_QUESTION, "checking before the assistant leaves"),
    TradeoffCase("can I ask you to exit", YES, CAT_ASSISTANT_QUESTION,
                 FORM_QUESTION, "a request wrapped in a question"),
    TradeoffCase("will you exit", YES, CAT_ASSISTANT_QUESTION, FORM_QUESTION,
                 "asking the assistant to leave"),
]

# ----------------------------------------------------------------------
# D. Permission and request formulations.
# ----------------------------------------------------------------------
PERMISSION_REQUESTS: list[TradeoffCase] = [
    TradeoffCase("please quit now", YES, CAT_PERMISSION, FORM_INSTRUCTION,
                 "a polite command, no subject"),
    TradeoffCase("you may exit", YES, CAT_PERMISSION, FORM_PERMISSION,
                 "granting permission, second person"),
    TradeoffCase("could you exit", YES, CAT_PERMISSION, FORM_QUESTION,
                 "polite request to the assistant"),
    TradeoffCase("I want you to exit", YES, CAT_PERMISSION, FORM_STATEMENT,
                 "names the speaker, but the order is to the assistant"),
    TradeoffCase("I need you to quit", YES, CAT_PERMISSION, FORM_STATEMENT,
                 "names the speaker, but the order is to the assistant"),
    TradeoffCase("may I close the assistant", YES, CAT_PERMISSION,
                 FORM_QUESTION, "no system trigger at all"),
]

# ----------------------------------------------------------------------
# E. Ordinary sentences. The seven Step 17 dangers plus four more. All
#    must stay refused; a single failure here is a FALSE_EXECUTION.
# ----------------------------------------------------------------------
ORDINARY_SENTENCES: list[TradeoffCase] = [
    TradeoffCase("I should quit smoking", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "the Step 15 finding"),
    TradeoffCase("I want to quit smoking", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "a personal decision"),
    TradeoffCase("she decided to quit smoking", NO, CAT_ORDINARY,
                 FORM_STATEMENT, "third person"),
    TradeoffCase("he plans to quit smoking", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "third person"),
    TradeoffCase("I need to quit smoking", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "a personal decision"),
    TradeoffCase("he quit smoking", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "third person"),
    TradeoffCase("I want to exit early", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "leaving a meeting"),
    TradeoffCase("the device will power off", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "no system trigger at all"),
    TradeoffCase("the heater will power off", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "no system trigger at all"),
    TradeoffCase("there is an exit sign", NO, CAT_ORDINARY, FORM_STATEMENT,
                 "a sign, not a command"),
    TradeoffCase("the quit smoking plan failed", NO, CAT_ORDINARY,
                 FORM_STATEMENT, "about a habit"),
]

# ----------------------------------------------------------------------
# F. Genuinely unclear. Asking for advice and asking for an exit read the
#    same way in English, so these are never scored either way.
# ----------------------------------------------------------------------
AMBIGUOUS_CONVERSATIONAL: list[TradeoffCase] = [
    TradeoffCase("should we quit", UNDECIDED, CAT_AMBIGUOUS, FORM_QUESTION,
                 "asking for advice, or ordering an exit"),
    TradeoffCase("can we quit", UNDECIDED, CAT_AMBIGUOUS, FORM_QUESTION,
                 "asking for advice, or ordering an exit"),
    TradeoffCase("do you want to quit", UNDECIDED, CAT_AMBIGUOUS,
                 FORM_QUESTION, "asking the assistant's preference"),
    TradeoffCase("I think we should exit", UNDECIDED, CAT_AMBIGUOUS,
                 FORM_STATEMENT, "an opinion, not clearly an order"),
    TradeoffCase("maybe we should quit", UNDECIDED, CAT_AMBIGUOUS,
                 FORM_STATEMENT, "an opinion, not clearly an order"),
    TradeoffCase("what does quit mean", UNDECIDED, CAT_AMBIGUOUS,
                 FORM_QUESTION, "asking about the word"),
    TradeoffCase("what does exit mean", UNDECIDED, CAT_AMBIGUOUS,
                 FORM_QUESTION, "asking about the word"),
]

CORPUS: list[TradeoffCase] = (
    DIRECT_COMMANDS
    + FIRST_PERSON_FORMS
    + ASSISTANT_QUESTIONS
    + PERMISSION_REQUESTS
    + ORDINARY_SENTENCES
    + AMBIGUOUS_CONVERSATIONAL
)


def by_category(category: str) -> list[TradeoffCase]:
    """Return only the cases in one category."""
    return [case for case in CORPUS if case.category == category]
