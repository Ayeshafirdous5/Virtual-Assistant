"""Text normalisation for the assistant's natural language layer.

This is the first stage of the pipeline and is deliberately the only part
built so far. It turns whatever speech recognition produced into a
comparable, deterministic form that later stages can score against.

Pipeline
--------
1. Unicode NFKC, so fullwidth and compatibility characters fold down.
2. ``casefold()`` rather than ``lower()``, so cases such as ``ß`` become
   ``ss`` and matching stops being locale dependent.
3. Fancy apostrophes are mapped to the plain one, so a smart-quoted
   contraction is still recognised.
4. Punctuation and symbols become spaces, except apostrophes, which are kept
   long enough for step 5 to see a whole contraction.
5. A small, fixed contraction table is expanded.
6. Number words from one to twenty become digits.
7. ``text`` is rebuilt from ``tokens``, so the two can never disagree.

Standard library only. Nothing here imports from the rest of the assistant.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Characters treated as an apostrophe during tokenisation.
APOSTROPHES = frozenset("\u0027\u2018\u2019\u02bc")

#: The approved contraction set. Kept small on purpose: each entry is a word
#: users actually say to a voice assistant. Expanding anything more would
#: start rewriting meaning rather than fixing punctuation.
CONTRACTIONS: dict[str, str] = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
    "let's": "let us",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "can not",
    "won't": "will not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
}

#: Number words converted to digits, so "delete note two" works.
NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

#: Words made of nothing but whitespace between two separators.
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Normalized:
    """A comparable view of one utterance.

    Attributes:
        text: the normalised string. Always equal to ``" ".join(tokens)``.
        tokens: the words, in order, as a tuple.
        numbers: every integer found in ``tokens``, in order, as a tuple.
    """

    text: str
    tokens: tuple[str, ...] = ()
    numbers: tuple[int, ...] = ()


def _is_separator(char: str) -> bool:
    """True when a character should become a space.

    Punctuation and symbols separate words. Apostrophes do not, because they
    sit inside a word and must survive until contractions are expanded.
    """
    if char in APOSTROPHES:
        return False
    category = unicodedata.category(char)
    return category.startswith("P") or category.startswith("S")


def _fold_apostrophes(text: str) -> str:
    """Map every apostrophe variant onto the plain one."""
    for char in APOSTROPHES:
        if char != "'":
            text = text.replace(char, "'")
    return text


def _expand_contractions(words: list[str]) -> list[str]:
    """Replace whole words found in :data:`CONTRACTIONS`.

    Any apostrophe left over belongs to a contraction that is not in the
    table, such as "ain't". Those are split as well, so the pipeline's
    invariant holds: **no token ever contains an apostrophe**. A token such
    as ``"ain't"`` could never usefully fuzzy-match anything, and a uniform
    token model keeps the later scoring stage simple.
    """
    expanded: list[str] = []
    for word in words:
        replacement = CONTRACTIONS.get(word)
        if replacement is not None:
            expanded.extend(replacement.split())
        else:
            expanded.extend(word.replace("'", " ").split())
    return expanded


def _to_digits(words: list[str]) -> list[str]:
    """Turn number words into digits, whole word only.

    Matching whole words is what keeps ``no one`` from becoming ``no 1``
    when the token is ``one`` on its own; ``someone`` is never touched.
    """
    return [str(NUMBER_WORDS[word]) if word in NUMBER_WORDS else word for word in words]


def normalize(text: str | None) -> Normalized:
    """Normalise one utterance.

    Args:
        text: the raw utterance. ``None`` is treated as empty.

    Returns:
        A :class:`Normalized` with deterministic ``text``, ``tokens`` and
        ``numbers``. Empty input yields empty strings and tuples rather than
        raising, because this runs on every command.
    """
    if not text:
        return Normalized(text="", tokens=(), numbers=())

    # 1-2: fold the text into a comparable, lower case form.
    folded = unicodedata.normalize("NFKC", text).casefold()

    # 3: normalise apostrophe variants before anything else looks at words.
    folded = _fold_apostrophes(folded)

    # 4: punctuation and symbols separate words; apostrophes do not.
    cleaned = "".join(" " if _is_separator(ch) else ch for ch in folded)

    # 4b: collapse repeated whitespace in one pass.
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    if not cleaned:
        return Normalized(text="", tokens=(), numbers=())

    # 5: expand contractions while whole words are still intact.
    words = _expand_contractions(cleaned.split())

    # 6: number words become digits.
    words = _to_digits(words)

    tokens = tuple(words)
    numbers = tuple(int(word) for word in words if word.isdigit())

    # 7: text is rebuilt from tokens so the two can never drift apart.
    return Normalized(text=" ".join(tokens), tokens=tokens, numbers=numbers)
