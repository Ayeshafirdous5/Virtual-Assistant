"""Intent vocabulary for the Natural Language Understanding layer.

This module turns plain data of the form ``(intent_name, patterns)`` into a
set of compiled, word-boundary matchers. It is deliberately the *dumbest*
part of the NLU pipeline: it knows what words belong to an intent, and
nothing else. No scoring, no fuzzy matching, no confidence, no entities.

Independence
------------
The lexicon knows nothing about the assistant's tools. It never imports
``assistant.tools``, a ``Tool`` subclass, the router, or any speech or HTTP
library. A caller supplies the vocabulary as data, which keeps the dependency
arrow pointing one way and lets the whole layer be tested with three fake
intents.

Word boundaries
---------------
Every pattern is compiled with ``re.escape`` and anchored on ``\\b``, so
``play`` matches "play music" but never "replay", "playback" or "display",
and ``note`` never matches "noted". That property is the whole reason this
module exists separately from the scoring stage that will use it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

#: A single source line of vocabulary: an intent name and its trigger words.
Entry = tuple[str, Sequence[str]]

#: Optional extra trigger words, merged on top of a caller's own patterns.
AliasMap = Mapping[str, Sequence[str]]

#: A small, deliberately short table of natural synonyms for the assistant's
#: existing commands. A caller merges these in; nothing here is applied
#: automatically, so the tools stay the single source of truth for their own
#: patterns and this file cannot drift into a second copy of them.
#:
#: Why generic nouns are absent
#: -----------------------------
#: An earlier draft listed ``"song"``, ``"video"``, ``"music"``, ``"stop"``
#: and ``"log"`` here. Each is an ordinary English noun, so a bare entry
#: turned any sentence that merely *mentions* the thing into a command:
#:
#:     "replay my song"       -> YouTube
#:     "this video is funny"   -> YouTube
#:     "listen to some music"  -> YouTube
#:     "the train will stop"   -> system
#:     "check the log"         -> history
#:
#: The first three even reached :class:`~assistant.tools.youtube.YouTubeTool`,
#: which opens a browser. The fourth could close the application.
#:
#: The fix is not a blacklist but a rule: **a generic noun may only enter
#: this table as part of a phrase that also names the action.** The
#: commands these aliases existed to catch are still caught, because the
#: tool's own patterns already anchor them:
#:
#:     "play a song"    -> youtube, via the tool pattern "play"
#:     "play music"     -> youtube, via the tool pattern "play"
#:     "play a video"   -> youtube, via the tool pattern "play"
#:     "play lofi beats"-> youtube, via the tool pattern "play"
#:
#: while a sentence that only mentions the noun is left alone. Entries below
#: are therefore either unambiguous on their own, such as "forecast", or
#: already carry their own verb, such as "jot down".
COMMON_ALIASES: dict[str, tuple[str, ...]] = {
    "weather": ("forecast", "how hot", "how cold"),
    "news": ("current affairs", "what happened today"),
    "jokes": ("make me laugh",),
    "information": ("look up", "search for", "tell me about"),
    "youtube": (
        "play a song",
        "play the song",
        "play a video",
        "play the video",
        "on youtube",
        "from youtube",
    ),
    "history": ("what did i say", "command log", "activity log"),
    "notes": ("jot down", "write down"),
    "system": ("stop the assistant", "stop listening", "shut down", "that is all"),
}


@dataclass(frozen=True)
class LexiconEntry:
    """One trigger word belonging to one intent.

    Attributes:
        intent: the name of the intent this word activates.
        pattern: the trigger word, already stripped and lower case.
        regex: compiled word-boundary matcher. Excluded from equality,
            because two separately compiled regexes never compare equal even
            when they are identical.
    """

    intent: str
    pattern: str
    regex: re.Pattern[str] = field(compare=False, repr=False)


@dataclass(frozen=True)
class Lexicon:
    """An immutable, deterministically ordered collection of entries."""

    entries: tuple[LexiconEntry, ...] = ()

    def __iter__(self) -> Iterator[LexiconEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, intent: object) -> bool:
        return any(entry.intent == intent for entry in self.entries)

    @property
    def intents(self) -> tuple[str, ...]:
        """Every distinct intent name, in canonical order."""
        seen: list[str] = []
        for entry in self.entries:
            if entry.intent not in seen:
                seen.append(entry.intent)
        return tuple(seen)

    def entries_for(self, intent: str) -> tuple[LexiconEntry, ...]:
        """Every entry belonging to one intent, in canonical order."""
        return tuple(entry for entry in self.entries if entry.intent == intent)

    def exact_matches(self, text: str) -> tuple[LexiconEntry, ...]:
        """Return every entry whose word-boundary regex matches ``text``.

        A plain presence test. It does not rank results and does not decide
        a winner; the scoring stage does that later.

        Args:
            text: already normalised text. Callers should pass the output of
                :func:`assistant.nlu.normalize.normalize`.

        Returns:
            Matching entries in canonical order. Empty when nothing matches.
        """
        if not text:
            return ()
        return tuple(entry for entry in self.entries if entry.regex.search(text))

    def matching_intents(self, text: str) -> tuple[str, ...]:
        """Distinct intent names with at least one exact match, in order."""
        seen: list[str] = []
        for entry in self.exact_matches(text):
            if entry.intent not in seen:
                seen.append(entry.intent)
        return tuple(seen)


def _compile(pattern: str) -> re.Pattern[str] | None:
    """Compile one trigger word with word boundaries, or ``None`` if unusable.

    The pattern is escaped, so a metacharacter in the data can never turn
    into regex syntax. Empty and whitespace-only patterns are rejected
    rather than silently producing a matcher that matches everywhere.

    Boundaries use ``(?<!\\w)`` and ``(?!\\w)`` rather than ``\\b``. A plain
    ``\\b`` only marks a position next to a *word* character, so a trigger
    ending in punctuation would never match: ``\\bc\\+\\+\\b`` cannot match
    "c++", because there is no word boundary after the final plus. These
    lookarounds express the real requirement, "not touching another word",
    and behave correctly for any pattern.
    """
    cleaned = (pattern or "").strip().lower()
    if not cleaned:
        return None
    return re.compile(rf"(?<!\w){re.escape(cleaned)}(?!\w)")


def build_lexicon(
    entries: Iterable[Entry], aliases: AliasMap | None = None
) -> Lexicon:
    """Build a :class:`Lexicon` from plain data.

    Args:
        entries: ``(intent_name, patterns)`` pairs. Several pairs may share
            an intent; their patterns are merged.
        aliases: optional ``{intent: extra_patterns}`` merged on top.

    Returns:
        A lexicon with one entry per distinct ``(intent, pattern)`` pair,
        sorted by intent then pattern, so the result never depends on the
        order the caller happened to supply.

    Notes:
        Duplicate ``(intent, pattern)`` pairs collapse to one entry. The same
        pattern under two different intents is kept for both, so a later
        scoring stage can arbitrate between them.
    """
    merged: dict[str, list[str]] = {}
    for intent, patterns in entries:
        key = str(intent).strip().lower()
        if not key:
            continue
        merged.setdefault(key, []).extend(patterns or ())

    for intent, extra in (aliases or {}).items():
        key = str(intent).strip().lower()
        if not key:
            continue
        merged.setdefault(key, []).extend(extra or ())

    compiled: list[LexiconEntry] = []
    seen: set[tuple[str, str]] = set()

    for intent, patterns in merged.items():
        for pattern in patterns:
            regex = _compile(pattern)
            if regex is None:
                continue
            cleaned = pattern.strip().lower()
            if (intent, cleaned) in seen:
                continue
            seen.add((intent, cleaned))
            compiled.append(
                LexiconEntry(intent=intent, pattern=cleaned, regex=regex)
            )

    # Canonical order: the lexicon must not depend on input order.
    compiled.sort(key=lambda entry: (entry.intent, entry.pattern))
    return Lexicon(entries=tuple(compiled))
