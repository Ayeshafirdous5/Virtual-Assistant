"""Natural language understanding for the Virtual Assistant.

This package turns an utterance into a :class:`ParsedIntent`. It is now the
primary way the application decides what a user asked for, with the legacy
substring router kept as a fallback.

The public surface is deliberately small, and everything a caller needs is
reachable from here, so integration code never imports a submodule directly:

    normalize  -> Normalized    the folded form of an utterance
    parse      -> ParsedIntent  the intent, or None when unsure
                    Alternative a runner-up intent

Independence
------------
No module in this package imports ``assistant.tools``, the router, the
context, the database, the speech stack or an HTTP client. The lexicon is
built from plain ``(intent, patterns)`` data that the application supplies,
so the dependency arrow points one way: ``assistant.app`` knows about the
tools, and :mod:`assistant.nlu` knows about words. Importing this package is
therefore cheap and safe, and it cannot pull in Selenium, pyttsx3 or
``requests``.

Circular imports
----------------
The submodules import each other in one direction only, and none of them
imports this package, so ``import assistant.nlu`` is always safe.
"""

from assistant.nlu.lexicon import (
    COMMON_ALIASES,
    Lexicon,
    LexiconEntry,
    build_lexicon,
)
from assistant.nlu.normalize import (
    CONTRACTIONS,
    NUMBER_WORDS,
    Normalized,
    normalize,
)
from assistant.nlu.parser import (
    AMBIGUOUS,
    CLEAR,
    Alternative,
    ParsedIntent,
    parse,
)

__all__ = [
    "AMBIGUOUS",
    "CLEAR",
    "COMMON_ALIASES",
    "CONTRACTIONS",
    "NUMBER_WORDS",
    "Alternative",
    "Lexicon",
    "LexiconEntry",
    "Normalized",
    "ParsedIntent",
    "build_lexicon",
    "normalize",
    "parse",
]
