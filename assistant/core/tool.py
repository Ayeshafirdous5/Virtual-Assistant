"""The tool contract every assistant capability implements.

A "tool" is one user-facing capability: weather, news, jokes, and so on.
Each tool declares the phrases that trigger it and knows how to produce the
text answer. The router decides *which* tool runs; the tool only decides
*what it says*.

Design rules
------------
1. ``execute()`` returns text. It never calls the speaker. The caller speaks
   the returned string, which keeps voice mode, text mode and the future
   dashboard working from one implementation.
2. ``execute()`` receives an ``AppContext`` and returns a string, so tools
   are testable in isolation with a fake context.
3. Patterns are plain lowercase substrings. A tool may override
   :meth:`Tool.extract_slots` to pull arguments out of the utterance.
4. ``execute()`` should not raise for expected failures. Return a helpful
   sentence instead, so the assistant stays conversational.
"""

from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

from assistant.core.context import AppContext

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
    from assistant.core.router import Router


class Tool(abc.ABC):
    """Abstract base class for a single assistant capability."""

    #: Stable unique identifier, e.g. ``"weather"``. Used in logs and tests.
    name: str = "tool"

    #: One-line summary shown by the ``help`` command.
    description: str = "No description provided."

    @abc.abstractmethod
    def patterns(self) -> list[str]:
        """Return the lowercase trigger phrases for this tool.

        The router matches these as substrings. Keep them specific; a very
        generic pattern such as ``"get"`` will capture unrelated commands.
        """

    @abc.abstractmethod
    def execute(self, ctx: AppContext, slots: dict[str, Any] | None = None) -> str:
        """Run the capability and return the text to speak or display.

        Args:
            ctx: the shared :class:`~assistant.core.context.AppContext`.
            slots: values extracted from the utterance by
                :meth:`extract_slots`. May be ``None``.

        Returns:
            The response text. Never raises for an expected failure such as
            a network error or a missing API key.
        """

    def extract_slots(self, utterance: str, ctx: AppContext) -> dict[str, Any]:
        """Pull arguments out of the utterance. Default: no arguments.

        Override to capture things like a city name or a note body.

        Args:
            utterance: the raw (lowercased) user request.
            ctx: the shared application context.

        Returns:
            A dict of argument names to values, possibly empty.
        """
        return {}

    def matches(self, utterance: str) -> str | None:
        """Return the matching pattern for this tool, or ``None``.

        Used by the router. When a tool declares several patterns, the
        longest one is returned so that "more specific wins" holds even
        within a single tool.
        """
        text = utterance.lower()
        best: str | None = None
        for pattern in self.patterns():
            candidate = pattern.lower().strip()
            if candidate and candidate in text:
                if best is None or len(candidate) > len(best):
                    best = candidate
        return best

    def handle(self, utterance: str, ctx: AppContext) -> str:
        """Extract slots then execute. The router calls this.

        Wrapping the two steps keeps each tool's ``execute()`` focused on the
        work rather than on argument parsing.
        """
        slots = self.extract_slots(utterance, ctx)
        return self.execute(ctx, slots)

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


def register_all(router: "Router", tools: list[type[Tool]]) -> None:
    """Instantiate each tool class and register it on the router.

    Keeps the tool list declarative and avoids repeating constructor calls.

    Args:
        router: the router to register with.
        tools: tool classes (not instances).
    """
    for tool_class in tools:
        router.register(tool_class())
