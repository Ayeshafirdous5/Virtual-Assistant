"""Tool registry and deterministic command dispatch.

The router is deliberately small: a list of registered tools, a scoring
pass over their patterns, and a call to the winning tool. There is no
fuzzy matching here by design, so behaviour stays predictable and testable;
natural-language understanding arrives later in its own layer.

Matching rules
--------------
1. A tool matches when one of its patterns appears as a substring of the
   lowercased utterance.
2. The **longest matching pattern wins**, so a specific phrase beats a
   generic one: "top headlines" beats "news", and a tool declaring
   "play music" is preferred over one declaring "play".
3. Ties are broken by registration order, so the result never depends on
   dict ordering or on hash randomisation.

Known limitation
----------------
Rule 2 picks the longer phrase, not the user's final intent. In
*"information about the news"* the word "information" is longer than
"news", so the utterance goes to the information tool. That is predictable
and consistent, but it is a limitation of substring matching. Resolving
this properly needs intent scoring, which arrives with the natural
language layer rather than being faked here.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.core.context import AppContext
from assistant.core.tool import Tool
from assistant.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Match:
    """The outcome of routing one utterance."""

    #: The tool that will handle the utterance.
    tool: Tool
    #: The pattern that matched (already lowercased).
    pattern: str
    #: The utterance, lowercased.
    utterance: str

    @property
    def tool_name(self) -> str:
        return self.tool.name


@dataclass(frozen=True)
class NoMatch:
    """Returned when no registered tool recognises the utterance."""

    utterance: str

    @property
    def tool_name(self) -> str:
        return ""


class Router:
    """Registry of tools plus the matching logic."""

    def __init__(self) -> None:
        self._tools: list[Tool] = []
        self._by_name: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        """Register a tool instance.

        Args:
            tool: an instance of a :class:`~assistant.core.tool.Tool`.

        Returns:
            The same instance, so calls can be chained.

        Raises:
            TypeError: if ``tool`` is not a :class:`Tool`.
            ValueError: if a tool with the same name is already registered.
        """
        if not isinstance(tool, Tool):
            raise TypeError(
                f"{tool!r} is not a Tool instance; "
                "pass an instance of a Tool subclass."
            )
        if tool.name in self._by_name:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered. "
                "Tool names must be unique."
            )

        self._tools.append(tool)
        self._by_name[tool.name] = tool
        logger.debug(
            "Registered tool %r with patterns %r", tool.name, tool.patterns()
        )
        return tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if something was removed."""
        tool = self._by_name.pop(name, None)
        if tool is None:
            return False
        self._tools.remove(tool)
        logger.debug("Unregistered tool %r", name)
        return True

    def clear(self) -> None:
        """Remove every registered tool. Mainly useful in tests."""
        self._tools.clear()
        self._by_name.clear()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def tools(self) -> list[Tool]:
        """Registered tools, in registration order."""
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        """Return a registered tool by name, or None."""
        return self._by_name.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def help_text(self) -> str:
        """Build the spoken/printed list of available commands."""
        if not self._tools:
            return "No commands are available right now."

        lines = ["These are the commands I understand:"]
        for tool in self._tools:
            examples = ", ".join(tool.patterns()[:3])
            lines.append(f"- {tool.name}: {tool.description}")
            if examples:
                lines.append(f"    try saying: {examples}")
        return "\n".join(lines)

    def find_match(self, utterance: str) -> Match | NoMatch:
        """Return the best :class:`Match`, or :class:`NoMatch`.

        Deterministic: the longest pattern wins, and equal-length patterns
        are decided by registration order.
        """
        text = (utterance or "").lower()

        best_tool: Tool | None = None
        best_pattern = ""

        for tool in self._tools:
            pattern = tool.matches(text)
            if pattern is None:
                continue
            # Strictly greater, so the first registered tool wins a tie.
            if len(pattern) > len(best_pattern):
                best_tool = tool
                best_pattern = pattern

        if best_tool is None:
            logger.debug("No tool matched utterance: %r", text)
            return NoMatch(utterance=text)

        logger.debug(
            "Routed %r to tool %r via pattern %r",
            text,
            best_tool.name,
            best_pattern,
        )
        return Match(tool=best_tool, pattern=best_pattern, utterance=text)

    def dispatch(self, utterance: str, ctx: AppContext) -> str | list[str]:
        """Route the utterance and return the tool's response.

        Never raises: a failing tool is logged and reported as a spoken
        message so the assistant can keep running.

        Returns:
            Either a single string, or a list of strings when a tool produces
            several separate utterances (a joke's setup and punchline, or a
            run of news headlines). Lists are passed through unchanged so the
            caller can speak each entry as its own call.

            Returns an empty string when nothing matched. The wording of the
            "not understood" reply is the application's decision, not the
            router's, so the original assistant wording stays in one place.
        """
        match = self.find_match(utterance)

        if isinstance(match, NoMatch):
            return ""

        tool = match.tool
        try:
            response = tool.handle(match.utterance, ctx)
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the app
            logger.exception("Tool %r raised while handling a request", tool.name)
            response = f"Sorry, something went wrong while running {tool.name}: {exc}"

        if response is None:
            return ""
        if isinstance(response, list):
            return [str(item) for item in response]
        return str(response)
