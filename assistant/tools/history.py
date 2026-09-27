"""Show the commands the assistant has handled this session and before.

Reads from the ``interactions`` table through
:mod:`assistant.storage.repositories`. The tool never touches
``ctx.speaker`` or ``ctx.listener``: it returns the lines and the app speaks
them.

Every failure is reported as a sentence rather than raised, so a missing or
broken database can never stop the assistant.
"""

from __future__ import annotations

from typing import Any

from assistant.core.context import AppContext
from assistant.core.tool import Tool
from assistant.storage import count_by_tool, recent_interactions, total_interactions

#: How many recent commands to read back by default.
DEFAULT_LIMIT = 5

#: Commands per line, so one very long utterance cannot flood the log.
MAX_UTTERANCE = 40


class HistoryTool(Tool):
    """Report recent commands and a small usage summary."""

    name = "history"
    description = "Show the commands used so far."

    def patterns(self) -> list[str]:
        return ["history", "command history", "recent commands", "what did i ask"]

    def execute(
        self, ctx: AppContext, slots: dict[str, Any] | None = None
    ) -> list[str]:
        limit = int((slots or {}).get("limit") or DEFAULT_LIMIT)

        if ctx.db is None:
            return ["Sorry, I have no history to show right now."]

        try:
            recent = recent_interactions(ctx.db, limit=limit)
            total = total_interactions(ctx.db)
            counts = count_by_tool(ctx.db)
        except Exception as exc:  # noqa: BLE001 - history is never critical
            return [f"Sorry, I could not read my history right now. {exc}"]

        if not recent:
            return ["You have not asked me anything yet."]

        lines = ["Here is what you have asked me recently:"]
        for number, item in enumerate(recent, start=1):
            spoken = " ".join(item.utterance.split())
            if len(spoken) > MAX_UTTERANCE:
                spoken = spoken[: MAX_UTTERANCE - 1] + "..."
            lines.append(f"{number}. {item.tool_name}: {spoken}")

        summary = ", ".join(f"{c.tool_name} {c.count}" for c in counts)
        lines.append(f"That is {total} commands in total: {summary}.")
        return lines
