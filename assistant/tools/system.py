"""System level commands: exiting the assistant.

Kept separate from the feature tools because it controls the application
lifecycle rather than answering a question. The app checks the tool name
after dispatch to decide when to stop the loop.
"""

from __future__ import annotations

from typing import Any

from assistant.core.context import AppContext
from assistant.core.tool import Tool


class SystemTool(Tool):
    """Handle exit and quit.

    The app sees this tool's name after dispatch and then stops the loop,
    so this class only produces the farewell.
    """

    name = "system"
    description = "Exit or quit the assistant."

    def patterns(self) -> list[str]:
        return ["exit", "quit", "bye", "goodbye"]

    def execute(self, ctx: AppContext, slots: dict[str, Any] | None = None) -> str:
        return "Goodbye! Have a great day!"

    def is_exit(self) -> bool:
        """True when this tool means the assistant should stop.

        Lets the app ask the question rather than hard-coding the name.
        """
        return True
