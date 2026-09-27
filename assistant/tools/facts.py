"""Random fun facts.

Thin wrapper around the ``randfacts`` package. The import is deliberately
inside ``execute()`` so that importing this module is free.
"""

from __future__ import annotations

from typing import Any

from assistant.core.context import AppContext
from assistant.core.tool import Tool


class FactsTool(Tool):
    """Tell the user one random fact."""

    name = "facts"
    description = "A random fun fact."

    def patterns(self) -> list[str]:
        return ["fact", "facts"]

    def execute(self, ctx: AppContext, slots: dict[str, Any] | None = None) -> str:
        import randfacts  # imported lazily so importing this module is cheap

        fact = randfacts.get_fact()
        return f"Did you know? {fact}"
