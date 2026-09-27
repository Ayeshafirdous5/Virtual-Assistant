"""Virtual Assistant tools.

Each module here is one user-facing capability. They are registered on the
router at startup by :func:`assistant.tools.build_default_router`.

Importing this package is cheap: it only pulls in the tool classes, and
every tool imports its own heavy dependency lazily inside ``execute()``.
So ``import assistant.tools`` does not load Selenium, requests, pyttsx3 or
randfacts.
"""

from __future__ import annotations

from assistant.core.context import AppContext
from assistant.core.router import Router
from assistant.core.tool import Tool
from assistant.tools.facts import FactsTool
from assistant.tools.history import HistoryTool
from assistant.tools.jokes import JokesTool
from assistant.tools.news import NewsTool
from assistant.tools.notes import NoteTool
from assistant.tools.system import SystemTool
from assistant.tools.weather import WeatherTool
from assistant.tools.wikipedia import WikipediaTool
from assistant.tools.youtube import YouTubeTool

__all__ = [
    "FactsTool",
    "HistoryTool",
    "JokesTool",
    "NewsTool",
    "NoteTool",
    "SystemTool",
    "WeatherTool",
    "WikipediaTool",
    "YouTubeTool",
    "build_default_router",
]

# Registration order is deliberate and deterministic. More specific tools are
# listed before broader ones; ties in the router are broken by this order.
# SystemTool stays last so exit and quit are matched as late as possible.
_DEFAULT_TOOLS: tuple[type[Tool], ...] = (
    WeatherTool,
    NewsTool,
    FactsTool,
    JokesTool,
    WikipediaTool,
    YouTubeTool,
    HistoryTool,
    NoteTool,
    SystemTool,
)


def build_default_router() -> Router:
    """Build a router with every built-in tool registered.

    Returns:
        A fresh :class:`~assistant.core.router.Router`. Nothing is shared
        between calls, so tests can build an isolated router.
    """
    router = Router()
    for tool_class in _DEFAULT_TOOLS:
        router.register(tool_class())
    return router
