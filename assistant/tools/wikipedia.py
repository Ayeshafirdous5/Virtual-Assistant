"""Wikipedia lookups.

Selenium is imported inside :meth:`Info.__init__` rather than at module
level, so importing this module (and therefore ``assistant.tools``) does not
load a browser automation stack. The dependency is only paid when a lookup
actually runs.
"""

from typing import Any

from assistant.core.context import AppContext
from assistant.core.tool import Tool


class Info:
    """Drives a Chrome window to Wikipedia and searches for a topic."""

    def __init__(self):
        # Imported here so merely importing this module stays cheap.
        from selenium import webdriver  # noqa: PLC0415

        # Initialize the Chrome WebDriver
        self.driver = webdriver.Chrome()

    def get_info(self, query):
        """Search Wikipedia for ``query`` and leave the page open."""
        from selenium.webdriver.common.by import By  # noqa: PLC0415
        from selenium.webdriver.common.keys import Keys  # noqa: PLC0415

        # Open Wikipedia
        self.driver.get("https://www.wikipedia.org")

        # Find the search input box
        search = self.driver.find_element(By.ID, "searchInput")
        search.send_keys(query)  # Enter the query
        search.send_keys(Keys.RETURN)  # Press Enter to search

        # Wait for the user to close manually
        input("Press Enter to close the browser...")  # Keeps the browser open

        # Quit the browser
        self.driver.quit()


def lookup(topic: str) -> str:
    """Run a Wikipedia search for ``topic``.

    Exposed as a plain function so the application layer can drive the
    second half of a two-turn request without the tool owning any speech.

    Args:
        topic: what to search for.

    Returns:
        A short confirmation line, or a message describing the failure.
    """
    try:
        Info().get_info(topic)
    except Exception as exc:  # noqa: BLE001 - a browser problem must not kill the app
        return f"Sorry, I could not open Wikipedia this time. {exc}"
    return ""


class WikipediaTool(Tool):
    """Look a topic up on Wikipedia.

    The tool never touches ``ctx.speaker`` or ``ctx.listener``. When the user
    has not said a topic, the tool returns :attr:`NEEDS_TOPIC` and lets the
    application run the follow-up question and pass the answer back in. That
    keeps the conversation loop in the app, where it belongs.
    """

    name = "information"
    description = "Look a topic up on Wikipedia."

    #: Returned when the user said "information" without saying what about.
    #: The app recognises this and asks the follow-up question.
    NEEDS_TOPIC = "__needs_topic__"

    #: Phrases that introduce the topic, longest first so "tell me about"
    #: is preferred over "about".
    TOPIC_PREFIXES = (
        "information about",
        "tell me about",
        "search for",
        "search wikipedia for",
        "look up",
        "information on",
        "wikipedia",
        "about",
    )

    def patterns(self) -> list[str]:
        return ["information", "wikipedia"]

    def extract_slots(self, utterance: str, ctx: AppContext) -> dict[str, Any]:
        """Pull the topic out of the utterance.

        "information about Python" yields ``{"topic": "python"}``. A bare
        "information" yields nothing, so the tool can ask for a topic.
        """
        text = utterance.lower()
        for prefix in self.TOPIC_PREFIXES:
            if text.startswith(prefix):
                topic = text[len(prefix):].strip()
                if topic:
                    return {"topic": topic}
                return {}
        return {}

    def execute(self, ctx: AppContext, slots: dict[str, Any] | None = None) -> str:
        topic = (slots or {}).get("topic")
        if not topic:
            # Let the app ask the follow-up question.
            return self.NEEDS_TOPIC
        return lookup(topic)


