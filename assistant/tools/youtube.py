"""YouTube search and playback, plus the :class:`YouTubeTool` wrapper.

The original ``Music`` class is preserved unchanged in behaviour. Selenium is
imported inside :meth:`Music.__init__` so that importing this module does not
load a browser automation stack; the cost is only paid when a video is
actually played.
"""

from __future__ import annotations

from typing import Any

from assistant.core.context import AppContext
from assistant.core.tool import Tool


class Music:
    def __init__(self):
        # Imported here so importing this module stays cheap.
        from selenium import webdriver  # noqa: PLC0415

        # Initialize the Chrome WebDriver
        # Make sure you have the correct version of ChromeDriver installed
        self.driver = webdriver.Chrome()

    def play(self, query):
        from selenium.webdriver.common.by import By  # noqa: PLC0415
        from selenium.webdriver.common.keys import Keys  # noqa: PLC0415
        from selenium.webdriver.support import expected_conditions as EC  # noqa: PLC0415
        from selenium.webdriver.support.ui import WebDriverWait  # noqa: PLC0415

        try:
            # Open YouTube
            self.driver.get("https://www.youtube.com")

            # Wait for the search bar to load
            WebDriverWait(self.driver, timeout=10).until(
                EC.presence_of_element_located((By.NAME, "search_query"))
            )

            # Find the search bar and input the query
            search_box = self.driver.find_element(By.NAME, "search_query")
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for search results to load
            WebDriverWait(self.driver, timeout=10).until(
                EC.presence_of_element_located((By.XPATH, '//a[@id="video-title"]'))
            )

            # Click on the first video
            video = self.driver.find_element(By.XPATH, '(//a[@id="video-title"])[1]')
            video.click()

            print("Playing the video on YouTube...")

        except Exception as e:
            print(f"An error occurred: {e}")
            self.driver.quit()

    def quit_driver(self):
        # Close the browser
        self.driver.quit()


class YouTubeTool(Tool):
    """Search YouTube and play the first result.

    Supports a single command, "play <query>", and the two-turn form where
    the user says "play" on its own and is asked what to play. Selenium is
    only imported when a video is actually opened.
    """

    name = "youtube"
    description = "Search YouTube and play a video."

    #: Returned when the user said "play" without saying what to play.
    NEEDS_QUERY = "__needs_youtube_query__"

    #: Phrases that introduce what to play, longest first.
    QUERY_PREFIXES = (
        "play the song",
        "play the video",
        "play song",
        "play video",
        "play music",
        "on youtube",
        "from youtube",
        "youtube",
        "play",
    )

    #: Words that name the service rather than the search term. They are
    #: trimmed from the end of the query, repeatedly, so that
    #: "play lofi beats on youtube" yields "lofi beats" and
    #: "play music on youtube" yields nothing.
    QUERY_NOISE_WORDS = frozenset(
        {"youtube", "youtube.com", "yt", "on", "in", "from", "the"}
    )

    def patterns(self) -> list[str]:
        return ["youtube", "play"]

    @classmethod
    def _strip_service_words(cls, query: str) -> str:
        """Remove trailing service words, keeping the real search terms.

        Works right to left so a phrase like "lofi beats on youtube" loses
        "youtube" first and then the now-trailing "on". Leading and trailing
        filler words are dropped too.

        Args:
            query: the candidate search term.

        Returns:
            The cleaned search term, possibly an empty string.
        """
        words = query.split()
        while words and words[-1] in cls.QUERY_NOISE_WORDS:
            words.pop()
        while words and words[0] in cls.QUERY_NOISE_WORDS:
            words.pop(0)
        return " ".join(words)

    def extract_slots(self, utterance: str, ctx: AppContext) -> dict[str, Any]:
        """Pull the search term out of the utterance.

        "play lofi beats on youtube" yields ``{"query": "lofi beats"}``.
        A bare "play" yields nothing, so the app can ask what to play.
        """
        text = utterance.lower()
        for prefix in self.QUERY_PREFIXES:
            if text.startswith(prefix):
                query = self._strip_service_words(text[len(prefix):])
                if query:
                    return {"query": query}
                return {}
        return {}

    def execute(self, ctx: AppContext, slots: dict[str, Any] | None = None) -> str:
        query = (slots or {}).get("query")
        if not query:
            # Let the app ask the follow-up question.
            return self.NEEDS_QUERY

        music = None
        try:
            music = Music()
            music.play(query)
        except Exception as exc:  # noqa: BLE001 - a browser problem must not kill the app
            return f"Sorry, I could not play that on YouTube. {exc}"
        finally:
            # Never leave an orphaned Chrome process behind.
            if music is not None:
                try:
                    music.quit_driver()
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass

        return f"Playing {query} on YouTube."

