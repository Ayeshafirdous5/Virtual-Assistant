"""Top news headlines for the Virtual Assistant.

The NewsAPI key is read from the NEWS_API_KEY environment variable.
See .env.example for setup instructions. No key is stored in this file.
"""

import os

from typing import Any

from dotenv import load_dotenv

from assistant.core.context import AppContext
from assistant.core.tool import Tool

# Load values from a local .env file if one exists.
load_dotenv()

API_URL = "https://newsapi.org/v2/top-headlines"
COUNTRY = "us"
HEADLINE_COUNT = 3

# Never allow a request to hang indefinitely.
REQUEST_TIMEOUT = 10  # seconds


def news():
    """Return the top headlines as a list of speakable strings.

    The HTTP request is made here, inside the function, rather than at import
    time. Importing this module therefore performs no network activity, and
    every call returns freshly fetched data.
    """
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        # A list is still returned so the caller's loop keeps working.
        return ["Sorry, I could not fetch the news. The News API key is missing."]

    # Imported here so that importing this module does not load the HTTP
    # client; it is only needed once a real request is about to be made.
    import requests

    try:
        response = requests.get(
            API_URL,
            params={"country": COUNTRY, "apiKey": api_key},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        # The text of a requests exception can contain the request URL, and
        # that URL contains the API key, so only the exception type is shown.
        return ["Sorry, I could not reach the news service. Please try again."]

    if response.status_code != 200:
        return [
            "Sorry, the news service is not available right now. "
            "The service returned error code " + str(response.status_code) + "."
        ]

    try:
        articles = response.json().get("articles") or []
    except ValueError:
        return ["Sorry, I could not read the response from the news service."]

    headlines = []
    for article in articles[:HEADLINE_COUNT]:
        title = article.get("title")
        if title:
            headlines.append("Number " + str(len(headlines) + 1) + "," + title + ".")

    if not headlines:
        return ["Sorry, there are no news headlines available right now."]

    return headlines


class NewsTool(Tool):
    """Voice command wrapper around the news() function above."""

    name = "news"
    description = "The latest top headlines."

    def patterns(self) -> list[str]:
        return ["news", "headlines", "top headlines"]

    def execute(
        self, ctx: AppContext, slots: dict[str, Any] | None = None
    ) -> list[str]:
        # Returned as a list so the app speaks each headline separately,
        # exactly as the original assistant did.
        return news()

