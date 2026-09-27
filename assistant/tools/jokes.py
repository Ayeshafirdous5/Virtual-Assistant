"""Jokes for the Virtual Assistant.

This module uses the free Official Joke API, which does not need a key.
The request is made inside the function so that importing this module does
not contact the network.
"""

from typing import Any

from assistant.core.context import AppContext
from assistant.core.tool import Tool

API_URL = "https://official-joke-api.appspot.com/random_joke"

# Never allow a request to hang indefinitely.
REQUEST_TIMEOUT = 10  # seconds


def joke():
    """Return a joke as [setup, punchline] so the caller speaks the two parts
    separately, exactly as before.

    The HTTP request happens here, so importing this module is side-effect
    free and every call returns a freshly fetched joke.
    """
    # Imported here so that importing this module does not load the HTTP
    # client; it is only needed once a real request is about to be made.
    import requests

    try:
        response = requests.get(API_URL, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        return ["Sorry, I could not reach the joke service. Please try again."]

    if response.status_code != 200:
        return [
            "Sorry, the joke service is not available right now. "
            "The service returned error code " + str(response.status_code) + "."
        ]

    try:
        json_data = response.json()
    except ValueError:
        return ["Sorry, I could not read the response from the joke service."]

    setup = json_data.get("setup")
    punchline = json_data.get("punchline")

    if not setup or not punchline:
        return ["Sorry, the joke service returned an incomplete joke."]

    # Kept as a two item list: main.py speaks these as two separate calls.
    return [setup, punchline]


class JokesTool(Tool):
    """Voice command wrapper around the joke() function above."""

    name = "jokes"
    description = "Tell a joke."

    def patterns(self) -> list[str]:
        return ["joke", "jokes"]

    def execute(
        self, ctx: AppContext, slots: dict[str, Any] | None = None
    ) -> list[str]:
        # A list of setup + punchline, spoken as two separate utterances.
        return joke()

