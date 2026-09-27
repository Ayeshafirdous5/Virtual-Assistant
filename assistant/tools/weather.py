"""Current weather for the Virtual Assistant.

The OpenWeatherMap API key is read from the OPENWEATHER_API_KEY environment
variable. See .env.example for setup instructions. No key is stored in this
file.
"""

import os
from typing import Any

from dotenv import load_dotenv

from assistant.core.context import AppContext
from assistant.core.tool import Tool

# Load values from a local .env file if one exists.
load_dotenv()

API_URL = "https://api.openweathermap.org/data/2.5/weather"
CITY = "Hyderabad"
KELVIN_OFFSET = 273

# Never allow a request to hang indefinitely.
REQUEST_TIMEOUT = 10  # seconds


def _fetch_weather():
    """Request the current weather and return the decoded JSON, or None.

    Kept private because callers only need the small values below. The
    request happens here, inside the function, so importing this module does
    not contact the network.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return None

    # Imported here so that importing this module does not load the HTTP
    # client; it is only needed once a real request is about to be made.
    import requests

    try:
        response = requests.get(
            API_URL,
            params={"q": CITY, "appid": api_key},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        # The text of a requests exception can contain the request URL, and
        # that URL contains the API key, so nothing from the error is shown.
        return None

    if response.status_code != 200:
        return None

    try:
        return response.json()
    except ValueError:
        return None


def temp():
    """Current temperature in degrees Celsius, or a message if unavailable."""
    json_data = _fetch_weather()

    try:
        return round(json_data["main"]["temp"] - KELVIN_OFFSET, 1)
    except (TypeError, KeyError):
        return "unavailable"


def des():
    """Current weather description, or a message if unavailable."""
    json_data = _fetch_weather()

    try:
        return json_data["weather"][0]["description"]
    except (TypeError, KeyError, IndexError):
        return "unavailable"


class WeatherTool(Tool):
    """Voice command wrapper around the functions above."""

    name = "weather"
    description = "Current weather for a city."

    def patterns(self) -> list[str]:
        return ["weather", "temperature"]

    def execute(self, ctx: AppContext, slots: dict[str, Any] | None = None) -> str:
        city = (slots or {}).get("city") or ctx.config.weather_city
        return f"Temperature in {city} is {temp()} degree Celsius and with {des()}"

