"""Central configuration for the Virtual Assistant.

This is the ONLY module in the project that reads environment variables.
Everything else imports :class:`Config` (or calls :func:`get_config`) so that
there is exactly one place to look when asking "where does this value come
from?".

Security notes
--------------
* API keys are never printed, logged, or included in ``repr()`` output.
  They are declared with ``repr=False`` so that accidentally logging a
  :class:`Config` object cannot leak them.
* :meth:`Config.describe` is the safe way to display settings: it reports
  only whether a key is *present*, never its value.
* Values come from the process environment first, then from a local ``.env``
  file. The ``.env`` file is git-ignored and must never be committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root: the folder that contains main.py and the .env file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the local secrets file.
ENV_FILE = PROJECT_ROOT / ".env"

# Log file lives beside the source, in an ignored "logs" folder.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "assistant.log"

# SQLite database lives in an ignored "data" folder next to the source.
DATA_DIR = PROJECT_ROOT / "data"
DB_FILE = DATA_DIR / "assistant.db"


@dataclass(frozen=True)
class Config:
    """Immutable application settings.

    Frozen so that nothing can mutate configuration at runtime, which keeps
    bugs easier to trace. The two API keys are excluded from ``repr`` so they
    can never be exposed by printing or logging this object.
    """

    # --- Credentials (never printed) ---------------------------------------
    news_api_key: str | None = field(default=None, repr=False)
    openweather_api_key: str | None = field(default=None, repr=False)

    # --- News feature -------------------------------------------------------
    news_country: str = "us"
    news_headline_count: int = 3

    # --- Weather feature ----------------------------------------------------
    weather_city: str = "Hyderabad"

    # --- Speech settings ----------------------------------------------------
    tts_rate: int = 130
    # Preferred voice name. When None, the assistant picks the first voice the
    # system reports, which is safer than assuming a fixed voice index.
    tts_voice_name: str | None = None

    # --- Networking ---------------------------------------------------------
    # Applied to every outgoing HTTP request so nothing can hang forever.
    request_timeout: int = 10

    # --- Logging ------------------------------------------------------------
    log_level: str = "INFO"
    log_file_max_bytes: int = 1_000_000  # rotate at ~1 MB
    log_file_backup_count: int = 3  # keep assistant.log.1 .. .3

    # --- Paths --------------------------------------------------------------
    project_root: Path = field(default=PROJECT_ROOT)
    log_dir: Path = field(default=LOG_DIR)
    log_file: Path = field(default=LOG_FILE)
    data_dir: Path = field(default=DATA_DIR)
    db_path: Path = field(default=DB_FILE)

    def has_news_api_key(self) -> bool:
        """True when a NewsAPI key is configured."""
        return bool(self.news_api_key)

    def has_openweather_api_key(self) -> bool:
        """True when an OpenWeatherMap key is configured."""
        return bool(self.openweather_api_key)

    def describe(self) -> str:
        """Return a human-readable summary that is safe to print or log.

        Reports only whether each key is present, never the key itself.
        """
        news_state = "configured" if self.has_news_api_key() else "MISSING"
        weather_state = (
            "configured" if self.has_openweather_api_key() else "MISSING"
        )
        return (
            "Config: "
            f"news_api_key={news_state}, "
            f"openweather_api_key={weather_state}, "
            f"news_country={self.news_country!r}, "
            f"news_headline_count={self.news_headline_count}, "
            f"weather_city={self.weather_city!r}, "
            f"tts_rate={self.tts_rate}, "
            f"tts_voice_name={self.tts_voice_name!r}, "
            f"request_timeout={self.request_timeout}s, "
            f"log_level={self.log_level!r}, "
            f"log_file={self.log_file.name!r}, "
            f"db_file={self.db_path.name!r}"
        )


def _read_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back on bad input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_str(name: str, default: str | None) -> str | None:
    """Read a string from the environment, treating blanks as unset."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def get_config() -> Config:
    """Build a :class:`Config` from the environment and the ``.env`` file.

    ``load_dotenv`` does not overwrite variables that are already set in the
    real environment, so a value exported in the shell always wins over the
    one stored in ``.env``.
    """
    load_dotenv(dotenv_path=ENV_FILE)

    return Config(
        news_api_key=_read_str("NEWS_API_KEY", None),
        openweather_api_key=_read_str("OPENWEATHER_API_KEY", None),
        news_country=_read_str("NEWS_COUNTRY", "us"),
        news_headline_count=_read_int("NEWS_HEADLINE_COUNT", 3),
        weather_city=_read_str("WEATHER_CITY", "Hyderabad"),
        tts_rate=_read_int("TTS_RATE", 130),
        tts_voice_name=_read_str("TTS_VOICE_NAME", None),
        request_timeout=_read_int("REQUEST_TIMEOUT", 10),
        log_level=(_read_str("LOG_LEVEL", "INFO") or "INFO").upper(),
        project_root=PROJECT_ROOT,
        log_dir=LOG_DIR,
        log_file=LOG_FILE,
        data_dir=DATA_DIR,
        db_path=DB_FILE,
    )


# Cached so that repeated calls do not re-read the environment each time.
_cached_config: Config | None = None


def get_config_cached() -> Config:
    """Return the process-wide :class:`Config`, building it on first use."""
    global _cached_config
    if _cached_config is None:
        _cached_config = get_config()
    return _cached_config
