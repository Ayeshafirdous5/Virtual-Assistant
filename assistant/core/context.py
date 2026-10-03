"""Shared application context passed to every tool.

The context is the single object a tool receives. It carries the things a
tool legitimately needs and nothing else. Passing it explicitly (rather than
having tools import globals) is what makes a tool testable: a test builds a
context with a temporary config and a throwaway logger, then calls
``execute()`` with no microphone, browser or network involved.

Deliberately plain
------------------
This is a dataclass, not a container framework and not a service locator.
Adding a new capability means adding a field here, which is a visible,
reviewable change.

Planned additions
-----------------
Later phases may attach more collaborators, each optional so the assistant
still runs without them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from assistant.config import Config, get_config_cached
from assistant.core.database import Database
from assistant.logging_config import get_logger
from assistant.speech.base import Listener, Speaker

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
    from assistant.ai import AIResponder


@dataclass
class AppContext:
    """Everything a tool is allowed to depend on.

    Attributes:
        config: immutable application settings.
        logger: a namespaced logger. Defaults to ``assistant.context``.
        speaker: a :class:`~assistant.speech.base.Speaker`, or ``None`` when
            the assistant runs without spoken output.
        listener: a :class:`~assistant.speech.base.Listener`, or ``None``
            when the assistant runs without speech input.
        db: the SQLite handle, or ``None`` when the assistant runs without
            persistence. Optional so the app still works with no database.
    """

    config: Config = field(default_factory=get_config_cached)
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))

    # --- Speech I/O, attached by the app at startup ------------------------
    # Typed as the protocols so any implementation can be plugged in.
    speaker: Speaker | None = None
    listener: Listener | None = None

    # --- Persistence ---------------------------------------------------------
    # Optional so the assistant still runs with no database at all.
    db: Database | None = None

    # --- Optional AI conversational layer -----------------------------------
    # Optional in exactly the same way, and for the same reason: the assistant
    # must run unchanged when the feature is off. ``None`` or a disabled
    # responder both mean "no AI", so the caller never needs to know which.
    ai: "AIResponder | None" = None

    def has_db(self) -> bool:
        """True when a database handle is attached."""
        return self.db is not None

    def has_speaker(self) -> bool:
        """True when speech output is attached."""
        return self.speaker is not None

    def log_command(
        self,
        tool_name: str,
        utterance: str,
        response: str,
        nlu: dict[str, object] | None = None,
    ) -> None:
        """Record one handled command. The app calls this after dispatch.

        Does two things, in this order:

        1. writes the usual log line, exactly as before;
        2. stores the same command in the ``interactions`` table when a
           database is attached.

        Both are best effort. If the database is missing, closed, locked or
        broken, the failure is logged and swallowed, because losing history
        must never stop the assistant from answering.

        Args:
            tool_name: the tool that handled the command.
            utterance: what the user said.
            response: what the assistant said, already truncated.
            nlu: optional understanding details, such as the chosen intent,
                its score, confidence, method, trigger and alternatives. This
                is **diagnostic only**: it is appended to the log line and
                never written to the database, so the stored schema and
                every existing log consumer are unaffected. Omitting it
                produces exactly the log line that was produced before.
        """
        preview = " ".join(response.split())[:120]
        if nlu:
            self.logger.info(
                "command=%s utterance=%r response=%r nlu=%s",
                tool_name,
                utterance,
                preview,
                _format_nlu(nlu),
            )
        else:
            self.logger.info(
                "command=%s utterance=%r response=%r",
                tool_name,
                utterance,
                preview,
            )
        self._record_interaction(tool_name, utterance, preview)

    def _record_interaction(
        self, tool_name: str, utterance: str, response: str
    ) -> None:
        """Insert one command into the ``interactions`` table.

        Kept separate from :meth:`log_command` so the logging path stays
        readable and so the database failure handling is in one place.
        """
        if self.db is None:
            return
        try:
            self.db.execute(
                "INSERT INTO interactions (tool_name, utterance, response) "
                "VALUES (?, ?, ?)",
                (tool_name, utterance, response),
            )
        except Exception as exc:  # noqa: BLE001 - history is never critical
            self.logger.warning("Could not record the command to the database: %s", exc)

    def __repr__(self) -> str:
        # Written by hand so a Config (which may hold API keys) is summarised
        # safely instead of being dumped, and so optional handles stay short.
        attached = [
            name
            for name in ("db", "speaker", "listener", "ai")
            if getattr(self, name) is not None
        ]
        return (
            f"AppContext(config={self.config.describe()!r}, "
            f"attached={attached or 'none'})"
        )


def _format_nlu(nlu: dict[str, object]) -> str:
    """Render understanding details compactly for a single log line.

    Kept short and predictable so a log stays greppable. Keys are emitted
    in a fixed order rather than dictionary order, so the same command
    always produces exactly the same line.
    """
    parts = []
    for key in ("intent", "score", "confidence", "method", "trigger"):
        if key in nlu:
            parts.append(f"{key}={nlu[key]}")
    alternatives = nlu.get("alternatives") or ()
    if alternatives:
        rendered = ",".join(str(item) for item in alternatives)
        parts.append(f"alternatives=[{rendered}]")
    return " ".join(parts)

