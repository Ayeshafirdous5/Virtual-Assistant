"""The provider abstraction used by the optional AI response layer.

This module defines *what* an AI provider must be able to do, not how any
particular vendor does it. The application depends on
:class:`AIProvider` only, so a future provider is a new subclass and no
existing code changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
# All of these are *expected* failures. The layer that catches them turns each
# one into the assistant's ordinary "didn't understand" reply, so none of them
# ever reaches the user as a crash. Messages must never contain credentials.
class AIProviderError(Exception):
    """Base class for every expected AI failure."""


class AIProviderUnavailable(AIProviderError):
    """The provider cannot be used at all: no key, no network, or no client."""


class AIProviderTimeout(AIProviderError):
    """The provider did not answer inside the configured timeout."""


class AIProviderResponseError(AIProviderError):
    """The provider answered, but not with usable conversational text."""


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AIResponse:
    """Text produced by a provider, plus what produced it.

    Attributes:
        text: the conversational reply, ready to be spoken.
        provider: the provider's stable name, for logging.
        model: the model identifier, for logging.
    """

    text: str
    provider: str
    model: str

    def __repr__(self) -> str:
        return (
            f"AIResponse(provider={self.provider!r}, model={self.model!r}, "
            f"text={self.text!r})"
        )


# ----------------------------------------------------------------------
# The abstraction
# ----------------------------------------------------------------------
class AIProvider(abc.ABC):
    """Turns one user message into conversational text. Nothing else.

    Safety is structural rather than advisory. A provider is handed a plain
    string and must return a plain string. It is never given the
    :class:`~assistant.core.router.Router`, an
    :class:`~assistant.core.context.AppContext`, a
    :class:`~assistant.core.tool.Tool`, the speaker, the listener or the
    database handle, so it has no way to dispatch a command, run a tool,
    execute a shell command, drive a browser, or read stored history.

    Anything a provider returns is spoken text and nothing more. It is never
    parsed, routed or executed, which is why an AI reply can answer a
    conversational question without ever becoming an action.
    """

    #: Short, stable identifier used in logs and in tests.
    name: str = "ai"

    @abc.abstractmethod
    def generate(self, message: str) -> AIResponse:
        """Return a conversational reply to one user message.

        Args:
            message: the raw user utterance, exactly as it was heard. It is
                passed through unchanged so the provider can use the user's
                own words.

        Returns:
            An :class:`AIResponse` carrying non-empty text.

        Raises:
            AIProviderUnavailable: the provider cannot be used.
            AIProviderTimeout: the call exceeded its timeout.
            AIProviderResponseError: the reply was empty or unparseable.
        """

    def __repr__(self) -> str:
        # Subclasses that hold credentials must override this. The default is
        # deliberately bland so a secret is never printed by accident.
        return f"<{type(self).__name__} name={self.name!r}>"