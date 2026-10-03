"""Optional AI conversational responses for unrecognised input.

What this is
------------
A thin, optional layer that answers a user utterance **only** when the
existing rule-based pipeline has already decided the utterance is not a
command. It exists to give a helpful reply to something like "what do you
think about the weather today" without a matching command.

What this is deliberately not
-----------------------------
* It is not a replacement for the NLU, the router, or any tool. Those run
  first and AI is never consulted for a recognised command.
* It cannot execute anything. A provider receives a string and returns a
  string; it is never given the router, a tool, the speaker or the database.
* It is not required. With no key, or with the feature switched off, the
  assistant behaves exactly as it did before this layer existed.

Importing this package is cheap and performs no I/O: the provider is
constructed only by :func:`build_responder`, and it imports ``requests``
lazily inside its own ``generate`` call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from assistant.ai.base import (
    AIProvider,
    AIProviderError,
    AIProviderResponseError,
    AIProviderTimeout,
    AIProviderUnavailable,
    AIResponse,
)
from assistant.ai.layer import AIResponder
from assistant.ai.openai_compatible import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    OpenAICompatibleProvider,
)

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
    from assistant.config import Config

__all__ = [
    "AIProvider",
    "AIProviderError",
    "AIProviderResponseError",
    "AIResponder",
    "AIProviderTimeout",
    "AIProviderUnavailable",
    "AIResponse",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "OpenAICompatibleProvider",
    "build_responder",
]


def build_responder(config: "Config") -> AIResponder:
    """Build the responder described by ``config``.

    Returns a disabled :class:`~assistant.ai.layer.AIResponder` when the
    feature is off or no key is configured, so a project with no credentials
    behaves exactly as it did before this layer existed. It never raises.

    Args:
        config: the application settings.

    Returns:
        An :class:`~assistant.ai.layer.AIResponder`, enabled only when both the
        feature flag is on and an API key is present.
    """
    # Both conditions matter. Being switched on without a key is the common
    # half-configured case, and it must degrade quietly rather than fail.
    if not getattr(config, "ai_enabled", False):
        return AIResponder(None)
    # Strip before testing, so a key that is only whitespace counts as absent.
    # ``get_config`` already blanks such values, but a Config built directly
    # (in code or a test) should not be able to produce a "configured" provider
    # that can only ever fail.
    key = (getattr(config, "ai_api_key", None) or "").strip()
    if not key:
        return AIResponder(None)

    provider = OpenAICompatibleProvider(
        key,
        model=config.ai_model,
        base_url=config.ai_base_url,
        timeout=config.ai_timeout,
        max_tokens=config.ai_max_tokens,
    )
    return AIResponder(provider)