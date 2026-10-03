"""The layer the application actually depends on.

:class:`AIResponder` is the single narrow seam between the assistant and any
provider. It owns three things and nothing else:

1. whether the feature is available at all;
2. turning one unknown utterance into one reply, or into ``None``;
3. guaranteeing that no provider failure can ever reach the user or the
   command loop.

The guarantee that matters is that :meth:`AIResponder.respond` **never
raises**. Any provider error, any timeout, any unexpected exception at all is
caught and reported as ``None``, which tells the caller to use the assistant's
ordinary "didn't understand" reply. That is what makes the feature safe to
leave switched on: the worst case is the behaviour that existed before it.
"""

from __future__ import annotations

from assistant.ai.base import AIProvider, AIProviderError
from assistant.logging_config import get_logger

logger = get_logger(__name__)


class AIResponder:
    """Answers an unrecognised utterance, or declines to.

    Holds a provider behind the abstraction and applies the availability rules
    in one place, so the application never has to ask whether AI is switched
    on, whether a key exists, or how to handle a failure.
    """

    def __init__(self, provider: AIProvider | None = None) -> None:
        """Wrap a provider, or none at all.

        Args:
            provider: the provider to use, or ``None`` to build a permanently
                disabled responder. Passing ``None`` is the normal case when the
                feature is off, and it keeps the calling code free of ``None``
                checks.
        """
        self._provider = provider

    @property
    def enabled(self) -> bool:
        """True when a provider is attached and the layer can answer."""
        return self._provider is not None

    @property
    def provider_name(self) -> str:
        """The provider's name for logs, or ``"none"`` when disabled."""
        return self._provider.name if self._provider is not None else "none"

    def __repr__(self) -> str:
        return f"<AIResponder enabled={self.enabled} provider={self.provider_name!r}>"

    def respond(self, utterance: str) -> str | None:
        """Return a conversational reply, or ``None`` to use the fallback.

        This is the whole contract with the rest of the assistant: a string
        means "speak this instead", and ``None`` means "say the ordinary
        didn't-understand reply". It never raises.

        Args:
            utterance: the raw, unrecognised user utterance.

        Returns:
            Non-empty conversational text on success; ``None`` when the feature
            is off, the utterance is empty, or the provider failed in any way.
        """
        if self._provider is None:
            return None

        text = (utterance or "").strip()
        if not text:
            return None

        try:
            response = self._provider.generate(text)
        except AIProviderError as exc:
            # Expected: no key, no network, a timeout, a bad body. Logged at
            # warning level so a misconfiguration is visible, but never shown
            # to the user and never fatal. The message is safe by construction:
            # providers must not put credentials in their exception text.
            logger.warning("AI provider unavailable, using the fallback: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - a provider must not kill the loop
            # A provider is third-party code. Anything it raises at all is
            # caught here, so a bug in a provider degrades to the fallback
            # instead of ending the command loop.
            logger.warning(
                "AI provider raised %s unexpectedly, using the fallback.",
                type(exc).__name__,
            )
            return None

        reply = (response.text or "").strip()
        if not reply:
            logger.warning("AI provider returned no usable text; using the fallback.")
            return None

        logger.info(
            "AI answered an unrecognised utterance via %s/%s",
            response.provider,
            response.model,
        )
        return reply