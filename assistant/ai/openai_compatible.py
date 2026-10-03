"""The one optional provider: any OpenAI-compatible chat-completions endpoint.

Scope and choices
-----------------
* **No new dependency.** ``requests`` is already required by the news and
  weather tools, so this module reuses it and imports it lazily inside
  :meth:`OpenAICompatibleProvider.generate`. Importing this module therefore
  performs no I/O and loads no HTTP client.
* **Conversation text only.** The request carries a system prompt and the
  user's message and nothing else. No tools, no functions, no tool-choice
  schema and no structured-output format is ever sent, so the endpoint has
  nothing it could call back with.
* **Credentials stay here.** The key is held privately, is never logged, and
  is redacted from ``repr()``.
"""

from __future__ import annotations

from assistant.ai.base import (
    AIProvider,
    AIProviderResponseError,
    AIProviderTimeout,
    AIProviderUnavailable,
    AIResponse,
)

#: Default endpoint. Any OpenAI-compatible base URL can be substituted, which
#: is why this is configuration rather than a hard-coded constant below.
DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: Default model. Deliberately a small, cheap, fast one: this layer answers a
#: short conversational reply, it does not run a reasoning task.
DEFAULT_MODEL = "gpt-4o-mini"

#: Refuse an unexpectedly large reply. A chat reply is a couple of sentences;
#: anything bigger means something unexpected came back.
MAX_REPLY_CHARS = 600

#: Told to the model. It describes the assistant's capabilities so the reply is
#: accurate, and it states the boundary the code also enforces: this layer can
#: produce words, never actions.
DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant built in Python. The user just said something "
    "that is not one of your commands.\n"
    "Reply with one short, friendly sentence (at most two) that answers "
    "conversationally.\n"
    "Rules:\n"
    "- Never claim to have performed an action. You cannot run commands, "
    "control a computer, browse the web, look anything up, or control smart "
    "home devices.\n"
    "- If asked to do one of those, say plainly that you cannot.\n"
    "- If you do not know, say so briefly.\n"
    "- Plain text only. No markdown, no lists, no code blocks, no emoji.\n"
    "- The user will hear this spoken aloud, so avoid symbols that do not "
    "read well."
)


class OpenAICompatibleProvider(AIProvider):
    """Calls an OpenAI-compatible ``/chat/completions`` endpoint.

    Constructed only when the feature is switched on and a key is present, so
    an app configured with no credentials builds no provider at all.
    """

    name = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 8,
        max_tokens: int = 200,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Store the settings needed for one call.

        Args:
            api_key: the credential. Never logged or printed.
            model: the model identifier to request.
            base_url: the API root; ``/chat/completions`` is appended.
            timeout: seconds to wait before giving up.
            max_tokens: upper bound on the reply length.
            system_prompt: instructions sent with every request.
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = max(1, int(timeout))
        self._max_tokens = max(1, int(max_tokens))
        self._system_prompt = system_prompt

    def __repr__(self) -> str:
        # Written by hand so the key can never reach a log line or a
        # traceback. Only presence is reported, never the value.
        key_state = "set" if self._api_key else "unset"
        return (
            f"<OpenAICompatibleProvider model={self._model!r} "
            f"base_url={self._base_url!r} api_key={key_state}>"
        )

    @property
    def endpoint(self) -> str:
        """The full URL this provider posts to."""
        return f"{self._base_url}/chat/completions"

    def _payload(self, message: str) -> dict[str, object]:
        """Build the request body.

        Deliberately contains no ``tools``, ``functions`` or ``response_format``
        key, so the endpoint is never offered anything to call.
        """
        return {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": message},
            ],
        }

    def generate(self, message: str) -> AIResponse:
        """Return one conversational reply, or raise an ``AIProviderError``.

        Raises:
            AIProviderUnavailable: no key, ``requests`` is missing, the request
                failed, or the endpoint answered with an error status.
            AIProviderTimeout: the request exceeded the configured timeout.
            AIProviderResponseError: the body was not JSON, or carried no
                usable text.
        """
        if not message or not message.strip():
            raise AIProviderResponseError("Refusing to send an empty message.")
        if not self._api_key:
            raise AIProviderUnavailable("No AI API key is configured.")

        # Imported here, inside the call, exactly as the news and weather tools
        # do it. Importing this module therefore loads no HTTP client.
        try:
            import requests
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise AIProviderUnavailable(
                "The 'requests' package is not installed."
            ) from exc

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=self._payload(message),
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise AIProviderTimeout(
                f"The AI provider did not respond within {self._timeout}s."
            ) from exc
        except requests.exceptions.RequestException as exc:
            # The exception text can echo the request, so only the type name is
            # reported. That keeps the key and the Authorization header out of
            # every log and traceback.
            raise AIProviderUnavailable(
                f"Could not reach the AI provider: {type(exc).__name__}"
            ) from None
        except Exception as exc:  # noqa: BLE001 - never let the app crash
            raise AIProviderUnavailable(
                f"Unexpected AI provider failure: {type(exc).__name__}"
            ) from None

        if response.status_code >= 400:
            # The status code is safe to log; the body may contain the key.
            raise AIProviderUnavailable(
                f"The AI provider returned HTTP {response.status_code}."
            )

        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - not every failure is ValueError
            raise AIProviderResponseError(
                "The AI provider returned a response that was not JSON."
            ) from exc

        text = self._extract_text(payload)
        if not text:
            raise AIProviderResponseError("The AI provider returned no text.")

        return AIResponse(
            text=text[:MAX_REPLY_CHARS], provider=self.name, model=self._model
        )

    def _extract_text(self, payload: object) -> str:
        """Pull the assistant's text out of a chat-completions body.

        Returns an empty string for any shape that is not the expected one, so
        a protocol change becomes a clean fallback rather than a crash.
        """
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if not isinstance(content, str):
            return ""
        return content.strip()