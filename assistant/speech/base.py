"""Protocols for speech input and output.

Kept deliberately tiny and free of heavy imports: this module pulls in
nothing but the standard library, so importing it is cheap and safe. The
pyttsx3 and SpeechRecognition imports live in ``voice.py`` and are only
loaded when that module is actually used.

Both protocols are synchronous by design. The assistant is a simple
request/response loop around a blocking microphone and a blocking speech
engine, so there is no event loop and no background thread.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Sentinel returned by Listener.listen() when nothing was heard. It is not a
# valid utterance, so callers can test for it with a plain identity check.
NOTHING_HEARD = ""


@runtime_checkable
class Speaker(Protocol):
    """Anything that can say a piece of text to the user."""

    def speak(self, text: str) -> None:
        """Speak ``text``.

        Implementations must not raise for an ordinary failure such as a
        missing speech engine; they should log and carry on, because losing
        the ability to speak should never end the assistant.

        Args:
            text: the text to say. May be multi-line.
        """
        ...


@runtime_checkable
class Listener(Protocol):
    """Anything that can obtain the next thing the user said."""

    def listen(self) -> str:
        """Return the next utterance, lowercased, or ``NOTHING_HEARD``.

        Returns:
            The user's speech as text, or ``NOTHING_HEARD`` when nothing was
            captured, the microphone was unavailable, or recognition failed.
            This method must not raise: a listening failure is reported by
            returning ``NOTHING_HEARD``.
        """
        ...
