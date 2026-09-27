"""Console implementation of the Speaker and Listener protocols.

Used for two things:

* the future ``--text`` mode, so the assistant can be driven with no
  microphone and no speakers, which also makes it easy to demo;
* automated tests, by handing a list of strings to :class:`TextListener`
  instead of real keyboard input.

This module imports nothing outside the standard library.
"""

from __future__ import annotations

import builtins
import sys
from typing import Callable, Sequence

from assistant.logging_config import get_logger
from assistant.speech.base import NOTHING_HEARD

logger = get_logger(__name__)


class TextSpeaker:
    """A :class:`~assistant.speech.base.Speaker` that prints to the console."""

    def __init__(self, stream=None) -> None:
        self._stream = stream  # None means sys.stdout at call time

    def speak(self, text: str) -> None:
        """Print ``text`` to the console."""
        if not text:
            return
        stream = self._stream if self._stream is not None else sys.stdout
        print(text, file=stream)
        try:
            stream.flush()
        except (AttributeError, ValueError):
            # A closed or replaced stream must not break the assistant.
            pass

    def __repr__(self) -> str:
        return "<TextSpeaker>"


class TextListener:
    """A :class:`~assistant.speech.base.Listener` backed by ``input()``.

    Args:
        responses: optional canned replies. When provided, each call to
            :meth:`listen` pops the next item, so a test can drive the
            assistant without a keyboard. The final item repeats once the
            list is exhausted, which keeps a scripted session flowing.
        prompt: text shown before each read. ``None`` prints nothing.
        reader: the read function, injectable for testing. Defaults to the
            built-in ``input``.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        prompt: str | None = "You: ",
        reader: Callable[[str], str] = builtins.input,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._prompt = prompt
        self._reader = reader
        self._index = 0

    def listen(self) -> str:
        """Return the next line typed by the user, lowercased.

        Returns:
            The typed text lowercased, or ``NOTHING_HEARD`` on EOF, on an
            interrupt, or when the user types nothing.
        """
        if self._responses:
            text = self._responses[min(self._index, len(self._responses) - 1)]
            self._index += 1
            return text.strip().lower()

        try:
            raw = self._reader(self._prompt) if self._prompt else self._reader()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C or Ctrl+Z means the user wants to finish.
            return NOTHING_HEARD

        if raw is None:
            return NOTHING_HEARD
        return raw.strip().lower()

    def __repr__(self) -> str:
        return f"<TextListener scripted={bool(self._responses)}>"
