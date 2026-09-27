"""Voice implementation using pyttsx3 and SpeechRecognition.

This preserves the original assistant's behaviour:

* the microphone is opened, calibrated against room noise, and listened to
  with the same 5 second timeout and 4000 energy threshold;
* recognition uses the same Google recogniser and lowercases the result;
* text is spoken through pyttsx3 at the configured rate;
* the familiar spoken error messages are used unchanged.

Two deliberate improvements
---------------------------
1. **Lazy imports.** ``pyttsx3`` and ``speech_recognition`` are imported
   inside the constructors, not at module import. Importing this file
   therefore does not touch a sound card or a microphone, which keeps the
   tests and the ``--text`` mode safe.
2. **Voice chosen by name.** The original code used ``voices[1]``, which
   raised ``IndexError`` on a machine with fewer than two voices and picked
   an arbitrary one. This selects by name when possible and otherwise falls
   back to the first available voice, so it always works.

Everything is synchronous. There is no thread and no event loop.
"""

from __future__ import annotations

from assistant.logging_config import get_logger
from assistant.speech.base import NOTHING_HEARD

logger = get_logger(__name__)

# Spoken messages, kept identical to the original assistant so that the
# experience the user already knows does not change.
MSG_DIDNT_UNDERSTAND = "Sorry, I couldn't understand. Could you repeat?"
MSG_SERVICE_ISSUE = "There was an issue with the speech recognition service."
MSG_UNEXPECTED = "An unexpected error occurred. Please try again."


class VoiceSpeaker:
    """A :class:`~assistant.speech.base.Speaker` backed by pyttsx3.

    The TTS engine is created in the constructor, not at import time.
    """

    def __init__(self, rate: int = 130, voice_name: str | None = None) -> None:
        import pyttsx3  # imported here so importing this module stays cheap

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._apply_voice(voice_name)

    def _apply_voice(self, voice_name: str | None) -> None:
        """Select a voice by name, falling back to the first available one.

        Never raises. A machine with no voices installed simply keeps the
        engine default rather than crashing the assistant on startup.
        """
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception as exc:  # noqa: BLE001 - engine may be unavailable
            logger.warning("Could not read the available voices: %s", exc)
            return

        if not voices:
            logger.warning("No TTS voices are installed; using engine default.")
            return

        if voice_name:
            for voice in voices:
                name = getattr(voice, "name", "") or ""
                if voice_name.lower() in name.lower():
                    self._engine.setProperty("voice", getattr(voice, "id", None))
                    logger.info("Using TTS voice: %s", name)
                    return
            logger.warning(
                "TTS voice %r not found; falling back to the first voice.",
                voice_name,
            )

        # Safe default: the first voice the system reports.
        first = voices[0]
        self._engine.setProperty("voice", getattr(first, "id", None))
        logger.info("Using default TTS voice: %s", getattr(first, "name", first))

    def speak(self, text: str) -> None:
        """Say ``text``.

        Failures are logged, never raised, so a speech problem cannot end
        the assistant.
        """
        if not text:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:  # noqa: BLE001 - pyttsx3 can fail on COM
            logger.exception("Text-to-speech failed while speaking: %s", exc)

    def close(self) -> None:
        """Release the speech engine. Safe to call more than once."""
        try:
            self._engine.stop()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass

    def __repr__(self) -> str:
        return "<VoiceSpeaker pyttsx3>"


class VoiceListener:
    """A :class:`~assistant.speech.base.Listener` using SpeechRecognition.

    Args:
        energy_threshold: the original value used by the assistant.
        listen_timeout: seconds of silence before a listen attempt gives up.
        ambient_duration: seconds of room noise used to calibrate.
    """

    def __init__(
        self,
        energy_threshold: int = 4000,
        listen_timeout: int = 5,
        ambient_duration: float = 1,
    ) -> None:
        # Imported here, not at module level, so that importing this file
        # does not load the audio stack.
        import speech_recognition as sr  # noqa: PLC0415

        self._sr = sr
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = energy_threshold
        self.listen_timeout = listen_timeout
        self.ambient_duration = ambient_duration

    def listen(self) -> str:
        """Listen once and return the utterance, lowercased.

        Returns:
            The lowercased transcript, or one of the original spoken error
            messages when the microphone was unavailable or recognition
            failed. Returns ``NOTHING_HEARD`` on plain silence so the caller
            can simply listen again without apologising.
        """
        sr = self._sr

        try:
            with sr.Microphone() as source:
                # Same calibration the original assistant used.
                self._recognizer.adjust_for_ambient_noise(
                    source, duration=self.ambient_duration
                )
                audio = self._recognizer.listen(
                    source, timeout=self.listen_timeout
                )
        except sr.WaitTimeoutError:
            # Silence: no message, just let the caller try again.
            return NOTHING_HEARD
        except (sr.RequestError, OSError) as exc:
            # Microphone missing or audio device unavailable.
            logger.error("Microphone unavailable: %s", exc)
            return MSG_SERVICE_ISSUE
        except Exception as exc:  # noqa: BLE001 - never kill the loop
            logger.exception("Unexpected error while listening: %s", exc)
            return MSG_UNEXPECTED

        try:
            utterance = self._recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return MSG_DIDNT_UNDERSTAND
        except sr.RequestError:
            return MSG_SERVICE_ISSUE
        except Exception as exc:  # noqa: BLE001 - never kill the loop
            logger.exception("Speech recognition failed: %s", exc)
            return MSG_UNEXPECTED

        return (utterance or "").strip().lower()

    def __repr__(self) -> str:
        return "<VoiceListener SpeechRecognition>"


def available_voices() -> list[str]:
    """Return the names of the installed TTS voices.

    Useful for the ``help`` command and for troubleshooting a wrong voice.
    Returns an empty list when the engine cannot be started.
    """
    try:
        import pyttsx3

        engine = pyttsx3.init()
        names = [
            getattr(voice, "name", "")
            for voice in (engine.getProperty("voices") or [])
        ]
        engine.stop()
        return [name for name in names if name]
    except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
        logger.warning("Could not list TTS voices: %s", exc)
        return []
