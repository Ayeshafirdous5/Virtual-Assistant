"""Application orchestration: greeting, the command loop, and startup.

``main.py`` is now only a launcher. Everything that happens at runtime lives
here.

This module contains no feature logic and defines no tools. Each capability
lives in its own module under :mod:`assistant.tools`, and
:func:`assistant.tools.build_default_router` registers them in a fixed order.

The app owns the conversation. When a tool needs something the user has not
said yet, it returns a sentinel such as
:attr:`~assistant.tools.wikipedia.WikipediaTool.NEEDS_TOPIC`; the app then
asks the follow-up question, listens once, and calls the tool again. Tools
therefore never touch ``ctx.speaker`` or ``ctx.listener``.

Everything is synchronous. One context is built, one speaker and listener
are attached, and the loop runs until the user says exit.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass
from typing import Any, Sequence

from assistant.config import Config, get_config
from assistant.core.context import AppContext
from assistant.core.database import Database
from assistant.core.router import Match, NoMatch, Router
from assistant.core.tool import Tool
from assistant.logging_config import get_logger, setup_logging
from assistant.nlu import (
    AMBIGUOUS,
    COMMON_ALIASES,
    Lexicon,
    ParsedIntent,
    build_lexicon,
    parse,
)
from assistant.nlu import framing as framing_mod
from assistant.nlu.normalize import normalize
from assistant.speech import voice as voice_mod
from assistant.speech.base import NOTHING_HEARD
from assistant.speech.text import TextListener, TextSpeaker
from assistant.tools import build_default_router
from assistant.tools.system import SystemTool
from assistant.tools.wikipedia import WikipediaTool
from assistant.tools.youtube import YouTubeTool

logger = get_logger(__name__)

# Strings the voice listener returns to mean "this is a message to speak",
# not a command. Compared here so a message is never routed as a request.
SPEECH_ERRORS = frozenset(
    {
        voice_mod.MSG_DIDNT_UNDERSTAND,
        voice_mod.MSG_SERVICE_ISSUE,
        voice_mod.MSG_UNEXPECTED,
    }
)

# The original unknown-command wording, restored exactly as it was before the
# router was introduced.
MSG_DIDNT_UNDERSTAND = "I'm sorry, I didn't understand. Can you please repeat?"

# Follow-up questions the app asks when a tool signals it needs more input.
# The keys are the sentinel values the tools return. Keeping the mapping here
# means the tools themselves never touch the speaker or the listener.
PROMPTS = {
    WikipediaTool.NEEDS_TOPIC: "You need information on what topic?",
    YouTubeTool.NEEDS_QUERY: "What would you like me to play?",
}

# ----------------------------------------------------------------------
# Greeting
# ----------------------------------------------------------------------
def wishme() -> str:
    """Return the time-of-day greeting, unchanged from the original."""
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 16:
        return "Afternoon"
    else:
        return "Evening"


def speak_all(ctx: AppContext, parts: Sequence[Any]) -> None:
    """Speak one or more parts.

    A tool may return a list (for example a joke's setup and punchline), in
    which case each entry is spoken as its own utterance. This is what keeps
    the joke delivered as two separate calls, exactly as before.

    Output goes through the attached speaker, which is the single output
    path. The console echo that the original assistant produced is kept for
    voice mode only: in text mode the speaker already prints, so echoing
    again would show every line twice.
    """
    if parts is None:
        return
    if isinstance(parts, str):
        parts = [parts]
    # A TextSpeaker already prints, so only voice mode needs an echo.
    echo = not isinstance(ctx.speaker, TextSpeaker)

    for part in parts:
        if not part:
            continue
        text = str(part)
        if echo:
            print(text)
        if ctx.speaker is not None:
            ctx.speaker.speak(text)


def greet(ctx: AppContext) -> None:
    """Speak the startup greeting. Wording is unchanged from the original."""
    today_date = datetime.datetime.now()
    speak_all(
        ctx,
        [
            "Hello, Good " + wishme() + ". I'm your voice assistant.",
            "Today is "
            + today_date.strftime("%d")
            + " of "
            + today_date.strftime("%B")
            + ", And it's currently "
            + today_date.strftime("%I")
            + ":"
            + today_date.strftime("%M")
            + " "
            + today_date.strftime("%p"),
        ],
    )

    # Weather is fetched live, as the original greeting did.
    from assistant.tools.weather import des, temp

    speak_all(
        ctx,
        [
            f"Temperature in {ctx.config.weather_city} is "
            f"{temp()} degree Celsius and with {des()}"
        ],
    )
    speak_all(ctx, ["What can I do for you?"])


# ----------------------------------------------------------------------
# Understanding
# ----------------------------------------------------------------------
def build_runtime_lexicon(router: Router) -> Lexicon:
    """Build the NLU lexicon from the tools the router actually registered.

    The dependency arrow points one way. This function lives in the app
    because the app is what knows about tools; :mod:`assistant.nlu` only
    ever receives the plain ``(intent, patterns)`` data it compiles, so the
    NLU package never imports a tool and the layer stays testable alone.

    Reading the patterns from the router rather than a hand-written table
    means a new tool is understood as soon as it is registered, with no
    second list to keep in step.

    Args:
        router: the router whose tools should be understood.

    Returns:
        A lexicon covering every registered tool, plus
        :data:`~assistant.nlu.COMMON_ALIASES`.
    """
    entries = [(tool.name, tuple(tool.patterns())) for tool in router.tools]
    return build_lexicon(entries, aliases=COMMON_ALIASES)


def _nlu_tool(router: Router, name: str) -> Tool | None:
    """Return the tool for an NLU intent name, or None if unregistered."""
    return router.get(name)


def _legacy_tool(router: Router, utterance: str) -> Tool | None:
    """Return the tool the legacy router chooses, or None."""
    match = router.find_match(utterance)
    if isinstance(match, NoMatch):
        return None
    return match.tool


#: Why the resolver finished, for diagnostics. These describe the path
#: taken, never a change in behaviour.
STATUS_RESOLVED = "resolved"
STATUS_NO_MATCH = "no-match"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_FALLBACK = "fallback"
#: The parser found an intent, but the framing layer judged the sentence to
#: be talking *about* the topic rather than asking for it.
STATUS_REJECTED = "rejected"


@dataclass(frozen=True)
class Resolution:
    """What the assistant decided, and why, for one utterance.

    This is diagnostic data. :func:`resolve` still returns only the tool,
    so nothing that already calls it has to change; the run loop asks for
    the richer record so it can log and optionally print the reasoning.
    """

    tool: Tool | None
    status: str
    parsed: ParsedIntent | None = None
    reason: str = ""
    candidates: tuple[tuple[str, float, str], ...] = ()
    framing: str = ""

    @property
    def tool_name(self) -> str:
        return self.tool.name if self.tool is not None else ""

    def metadata(self) -> dict[str, object] | None:
        """Return log-friendly understanding details, or None if there are none.

        A command that produced no understanding at all, a bare "didn't
        understand", has nothing worth recording, so ``None`` is returned
        and the log line stays exactly what it has always been.
        """
        if self.parsed is None and not self.reason:
            return None

        data: dict[str, object] = {"status": self.status}
        if self.parsed is not None:
            data["intent"] = self.parsed.name
            data["score"] = round(self.parsed.score, 3)
            data["confidence"] = self.parsed.confidence
            data["method"] = self.parsed.method
            data["trigger"] = self.parsed.trigger
        else:
            data["intent"] = "none"
            data["confidence"] = "none"
        if self.reason:
            data["reason"] = self.reason
        if self.framing and self.framing != framing_mod.NEUTRAL:
            data["framing"] = self.framing
        if self.parsed is not None and self.parsed.alternatives:
            data["alternatives"] = [
                f"{item.name}({round(item.score, 3)})"
                for item in self.parsed.alternatives
            ]
        return data


def _ranked_candidates(utterance: str, lexicon: Lexicon) -> tuple[tuple[str, float, str], ...]:
    """Return the full scoring ranking for diagnostics.

    The parser deliberately reduces its result to a winner plus
    alternatives, so the runners-up lose their method. Tuning needs the
    whole ordering, including the weak candidates the parser discarded, so
    this re-runs the scorer directly. It is a pure function and is only
    ever called when diagnostics are being printed.
    """
    from assistant.nlu.normalize import normalize
    from assistant.nlu.scoring import score_intents

    return tuple(
        (item.intent, item.score, item.method)
        for item in score_intents(normalize(utterance), lexicon)
    )


def resolve_detail(
    ctx: AppContext, router: Router, utterance: str, lexicon: Lexicon
) -> Resolution:
    """Resolve an utterance and describe the decision.

    This is :func:`resolve` with the reasoning kept alongside the answer.
    The two share one implementation, so a diagnostic can never describe a
    different decision from the one that was actually made.

    Args:
        ctx: the shared application context.
        router: the router, used for the fallback and to look tools up.
        utterance: the raw utterance, read but never rewritten.
        lexicon: the runtime vocabulary.

    Returns:
        A :class:`Resolution` describing the tool and the path taken.
    """
    try:
        parsed = parse(utterance, lexicon)
    except Exception as exc:  # noqa: BLE001 - understanding must never end the run
        logger.warning(
            "Language understanding failed, falling back to the router: %s", exc
        )
        tool = _legacy_tool(router, utterance)
        return Resolution(
            tool=tool,
            status=STATUS_FALLBACK,
            reason=f"parser exception: {type(exc).__name__}",
        )

    if parsed is None:
        # The parser ran and recognised nothing. That is a decision, not a
        # gap, so the substring router is not consulted: it would match
        # inside words such as "quite" or "denote" and run the wrong tool.
        return Resolution(tool=None, status=STATUS_NO_MATCH)

    if parsed.confidence == AMBIGUOUS:
        # Never silently pick a side between two real candidates.
        asked = clarify(ctx, parsed)
        if asked is not None:
            chosen = _nlu_tool(router, asked)
            if chosen is not None:
                return Resolution(
                    tool=chosen, status=STATUS_AMBIGUOUS, parsed=parsed
                )
        return Resolution(tool=None, status=STATUS_AMBIGUOUS, parsed=parsed)

    chosen = _nlu_tool(router, parsed.name)
    if chosen is not None:
        # The parser matched a word. The framing layer asks the separate
        # question of whether the user was actually asking for anything.
        # A mention runs no tool and, deliberately, does not fall back to
        # the legacy router: that router matches raw substrings and is
        # exactly what would undo this decision.
        verdict = framing_mod.assess(normalize(utterance), parsed.name)
        if verdict == framing_mod.MENTION:
            logger.debug(
                "Framing judged %r to be a mention of %r, not a request",
                utterance,
                parsed.name,
            )
            return Resolution(
                tool=None,
                status=STATUS_REJECTED,
                parsed=parsed,
                reason="framing: mention",
                framing=verdict,
            )
        return Resolution(
            tool=chosen, status=STATUS_RESOLVED, parsed=parsed, framing=verdict
        )

    # The parser named a tool that is not registered. That is a wiring
    # problem rather than a user error, so let the router try.
    logger.warning(
        "The parser chose %r, which no registered tool provides.", parsed.name
    )
    tool = _legacy_tool(router, utterance)
    return Resolution(
        tool=tool,
        status=STATUS_FALLBACK,
        parsed=parsed,
        reason="unregistered intent",
    )


def resolve(
    ctx: AppContext, router: Router, utterance: str, lexicon: Lexicon
) -> Tool | None:
    """Decide which tool should handle one utterance.

    The NLU decides, and the legacy router remains the fallback. Three
    outcomes are possible, in this order:

    1. the NLU returns a confident intent, so that tool is used;
    2. the NLU is genuinely torn, so the user is asked which one they meant;
    3. the NLU cannot decide, so the legacy router is consulted.

    A ``None`` result means "run no tool at all", which is a real answer and
    not an error. It is what keeps a false positive from reaching a tool.

    The legacy fallback is deliberately narrower than it looks. The legacy
    router matches raw substrings, so it reads *"quite good"* as the exit
    command, because ``"quit"`` hides inside *"quite"*; it reads *"i don't
    want to exit"* and *"goodbye is in the dictionary"* as requests to
    quit. Falling back unconditionally would undo every guarantee the NLU
    adds, so the fallback runs **only when the NLU could not decide** -- an
    exception, or an intent no registered tool provides. When the parser
    runs and returns ``None``, that is a considered judgement that this is
    not a command, and the legacy router is not asked.

    Args:
        ctx: the shared application context.
        router: the router, used for the fallback and to look tools up.
        utterance: the raw utterance, exactly as the listener produced it.
            It is only ever read, never rewritten.
        lexicon: the runtime vocabulary.

    Returns:
        The tool to run, or ``None`` when nothing should run.
    """
    return resolve_detail(ctx, router, utterance, lexicon).tool


def format_nlu_debug(
    utterance: str, resolution: Resolution, candidates: Sequence[tuple[str, float, str]] = ()
) -> str:
    """Render one human-readable ``[NLU]`` diagnostic block.

    Printed only when ``--debug-nlu`` is given, and written to stdout, so
    it is never spoken aloud and never reaches the speaker or the listener.
    Purely a function of its arguments, so the same command always produces
    byte-identical output.

    Args:
        utterance: what the user said.
        resolution: the decision that was actually taken.
        candidates: the full scoring ranking, when it was gathered.

    Returns:
        A multi-line block beginning with ``[NLU]``.
    """
    lines = ["[NLU]", f"input: {utterance}", f"status: {resolution.status}"]

    if resolution.parsed is None:
        lines.append("intent: none")
        lines.append("confidence: none")
    else:
        parsed = resolution.parsed
        lines.append(f"intent: {parsed.name}")
        lines.append(f"score: {round(parsed.score, 3):.3f}")
        lines.append(f"confidence: {parsed.confidence}")
        lines.append(f"method: {parsed.method}")
        lines.append(f"trigger: {parsed.trigger}")

    if resolution.reason:
        lines.append(f"reason: {resolution.reason}")

    if resolution.framing:
        lines.append(f"framing: {resolution.framing}")

    if resolution.parsed is not None and resolution.parsed.alternatives:
        lines.append("alternatives:")
        for item in resolution.parsed.alternatives:
            lines.append(f"  - {item.name} ({round(item.score, 3):.3f})")

    if candidates:
        lines.append("candidates:")
        # Pad the name column so the scores line up and a long ranking
        # stays readable. The width is derived from this ranking only, so
        # the output stays deterministic.
        width = max(len(name) for name, _, _ in candidates)
        for position, (name, score, method) in enumerate(candidates, start=1):
            lines.append(
                f"  {position}. {name.ljust(width)}  {round(score, 3):.3f}  {method}"
            )

    lines.append(f"resolved_tool: {resolution.tool_name or 'none'}")
    return "\n".join(lines)


def clarification_question(names: Sequence[str]) -> str:
    """Build the spoken question for an ambiguous utterance.

    The wording is generated from the real candidates rather than looked up
    per intent, so any pair of tools produces a sensible question and none
    is hard-coded. Ordering is the parser's ranking, so the same utterance
    always produces the same sentence.

    Args:
        names: candidate tool names, best first. Only the first two are
            used, because a question offering more than two choices is
            harder to answer than simply repeating the command.

    Returns:
        A single sentence to speak.
    """
    offered = [name for name in names[:2] if name]
    if len(offered) < 2:
        return "Could you say that again?"
    return f"Did you mean {offered[0]} or {offered[1]}?"


def clarify(ctx: AppContext, parsed: ParsedIntent) -> str | None:
    """Ask which intent the user meant, and return the one they picked.

    Reuses the app's existing one-question-then-one-listen shape, the same
    mechanism :func:`ask_follow_up` uses, rather than introducing a
    conversation framework.

    The reply is matched against the offered candidates only, and only as a
    whole word, so a clarification can never become a way to run an
    arbitrary tool.

    Args:
        ctx: the shared application context.
        parsed: the ambiguous result from the parser.

    Returns:
        The chosen tool name, or ``None`` when the reply picked none.
    """
    offered = [parsed.name] + [item.name for item in parsed.alternatives]
    speak_all(ctx, [clarification_question(offered)])

    if ctx.listener is None:
        return None

    answer = ctx.listener.listen()
    if not answer or answer == NOTHING_HEARD or answer in SPEECH_ERRORS:
        return None

    words = set(answer.lower().split())
    for name in offered:
        if name in words:
            return name
    return None


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def ask_follow_up(ctx: AppContext, router: Router, sentinel: str) -> str:
    """Run the second turn of a two-step request.

    A tool returns a sentinel such as ``NEEDS_TOPIC`` when the user has not
    said what the request is about. This asks the matching question, listens
    once, and hands the answer back to the tool.

    Keeping this in the app is what lets the tools stay free of any speech
    dependency: they return data, the conversation lives here.

    Args:
        ctx: the shared application context.
        router: the router, used to re-dispatch the completed request.
        sentinel: the sentinel value the tool returned.

    Returns:
        The tool's response for the completed request.
    """
    prompt = PROMPTS.get(sentinel, "Could you repeat that?")
    speak_all(ctx, [prompt])

    if ctx.listener is None:
        return "Sorry, I could not hear the answer."

    answer = ctx.listener.listen()
    if not answer or answer == NOTHING_HEARD or answer in SPEECH_ERRORS:
        return "Sorry, I didn't catch that. Please try again."

    # Re-dispatch using the phrase shape the tool's own extractor understands,
    # so the follow-up answer is parsed exactly as a single-turn request
    # would be. "information python" would not match any topic prefix.
    if sentinel == WikipediaTool.NEEDS_TOPIC:
        completed = f"information about {answer}"
    else:
        completed = f"play {answer}"

    response = router.dispatch(completed, ctx)
    if isinstance(response, str) and response in PROMPTS:
        # The answer still had no usable value; do not loop.
        return "Sorry, I didn't catch that. Please try again."
    return response



def run(ctx: AppContext, router: Router, debug_nlu: bool = False) -> int:
    """Listen, understand, route, and speak until the user exits.

    Args:
        ctx: the shared application context.
        router: the router holding the registered tools.
        debug_nlu: when true, print a diagnostic block for each command.
            This only prints: it never changes which tool runs, what is
            said, or what is recorded.

    Returns:
        A process exit code: 0 for a clean exit.
    """
    if ctx.listener is None:
        logger.error("No listener is attached; cannot start the command loop.")
        return 1

    # Built once from the registered tools, so every command in the session
    # is understood with exactly the same vocabulary.
    lexicon = build_runtime_lexicon(router)

    while True:
        print("Listening for your command...")
        utterance = ctx.listener.listen()

        if not utterance or utterance == NOTHING_HEARD:
            # Silence: listen again without apologising.
            continue

        if utterance in SPEECH_ERRORS:
            # The listener already produced a message to speak.
            speak_all(ctx, [utterance])
            continue

        print(f"You said: {utterance}")

        resolution = resolve_detail(ctx, router, utterance, lexicon)
        tool = resolution.tool
        tool_name = resolution.tool_name

        if debug_nlu:
            print(
                format_nlu_debug(
                    utterance, resolution, _ranked_candidates(utterance, lexicon)
                )
            )

        # The tool always receives the original utterance, exactly as the
        # router used to pass it. Understanding happens alongside the
        # request and never rewrites it, so slot extraction sees the user's
        # own words rather than the parser's normalised text.
        response = _run_tool(tool, utterance, ctx)

        # A tool that needs more input triggers the second turn here.
        if isinstance(response, str) and response in PROMPTS:
            response = ask_follow_up(ctx, router, response)
        elif not isinstance(response, list) and not response:
            # An empty response means nothing matched; keep the original
            # wording the assistant used before the router existed. When the
            # optional AI layer is available it may answer first, and its
            # answer replaces this line only in the one case it is allowed.
            response = ai_reply(ctx, resolution.status, utterance) or MSG_DIDNT_UNDERSTAND

        speak_all(ctx, response)

        # One log line per command. A multi-part response is joined so the
        # log stays readable, and log_command truncates long text.
        if isinstance(response, list):
            logged = " | ".join(response)
        else:
            logged = str(response)
        ctx.log_command(tool_name or "none", utterance, logged, nlu=resolution.metadata())

        if tool_name == SystemTool.name:
            return 0


def _run_tool(tool: Tool | None, utterance: str, ctx: AppContext) -> str | list[str]:
    """Run one tool and normalise its answer, mirroring the router.

    Dispatch moved out of the router when the NLU took over, and this keeps
    the router's contract intact: a tool that raises is logged and reported
    as a spoken message instead of ending the run, a ``None`` becomes an
    empty string, a list is passed through as a list of strings, and
    anything else is stringified.

    Args:
        tool: the tool to run, or ``None`` when nothing matched.
        utterance: the raw utterance, passed through untouched.
        ctx: the shared application context.

    Returns:
        Either a string, or a list of strings for a multi-part reply.
    """
    if tool is None:
        return ""

    try:
        response = tool.handle(utterance, ctx)
    except Exception as exc:  # noqa: BLE001 - a tool must not kill the app
        logger.exception("Tool %r raised while handling a request", tool.name)
        return f"Sorry, something went wrong while running {tool.name}: {exc}"

    if response is None:
        return ""
    if isinstance(response, list):
        return [str(item) for item in response]
    return str(response)



# ----------------------------------------------------------------------
# Optional AI response layer
# ----------------------------------------------------------------------
def ai_reply(ctx: AppContext, status: str, utterance: str) -> str:
    """Return an optional AI answer for an utterance nothing else handled.

    This is the single integration point of the AI feature, and it is
    deliberately narrow. It is reached only from the "empty response" branch of
    :func:`run`, which the loop enters **after** normalisation, parsing,
    scoring, the framing guards and the router have all already declined the
    utterance. Recognised commands therefore never come here.

    Two gates protect the existing behaviour:

    1. **Status gate.** AI is consulted only for
       :data:`STATUS_NO_MATCH`, the single status meaning "the parser decided
       this is not a command". The other non-matching statuses are refused on
       purpose:

       * :data:`STATUS_REJECTED` -- the framing layer judged the sentence to be
         *talking about* a topic rather than requesting it. Letting AI answer
         would replace a deliberate safety decision with a generated sentence,
         so it is excluded.
       * :data:`STATUS_AMBIGUOUS` -- the app already asked a clarifying
         question; answering the original utterance would skip that turn.
       * :data:`STATUS_FALLBACK` -- an internal resolver problem, not user
         input worth answering.

    2. **Availability gate.** The responder declines when the feature is off,
       the key is missing, or the provider fails, in which case this returns
       ``""`` and the caller speaks the ordinary
       :data:`MSG_DIDNT_UNDERSTAND` reply.

    The returned text is spoken as-is. It is never routed, parsed or executed,
    so an AI reply can say something but cannot *do* something.

    Args:
        ctx: the shared application context.
        status: the resolver status for this utterance.
        utterance: the raw utterance, passed to the provider unchanged.

    Returns:
        The AI reply, or ``""`` when AI must not answer or could not.
    """
    # Imported here, inside the function, so that starting the assistant loads
    # no AI code at all when the feature is not in use.
    from assistant.ai import AIResponder

    responder = ctx.ai
    if not isinstance(responder, AIResponder) or not responder.enabled:
        return ""

    # The one status that means "this is not a command I know".
    if status != STATUS_NO_MATCH:
        logger.debug(
            "Not consulting the AI layer: resolver status is %r.", status
        )
        return ""

    try:
        reply = responder.respond(utterance)
    except Exception as exc:  # noqa: BLE001 - belt and braces; respond() is safe
        # AIResponder.respond is already written never to raise. This guard
        # exists so that no future change to it can end the command loop.
        logger.warning("The AI layer failed unexpectedly; using the fallback: %s", exc)
        return ""

    return reply or ""


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def build_context(config: Config, text_mode: bool) -> AppContext:
    """Create the single AppContext, with speech I/O attached.

    In text mode nothing touches a sound card or a microphone: pyttsx3 and
    SpeechRecognition are not even imported.
    """
    ctx = AppContext(config=config, logger=get_logger("assistant.app"))

    if text_mode:
        ctx.speaker = TextSpeaker()
        ctx.listener = TextListener()
        print("Text mode: type a command. No microphone or speech is used.")
    else:
        # These constructors are the only place the audio stack is loaded.
        ctx.speaker = voice_mod.VoiceSpeaker(
            rate=config.tts_rate, voice_name=config.tts_voice_name
        )
        ctx.listener = voice_mod.VoiceListener()

    ctx.ai = attach_ai(config)
    return ctx


def attach_ai(config: Config) -> Any:
    """Build the optional AI responder and attach it to the context.

    Attached alongside the database and the speech I/O, and optional in the
    same way: when the feature is off or no key is configured the context
    simply gets a disabled responder, and the assistant behaves exactly as it
    did before the AI layer existed.

    Never raises. A failure to build the responder is logged and produces a
    disabled one, because an optional feature must not be able to stop the
    assistant from starting.

    Args:
        config: supplies the AI settings.

    Returns:
        An :class:`~assistant.ai.AIResponder`, enabled or disabled.
    """
    try:
        from assistant.ai import build_responder

        responder = build_responder(config)
    except Exception as exc:  # noqa: BLE001 - optional feature must not block startup
        logger.warning("AI layer unavailable, continuing without it: %s", exc)
        from assistant.ai import AIResponder

        return AIResponder(None)

    logger.info("AI response layer %s", "enabled" if responder.enabled else "disabled")
    return responder


def attach_database(ctx: AppContext, config: Config) -> None:
    """Open and initialise the command-history database.

    Exactly one :class:`~assistant.core.database.Database` is created per
    run, here, and it is the only place that opens a connection. On any
    failure the context is left with ``db = None`` and the assistant carries
    on in logging-only mode: persistence is a bonus, never a requirement.

    Args:
        ctx: the context to attach the database to.
        config: supplies the database path.
    """
    try:
        database = Database(config.db_path)
        version = database.initialize()
        ctx.db = database
        logger.info(
            "Command history enabled at %s (schema version %s)",
            database.path,
            version,
        )
    except Exception as exc:  # noqa: BLE001 - never block startup
        logger.warning(
            "Command history disabled, continuing without a database: %s", exc
        )
        ctx.db = None


def close_database(ctx: AppContext) -> None:
    """Close the database if one is attached. Safe to call when there is none."""
    if ctx.db is None:
        return
    try:
        ctx.db.close()
        logger.debug("Command history database closed cleanly.")
    except Exception as exc:  # noqa: BLE001 - never block shutdown
        logger.warning("Could not close the database cleanly: %s", exc)
    finally:
        ctx.db = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line. Voice mode is the default."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Voice controlled virtual assistant.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Talk to the assistant by typing instead of speaking.",
    )
    parser.add_argument(
        "--debug-nlu",
        action="store_true",
        help="Print a diagnostic block for each command the NLU interprets.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the log level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Start the assistant. Returns a process exit code."""
    args = parse_args(argv)

    config = get_config()
    if args.log_level:
        config = replace_config_log_level(config, args.log_level)

    setup_logging(config)
    logger.info("Assistant starting (text_mode=%s)", args.text)
    logger.info(config.describe())
    if args.debug_nlu:
        logger.info("Language understanding diagnostics are enabled.")

    ctx = build_context(config, args.text)
    router = build_default_router()
    attach_database(ctx, config)

    try:
        greet(ctx)
        return run(ctx, router, debug_nlu=args.debug_nlu)
    finally:
        # Runs on a clean exit and on an error, so the database is never
        # left holding an open connection.
        close_database(ctx)


def replace_config_log_level(config: Config, level: str) -> Config:
    """Return a copy of ``config`` with a different log level."""
    import dataclasses

    return dataclasses.replace(config, log_level=level.upper())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
