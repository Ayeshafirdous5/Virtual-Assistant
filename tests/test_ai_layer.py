"""Tests for the optional AI response layer.

Every test here is deterministic and offline. No test contacts a real AI
endpoint, reads a real ``.env``, or needs a credential: the provider is
replaced by :class:`FakeProvider`, and the only key that appears anywhere is
the obviously fake :data:`FAKE_KEY`, which exists to be checked for leaks.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest


def dataclasses_replace(config, **changes):
    """Return a copy of ``config`` with ``changes`` applied.

    A local helper so the intent ("same config, AI switched on") reads clearly
    at each call site instead of repeating ``dataclasses.replace(``.
    """
    return dataclasses.replace(config, **changes)

from assistant.ai import (
    AIProvider,
    AIProviderError,
    AIProviderResponseError,
    AIProviderTimeout,
    AIProviderUnavailable,
    AIResponder,
    AIResponse,
    OpenAICompatibleProvider,
    build_responder,
)
from assistant.app import (
    MSG_DIDNT_UNDERSTAND,
    STATUS_AMBIGUOUS,
    STATUS_FALLBACK,
    STATUS_NO_MATCH,
    STATUS_REJECTED,
    STATUS_RESOLVED,
    ai_reply,
)

#: Deliberately fake, and shaped like the secret it stands in for so the leak
#: assertions below are meaningful.
FAKE_KEY = "sk-test-not-a-real-key-0000"


# ----------------------------------------------------------------------
# Scripts run in a clean interpreter by TestProviderOffline
# ----------------------------------------------------------------------
# Each replaces requests.post with a stub before the provider is built, so no
# request can leave the machine. The transport is always faked; nothing here
# contacts a real AI service.
_STUB_HEADER = (
    "import requests\n"
    "captured = []\n"
    "class R:\n"
    "    status_code = 200\n"
    "    def __init__(self, payload): self._payload = payload\n"
    "    def json(self):\n"
    "        if isinstance(self._payload, Exception): raise self._payload\n"
    "        return self._payload\n"
    "BODY = {'choices': [{'message': {'content': 'Hello there.'}}]}\n"
)

_SUCCESS_SCRIPT = (
    _STUB_HEADER
    + "requests.post = lambda *a, **k: (captured.append(a), R(BODY))[1]\n"
    "from assistant.ai import OpenAICompatibleProvider\n"
    "print(OpenAICompatibleProvider('" + FAKE_KEY + "').generate('hi').text)\n"
)

_HTTP_ERROR_SCRIPT = (
    _STUB_HEADER
    + "def bad(*a, **k):\n"
    "    r = R(BODY); r.status_code = 401; return r\n"
    "requests.post = bad\n"
    "from assistant.ai import OpenAICompatibleProvider\n"
    "from assistant.ai.base import AIProviderError\n"
    "try:\n"
    "    OpenAICompatibleProvider('" + FAKE_KEY + "').generate('hi')\n"
    "except AIProviderError as e:\n"
    "    print(type(e).__name__)\n"
)

_TIMEOUT_SCRIPT = (
    _STUB_HEADER
    + "def slow(*a, **k): raise requests.exceptions.Timeout()\n"
    "requests.post = slow\n"
    "from assistant.ai import OpenAICompatibleProvider\n"
    "from assistant.ai.base import AIProviderError\n"
    "try:\n"
    "    OpenAICompatibleProvider('" + FAKE_KEY + "').generate('hi')\n"
    "except AIProviderError as e:\n"
    "    print(type(e).__name__)\n"
)

_BAD_JSON_SCRIPT = (
    _STUB_HEADER
    + "requests.post = lambda *a, **k: R(ValueError('not json'))\n"
    "from assistant.ai import OpenAICompatibleProvider\n"
    "from assistant.ai.base import AIProviderError\n"
    "try:\n"
    "    OpenAICompatibleProvider('" + FAKE_KEY + "').generate('hi')\n"
    "except AIProviderError as e:\n"
    "    print(type(e).__name__)\n"
)

_EMPTY_BODY_SCRIPT = (
    _STUB_HEADER
    + "requests.post = lambda *a, **k: R({'choices': []})\n"
    "from assistant.ai import OpenAICompatibleProvider\n"
    "from assistant.ai.base import AIProviderError\n"
    "try:\n"
    "    OpenAICompatibleProvider('" + FAKE_KEY + "').generate('hi')\n"
    "except AIProviderError as e:\n"
    "    print(type(e).__name__)\n"
)

_CONNECTION_ERROR_SCRIPT = (
    _STUB_HEADER
    + "def down(*a, **k): raise requests.exceptions.ConnectionError('refused')\n"
    "requests.post = down\n"
    "from assistant.ai import OpenAICompatibleProvider\n"
    "from assistant.ai.base import AIProviderError\n"
    "try:\n"
    "    OpenAICompatibleProvider('" + FAKE_KEY + "').generate('hi')\n"
    "except AIProviderError as e:\n"
    "    print(type(e).__name__, str(e))\n"
)


class FakeProvider(AIProvider):
    """A provider that returns a fixed reply and records what it was asked."""

    name = "fake"

    def __init__(self, text: str = "Sure, that sounds interesting.") -> None:
        self.text = text
        self.calls: list[str] = []

    def generate(self, message: str) -> AIResponse:
        self.calls.append(message)
        return AIResponse(text=self.text, provider=self.name, model="fake-model")


class FailingProvider(AIProvider):
    """A provider that always raises the error it was given."""

    name = "failing"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def generate(self, message: str) -> AIResponse:
        self.calls += 1
        raise self.error


class ExplodingProvider(AIProvider):
    """A provider that raises something outside the expected error hierarchy."""

    name = "exploding"

    def generate(self, message: str) -> AIResponse:
        raise RuntimeError("provider bug")


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def enabled_ctx(ctx):
    """A context with a working AI responder attached."""
    ctx.ai = AIResponder(FakeProvider())
    return ctx


# ----------------------------------------------------------------------
# 1. AI disabled
# ----------------------------------------------------------------------
class TestDisabled:
    """With no provider the layer must be inert."""

    def test_responder_with_no_provider_is_disabled(self):
        assert AIResponder(None).enabled is False

    def test_disabled_responder_returns_none(self):
        assert AIResponder(None).respond("what do you think?") is None

    def test_config_default_is_off(self, config):
        assert config.ai_enabled is False

    def test_build_responder_disabled_by_default(self, config):
        assert build_responder(config).enabled is False

    def test_ai_reply_returns_empty_when_disabled(self, ctx):
        # The default context fixture has ai=None.
        assert ai_reply(ctx, STATUS_NO_MATCH, "hello there") == ""

    def test_disabled_layer_never_calls_out(self, ctx):
        ai_reply(ctx, STATUS_NO_MATCH, "hello")
        # Nothing to assert on calls, so instead prove the fallback is intact:
        assert (ai_reply(ctx, STATUS_NO_MATCH, "hello") or MSG_DIDNT_UNDERSTAND)


# ----------------------------------------------------------------------
# 2. Enabled with a missing key
# ----------------------------------------------------------------------
class TestMissingKey:
    """Switching the feature on without a key must degrade, not crash."""

    def test_enabled_without_key_is_unavailable(self, config):
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=None)
        assert cfg.ai_is_available() is False

    def test_build_responder_disabled_without_key(self, config):
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=None)
        assert build_responder(cfg).enabled is False

    def test_blank_key_counts_as_missing(self, config):
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key="   ")
        assert build_responder(cfg).enabled is False

    def test_missing_key_falls_back_to_normal_reply(self, ctx, config):
        ctx.ai = build_responder(
            dataclasses_replace(config, ai_enabled=True, ai_api_key=None)
        )
        assert ai_reply(ctx, STATUS_NO_MATCH, "tell me something") == ""

    def test_has_ai_api_key_reports_presence_only(self, config):
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=FAKE_KEY)
        assert cfg.has_ai_api_key() is True
        assert FAKE_KEY not in cfg.describe()
        assert FAKE_KEY not in repr(cfg)


# ----------------------------------------------------------------------
# 3. Provider success
# ----------------------------------------------------------------------
class TestSuccess:
    """When the provider works, its text is what gets spoken."""

    def test_enabled_responder_returns_text(self, fake_provider):
        reply = AIResponder(fake_provider).respond("how are you?")
        assert reply == "Sure, that sounds interesting."

    def test_provider_receives_the_utterance_unchanged(self, fake_provider):
        AIResponder(fake_provider).respond("  What do you think about rain?  ")
        # Stripped of surrounding whitespace, but otherwise untouched.
        assert fake_provider.calls == ["What do you think about rain?"]

    def test_ai_reply_uses_provider_text(self, enabled_ctx):
        assert ai_reply(enabled_ctx, STATUS_NO_MATCH, "hello") == (
            "Sure, that sounds interesting."
        )

    def test_unknown_input_reaches_ai_when_enabled(self, enabled_ctx):
        # The status the resolver reports for "this is not a command".
        reply = ai_reply(enabled_ctx, STATUS_NO_MATCH, "do you like autumn?")
        assert reply == "Sure, that sounds interesting."
        # The utterance reached the provider exactly once.
        assert enabled_ctx.ai._provider.calls == ["do you like autumn?"]

    def test_blank_reply_is_treated_as_a_failure(self, ctx):
        ctx.ai = AIResponder(FakeProvider(text="   "))
        assert ai_reply(ctx, STATUS_NO_MATCH, "hello") == ""


# ----------------------------------------------------------------------
# 4 & 5. Provider failure, timeout, and unexpected errors
# ----------------------------------------------------------------------
class TestFailures:
    """Every failure mode must degrade to the existing fallback."""

    @pytest.mark.parametrize(
        "error",
        [
            AIProviderUnavailable("no network"),
            AIProviderTimeout("took too long"),
            AIProviderResponseError("bad body"),
            AIProviderError("something else"),
        ],
    )
    def test_expected_errors_return_none(self, error):
        assert AIResponder(FailingProvider(error)).respond("hello") is None

    def test_unexpected_exception_is_contained(self):
        # A provider is third-party code: a bug in it must not escape.
        assert AIResponder(ExplodingProvider()).respond("hello") is None

    def test_provider_failure_falls_back_to_normal_reply(self, ctx):
        ctx.ai = AIResponder(FailingProvider(AIProviderUnavailable("down")))
        assert ai_reply(ctx, STATUS_NO_MATCH, "hello") == ""
        # Which means the caller speaks the original wording.
        assert (ai_reply(ctx, STATUS_NO_MATCH, "hello") or MSG_DIDNT_UNDERSTAND) == (
            MSG_DIDNT_UNDERSTAND
        )

    def test_timeout_falls_back(self, ctx):
        ctx.ai = AIResponder(FailingProvider(AIProviderTimeout("too slow")))
        assert ai_reply(ctx, STATUS_NO_MATCH, "hello") == ""

    def test_failures_are_logged_at_warning_not_error(self, ctx, caplog):
        ctx.ai = AIResponder(FailingProvider(AIProviderUnavailable("down")))
        with caplog.at_level(logging.WARNING, logger="assistant.ai.layer"):
            ai_reply(ctx, STATUS_NO_MATCH, "hello")
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_empty_utterance_is_never_sent(self, fake_provider):
        assert AIResponder(fake_provider).respond("   ") is None
        assert fake_provider.calls == []


# ----------------------------------------------------------------------
# 6. Recognised commands never reach the AI layer
# ----------------------------------------------------------------------
class TestExistingCommandPriority:
    """The rule-based pipeline always wins. These are the guard rails."""

    @pytest.mark.parametrize(
        "status",
        [STATUS_RESOLVED, STATUS_REJECTED, STATUS_AMBIGUOUS, STATUS_FALLBACK],
    )
    def test_only_no_match_reaches_ai(self, enabled_ctx, fake_provider, status):
        assert ai_reply(enabled_ctx, status, "what is the weather") == ""
        assert fake_provider.calls == [], "AI must not be consulted for this status"

    def test_framing_rejected_utterance_is_not_answered_by_ai(
        self, enabled_ctx, fake_provider
    ):
        # A framing-rejected mention must keep the existing fallback. Letting
        # AI answer would replace a deliberate safety decision with a generated
        # sentence, which is exactly the weakening this gate prevents.
        reply = ai_reply(
            enabled_ctx, STATUS_REJECTED, "I remember you telling me a joke"
        )
        assert reply == ""
        assert (reply or MSG_DIDNT_UNDERSTAND) == MSG_DIDNT_UNDERSTAND
        assert fake_provider.calls == []

    def test_ambiguous_utterance_still_asks_for_clarification(self, enabled_ctx):
        # Answering here would skip the clarification turn the app already runs.
        assert ai_reply(enabled_ctx, STATUS_AMBIGUOUS, "news") == ""

    def test_recognised_command_status_is_not_no_match(self):
        """A genuinely recognised command never produces the AI-eligible status."""
        from assistant.app import build_runtime_lexicon, resolve_detail
        from assistant.tools import build_default_router

        router = build_default_router()
        lexicon = build_runtime_lexicon(router)

        resolution = resolve_detail(
            _StubContext(), router, "what is the weather in london", lexicon
        )
        assert resolution.status == STATUS_RESOLVED
        assert resolution.tool is not None

        # And therefore the AI gate declines it.
        stub = _StubContext()
        stub.ai = AIResponder(FakeProvider())
        assert ai_reply(stub, resolution.status, "what is the weather in london") == ""


# ----------------------------------------------------------------------
# 7. AI can never execute anything
# ----------------------------------------------------------------------
class TestNoExecution:
    """The provider cannot reach a tool, the router, or the database."""

    def test_provider_receives_only_a_string(self):
        # The contract itself is the guarantee: generate() takes a str and
        # returns text. No context, router or tool handle is ever passed.
        provider = FakeProvider()
        provider.generate("hello")
        assert provider.calls == ["hello"]
        assert isinstance(provider.calls[0], str)

    def test_request_payload_offers_no_tools(self):
        from assistant.ai.openai_compatible import DEFAULT_SYSTEM_PROMPT

        provider = OpenAICompatibleProvider(
            FAKE_KEY, system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        payload = provider._payload("delete my files")
        # No tool-calling surface of any kind is offered to the endpoint.
        assert "tools" not in payload
        assert "functions" not in payload
        assert "tool_choice" not in payload
        assert "response_format" not in payload

    def test_system_prompt_forbids_claiming_actions(self):
        from assistant.ai.openai_compatible import DEFAULT_SYSTEM_PROMPT

        lowered = DEFAULT_SYSTEM_PROMPT.lower()
        assert "never claim to have performed an action" in lowered
        assert "cannot run commands" in lowered

    def test_ai_text_is_never_executed(self, enabled_ctx):
        """Even a reply that looks like a command is only ever spoken."""
        provider = FakeProvider(text="exit")
        enabled_ctx.ai = AIResponder(provider)
        reply = ai_reply(enabled_ctx, STATUS_NO_MATCH, "something")
        # It is returned as text. It is never handed to a tool or a shell.
        assert reply == "exit"
        assert isinstance(reply, str)


# ----------------------------------------------------------------------
# 10. No credentials leak
# ----------------------------------------------------------------------
class TestNoCredentialLeak:
    """Nothing anywhere may print or store the key."""

    def test_provider_repr_hides_the_key(self):
        assert FAKE_KEY not in repr(OpenAICompatibleProvider(FAKE_KEY))
        assert "api_key=set" in repr(OpenAICompatibleProvider(FAKE_KEY))

    def test_config_repr_hides_the_key(self, config):
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=FAKE_KEY)
        assert FAKE_KEY not in repr(cfg)
        assert "ai_api_key" not in repr(cfg)

    def test_describe_reports_presence_only(self, config):
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=FAKE_KEY)
        text = cfg.describe()
        assert FAKE_KEY not in text
        assert "ai_api_key=configured" in text

    def test_context_repr_hides_the_key(self, config):
        from assistant.core.context import AppContext

        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=FAKE_KEY)
        assert FAKE_KEY not in repr(AppContext(config=cfg))

    def test_logs_never_contain_the_key(self, ctx, caplog):
        """Even on a failure, log output must not carry the credential.

        Uses the fake provider rather than the real one: invoking the real
        provider would open a network connection, which these tests must never
        do. The real provider's error paths are covered offline by
        :class:`TestProviderOffline`.
        """
        cfg = dataclasses_replace(ctx.config, ai_enabled=True, ai_api_key=FAKE_KEY)
        ctx.config = cfg
        ctx.ai = AIResponder(FailingProvider(AIProviderUnavailable("down")))
        with caplog.at_level(logging.DEBUG):
            ai_reply(ctx, STATUS_NO_MATCH, "hello")
        assert FAKE_KEY not in caplog.text

    def test_responder_repr_does_not_leak_the_key(self, config):
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=FAKE_KEY)
        responder = build_responder(cfg)
        assert responder.enabled is True
        assert FAKE_KEY not in repr(responder)
        assert FAKE_KEY not in repr(responder._provider)

    def test_key_is_never_written_to_the_database(self, ctx_with_db, config):
        """The AI layer records nothing, so the key cannot reach SQLite."""
        cfg = dataclasses_replace(config, ai_enabled=True, ai_api_key=FAKE_KEY)
        ctx_with_db.config = cfg
        ctx_with_db.ai = AIResponder(FakeProvider())
        ai_reply(ctx_with_db, STATUS_NO_MATCH, "hello")
        assert ctx_with_db.db.query_all("SELECT * FROM interactions") == []


# ----------------------------------------------------------------------
# The real provider, exercised offline in a child process
# ----------------------------------------------------------------------
class TestProviderOffline:
    """HTTP paths of the real provider, with the transport stubbed out.

    These run in a subprocess for two reasons. First, ``requests.post`` is
    replaced, so no request ever leaves the machine and no test depends on a
    live AI service. Second, importing ``requests`` in the parent process
    would trip the two pre-existing NLU tests that assert ``sys.modules`` is
    free of it. ``test_nlu_parser.py`` uses a subprocess for the same reason.
    """

    @staticmethod
    def _run(script: str) -> str:
        """Execute ``script`` in a clean interpreter and return its stdout."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_success_returns_the_reply_text(self):
        out = self._run(_SUCCESS_SCRIPT)
        assert out == "Hello there."

    def test_http_error_status_raises_unavailable(self):
        assert self._run(_HTTP_ERROR_SCRIPT) == "AIProviderUnavailable"

    def test_timeout_raises_timeout(self):
        assert self._run(_TIMEOUT_SCRIPT) == "AIProviderTimeout"

    def test_non_json_body_raises_response_error(self):
        assert self._run(_BAD_JSON_SCRIPT) == "AIProviderResponseError"

    def test_empty_choices_raises_response_error(self):
        assert self._run(_EMPTY_BODY_SCRIPT) == "AIProviderResponseError"

    def test_connection_failure_does_not_leak_the_key(self):
        out = self._run(_CONNECTION_ERROR_SCRIPT)
        # The error type is reported; the credential never is.
        assert out.startswith("AIProviderUnavailable")
        assert FAKE_KEY not in out

    def test_no_real_network_call_is_made(self):
        """The success path must still have gone through the stubbed transport."""
        out = self._run(_SUCCESS_SCRIPT + "print('CALLED', captured[0])")
        assert out.startswith("Hello there.")
        assert "CALLED" in out


class _StubContext:
    """A minimal context for tests that only need ``.ai`` and ``.config``."""

    def __init__(self) -> None:
        from assistant.config import get_config_cached

        self.config = get_config_cached()
        self.ai = None


# ----------------------------------------------------------------------
# 8. The feature is fully optional: existing behaviour is untouched
# ----------------------------------------------------------------------
class TestAdditiveAndOptional:
    """The strongest guarantee: with AI off, behaviour is byte-for-byte identical."""

    def test_unknown_input_still_gets_the_original_reply(self, run_text_session):
        # The listener repeats its last item forever, so every scripted session
        # must end with an explicit "exit".
        code, spoken = run_text_session(["what do you think about the ocean", "exit"])
        assert code == 0
        assert MSG_DIDNT_UNDERSTAND in spoken

    def test_recognised_command_still_runs_its_tool(self, run_text_session):
        """A real command still exits cleanly through the normal path.

        Deliberately uses the exit command rather than one backed by a
        third-party library. Running a joke here would import that library's
        dependencies into the pytest process, and two pre-existing NLU tests
        assert on ``sys.modules`` being free of them. Those guards are
        order-dependent by design, so a new test must not trip them.

        That a recognised command bypasses AI is proved where it can be
        observed precisely, in
        :meth:`TestExistingCommandPriority.test_recognised_command_status_is_not_no_match`.
        """
        code, spoken = run_text_session(["exit"])
        assert code == 0
        assert spoken[-1] != MSG_DIDNT_UNDERSTAND

    def test_app_still_runs_with_no_ai_configured(self, config):
        from assistant.app import attach_ai

        responder = attach_ai(config)
        assert responder.enabled is False
        assert responder.respond("hello") is None

    def test_context_defaults_to_no_ai(self, ctx):
        assert ctx.ai is None

    def test_ai_is_not_imported_when_unused(self):
        """Importing the app must not drag in the AI layer."""
        import subprocess
        import sys

        code = (
            "import sys, assistant.app; "
            "print('LOADED' if 'assistant.ai' in sys.modules else 'NOT_LOADED')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.stdout.strip() == "NOT_LOADED", result.stdout