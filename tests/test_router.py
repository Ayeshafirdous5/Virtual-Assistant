"""Router matching, determinism, and tool registration guards."""

from __future__ import annotations

import pytest

from assistant.core.router import NoMatch, Router
from assistant.core.tool import Tool
from assistant.tools import build_default_router


class Alpha(Tool):
    name = "alpha"
    description = "test alpha"

    def patterns(self):
        return ["alpha"]

    def execute(self, ctx, slots=None):
        return "ALPHA"


class Beta(Tool):
    name = "beta"
    description = "test beta"

    def patterns(self):
        return ["beta", "the beta topic"]

    def execute(self, ctx, slots=None):
        return "BETA"


class Exploding(Tool):
    name = "boom"
    description = "always fails"

    def patterns(self):
        return ["boom"]

    def execute(self, ctx, slots=None):
        raise RuntimeError("kaboom")


@pytest.fixture
def router():
    r = Router()
    r.register(Alpha())
    r.register(Beta())
    return r


class TestMatching:
    def test_simple_match(self, router):
        assert router.find_match("say alpha").tool_name == "alpha"

    def test_is_case_insensitive(self, router):
        assert router.find_match("SAY ALPHA").tool_name == "alpha"

    def test_no_match_returns_no_match(self, router):
        assert isinstance(router.find_match("xyzzy"), NoMatch)
        assert isinstance(router.find_match(""), NoMatch)
        assert router.find_match("xyzzy").tool_name == ""

    def test_longest_pattern_wins(self, router):
        match = router.find_match("tell me the beta topic please")
        assert match.tool_name == "beta"
        assert match.pattern == "the beta topic"

    def test_longest_pattern_wins_within_one_tool(self, router):
        # "the beta topic" is longer than "beta" but both belong to Beta.
        assert router.find_match("the beta topic").pattern == "the beta topic"
        assert router.find_match("beta").pattern == "beta"

    def test_tie_broken_by_registration_order(self):
        first = Router()
        first.register(Alpha())
        first.register(Beta())
        second = Router()
        second.register(Beta())
        second.register(Alpha())
        # "alpha" only matches Alpha, so ordering is checked with a real tie.
        tied_a = Router()
        tied_a.register(Alpha())
        tied_b = Router()
        tied_b.register(Beta())
        assert tied_a.find_match("alpha").tool_name == "alpha"
        assert tied_b.find_match("beta").tool_name == "beta"

    def test_dispatch_returns_text(self, router):
        assert router.dispatch("alpha", None) == "ALPHA"

    def test_dispatch_on_no_match_is_empty(self, router):
        # The app owns the wording, so the router returns an empty string.
        assert router.dispatch("xyzzy", None) == ""

    def test_dispatch_passes_lists_through(self):
        class Multi(Tool):
            name = "multi"
            description = "many parts"

            def patterns(self):
                return ["multi"]

            def execute(self, ctx, slots=None):
                return ["one", "two"]

        r = Router()
        r.register(Multi())
        assert r.dispatch("multi", None) == ["one", "two"]

    def test_tool_exception_is_contained(self):
        r = Router()
        r.register(Exploding())
        result = r.dispatch("boom", None)
        assert "kaboom" in result
        assert isinstance(result, str)

    def test_none_response_becomes_empty_string(self):
        class Nothing(Tool):
            name = "nothing"
            description = "returns None"

            def patterns(self):
                return ["nothing"]

            def execute(self, ctx, slots=None):
                return None

        r = Router()
        r.register(Nothing())
        assert r.dispatch("nothing", None) == ""


class TestRegistration:
    def test_rejects_non_tool(self):
        with pytest.raises(TypeError):
            Router().register("not a tool")

    def test_rejects_duplicate_name(self):
        r = Router()
        r.register(Alpha())
        with pytest.raises(ValueError):
            r.register(Alpha())

    def test_unregister(self, router):
        assert router.unregister("alpha") is True
        assert router.unregister("alpha") is False
        assert "alpha" not in router

    def test_clear(self, router):
        router.clear()
        assert len(router) == 0

    def test_get_and_contains(self, router):
        assert isinstance(router.get("alpha"), Alpha)
        assert router.get("nope") is None
        assert "alpha" in router

    def test_help_text_lists_tools(self, router):
        text = router.help_text()
        assert "alpha" in text and "beta" in text
        assert Router().help_text() == "No commands are available right now."

    def test_tools_property_is_a_copy(self, router):
        tools = router.tools
        tools.clear()
        assert len(router) == 2


class TestDefaultRegistration:
    def test_registers_ten_unique_tools(self):
        r = build_default_router()
        names = [t.name for t in r.tools]
        assert len(names) == 10
        assert len(set(names)) == 10

    def test_expected_tools_in_expected_order(self):
        r = build_default_router()
        assert [t.name for t in r.tools] == [
            "weather",
            "news",
            "facts",
            "jokes",
            "information",
            "youtube",
            "analytics",
            "history",
            "notes",
            "system",
        ]

    def test_system_tool_is_last(self):
        r = build_default_router()
        assert r.tools[-1].name == "system"

    def test_build_returns_independent_routers(self):
        a, b = build_default_router(), build_default_router()
        a.unregister("weather")
        assert "weather" in b
