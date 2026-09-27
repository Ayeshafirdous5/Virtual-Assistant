"""Backward compatibility between the NLU and the legacy router.

The promise this file protects is narrow and important: **integrating the
NLU must not break a command that already worked.** Every entry in
:data:`CORPUS` is a clean, intended command for one of the nine tools. For
each one the legacy router and the NLU must name the same tool.

The corpus holds only unambiguous commands. Semantically tangled sentences
belong in ``test_nlu_integration.py``, which asserts what the assistant
should do with them rather than what the old router happened to do.
"""

from __future__ import annotations

import pytest

from assistant.app import build_runtime_lexicon, resolve
from assistant.core.router import NoMatch
from assistant.tools import build_default_router

#: Every tool, with the plain commands a user would actually say. The nine
#: keys are the complete registered tool list, so this doubles as a check
#: that no tool has been left out of the integration.
CORPUS: dict[str, list[str]] = {
    "weather": ["weather", "what is the weather", "weather in Hyderabad"],
    "news": ["news", "tell me the news", "top headlines"],
    "facts": ["fact", "tell me a fact"],
    "jokes": ["joke", "tell me a joke"],
    "information": ["information", "information about python"],
    "youtube": ["play lofi beats", "play a song", "play music", "play a video"],
    "history": ["history", "what did I ask"],
    "notes": ["note buy milk", "notes", "list notes", "show me my notes"],
    "system": ["exit", "quit", "goodbye"],
}


@pytest.fixture
def router():
    return build_default_router()


@pytest.fixture
def lexicon(router):
    return build_runtime_lexicon(router)


def _legacy_name(router, utterance):
    match = router.find_match(utterance)
    return "" if isinstance(match, NoMatch) else match.tool_name


class TestCorpusCoversEveryTool:
    def test_all_nine_tools_are_represented(self):
        assert len(CORPUS) == 9
        assert set(CORPUS) == {t.name for t in build_default_router().tools}

    def test_corpus_is_not_empty(self):
        assert all(CORPUS.values())


class TestLegacyAndNluAgree:
    @pytest.mark.parametrize(
        "expected,utterance",
        [(tool, text) for tool, texts in CORPUS.items() for text in texts],
    )
    def test_same_tool(self, router, lexicon, ctx, expected, utterance):
        assert _legacy_name(router, utterance) == expected
        tool = resolve(ctx, router, utterance, lexicon)
        assert tool is not None, f"the NLU dropped {utterance!r}"
        assert tool.name == expected

    @pytest.mark.parametrize(
        "utterance", [text for texts in CORPUS.values() for text in texts]
    )
    def test_legacy_router_still_works_on_its_own(self, router, utterance):
        # The fallback must remain genuinely reachable, not merely present.
        assert _legacy_name(router, utterance) != ""


class TestRuntimeLexicon:
    def test_is_built_from_the_registered_tools(self, router, lexicon):
        registered = {t.name for t in router.tools}
        assert set(lexicon.intents) == registered

    def test_every_tool_pattern_is_present(self, router, lexicon):
        # A subset check, not equality: the lexicon also carries the
        # approved aliases, which are additional vocabulary.
        for tool in router.tools:
            patterns = {e.pattern for e in lexicon.entries_for(tool.name)}
            assert set(tool.patterns()) <= patterns, tool.name

    def test_aliases_are_added_on_top(self, router, lexicon):
        from assistant.nlu import COMMON_ALIASES

        for intent, aliases in COMMON_ALIASES.items():
            patterns = {e.pattern for e in lexicon.entries_for(intent)}
            assert set(aliases) <= patterns, intent

    def test_covers_every_tool_even_without_aliases(self, router):
        from assistant.nlu import build_lexicon

        bare = build_lexicon(
            [(t.name, tuple(t.patterns())) for t in router.tools]
        )
        assert set(bare.intents) == {t.name for t in router.tools}

    def test_is_deterministic(self, router):
        first = build_runtime_lexicon(router)
        second = build_runtime_lexicon(router)
        assert [e.pattern for e in first] == [e.pattern for e in second]

    def test_nlu_package_never_imports_a_tool(self):
        # The dependency arrow must point one way: the app knows about
        # tools, the NLU only ever sees plain data. Parsed with ast rather
        # than grepped, so prose in a docstring that merely names another
        # package is not mistaken for an import.
        import ast
        import pathlib

        import assistant.nlu as nlu

        forbidden = ("assistant.tools", "assistant.core", "assistant.app")
        for name in nlu.__all__:
            assert "tool" not in name

        root = pathlib.Path(nlu.__file__).parent
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for imported in names:
                    assert not imported.startswith(forbidden), path.name
