"""Tests for the NLU lexicon layer.

Everything here uses invented intents. No real assistant tool is
instantiated and ``assistant.tools`` is never imported, which is exactly the
independence the lexicon promises.
"""

from __future__ import annotations

import re

import pytest

from assistant.nlu.lexicon import (
    COMMON_ALIASES,
    Lexicon,
    LexiconEntry,
    build_lexicon,
)

# --- Fake vocabulary, deliberately shaped like the real commands -----------
FAKE_ENTRIES = [
    ("weather", ("weather", "temperature")),
    ("news", ("news", "headlines")),
    ("jokes", ("joke", "jokes")),
    ("youtube", ("play", "youtube")),
    ("notes", ("note",)),
]


@pytest.fixture
def lexicon() -> Lexicon:
    return build_lexicon(FAKE_ENTRIES)


class TestConstruction:
    def test_builds_one_entry_per_pattern(self, lexicon):
        # 2 + 2 + 2 + 2 + 1 patterns
        assert len(lexicon) == 9

    def test_entry_fields(self, lexicon):
        entry = lexicon.entries[0]
        assert isinstance(entry, LexiconEntry)
        assert entry.intent and entry.pattern
        assert isinstance(entry.regex, re.Pattern)

    def test_multiple_patterns_for_one_intent(self, lexicon):
        patterns = {e.pattern for e in lexicon.entries_for("weather")}
        assert patterns == {"weather", "temperature"}

    def test_repeated_pairs_merge_into_one_intent(self):
        lex = build_lexicon([("news", ("news",)), ("news", ("headlines",))])
        assert lex.intents == ("news",)
        assert len(lex.entries_for("news")) == 2

    def test_empty_input(self):
        assert len(build_lexicon([])) == 0

    def test_empty_lexicon_matches_nothing(self):
        assert build_lexicon([]).exact_matches("news") == ()

    def test_intent_names_are_normalised(self):
        assert build_lexicon([("  NEWS  ", ("news",))]).intents == ("news",)

    def test_patterns_are_normalised(self):
        lex = build_lexicon([("news", ("  HeadLines ",))])
        assert lex.entries[0].pattern == "headlines"

    def test_blank_intent_is_skipped(self):
        assert len(build_lexicon([("", ("news",))])) == 0


class TestRegexCompilation:
    def test_patterns_are_compiled(self, lexicon):
        for entry in lexicon:
            assert entry.regex.pattern == rf"(?<!\w){re.escape(entry.pattern)}(?!\w)"

    def test_pattern_ending_in_punctuation_still_matches(self):
        # A plain \b would break here: there is no word boundary after "+".
        lex = build_lexicon([("code", ("c++",))])
        assert lex.exact_matches("i need c++")
        assert lex.exact_matches("c++")
        # The guarantee that still holds: a neighbouring word character
        # blocks the match on either side.
        assert lex.exact_matches("ac++") == ()
        assert lex.exact_matches("c++x") == ()

    def test_metacharacters_are_escaped(self):
        # A raw "." would otherwise match any character.
        lex = build_lexicon([("weird", ("c++", "a.b"))])
        assert lex.exact_matches("c++")
        assert lex.exact_matches("a.b")
        # "axb" must not match the pattern "a.b".
        assert lex.exact_matches("axb") == ()

    def test_regex_operators_in_data_are_inert(self):
        lex = build_lexicon([("odd", ("x|y", "(a)", "[bc]"))])
        assert lex.exact_matches("x|y")
        assert lex.exact_matches("(a)")
        assert lex.exact_matches("[bc]")
        assert lex.exact_matches("y") == ()


class TestWordBoundaries:
    @pytest.mark.parametrize("text", ["play music", "play lofi", "play"])
    def test_play_matches_real_commands(self, lexicon, text):
        assert "youtube" in lexicon.matching_intents(text)

    @pytest.mark.parametrize("text", ["replay", "playback", "display"])
    def test_play_does_not_match_embedded_words(self, lexicon, text):
        assert "youtube" not in lexicon.matching_intents(text)

    @pytest.mark.parametrize("text", ["noted", "notes are good", "denote"])
    def test_note_does_not_match_noted(self, lexicon, text):
        assert "notes" not in lexicon.matching_intents(text)

    def test_note_matches_whole_word(self, lexicon):
        assert "notes" in lexicon.matching_intents("note buy milk")

    def test_empty_text_matches_nothing(self, lexicon):
        assert lexicon.exact_matches("") == ()


class TestMultipleIntents:
    def test_several_intents_can_match_one_utterance(self, lexicon):
        found = lexicon.matching_intents("what is the news and the weather")
        assert "news" in found and "weather" in found

    def test_exact_matches_returns_every_entry(self, lexicon):
        matches = lexicon.exact_matches("news headlines")
        assert {m.intent for m in matches} == {"news"}
        assert len(matches) == 2

    def test_matching_intents_are_deduplicated(self, lexicon):
        found = lexicon.matching_intents("news headlines news")
        assert found.count("news") == 1

    def test_same_pattern_under_two_intents_is_kept(self):
        lex = build_lexicon([("a", ("shared",)), ("b", ("shared",))])
        assert {e.intent for e in lex.exact_matches("shared")} == {"a", "b"}


class TestDeterminism:
    def test_input_order_does_not_matter(self):
        forward = build_lexicon(FAKE_ENTRIES)
        backward = build_lexicon(list(reversed(FAKE_ENTRIES)))
        assert forward.entries == backward.entries

    def test_canonical_sort_order(self, lexicon):
        keys = [(e.intent, e.pattern) for e in lexicon]
        assert keys == sorted(keys)

    def test_repeated_builds_are_identical(self, lexicon):
        again = build_lexicon(FAKE_ENTRIES)
        assert [e.intent for e in lexicon] == [e.intent for e in again]
        assert [e.pattern for e in lexicon] == [e.pattern for e in again]

    def test_match_order_is_canonical(self, lexicon):
        intents = lexicon.matching_intents("news and weather and note")
        assert intents == ("news", "notes", "weather")


class TestDuplicatesAndBlanks:
    def test_duplicate_pattern_in_same_intent_collapses(self):
        assert len(build_lexicon([("news", ("news", "news", "NEWS"))])) == 1

    def test_duplicate_across_repeated_pairs_collapses(self):
        assert len(build_lexicon([("news", ("news",)), ("news", ("news",))])) == 1

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_patterns_are_dropped(self, blank):
        assert len(build_lexicon([("news", (blank,))])) == 0

    def test_blank_pattern_does_not_match_everything(self):
        # An empty pattern must never become a catch-all matcher.
        lex = build_lexicon([("news", ("", "news"))])
        assert len(lex) == 1
        assert lex.exact_matches("weather today") == ()


class TestAliases:
    def test_aliases_are_merged(self):
        lex = build_lexicon([("news", ("news",))], aliases={"news": ("headlines",)})
        assert {e.pattern for e in lex.entries} == {"news", "headlines"}

    def test_aliases_can_introduce_an_intent(self):
        lex = build_lexicon([("news", ("news",))], aliases={"weather": ("forecast",)})
        assert "weather" in lex

    def test_alias_for_intent_with_no_entries(self):
        lex = build_lexicon([], aliases={"news": ("news",)})
        assert lex.intents == ("news",)

    def test_common_aliases_are_well_formed(self):
        for intent, patterns in COMMON_ALIASES.items():
            assert isinstance(intent, str) and intent
            assert patterns and all(p.strip() for p in patterns)

    def test_common_aliases_build_cleanly(self):
        lex = build_lexicon(FAKE_ENTRIES, aliases=COMMON_ALIASES)
        assert len(lex) >= len(build_lexicon(FAKE_ENTRIES))
        # Every alias must compile, never be dropped as malformed.
        for entry in lex:
            assert entry.regex.pattern.endswith(r"(?!\w)")


class TestContainerBehaviour:
    def test_iteration(self, lexicon):
        assert len(list(lexicon)) == 9

    def test_contains_intent(self, lexicon):
        assert "news" in lexicon
        assert "nope" not in lexicon

    def test_entries_for_unknown_intent_is_empty(self, lexicon):
        assert lexicon.entries_for("nope") == ()

    def test_entry_equality_ignores_regex_identity(self):
        a = build_lexicon([("news", ("news",))]).entries[0]
        b = build_lexicon([("news", ("news",))]).entries[0]
        assert a == b  # separately compiled regexes still compare equal

    def test_lexicon_is_frozen(self, lexicon):
        with pytest.raises(Exception):
            lexicon.entries = ()

    def test_entry_is_frozen(self, lexicon):
        with pytest.raises(Exception):
            lexicon.entries[0].pattern = "other"


class TestIndependence:
    def test_no_heavy_modules_imported(self):
        import sys

        forbidden = {
            "requests", "pyttsx3", "speech_recognition",
            "pyaudio", "selenium", "randfacts",
        }
        assert not ({m.split(".")[0] for m in sys.modules} & forbidden)

    def test_module_imports_nothing_from_the_assistant(self):
        # Parsed with ast rather than grepped, so prose in the docstring
        # that merely mentions another package is not a false positive.
        import ast
        import pathlib

        import assistant.nlu.lexicon as module

        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        outside = {
            name
            for name in imported
            if name.startswith("assistant") and not name.startswith("assistant.nlu")
        }
        assert outside == set(), f"lexicon must not import {outside}"

    def test_works_with_arbitrary_fake_intents(self):
        lex = build_lexicon(
            [("teapot", ("brew", "pour")), ("umbrella", ("rain", "forecast"))]
        )
        assert lex.matching_intents("please brew some tea") == ("teapot",)
        assert lex.matching_intents("take the umbrella it will rain") == (
            "umbrella",
        )
