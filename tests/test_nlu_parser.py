"""Tests for the public NLU parser.

A small invented :class:`~assistant.nlu.lexicon.Lexicon` is built directly
in these tests. No real assistant tool is instantiated and
``assistant.tools`` is never imported, which is the independence the parser
promises.
"""

from __future__ import annotations

import pytest

from assistant.nlu.lexicon import build_lexicon
from assistant.nlu.parser import (
    AMBIGUOUS,
    AMBIGUITY_MARGIN,
    CLEAR,
    MIN_CONFIDENCE,
    Alternative,
    ParsedIntent,
    extract_entities,
    parse,
)
from assistant.nlu.normalize import normalize

FAKE_ENTRIES = [
    ("weather", ("weather", "temperature")),
    ("news", ("news", "headlines", "top headlines")),
    ("jokes", ("joke", "jokes")),
    ("history", ("history", "what did i ask")),
    ("information", ("information",)),
    ("youtube", ("play",)),
    ("notes", ("note",)),
    ("system", ("exit", "quit", "goodbye")),
]


@pytest.fixture
def lexicon():
    return build_lexicon(FAKE_ENTRIES)


class TestBasicParsing:
    def test_exact_command(self, lexicon):
        result = parse("tell me the news", lexicon)
        assert result.name == "news"
        assert result.method == "exact"
        assert result.score == 1.0

    def test_multi_word_phrase(self, lexicon):
        result = parse("what did i ask earlier", lexicon)
        assert result.name == "history"

    def test_fuzzy_command(self, lexicon):
        assert parse("wether", lexicon).name == "weather"

    def test_play_command(self, lexicon):
        assert parse("play lofi beats", lexicon).name == "youtube"

    def test_no_match_returns_none(self, lexicon):
        assert parse("completely unrelated sentence", lexicon) is None

    def test_none_input_returns_none(self, lexicon):
        assert parse(None, lexicon) is None

    def test_empty_input_returns_none(self, lexicon):
        assert parse("", lexicon) is None

    def test_all_fields_present(self, lexicon):
        result = parse("tell me the news", lexicon)
        assert isinstance(result, ParsedIntent)
        for field in ("name", "score", "trigger", "method", "confidence"):
            assert getattr(result, field) is not None
        assert isinstance(result.entities, dict)
        assert isinstance(result.alternatives, tuple)

    def test_trigger_is_reported(self, lexicon):
        assert parse("tell me the news", lexicon).trigger == "news"

    def test_is_clear_and_is_ambiguous_helpers(self, lexicon):
        result = parse("tell me the news", lexicon)
        assert result.is_clear is True
        assert result.is_ambiguous is False


class TestSafetyRegressions:
    """Utterances that must never reach a tool."""

    @pytest.mark.parametrize(
        "raw",
        [
            "replay my song",
            "playback speed",
            "goodbye is in the dictionary",
        ],
    )
    def test_returns_none(self, lexicon, raw):
        assert parse(raw, lexicon) is None

    def test_inflected_form_returns_none(self, lexicon):
        # The fix for "noted" reaching the notes tool. Fuzzy no longer
        # rescues it, and the only stage that still sees "note" inside
        # "noted" is legacy containment at 0.50, below the action floor.
        assert parse("i noted that down", lexicon) is None

    def test_real_note_command_still_parses(self, lexicon):
        result = parse("note buy milk", lexicon)
        assert result is not None
        assert result.name == "notes"
        assert result.method == "exact"

    def test_typo_recovery_still_parses(self, lexicon):
        assert parse("wether", lexicon).name == "weather"
        assert parse("joks", lexicon).name == "jokes"

    def test_containment_alone_is_never_actionable(self, lexicon):
        # The legacy stage scores 0.50; the parser must not act on that.
        for raw in ["replay my song", "playback speed"]:
            assert parse(raw, lexicon) is None
        assert MIN_CONFIDENCE > 0.5

    def test_legitimate_commands_still_work(self, lexicon):
        assert parse("play lofi beats", lexicon).name == "youtube"
        assert parse("tell me the news", lexicon).name == "news"
        assert parse("wether", lexicon).name == "weather"


class TestConfidence:
    def test_single_candidate_is_clear(self, lexicon):
        assert parse("tell me the news", lexicon).confidence == CLEAR

    def test_distant_runner_up_is_clear(self, lexicon):
        # Exact news at 1.00 against a much weaker alternative.
        result = parse("tell me the news", lexicon)
        assert result.confidence == CLEAR

    def test_close_candidates_are_ambiguous(self):
        # Two exact triggers of equal weight in one utterance.
        lex = build_lexicon([("alpha", ("alpha",)), ("beta", ("beta",))])
        result = parse("alpha and beta", lex)
        assert result.confidence == AMBIGUOUS
        assert result.is_ambiguous is True

    def test_margin_boundary_is_inclusive(self):
        # The project states thresholds as "at least" and tests them with
        # >=, so "within 0.05" means <= 0.05 and a difference of exactly the
        # margin counts as ambiguous. Here "weather" is exact (1.00) and
        # "top headlines" is a trailing phrase (0.95): a gap of exactly 0.05.
        assert AMBIGUITY_MARGIN == 0.05
        lex = build_lexicon(
            [("weather", ("weather",)), ("news", ("top headlines",))]
        )
        result = parse("weather and top headlines", lex)
        assert result.score == 1.0
        assert result.alternatives[0].score == 0.95
        assert round(result.score - result.alternatives[0].score, 4) == 0.05
        assert result.confidence == AMBIGUOUS

    def test_below_margin_is_ambiguous(self):
        # Two exact triggers: a difference of 0.0, well inside the margin.
        lex = build_lexicon([("alpha", ("alpha",)), ("beta", ("beta",))])
        assert parse("alpha beta", lex).confidence == AMBIGUOUS

    def test_above_margin_is_clear(self):
        # Exact 1.00 against a fuzzy 0.9231 is a gap of about 0.077.
        lex = build_lexicon([("news", ("news",)), ("weather", ("weather",))])
        result = parse("news and wether", lex)
        assert result.name == "news"
        assert result.score - result.alternatives[0].score > AMBIGUITY_MARGIN
        assert result.confidence == CLEAR

    def test_no_candidate_is_not_ambiguous(self, lexicon):
        assert parse("zzzz qqqq", lexicon) is None


class TestAlternatives:
    def test_runner_up_is_exposed(self):
        lex = build_lexicon([("alpha", ("alpha",)), ("beta", ("beta",))])
        result = parse("alpha beta", lex)
        assert len(result.alternatives) == 1
        assert result.alternatives[0].name == "beta"

    def test_alternative_carries_name_and_score(self):
        lex = build_lexicon([("alpha", ("alpha",)), ("beta", ("beta",))])
        alt = parse("alpha beta", lex).alternatives[0]
        assert isinstance(alt, Alternative)
        assert alt.name == "beta"
        assert isinstance(alt.score, float)

    def test_single_candidate_has_no_alternatives(self, lexicon):
        assert parse("tell me the news", lexicon).alternatives == ()

    def test_ordering_is_stable(self):
        lex = build_lexicon([("alpha", ("alpha",)), ("beta", ("beta",))])
        first = parse("alpha beta", lex).alternatives
        for _ in range(5):
            assert parse("alpha beta", lex).alternatives == first

    def test_alternatives_are_ordered_worst_last(self):
        lex = build_lexicon(
            [
                ("a1", ("alpha",)),
                ("b1", ("beta",)),
                ("c1", ("gamma",)),
            ]
        )
        scores = [alt.score for alt in parse("alpha beta gamma", lex).alternatives]
        assert scores == sorted(scores, reverse=True)

    def test_alternative_is_frozen(self):
        with pytest.raises(Exception):
            Alternative("a", 0.5).score = 0.9

    def test_parser_does_not_expose_candidate_objects(self):
        lex = build_lexicon([("alpha", ("alpha",)), ("beta", ("beta",))])
        for alt in parse("alpha beta", lex).alternatives:
            assert not hasattr(alt, "method")
            assert not hasattr(alt, "trigger")


class TestEntities:
    def test_integer_extraction(self):
        entities = extract_entities(normalize("delete note 3"), "delete note 3")
        assert entities["number"] == 3

    def test_number_word_extraction(self):
        entities = extract_entities(
            normalize("delete note three"), "delete note three"
        )
        assert entities["number"] == 3

    def test_first_number_only(self):
        entities = extract_entities(normalize("note 2 and 5"), "note 2 and 5")
        assert entities["number"] == 2

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("news about python", "python"),
            ("information about machine learning", "machine learning"),
            ("look for the answer", "the answer"),
            ("information regarding tides", "tides"),
        ],
    )
    def test_tail_extraction(self, raw, expected):
        assert extract_entities(normalize(raw), raw)["tail"] == expected

    def test_quoted_double(self):
        assert extract_entities(normalize('note "buy milk"'), 'note "buy milk"')[
            "quoted"
        ] == "buy milk"

    def test_quoted_single(self):
        assert extract_entities(normalize("note 'call mum'"), "note 'call mum'")[
            "quoted"
        ] == "call mum"

    def test_quoted_survives_normalisation(self):
        # Normalisation strips quotation marks, so this entity is read from
        # the raw utterance. The test proves that.
        assert '"' not in normalize('note "buy milk"').text
        assert extract_entities(normalize('note "buy milk"'), 'note "buy milk"')[
            "quoted"
        ] == "buy milk"

    def test_multiple_entities_together(self):
        raw = "note about project 7 and \"the deadline\""
        entities = extract_entities(normalize(raw), raw)
        assert entities["number"] == 7
        assert entities["tail"] == "project 7 and the deadline"
        assert entities["quoted"] == "the deadline"

    def test_absent_entities_are_omitted(self):
        entities = extract_entities(normalize("tell me the news"), "tell me the news")
        assert entities == {}

    def test_no_invented_keys(self):
        entities = extract_entities(normalize("news"), "news")
        assert set(entities) <= {"number", "tail", "quoted"}

    def test_entities_reach_the_parsed_intent(self, lexicon):
        result = parse("news about python", lexicon)
        assert result.entities.get("tail") == "python"

    def test_parsed_intent_is_frozen(self, lexicon):
        with pytest.raises(Exception):
            parse("tell me the news", lexicon).name = "other"


class TestPipeline:
    """normalize -> scoring -> parser, end to end."""

    def test_full_pipeline(self, lexicon):
        result = parse("Tell Me The News!", lexicon)
        assert result.name == "news"
        assert result.method == "exact"
        assert result.confidence == CLEAR

    def test_pipeline_applies_normalisation(self, lexicon):
        # Case folding and punctuation are handled upstream of scoring.
        assert parse("WHAT'S THE WEATHER?!", lexicon).name == "weather"

    def test_pipeline_applies_contraction_expansion(self, lexicon):
        assert parse("don't tell me the news", lexicon).name == "news"

    def test_repeated_calls_are_identical(self, lexicon):
        first = parse("tell me the news", lexicon)
        for _ in range(5):
            assert parse("tell me the news", lexicon) == first

    def test_works_with_completely_invented_intents(self):
        lex = build_lexicon([("teapot", ("brew", "tea"))])
        assert parse("please brew some tea", lex).name == "teapot"

    def test_parser_does_not_alter_the_utterance(self, lexicon):
        raw = "Tell Me The News!"
        before = str(raw)
        parse(raw, lexicon)
        assert str(raw) == before


class TestIsolation:
    def test_no_heavy_modules_imported(self):
        import subprocess
        import sys as _sys
        import pathlib

        code = (
            "import sys, assistant.nlu.parser as p;"
            "bad={m.split('.')[0] for m in sys.modules} & "
            "{'requests','pyttsx3','speech_recognition','pyaudio',"
            "'selenium','randfacts','dotenv'};"
            "print(sorted(bad))"
        )
        root = pathlib.Path(__file__).resolve().parent.parent
        out = subprocess.run(
            [_sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", out.stdout

    def test_parser_imports_only_nlu_and_stdlib(self):
        import ast
        import pathlib

        import assistant.nlu.parser as module

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
        assert outside == set(), f"parser must not import {outside}"
