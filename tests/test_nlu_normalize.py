"""Tests for the normalisation stage of the NLU pipeline.

These cover only :mod:`assistant.nlu.normalize`. Scoring, fuzzy matching,
entities and router integration are later steps and are not tested here.
"""

from __future__ import annotations

import dataclasses

import pytest

from assistant.nlu import NUMBER_WORDS, Normalized, normalize


class TestCaseFolding:
    def test_lowercases_ascii(self):
        assert normalize("NEWS").text == "news"

    def test_casefold_beats_lower_for_sharp_s(self):
        # lower() leaves the German sharp s alone; casefold() maps it to ss.
        assert "straße".lower() != "strasse"
        assert normalize("STRASSE").text == normalize("straße").text == "strasse"

    def test_mixed_case_is_uniform(self):
        assert normalize("Tell Me The NEWS").text == "tell me the news"


class TestUnicode:
    def test_nfkc_folds_fullwidth(self):
        assert normalize("ＮＥＷＳ").text == "news"

    def test_nfkc_folds_ligature(self):
        assert normalize("ﬁle").text == "file"

    def test_accents_are_preserved(self):
        # Stripping accents would lose meaning, so they are kept.
        assert normalize("café").text == "café"

    def test_smart_apostrophe_is_folded(self):
        assert normalize("what’s").text == "what is"


class TestWhitespace:
    def test_strips_surrounding(self):
        assert normalize("   news   ").text == "news"

    def test_collapses_internal_runs(self):
        assert normalize("tell    me\tthe\nnews").text == "tell me the news"

    def test_all_whitespace_is_empty(self):
        assert normalize("     ").text == ""


class TestPunctuation:
    def test_commas_and_dots_become_spaces(self):
        assert normalize("tell me, the news.").text == "tell me the news"

    def test_question_mark_is_separator(self):
        assert normalize("what is the weather?").text == "what is the weather"

    def test_symbols_are_separators(self):
        assert normalize("play + one").text == "play 1"

    def test_hyphen_splits_into_two_words(self):
        assert normalize("top-headlines").text == "top headlines"

    def test_apostrophe_is_not_a_separator(self):
        # "don't" must stay one word until contractions are expanded.
        assert normalize("don't").tokens == ("do", "not")


class TestContractions:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("don't", "do not"),
            ("Don't", "do not"),
            ("what's", "what is"),
            ("What's", "what is"),
            ("i'm", "i am"),
            ("I'm", "i am"),
            ("it's", "it is"),
            ("that's", "that is"),
            ("let's", "let us"),
            ("isn't", "is not"),
        ],
    )
    def test_expansion(self, raw, expected):
        assert normalize(raw).text == expected

    def test_unknown_contraction_splits(self):
        # Not in the table, but the apostrophe still separates the word, so
        # no token is ever left holding one.
        assert normalize("ain't").tokens == ("ain", "t")

    def test_no_token_ever_keeps_an_apostrophe(self):
        for raw in ["don't", "ain't", "rock'n'roll", "y'all"]:
            assert "'" not in "".join(normalize(raw).tokens)

    def test_contraction_inside_sentence(self):
        assert normalize("don't play that").text == "do not play that"

    def test_partial_word_is_not_expanded(self):
        assert normalize("contraction").tokens == ("contraction",)


class TestNumberWords:
    @pytest.mark.parametrize(
        "word,digit",
        [
            ("one", 1),
            ("two", 2),
            ("three", 3),
            ("five", 5),
            ("ten", 10),
            ("fifteen", 15),
            ("twenty", 20),
        ],
    )
    def test_conversion(self, word, digit):
        result = normalize(f"delete note {word}")
        assert result.tokens[-1] == str(digit)
        assert result.numbers == (digit,)

    @pytest.mark.parametrize("index", list(range(1, 21)))
    def test_full_range_one_to_twenty(self, index):
        # Guards that every number word in the documented range converts.
        word = next(w for w, d in NUMBER_WORDS.items() if d == index)
        assert normalize(f"note {word}").numbers == (index,)

    def test_number_word_table_is_complete(self):
        assert set(NUMBER_WORDS.values()) == set(range(1, 21))

    def test_existing_digits_are_kept(self):
        assert normalize("delete note 7").numbers == (7,)

    def test_mixed_words_and_digits(self):
        result = normalize("note three about 2026")
        assert result.tokens == ("note", "3", "about", "2026")
        assert result.numbers == (3, 2026)

    def test_number_word_inside_a_word_is_ignored(self):
        assert normalize("someone").tokens == ("someone",)
        assert normalize("no one").tokens == ("no", "1")

    def test_numbers_follow_token_order(self):
        assert normalize("one two three").numbers == (1, 2, 3)


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize("") == Normalized(text="", tokens=(), numbers=())

    def test_none_is_treated_as_empty(self):
        assert normalize(None).text == ""

    def test_punctuation_only_input(self):
        assert normalize("!!! ???").text == ""

    @pytest.mark.parametrize(
        "raw",
        [
            "Tell me the news!",
            "delete note two",
            "what's the wether?",
            "play   lofi   beats",
        ],
    )
    def test_text_always_equals_joined_tokens(self, raw):
        result = normalize(raw)
        assert result.text == " ".join(result.tokens)

    def test_tokens_and_numbers_are_tuples(self):
        result = normalize("note two")
        assert isinstance(result.tokens, tuple)
        assert isinstance(result.numbers, tuple)

    def test_deterministic(self):
        raw = "What's the weather in Delhi? delete note two"
        first = normalize(raw)
        for _ in range(5):
            assert normalize(raw) == first

    def test_repeated_separators_do_not_create_empty_tokens(self):
        assert normalize("a -- b ,, c").tokens == ("a", "b", "c")


class TestNormalizedDataclass:
    def test_is_frozen(self):
        result = normalize("news")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.text = "other"

    def test_equality_works(self):
        assert normalize("news") == normalize("news")

    def test_hashable(self):
        assert hash(normalize("news")) == hash(normalize("news"))

    def test_defaults_are_empty(self):
        result = Normalized(text="x")
        assert result.tokens == ()
        assert result.numbers == ()


class TestRealUtterances:
    """Cases taken from how people actually phrase voice commands."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Tell me the news!", "tell me the news"),
            ("What's the weather like?", "what is the weather like"),
            ("Delete note two please", "delete note 2 please"),
            ("Play   lofi   beats", "play lofi beats"),
            ("I want a joke", "i want a joke"),
            ("What did I ask?", "what did i ask"),
        ],
    )
    def test_examples(self, raw, expected):
        assert normalize(raw).text == expected
