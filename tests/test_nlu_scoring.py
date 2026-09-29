"""Tests for the NLU scoring stage.

All vocabulary is invented. No real assistant tool is instantiated and
``assistant.tools`` is never imported, which is the independence the
scoring layer promises.
"""

from __future__ import annotations

import pytest

from assistant.nlu.lexicon import build_lexicon
from assistant.nlu.normalize import normalize
from assistant.nlu.scoring import (
    CONTAINMENT_SCORE,
    DEFINITION_MARKERS,
    EXACT_SCORE,
    PHRASE_SCORE,
    SELF_OTHER_SUBJECTS,
    SYSTEM_MIN_SCORE,
    SYSTEM_TAIL_TOKENS,
    Candidate,
    is_inflected_form,
    is_negated,
    rank_candidates,
    score_intents,
)

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

# A vocabulary where the phrase is the only way to reach the intent, so the
# phrase stage is actually exercised rather than shadowed by a single word.
PHRASE_ENTRIES = [
    ("news", ("top headlines",)),
    ("history", ("what did i ask",)),
]


@pytest.fixture
def lexicon():
    return build_lexicon(FAKE_ENTRIES)


@pytest.fixture
def phrase_lexicon():
    return build_lexicon(PHRASE_ENTRIES)


def top(raw, lex):
    """Return the winning candidate's intent, or None."""
    found = score_intents(normalize(raw), lex)
    return found[0].intent if found else None


def find(raw, lex, intent):
    for candidate in score_intents(normalize(raw), lex):
        if candidate.intent == intent:
            return candidate
    return None


def pathlib_root():
    """Project root, so the subprocess imports the package under test."""
    import pathlib

    return pathlib.Path(__file__).resolve().parent.parent


class TestExact:
    def test_single_word_trigger_scores_one(self, lexicon):
        cand = find("what is the weather", lexicon, "weather")
        assert cand is not None
        assert cand.score == EXACT_SCORE
        assert cand.method == "exact"

    def test_trigger_is_reported(self, lexicon):
        assert find("tell me a joke", lexicon, "jokes").trigger == "joke"

    def test_several_intents_can_be_candidates(self, lexicon):
        found = score_intents(normalize("news and the weather"), lexicon)
        assert {c.intent for c in found} == {"news", "weather"}

    @pytest.mark.parametrize(
        "raw,forbidden",
        [
            ("replay my song", "youtube"),
            ("playback speed", "youtube"),
            ("i noted that down", "notes"),
            ("denote this", "notes"),
        ],
    )
    def test_no_substring_false_positives(self, lexicon, raw, forbidden):
        # The exact and phrase stages must never fire on an embedded word.
        # This is the guarantee the word-boundary lexicon exists to provide.
        for candidate in score_intents(normalize(raw), lexicon):
            if candidate.intent == forbidden:
                assert candidate.method not in ("exact", "phrase")

    def test_replay_does_not_reach_youtube_by_fuzzy(self, lexicon):
        # "replay" is longer than "play" and so fails the fuzzy length
        # tolerance, leaving only the low-confidence legacy containment.
        for candidate in score_intents(normalize("replay my song"), lexicon):
            if candidate.intent == "youtube":
                assert candidate.method == "containment"

    def test_play_still_matches_whole_word(self, lexicon):
        assert find("play lofi beats", lexicon, "youtube") is not None


class TestPhrases:
    def test_phrase_scores_ninety_five(self, phrase_lexicon):
        # "top headlines" sits at the end here, so no leading bonus applies.
        cand = find("please top headlines", phrase_lexicon, "news")
        assert cand.method == "phrase"
        assert cand.score == PHRASE_SCORE

    def test_multi_word_history_phrase(self, phrase_lexicon):
        assert find("what did i ask earlier", phrase_lexicon, "history") is not None

    def test_exact_takes_precedence_over_phrase(self, lexicon):
        # "headlines" is a single-word trigger, so exact wins over the
        # "top headlines" phrase in the same utterance.
        cand = find("give me top headlines", lexicon, "news")
        assert cand.method == "exact"
        assert cand.score == EXACT_SCORE

    def test_leading_phrase_receives_bonus(self, phrase_lexicon):
        leading = find("top headlines now", phrase_lexicon, "news")
        assert leading.method == "phrase"
        assert leading.score == pytest.approx(1.0)

    def test_trailing_phrase_gets_no_bonus(self, phrase_lexicon):
        trailing = find("please top headlines", phrase_lexicon, "news")
        assert trailing.score == PHRASE_SCORE

    def test_score_never_exceeds_one(self, lexicon):
        for raw in ["news", "top headlines", "weather", "what did i ask"]:
            for candidate in score_intents(normalize(raw), lexicon):
                assert candidate.score <= 1.0


class TestFuzzy:
    def test_fuzzy_does_not_rescue_an_inflected_form(self, lexicon):
        # "noted" is "note" plus "d": the user said every letter of the
        # trigger correctly and added an ending, so this is a different
        # word, not a misspelling. Only the legacy containment stage, at
        # 0.50 and below the parser's action threshold, may still see it.
        cand = find("i noted that down", lexicon, "notes")
        assert cand is None or cand.method == "containment"
        assert cand is None or cand.score < 0.65

    def test_note_still_matches_whole_word(self, lexicon):
        assert find("note buy milk", lexicon, "notes").method == "exact"

    @pytest.mark.parametrize(
        "raw,forbidden",
        [
            ("i noted that down", "notes"),
            ("play the song", "youtube"),
            ("my notes are here", "notes"),
        ],
    )
    def test_inflections_never_reach_the_fuzzy_stage(self, lexicon, raw, forbidden):
        for candidate in score_intents(normalize(raw), lexicon):
            if candidate.intent == forbidden:
                assert candidate.method != "fuzzy"

    @pytest.mark.parametrize(
        "token,pattern",
        [
            ("noted", "note"),
            ("notes", "note"),
            ("played", "play"),
            ("plays", "play"),
            ("playing", "play"),
        ],
    )
    def test_is_inflected_form_helper(self, token, pattern):
        assert is_inflected_form(token, pattern) is True

    @pytest.mark.parametrize(
        "token,pattern",
        [
            ("wether", "weather"),         # substitution
            ("joks", "jokes"),             # deletion at the end
            ("umbrela", "umbrella"),       # deletion
            ("temprature", "temperature"),  # transposition
            ("histry", "history"),
            ("news", "news"),              # identical
        ],
    )
    def test_typos_are_not_inflected_forms(self, token, pattern):
        assert is_inflected_form(token, pattern) is False

    def test_long_extension_is_not_treated_as_inflection(self):
        # "weather" plus a four character tail is outside the cap, so the
        # rule does not silently discard longer words; they simply fall
        # below the fuzzy cutoff on their own.
        assert is_inflected_form("weatherzzzz", "weather") is False
        assert is_inflected_form("weathersxx", "weather") is True

    def test_plural_of_the_same_intent_cannot_bypass_the_guard(self):
        # The real notes tool carries both "note" and "notes". "noted"
        # extends "note" so that pair is refused, but it is only one letter
        # from the plural and fuzzy-matches "notes" on its own. The guard is
        # therefore per intent, not per word pair.
        multi = build_lexicon(
            [("notes", ("note", "notes")), ("weather", ("weather",))]
        )
        patterns = {e.pattern for e in multi.entries_for("notes")}
        assert {"note", "notes"} <= patterns
        # The plural alone is NOT an extension, so a pair level guard misses.
        assert is_inflected_form("noted", "notes") is False
        cand = find("i noted that down", multi, "notes")
        assert cand is None or cand.method != "fuzzy"
        # The exact command still works through either form.
        assert find("note buy milk", multi, "notes").method == "exact"
        assert find("notes buy milk", multi, "notes").method == "exact"

    def test_guard_does_not_leak_across_intents(self, lexicon):
        # Blocking "notes" for "noted" must not stop another intent from
        # legitimately recovering a typo from the same token.
        for candidate in score_intents(normalize("wether"), lexicon):
            if candidate.intent == "weather":
                assert candidate.method == "fuzzy"

    def test_wether_recovers_weather(self, lexicon):
        cand = find("wether", lexicon, "weather")
        assert cand is not None
        assert cand.method == "fuzzy"

    def test_joks_recovers_jokes(self, lexicon):
        cand = find("joks", lexicon, "jokes")
        assert cand is not None
        assert cand.method == "fuzzy"

    def test_nwes_is_rejected_honestly(self, lexicon):
        # SequenceMatcher gives nwes/news a ratio of 0.75. The length penalty
        # cannot lift it, so it lands under the 0.80 cutoff and is dropped.
        # This is recorded rather than massaged: shortening the threshold to
        # make one example pass would let far worse matches through.
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, "nwes", "news").ratio()
        assert ratio < 0.80
        assert find("nwes", lexicon, "news") is None

    def test_short_tokens_are_ignored(self, lexicon):
        # "wes" is 3 characters, below FUZZY_MIN_TOKEN, so it is not scored.
        assert find("wes", lexicon, "weather") is None

    def test_below_cutoff_is_rejected(self, lexicon):
        assert find("zzzzzzzzzzzz", lexicon, "weather") is None

    def test_length_penalty_softens_short_tokens(self, lexicon):
        lex = build_lexicon([("umbrella", ("umbrella",))])
        cand = find("umbrela", lex, "umbrella")
        assert cand is not None
        assert cand.method == "fuzzy"
        assert cand.score < 1.0

    def test_one_winner_per_intent(self, lexicon):
        found = score_intents(normalize("wether tempreature"), lexicon)
        assert [c.intent for c in found].count("weather") == 1

    def test_exact_beats_fuzzy(self, lexicon):
        # The real word is present, so the exact candidate must win.
        assert find("what is the weather", lexicon, "weather").method == "exact"

    def test_repeated_calls_are_identical(self, lexicon):
        raw = "wether and joks"
        first = score_intents(normalize(raw), lexicon)
        for _ in range(5):
            assert score_intents(normalize(raw), lexicon) == first


class TestNegation:
    @pytest.mark.parametrize(
        "raw", ["do not play music", "don't play anything", "never play", "stop"]
    )
    def test_negated_trigger_is_dropped(self, lexicon, raw):
        assert find(raw, lexicon, "youtube") is None

    def test_stop_playing_does_not_play(self, lexicon):
        assert top("stop playing that", lexicon) is None

    def test_unrelated_negation_does_not_block(self, lexicon):
        # "no" belongs to an earlier clause, so the news request stands.
        assert find("i have no idea, tell me the news", lexicon, "news") is not None

    def test_window_is_local_not_sentence_wide(self, lexicon):
        # Documented limitation. "not" sits five tokens before the trigger,
        # outside the window, so the weather request still surfaces. Widening
        # the window to catch it would break the previous test, so the trade
        # is accepted and recorded; a real fix belongs in the parser.
        assert find("do not ask me about the weather", lexicon, "weather") is not None

    def test_is_negated_helper_directly(self):
        assert is_negated("do not play music", 2) is True
        assert is_negated("play music", 0) is False
        assert is_negated("i have no idea and tell me the news", 6) is False

    def test_negation_stops_at_clause_break(self):
        # "no" is before "but", so it does not reach the trigger after it.
        assert is_negated("no thanks but tell me the news", 6) is False


class TestSystemSafety:
    def test_plain_exit_is_valid(self, lexicon):
        assert find("exit", lexicon, "system") is not None

    def test_trailing_goodbye_is_valid(self, lexicon):
        assert find("ok goodbye", lexicon, "system") is not None

    def test_goodbye_inside_a_sentence_is_rejected(self, lexicon):
        assert find("goodbye is in the dictionary", lexicon, "system") is None

    def test_containment_system_candidate_is_rejected(self, lexicon):
        # "exit" sits inside "exiting", so only containment could fire.
        for candidate in score_intents(normalize("he is exiting now"), lexicon):
            if candidate.intent == "system":
                assert candidate.score >= 0.95 and candidate.method == "containment"

    def test_exit_among_other_words_is_valid(self, lexicon):
        # "exit" is a genuine command and sits in the closing words, so it
        # must still work even when the utterance is longer.
        assert find("ok pls exit now", lexicon, "system") is not None

    def test_system_scores_highest_bar(self, lexicon):
        for candidate in score_intents(normalize("exit"), lexicon):
            if candidate.intent == "system":
                assert candidate.score >= 0.95

    def test_other_intents_are_unaffected(self, lexicon):
        assert find("quit", lexicon, "system").score == EXACT_SCORE


class TestContainment:
    def test_legacy_substring_still_matches_as_last_resort(self, lexicon):
        # "display" contains "play", so only the legacy stage can catch it.
        found = score_intents(normalize("display"), lexicon)
        youtube = [c for c in found if c.intent == "youtube"]
        assert youtube and youtube[0].method == "containment"
        assert youtube[0].score == CONTAINMENT_SCORE

    def test_containment_never_beats_exact(self, lexicon):
        for candidate in score_intents(normalize("play the weather news"), lexicon):
            if candidate.method == "containment":
                assert candidate.score == CONTAINMENT_SCORE


# ----------------------------------------------------------------------
# Definition and explanation apposition (Step 20)
# ----------------------------------------------------------------------
#: The six sentences that used to terminate the application. Each puts
#: the trigger in apposition: the word is being named, not acted on.
DEFINITION_FPS = (
    "what does quit mean",
    "what does exit mean",
    "what does goodbye mean",
    "what is the meaning of quit",
    "define exit",
    "explain quit",
)

#: Genuine orders, which must all keep working.
GENUINE_SYSTEM = (
    "exit",
    "quit",
    "goodbye",
    "ok goodbye",
    "ok pls exit now",
    "goodbye assistant",
)

#: The seven Step 17 dangers, all of which must stay blocked.
STEP_17_FPS = (
    "I should quit smoking",
    "I want to quit smoking",
    "she decided to quit smoking",
    "he plans to quit smoking",
    "I need to quit smoking",
    "he quit smoking",
    "I want to exit early",
)

#: Ordinary sentences that merely contain system vocabulary.
ORDINARY_SENTENCES = (
    "the exit sign is red",
    "quit your browser",
    "goodbye is in the dictionary",
    "define my own words",
    "explain the rule to me",
)

#: Deliberately **not** fixed by this step. Recorded so it cannot be
#: half-fixed by accident and forgotten. Fixing it needs a different rule
#: and is not attempted here.
STILL_OPEN_UTTERANCE = "do you want to quit"


class TestDefinitionAppositionIsRejected:
    """A trigger in apposition is a word being explained, not an order."""

    @pytest.mark.parametrize("utterance", DEFINITION_FPS)
    def test_no_system_candidate_is_produced(self, utterance, lexicon):
        assert find(utterance, lexicon, "system") is None

    @pytest.mark.parametrize(
        "utterance,marker",
        [
            ("what does quit mean", "does"),
            ("what does exit mean", "does"),
            ("what does goodbye mean", "does"),
            ("what is the meaning of quit", "of"),
            ("define exit", "define"),
            ("explain quit", "explain"),
        ],
    )
    def test_each_is_refused_by_its_own_marker(self, utterance, marker, lexicon):
        """The refusal is the marker sitting immediately before it."""
        assert marker in DEFINITION_MARKERS
        assert find(utterance, lexicon, "system") is None

    def test_the_markers_are_a_closed_set(self):
        assert DEFINITION_MARKERS == frozenset(
            {"does", "do", "of", "define", "explain"}
        )

    def test_you_was_not_added_to_the_subject_set(self):
        """A constraint of this step, asserted rather than trusted."""
        assert "you" not in SELF_OTHER_SUBJECTS
        assert "your" not in SELF_OTHER_SUBJECTS

    def test_ordinary_sentences_are_unaffected(self, lexicon):
        for utterance in ORDINARY_SENTENCES:
            assert find(utterance, lexicon, "system") is None, utterance

    def test_the_guard_is_still_scoped_to_system(self, lexicon):
        """A definition word must not become a general question rule.

        Scored with and without one, so the comparison is evidence
        rather than a claim: a real request keeps exactly its score.
        """
        without = score_intents(normalize("the weather now"), lexicon)
        with_marker = score_intents(normalize("define the weather now"), lexicon)
        plain = {c.intent: c.score for c in without}
        marked = {c.intent: c.score for c in with_marker}
        assert plain["weather"] == marked["weather"]
        assert marked["weather"] == EXACT_SCORE


class TestGenuineSystemCommandsSurvive:
    @pytest.mark.parametrize("utterance", GENUINE_SYSTEM)
    def test_still_resolves(self, utterance, lexicon):
        assert find(utterance, lexicon, "system") is not None, utterance

    @pytest.mark.parametrize("utterance", STEP_17_FPS)
    def test_the_step_seventeen_dangers_stay_blocked(self, utterance, lexicon):
        assert find(utterance, lexicon, "system") is None, utterance

    def test_no_constant_was_moved(self):
        """Scope of this step, asserted rather than described."""
        assert SYSTEM_TAIL_TOKENS == 2


class TestMultiWordTriggersAreExempt:
    """The guard must measure from the trigger's **start**.

    "shut down" is the case that proves it. Measured from the last word
    the guard would read the wrong neighbour and could refuse a genuine
    order, because "please" sits before "shut" but nothing relevant
    sits before "down".
    """

    @pytest.fixture
    def phrase_system_lexicon(self):
        return build_lexicon([("system", ("shut down", "exit", "quit"))])

    def test_a_phrase_command_still_works(self, phrase_system_lexicon):
        assert find("shut down", phrase_system_lexicon, "system") is not None
        assert find("please shut down", phrase_system_lexicon, "system") is not None

    def test_a_marker_before_a_phrase_does_not_refuse_it(self, phrase_system_lexicon):
        """The exemption is what keeps this a single-word rule."""
        assert find("define shut down", phrase_system_lexicon, "system") is not None

    def test_the_index_is_the_start_of_the_phrase(self, phrase_system_lexicon):
        """Proved on the real matcher, not on the string."""
        text = normalize("please shut down").text
        assert text.split() == ["please", "shut", "down"]
        candidate = find("please shut down", phrase_system_lexicon, "system")
        assert candidate is not None
        assert candidate.trigger == "shut down"

    def test_what_does_shut_down_mean_produces_nothing(self, phrase_system_lexicon):
        assert find("what does shut down mean", phrase_system_lexicon, "system") is None

    def test_a_single_word_trigger_beside_the_phrase_is_still_guarded(
        self, phrase_system_lexicon
    ):
        """The exemption is per trigger, not a hole in the rule."""
        assert find("what does exit mean", phrase_system_lexicon, "system") is None


class TestStillOpenDeliberately:
    """A separate safety problem, left measured rather than half-fixed."""

    def test_do_you_want_to_quit_is_not_caught(self, lexicon):
        """The token before "quit" is "to", which is not a marker."""
        tokens = normalize("do you want to quit").tokens
        assert tokens[tokens.index("quit") - 1] == "to"
        assert "to" not in DEFINITION_MARKERS
        assert find("do you want to quit", lexicon, "system") is not None

    def test_it_is_recorded_rather_than_ignored(self):
        """Named here so the next step starts from a number, not a search."""
        assert STILL_OPEN_UTTERANCE == "do you want to quit"

        assert SYSTEM_MIN_SCORE == 0.95


    def test_exact_outranks_containment_for_same_intent(self, lexicon):
        cand = find("play lofi", lexicon, "youtube")
        assert cand.method != "containment"

    def test_lower_stage_cannot_lower_a_score(self, lexicon):
        # "notes" exists, so containment would also match; exact must stand.
        cand = find("note the weather", lexicon, "notes")
        assert cand.method == "exact" and cand.score == EXACT_SCORE


class TestRanking:
    def _cands(self):
        return [
            Candidate("low", 0.4, "xx", "exact"),
            Candidate("mid", 0.6, "yy", "exact"),
            Candidate("tie1", 0.9, "yy", "exact"),
            Candidate("tie2", 0.9, "longer", "exact"),
        ]

    def test_higher_score_wins(self):
        assert rank_candidates(self._cands())[0].intent == "tie2"

    def test_lower_score_ranks_last(self):
        assert rank_candidates(self._cands())[-1].intent == "low"

    def test_longer_trigger_breaks_ties(self):
        ordered = rank_candidates(
            [Candidate("a", 0.9, "yy", "exact"), Candidate("b", 0.9, "longer", "exact")]
        )
        assert ordered[0].intent == "b"

    def test_lexical_intent_breaks_remaining_ties(self):
        ordered = rank_candidates(
            [Candidate("z", 0.9, "same", "exact"), Candidate("a", 0.9, "same", "exact")]
        )
        assert [c.intent for c in ordered] == ["a", "z"]

    def test_ranking_is_order_independent(self):
        forward = rank_candidates(self._cands())
        backward = rank_candidates(list(reversed(self._cands())))
        assert forward == backward

    def test_ranking_does_not_mutate_input(self):
        original = self._cands()
        rank_candidates(original)
        assert original[0].intent == "low"

    def test_candidate_is_frozen(self):
        with pytest.raises(Exception):
            Candidate("a", 0.5, "x", "exact").score = 0.9

    def test_candidate_equality_ignores_nothing(self):
        assert Candidate("a", 0.5, "x", "exact") == Candidate("a", 0.5, "x", "exact")

    def test_empty_input(self):
        assert rank_candidates([]) == []

    def test_results_come_back_sorted(self, lexicon):
        found = score_intents(normalize("news weather joke"), lexicon)
        scores = [c.score for c in found]
        assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    def test_empty_utterance(self, lexicon):
        assert score_intents(normalize(""), lexicon) == []

    def test_no_match_returns_empty(self, lexicon):
        assert score_intents(normalize("zzzz qqqq wwww"), lexicon) == []

    def test_result_is_a_new_list_each_call(self, lexicon):
        raw = "news"
        first = score_intents(normalize(raw), lexicon)
        first.clear()
        assert score_intents(normalize(raw), lexicon)


class TestIsolation:
    def test_no_heavy_modules_imported(self):
        # Run in a fresh interpreter. Inside pytest, conftest has already
        # imported assistant.config, which legitimately loads dotenv, so
        # checking sys.modules here would measure the harness rather than
        # this module.
        import subprocess
        import sys as _sys

        code = (
            "import sys, assistant.nlu.scoring as s;"
            "bad={m.split('.')[0] for m in sys.modules} & "
            "{'requests','pyttsx3','speech_recognition','pyaudio',"
            "'selenium','randfacts','dotenv'};"
            "print(sorted(bad))"
        )
        out = subprocess.run(
            [_sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(pathlib_root()),
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", f"unexpected imports: {out.stdout}"

    def test_module_imports_nothing_from_the_assistant_core(self):
        import ast
        import pathlib

        import assistant.nlu.scoring as module

        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        banned = {"assistant.tools", "assistant.app", "assistant.core"}
        assert not (imported & banned), f"scoring must not import {imported & banned}"

    def test_only_nlu_and_stdlib_are_imported(self):
        import ast
        import pathlib

        import assistant.nlu.scoring as module

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
        assert outside == set()

    def test_works_with_completely_invented_intents(self):
        lex = build_lexicon([("teapot", ("brew", "tea"))])
        assert top("please brew some tea", lex) == "teapot"
