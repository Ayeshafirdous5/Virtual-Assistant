"""NoteTool and HistoryTool behaviour."""

from __future__ import annotations

import io

import pytest

from assistant.core.context import AppContext
from assistant.speech.base import NOTHING_HEARD, Listener, Speaker
from assistant.speech.text import TextListener, TextSpeaker
from assistant.tools.history import HistoryTool
from assistant.tools.notes import NoteTool
from assistant.tools.wikipedia import WikipediaTool
from assistant.tools.youtube import YouTubeTool

from tests.conftest import ExplodingSpeaker


@pytest.fixture
def notes_ctx(ctx, memory_db) -> AppContext:
    ctx.db = memory_db
    return ctx


class TestNoteToolSlots:
    @pytest.mark.parametrize(
        "utterance,action",
        [
            ("note buy milk", "add"),
            ("add a note test this", "add"),
            ("remember that the roof leaks", "add"),
            ("list notes", "list"),
            ("my notes", "list"),
            ("notes", "list"),
            ("note", "list"),
            ("show me my notes", "list"),
            ("what are my notes", "list"),
        ],
    )
    def test_action_detection(self, notes_ctx, utterance, action):
        assert NoteTool().extract_slots(utterance, notes_ctx)["action"] == action

    def test_add_captures_body(self, notes_ctx):
        slots = NoteTool().extract_slots("note buy milk", notes_ctx)
        assert slots["body"] == "buy milk"

    @pytest.mark.parametrize(
        "utterance,note_id",
        [("delete note 3", 3), ("remove note 7", 7)],
    )
    def test_delete_captures_id(self, notes_ctx, utterance, note_id):
        assert NoteTool().extract_slots(utterance, notes_ctx)["note_id"] == note_id

    def test_delete_without_id_still_deletes(self, notes_ctx):
        # Must not fall through and be stored as a note called "delete note".
        slots = NoteTool().extract_slots("delete note", notes_ctx)
        assert slots == {"action": "delete", "note_id": None}


class TestNoteToolExecute:
    def test_add(self, notes_ctx):
        out = NoteTool().execute(notes_ctx, {"action": "add", "body": "milk"})
        assert "Saved note" in out[0]
        assert notes_ctx.db.query_one("SELECT COUNT(*) AS n FROM notes")["n"] == 1

    def test_list_when_empty(self, notes_ctx):
        out = NoteTool().execute(notes_ctx, {"action": "list"})
        assert out == ["You have not saved any notes yet."]

    def test_list_shows_notes(self, notes_ctx):
        NoteTool().execute(notes_ctx, {"action": "add", "body": "milk"})
        out = NoteTool().execute(notes_ctx, {"action": "list"})
        assert "milk" in " ".join(out)

    def test_delete_existing(self, notes_ctx):
        NoteTool().execute(notes_ctx, {"action": "add", "body": "x"})
        note_id = notes_ctx.db.query_one("SELECT id FROM notes")["id"]
        out = NoteTool().execute(notes_ctx, {"action": "delete", "note_id": note_id})
        assert "Deleted note" in out[0]
        assert notes_ctx.db.query_one("SELECT COUNT(*) AS n FROM notes")["n"] == 0

    def test_delete_missing_id(self, notes_ctx):
        out = NoteTool().execute(notes_ctx, {"action": "delete", "note_id": 999})
        assert "could not find" in out[0]

    def test_delete_without_id_asks(self, notes_ctx):
        out = NoteTool().execute(notes_ctx, {"action": "delete", "note_id": None})
        assert "which note" in out[0]

    def test_delete_without_id_stores_nothing(self, notes_ctx):
        NoteTool().execute(notes_ctx, {"action": "delete", "note_id": None})
        assert notes_ctx.db.query_one("SELECT COUNT(*) AS n FROM notes")["n"] == 0

    def test_no_database(self, ctx):
        out = NoteTool().execute(ctx, {"action": "list"})
        assert "no place to keep notes" in out[0]

    def test_broken_database_is_graceful(self, ctx):
        class Broken:
            def query_all(self, *a, **k):
                raise RuntimeError("database is locked")

            def query_one(self, *a, **k):
                raise RuntimeError("database is locked")

            def execute(self, *a, **k):
                raise RuntimeError("database is locked")

        ctx.db = Broken()
        out = NoteTool().execute(ctx, {"action": "list"})
        assert "could not do that" in out[0]


class TestHistoryTool:
    def test_no_database(self, ctx):
        assert "no history" in HistoryTool().execute(ctx, {})[0]

    def test_empty_database(self, notes_ctx):
        assert HistoryTool().execute(notes_ctx, {}) == [
            "You have not asked me anything yet."
        ]

    def test_reports_recent_commands(self, notes_ctx):
        notes_ctx.log_command("jokes", "tell me a joke", "setup | punchline")
        joined = " ".join(HistoryTool().execute(notes_ctx, {}))
        assert "tell me a joke" in joined
        assert "jokes" in joined

    def test_respects_limit(self, notes_ctx):
        for i in range(6):
            notes_ctx.log_command("facts", f"fact {i}", "x")
        out = HistoryTool().execute(notes_ctx, {"limit": 2})
        assert len([line for line in out if line[:1].isdigit()]) == 2

    def test_broken_database(self, ctx):
        class Broken:
            def query_all(self, *a, **k):
                raise RuntimeError("locked")

            def query_one(self, *a, **k):
                raise RuntimeError("locked")

        ctx.db = Broken()
        assert "could not read" in HistoryTool().execute(ctx, {})[0]


class TestToolIsolation:
    """No tool may speak or listen for itself."""

    @pytest.mark.parametrize("tool", [NoteTool(), HistoryTool()])
    def test_tools_never_touch_speech(self, ctx, memory_db, tool):
        ctx.db = memory_db
        ctx.speaker = ExplodingSpeaker()
        ctx.listener = ExplodingSpeaker()
        tool.execute(ctx, {})  # must not raise

    def test_wikipedia_and_youtube_sentinels(self, ctx, memory_db):
        ctx.db = memory_db
        ctx.speaker = ExplodingSpeaker()
        ctx.listener = ExplodingSpeaker()
        assert WikipediaTool().execute(ctx, {}) == WikipediaTool.NEEDS_TOPIC
        assert YouTubeTool().execute(ctx, {}) == YouTubeTool.NEEDS_QUERY

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("information about python", {"topic": "python"}),
            ("tell me about python", {"topic": "python"}),
            ("information", {}),
        ],
    )
    def test_wikipedia_slot_extraction(self, ctx, utterance, expected):
        assert WikipediaTool().extract_slots(utterance, ctx) == expected

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("play lofi beats on youtube", {"query": "lofi beats"}),
            ("play music on youtube", {}),
            ("play jazz", {"query": "jazz"}),
            ("youtube lofi", {"query": "lofi"}),
            ("play", {}),
        ],
    )
    def test_youtube_slot_extraction(self, ctx, utterance, expected):
        assert YouTubeTool().extract_slots(utterance, ctx) == expected


class TestSpeechProtocols:
    def test_implementations_satisfy_protocols(self):
        assert isinstance(TextSpeaker(), Speaker)
        assert isinstance(TextListener(), Listener)

    def test_speaker_prints(self):
        buf = io.StringIO()
        TextSpeaker(stream=buf).speak("hello")
        assert buf.getvalue().strip() == "hello"

    def test_speaker_ignores_empty(self):
        buf = io.StringIO()
        TextSpeaker(stream=buf).speak("")
        assert buf.getvalue() == ""

    def test_scripted_listener_lowercases(self):
        listener = TextListener(responses=["  Tell Me The News  "])
        assert listener.listen() == "tell me the news"

    def test_scripted_listener_repeats_last(self):
        listener = TextListener(responses=["a", "b"])
        listener.listen()
        listener.listen()
        assert listener.listen() == "b"

    def test_reader_injection(self):
        seen = []

        def reader(prompt):
            seen.append(prompt)
            return " Weather In Delhi "

        assert TextListener(prompt="You: ", reader=reader).listen() == (
            "weather in delhi"
        )
        assert seen == ["You: "]

    @pytest.mark.parametrize("error", [EOFError("eof"), KeyboardInterrupt("ctrl-c")])
    def test_interrupts_return_nothing_heard(self, error):
        def reader(prompt):
            raise error

        assert TextListener(reader=reader).listen() == NOTHING_HEARD

    def test_blank_input(self):
        assert TextListener(reader=lambda p: "   ").listen() == NOTHING_HEARD
