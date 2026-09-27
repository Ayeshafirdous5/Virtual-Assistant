"""End-to-end orchestration tests driven by scripted text input.

These exercise the real ``assistant.app.run`` loop with a stubbed speaker and
a scripted listener. No microphone, no network, no browser, and the database
is the temporary one from ``conftest``.
"""

from __future__ import annotations

import dataclasses

from assistant.app import (
    MSG_DIDNT_UNDERSTAND,
    PROMPTS,
    attach_database,
    close_database,
    parse_args,
    run,
    speak_all,
    wishme,
)
from assistant.core.context import AppContext
from assistant.core.database import Database
from assistant.tools.wikipedia import WikipediaTool
from assistant.tools.youtube import YouTubeTool


class TestCli:
    def test_text_flag(self):
        assert parse_args(["--text"]).text is True
        assert parse_args([]).text is False

    def test_log_level_flag(self):
        assert parse_args(["--log-level", "debug"]).log_level == "debug"


class TestGreeting:
    def test_wishme_returns_a_time_of_day(self):
        assert wishme() in ("Morning", "Afternoon", "Evening")

    def test_speak_all_skips_empty(self, ctx, capsys):
        speak_all(ctx, ["", "hello", None])
        assert capsys.readouterr().out.strip() == "hello"

    def test_speak_all_accepts_a_list(self, ctx, capsys):
        speak_all(ctx, ["one", "two"])
        out = capsys.readouterr().out
        assert "one" in out and "two" in out


class TestRunLoop:
    def test_exits_cleanly(self, run_text_session):
        code, spoken = run_text_session(["exit"])
        assert code == 0
        assert "Goodbye! Have a great day!" in spoken

    def test_unknown_command_uses_original_wording(self, run_text_session):
        _, spoken = run_text_session(["xyzzy gibberish", "exit"])
        assert MSG_DIDNT_UNDERSTAND in spoken
        assert MSG_DIDNT_UNDERSTAND == (
            "I'm sorry, I didn't understand. Can you please repeat?"
        )

    def test_persists_every_handled_command(self, config, db, run_text_session):
        code, _ = run_text_session(
            ["note buy milk", "list notes", "exit"], database=db
        )
        assert code == 0
        rows = db.query_all("SELECT tool_name FROM interactions ORDER BY id")
        tools = [r["tool_name"] for r in rows]
        # Two note commands plus the exit; one row per handled command.
        assert tools == ["notes", "notes", "system"]
        assert db.query_one("SELECT COUNT(*) AS n FROM interactions")["n"] == 3

    def test_no_database_is_logging_only(self, run_text_session):
        # database=None means ctx.db stays None; the loop must still work.
        code, spoken = run_text_session(["note hello", "exit"], database=None)
        assert code == 0
        assert "no place to keep notes" in " ".join(spoken)

    def test_broken_database_does_not_stop_the_loop(self, config, run_text_session):
        class Broken:
            def query_all(self, *a, **k):
                raise RuntimeError("database is locked")

            def query_one(self, *a, **k):
                raise RuntimeError("database is locked")

            def execute(self, *a, **k):
                raise RuntimeError("database is locked")

        code, spoken = run_text_session(
            ["note hello", "list notes", "exit"], database=Broken()
        )
        assert code == 0
        assert "could not do that" in " ".join(spoken)

    def test_silence_does_not_crash(self, run_text_session):
        # Blank replies are treated as nothing heard and simply retried.
        code, _ = run_text_session(["", "", "exit"])
        assert code == 0


class TestDatabaseAttachment:
    def test_attach_and_close(self, config, tmp_path):
        ctx = AppContext(config=config)
        attach_database(ctx, config)
        assert ctx.has_db() is True
        assert ctx.db.schema_version() >= 2
        close_database(ctx)
        assert ctx.db is None

    def test_close_without_database_is_safe(self, ctx):
        close_database(ctx)  # must not raise

    def test_attach_failure_leaves_db_none(self, config, tmp_path):
        bad = dataclasses.replace(config, db_path=tmp_path)  # a directory
        ctx = AppContext(config=bad)
        attach_database(ctx, bad)
        assert ctx.db is None

    def test_two_turn_wikipedia(self, config, run_text_session, monkeypatch):
        seen = {}

        import assistant.tools.wikipedia as wiki

        monkeypatch.setattr(wiki, "lookup", lambda topic: f"LOOKED_UP:{topic}")

        code, spoken = run_text_session(["information", "python", "exit"])
        assert code == 0
        assert "You need information on what topic?" in spoken
        assert "LOOKED_UP:python" in spoken

    def test_two_turn_youtube(self, config, run_text_session, monkeypatch):
        import assistant.tools.youtube as yt

        class FakeMusic:
            def play(self, query):
                pass

            def quit_driver(self):
                pass

        monkeypatch.setattr(yt, "Music", FakeMusic)
        code, spoken = run_text_session(["play", "lofi beats", "exit"])
        assert code == 0
        assert "What would you like me to play?" in spoken
        assert any("lofi beats" in line for line in spoken)

    def test_prompts_cover_both_sentinels(self):
        assert WikipediaTool.NEEDS_TOPIC in PROMPTS
        assert YouTubeTool.NEEDS_QUERY in PROMPTS
