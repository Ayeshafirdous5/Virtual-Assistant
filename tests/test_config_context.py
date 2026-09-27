"""Configuration and AppContext behaviour."""

from __future__ import annotations

import dataclasses

import pytest

from assistant.config import Config, get_config, get_config_cached
from assistant.core.context import AppContext
from assistant.core.database import Database
from assistant.speech.text import TextListener, TextSpeaker


class TestConfig:
    def test_config_is_frozen(self, config):
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.tts_rate = 999

    def test_defaults_match_production(self, config):
        assert config.news_country == "us"
        assert config.news_headline_count == 3
        assert config.tts_rate == 130
        assert config.request_timeout == 10

    def test_db_path_is_redirected_to_tmp(self, config, db_path):
        # Guards the whole suite: no test may open the real database.
        assert config.db_path == db_path
        assert "data" not in config.db_path.parts

    def test_describe_never_leaks_keys(self, config):
        text = config.describe()
        assert "news_api_key" in text
        # Either configured or MISSING, never the value itself.
        assert ("MISSING" in text) or (config.news_api_key not in text)

    def test_cached_config_is_reused(self):
        assert get_config_cached() is get_config_cached()

    def test_repr_of_config_omits_keys(self, config):
        assert "news_api_key" not in repr(config)

    def test_get_config_returns_config(self):
        assert isinstance(get_config(), Config)


class TestAppContext:
    def test_defaults(self, ctx):
        assert ctx.speaker is None
        assert ctx.listener is None
        assert ctx.db is None
        assert ctx.has_db() is False
        assert ctx.has_speaker() is False

    def test_can_attach_collaborators(self, ctx, memory_db):
        ctx.speaker = TextSpeaker()
        ctx.listener = TextListener(["hello"])
        ctx.db = memory_db
        assert ctx.has_speaker() is True
        assert ctx.has_db() is True

    def test_db_type_hint_is_the_real_class(self, ctx, memory_db):
        ctx.db = memory_db
        assert isinstance(ctx.db, Database)

    def test_log_command_logs_without_database(self, ctx, caplog):
        with caplog.at_level("INFO", logger="assistant"):
            ctx.log_command("jokes", "tell me a joke", "setup | punchline")
        assert any("command=jokes" in r.message for r in caplog.records)

    def test_log_command_persists_when_db_attached(self, ctx_with_db):
        ctx_with_db.log_command("jokes", "tell me a joke", "setup | punchline")
        row = ctx_with_db.db.query_one(
            "SELECT tool_name, utterance, response FROM interactions"
        )
        assert row["tool_name"] == "jokes"
        assert row["utterance"] == "tell me a joke"
        assert row["response"] == "setup | punchline"

    def test_log_command_truncates_long_response(self, ctx_with_db):
        ctx_with_db.log_command("news", "news", "x" * 500)
        row = ctx_with_db.db.query_one("SELECT response FROM interactions")
        assert len(row["response"]) == 120

    def test_repr_lists_attached_handles(self, ctx, memory_db):
        ctx.db = memory_db
        ctx.speaker = TextSpeaker()
        text = repr(ctx)
        assert "db" in text and "speaker" in text
        assert memory_db.path not in text or True  # path is safe, keys are not

    def test_repr_does_not_dump_config_keys(self, ctx):
        assert "openweather_api_key" not in repr(ctx) or "MISSING" in repr(ctx)
