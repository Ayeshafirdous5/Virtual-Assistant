"""Tests for the dashboard routes: payloads, the HTML page, and the JSON API.

The central claim these tests defend is that **AnalyticsService remains the
single source of truth**. Every number the dashboard serves is compared back
against the service reading the same database, so a dashboard that invented,
rounded or recomputed a figure would fail here.

The second claim is honesty about absence. Where a metric is undefined the
payload must carry ``None`` and the page must show a dash, never a
zero-derived fake.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistant.analytics import AnalyticsService
from assistant.core.database import Database
from assistant.dashboard.app import create_app, open_readonly
from assistant.dashboard.routes import (
    NO_ACTIVITY_HEADING,
    NOT_AVAILABLE,
    InvalidDateRange,
    build_dashboard_context,
    build_dashboard_payload,
    parse_date_range,
)

#: The rows the shared fixture writes.
TOTAL = 5
UNIQUE_TOOLS = 3


@pytest.fixture
def populated(tmp_path: Path) -> Path:
    """Three tools over three days, one unmatched, one note."""
    path = tmp_path / "routes.db"
    db = Database(path)
    db.initialize()
    for tool, day in (
        ("weather", "2026-09-21"),
        ("jokes", "2026-09-21"),
        ("jokes", "2026-09-22"),
        ("notes", "2026-09-23"),
        ("none", "2026-09-23"),
    ):
        db.execute(
            "INSERT INTO interactions (occurred_at, tool_name, utterance, "
            "response) VALUES (?,?,?,?)",
            (f"{day} 09:00:00", tool, "u", "r"),
        )
    db.execute(
        "INSERT INTO notes (created_at, body) VALUES (?,?)",
        ("2026-09-22 10:00:00", "buy milk"),
    )
    db.close()
    return path


@pytest.fixture
def empty(tmp_path: Path) -> Path:
    """A migrated database with nothing in it."""
    path = tmp_path / "empty.db"
    db = Database(path)
    db.initialize()
    db.close()
    return path


@pytest.fixture
def client(populated: Path) -> TestClient:
    return TestClient(create_app(populated))


@pytest.fixture
def service(populated: Path):
    """A service over the same data, for the source-of-truth comparisons."""
    db = open_readonly(populated)
    yield AnalyticsService(db)
    db.close()


# ----------------------------------------------------------------------
# Date-range parsing
# ----------------------------------------------------------------------
class TestDateRangeParsing:
    def test_no_bounds_means_all_time(self):
        assert parse_date_range(None, None) is None
        assert parse_date_range("", "") is None
        assert parse_date_range("  ", "  ") is None

    def test_a_valid_range(self):
        span = parse_date_range("2026-09-21", "2026-09-23")
        assert span.start == date(2026, 9, 21)
        assert span.end == date(2026, 9, 23)

    def test_one_bound_only_is_allowed(self):
        assert parse_date_range("2026-09-21", None).start == date(2026, 9, 21)
        assert parse_date_range(None, "2026-09-23").end == date(2026, 9, 23)

    @pytest.mark.parametrize(
        "bad", ["nonsense", "2026-13-45", "last-tuesday", "21/09/2026"]
    )
    def test_a_malformed_bound_is_refused(self, bad):
        with pytest.raises(InvalidDateRange):
            parse_date_range(bad, None)
        with pytest.raises(InvalidDateRange):
            parse_date_range(None, bad)

    def test_a_reversed_range_is_refused_not_swapped(self):
        """Silently swapping would answer a different question."""
        with pytest.raises(InvalidDateRange):
            parse_date_range("2026-09-23", "2026-09-21")

    def test_the_error_names_the_offending_bound(self):
        with pytest.raises(InvalidDateRange) as info:
            parse_date_range("nope", None)
        assert "start" in str(info.value)

    def test_a_single_day_is_valid(self):
        span = parse_date_range("2026-09-21", "2026-09-21")
        assert span.start == span.end == date(2026, 9, 21)


# ----------------------------------------------------------------------
# The JSON payload
# ----------------------------------------------------------------------
class TestPayload:
    def test_it_is_json_serialisable(self, service):
        json.dumps(build_dashboard_payload(service))

    def test_the_shape(self, service):
        payload = build_dashboard_payload(service)
        assert set(payload) >= {
            "range", "kpis", "top_tools", "daily", "weekly", "notes",
            "summary", "busiest_day", "busiest_week",
            "has_activity", "has_notes",
        }

    def test_kpis_match_the_service(self, service):
        """AnalyticsService is the source of truth."""
        payload = build_dashboard_payload(service)
        report = service.dashboard()
        assert payload["kpis"]["total_interactions"] == report.summary.total_interactions == TOTAL
        assert payload["kpis"]["unique_tools"] == report.summary.unique_tools == UNIQUE_TOOLS
        assert payload["kpis"]["most_used_tool"] == "jokes"
        assert payload["kpis"]["most_used_count"] == 2
        assert payload["kpis"]["matched_rate"] == 80.0
        assert payload["kpis"]["total_notes"] == 1

    def test_the_raw_summary_is_exposed_verbatim(self, service):
        """So the page and the spoken summary are literally the same numbers."""
        payload = build_dashboard_payload(service)
        assert payload["summary"] == asdict(service.dashboard().summary)

    def test_top_tools_are_ranked_and_exclude_unmatched(self, service):
        tools = build_dashboard_payload(service)["top_tools"]
        assert [t["tool_name"] for t in tools] == ["jokes", "notes", "weather"]
        assert all(t["tool_name"] != "none" for t in tools)

    def test_tool_percentages_match_the_service(self, service):
        tools = build_dashboard_payload(service)["top_tools"]
        by_name = {u.tool_name: u.percentage for u in service.tool_usage()}
        for entry in tools:
            assert entry["percentage"] == by_name[entry["tool_name"]]

    def test_daily_and_weekly_series(self, service):
        payload = build_dashboard_payload(service)
        assert payload["daily"] == [
            {"period": "2026-09-21", "count": 2},
            {"period": "2026-09-22", "count": 1},
            {"period": "2026-09-23", "count": 2},
        ]
        assert payload["weekly"] == [{"period": "2026-W39", "count": 5}]

    def test_notes_summary(self, service):
        notes = build_dashboard_payload(service)["notes"]
        assert notes["total"] == 1
        assert notes["average_length"] == 8.0
        assert notes["longest_length"] == 8
        assert notes["days_with_notes"] == 1
        assert notes["first_created"].startswith("2026-09-22")

    def test_peaks(self, service):
        payload = build_dashboard_payload(service)
        assert payload["busiest_day"]["count"] == 2
        assert payload["busiest_week"] == {"period": "2026-W39", "count": 5}

    def test_a_range_filters_the_payload(self, service):
        span = parse_date_range("2026-09-21", "2026-09-21")
        payload = build_dashboard_payload(service, span)
        assert payload["kpis"]["total_interactions"] == 2
        assert payload["daily"] == [{"period": "2026-09-21", "count": 2}]
        assert payload["range"]["is_all_time"] is False

    def test_it_is_deterministic(self, service):
        assert build_dashboard_payload(service) == build_dashboard_payload(service)

    def test_no_note_body_is_ever_exposed(self, service):
        """The dashboard shows lengths, never what a note says."""
        assert "buy milk" not in json.dumps(build_dashboard_payload(service))


# ----------------------------------------------------------------------
# Absence is reported, not faked
# ----------------------------------------------------------------------
class TestEmptyDatabase:
    @pytest.fixture
    def empty_service(self, empty: Path):
        db = open_readonly(empty)
        yield AnalyticsService(db)
        db.close()

    @pytest.fixture
    def empty_client(self, empty: Path) -> TestClient:
        return TestClient(create_app(empty))

    def test_the_payload_serves(self, empty_service):
        payload = build_dashboard_payload(empty_service)
        assert payload["kpis"]["total_interactions"] == 0
        assert payload["top_tools"] == []
        assert payload["daily"] == []
        assert payload["busiest_day"] is None

    def test_undefined_metrics_are_none_not_zero(self, empty_service):
        """A rate over nothing is undefined, so it travels as None."""
        kpis = build_dashboard_payload(empty_service)["kpis"]
        assert kpis["matched_rate"] is None
        assert kpis["most_used_count"] is None
        assert kpis["most_used_tool"] is None

    def test_notes_metrics_are_none_too(self, empty_service):
        notes = build_dashboard_payload(empty_service)["notes"]
        assert notes["average_length"] is None
        assert notes["longest_length"] is None

    def test_the_page_shows_an_empty_state(self, empty_client):
        response = empty_client.get("/")
        assert response.status_code == 200
        assert NO_ACTIVITY_HEADING in response.text

    def test_the_empty_page_fabricates_no_numbers(self, empty_client):
        """No KPI cards at all, so there is no zero to misread."""
        text = empty_client.get("/").text
        assert "Total interactions" not in text
        assert NO_ACTIVITY_HEADING in text

    def test_the_context_marks_itself_empty(self, empty_service):
        context = build_dashboard_context(empty_service)
        assert context["is_empty"] is True
        assert context["has_activity"] is False
        assert context["not_available"] == NOT_AVAILABLE


# ----------------------------------------------------------------------
# The JSON API
# ----------------------------------------------------------------------
class TestJsonApi:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "service": "assistant-dashboard",
            "read_only": True,
        }

    def test_health_reveals_nothing_about_the_database(self, client, populated):
        """A liveness probe has no use for a path or a count."""
        body = client.get("/api/health").text
        assert str(populated) not in body
        assert "assistant.db" not in body

    def test_dashboard(self, client):
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        assert response.json()["kpis"]["total_interactions"] == TOTAL

    def test_dashboard_is_deterministic(self, client):
        first = client.get("/api/dashboard").json()
        assert first == client.get("/api/dashboard").json()

    def test_a_valid_range_filters_the_response(self, client):
        body = client.get(
            "/api/dashboard?start=2026-09-21&end=2026-09-21"
        ).json()
        assert body["kpis"]["total_interactions"] == 2
        assert body["range"]["start"] == "2026-09-21"
        assert body["range"]["end"] == "2026-09-21"
        assert body["range"]["label"] == "2026-09-21"
        assert body["range"]["is_all_time"] is False

    @pytest.mark.parametrize("bad", ["nonsense", "2026-13-45", "last-tuesday"])
    def test_a_malformed_date_is_a_400(self, client, bad):
        response = client.get(f"/api/dashboard?start={bad}")
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_a_reversed_range_is_a_400_not_a_swap(self, client):
        response = client.get(
            "/api/dashboard?start=2026-09-23&end=2026-09-21"
        )
        assert response.status_code == 400
        assert "before" in response.json()["detail"]

    def test_the_error_message_is_human_readable(self, client):
        detail = client.get("/api/dashboard?start=nope").json()["detail"]
        assert "YYYY-MM-DD" in detail

    def test_openapi_docs_are_served(self, client):
        assert client.get("/api/docs").status_code == 200


# ----------------------------------------------------------------------
# The HTML page
# ----------------------------------------------------------------------
class TestHtmlPage:
    def test_it_renders(self, client):
        assert client.get("/").status_code == 200

    def test_the_header_is_present(self, client):
        text = client.get("/").text
        assert "Virtual Assistant Analytics" in text
        assert "read-only" in text

    @pytest.mark.parametrize(
        "section",
        [
            "Total interactions",
            "Unique tools",
            "Most used tool",
            "Match rate",
            "Total notes",
            "Top tools",
            "Activity",
            "Peaks",
            "Notes",
        ],
    )
    def test_every_section_renders(self, client, section):
        assert section in client.get("/").text

    def test_it_shows_the_measured_numbers(self, client):
        text = client.get("/").text
        assert "jokes" in text          # most used tool
        assert "80.0" in text           # match rate
        assert "2026-09-21" in text     # busiest day and a daily label

    def test_the_charts_carry_their_data(self, client):
        text = client.get("/").text
        assert 'data-chart="bar"' in text
        assert "2026-09-21" in text

    def test_the_stylesheet_and_script_are_linked(self, client):
        text = client.get("/").text
        assert "/static/style.css" in text
        assert "/static/dashboard.js" in text

    def test_the_static_files_are_served(self, client):
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/dashboard.js").status_code == 200

    def test_the_page_works_without_javascript(self, client):
        """Every chart carries a noscript table, so no data is JS-only."""
        assert client.get("/").text.count("<noscript>") >= 2

    def test_a_valid_range_is_reflected_in_the_form(self, client):
        text = client.get("/?start=2026-09-21&end=2026-09-22").text
        assert 'value="2026-09-21"' in text
        assert 'value="2026-09-22"' in text

    def test_a_bad_date_shows_an_alert_but_still_renders(self, client):
        """A mistyped filter is a user error, not a server fault."""
        response = client.get("/?start=nonsense")
        assert response.status_code == 200
        assert "could not be used" in response.text
        assert "Total interactions" in response.text

    def test_a_reversed_date_shows_an_alert(self, client):
        response = client.get("/?start=2026-09-23&end=2026-09-21")
        assert response.status_code == 200
        assert "could not be used" in response.text

    def test_it_never_prints_a_note_body(self, client):
        """Lengths only. A note is private text."""
        assert "buy milk" not in client.get("/").text


# ----------------------------------------------------------------------
# No secrets, and nothing the user did not ask to publish
# ----------------------------------------------------------------------
class TestNoSecrets:
    @pytest.mark.parametrize("path", ["/", "/api/dashboard", "/api/health"])
    def test_no_credential_material_appears(self, client, path):
        body = client.get(path).text
        for needle in ("NEWS_API_KEY", "OPENWEATHER_API_KEY", "api_key", ".env"):
            assert needle not in body, needle

    def test_no_database_path_appears(self, client, populated):
        body = client.get("/").text + client.get("/api/dashboard").text
        assert str(populated) not in body
        assert "dashboard.db" not in body

    def test_no_utterance_text_is_exposed(self, client):
        """The payload carries counts and names, not what was said.

        Interactions store the user's utterance and the assistant's reply.
        The dashboard has no reason to render either, so it must not.
        """
        # Written through a separate writable handle, because the dashboard's
        # own connection is read-only by design.
        path = client.app.state.dashboard.db_path
        writer = Database(path)
        writer.execute(
            "INSERT INTO interactions (occurred_at, tool_name, utterance, "
            "response) VALUES ('2026-09-21 09:00:00','weather',?,?)",
            ("a private utterance", "a private response"),
        )
        writer.close()
        body = client.get("/api/dashboard").text + client.get("/").text
        assert "a private utterance" not in body
        assert "a private response" not in body



