"""HTTP routes for the dashboard, and the JSON shape they serve.

This module is the whole request layer. It holds **no SQL and no aggregation
logic**: every figure comes from
:class:`~assistant.analytics.service.AnalyticsService`, and this file only
decides how to *present* what that service already measured.

Two payloads are built from one function, so the HTML page and the JSON API
can never disagree:

* :func:`build_dashboard_payload` -- the JSON-safe dictionary.
* :func:`build_dashboard_context` -- the template context, which is that same
  payload plus the small amount of presentation state the template needs.

The honesty rule
----------------
The analytics layer deliberately reports ``0.0`` for a metric it cannot
compute, such as a match rate over zero interactions or an average note
length with no notes. Those are *undefined*, not *zero*, and printing "0.0%"
beside an empty dashboard would be a lie told by a rounding step. So every
such value travels as ``None`` on the way out and the templates render a
dash. Nothing is invented here -- only the decision to show "not available"
instead of "0".
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import date
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from assistant.analytics import AnalyticsService, DateRange, ToolUsage
from assistant.logging_config import get_logger

logger = get_logger(__name__)

#: Shown wherever a number is genuinely unavailable, so an empty dashboard
#: never displays a zero-derived statistic dressed up as a measurement.
NOT_AVAILABLE = "--"

#: Wording for a database that exists but holds nothing.
NO_ACTIVITY_HEADING = "No recorded activity"
NO_ACTIVITY_BODY = (
    "There are no interactions and no notes in this database yet. "
    "Use the assistant a few times, then reload this page."
)

#: Wording for the case where the database file itself is missing.
NO_DATABASE_BODY = (
    "No analytics database was found at the configured location. "
    "Start the assistant once so it can create one, then reload this page."
)


class InvalidDateRange(ValueError):
    """Raised when a caller supplies a date the dashboard cannot use.

    A distinct type so the route can turn it into a friendly message instead
    of a stack trace. A malformed ``?start=`` is a typo, not a server fault.
    """


def parse_date_range(start: str | None, end: str | None) -> DateRange | None:
    """Turn the ``start``/``end`` query parameters into a :class:`DateRange`.

    Both bounds are optional and inclusive. Passing neither means "all
    available data", which is the dashboard's default.

    Args:
        start: an ISO ``YYYY-MM-DD`` date, or ``None``.
        end: an ISO ``YYYY-MM-DD`` date, or ``None``.

    Returns:
        ``None`` for all time, otherwise the requested range.

    Raises:
        InvalidDateRange: if a bound is not a real ISO date, if a bound is
            not a date at all (``"last-tuesday"``), or if ``end`` falls
            before ``start``. Silently swapping the two would return a
            confident wrong answer, so it is refused instead.
    """
    start_date = _parse_bound(start, "start")
    end_date = _parse_bound(end, "end")

    if start_date is None and end_date is None:
        return None
    if start_date is not None and end_date is not None and end_date < start_date:
        raise InvalidDateRange(
            f"The end date ({end_date}) falls before the start date "
            f"({start_date})."
        )
    return DateRange(start=start_date, end=end_date)


def _parse_bound(raw: str | None, name: str) -> date | None:
    """Parse one bound, treating blank input as "not supplied".

    Raises:
        InvalidDateRange: if the value is not a real ISO date. The message
            names the offending bound and shows the expected format, so a
            mistyped query parameter is fixable without reading the source.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise InvalidDateRange(
            f"The {name} date {text!r} is not valid. "
            "Use the ISO format YYYY-MM-DD, for example 2026-09-01."
        ) from None


# ----------------------------------------------------------------------
# JSON-safe payloads
# ----------------------------------------------------------------------
def build_dashboard_payload(
    service: AnalyticsService, date_range: DateRange | None = None
) -> dict[str, Any]:
    """Return the whole dashboard as a JSON-safe dictionary.

    Everything the service returns is already a primitive, so this is a
    mechanical conversion rather than a calculation. The one judgement call
    is :func:`_available`, applied to every metric that is undefined when
    there is no data.

    The shape is grouped the way the page is grouped: headline numbers, then
    tools, then activity, then notes.
    """
    report = service.dashboard(date_range)
    summary = report.summary
    notes = report.notes

    # Averages over an empty set, and a rate over zero interactions, are
    # undefined rather than zero. They travel as None so the template can
    # print a dash instead of a fabricated statistic.
    has_interactions = summary.total_interactions > 0
    has_notes = notes.total > 0

    return {
        "range": {
            "start": _iso(date_range.start if date_range else None),
            "end": _iso(date_range.end if date_range else None),
            "label": report.range_used,
            "is_all_time": date_range is None or date_range.is_unbounded,
        },
        "kpis": {
            "total_interactions": summary.total_interactions,
            "unique_tools": summary.unique_tools,
            "most_used_tool": summary.most_used_tool,
            "most_used_count": _available(summary.most_used_count, has_interactions),
            "matched_rate": _available(summary.match_rate, has_interactions),
            "total_notes": notes.total,
        },
        "top_tools": _top_tools_payload(service, report, date_range),
        "daily": [_bucket_payload(bucket) for bucket in report.daily],
        "weekly": [_bucket_payload(bucket) for bucket in report.weekly],
        "notes": {
            "total": notes.total,
            "average_length": _available(notes.average_length, has_notes),
            "longest_length": _available(notes.longest_length, has_notes),
            "first_created": notes.first_created,
            "last_created": notes.last_created,
            "days_with_notes": notes.days_with_notes,
        },
        # The existing AnalyticsService summary, exposed verbatim so the
        # dashboard's summary panel and the assistant's spoken summary are
        # literally the same numbers.
        "summary": asdict(summary),
        "busiest_day": _bucket_or_none(report.busiest_day),
        "busiest_week": _bucket_or_none(report.busiest_week),
        "has_activity": summary.has_activity,
        "has_notes": notes.has_notes,
    }


def _top_tools_payload(
    service: AnalyticsService,
    report: Any,
    date_range: DateRange | None,
) -> list[dict[str, Any]]:
    """Return the ranked tools with their share of the range.

    The ranking comes from the report's ``top_tools``, which already excludes
    unresolved commands. The percentage comes from
    :meth:`~assistant.analytics.service.AnalyticsService.tool_usage`, so the
    service stays the single place that knows how a share is computed; this
    function only joins the two results by tool name.

    The denominator is every interaction in the range, including the ones
    that matched no tool, so the real tools add up to less than 100 when some
    commands went unresolved. That is the honest reading, not a rounding bug.
    """
    shares = {
        usage.tool_name: usage.percentage
        for usage in service.tool_usage(date_range)
    }
    return [
        {
            "tool_name": tool.tool_name,
            "count": tool.count,
            "percentage": shares.get(tool.tool_name),
        }
        for tool in report.top_tools
    ]


def _bucket_payload(bucket: Any) -> dict[str, Any]:
    """Return one daily or weekly activity row."""
    return {"period": bucket.period, "count": bucket.count}


def _bucket_or_none(bucket: Any) -> dict[str, Any] | None:
    """Return a peak row, or ``None`` when there is nothing to peak."""
    return None if bucket is None else _bucket_payload(bucket)


def _available(value: Any, has_data: bool) -> Any:
    """Return ``value``, or ``None`` when it is undefined.

    An average over zero rows, or a match rate over zero interactions, has no
    value at all. Showing the service's ``0.0`` would imply a measurement was
    taken and came out zero, which is not what happened.
    """
    return value if has_data else None


def _iso(day: date | None) -> str | None:
    """Render a date for JSON, or ``None``.

    ISO 8601 (``YYYY-MM-DD``) is both machine-parseable and readable, so the
    API needs no second format. ``None`` is preserved rather than turned into
    an empty string, so a client can tell "no bound" from "the epoch".
    """
    return day.isoformat() if day is not None else None


def build_dashboard_context(
    service: AnalyticsService,
    date_range: DateRange | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Return the template context for the dashboard page.

    Wraps :func:`build_dashboard_payload` with the few presentation values
    the template needs and the payload itself cannot supply, such as the
    human wording for an empty database.
    """
    payload = build_dashboard_payload(service, date_range)
    return {
        "payload": payload,
        "kpis": payload["kpis"],
        "top_tools": payload["top_tools"],
        "daily": payload["daily"],
        "weekly": payload["weekly"],
        "notes": payload["notes"],
        "summary": payload["summary"],
        "busiest_day": payload["busiest_day"],
        "busiest_week": payload["busiest_week"],
        "range_label": payload["range"]["label"],
        "range_start": payload["range"]["start"] or "",
        "range_end": payload["range"]["end"] or "",
        "has_activity": payload["has_activity"],
        "has_notes": payload["has_notes"],
        "is_empty": not payload["has_activity"] and not payload["has_notes"],
        "not_available": NOT_AVAILABLE,
        "empty_heading": NO_ACTIVITY_HEADING,
        "empty_body": NO_ACTIVITY_BODY,
        "error": error,
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
router = APIRouter()


def get_service(request: Request) -> Iterator[AnalyticsService]:
    """Yield a fresh AnalyticsService over a read-only connection.

    A dependency rather than a module-level singleton, for two reasons. The
    connection is opened with ``mode=ro`` and closed once the response is
    rendered, so the dashboard holds no lock between requests. And the service
    is rebuilt per request, so the page always shows the assistant's most
    recent rows without needing a restart.

    Raises:
        HTTPException: 503 when the configured database cannot be opened.
            A missing file is a "nothing to show yet" condition, not a crash.
    """
    from assistant.dashboard.app import open_readonly

    settings = request.app.state.dashboard
    try:
        database = open_readonly(settings.db_path)
    except sqlite3.Error as exc:
        logger.warning("Could not open the analytics database: %s", exc)
        raise HTTPException(
            status_code=503, detail="The analytics database is not available."
        ) from exc

    try:
        yield AnalyticsService(database)
    finally:
        database.close()


def _range_or_error(
    start: str | None, end: str | None
) -> tuple[DateRange | None, str | None]:
    """Parse the range, turning a bad value into a message rather than a crash."""
    try:
        return parse_date_range(start, end), None
    except InvalidDateRange as exc:
        logger.info("Rejected dashboard date range: %s", exc)
        return None, str(exc)


@router.get("/", response_class=HTMLResponse)
def dashboard_page(
    request: Request,
    start: str | None = None,
    end: str | None = None,
    service: AnalyticsService = Depends(get_service),
) -> HTMLResponse:
    """Render the single-page dashboard.

    Both query parameters are optional. An unparseable date is reported in
    the page itself rather than raised: a mistyped filter should not look
    like a broken server. The page shows the message and falls back to all
    available data, so the reader still gets real numbers.
    """
    date_range, error = _range_or_error(start, end)
    context = build_dashboard_context(service, date_range, error=error)
    return request.app.state.templates.TemplateResponse(
        request=request, name="index.html", context=context
    )


@router.get("/api/dashboard")
def dashboard_api(
    start: str | None = None,
    end: str | None = None,
    service: AnalyticsService = Depends(get_service),
) -> JSONResponse:
    """Return the same dashboard data as JSON.

    A bad date range is a 400 carrying a readable ``detail`` string, so a
    client can show the reason instead of silently reporting the wrong
    window.
    """
    date_range, error = _range_or_error(start, end)
    if error is not None:
        return JSONResponse(status_code=400, content={"detail": error})
    return JSONResponse(content=build_dashboard_payload(service, date_range))


@router.get("/api/health")
def dashboard_health() -> dict[str, Any]:
    """Report that the dashboard process is alive.

    Deliberately tiny, and deliberately free of anything about the database:
    no path, no counts, no configuration. It answers "is this process
    running", which is the only question a liveness probe should ask.
    """
    return {"status": "ok", "service": "assistant-dashboard", "read_only": True}
