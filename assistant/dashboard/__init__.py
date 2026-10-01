"""A local, read-only web dashboard over the assistant's own history.

This package is the first web surface for the project, and it is deliberately
boring: it renders what :class:`~assistant.analytics.service.AnalyticsService`
already knows, and adds no analysis of its own.

Design rules
------------
* **No SQL here.** Routes call typed methods on
  :class:`~assistant.analytics.service.AnalyticsService` and format the
  frozen dataclasses it returns. Every number on the page is a number that
  service measured, so the dashboard can never disagree with the analytics
  layer or with the assistant's spoken summary.
* **Structurally read-only.** The dashboard connects with SQLite's
  ``mode=ro`` URI and wraps it in :class:`~assistant.dashboard.app.ReadOnlyDatabase`,
  which exposes only ``query_one`` and ``query_one``'s sibling ``query_all``.
  There is no code path from a request to an ``INSERT``, so "the dashboard
  never writes" is enforced by the connection itself rather than by
  convention.
* **No secrets.** Templates and JSON carry counts, tool names and timestamps
  only. No API key, no ``.env`` content and no absolute database path is
  ever rendered or served.
* **Honest about absence.** A metric the schema cannot supply is rendered as
  unavailable rather than as a zero-derived fake. See
  :func:`assistant.dashboard.routes.build_dashboard_payload`.
* **Nothing on import.** Building the app opens no connection; the database
  is opened per request and closed again, so importing this package cannot
  create or touch a file.

Run it locally with::

    python -m assistant.dashboard.app
"""

from __future__ import annotations

from assistant.dashboard.app import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ReadOnlyDatabase,
    create_app,
    main,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ReadOnlyDatabase",
    "create_app",
    "main",
]
