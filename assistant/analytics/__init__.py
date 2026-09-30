"""Analytics and insights over the assistant's own history.

A read-only layer that answers "how much have I used you, and what do I use
most" from the data already in SQLite. It adds no tables, no dependencies and
no writes: the same ``interactions`` and ``notes`` rows that
:mod:`assistant.storage` reads are aggregated here.

The layer is split so it can be reused without the speech layer:
:mod:`assistant.analytics.service` holds the typed queries and returns
frozen dataclasses, and :class:`~assistant.tools.analytics.AnalyticsTool`
turns one of those summaries into a sentence.

Importing this package is cheap. It pulls in only the standard library and
the existing storage layer.
"""

from __future__ import annotations

from assistant.analytics.service import (
    MAX_NOTE_LENGTH,
    NOT_A_TOOL,
    AnalyticsService,
    AnalyticsSummary,
    TimeBucket,
)

__all__ = [
    "MAX_NOTE_LENGTH",
    "NOT_A_TOOL",
    "AnalyticsService",
    "AnalyticsSummary",
    "TimeBucket",
]
