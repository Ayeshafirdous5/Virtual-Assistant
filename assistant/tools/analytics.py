"""Report how much the assistant has been used and what it is used for.

Reads the ``interactions`` and ``notes`` tables through
:class:`~assistant.analytics.service.AnalyticsService` and renders a short
spoken summary. The tool holds no SQL and no aggregation logic of its own: it
asks the service for an :class:`~assistant.analytics.service.AnalyticsSummary`
and formats it.

Every number comes from a real query. There are no placeholder or demo
figures, and a command that matched no tool is counted as such rather than
being quietly dropped from the totals.

The tool never touches ``ctx.speaker`` or ``ctx.listener``: it returns lines
and the app speaks them. A missing or broken database becomes a sentence, so
analytics can never stop the assistant.
"""

from __future__ import annotations

from typing import Any, Sequence

from assistant.analytics import AnalyticsService, AnalyticsSummary
from assistant.core.context import AppContext
from assistant.core.tool import Tool

#: How many days of the daily breakdown to include. Long enough to show a
#: pattern, short enough to stay speakable.
TOP_DAYS = 5

#: Wording used when nothing has been recorded yet.
NO_ACTIVITY = "You have not used me yet, so I have no usage data to show."

#: Wording used when every recorded command failed to match a tool.
NO_MATCHED_ACTIVITY = (
    "You have not asked me anything I recognised yet, so I have no usage data."
)

#: The length of the stored timestamp format, "YYYY-MM-DD HH:MM:SS".
TIMESTAMP_LENGTH = 19


class AnalyticsTool(Tool):
    """Summarise recorded usage."""

    name = "analytics"
    description = "Show usage statistics and an activity summary."

    def patterns(self) -> list[str]:
        """Return the trigger phrases.

        Deliberately a small, explicit set rather than a family of loose
        aliases. "analytics" and "statistics" appear in no other tool, so
        there is no competition for them. The list avoids "summary" on its
        own, which is far too generic to claim, and avoids a bare "how many",
        which belongs to the weather aliases.
        """
        return [
            "show my analytics",
            "show analytics",
            "usage statistics",
            "how many commands have i used",
            "what do i use most",
            "my activity summary",
        ]

    def execute(
        self, ctx: AppContext, slots: dict[str, Any] | None = None
    ) -> list[str]:
        if ctx.db is None:
            return ["Sorry, I have no usage data to show right now."]

        try:
            service = AnalyticsService(ctx.db)
            summary = service.summary()
            if not summary.has_activity:
                return [NO_ACTIVITY]
            if summary.unique_tools == 0:
                return [NO_MATCHED_ACTIVITY]
            return self._render(service, summary)
        except Exception as exc:  # noqa: BLE001 - analytics is never critical
            return [f"Sorry, I could not work out my usage right now. {exc}"]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(
        self, service: AnalyticsService, summary: AnalyticsSummary
    ) -> list[str]:
        """Turn a summary into speakable lines.

        Kept separate from :meth:`execute` so the wording can be tested
        against a hand-built summary, with no database behind it.
        """
        lines = [self._headline(summary)]

        if summary.most_used_tool:
            lines.append(
                f"You use {summary.most_used_tool} most, "
                f"{summary.most_used_count} times."
            )

        unmatched = service.unmatched_count()
        if unmatched:
            lines.append(
                f"{unmatched} command{'' if unmatched == 1 else 's'} "
                "did not match anything I recognise."
            )

        first = _readable_timestamp(summary.first_interaction)
        last = _readable_timestamp(summary.last_interaction)
        if first and last:
            lines.append(
                f"Your first command was on {first}, the latest on {last}."
            )
        elif first or last:
            lines.append(
                f"The only command I have on record was on {first or last}."
            )

        days = service.daily_interaction_counts(limit=TOP_DAYS)
        if len(days) > 1:
            lines.append(self._daily_sentence(days))
        elif days:
            lines.append(f"All of that happened on {days[0].period}.")

        lines.append(self._notes_sentence(service, summary.total_notes))
        return lines

    def _headline(self, summary: AnalyticsSummary) -> str:
        """The opening line, carrying the headline numbers."""
        return (
            f"You have made {summary.total_interactions} command"
            f"{'' if summary.total_interactions == 1 else 's'} "
            f"across {summary.unique_tools} tool"
            f"{'' if summary.unique_tools == 1 else 's'} "
            f"over {summary.active_days} day"
            f"{'' if summary.active_days == 1 else 's'}."
        )

    def _daily_sentence(self, days: Sequence[Any]) -> str:
        """Render the recent-days breakdown, busiest first.

        Ordering is by count descending then date descending, so the busiest
        day is named first and equal counts resolve to the most recent day.
        The service returns days chronologically; this is presentation order
        only, and it is a total order on (count, period), so repeated calls on
        unchanged data always produce identical wording.
        """
        ordered = sorted(days, key=lambda bucket: (bucket.count, bucket.period))
        detail = ", ".join(
            f"{bucket.period} with {bucket.count}" for bucket in reversed(ordered)
        )
        return f"By day: {detail}."

    def _notes_sentence(self, service: AnalyticsService, total: int) -> str:
        """Report the note count, and when the notes were added."""
        if not total:
            return "You have not saved any notes."

        sentence = f"You have {total} note{'' if total == 1 else 's'}."
        note_days = service.daily_note_counts()
        if len(note_days) > 1:
            sentence += (
                f" The first was on {note_days[0].period}"
                f" and the latest on {note_days[-1].period}."
            )
        elif note_days:
            sentence += f" They were all added on {note_days[0].period}."
        return sentence


def _readable_timestamp(value: str | None) -> str | None:
    """Return just the date part of a stored timestamp.

    The stored format is ``"YYYY-MM-DD HH:MM:SS"`` in UTC. Spoken output
    gains nothing from the clock time, and dropping it keeps the sentence
    short and avoids implying a local time that was never recorded. A value
    of an unexpected shape is returned unchanged rather than mangled into a
    wrong-looking date.
    """
    if not value:
        return None
    if len(value) == TIMESTAMP_LENGTH and value[4] == "-" and value[7] == "-":
        return value[:10]
    return value

