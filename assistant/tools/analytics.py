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

from datetime import date
from typing import Any, Sequence

from assistant.analytics import AnalyticsService, AnalyticsSummary, DateRange
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


def _times(count: int) -> str:
    """Return ``"time"`` or ``"times"`` to agree with ``count``.

    "1 times" is plainly wrong when spoken, and this tool exists to speak.
    Every count that reaches a sentence goes through here, so the grammar is
    decided in one place rather than at each of the several call sites.
    """
    return "time" if count == 1 else "times"


#: Reported modes. Each phrase is checked against these, longest first, so a
#: more specific request wins over the general summary.
MODE_TODAY = "today"
MODE_RECENT = "recent"
MODE_MOST_USED = "most_used"
MODE_SUMMARY = "summary"

#: Phrase fragments that select a mode. Every entry is at least two words, so
#: no single generic word can select a mode on its own. The order matters and
#: is deliberate: a longer, more specific phrase is listed before the shorter
#: one it contains, which is what makes "what do I use most today" a
#: most-used question rather than a today question.
_MODE_PHRASES: tuple[tuple[str, str], ...] = (
    ("what do i use most today", MODE_MOST_USED),
    ("what do i use most", MODE_MOST_USED),
    ("most used command today", MODE_MOST_USED),
    ("most used command", MODE_MOST_USED),
    ("my most used command", MODE_MOST_USED),
    ("which command do i use most", MODE_MOST_USED),
    ("today's activity", MODE_TODAY),
    ("todays activity", MODE_TODAY),
    ("what did i do today", MODE_TODAY),
    ("commands today", MODE_TODAY),
    ("recent activity", MODE_RECENT),
    ("last 7 days", MODE_RECENT),
    ("last seven days", MODE_RECENT),
    ("last 30 days", MODE_RECENT),
    # "this week" is deliberately absent. Sprint 2 measured it: it is an
    # 8-character pattern, so it outranks the weather alias "is it going to
    # rain" inside "is it going to rain this week" and turned that genuine
    # weather question into an ambiguous match. A time window is not specific
    # enough on its own to claim an utterance.
    ("usage summary", MODE_SUMMARY),
)

#: How many days the ``recent`` mode covers. One week is long enough to show
#: a habit and short enough that every day in it is worth speaking about.
RECENT_DAYS = 7


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

        The mode phrases are included so a routed utterance can be
        interpreted, but they never claim an utterance on their own merit:
        every one contains a word specific enough that no other tool competes
        for it.

        The list is de-duplicated while preserving order, because a phrase can
        legitimately appear in both halves -- "what do i use most" is both a
        base trigger and a mode selector -- and a duplicate pattern would make
        the help text list the same example twice.
        """
        phrases = [
            "show my analytics",
            "show analytics",
            "usage statistics",
            "how many commands have i used",
            "what do i use most",
            "my activity summary",
        ]
        phrases += [phrase for phrase, _mode in _MODE_PHRASES]

        seen: set[str] = set()
        unique: list[str] = []
        for phrase in phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique.append(phrase)
        return unique

    def extract_slots(self, utterance: str, ctx: AppContext) -> dict[str, Any]:
        """Work out which report the user asked for.

        Returns a dict with a ``mode`` key, defaulting to
        :data:`MODE_SUMMARY`. ``ctx`` is unused, as in the base class.
        """
        text = (utterance or "").lower()
        for phrase, mode in _MODE_PHRASES:
            if phrase in text:
                return {"mode": mode}
        return {"mode": MODE_SUMMARY}

    def execute(
        self, ctx: AppContext, slots: dict[str, Any] | None = None
    ) -> list[str]:
        if ctx.db is None:
            return ["Sorry, I have no usage data to show right now."]

        mode = (slots or {}).get("mode", MODE_SUMMARY)
        try:
            service = AnalyticsService(ctx.db)
            summary = service.summary()
            if not summary.has_activity:
                return [NO_ACTIVITY]
            if mode == MODE_TODAY:
                return self._render_today(service)
            if mode == MODE_MOST_USED:
                return self._render_most_used(service)
            if summary.unique_tools == 0:
                return [NO_MATCHED_ACTIVITY]
            if mode == MODE_RECENT:
                return self._render_recent(service)
            return self._render(service, summary)
        except Exception as exc:  # noqa: BLE001 - analytics is never critical
            return [f"Sorry, I could not work out my usage right now. {exc}"]

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------
    def _render_today(self, service: AnalyticsService) -> list[str]:
        """Report only what happened on the current day.

        "Today" is the one figure that genuinely depends on the clock, so it
        is the one place this tool reads the system date. The report is
        whatever the stored rows say; nothing is assumed about which timezone
        the user is in.
        """
        today = date.today()
        window = DateRange.between(today, today)
        total = service.total_interactions(window)
        if total == 0:
            return [f"You have not used me today, {today.isoformat()}."]

        lines = [
            f"Today, {today.isoformat()}, you have made {total} command"
            f"{'' if total == 1 else 's'}."
        ]
        top = service.most_used_command(window)
        if top:
            lines.append(
                f"The most used was {top.tool_name}, {top.count} {_times(top.count)}."
            )
        return lines

    def _render_most_used(self, service: AnalyticsService) -> list[str]:
        """Report the top few tools and nothing else.

        This answers "what do I use most" with a ranking rather than the full
        summary. Unresolved commands are excluded by
        :meth:`AnalyticsService.top_tools`, since "none" is not a feature.
        """
        top = service.top_tools(TOP_DAYS)
        if not top:
            return [NO_MATCHED_ACTIVITY]
        lines = ["Here is what you use most:"]
        for count in top:
            lines.append(f"{count.tool_name}, {count.count} {_times(count.count)}.")
        return lines

    def _render_recent(self, service: AnalyticsService) -> list[str]:
        """Report the last :data:`RECENT_DAYS` days.

        The window is anchored on today and built with
        :meth:`DateRange.last_days`, so it is explicit rather than a hidden
        "30 days" the caller never asked for.
        """
        window = DateRange.last_days(RECENT_DAYS, date.today())
        total = service.total_interactions(window)
        if total == 0:
            return [f"You have not used me in the last {RECENT_DAYS} days."]

        lines = [
            f"In the last {RECENT_DAYS} days you have made {total} command"
            f"{'' if total == 1 else 's'}."
        ]
        top = service.most_used_command(window)
        if top:
            lines.append(
                f"You use {top.tool_name} most, {top.count} {_times(top.count)}."
            )
        return lines

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
                f"{summary.most_used_count} {_times(summary.most_used_count)}."
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

