# Virtual Assistant

A voice- and text-driven assistant written in Python, with a spoken natural
language understanding layer, SQLite persistence, and a read-only web
dashboard over its own command history.

```
main.py                 launcher
assistant/
  app.py                orchestration: greeting, command loop, startup
  config.py             the only module that reads environment variables
  core/                 database, router, tool contract, shared context
  nlu/                  normalize, lexicon, scoring, parser, framing
  storage/              typed reads over the SQLite tables, notes CRUD
  analytics/            read-only aggregation over interactions and notes
  dashboard/            optional read-only web dashboard
  tools/                one module per user-facing capability
tests/                  the full suite
data/assistant.db       local SQLite history (git-ignored)
```

## Running the assistant

```bash
pip install -r requirements.txt
python main.py            # voice
python main.py --text     # typed instead of spoken
```

API keys are read from the environment or a local `.env` file. See
`assistant/config.py`; nothing prints or logs a key.

## Analytics Dashboard

A small local web dashboard that charts the assistant's own usage. It reads
the same SQLite history the assistant writes, and every figure on the page is
produced by `assistant/analytics/service.py` — the same service that answers
"show my analytics" when you ask the assistant in person. The dashboard adds
no analysis of its own, so the page and the spoken summary can never disagree.

**It is read-only.** The database is opened with SQLite's `mode=ro`, so the
engine itself rejects any write, and the wrapper the dashboard uses exposes
only `query_one` and `query_all` — there is no `execute` to call. Nothing in
the request path can modify your history. Note *contents* are never rendered:
the page shows counts and lengths only.

### Install

```bash
pip install -r requirements.txt
```

FastAPI, Uvicorn and Jinja2 are only needed for the dashboard. The assistant
and its test suite do not import them, so they can be omitted if you never
open the dashboard.

### Launch

```bash
python -m assistant.dashboard.app
```

Then open <http://127.0.0.1:8765>.

The server binds to loopback only, because this sprint has no authentication
and the dashboard shows a user's own command history.

### What it shows

| Section | Contents |
| --- | --- |
| Header | Reporting range, plus a from/to date filter |
| KPI cards | Total interactions, unique tools, most used tool, match rate, total notes |
| Top tools | Up to five tools with counts and share of the range |
| Activity | Daily and weekly interaction trends, drawn as SVG bar charts |
| Peaks | Busiest day, busiest week, active days, unmatched commands, first and most recent command |
| Notes | Total, average and longest length, days with notes, first and latest timestamps |

A metric that has no value — a match rate with nothing measured, an average
over no notes — is shown as a dash rather than a zero, because "not measured"
and "measured as zero" are different facts. With no data at all the page shows
an empty state explaining what to do next.

### API

| Endpoint | Returns |
| --- | --- |
| `GET /` | The HTML dashboard |
| `GET /api/dashboard` | The same data as JSON |
| `GET /api/health` | `{"status": "ok", "service": "assistant-dashboard", "read_only": true}` |
| `GET /api/docs` | Interactive OpenAPI documentation |

`/api/dashboard` accepts optional `start` and `end` query parameters in ISO
`YYYY-MM-DD` form, both inclusive:

```bash
curl "http://127.0.0.1:8765/api/dashboard?start=2026-09-01&end=2026-09-30"
```

A malformed or reversed range returns `400` with a readable `detail` string.
Dates are never silently swapped, because doing so would return a confident
answer to a different question. On the HTML page the same mistake is reported
in a banner and the page falls back to all available data.

## Tests

```bash
python -m pytest
```

The suite covers the NLU layer, the storage layer, the analytics layer and the
dashboard. No test opens the real `data/assistant.db`; every test that needs
storage points at a temporary database.

## Database

SQLite, standard library only — no ORM. The schema is at version 2 with two
tables: `interactions` (one row per handled command) and `notes`. Migrations
are additive and recorded in `schema_version`, so an existing database
upgrades in place without data loss.
