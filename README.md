# Virtual Assistant

A modular Python assistant that began as a voice-driven demo and grew into a
small automation and analytics platform. You can talk to it, type to it, and
then look at what it has been doing — in a browser, over a read-only web
dashboard, using the same analytics service the assistant itself answers from.

Everything is standard library where it can be: SQLite through `sqlite3`, the
NLU through hand-written rules rather than a model, and the analytics through
plain SQL. There is no ORM, no data-science framework, and no cloud service
anywhere in the request path.

---

## Table of contents

- [Key capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Optional AI responses](#optional-ai-responses)
- [Running it](#running-it)
- [The dashboard](#the-dashboard)
- [Testing](#testing)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [What makes this more than a voice assistant](#what-makes-this-more-than-a-voice-assistant)

---

## Key capabilities

**Interfaces**
- **Voice mode** — spoken input through `SpeechRecognition`, spoken output
  through `pyttsx3`, with follow-up questions handled without the tool layer
  ever touching the microphone.
- **Text mode** — the same assistant typed, for when a microphone is not
  available or wanted.

**Understanding**
- **Natural language understanding** — normalisation, a lexicon, weighted
  scoring and a parser, producing a confidence and a runner-up so genuinely
  ambiguous requests are reported rather than guessed.
- **Intent detection** — ten registered capabilities: `weather`, `news`,
  `facts`, `jokes`, `information`, `youtube`, `analytics`, `history`,
  `notes`, `system`.
- **Tool routing** — a deterministic router that resolves an intent to a tool
  and formats the reply.

**Capabilities**
- Weather, news, jokes and facts, Wikipedia-style information lookup, and
  YouTube automation via Selenium.
- **Notes** — add, list and delete short text notes.
- **History** — recent commands and a usage summary.
- **Analytics** — interaction totals, per-tool usage, daily and weekly
  activity, peaks, and notes analytics.
- **System** — `exit` / `quit`, held behind the strictest safety rules in the
  project.

**Persistence and insight**
- **SQLite** — every handled command and every note, schema-versioned with
  additive migrations.
- **AnalyticsService** — one read-only layer that both the spoken summary and
  the dashboard read from, so the two can never disagree.
- **Read-only dashboard** — a local FastAPI + Jinja2 web dashboard with vanilla
  SVG charts.

---

## Architecture

```text
              ┌──────────────────────────────┐
              │        User                  │
              └───────────────┬──────────────┘
                              │
              ┌───────────────▼──────────────┐
              │  Voice  /  Text  interface    │   speech/, app.py
              └───────────────┬──────────────┘
                              │
              ┌───────────────▼──────────────┐
              │     Normalization + NLU       │   nlu/
              │  normalize → score → parse    │
              │  → confidence → framing       │
              └───────────────┬──────────────┘
                              │  ParsedIntent
              ┌───────────────▼──────────────┐
              │      Intent resolution       │   core/router.py
              └───────────────┬──────────────┘
                              │
              ┌───────────────▼──────────────┐
              │   Tool router + capabilities │   tools/
              └───────┬───────────────┬───────┘
                      │               │
          ┌───────────▼──────┐  ┌────▼─────────────┐
          │  SQLite (write)  │  │ external services │
          │  interactions    │  │ weather, news,    │
          │  notes           │  │ wikipedia, yt     │
          └───────────┬──────┘  └───────────────────┘
                      │
          ┌───────────▼──────────────┐
          │      AnalyticsService     │   analytics/ (read-only)
          └───────────┬───────────────┘
                      │
          ┌───────────▼──────────────┐
          │  Read-only dashboard      │   dashboard/  FastAPI
          └──────────────────────────┘
```

The important arrow is the one into `AnalyticsService`: the dashboard reads
the same tables the assistant writes, through the same service, and it can
never write.

**The optional AI branch.** When intent resolution finds no command at all, the
assistant *may* ask an AI service for a short conversational reply:

```text
   no command recognised
              │
   ┌──────────▼───────────┐
   │  AI response layer   │   ai/   OPTIONAL, off by default
   │  text only, no tools │   ┌──────────────────────────────┐
   └──────────┬───────────┘   │ disabled / no key / failure │
              │               │   → the normal "didn't        │
              └── reply ──────┤     understand" reply         │
                              └──────────────────────────────┘
```

It sits *after* the NLU and the router, so a recognised command never reaches
it, and it can only ever produce text. See
[Optional AI responses](#optional-ai-responses).

---

## Project structure

```text
main.py                    launcher; all runtime logic lives in assistant/app.py
requirements.txt           pinned runtime, dashboard and test dependencies

assistant/
  app.py                   orchestration: greeting, command loop, startup
  config.py                the only module that reads environment variables
  core/
    context.py             the AppContext every tool receives
    database.py            SQLite foundation and additive migrations
    router.py              deterministic tool matching and dispatch
    tool.py                the Tool contract every capability implements
  nlu/
    normalize.py           text to tokens
    lexicon.py             trigger vocabulary, built from registered tools
    scoring.py             candidate scoring and the system-safety guard
    parser.py              ranked candidates to one ParsedIntent
    framing.py             sentence framing (request / mention / directive)
  speech/                  text and voice input/output implementations
  storage/
    repositories.py        typed reads over `interactions`
    notes.py               note CRUD over `notes`
  ai/                      optional AI layer, off unless configured
    base.py                the AIProvider abstraction (text in, text out)
    layer.py               AIResponder: availability rules and the fallback
    openai_compatible.py   the one bundled provider
  tools/                   one module per user-facing capability
  analytics/
    service.py             read-only aggregation; the single source of truth
  dashboard/
    app.py                 read-only SQLite access and the ASGI app
    routes.py              routes and the JSON payload
    templates/             Jinja2 templates
    static/                stylesheet and the vanilla SVG chart script

tests/                     the automated suite (see Testing)
docs/nlu-tuning.md         the NLU tuning record and its evidence
data/                      local SQLite database (git-ignored)
```

**The dependency direction is one-way.** `core` knows nothing about `tools`;
`tools` know nothing about `analytics`; the dashboard's presentation layer
contains no SQL at all. That is what lets the analytics layer be tested, and
reused, without importing a single tool.

---

## Setup

**Requirements:** Python 3.12 on Windows (voice and screenshot features use
Windows-specific libraries). The non-voice paths, the tests and the dashboard
run on other platforms as well.

```bash
git clone https://github.com/Ayeshafirdous5/Virtual-Assistant.git
cd Virtual-Assistant
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

`requirements.txt` pins everything, in three groups: voice I/O and browser
automation, the HTTP client and helper libraries the tools need, and an
optional block for the dashboard. The dashboard's three packages are needed
only if you intend to open it.

### Optional API keys

Two tools need a key. Everything else works without one, and the assistant
degrades gracefully rather than failing.

```bash
copy .env.example .env         # Windows PowerShell
# cp .env.example .env         # macOS / Linux
```

`.env.example` lists the two variables:

| Variable | Needed for | Without it |
| --- | --- | --- |
| `NEWS_API_KEY` | the news tool | News is unavailable; nothing else changes |
| `OPENWEATHER_API_KEY` | the weather tool | Weather is unavailable; nothing else changes |

`.env` is git-ignored and must never be committed. `assistant/config.py` is the
only module that reads the environment, and it never prints or logs a key.

---

## Optional AI responses

**Off by default.** With nothing configured the assistant behaves exactly as it
does today, and never contacts an AI service.

When the NLU decides an utterance simply is not a command, the assistant can
optionally ask an AI service for a short conversational reply instead of
saying it did not understand.

```bash
# .env
AI_ENABLED=true
AI_API_KEY=your_ai_api_key_here
AI_MODEL=gpt-4o-mini
AI_BASE_URL=https://api.openai.com/v1
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_ENABLED` | `false` | Master switch. Nothing happens unless this is true. |
| `AI_API_KEY` | *(none)* | Credential. Required, and never logged or stored. |
| `AI_MODEL` | `gpt-4o-mini` | Model to request. |
| `AI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint. |
| `AI_TIMEOUT` | `8` | Seconds to wait before falling back. |
| `AI_MAX_TOKENS` | `200` | Upper bound on the reply length. |

### What happens when it is off, or fails

In every one of these cases the assistant says its normal
*"I'm sorry, I didn't understand"* reply, exactly as before, and never
crashes:

- `AI_ENABLED` is false or unset
- `AI_API_KEY` is missing or blank
- the provider is unreachable
- the request times out
- the provider returns an error or an unusable reply
- the provider raises anything at all

### Command routing stays rule-based

This does **not** make the assistant an AI agent.

- The NLU, router and tools are unchanged and always run first.
- A recognised command is **never** sent to the AI layer.
- Input the framing guards reject is **never** sent to the AI layer.
- The AI layer is handed a string and returns a string. It is never given the
  router, a tool, the speaker or the database, so it **cannot** run a tool,
  a command, or anything else. Its output is spoken and never executed.

### Design

`assistant/ai/base.py` defines `AIProvider`, a one-method abstraction: a string
in, a string out. `assistant/ai/layer.py` (`AIResponder`) is the only thing the
application depends on, and its `respond()` is written never to raise.
`assistant/ai/openai_compatible.py` is the one bundled provider; it reuses
`requests`, which is already a dependency, and adds none.

Swapping in a different provider means subclassing `AIProvider` and changing
one factory function. No other code moves.

---

## Running it

```bash
python main.py                  # voice mode
python main.py --text           # type commands instead
python main.py --text --debug-nlu   # type, showing an NLU diagnostic per command
python main.py --help           # all options
python -m assistant.dashboard.app   # the read-only dashboard
```

In text mode, try:

```text
weather
tell me a joke
note buy milk
notes
history
show my analytics
what do I use most
exit
```

---

## The dashboard

```bash
python -m assistant.dashboard.app
```

Then open **http://127.0.0.1:8765**.

It shows:

| Section | Contents |
| --- | --- |
| **Overview** | Total interactions, unique tools, most used tool, match rate, total notes |
| **Usage** | Top tools with counts and share, plus activity peaks |
| **Activity** | Daily and weekly interaction trends, drawn as SVG bar charts |
| **Notes** | Total, average and longest length, days with notes, first and latest timestamps |

A from/to date filter narrows every figure. A malformed or reversed range is
refused with an explanation rather than silently corrected, because swapping
two dates would answer a different question than the one you asked.

The page is arranged in four reading tiers and works from a 390px phone to a
wide desktop. Every figure is also present as text, so it stays complete and
readable with JavaScript disabled: each chart carries a fallback table with the
same numbers.

**It is read-only.** The database is opened with SQLite's `mode=ro`, so the
engine itself refuses a write, and the wrapper the dashboard uses exposes only
`query_one` and `query_all` — there is no `execute` method to call. Note
contents and the utterances you typed are never rendered; only counts, tool
names and timestamps.

**It has no authentication**, so it binds to loopback only.

### JSON API

| Endpoint | Returns |
| --- | --- |
| `GET /` | the HTML dashboard |
| `GET /api/dashboard` | the same data as JSON, accepting `start` and `end` |
| `GET /api/health` | `{"status": "ok", "service": "assistant-dashboard", "read_only": true}` |
| `GET /api/docs` | interactive OpenAPI documentation |

```bash
curl "http://127.0.0.1:8765/api/dashboard?start=2026-09-01&end=2026-09-30"
```

---

## Testing

```bash
python -m pytest              # the whole suite
python -m pytest -q           # quieter
python -m pytest tests/test_nlu_parser.py    # one module
```

The repository includes a comprehensive automated test suite covering the
NLU layer, the storage layer, the analytics layer and the dashboard, along with
the corpora that pin each safety fix. It is deliberately more than the code
needs: several behaviours are *measured* rather than assumed, and a test holds
the measurement so it cannot quietly drift.

**No test touches the real database.** Anything needing storage points at a
temporary one; the `data/assistant.db` in your checkout is never opened by the
suite.

CI runs the same suite on every push and pull request.

---

## Design decisions

**The NLU is rules-first, deliberately.** No model, no embeddings, no
inference at runtime. Every decision is inspectable and every safety rule is a
named condition with a test beside it. For a system that can terminate the
process, a rule you can read is worth more than a score you cannot.

**Ending the assistant is the hard problem, so it got the strictest rules.**
`exit` and `quit` are only honoured when the sentence is genuinely an order.
Three layers of evidence each closed a family of false executions: a
first/third-person subject check, a definition-frame check (`what does quit
mean`), and a want-frame check (`do you want to quit`). Each is a named
condition in `_passes_system_guard`, and each has tests that fail if it is
undone.

**Ambiguity is reported, not resolved by guessing.** The parser exposes its
runner-up and a confidence, so a genuinely ambiguous sentence is surfaced
instead of silently dispatched.

**Analytics never invents a metric.** Every figure is a fact the schema can
support. Where a metric is undefined — a match rate over nothing, an average
over no notes — it is shown as unavailable rather than as a zero, because "not
measured" and "measured as zero" are different things. Where the schema cannot
answer a question at all, such as an "active notes" count when there is no
status column, the limitation is documented instead of approximated.

**The dashboard cannot write.** Not by convention and not by review — by
construction. `mode=ro` means the SQLite engine rejects the write, and the
access wrapper exposes no method that could issue one.

**Failures degrade, they do not cascade.** A tool returns a sentence when its
network call, API key or database misbehaves, so one broken integration can
never stop the assistant answering anything else.

---

## Limitations

Stated plainly, because a project that lists no limitations is not telling you
the truth.

- **The NLU is rule-based and bounded.** It handles the phrasings in its
  corpora well and degrades outside them. Two conversational sentences —
  `plans to quit` and `nothing to quit over` — are still misread as commands,
  and are documented as such rather than papered over.
- **`if you want to quit now` is refused.** A genuine conditional order that the
  safety guard cannot distinguish from a question. Refusing one retry is
  cheaper than acting on the wrong reading.
- **Voice mode is Windows-oriented.** `pyttsx3` and the speech stack are
  configured for SAPI5.
- **YouTube automation needs a local browser and driver.** It is the least
  portable feature in the project.
- **The dashboard has no authentication** and binds to loopback only.
- **Notes have no lifecycle.** No archived, done or active state exists, so no
  such figure is reported.
- **Weekly activity is never zero-filled**, and daily gaps are omitted unless a
  range is explicitly requested — a day with no commands is a day the assistant
  was not running, not a measured zero.
- **Analytics are read-only by design**, so there is no charting of live or
  streaming data.

---

## What makes this more than a voice assistant

Most voice-assistant projects stop once the assistant answers a question. The
interesting part of this one is everything built *around* that:

- **Safety is treated as an engineering problem, not a precaution.** Ending the
  process is the one action that cannot be undone, so it earned the deepest
  analysis in the codebase — three distinct families of false execution, each
  diagnosed by measuring the failure, closed by a named condition, and pinned
  by tests. The reasoning behind each is written down.

- **The NLU was tuned against a corpus, not a hunch.** Thresholds were chosen
  by sweeping them and observing where behaviour actually changes, and the
  residual errors are listed instead of hidden. The record is in
  [`docs/nlu-tuning.md`](docs/nlu-tuning.md).

- **One analytics layer serves both consumers.** The spoken summary and the
  dashboard read the same service, so a number cannot differ between what the
  assistant says and what the page shows.

- **The dashboard is genuinely read-only, and provably so.** The guarantee
  lives in the database connection, not in a code-review promise, and there
  are tests that assert both the wrapper's shape and the engine's refusal.

- **Honesty is designed into the output.** Undefined metrics render as
  unavailable rather than zero, empty states say what to do next instead of
  showing zeroes, and there are three different empty states because three
  different situations genuinely need different advice.

- **The interface degrades honestly.** A missing API key, an unreachable
  service or a broken database becomes a sentence, never a crash.

- **It was verified, not assumed.** The dashboard was inspected in a real
  browser at four viewport widths, which caught a broken stylesheet and a
  JavaScript syntax error that every static check had passed.