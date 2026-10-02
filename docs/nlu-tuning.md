# NLU Tuning Record

## Status

This is a **historical record of one tuning pass**, dated 2026-09-27. The
baseline figures below describe the suite as it stood on that date, not as it
stands today — the project has grown substantially since, and several later
sprints added both capabilities and tests.

The numbers here are deliberately left as they were measured. Changing them
would rewrite the evidence that justifies the shipped thresholds, which is
the one thing this document exists to preserve.

Every claim in it is asserted by a test in `tests/test_nlu_tuning.py`, so the
thresholds and the residual errors can be re-verified at any time by running
that file. The sections added after the tuning pass — the system-safety
closures and the analytics work — are later entries in the same record, and
are labelled with the phase they belong to.

For the current state of the project, see the project README.

## Date

2026-09-27

## Baseline

Freshly measured before any change was made:

```text
597 passed, 0 failed, 0 skipped
```

After this step:

```text
628 passed, 0 failed, 0 skipped   (597 original + 31 new)
```

The corpus, the evaluator and the sweeps live in
`tests/nlu_tuning_corpus.py` and `tests/test_nlu_tuning.py`. Every claim in
this document is asserted by a test in that file, so the numbers can be
reproduced by running the suite.

## Corpus

| | Count |
| --- | ---: |
| **Total utterances** | **88** |
| Group A, known-good commands | 25 |
| Group B, safety cases | 19 |
| Group C, natural language | 39 |
| Group D, near-ties | 5 |

Group A is **imported** from `tests/test_nlu_backward_compat.py` rather
than copied, so the two corpora cannot drift apart. Duplicates are dropped
by `_dedupe`, because `exit`, `note buy milk` and `play lofi beats` appear
in more than one group and counting them twice would have given them double
weight in the metrics.

The corpus is larger than the 60-70 originally suggested because the brief
itself enumerated roughly 40 Group C sentences on top of Groups A and B.
Every entry is one the brief named, or one the sweeps needed in order to be
meaningful. Nothing was added to pad it.

### Expected intent categories

| Label | Count |
| --- | ---: |
| a specific tool | 54 |
| `<no-match>` | 30 |
| `<ambiguous>` | 3 |

All nine registered tools are covered as expected outcomes.

### Labels record intent, not behaviour

Where the NLU disagreed with a label, the label was **not** changed to
make the number look better. One label was corrected because the label
itself was wrong: `play the song back` was originally marked
`<no-match>` as a suspected "inflected noun" near miss, but it is an
ordinary request to play a track in reverse, so it is relabelled
`youtube`. That correction reduced the error count by one and made the
corpus more accurate, not the code.

## MIN_CONFIDENCE

Swept with `AMBIGUITY_MARGIN` held at 0.05. Corpus of 88.

| Value | Correct | False Positive | Missed | Ambiguous | Errors |
| ----: | ------: | -------------: | -----: | --------: | -----: |
| 0.50 | 78 | 7 | 2 | 1 | 10 |
| 0.55 | 83 | 2 | 2 | 1 | 5 |
| 0.60 | 83 | 2 | 2 | 1 | 5 |
| 0.62 | 83 | 2 | 2 | 1 | 5 |
| **0.65** | **83** | **2** | **2** | **1** | **5** |
| 0.66 | 83 | 2 | 2 | 1 | 5 |
| 0.68 | 83 | 2 | 2 | 1 | 5 |
| 0.70 | 83 | 2 | 2 | 1 | 5 |
| 0.75 | 83 | 2 | 2 | 1 | 5 |
| 0.80 | 83 | 2 | 2 | 1 | 5 |
| 0.85 | 83 | 2 | 2 | 1 | 5 |
| 0.90 | 83 | 2 | 2 | 1 | 5 |

**Result: unchanged.** Every value from 0.55 to 0.90 produces identical
results, so 0.65 sits in the middle of a wide plateau rather than on a
cliff. Nothing in the corpus prefers any other value.

The only value that behaves differently is 0.50, and it is worse: the
legacy containment stage scores exactly 0.50, so a floor of 0.50 makes
containment actionable and false positives rise from 2 to 7. That is the
mechanism the floor exists to prevent, and it is why the value is above
0.50 at all. Raising the floor never removes a false positive in this
corpus; it only ever loses commands.

## AMBIGUITY_MARGIN

Swept with `MIN_CONFIDENCE` held at 0.65. Corpus of 88.

The first sweep of a corpus without Group D was **completely
uninformative**: every value from 0.00 to 0.20 gave identical results.
Measuring the reason showed that only one utterance had two actionable
candidates, and its gap was exactly 0.000. Group D was then added to make
the margin measurable at all.

| Value | Correct | False Positive | Missed | Unnecessary Ambiguity | Missed Ambiguity | Errors |
| ----: | ------: | -------------: | -----: | -------------------: | ---------------: | -----: |
| 0.00 | 82 | 2 | 2 | 0 | 2 | 6 |
| 0.01 | 82 | 2 | 2 | 0 | 2 | 6 |
| 0.02 | 82 | 2 | 2 | 0 | 2 | 6 |
| 0.03 | 82 | 2 | 2 | 0 | 2 | 6 |
| 0.04 | 82 | 2 | 2 | 0 | 2 | 6 |
| **0.05** | **83** | **2** | **2** | **1** | **0** | **5** |
| 0.06 | 83 | 2 | 2 | 1 | 0 | 5 |
| 0.07 | 83 | 2 | 2 | 1 | 0 | 5 |
| 0.10 | 82 | 2 | 2 | 2 | 0 | 6 |
| 0.15 | 82 | 2 | 2 | 2 | 0 | 6 |
| 0.20 | 82 | 2 | 2 | 2 | 0 | 6 |

**Result: unchanged.** 0.05 is at the minimum, tied with 0.06 and 0.07,
and both directions of travel are worse. The reason is structural rather
than accidental.

### Why 0.05 is the right value

The reachable gaps between two actionable candidates were enumerated by
crossing every single-word trigger in the real runtime lexicon with a set
of realistic typos. They fall into two families:

```text
0.000   two exact matches, both 1.000. A genuine tie.
0.048   1.000 exact against 0.952 fuzzy ("temprature" -> temperature)
0.077   1.000 exact against 0.923 fuzzy ("wether" -> weather)
0.091, 0.111, 0.143, 0.147, 0.200   further, weaker fuzzy matches
```

plus one family the probe did not reach but the design guarantees:

```text
0.050   1.000 exact against 0.950 phrase (PHRASE_SCORE)
```

`EXACT_SCORE` is 1.000 and `PHRASE_SCORE` is 0.950, so an exact match
against a phrase match is **always** exactly 0.050 apart, and real
sentences reach it: `note what did i ask` produces `notes` 1.000 exact
against `history` 0.950 phrase. That gap is the one the margin exists to
catch, and 0.05 sits exactly on it.

The margin therefore has a narrow window, and the current value is inside
it:

* Below 0.05 the exact-versus-phrase ties are silently decided instead of
  asked about. Lowering to 0.04 buys the 0.048 fuzzy case and loses two
  phrase ties, netting one error worse.
* Above about 0.077 a decisive exact-over-fuzzy win becomes a pointless
  question. Raising to 0.10 does exactly that.

The acceptable band is approximately `[0.050, 0.077)`, and 0.05 is the
value already shipped, chosen to sit exactly on the structurally
guaranteed phrase boundary.

### The one cost, stated plainly

`joke about temprature` (`jokes` 1.000 exact against `weather` 0.952
fuzzy, gap 0.048) is currently reported as **ambiguous** when it is
arguably clear. This is the price of catching the phrase ties. It is
recorded in the corpus with the label it ought to have, and it is a
deliberate trade rather than an oversight. The corpus cannot separate
these two effects, so no change is justified in either direction.

## FUZZY_CUTOFF

```text
Not tuned.

The existing cutoff is retained because nwes -> news
has a measured similarity of 0.750 and remains below the
configured fuzzy threshold by design.
```

Verified and asserted by `TestCurrentConfiguration` in
`tests/test_nlu_tuning.py`:

* `FUZZY_CUTOFF == 0.80`
* `SequenceMatcher(None, "nwes", "news").ratio()` rounds to `0.750`
* `0.750 < 0.80`
* `nwes` resolves to no intent at the shipped settings

The cutoff could not have been reached by this sweep in any case. The
smallest surviving fuzzy score in the corpus is 0.853 (`joks` to `jokes`),
and no candidate scores between 0.750 and 0.853, so no value of
`MIN_CONFIDENCE` could admit `nwes` while keeping `joks` without changing
the cutoff itself.

## Decision

```text
MIN_CONFIDENCE    = 0.65   unchanged
AMBIGUITY_MARGIN  = 0.05   unchanged
FUZZY_CUTOFF      = 0.80   unchanged (not tuned)
```

No constant was changed, because the measured corpus did not provide
sufficient evidence for a change to any of them.

* `MIN_CONFIDENCE` sits in a 0.55 to 0.90 plateau where every value is
  indistinguishable. Moving it would be a change with no measured
  justification.
* `AMBIGUITY_MARGIN` sits at the measured minimum, and both neighbouring
  directions of travel were shown to be worse for concrete, identified
  reasons rather than by preference.

Both values are now pinned by tests (`TestShippedConstants`), so a future
change has to be deliberate and will fail the suite until this record is
revisited.

## Residual errors

Five of 88 cases do not match their label. **None of them is reachable by
threshold tuning**, which is asserted by
`TestRemainingFailuresAreNotThresholdFixable`: every one scores either
exactly 1.000 or exactly 0.000, or is the margin trade-off above.

| Utterance | Outcome | Why it cannot be tuned away |
| --- | --- | --- |
| `I want to log this information` | resolves `information` at 1.000 | `information` is a real trigger present as a whole word. Separating a meta-mention from a request needs intent semantics, not a threshold. |
| `the weather is nice today I guess` | resolves `weather` at 1.000 | A statement about the weather, not a request for a report. Word-boundary matching cannot see the difference. |
| `quit the assistant` | no match | The system safety guard requires the trigger to be the **last** token. `assistant please quit` resolves correctly; `quit the assistant` does not. |
| `stop the assistant` | no match | Same guard. |
| `joke about temprature` | ambiguous at gap 0.048 | The documented margin trade-off above. |

The first two are the same underlying limitation: matching is lexical, so
a sentence that *mentions* a command word as a command word. This is the
same class as the `i noted that down` problem fixed in Step 4, but the
inflection guard cannot help when the word is already correctly inflected.
Closing it would need sentence-level intent modelling, a different piece
of work.

The third and fourth are a deliberate safety trade. The system guard is
what stops `goodbye is in the dictionary` from closing the application; it
also rejects `quit the assistant`. Relaxing it to allow the latter would
put the former back, and the brief forbids changing routing semantics in
this step.

## Phase 5 closure: the last two conversational cases

Steps 17, 20 and 22 each closed one family of false executions, using a
signal local to that family. Two cases survived all three and were left
for a closure pass:

```text
plans to quit            nothing to quit over
```

**No production change was made for them, because no reliable boundary
exists.** This is the measured reason, not a guess.

### What the two cases have in common

Both are a 1.000 exact match on `quit`, in the closing window, with no
self/third-party subject before the trigger, no definition marker, and
no want frame. Every condition in `_passes_system_guard` passes, so
`parse` returns `system` and the router dispatches `SystemTool`. The
parser is never given a reason to refuse; it reads a bare imperative.

The only structural feature they share is the one immediately before the
trigger: `to`. `"plans to quit"` is `plans → to → quit`, and
`"nothing to quit over"` is `nothing → to → quit → over`.

### Why "to" cannot be the fix

`to` is the obvious candidate and it does not survive contact with the
corpora. Three rules were measured against the pinned expectations:

| Candidate rule | Catches both | Breaks genuine commands | Breaks cases that must still parse |
| --- | --- | --- | --- |
| A: `to` immediately before the trigger | yes | none | **3 of 8** |
| B: nothing permitted after the trigger | **no** | **7 of 15** | none |
| C: word list `{plans, nothing}` | yes | none | none |

* **Rule A** refuses `time to quit`, `ready to quit`, `try to quit`,
  `attempt to exit`, `about to quit` and `go to exit` — all of which
  `test_nlu_system_want_frame_step22.py` pins as still parsing — and it
  would also newly refuse `I need to exit`, `I want to exit`,
  `are you ready to exit`, `can I ask you to exit`, `I want you to exit`
  and `I need you to quit`, all labelled `YES` in
  `nlu_system_tradeoff_corpus.py`. That is a straight trade of six real
  commands for two unsafe ones.
* **Rule B** does not even catch `plans to quit`, which has nothing after
  the trigger, and it breaks `goodbye assistant`, `shut down`,
  `you can quit now`, `would you quit now`, `can you shut down`,
  `would you exit now` and `should you quit now`.
* **Rule C** works only because it lists the two words in the two
  sentences. It is not a rule, it is the answer key, and it generalises
  to nothing.

### The precise reason there is no boundary

`plans to quit` and `time to quit` are identical on every feature the
guard actually reads:

```text
feature              plans to quit    time to quit
trigger_index        2                 2
token_before         to                to
in_tail              True              True
self_other_subject   False             False
definition_marker    False             False
want_frame           False             False
has_you              False             False
```

Token *count* differs between the two original cases — `nothing to quit
over` carries a trailing `over` — but nothing in the guard reads length,
and every feature that is read agrees.

The only real difference is the meaning of the token at index 0.
Telling `plans` (a speaker's private intention) from `time` (a noun)
requires knowing what the words *mean*, not how the sentence is
*shaped*. The scoring layer is deliberately a shape reader — it produces
numbers from tokens and is documented as containing no grammar. A rule
that separated these two would be the first genuinely semantic rule in
the layer, and it would be a two-word lookup table pretending to be one.

The honest options were therefore: ship rule C and mislabel a word list
as a structural fix, or ship rule A and break six commands. Neither was
acceptable, so the sprint stopped and recorded the limitation.

### What would actually close it

A subject or beneficiary analysis of the clause in front of the
infinitive — recognising that `plans` has an understood first-person
subject and `nothing` is a quantifier over no one. That is a parser
responsibility, not a scoring one, and it is the natural next increment
if these two sentences are ever worth refusing.

## Known limitations

All of these remain true. None was removed because the tuning corpus
happens not to cover it.

* **Lexical matching cannot separate a mention from a request.**
  `I want to log this information` and `the weather is nice today I guess`
  both resolve to a tool at 1.000. Newly recorded by this step.
* **The system guard rejects natural exit phrasings.** `quit the
  assistant` and `stop the assistant` do not resolve; only a trailing
  `exit`/`quit`/`goodbye` does. Newly recorded by this step.
* **Vocabulary gaps.** `tell me something funny`, `show me my previous
  commands`, `what have I asked you`, `remember that I need milk`, `what
  do you know about machine learning` and `I want to watch something` are
  all ordinary requests the lexicon does not cover. They are labelled
  `<no-match>` in the corpus and recorded here rather than added to
  `COMMON_ALIASES`, because that table is deliberately restricted to
  phrases that name an action.
* **Two conversational system intents are still unsafe and cannot be
  closed without breaking correct behaviour.** `plans to quit` and
  `nothing to quit over` still reach `SystemTool`. Newly recorded by
  the Phase 5 closure sprint; the reasoning is in
  [Phase 5 closure](#phase-5-closure-the-last-two-conversational-cases)
  below. Both are pinned as *known-unsafe* rather than dropped, so the
  next person to see them knows they were measured and not missed.
* **Single-shot clarification.** An unusable reply to "Did you mean ...?"
  returns the "didn't understand" wording and does not re-ask.
* **`ask_follow_up` still bypasses the NLU**, using `router.dispatch` with
  a hard-coded phrase. Unchanged from Step 5.
* **Aliases are hand-maintained** in `COMMON_ALIASES`, and a new tool gets
  no aliases at all.
* **Fuzzy limitations.** A genuine typo that *appends* a character is not
  recovered, by the documented choice in Step 4.
* **`nwes` is permanently unrecognised**, by design, at 0.750 similarity.
* **The corpus is synthetic.** It was written by the author of the code,
  so it reflects assumptions about what users say rather than a real
  transcript. The clearest evidence of that is the corpus bug this step
  found and corrected (`play the song back`).
* **`MIN_CONFIDENCE` is insensitive over a wide range.** Being on a broad
  plateau is reassuring for stability, but it means the value carries
  little behavioural information: it is a guard rail, not a tuned
  parameter.
* **The margin has a narrow valid window**, roughly `[0.050, 0.077)`. That
  is a small safety buffer for future scoring changes, and it is the main
  thing to re-measure if `PHRASE_SCORE` ever moves.

## How to reproduce

```text
python -m pytest tests/test_nlu_tuning.py -q
```

The suite asserts the corpus shape, the shipped values, that both sweeps
find the shipped value optimal, that `nwes` stays rejected, and that the
five residual errors are exactly the ones documented above. If a future
change to the scoring layer makes any of those fail, this record is the
thing to update alongside it.
