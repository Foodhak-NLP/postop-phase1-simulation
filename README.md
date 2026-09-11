# Post-op Phase 1 — end-to-end simulation

A post-op patient pushed through the **real** Layer 1-4 code and section 10's
gate, **one day at a time, for up to six weeks**. Every day gives an insight and
a recommendation from each layer.

```bash
# The app. --server.port only matters if something else already holds 8501.
../Postop-Phase1/Staging/phase1/bin/streamlit run app.py --server.port 8502

# The same simulation, in a terminal.
../Postop-Phase1/Staging/phase1/bin/python run_simulation.py
```

The port is passed on the command line rather than set in
`.streamlit/config.toml`, because that file is committed: a port pinned there
would follow the app to any host and make it start somewhere the platform is
not looking.

No clinical logic lives here. The app imports `Postop-Phase1/Staging/*` and
calls `run_phase1()`, so every threshold, matrix and rule on screen is the
service's. Move a number there and it moves here on the next reload.

## Driving it

The sidebar holds the patient and nothing else: **surgery type, age, sex,
height, weight, allergens, dietary restrictions, existing conditions** and the
**Foodhak user id**. Weight sets Layer 3's per-kilogram targets; age, sex and
height are read only by the recipe pool, to pick a demographic band; the user
id is passed to the pool and read by no layer.

The **stay** is the six weeks Phase 1 covers. There is nothing to set: the
calendar counts post-op days, so a date would only be a label.

The calendar sits on the left and that day's **inputs** sit beside it. Click a
day and every tab below shows that morning's insights, recommendations and
recipes. The grid is six rows of seven — one week each, counting **post-op days 0 to
41**. Nothing on screen is dated: the post-op day is the unit every threshold in
Phase 1 is written in, and a calendar date would only be a label over the top of
it. Each cell is marked: `·` nothing flagged, `•` a deviation, `!` a critical
one. An infection on post-op day 8 is a wall of `!` from that day onward without
opening a single tab.

Both blocks are sized to sit above the fold, so the layer tabs are visible
without scrolling past the inputs to reach them.

### Editing a day

The inputs panel carries the fields the service is actually posted — `labs`,
`vitals`, `self_report` — pre-filled with what that day recorded. Change
anything, press **Apply and re-run**, and every prediction from that morning
onward is recomputed. Earlier days cannot move, because day D is judged on days
0..D, and only the days that could change are re-run: editing day 30 of a
six-week stay costs twelve runs rather than forty-two.

An empty lab is one that was not drawn — absent, which the layers treat as not
measured rather than zero.

**Vitals are daily means, plus how long each was held above its threshold.**
The mean alone is not enough: section 5.3 measures an *unbroken run* above 38.5,
so a mean of 37.6 could be a steady low grade or a two-hour spike, and those are
different findings. Type a mean and a duration and the rest of the day is set to
keep the mean.

The three ward-round sign-offs are not editable. They are written by the
generator on the days that operation would realistically see them, and they are
the only inputs that are a fact about the patient rather than a measurement of
one day.

Phase 1 covers wound healing, so the window is capped at six weeks; asking for
longer says so and shows the first 42 days.

## One operation, four ways it can go

No layer reads `surgery_type` — Layer 1 carries it through and Layer 4 prints
it, and nothing computes with it. The axis the layers actually see is **how the
recovery goes**, so that is the one worth walking through:

| Recovery | What happens |
| -------- | ------------ |
| **Textbook** | All four stages in order, gate opens on day 28, nothing flagged |
| **Slow, but gets there** | Same order, everything about half again as long. Gate opens day 38 |
| **Stalls in inflammation** | Never reaches remodelling. Flagged as not settling and stuck; the gate never opens |
| **Infection from day 8** | Textbook until day 8, then fever, tachycardia and a CRP rebound — six findings |

Surgery type is still there, and still only changes the signals: three
operations, with default weights inside the 65–80 kg band the recipe pool has
been generated for.

| Surgery | Weight | CRP peak |
| ------- | ------ | -------- |
| Laparoscopic cholecystectomy | 72 kg | 70 |
| Open abdominal — bowel resection | 80 kg | 92 |
| Below-knee amputation | 78 kg | 160 |

The CRP curve is the one in `fixtures/patient_six_week.json`, normalised and
rescaled, so the bowel resection is a transform of a patient the team already
reviewed rather than something invented alongside it.

## How well the patient eats

The **NSS — Nutritional Sufficiency Score** is how much of what a patient needs
to heal they actually got, scored out of 1 every day. It is an output, not a
setting: it is worked out from what was eaten. What the sidebar chooses is how
well they eat across the six weeks, and the score follows.

| Pattern | Mean NSS over 42 days |
| ------- | --------------------- |
| Eats everything recommended | 1.00 |
| Eats most of it | 0.97 |
| Starts poorly, improves | 0.80 |
| Starts well, tails off | 0.84 — ends escalating |
| Erratic, some days nothing logged | 0.78 |
| Barely eats | 0.58 — escalating throughout |

Two of those are worth watching past the score. **Starts well, tails off** shows
prealbumin stop rising as the eating falls away. **Erratic** takes prealbumin
below its clinical floor, so Layer 2 raises a protein-synthesis finding on its
own — the feeding problem becoming a clinical one.

A day with nothing logged is scored *nothing at all* rather than zero: not
knowing what someone ate is not the same as knowing they ate nothing.

## Recipes across the stay

Every day from oral intake onward has recipes — 41 of the 42 days, for all
three surgeries. Day 0 has none, and should not: the patient is nil by mouth,
which is also why nothing is scored for nutrition that day. The tab says so
rather than showing an empty list.

Recipes depend on the healing **stage**, not the surgery, so a six-week stay
needs at most four pool queries. All of them are cached locally.

## Meals logged, not adherence

Phase 2 scores adherence — did the patient eat what was prescribed. Phase 1 has
no prescription to adhere to: section 6.2 scores the previous 24 hours against a
phase target, and the input is whatever the dietitian wrote down. So the control
is **meals logged**, split 20/40/40 across breakfast, lunch and dinner using
`MEAL_SPLITS` from `populate_phase1_pool.py`.

| Meals logged | Share of target | NSS |
| ------------ | --------------- | --- |
| 3 of 3 | 100% | 1.00 |
| 2 of 3 | 80% | 0.80 |
| 1 of 3 | 40% | 0.65 → immediate dietitian escalation |
| nothing | no intake block at all | **unknown, not zero** |

That last row is the point. A day the dietitian did not write up is unknown, and
section 6.2 leaves it unscored rather than scoring it as starvation.

Intake also reaches Layer 2. Prealbumin rises only if the patient is being fed,
so sustained underfeeding stalls it, Layer 2 flags protein synthesis impairment,
and section 6.3 pairs that with the protein gap — the finding neither layer can
make alone.

## Layer 4 writes with the model, on demand

`claude-sonnet-5` can phrase any alert **and any routine note**. The wording is
checked in code after it is generated and discarded if it breaks a rule — it may
not use a number the layers did not produce, recommend anything they did not
flag, say anything about the wound itself, or reassure while something is
flagged. Who phrases the note changes; what it may say does not.

It is asked for with a **Generate insights** button rather than called
automatically, because the numbers are lopsided:

| | Time |
| --- | ---- |
| All four layers and the gate | 0.73 s |
| The model phrasing one day's note | 7.68 s |

Calling it on every click made moving through the calendar feel broken. On
demand, switching days is 0.1 s, and a day already written stays written for as
long as the day and the patient are unchanged.

Until it is asked for there is **no note on screen at all** — only the insight
and the recommendation, which are assembled from the layers' own fields. The
deterministic template is the service's documented default and it is what a ward
would get with the network down, but putting it on screen unasked showed a block
of machine-assembled prose that reads like the finished article and is not. With
no key reachable it is shown, labelled as the fallback.

One thing to know: the **service** deliberately does not call the model on a day
with nothing flagged — a routine note must not depend on a network call, so it
returns fixed text and stops. The simulator asks the model to phrase that note
too, from the same data and through the same checks, and falls back to the
service's own text if the wording fails one. `alert_generated` is never touched,
so a quiet day still escalates to nobody. That extension lives in
`sim_alert.py`; the service is untouched.

With no key reachable everything falls back to the fixed text.

The key is read from the environment, then Streamlit secrets, then a paste.
`Postop-Phase1/Staging/.env` already carries one, so it works out of the box.

## Five tabs

Each layer tab opens with two lines — what that layer asks, and what its result
means — then its insight and its recommendation for the selected day. The
Nutrition tab carries a plain-English explanation of the NSS.

| Tab | What it holds |
| --- | ------------- |
| **Today** | The day's metrics, whether the patient is ready for Phase 2, **what to do today**, and one line per layer |
| **Layer 1 · Phase** | Insight + recommendation, the posterior across the stay, today's symbols and why |
| **Layer 2 · Deviations** | Insight + recommendation, the GP against observed, findings, and whether each check could run |
| **Layer 3 · Nutrition** | Insight + recommendation, meals logged, all seven nutrients, section 6.3's interactions, and the day's recipes |
| **Layer 4 · Alert** | Insight + recommendation, the narrative, and section 7's five prohibitions |

Every recommendation is a string a layer already produced — Layer 2's from a
deviation's `recommended_action`, Layer 3's from `nss_action` and section 6.3,
Layer 4's from its escalation list. Nothing is written by this app, so a reader
can put a finger on any sentence and find the field it came from.

## Recipes survive the database

Whatever the pool returns is written to `.recipe_cache/` and served from there
when the RDS is unreachable, labelled with when it was fetched and with the
reason the live call failed. Every row in that cache came from `recipe_pool`.

What is still deliberately absent is a *composer*: nothing invents a recipe,
because unapproved food sitting beside approved food with nothing to tell them
apart is the failure worth avoiding. A cache of real rows is not that.

## Exporting the stay

**Prepare export** at the foot of the page collects every day — insights,
recommendations, and the recipes for that day's healing phase — into one JSON
file. Recipes come from `recipe_pool` through the same `select_phase1_recipes`
call the orchestrator makes, and because they depend only on the phase and the
demographic band, a six-week stay needs at most four queries rather than
forty-two.

When the pool is unreachable the export carries the last rows it returned,
labelled with when they were fetched.

`run_simulation.py --json out.json` writes the same thing for the whole cohort.

**Nothing is written back to the RDS or to LangGraph.** Storing recommendations
in the staging store is a side effect on shared data, so it needs asking for
rather than slipping into an export.

## The replay

Post-op day D is judged on days 0..D and nothing later — the slice
`run_daily_simulation.py` posts to LangGraph one calendar day at a time. No
state is carried between runs: Phase 1 is stateless per call and the prefix is
the whole input.

It costs one full Phase 1 run per day, so six weeks takes about thirty seconds
the first time and is then instant to click through. The replay itself is
deterministic and never calls the model; only the day on screen is re-run
through Layer 4.

## Files

| File | Purpose |
| ---- | ------- |
| `app.py` | The UI |
| `run_simulation.py` | The same simulation, in a terminal. `--surgery`, `--days`, `--weeks`, `--meals`, `--complication`, `--per-day`, `--json` |
| `sim_timeline.py` | The generated patient: seven surgeries, up to six weeks |
| `sim_meals.py` | Meals logged → intake, using the service's own meal splits |
| `sim_fixtures.py` | The real patients, read from the service's `fixtures/` |
| `sim_replay.py` | One Phase 1 run per day, plus the diff between consecutive days |
| `sim_insights.py` | One insight and one recommendation per layer, per day |
| `sim_recipes.py` | The pool call, the local cache, and which bands the pool holds |
| `sim_store.py` | The whole stay collected for export — one entry per day |
| `sim_alert.py` | Lets the model phrase the quiet-day note, through the same checks an alert faces |
| `sim_fixtures.py` | **Test only.** Loads the service's real fixtures so `test_replay.py` can check the generator against data the team curated |

## Tests

```bash
../Postop-Phase1/Staging/phase1/bin/python test_sim_timeline.py   # surgeries and meals
../Postop-Phase1/Staging/phase1/bin/python test_replay.py         # the real fixtures
../Postop-Phase1/Staging/phase1/bin/python test_app.py            # renders every tab
```

The assertions worth knowing about: seven surgeries produce seven different
courses rather than one course with different labels; `as_of(D)` is a **prefix**
and never a filter, so no run can see a day that had not happened; nothing
logged gives an unknown NSS and no intake block; and the negative control fires
nothing on any day.

## Known limits

* **The recipe pool needs network reach** to the staging RDS. Without it the tab
  reports the status `select_phase1_recipes` returned. There is deliberately no
  offline substitute — composing a stand-in would put unapproved food beside
  approved food with nothing to tell them apart.
* **The surgery profiles are the simulator's.** Nothing in the platform defines
  how an amputation differs from a cholecystectomy, because no layer reads the
  surgery type.
* Everything the layers cannot do — no wound imaging, within-patient rather than
  cohort deviation detection, NSS weights that are ours — is stated on the tab
  where it applies and listed in `Postop-Phase1/README.md`.

## Running it somewhere else

The simulator imports the Phase 1 service rather than reimplementing it, and it
looks in two places:

1. `../Postop-Phase1/Staging` — the real service, when this sits beside it.
   Always preferred, so a threshold changed there shows up here on reload.
2. `phase1_service/` — a vendored copy, for a host that only sees this
   repository.

`test_vendored_parity.py` diffs the two whenever both are present, so the copy
cannot drift unnoticed. It is byte-for-byte identical except for one file.

### The one file that differs

`phase1_service/phase1_config.py` carries **no credentials**. The service's own
copy spells the staging database out in code so it starts with a bare uvicorn
command — reasonable on a laptop, not in a repository. The vendored one reads
everything from the environment, Streamlit's secrets, or a git-ignored `.env`,
and has no default to fall back to.

### Streamlit Cloud

Point it at this repository, `app.py` as the entry point. Then in **Secrets**:

```toml
ANTHROPIC_API_KEY = "<your Anthropic API key>"
# Optional. Without it the Recipes tab serves the local cache.
RECIPE_POOL_DATABASE_URL = "<postgres url for the recipe pool>"
```

Neither is required. Without the key, Layer 4 shows its deterministic template
labelled as the fallback. Without the database, recipes come from
`.recipe_cache/` — 22 entries of real `recipe_pool` rows, labelled with when
they were fetched.

Expect the database to be unreachable from Streamlit Cloud unless the RDS
security group allows it: the cache is what makes the deployment work anyway,
and is why it is committed rather than ignored.
