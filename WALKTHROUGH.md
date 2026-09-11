# Post-op Phase 1 — demo walkthrough

Fifteen minutes, five scenes. Everything below is a real run; the numbers are
what the app shows.

```bash
cd postop-phase1-simulation
../Postop-Phase1/Staging/phase1/bin/streamlit run app.py
```

---

## Before you start: the one idea

A wound heals in four stages, and **each stage needs different feeding**. Phase 1
watches a patient through those stages and answers four questions every morning
— one per layer:

1. Which stage is this wound in? *(Layer 1)*
2. Is it going the way it should? *(Layer 2)*
3. Is the patient being fed enough to heal? *(Layer 3)*
4. What should the physician be told? *(Layer 4)*

| Stage | Typically lasts | What is happening |
| ----- | --------------- | ----------------- |
| Haemostasis | hours | Bleeding stops, clot forms. Pinned to the day of surgery — daily bloods cannot resolve a three-hour phase |
| Inflammation | ~5 days | Immune cells clear the wound. CRP climbs, peaks, must then fall |
| **Proliferation** | ~11 days | New tissue built. **The hungriest stage** |
| Remodelling | ~50 days | Scar reorganises. Targets fall back towards normal |

Those durations are read out of the model's own transition matrix, not typed
into the page — retune the matrix and the table follows it.

All of this is in an expander at the top of the app, so you can show it once and
collapse it.

The system **monitors and flags. It never decides.** It also cannot see the
wound — everything is inferred from bloods, wearables and the intake log — and
it says so on every screen.

---

## Scene 1 — the ideal recovery

Leave the sidebar as it opens: **Laparoscopic cholecystectomy · Ideal case ·
Eats everything recommended.**

Click through the calendar and watch the patient change. What is actually
happening to them:

| Day | Stage | CRP | Prealbumin | Pain | What is going on |
| --- | ----- | --- | ---------- | ---- | ---------------- |
| 0 | Haemostasis | 12 | 180 | 7/10 | Out of theatre. Bleeding stops, clot forms. **Nil by mouth — the Recipes tab is empty, and correctly so** |
| 1 | Inflammation | 45 | — | 6/10 | Immune cells flood the wound. CRP climbing hard. Oral intake starts |
| 3 | Proliferation | **92 — the peak** | — | 4/10 | CRP tops out and turns. New tissue starts being built |
| 8 | Proliferation | 17 | 212 | 0/10 | Collagen laid down fast. **This is the hungriest the patient will be** |
| 15 | Remodelling | 5 | — | 0 | Scar reorganising and strengthening |
| 28 | Remodelling | 3 | 292 | 0 | Everything settled. **Gate opens** |

**The one number to watch is CRP.** It peaks around day 3 and must then fall. A
CRP that stops falling is the earliest sign something is wrong — earlier than a
fever.

**The second is prealbumin**, 180 → 292. It has a two-day half-life, so it
answers *is this patient building protein right now?* That is why it is the
marker, not albumin.

### The gate closing

Open the **Today** tab on each of these days and watch the line at the top:

```
day  1   glucose controlled           5 criteria left
day  7   diet fully advanced          4 left   ← dietitian signs
day 11   inflammation resolved        3 left   ← CRP under 10 and falling
day 16   healing stage confirmed      2 left
day 21   wound closure                1 left   ← physician signs
day 28   discharge sign-off           0 left   ← physician signs
day 28   READY FOR PHASE 2
```

Three of those seven can only be signed by a human. **Their absence blocks, it
never passes.** That asymmetry is the safety property: Phase 2 optimises for
biomarkers and can recommend eating *less*, which is actively harmful while
collagen is still being laid down.

---

## Scene 2 — what the four layers are saying

Stay on day 8 and walk the tabs left to right. Each opens with what that layer
asks and what its answer means.

**Layer 1 · Phase** — *Which stage is this wound in?*
Reads seven daily signals and names the stage with a confidence. On day 8:
Proliferation, 98%. The table shows which readings pushed it there. Everything
downstream depends on this being right, because the feeding targets follow from
the stage.

**Layer 2 · Deviations** — *Is this going the way it should?*
Seven checks, every day, each with a rule in plain words. The table shows all
seven and whether each **ran**. A check with no data is reported as *not
assessed* — never as clear. A patient with no thermometer must not look like a
patient with no fever.

**Layer 3 · Nutrition** — *Is the patient being fed enough?*
Scroll to the recipes: three options per meal, in three columns, with a ✓ on
the one the patient actually had. A meal with no tick was not eaten — and that,
not the menu, is what the score is built from. Carbohydrate is pulled out
because it is the macro a ward adjusts first; the rest sits behind the
dropdowns.

The targets change with the stage — for an 80 kg patient:

| Stage | Protein | Calories | Vitamin C |
| ----- | ------- | -------- | --------- |
| Inflammation | 120 g | 2000 kcal | 200 mg |
| **Proliferation** | **144 g** | **2400 kcal** | **500 mg** |
| Remodelling | 96 g | 2000 kcal | 100 mg |

Proliferation is the demanding one — that is when tissue is being built. 144 g
of protein is roughly twice what this patient would need if they were well.

**Layer 4 · Alert** — *What should the physician be told?*
On a quiet day it writes a routine note; when something is flagged it writes an
alert with a priority and who to page.

Press **Generate insights** and `claude-sonnet-5` writes it. That takes about
eight seconds against seven tenths of a second for all four layers, which is why
it is a button rather than automatic — otherwise every click on the calendar
would stall. Until you press it there is no note, only the insight and the
recommendation, which come from the layers themselves.

Either way the wording is checked before it is shown: it may not use a number
the layers did not produce, recommend anything they did not flag, say anything
about the wound itself, or reassure while something is flagged.

---

## Scene 3 — when it goes wrong

Change **Wound healing** in the sidebar.

### Stalls in inflammation

The wound never moves on. CRP plateaus around 48, the white count stays up, a
low-grade temperature persists, the pain does not improve.

- Stages reached: Haemostasis → Inflammation → Proliferation. **Never
  remodelling.**
- Flagged: *inflammation not settling* and *stuck in one stage*
- **The gate never opens.** The wound never closes, so nobody signs it off

### Infection from day 8

A textbook course, then it turns. Click day 7, then day 8:

```
day  8   Proliferation → Inflammation      ← the stage goes BACKWARDS
day  8   On Track → Critical
day  8   fever · inflammation not settling · stuck · tachycardia flagged
day  8   Layer 4: routine note → ALERT
day 10   blood sugar too high flagged
```

Backwards transitions are allowed on purpose: an infection genuinely re-enters
inflammation, and Layer 3 reads the stage to set the protein target — so the
targets must follow the patient back.

The calendar shows the whole story at a glance: dots until day 8, then a wall
of `!`.

---

## Scene 4 — the nutrition axis

Set **Wound healing** back to *Ideal case*, then change **Meals actually
eaten**. The NSS is not something you set — it is worked out from what was
eaten, and it moves:

| Meals actually eaten | Mean NSS | What you see |
| -------------------- | -------- | ------------ |
| Eats everything recommended | 1.00 | Nothing flagged |
| Eats most of it | 0.97 | Occasional moderate gap |
| Starts poorly, improves | 0.80 | Score climbs with the patient |
| **Starts well, tails off** | 0.84 → **0.65** | Score falls, prealbumin stops rising, dietitian escalation |
| **Erratic** | 0.78 | Days with **nothing logged**, and prealbumin drops below its floor — Layer 2 raises a protein finding on its own |
| Barely eats | 0.58 | Escalating throughout |

Two things worth pausing on:

**A day with nothing logged scores *nothing*, not zero.** Not knowing what
someone ate is not the same as knowing they ate nothing. Pick *Erratic* and
find a day with 0 meals — the score is blank, not 0.00.

**Feeding becomes a clinical problem.** On *Erratic*, prealbumin falls below
150 and Layer 2 — which is not a nutrition layer — raises *not building
protein*. Set **Wound healing: Stalls** and **Meals: Barely eats** together and
Layer 3 pairs them: *"protein intake 52% below target, CRP not resolving"*.
Neither half means much alone; together they say the feeding is holding the
healing back.

---

## Scene 5 — change one day and watch it move

The panel beside the calendar is the day's actual inputs — the same fields the
service is posted. Pick any day and type into it; there is no Apply button,
it re-runs as soon as you leave the field.

Try this on day 20 of the ideal case:

1. **Mean temp** → `38.4`
2. **h > 38.5** → `9`

The calendar cell turns `!`, the status flips to **Critical**, Layer 2 raises a
fever, and Layer 4 turns its routine note into an alert naming who to page.

Two details worth calling out:

- **Only day 20 onward is recomputed.** Days 0–19 cannot move, because each day
  is judged only on the days up to it. That is also why an edit is fast.
- **A fever needs a duration, not just a temperature.** Set the hours to `2`
  instead of `9` and nothing fires — the rule is six unbroken hours. An average
  alone cannot tell a steady low temperature from a short spike.

Undo with **Undo this day**.

---

## What to say if someone asks "is any of this real?"

- The four layers and the gate are the **service's own code**, imported, not
  reimplemented. Change a threshold there and it changes here.
- The recipes are **real rows from `recipe_pool`**, the same query the
  orchestrator makes. They are cached locally, so the demo survives the database
  being down — and it says when they were fetched.
- The CRP curve is the team's own six-week fixture, rescaled.
- The **patient is simulated.** The surgery type changes nothing the layers
  compute with — it is context. What changes their answers is the recovery
  pattern and what was eaten, which is exactly what the two sidebar controls do.
