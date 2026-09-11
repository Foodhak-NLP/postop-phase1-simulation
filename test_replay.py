"""The real-patient demo path: fixtures, slicing, and the day-by-day replay.

`test_sim_timeline.py` covers the synthetic generator. This covers the part the
demo actually runs on — the service's own fixtures — and the one property the
replay must never violate: post-op day D is judged on days 0..D and nothing
later. A replay that leaked a future day would make every finding look
prescient and would be wrong in the one way that is hardest to notice.
"""
from __future__ import annotations

import sys

import sim_fixtures as F
import sim_paths  # noqa: F401
import sim_replay as R
from layer1_phase_estimator import validate_days

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


# --- 1. The fixtures the demo offers exist and are loadable ----------------
print("\nfixtures")
catalogue = F.available()
check("fixtures were found", bool(catalogue), str(F.FIXTURE_DIR))
names = [f["name"] for f in catalogue]
check("the six-week patient is present and listed first",
      names and names[0] == "patient_six_week", str(names[:3]))
for meta in catalogue:
    timeline = F.load(meta["name"])
    try:
        validate_days(timeline)
        ok, detail = True, ""
    except Exception as exc:                          # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    check(f"{meta['name']} is a valid timeline", ok, detail)
    check(f"{meta['name']} is described",
          bool(meta["label"]) and bool(meta["blurb"]), meta["name"])


# --- 2. as_of never leaks a future day -------------------------------------
print("\nas_of")
six = F.load("patient_six_week")
for day in (0, 1, 7, 20, 28, 41):
    sliced = F.as_of(six, day)
    numbers = F.day_numbers(sliced)
    check(f"as_of({day}) stops at {day}",
          numbers and max(numbers) == day, str(numbers[-3:] if numbers else []))
    check(f"as_of({day}) is a prefix, not a filter",
          numbers == F.day_numbers(six)[:len(numbers)], str(numbers[:3]))
check("as_of does not mutate the source",
      len(F.day_numbers(six)) == 42, str(len(F.day_numbers(six))))
check("a day before the first is empty, not an error",
      F.day_numbers(F.as_of(F.load("patient_ready_for_phase2"), 5)) == [])


# --- 3. The six-week arc — the demo's spine --------------------------------
print("\nsix-week arc")
arc = R.replay(six)
check("one record per post-op day", len(arc["days"]) == 42, str(len(arc["days"])))
check("each day's bundle is that day",
      all(r["bundle"]["days_post_op"] == r["day"] for r in arc["days"]))
check("all four phases are entered", len(arc["phase_first_seen"]) == 4,
      str(arc["phase_first_seen"]))
_order = ["HAEMOSTASIS", "INFLAMMATION", "PROLIFERATION", "REMODELLING"]
check("the phases are entered in order",
      [arc["phase_first_seen"][p] for p in _order]
      == sorted(arc["phase_first_seen"][p] for p in _order),
      str(arc["phase_first_seen"]))
check("the patient is never flagged",
      all(r["deviation_count"] == 0 for r in arc["days"]),
      str({r["day"]: r["deviations"] for r in arc["days"] if r["deviations"]}))
check("the gate opens on day 28", arc["gate_opened_on"] == 28,
      str(arc["gate_opened_on"]))
check("the gate stays open once open", arc["gate_stayed_open"] is True)
check("the gate is shut before day 28",
      all(not r["phase2_unlocked"] for r in arc["days"] if r["day"] < 28))
check("blockers only ever fall",
      all(a["gate_blocking_count"] >= b["gate_blocking_count"]
          for a, b in zip(arc["days"], arc["days"][1:])),
      str([r["gate_blocking_count"] for r in arc["days"]]))

events = R.events(arc)
check("the arc produces a narrative spine", len(events) >= 8, str(len(events)))
check("the spine records the unlock",
      any(e["event"] == "Phase 2 unlocked" and e["day"] == 28 for e in events))
check("the spine records every phase change",
      sum(1 for e in events if e["event"] == "Phase change") == 3,
      str([e for e in events if e["event"] == "Phase change"]))
check("the arc table has a row per day",
      len(R.arc_frame(arc)) == 42)


# --- 4. Sign-offs are facts about the patient, not about a day -------------
# The fixture records the dietitian on day 7, wound closure on day 21 and
# discharge on day 28, and repeats them thereafter. The gate must still hold
# them on days that repeat nothing, which is what latest_clinical() added.
print("\nsign-offs")
_gate_on = {r["day"]: r["bundle"]["transition_gate"] for r in arc["days"]}
_diet = [d for d, g in sorted(_gate_on.items())
         if "Diet fully advanced" not in g["blocking"]]
check("the dietitian's day-7 sign-off holds on every later day",
      _diet and _diet[0] == 7 and _diet == list(range(7, 42)),
      str(_diet[:5]))
_wound = [d for d, g in sorted(_gate_on.items())
          if "Wound closure" not in g["blocking"]]
check("the day-21 wound confirmation holds on every later day",
      _wound and _wound[0] == 21 and _wound == list(range(21, 42)),
      str(_wound[:5]))


# --- 5. A quiet day is no longer silent ------------------------------------
print("\nlayer 4 on a quiet day")
_quiet = arc["days"][-1]["bundle"]["layer4_alert"]
check("no alert is generated for a patient with nothing flagged",
      _quiet["alert_generated"] is False)
check("a routine daily summary is written instead",
      bool(_quiet.get("narrative")), str(_quiet.get("narrative_kind")))
check("the summary is marked as a summary, not an alert",
      _quiet.get("narrative_kind") == "daily_summary",
      str(_quiet.get("narrative_kind")))
check("the summary escalates to nobody", not _quiet.get("escalate_to"))
check("the summary states that no wound imaging was available",
      "imaging" in (_quiet.get("narrative") or "").lower())


# --- 6. A flagged patient still alerts -------------------------------------
print("\nssi arc")
ssi = R.replay(F.load("patient_ssi"))
check("the infection is flagged from day 8",
      ssi["first_flagged"].get("fever_pattern") == 8,
      str(ssi["first_flagged"]))
check("nothing is flagged before day 8",
      all(r["deviation_count"] == 0 for r in ssi["days"] if r["day"] < 8),
      str([(r["day"], r["deviations"]) for r in ssi["days"] if r["day"] < 8]))
check("the alert is an alert, not a summary",
      ssi["days"][-1]["narrative_kind"] == "alert",
      str(ssi["days"][-1]["narrative_kind"]))
check("the gate never opens on an infected patient",
      ssi["gate_opened_on"] is None, str(ssi["gate_opened_on"]))
check("a deviation is reported from the first day it appears, not before",
      all(name in [d for r in ssi["days"] if r["day"] >= day
                   for d in r["deviations"]]
          for name, day in ssi["first_flagged"].items()))


# --- 7. The edges replay without special-casing ----------------------------
print("\nedges")
for name in ("edge_single_day", "edge_labs_only", "patient_ready_for_phase2"):
    try:
        edge = R.replay(F.load(name))
        ok, detail = True, ""
    except Exception as exc:                          # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    check(f"{name} replays", ok, detail)
    if ok:
        check(f"{name} produces a record per day",
              len(edge["days"]) == len(F.day_numbers(F.load(name))))
        check(f"{name} produces an events spine", isinstance(R.events(edge), list))


# --- 8. The quiet-day note, phrased by the model ---------------------------
# The service returns a deterministic summary and never calls the model on a
# day with nothing flagged. The simulator asks the model to phrase it, through
# the same checks an alert goes through. Without a key it must leave it alone.
print("\nlayer 4 on a quiet day")
import os

import sim_alert

_quiet_l4 = arc["days"][-1]["bundle"]["layer4_alert"]
check("the service's own summary is deterministic",
      _quiet_l4["source"] == "deterministic", _quiet_l4["source"])

_no_key = dict(os.environ)
os.environ.pop("ANTHROPIC_API_KEY", None)
try:
    untouched = sim_alert.phrase_summary(_quiet_l4)
finally:
    os.environ.update(_no_key)
check("with no key it is left exactly as it was",
      untouched["narrative"] == _quiet_l4["narrative"]
      and untouched["source"] == "deterministic")

check("an alert is never re-phrased by this path",
      sim_alert.phrase_summary(
          {**_quiet_l4, "alert_generated": True}) is not None
      and sim_alert.phrase_summary(
          {**_quiet_l4, "alert_generated": True})["source"] == "deterministic")

if os.environ.get("ANTHROPIC_API_KEY"):
    phrased = sim_alert.phrase_summary(_quiet_l4)
    check("with a key the model phrases it",
          phrased["source"] in ("llm", "deterministic_after_guardrail_rejection"),
          phrased["source"])
    check("and the wording passed the same checks an alert faces",
          phrased["source"] == "llm" and not phrased["guardrail_violations"],
          str(phrased.get("guardrail_violations")))
    check("it is still not an alert and still escalates to nobody",
          phrased["alert_generated"] is False and not phrased["escalate_to"])
else:
    print("  skip  no ANTHROPIC_API_KEY — the model path was not exercised")


print()
if FAILURES:
    print(f"{len(FAILURES)} failed")
    for line in FAILURES:
        print(f"  - {line}")
    sys.exit(1)
print("all passed")
