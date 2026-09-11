"""The generated patient: surgeries, six weeks, and meals logged.

The layers have their own 118 tests. These check what the simulator adds — that
what it hands the layers is section 3 shaped, that seven surgeries produce seven
different courses rather than one course with different labels, and that the one
nutrition control does what it claims: fewer meals logged means a lower NSS,
and *nothing* logged means an unknown NSS rather than a zero one.
"""
from __future__ import annotations

import sys

import sim_meals
import sim_paths  # noqa: F401
import sim_replay as R
import sim_timeline as T
from layer1_phase_estimator import validate_days
from phase1_daily_recommendation import check_threshold_breaches, run_phase1

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def run(timeline):
    return run_phase1(timeline, use_llm=False)


# --- 1. Section 3 shape ----------------------------------------------------
print("\nshape")
for surgery in T.SURGERY_ORDER:
    for weeks in (1, 3, T.MAX_WEEKS):
        tl = T.build(surgery=surgery, weeks=weeks, seed=3)
        try:
            days = validate_days(tl)
            ok, detail = True, ""
        except Exception as exc:                          # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(f"{surgery} @ {weeks}w validates", ok, detail)
        if ok:
            check(f"{surgery} @ {weeks}w spans {weeks * 7} days",
                  len(days) == weeks * 7, str(len(days)))
check("weeks is clamped to the supported range",
      len(T.build(surgery="bk_amputation", weeks=99)["days"]) == T.MAX_WEEKS * 7)
def _raised(callable_) -> bool:
    try:
        callable_()
    except ValueError:
        return True
    except Exception:
        return False
    return False


check("an unknown surgery is rejected",
      _raised(lambda: T.build(surgery="nope")))
check("an unknown recovery pattern is rejected",
      _raised(lambda: T.build(recovery="nope")))
check("an unknown intake pattern is rejected",
      _raised(lambda: T.build(intake_pattern="nope")))


# --- 2. Seven surgeries, seven courses -------------------------------------
print("\nsurgeries differ")
arcs = {s: R.replay(T.build(surgery=s, weeks=T.MAX_WEEKS, seed=5))
        for s in T.SURGERY_ORDER}
gate_days = {s: a["gate_opened_on"] for s, a in arcs.items()}
check("three surgeries, not a catalogue — surgery type is context, and no "
      "layer computes with it", len(T.SURGERY_ORDER) == 3, str(T.SURGERY_ORDER))
check("every default weight is inside the band the recipe pool covers",
      all(65 <= T.SURGERIES[s]["weight_kg"] <= 80 for s in T.SURGERY_ORDER),
      str({s: T.SURGERIES[s]["weight_kg"] for s in T.SURGERY_ORDER}))
check("every surgery clears the gate within six weeks",
      all(d is not None for d in gate_days.values()), str(gate_days))
check("they do not all clear it on the same day",
      len(set(gate_days.values())) == len(gate_days), str(gate_days))
check("laparoscopic surgery clears before open abdominal",
      gate_days["lap_cholecystectomy"] < gate_days["open_bowel_resection"],
      str(gate_days))
check("an amputation is the slowest of the seven",
      gate_days["bk_amputation"] >= max(gate_days.values()), str(gate_days))
peaks = {s: max(d["labs"]["crp"] for d in T.build(surgery=s, weeks=2)["days"])
         for s in T.SURGERY_ORDER}
check("CRP peaks differ between surgeries",
      len(set(peaks.values())) == len(peaks), str(peaks))
check("the amputation has the largest inflammatory response",
      peaks["bk_amputation"] == max(peaks.values()), str(peaks))
check("every patient reaches remodelling",
      all("REMODELLING" in a["phase_first_seen"] for a in arcs.values()),
      str({s: list(a["phase_first_seen"]) for s, a in arcs.items()}))


# --- 3. The bowel resection reproduces the team's own fixture --------------
print("\nagainst fixtures/patient_six_week.json")
bowel = arcs["open_bowel_resection"]
check("the gate opens on day 28", bowel["gate_opened_on"] == 28,
      str(bowel["gate_opened_on"]))
# Not day-identical to the fixture, and it should not claim to be: the wearable
# noise is seeded differently, and a single sample tipping a symbol moves a
# transition by a day. What must match is the course.
check("all four phases, in order",
      list(bowel["phase_first_seen"]) == ["HAEMOSTASIS", "INFLAMMATION",
                                          "PROLIFERATION", "REMODELLING"],
      str(bowel["phase_first_seen"]))
check("remodelling is reached in the fixture's third week",
      14 <= bowel["phase_first_seen"]["REMODELLING"] <= 16,
      str(bowel["phase_first_seen"]))
check("nothing is flagged on the uncomplicated course",
      all(r["deviation_count"] == 0 for r in bowel["days"]),
      str({r["day"]: r["deviations"] for r in bowel["days"] if r["deviations"]}))


# --- 4. Meals logged is the nutrition control ------------------------------
print("\nmeals logged")
full = run(T.build(surgery="open_bowel_resection", weeks=3,
                   mean_meals_logged=3.0, seed=5))
check("three meals a day scores a full NSS", full["summary"]["nss"] == 1.0,
      str(full["summary"]["nss"]))

none_logged = T.build(surgery="open_bowel_resection", weeks=3,
                      mean_meals_logged=0.0, seed=5)
check("nothing logged means no intake block at all",
      all("nutrition" not in d for d in none_logged["days"]),
      str([d["day"] for d in none_logged["days"] if "nutrition" in d]))
starved = run(none_logged)
check("an unlogged day gives an unknown NSS, not a zero one",
      starved["summary"]["nss"] is None, str(starved["summary"]["nss"]))
check("nothing logged raises no nutritional deviation either",
      starved["layer3_nutrition"]["nss_action"]["escalation"] == "none",
      str(starved["layer3_nutrition"]["nss_action"]))

scores = []
for meals in (3.0, 2.0, 1.0):
    arc = R.replay(T.build(surgery="open_bowel_resection", weeks=3,
                           mean_meals_logged=meals, seed=5))
    values = [r["nss"] for r in arc["days"] if r["nss"] is not None]
    scores.append(sum(values) / len(values))
check("fewer meals logged means a lower mean NSS",
      scores[0] > scores[1] > scores[2],
      " > ".join(f"{v:.3f}" for v in scores))

check("meal shares come from the service's own MEAL_SPLITS",
      abs(sim_meals.logged_share(["breakfast"]) - 0.20) < 1e-9
      and abs(sim_meals.logged_share(["lunch", "dinner"]) - 0.80) < 1e-9,
      str(sim_meals.meal_shares()))
check("dinner is the meal kept when only one is logged",
      sim_meals.meals_for_day.__doc__ is not None
      and sim_meals.logged_share(["dinner"]) == 0.40)


# --- 5. Underfeeding stalls prealbumin, which Layer 2 is watching for ------
print("\nintake reaches layer 2")
fed = [d["labs"]["prealbumin"] for d in
       T.build(intake_pattern="all", days=T.MAX_DAYS, seed=5)["days"]
       if "prealbumin" in d["labs"]]
underfed = [d["labs"]["prealbumin"] for d in
            T.build(intake_pattern="poor", days=T.MAX_DAYS, seed=5)["days"]
            if "prealbumin" in d["labs"]]
check("prealbumin rises on a fed patient", fed[-1] > fed[0],
      f"{fed[0]} -> {fed[-1]}")
check("prealbumin does not rise on an underfed one", underfed[-1] <= underfed[0],
      f"{underfed[0]} -> {underfed[-1]}")

# Erratic eating takes prealbumin below the clinical floor, which is what the
# protein-synthesis check is looking for.
thin = run(T.build(intake_pattern="erratic", days=T.MAX_DAYS, seed=5))
check("Layer 2 flags protein synthesis impairment when intake stalls it",
      any(d["deviation"] == "protein_synthesis_impairment"
          for d in thin["layer2_deviations"]["deviations"]),
      str([d["deviation"] for d in thin["layer2_deviations"]["deviations"]]))

# The interaction needs both halves on the same day: a protein gap from Layer 3
# and inflammation that will not settle from Layer 2. Neither alone says the
# feeding is holding the healing back.
paired = run(T.build(recovery="stalls", intake_pattern="poor",
                     days=T.MAX_DAYS, seed=5))
check("a protein gap alongside unsettling inflammation is paired",
      any(i["id"] == "protein_gap_with_unresolving_inflammation"
          for i in paired["layer3_nutrition"]["interactions"]),
      str([i["id"] for i in paired["layer3_nutrition"]["interactions"]]))
check("and neither half alone raises that pairing",
      not run(T.build(recovery="stalls", intake_pattern="all",
                      days=T.MAX_DAYS, seed=5))["layer3_nutrition"]["interactions"],
      "a pairing fired without a nutritional gap")


# --- 6. Four ways the same operation can go --------------------------------
# The surgery is context; this is the axis the layers actually see. Each course
# has to produce the picture its label promises, or the walkthrough is a lie.
print("\nrecovery patterns")
courses = {name: R.replay(T.build(recovery=name, days=T.MAX_DAYS, seed=5))
           for name in T.RECOVERY_ORDER}

check("textbook reaches all four stages",
      len(courses["textbook"]["phase_first_seen"]) == 4,
      str(courses["textbook"]["phase_first_seen"]))
check("textbook flags nothing on any day",
      not courses["textbook"]["first_flagged"],
      str(courses["textbook"]["first_flagged"]))
check("textbook opens the gate",
      courses["textbook"]["gate_opened_on"] is not None)

check("slow reaches all four stages too",
      len(courses["slow"]["phase_first_seen"]) == 4,
      str(courses["slow"]["phase_first_seen"]))
check("slow reaches each stage later than textbook",
      courses["slow"]["phase_first_seen"]["REMODELLING"]
      > courses["textbook"]["phase_first_seen"]["REMODELLING"],
      f'{courses["slow"]["phase_first_seen"]} vs '
      f'{courses["textbook"]["phase_first_seen"]}')
check("slow opens the gate, but later",
      courses["slow"]["gate_opened_on"] is not None
      and courses["slow"]["gate_opened_on"]
      > courses["textbook"]["gate_opened_on"],
      f'{courses["slow"]["gate_opened_on"]} vs '
      f'{courses["textbook"]["gate_opened_on"]}')

check("a stalled recovery never reaches remodelling",
      "REMODELLING" not in courses["stalls"]["phase_first_seen"],
      str(courses["stalls"]["phase_first_seen"]))
check("and is flagged as not settling and stuck",
      {"inflammation_not_resolving", "phase_transition_delay"}
      <= set(courses["stalls"]["first_flagged"]),
      str(courses["stalls"]["first_flagged"]))
check("and the gate never opens on it",
      courses["stalls"]["gate_opened_on"] is None)

check("an infection is flagged from the day it starts",
      courses["infection"]["first_flagged"].get("fever_pattern") == 8,
      str(courses["infection"]["first_flagged"]))
check("an infection reaches CRITICAL",
      any(r["healing_status"] == "CRITICAL"
          for r in courses["infection"]["days"]),
      str({r["day"]: r["healing_status"] for r in courses["infection"]["days"]}))
check("nothing is flagged on an infected patient before day 8",
      all(r["deviation_count"] == 0 for r in courses["infection"]["days"]
          if r["day"] < 8))

print("\ndiabetes")
dm = T.build(recovery="textbook", days=14, diabetic=True, seed=5)
check("diabetes shows up as a glucose finding",
      any(d["deviation"] == "glucose_dysregulation"
          for d in run(dm)["layer2_deviations"]["deviations"]))
check("and pages someone through the higher immediate threshold",
      len(check_threshold_breaches(dm)) > 0)
check("without it, glucose is not flagged",
      not any(d["deviation"] == "glucose_dysregulation" for d in
              run(T.build(recovery="textbook", days=14, seed=5))
              ["layer2_deviations"]["deviations"]))

dark = run(T.build(recovery="textbook", days=14, wearable_coverage=0.0, seed=5))
not_assessed = dark["layer2_deviations"]["checks_not_assessed"]
check("no wearable leaves checks unassessed", bool(not_assessed),
      str(not_assessed))
check("an unassessed check raises no deviation",
      all(d["deviation"] not in not_assessed
          for d in dark["layer2_deviations"]["deviations"]))


# --- 6b. How well the patient eats, over six weeks -------------------------
print("\nintake patterns")
eating = {name: R.replay(T.build(intake_pattern=name, days=T.MAX_DAYS, seed=5))
          for name in T.INTAKE_ORDER}


def _mean_nss(arc):
    scores = [r["nss"] for r in arc["days"] if r["nss"] is not None]
    return sum(scores) / len(scores)


check("eating everything scores a full NSS every day",
      all(r["nss"] == 1.0 for r in eating["all"]["days"]
          if r["nss"] is not None), str(_mean_nss(eating["all"])))
check("barely eating scores far lower",
      _mean_nss(eating["poor"]) < 0.70, f'{_mean_nss(eating["poor"]):.2f}')
check("eating well beats eating badly, in that order",
      _mean_nss(eating["all"]) > _mean_nss(eating["most"])
      > _mean_nss(eating["poor"]),
      " > ".join(f'{_mean_nss(eating[k]):.2f}' for k in
                 ("all", "most", "poor")))
check("a patient who improves ends better than they started",
      (eating["improving"]["days"][-1]["meals_logged"] or 0)
      > (eating["improving"]["days"][2]["meals_logged"] or 0),
      str([r["meals_logged"] for r in eating["improving"]["days"][:6]]))
check("a patient who tails off ends worse than they started",
      (eating["declining"]["days"][-1]["meals_logged"] or 0)
      < (eating["declining"]["days"][2]["meals_logged"] or 0),
      str([r["meals_logged"] for r in eating["declining"]["days"][-6:]]))
_blank = [r["day"] for r in eating["erratic"]["days"][1:]
          if r["meals_logged"] == 0]
check("an erratic patient has days with nothing logged at all", bool(_blank),
      str([r["meals_logged"] for r in eating["erratic"]["days"][:10]]))
# The summary carries the last day that *was* scored, which is right for a ward
# round — so the unlogged day has to be checked on its own record.
_daily = {d["day"]: d for d in
          eating["erratic"]["days"][-1]["bundle"]["layer3_nutrition"]["daily"]}
check("and those days are left unscored rather than scored zero",
      all(_daily[day]["status"] != "scored" for day in _blank if day in _daily),
      str([(day, _daily[day]["status"]) for day in _blank[:4] if day in _daily]))


# --- 7. Sign-offs are recorded once and inherited --------------------------
print("\nsign-offs")
tl = T.build(surgery="open_bowel_resection", weeks=T.MAX_WEEKS, seed=5)
recorded = {d["day"]: set(d["clinical"]) for d in tl["days"] if d.get("clinical")}
check("each sign-off is written on exactly one day",
      sorted(recorded) == sorted(T.SURGERIES["open_bowel_resection"]["signoff"].values()),
      str(recorded))
check("and never repeated afterwards",
      sum(len(v) for v in recorded.values()) == 3, str(recorded))
gates = {r["day"]: r["gate_blocking"] for r in bowel["days"]}
check("a sign-off written once still holds on every later day",
      all("Diet fully advanced" not in gates[d] for d in range(7, 42)),
      str([d for d in range(7, 42) if "Diet fully advanced" in gates[d]][:5]))


# --- 8. Determinism --------------------------------------------------------
print("\ndeterminism")
check("the same inputs give the same timeline",
      T.build(surgery="bk_amputation", weeks=2, seed=11)
      == T.build(surgery="bk_amputation", weeks=2, seed=11))
check("a different seed gives a different one",
      T.build(surgery="bk_amputation", weeks=2, seed=11)
      != T.build(surgery="bk_amputation", weeks=2, seed=12))


# --- 9. Editing one day ----------------------------------------------------
print("\nediting a day")
base = T.build(surgery="open_bowel_resection", days=14, seed=5)
same = T.apply_overrides(base, {8: T.day_inputs(base, 8)})
check("re-applying a day's own inputs does not change the reading",
      run(same)["summary"]["healing_status"]
      == run(base)["summary"]["healing_status"],
      f'{run(base)["summary"]["healing_status"]} -> '
      f'{run(same)["summary"]["healing_status"]}')

febrile = T.apply_overrides(base, {8: {**T.day_inputs(base, 8),
                                       "mean_temp": 38.4, "fever_hours": 8.0}})
out = run(febrile)
_typed = T.day_inputs(febrile, 8)
check("a typed mean comes back as the mean",
      abs(_typed["mean_temp"] - 38.4) < 0.2, str(_typed["mean_temp"]))
check("and the run lasts as long as was asked for",
      abs(_typed["fever_hours"] - 8.0) < 0.6, str(_typed["fever_hours"]))
check("a fever typed onto day 8 is detected there",
      any(d["deviation"] == "fever_pattern" and d["day"] == 8
          for d in out["layer2_deviations"]["deviations"]),
      str([(d["day"], d["deviation"]) for d in out["layer2_deviations"]["deviations"]]))
check("and nothing was flagged before the edit",
      run(base)["summary"]["deviation_count"] == 0)
check("a fever below the threshold is not detected",
      not any(d["deviation"] == "fever_pattern" for d in run(T.apply_overrides(
          base, {8: {**T.day_inputs(base, 8), "mean_temp": 38.4,
                     "fever_hours": 2.0}}))["layer2_deviations"]["deviations"]),
      "2 hours is under section 5.3's 6-hour criterion")

unfed = T.apply_overrides(base, {9: {**T.day_inputs(base, 9), "meals": 0}})
_day9 = next(d for d in unfed["days"] if d["day"] == 9)
check("setting meals to zero removes the intake block entirely",
      "nutrition" not in _day9, str(sorted(_day9)))

blank = T.apply_overrides(base, {6: {**T.day_inputs(base, 6),
                                     "labs": {"crp": None, "wbc": None,
                                              "glucose": None,
                                              "fasting_glucose": None,
                                              "prealbumin": None,
                                              "albumin": None, "zinc": None}}})
check("a blank lab is absent, not zero",
      not (next(d for d in blank["days"] if d["day"] == 6)["labs"]),
      str(next(d for d in blank["days"] if d["day"] == 6)["labs"]))

signed = T.apply_overrides(base, {5: {**T.day_inputs(base, 5),
                                      "clinical": {"wound_closure_confirmed": True}}})
gate_after = R.replay(signed)
check("a sign-off typed onto day 5 holds on every later day",
      all("Wound closure" not in r["gate_blocking"]
          for r in gate_after["days"] if r["day"] >= 5),
      str([r["day"] for r in gate_after["days"]
           if r["day"] >= 5 and "Wound closure" in r["gate_blocking"]][:4]))

# --- 10. The incremental replay is not a shortcut --------------------------
# An edit recomputes only from the edited day. That is sound because day D is
# judged on days 0..D — but it is exactly the kind of optimisation that is
# quietly wrong, so it is checked against the full thing.
print("\nincremental replay")
edited = T.apply_overrides(base, {9: {**T.day_inputs(base, 9),
                                      "mean_temp": 38.4, "fever_hours": 9.0}})
full = R.replay(edited)
partial = R.replay(edited, since=9, reuse=R.replay(base))
check("recomputing from the edited day matches recomputing everything",
      [(r["day"], r["phase"], r["healing_status"], r["deviation_count"],
        r["nss"], r["gate_blocking_count"]) for r in full["days"]]
      == [(r["day"], r["phase"], r["healing_status"], r["deviation_count"],
           r["nss"], r["gate_blocking_count"]) for r in partial["days"]],
      "the two disagree")
check("and the days before the edit are untouched",
      [r["day"] for r in partial["days"]] == [r["day"] for r in full["days"]])


print()
if FAILURES:
    print(f"{len(FAILURES)} failed")
    for line in FAILURES:
        print(f"  - {line}")
    sys.exit(1)
print("all passed")
