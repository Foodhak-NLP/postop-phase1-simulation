"""Generate a post-op patient: one surgery, up to six weeks, day by day.

Section 3 shapes the output. What varies is the surgery, how long you watch, what
the dietitian managed to log, and whether something went wrong.

**No layer reads `surgery_type`.** Layer 1 carries it through and Layer 4 prints
it; nothing computes with it. Every difference between two surgeries here is a
difference in the *signals* this module produces, and those profiles are the
simulator's, not the platform's. What is not invented is the shape: the CRP curve
is the one in `fixtures/patient_six_week.json`, normalised and rescaled, so an
open bowel resection generated here reproduces that fixture rather than competing
with it.

Intake is not an adherence percentage. It is meals logged — see `sim_meals`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import sim_meals
import sim_paths  # noqa: F401

SAMPLES_PER_DAY = 96                                # 15-minute wearable, section 3
PAIN_CLOCK = ("06:00", "12:00", "18:00", "00:00")   # 4-6 hourly, section 3
MAX_WEEKS = 6
MAX_DAYS = MAX_WEEKS * 7

# ---------------------------------------------------------------------------
# 1. The CRP shape, taken from the team's own six-week patient
# ---------------------------------------------------------------------------
# Normalised to peak 1.0 / floor 0.0 over its 21 days of resolution. A surgery
# with a different peak or a different resolution rescales this curve rather
# than getting a curve of its own, so every generated patient is a transform of
# a patient the team already reviewed.

_SHAPE_DAYS = 21.0
_SHAPE = (0.1011, 0.4719, 0.9551, 1.0000, 0.7528, 0.4719, 0.3034, 0.2135,
          0.1573, 0.1124, 0.0787, 0.0562, 0.0449, 0.0337, 0.0281, 0.0225,
          0.0169, 0.0112, 0.0112, 0.0056, 0.0056, 0.0000)


def _shape_at(t: float) -> float:
    """The normalised CRP curve at time `t`, linearly interpolated."""
    if t <= 0:
        return _SHAPE[0]
    if t >= len(_SHAPE) - 1:
        return _SHAPE[-1]
    low = int(t)
    frac = t - low
    return _SHAPE[low] * (1 - frac) + _SHAPE[low + 1] * frac


# ---------------------------------------------------------------------------
# 2. Surgeries
# ---------------------------------------------------------------------------
# `resolution_days` is how long CRP takes to settle, which stretches the shape.
# The sign-off days are when a ward round would realistically confirm each of
# section 10's three human criteria for that operation.

SURGERIES: Dict[str, Dict[str, Any]] = {
    "open_bowel_resection": {
        "label": "Open abdominal — bowel resection",
        "note": "Major open abdominal surgery. Reproduces "
                "`fixtures/patient_six_week.json`.",
        "crp_peak": 92.0, "resolution_days": 21.0, "wbc_peak": 13.0,
        "prealbumin_start": 180.0, "albumin_start": 38.0,
        "hr_base": 95.0, "temp_base": 37.2, "spo2_base": 95.5,
        "oral_from_day": 1, "pain_start": 7,
        "signoff": {"full_oral_diet_tolerated": 7,
                    "wound_closure_confirmed": 21,
                    "physician_discharge_signoff": 28},
        "weight_kg": 80.0,
    },
    "lap_cholecystectomy": {
        "label": "Laparoscopic cholecystectomy",
        "note": "Day-case in most patients; the mildest inflammatory response.",
        "crp_peak": 70.0, "resolution_days": 9.0, "wbc_peak": 11.5,
        "prealbumin_start": 205.0, "albumin_start": 41.0,
        "hr_base": 84.0, "temp_base": 37.0, "spo2_base": 97.5,
        "oral_from_day": 1, "pain_start": 4,
        "signoff": {"full_oral_diet_tolerated": 2,
                    "wound_closure_confirmed": 11,
                    "physician_discharge_signoff": 13},
        "weight_kg": 72.0,
    },
    "bk_amputation": {
        "label": "Below-knee amputation",
        "note": "Slowest wound. Discharge sign-off is the last criterion to fall.",
        "crp_peak": 160.0, "resolution_days": 25.0, "wbc_peak": 14.0,
        "prealbumin_start": 165.0, "albumin_start": 33.0,
        "hr_base": 94.0, "temp_base": 37.3, "spo2_base": 95.5,
        "oral_from_day": 1, "pain_start": 8,
        "signoff": {"full_oral_diet_tolerated": 4,
                    "wound_closure_confirmed": 28,
                    "physician_discharge_signoff": 35},
        "weight_kg": 78.0,
    },
}

SURGERY_ORDER = ("lap_cholecystectomy", "open_bowel_resection",
                 "bk_amputation")

# ---------------------------------------------------------------------------
# How the recovery goes
# ---------------------------------------------------------------------------
# The surgery is context — no layer computes with it. This is the axis that
# actually changes what the four layers see, so it is the one worth walking
# through: the same operation, and four different courses through it.

RECOVERY_PATTERNS: Dict[str, Dict[str, Any]] = {
    # `signoff_scale` stretches the days the ward round confirms each of
    # section 10's three human criteria; None means they are never confirmed,
    # which is what a wound that has not closed looks like. The floors hold a
    # signal above the level the HMM reads as settled — a stall is not one
    # blood test staying high, it is the whole picture failing to move.
    "textbook": {
        "label": "Ideal case — clears all four stages",
        "note": "Inflammation settles on schedule, the wound moves through "
                "every stage, and the gate opens. Nothing should be flagged.",
        "resolution_scale": 1.0, "crp_floor": 3.0, "infection_day": None,
        "signoff_scale": 1.0, "wbc_floor": None, "temp_floor": None,
        "hr_floor": None, "pain_floor": None,
    },
    "slow": {
        "label": "Slow, but gets there",
        "note": "Everything takes about half again as long — the bloods, the "
                "stages and the ward round. The order is the same; the gate "
                "just opens later.",
        "resolution_scale": 1.6, "crp_floor": 4.0, "infection_day": None,
        "signoff_scale": 1.35, "wbc_floor": None, "temp_floor": None,
        "hr_floor": None, "pain_floor": None,
    },
    "stalls": {
        "label": "Stalls in inflammation",
        "note": "Inflammation never settles: CRP plateaus, the white count "
                "stays up, a low-grade temperature persists and the pain does "
                "not improve. The wound does not move on and never closes, so "
                "the gate stays shut.",
        "resolution_scale": 1.0, "crp_floor": 48.0, "infection_day": None,
        "signoff_scale": None, "wbc_floor": 12.4, "temp_floor": 37.7,
        "hr_floor": 94.0, "pain_floor": 4,
    },
    "infection": {
        "label": "Infection from day 8",
        "note": "A textbook course until day 8, then CRP rebounds, a fever "
                "runs most of the day, and the heart rate climbs.",
        "resolution_scale": 1.0, "crp_floor": 3.0, "infection_day": 8,
        "signoff_scale": None, "wbc_floor": None, "temp_floor": None,
        "hr_floor": None, "pain_floor": None,
    },
}
RECOVERY_ORDER = ("textbook", "slow", "stalls", "infection")


# ---------------------------------------------------------------------------
# How well the patient eats
# ---------------------------------------------------------------------------
# Meals logged per day, which is what drives the Nutritional Sufficiency Score.
# A function of the day rather than one number, so a patient can start badly and
# improve — which is what most of them actually do.

def _flat(value: float):
    return lambda day, total: value


def _ramp(start: float, end: float):
    return lambda day, total: start + (end - start) * (day / max(1, total - 1))


INTAKE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "all": {"label": "Eats everything recommended",
            "note": "Three meals logged every day. The score sits at 1.00.",
            "meals": _flat(3.0), "jitter": 0.0},
    "most": {"label": "Eats most of it",
             "note": "Two or three meals most days — the ordinary case.",
             "meals": _flat(2.5), "jitter": 0.35},
    "improving": {"label": "Starts poorly, improves",
                  "note": "Barely eating early on, back to full meals by the "
                          "end. The score climbs with the patient.",
                  "meals": _ramp(0.8, 3.0), "jitter": 0.3},
    "declining": {"label": "Starts well, tails off",
                  "note": "Full meals at first, then less and less. Watch the "
                          "score fall and prealbumin stop rising with it.",
                  "meals": _ramp(3.0, 0.8), "jitter": 0.3},
    "erratic": {"label": "Erratic — some days nothing is logged",
                "note": "Wildly variable, including days with no intake logged "
                        "at all. Those days score nothing rather than zero.",
                "meals": _flat(1.6), "jitter": 1.1},
    "poor": {"label": "Barely eats",
             "note": "About one meal a day throughout. The score sits below "
                     "the escalation line and prealbumin stalls.",
             "meals": _flat(1.0), "jitter": 0.25},
}
INTAKE_ORDER = ("all", "most", "improving", "declining", "erratic", "poor")


# ---------------------------------------------------------------------------
# 3. Trajectories
# ---------------------------------------------------------------------------

def _crp(spec: Dict[str, Any], day: int, floor: float = 3.0) -> float:
    t = day * (_SHAPE_DAYS / float(spec["resolution_days"]))
    return round(floor + (spec["crp_peak"] - floor) * _shape_at(t), 1)


def _wbc(spec: Dict[str, Any], day: int) -> float:
    settled, span = 6.0, float(spec["resolution_days"]) * 1.6
    return round(max(settled, spec["wbc_peak"]
                     - (spec["wbc_peak"] - settled) * min(1.0, day / span)), 1)


def _prealbumin(spec: Dict[str, Any], day: int, fed: float) -> float:
    """Prealbumin rises only if the patient is actually being fed.

    Half-life about two days, so it is the marker that answers "is this patient
    building protein *now*" — which is why section 5.3 watches it and why it
    must respond to intake here. `fed` is the mean share of the daily target
    logged over the preceding few days. At 1.0 it reproduces the fixture's
    +4 mg/L per day; at 0 it stalls and drifts down, which is what Layer 2's
    protein-synthesis check is looking for and what section 6.3 pairs with a
    protein gap.
    """
    rise = 4.0 * (fed - 0.45) / 0.55        # fed 1.0 -> +4/day, fed 0.45 -> flat
    return round(min(320.0, max(90.0, spec["prealbumin_start"] + rise * day)), 1)


def _albumin(spec: Dict[str, Any], day: int) -> float:
    return round(max(spec["albumin_start"] - min(day, 4), 28.0), 1)


# ---------------------------------------------------------------------------
# 4. Streams
# ---------------------------------------------------------------------------

def _vitals(rng: np.random.Generator, row: Dict[str, Any],
            coverage: float) -> Dict[str, List[Dict[str, Any]]]:
    """15-minute HR, temperature and SpO2.

    `coverage` below 1.0 drops readings independently, which is what a device
    coming on and off looks like — and what breaks the unbroken run Layer 2
    needs before it will call a fever sustained.
    """
    hr: List[Dict[str, Any]] = []
    temp: List[Dict[str, Any]] = []
    spo2: List[Dict[str, Any]] = []
    fever, tachy = row.get("fever"), row.get("tachy")
    for i in range(SAMPLES_PER_DAY):
        hour = i * 0.25
        if coverage < 1.0 and rng.random() > coverage:
            continue
        clock = f"{int(hour):02d}:{int(round((hour % 1) * 60)):02d}"
        # `temp_rest` / `hr_rest` are set when a day was rebuilt from a typed
        # mean; without them the baseline is the generator's own diurnal curve.
        _t_rest = row.get("temp_rest")
        _h_rest = row.get("hr_rest")
        t = (row["temp"] if (fever and fever[0] <= hour <= fever[1])
             else (_t_rest if _t_rest is not None
                   else min(row["temp"], 37.1))
             - 0.3 * np.cos(hour / 24 * 2 * np.pi))
        h = (row["hr"] if (tachy and tachy[0] <= hour <= tachy[1])
             else (_h_rest if _h_rest is not None
                   else min(row["hr"], 92))
             + 6 * np.sin(hour / 24 * 2 * np.pi))
        temp.append({"t": clock, "v": round(float(t + rng.normal(0, 0.06)), 2)})
        hr.append({"t": clock, "v": round(float(h + rng.normal(0, 3)), 1)})
        spo2.append({"t": clock, "v": round(
            float(np.clip(row["spo2"] + rng.normal(0, 0.5), 85, 100)), 1)})
    return {"hr": hr, "temp": temp, "spo2": spo2}


def _labs(row: Dict[str, Any], day: int, cadence: Dict[str, int]) -> Dict[str, float]:
    """Only the bloods drawn that day. A lab not drawn is absent, not imputed."""
    labs = {"crp": row["crp"], "wbc": row["wbc"],
            "glucose": row["glucose"], "fasting_glucose": row["fasting_glucose"]}
    if cadence["protein"] and day % cadence["protein"] == 0:
        labs["prealbumin"] = row["prealbumin"]
        labs["albumin"] = row["albumin"]
    if cadence["zinc"] and day % cadence["zinc"] == 0:
        labs["zinc"] = row["zinc"]
    return labs


# ---------------------------------------------------------------------------
# 5. Build
# ---------------------------------------------------------------------------

def build(
    *,
    surgery: str = "open_bowel_resection",
    weeks: int = 6,
    days: Optional[int] = None,
    recovery: str = "textbook",
    intake_pattern: str = "all",
    diabetic: bool = False,
    mean_meals_logged: Optional[float] = None,
    wearable_coverage: float = 1.0,
    lab_cadence: Optional[Dict[str, int]] = None,
    patient_ref: str = "SIM-0001",
    age: float = 54,
    sex: str = "male",
    height_cm: float = 174,
    weight_kg: Optional[float] = None,
    vitamin_a_at_risk: bool = False,
    ssi_onset_day: int = 8,
    phase_overrides: Optional[Sequence[Dict[str, Any]]] = None,
    seed: int = 7,
) -> Dict[str, Any]:
    """One patient timeline in the shape section 3 specifies."""
    if surgery not in SURGERIES:
        raise ValueError(f"Unknown surgery {surgery!r}. "
                         f"Known: {', '.join(SURGERY_ORDER)}")
    if recovery not in RECOVERY_PATTERNS:
        raise ValueError(f"Unknown recovery pattern {recovery!r}. "
                         f"Known: {', '.join(RECOVERY_ORDER)}")
    if intake_pattern not in INTAKE_PATTERNS:
        raise ValueError(f"Unknown intake pattern {intake_pattern!r}. "
                         f"Known: {', '.join(INTAKE_ORDER)}")
    course = RECOVERY_PATTERNS[recovery]
    eating = INTAKE_PATTERNS[intake_pattern]
    # A recovery that stalls or drags is the same curve, stretched and floored.
    spec = {**SURGERIES[surgery],
            "resolution_days": SURGERIES[surgery]["resolution_days"]
            * course["resolution_scale"]}
    # Days win when given. A stay is a number of mornings, not a whole number
    # of weeks — a patient reviewed 17 days after surgery is a normal thing to
    # ask for and rounding it to two or three weeks would answer a different
    # question.
    days_total = int(np.clip(days if days is not None else weeks * 7,
                             1, MAX_DAYS))
    weight = float(weight_kg if weight_kg is not None else spec["weight_kg"])
    cadence = {"protein": 2, "zinc": 7, **(lab_cadence or {})}
    rng = np.random.default_rng(int(seed))

    # --- meals first: prealbumin has to respond to what was actually logged --
    meals_by_day: Dict[int, List[str]] = {}
    for day in range(days_total):
        if day < spec["oral_from_day"]:
            meals_by_day[day] = []          # nothing to log yet — not a failure
            continue
        target = (float(mean_meals_logged) if mean_meals_logged is not None
                  else float(eating["meals"](day, days_total)))
        meals_by_day[day] = sim_meals.meals_for_day(
            rng, target, jitter=float(eating["jitter"]))

    def fed_before(day: int, window: int = 5) -> float:
        window_days = [d for d in range(max(0, day - window), day + 1)
                       if d >= spec["oral_from_day"]]
        if not window_days:
            return 1.0                      # nothing to judge yet
        return float(np.mean([sim_meals.logged_share(meals_by_day[d])
                              for d in window_days]))

    # --- the day rows -------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for day in range(days_total):
        rows.append({
            "day": day,
            "crp": _crp(spec, day, floor=course["crp_floor"]),
            "wbc": max(course["wbc_floor"] or 0.0, _wbc(spec, day)),
            "prealbumin": _prealbumin(spec, day, fed_before(day)),
            "albumin": _albumin(spec, day),
            "glucose": 118.0, "fasting_glucose": 96.0, "zinc": 82.0,
            # Decay rates are the fixture's own: -0.03 C and -2 bpm a day, with
            # SpO2 recovering at +0.2. Keeping them identical is what makes a
            # generated bowel resection reproduce patient_six_week rather than
            # merely resemble it.
            "temp": round(max(course["temp_floor"] or 36.6,
                              spec["temp_base"] - day * 0.03), 2),
            "hr": float(max(course["hr_floor"] or 66.0,
                            spec["hr_base"] - day * 2.0)),
            "spo2": round(min(98.5, spec["spo2_base"] + day * 0.2), 1),
            "pain": [max(course["pain_floor"] or 0, spec["pain_start"] - day)] * 2
                    + [max(course["pain_floor"] or 0,
                           spec["pain_start"] - 1 - day)] * 2,
            "fever": None, "tachy": None,
        })

    if course["infection_day"] is not None:
        _apply_ssi(rows, int(ssi_onset_day if ssi_onset_day is not None
                             else course["infection_day"]))
    if diabetic:
        for row in rows:
            row["glucose"] = max(row["glucose"], 196.0 + (row["day"] % 3) * 9)
            row["fasting_glucose"] = max(row["fasting_glucose"],
                                         168.0 + (row["day"] % 3) * 6)

    # --- assemble -----------------------------------------------------------
    days: List[Dict[str, Any]] = []
    for row in rows:
        day = row["day"]
        entry: Dict[str, Any] = {
            "day": day,
            "labs": _labs(row, day, cadence),
            "vitals": _vitals(rng, row, wearable_coverage),
            "self_report": {
                "pain": [{"t": t, "v": int(v)}
                         for t, v in zip(PAIN_CLOCK, row["pain"])],
                "appetite": _appetite(meals_by_day[day],
                                      early=day < spec["oral_from_day"] + 2),
                "nausea": bool(day < 2 or (course["infection_day"] is not None
                                           and day >= int(ssi_onset_day))),
            },
        }
        nutrition = sim_meals.intake(
            weight, meals_by_day[day],
            provisioned_only=day < spec["oral_from_day"])
        if nutrition:
            entry["nutrition"] = nutrition

        # A sign-off is recorded on the ward round that made it and not
        # repeated. `latest_clinical` carries it forward, so repeating it would
        # hide the very behaviour worth showing.
        signed = {} if course["signoff_scale"] is None else {
            field: True for field, on_day in spec["signoff"].items()
            if round(on_day * course["signoff_scale"]) == day}
        if signed:
            entry["clinical"] = signed
        days.append(entry)

    timeline: Dict[str, Any] = {
        "patient_ref": patient_ref,
        "patient": {
            "surgery_type": spec["label"],
            "weight_kg": weight,
            "height_cm": float(height_cm),
            "age": float(age),
            "sex": sex,
            "vitamin_a_at_risk": bool(vitamin_a_at_risk),
        },
        "days": days,
    }
    if phase_overrides:
        timeline["clinician_phase_overrides"] = list(phase_overrides)

    timeline["_simulation"] = {
        "surgery": surgery,
        "recovery": recovery,
        "intake_pattern": intake_pattern,
        "diabetic": bool(diabetic),
        "days": days_total,
        "mean_meals_logged": (float(mean_meals_logged)
                              if mean_meals_logged is not None else None),
        "wearable_coverage": float(wearable_coverage),
        "oral_from_day": spec["oral_from_day"],
        "meals_by_day": {d: list(m) for d, m in meals_by_day.items()},
        "signoff_days": dict(spec["signoff"]),
        "seed": int(seed),
    }
    return timeline


def _appetite(meals: List[str], early: bool = False) -> str:
    """Appetite is a section 3 signal in its own right, and Layer 1 reads it.

    It mostly tracks what was logged — a patient who ate three meals did not
    have a poor appetite. But the first days after surgery are poor whatever
    ends up on the chart, and saying otherwise moves the HMM's proliferation
    transition a day early against the team's own six-week patient.
    """
    if early:
        return "poor"
    return {0: "poor", 1: "poor", 2: "fair", 3: "good"}[len(meals)]


def _apply_ssi(rows: List[Dict[str, Any]], onset: int) -> None:
    """Infection from `onset`: CRP rebound, fever, tachycardia, rising glucose.

    The endpoints are `fixtures/patient_ssi.json` — the same numbers the team
    already reviews, applied on top of whichever surgery is running.
    """
    peaks = (
        {"crp": 62.0, "wbc": 16.1, "temp": 38.7, "hr": 105.0, "spo2": 94.3,
         "glucose": 171.0, "fever": (10.0, 21.0), "tachy": (12.0, 22.0)},
        {"crp": 112.0, "wbc": 18.3, "temp": 38.9, "hr": 110.0, "spo2": 94.0,
         "glucose": 188.0, "fever": (6.0, 22.0), "tachy": (8.0, 23.0)},
        {"crp": 131.0, "wbc": 17.5, "temp": 38.6, "hr": 108.0, "spo2": 93.6,
         "glucose": 205.0, "fever": (8.0, 20.0), "tachy": (9.0, 21.0)},
    )
    for row in rows:
        offset = row["day"] - onset
        if offset < 0:
            continue
        peak = peaks[min(offset, 2)]
        for key in ("crp", "wbc", "temp", "hr", "glucose", "spo2"):
            row[key] = max(row[key], peak[key]) if key != "spo2" \
                else min(row[key], peak[key])
        row["fasting_glucose"] = round(row["glucose"] - 22.0, 1)
        row["prealbumin"] = round(max(120.0, row["prealbumin"] - 10.0 * (offset + 1)), 1)
        if row["temp"] >= 38.5:
            row["fever"] = peak["fever"]
        if row["hr"] >= 100.0:
            row["tachy"] = peak["tachy"]
        row["pain"] = [min(10, 3 + offset)] * 4


# ---------------------------------------------------------------------------
# 6. Views
# ---------------------------------------------------------------------------

def signal_frame(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The labs as one row per day. No inference — a reshape for charting."""
    sim = timeline.get("_simulation") or {}
    meals = sim.get("meals_by_day") or {}
    out = []
    for day in timeline["days"]:
        labs = day.get("labs") or {}
        vitals = day.get("vitals") or {}
        temps = [p["v"] for p in vitals.get("temp") or []]
        hrs = [p["v"] for p in vitals.get("hr") or []]
        spo2s = [p["v"] for p in vitals.get("spo2") or []]
        pains = [p["v"] for p in (day.get("self_report") or {}).get("pain") or []]
        logged = meals.get(day["day"], meals.get(str(day["day"])))
        out.append({
            "day": day["day"],
            "CRP": labs.get("crp"), "WBC": labs.get("wbc"),
            "Prealbumin": labs.get("prealbumin"), "Albumin": labs.get("albumin"),
            "Glucose": labs.get("glucose"),
            "Fasting glucose": labs.get("fasting_glucose"),
            "Temp max": round(max(temps), 2) if temps else None,
            "HR max": round(max(hrs), 1) if hrs else None,
            "SpO2 min": round(min(spo2s), 1) if spo2s else None,
            "Pain max": max(pains) if pains else None,
            "Meals logged": (len(logged) if logged is not None
                             else (3 if day.get("nutrition") else None)),
        })
    return out


# ---------------------------------------------------------------------------
# 7. Editing one day
# ---------------------------------------------------------------------------
# A generated patient is a starting point, not the answer. Being able to change
# what a single morning recorded — and watch the four layers change their minds
# — is the difference between a demo that plays and a demo you can argue with.
#
# Overrides are stored separately from the timeline and applied on top of it, so
# the generator stays the single description of the patient and an edit can
# always be taken back.

EDITABLE_LABS = ("crp", "wbc", "glucose", "fasting_glucose",
                 "prealbumin", "albumin", "zinc")
APPETITE_CHOICES = ("poor", "fair", "good")


def day_inputs(timeline: Dict[str, Any], day: int) -> Dict[str, Any]:
    """What one day currently records, in the shape the editor works in.

    The wearable streams are 96 samples each; nobody edits those by hand. What
    a clinician would actually change is the peak and how long it lasted, which
    is also exactly what section 5.3's sustained-duration checks read — so that
    is what this exposes, and `apply_overrides` rebuilds the stream from it.
    """
    entry = next((d for d in timeline["days"] if int(d["day"]) == int(day)), None)
    if entry is None:
        return {}
    vitals = entry.get("vitals") or {}
    temps = [p["v"] for p in vitals.get("temp") or []]
    hrs = [p["v"] for p in vitals.get("hr") or []]
    spo2s = [p["v"] for p in vitals.get("spo2") or []]
    pains = [p["v"] for p in (entry.get("self_report") or {}).get("pain") or []]
    sim = timeline.get("_simulation") or {}
    meals = (sim.get("meals_by_day") or {}).get(int(day))

    return {
        "labs": {key: (entry.get("labs") or {}).get(key) for key in EDITABLE_LABS},
        # Means, because a day's worth of 15-minute samples is not something
        # anyone edits by hand and a mean is how a chart would summarise it.
        # The two durations come with them: section 5.3 measures an *unbroken
        # run* above a threshold, so a mean temperature of 37.6 could be a
        # steady low grade or a two-hour spike, and those are different
        # findings. Without the duration the fever and tachycardia checks
        # cannot be driven at all.
        "mean_temp": round(sum(temps) / len(temps), 2) if temps else None,
        "mean_hr": round(sum(hrs) / len(hrs), 1) if hrs else None,
        "mean_spo2": round(sum(spo2s) / len(spo2s), 1) if spo2s else None,
        "fever_hours": _run_hours(vitals.get("temp") or [], TEMP_FEVER),
        "tachy_hours": _run_hours(vitals.get("hr") or [], HR_TACHY),
        # Both: the series is what section 3 specifies and what the payload
        # carries, the maximum is what a summary line wants.
        "pain_series": [int(v) for v in pains] or [0] * len(PAIN_CLOCK),
        "pain": max(pains) if pains else 0,
        "appetite": (entry.get("self_report") or {}).get("appetite", "fair"),
        "nausea": bool((entry.get("self_report") or {}).get("nausea")),
        "meals": len(meals) if meals is not None else (
            3 if entry.get("nutrition") else 0),
        "clinical": dict(entry.get("clinical") or {}),
    }


# The two thresholds section 5.3 measures a sustained run against. Imported
# rather than restated so the editor cannot drift from the detector.
from layer2_deviation_detector import HR_HIGH as HR_TACHY  # noqa: E402
from layer2_deviation_detector import TEMP_HIGH as TEMP_FEVER  # noqa: E402


def _run_hours(points: Sequence[Dict[str, Any]], threshold: float) -> float:
    """Longest unbroken stretch above `threshold`, in hours."""
    best = run = 0
    for point in points:
        run = run + 1 if float(point["v"]) > threshold else 0
        best = max(best, run)
    return round(best * 0.25, 2)


# How far above the threshold a sustained run sits. Just enough for section
# 5.3 to see it: typing "6 hours above 38.5" should produce a detectable fever,
# not a 40 C one.
FEVER_MARGIN = 0.25      # degrees C above TEMP_FEVER
TACHY_MARGIN = 6.0       # bpm above HR_TACHY


def _elevated_day(mean: float, hours: float, level: float,
                  floor: float, ceiling: float) -> Tuple[float, float]:
    """Split a day into `hours` at `level` and the rest at whatever keeps the
    mean. Returns (rest_of_day, level).

    Typing a mean and a duration over-determines nothing: the run has to sit
    above the threshold to be detectable, so the remainder of the day carries
    the difference. Where that would need an impossible value the mean wins and
    the remainder is clamped — a mean is a measurement, the split is a
    presentation choice.
    """
    if hours <= 0:
        return mean, mean
    hours = min(hours, 24.0)
    if hours >= 24.0:
        return level, level
    rest = (mean * 24.0 - level * hours) / (24.0 - hours)
    return float(np.clip(rest, floor, ceiling)), level


def _vitals_from_means(rng: np.random.Generator, *, mean_temp: float,
                       fever_hours: float, mean_hr: float, tachy_hours: float,
                       mean_spo2: float,
                       coverage: float) -> Dict[str, List[Dict[str, Any]]]:
    """Rebuild a day's 15-minute streams from means and sustained durations."""
    temp_rest, temp_level = _elevated_day(
        mean_temp, fever_hours, max(TEMP_FEVER + FEVER_MARGIN, mean_temp),
        35.5, 38.4)
    hr_rest, hr_level = _elevated_day(
        mean_hr, tachy_hours, max(HR_TACHY + TACHY_MARGIN, mean_hr), 45.0, 99.0)
    return _vitals(rng, {
        "temp": temp_level, "hr": hr_level, "spo2": mean_spo2,
        "fever": _window(fever_hours), "tachy": _window(tachy_hours),
        "temp_rest": temp_rest, "hr_rest": hr_rest,
    }, coverage)


def _window(hours: float) -> Optional[Tuple[float, float]]:
    """A run of `hours` placed in the middle of the day."""
    if hours <= 0:
        return None
    half = min(hours, 24.0) / 2
    return (max(0.0, 12.0 - half), min(24.0, 12.0 + half))


def apply_overrides(timeline: Dict[str, Any],
                    overrides: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of the timeline with the edited days replaced."""
    if not overrides:
        return timeline
    spec = SURGERIES[(timeline.get("_simulation") or {}).get(
        "surgery", "open_bowel_resection")]
    weight = float((timeline.get("patient") or {}).get("weight_kg") or 70.0)
    seed = int((timeline.get("_simulation") or {}).get("seed", 7))
    coverage = float((timeline.get("_simulation") or {})
                     .get("wearable_coverage", 1.0))

    days: List[Dict[str, Any]] = []
    meals_by_day = dict((timeline.get("_simulation") or {}).get("meals_by_day") or {})

    for entry in timeline["days"]:
        number = int(entry["day"])
        edit = overrides.get(number) or overrides.get(str(number))
        if not edit:
            days.append(entry)
            continue

        new = {**entry}

        labs = {k: v for k, v in (edit.get("labs") or {}).items() if v is not None}
        new["labs"] = labs

        # One generator per edited day, seeded from the day, so an edit is
        # reproducible and editing day 9 does not reshuffle day 10.
        rng = np.random.default_rng(seed + 1000 + number)
        new["vitals"] = _vitals_from_means(
            rng,
            mean_temp=float(edit.get("mean_temp") or spec["temp_base"]),
            fever_hours=float(edit.get("fever_hours") or 0.0),
            mean_hr=float(edit.get("mean_hr") or spec["hr_base"]),
            tachy_hours=float(edit.get("tachy_hours") or 0.0),
            mean_spo2=float(edit.get("mean_spo2") or spec["spo2_base"]),
            coverage=coverage)

        series = edit.get("pain_series")
        if not series:
            series = [int(edit.get("pain", 0))] * len(PAIN_CLOCK)
        series = [int(v) for v in series][:len(PAIN_CLOCK)]
        series += [series[-1] if series else 0] * (len(PAIN_CLOCK) - len(series))
        new["self_report"] = {
            "pain": [{"t": t, "v": v} for t, v in zip(PAIN_CLOCK, series)],
            "appetite": edit.get("appetite", "fair"),
            "nausea": bool(edit.get("nausea")),
        }

        count = int(np.clip(int(edit.get("meals", 0)), 0, 3))
        meals = list(MEALS_ORDER[3 - count:]) if count else []
        meals_by_day[number] = meals
        nutrition = sim_meals.intake(
            weight, meals, provisioned_only=number < spec["oral_from_day"])
        if nutrition:
            new["nutrition"] = nutrition
        else:
            new.pop("nutrition", None)

        clinical = {k: bool(v) for k, v in (edit.get("clinical") or {}).items()
                    if v is not None}
        if clinical:
            new["clinical"] = clinical
        else:
            new.pop("clinical", None)

        days.append(new)

    out = {**timeline, "days": days}
    out["_simulation"] = {**(timeline.get("_simulation") or {}),
                          "meals_by_day": meals_by_day,
                          "edited_days": sorted(int(d) for d in overrides)}
    return out


MEALS_ORDER = ("breakfast", "lunch", "dinner")


# ---------------------------------------------------------------------------
# 8. The request body
# ---------------------------------------------------------------------------

def request_payload(timeline: Dict[str, Any], day: int, *,
                    user_id: str = "4d1c2fc8-77ad-4205-8226-cf58a175e910",
                    surgery_date: Optional[str] = None,
                    date: Optional[str] = None,
                    hourly: bool = True) -> Dict[str, Any]:
    """One day as the body that is POSTed to `/langgraph/postop/phase1/run`.

    The same shape `day_requests/*.json` carries, so what the editor changes and
    what the service is actually sent cannot drift apart.

    `hourly` downsamples the wearable streams from 15-minute to hourly, which is
    what `make_curls.py` emits and the coarsest rate that still reproduces the
    15-minute answer: section 5.3 breaks a sustained run on any gap wider than
    an hour, so anything coarser makes sustained fever and tachycardia
    undetectable — silently.
    """
    entry = next((d for d in timeline["days"] if int(d["day"]) == int(day)), None)
    if entry is None:
        return {}

    def _series(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [p for p in points if str(p["t"]).endswith(":00")] if hourly \
            else list(points)

    vitals = entry.get("vitals") or {}
    body: Dict[str, Any] = {
        "user_id": user_id,
        "date": date,
        "surgery_date": surgery_date,
        "patient": {"surgery_type": (timeline.get("patient") or {})
                    .get("surgery_type")},
        "days": [{
            "day": int(entry["day"]),
            "labs": dict(entry.get("labs") or {}),
            "vitals": {name: _series(vitals.get(name) or [])
                       for name in ("hr", "temp", "spo2")},
            "self_report": dict(entry.get("self_report") or {}),
        }],
    }
    if entry.get("nutrition"):
        body["days"][0]["nutrition"] = dict(entry["nutrition"])
    if entry.get("clinical"):
        body["days"][0]["clinical"] = dict(entry["clinical"])
    return body
