"""
Post-op Phase 1 — Layer 2: Deviation Detector.

Implements section 5 of Phase_1_Wound_Recovery.docx.

Layer 1 answers *which phase*. Layer 2 answers *is this going the way it should*.

    timeline + Layer 1 result  ->  healing_status + prioritised deviations

Seven deviations, from doc section 5.3. Three need a trajectory model; four are
threshold rules; two read Layer 1's phase estimate.

The trajectory model is a Gaussian Process with a Matern 3/2 kernel, as section
5.1 specifies — but with one deliberate departure that must be understood before
reading any output.

    Section 5.1 conditions the GP mean function on the patient's *cohort*:
    surgery type, ASA grade, BMI, diabetes, NRS-2002. Section 9 then says those
    cohort curves are learned as real patients accumulate. Both are right, and
    together they mean the thing Layer 2 is specified to compare against does
    not exist until there are real patients.

    So this GP is fitted *within patient*. It learns the shape of this patient's
    own CRP and prealbumin trajectory and flags when a new reading breaks it.
    That detects "this patient has departed from their own course" — not "this
    patient is healing slower than comparable patients". Those are different
    clinical claims. Every output says which one it is making.

Detection is causal: to judge day t, the GP is fitted on days 0..t-1 only. It
never sees the day it is judging, so a flag is something a clinician could have
been told that morning.

Run it:

    python3 layer2_deviation_detector.py --scenario ssi_regression
    python3 layer2_deviation_detector.py --timeline patient.json --json
    uvicorn layer2_deviation_detector:app --port 8011 --reload
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Layer 2 interprets Layer 1's output, so it shares Layer 1's vocabulary rather
# than duplicating it. A hard import: without Layer 1 there is nothing to detect
# deviations against.
from layer1_phase_estimator import (
    PHASES, estimate_phase, generate_timeline, validate_days, SCENARIOS,
)

MODEL_VERSION = "phase1-deviation-v1"


# ===========================================================================
# 1. CONFIG — the seven deviations, their thresholds and their escalation
#    Every number and every alert phrase traces to doc section 5.3.
# ===========================================================================

# --- Gaussian Process hyperparameters --------------------------------------
#
# Matern 3/2 (doc section 5.1): handles smooth but non-differentiable
# trajectories, which is the right shape for a biological process.
#
#   length scale  — how many days a reading stays informative about the next.
#                   Post-op inflammatory markers move over roughly this window.
#   noise         — assay and sampling noise, as a fraction of the signal's own
#                   spread. Keeps a single odd draw from breaking the fit.
#   min history   — below this the GP reports insufficient history rather than
#                   extrapolating from almost nothing.
GP_LENGTH_SCALE_DAYS = 4.0
GP_NOISE_FRACTION = 0.18
GP_MIN_HISTORY = 4
GP_CREDIBLE_Z = 1.96          # 95% credible interval

GP_SIGNALS = ("crp", "prealbumin")

# Both GP signals are concentrations, so they cannot be negative. A Gaussian
# interval can be: when the predicted mean sits near zero and the spread is
# wide, the lower bound goes below it. Mathematically correct, biologically
# impossible, and it reads as a defect on a clinical screen. Clamp the floor.
GP_NON_NEGATIVE = True

# --- deviation thresholds ---------------------------------------------------
CRP_WATCH_FROM_DAY = 3            # "after Day 3"
CRP_PLATEAU_READINGS = 3          # consecutive non-declining readings
CRP_DECLINE_PCT = 0.10            # what counts as declining

# Clinical floors. Section 5.3 defines these deviations purely as shapes — "CRP
# plateau", "prealbumin flat" — with no floor, which flags a patient whose CRP is
# sitting at 8 mg/L inside the normal range. A plateau only means "inflammation
# not resolving" if there is inflammation to resolve. These are added from the
# reference ranges, not read from the document:
#   CRP        normal is < 10 mg/L
#   Prealbumin 150 mg/L is the document's own Phase 1->2 gate (section 10)
CRP_CLINICAL_FLOOR = 10.0
PREALBUMIN_CLINICAL_FLOOR = 150.0

PREALBUMIN_WATCH_FROM_DAY = 5     # "from Day 5 onward"
PREALBUMIN_FLAT_READINGS = 3
PREALBUMIN_RISE_PCT = 0.05

PHASE_DELAY_AFTER_DAY = 7         # "HMM posterior stuck in Inflammation > Day 7"

SPO2_CONCERN_RANGE = (93.0, 95.0)  # "consistently 93-95% during proliferation"
SPO2_CONSECUTIVE_DAYS = 2
SPO2_CONSISTENT_FRACTION = 0.6    # share of the day's 12h rolling means in band

GLUCOSE_HIGH = 180.0              # "> 180 mg/dL on 2+ consecutive readings"
GLUCOSE_CONSECUTIVE = 2

HR_HIGH = 100.0                   # "HR > 100 sustained > 4 hours post Day 2"
HR_SUSTAINED_HOURS = 4.0
HR_WATCH_FROM_DAY = 2

TEMP_HIGH = 38.5                  # "Temp > 38.5 sustained > 6 hours after Day 2"
TEMP_SUSTAINED_HOURS = 6.0
TEMP_WATCH_FROM_DAY = 2

# --- wearable stream ---------------------------------------------------------
#
# Section 3: "Wearables provide continuous signals (HR, temperature, SpO2 —
# hundreds of readings per day)." Section 8: ingested every 15 minutes, with a
# feature engine computing rolling statistics over 1h, 4h and 12h windows.
#
# So vitals arrive as streams, and duration is always computable. Section 5.3's
# word is *sustained* — an unbroken stretch above threshold, not a count of
# crossings. That distinction is what stops normal physiological variation in a
# 96-reading day from firing an alert.
STREAM_SIGNALS = ("hr", "temp", "spo2")
FEATURE_WINDOWS_HOURS = (1.0, 4.0, 12.0)

# A wearable comes off for showers, charging and procedures. A gap wider than
# this breaks a run: we cannot claim a fever was sustained across a period when
# nothing was measured.
MAX_SAMPLE_GAP_HOURS = 1.0

# A run needs at least this many samples before its span means anything.
MIN_RUN_SAMPLES = 2

SEVERITY_ORDER = ("MODERATE", "HIGH", "CRITICAL")

# Escalation routing, doc section 5.3 "Alert Priority" column.
DEVIATIONS: Dict[str, Dict[str, Any]] = {
    "inflammation_not_resolving": {
        "signal": "CRP",
        "severity": "HIGH",
        "interpretation": "Potential wound infection or systemic complication",
        "action": "Same-day physician review",
        "escalate_to": ["Physician"],
        "source": "doc section 5.3 — CRP plateau or re-elevation after Day 3",
    },
    "protein_synthesis_impairment": {
        "signal": "Prealbumin",
        "severity": "HIGH",
        "interpretation": "Nutritional insufficiency during peak collagen synthesis",
        "action": "Immediate dietitian review",
        "escalate_to": ["Dietitian"],
        "source": "doc section 5.3 — Prealbumin flat or declining from Day 5 onward",
    },
    "phase_transition_delay": {
        "signal": "Healing phase",
        "severity": "HIGH",
        "interpretation": "Impaired healing — check nutrition, diabetes control, SSI",
        "action": "Dietitian and physician review",
        "escalate_to": ["Dietitian", "Physician"],
        "source": "doc section 5.3 — HMM posterior stuck in Inflammation > Day 7",
    },
    "oxygenation_concern": {
        "signal": "SpO2",
        "severity": "MODERATE",
        "interpretation": "Suboptimal O2 delivery to wound bed — impairs collagen synthesis",
        "action": "Respiratory review and position optimisation",
        "escalate_to": ["Physician"],
        "source": "doc section 5.3 — SpO2 consistently 93-95% during proliferation",
    },
    "glucose_dysregulation": {
        "signal": "Glucose",
        "severity": "HIGH",
        "interpretation": "Neutrophil function impaired — infection risk elevated",
        "action": "Endocrine and pharmacy review",
        "escalate_to": ["Physician", "Pharmacy"],
        "source": "doc section 5.3 — Glucose > 180 mg/dL on 2+ consecutive readings",
    },
    "unexplained_tachycardia": {
        "signal": "Heart rate",
        "severity": "HIGH",
        "interpretation": "Infection, pain, hypovolaemia or pulmonary complication",
        "action": "Immediate physician review",
        "escalate_to": ["Physician"],
        "source": "doc section 5.3 — HR > 100 sustained > 4 hours post Day 2",
    },
    "fever_pattern": {
        "signal": "Temperature",
        "severity": "CRITICAL",
        "interpretation": "Surgical site infection or systemic sepsis",
        "action": "Immediate review",
        "escalate_to": ["Physician"],
        "source": "doc section 5.3 — Temp > 38.5 sustained > 6 hours after Day 2",
    },
}

WITHIN_PATIENT_LIMITATION = (
    "Trajectory deviations are detected within patient, not against a cohort. The "
    "model flags departures from this patient's own established course; it cannot "
    "yet say whether that course is slower than comparable patients. Cohort norms "
    "require accumulated real patient outcomes (doc section 9)."
)

NO_IMAGING_LIMITATION = (
    "No wound imaging available. Deviations are inferred from indirect signal "
    "trajectory. Clinical wound assessment recommended to corroborate."
)


# ===========================================================================
# 2. SIGNAL SERIES — pull one signal out of the timeline as (day, value) pairs
# ===========================================================================

def series(days: Sequence[Dict[str, Any]], group: str, key: str) -> List[Tuple[int, float]]:
    """Every recorded reading of one signal, in day order.

    Sparse by design — a lab drawn every third day yields a third of the points,
    and the GP handles the irregular spacing natively (doc section 5.2). No
    interpolation, no resampling.
    """
    out: List[Tuple[int, float]] = []
    for day in days:
        block = day.get(group)
        if not isinstance(block, dict):
            continue
        value = block.get(key)
        if value in (None, ""):
            continue
        try:
            out.append((int(day.get("day", 0)), float(value)))
        except (TypeError, ValueError):
            continue
    return out


def value_on(points: Sequence[Tuple[int, float]], day: int) -> Optional[float]:
    for d, v in points:
        if d == day:
            return v
    return None


def parse_clock(value: Any) -> Optional[float]:
    """A reading's time of day, in hours. Accepts "14:30" or an ISO timestamp."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[1]
    parts = text.split(":")
    try:
        hours = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
    except (ValueError, IndexError):
        return None
    return hours + minutes / 60.0


def stream(day: Dict[str, Any], signal: str) -> List[Tuple[float, float]]:
    """One day of a wearable signal as (hour, value) pairs, in time order.

    Section 3's continuous stream. Each entry is {"t": "14:15", "v": 38.7}.
    An unworn device yields an empty list, which the caller reports as not
    assessed rather than as clear.
    """
    vitals = day.get("vitals") if isinstance(day.get("vitals"), dict) else {}
    raw = vitals.get(signal)
    if not isinstance(raw, list):
        return []
    readings: List[Tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        hour = parse_clock(item.get("t") or item.get("time"))
        value = item.get("v") if "v" in item else item.get("value")
        if hour is None or value is None:
            continue
        try:
            readings.append((hour, float(value)))
        except (TypeError, ValueError):
            continue
    return sorted(readings)


# ===========================================================================
# 2b. FEATURE ENGINE — doc section 8
#
#   "Feature engine computes rolling statistics (mean, trend slope,
#    variability over 1h, 4h, 12h windows)"
#
# Rolling rather than daily, because a 24-hour mean hides everything that
# matters: an afternoon of fever averages away against a calm morning. Each
# statistic is computed over the window *preceding* each reading, so it is
# available in real time rather than only at the end of the day.
# ===========================================================================

def rolling_features(readings: Sequence[Tuple[float, float]],
                     window_hours: float) -> List[Dict[str, float]]:
    """Trailing-window mean, trend slope and variability at each reading."""
    out: List[Dict[str, float]] = []
    for i, (hour, _) in enumerate(readings):
        window = [(h, v) for h, v in readings[:i + 1] if h > hour - window_hours]
        values = [v for _, v in window]
        times = [h for h, _ in window]
        mean = sum(values) / len(values)
        variability = float(np.std(values)) if len(values) > 1 else 0.0
        # Least-squares slope, in units per hour. Flat when the window has no
        # time spread — a slope through coincident points is undefined.
        if len(window) > 1 and max(times) > min(times):
            slope = float(np.polyfit(times, values, 1)[0])
        else:
            slope = 0.0
        out.append({"t": round(hour, 3), "mean": round(mean, 3),
                    "slope": round(slope, 4), "variability": round(variability, 3)})
    return out


def feature_summary(readings: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Per-window summary of one signal's day, for the output payload.

    The full rolling series is hundreds of rows per signal per day. What a
    reviewer needs is its shape: how high the rolling mean got, how fast the
    signal was moving, how unstable it was.
    """
    if not readings:
        return {"samples": 0}
    summary: Dict[str, Any] = {
        "samples": len(readings),
        "min": round(min(v for _, v in readings), 2),
        "max": round(max(v for _, v in readings), 2),
        "mean": round(sum(v for _, v in readings) / len(readings), 2),
    }
    for window in FEATURE_WINDOWS_HOURS:
        rows = rolling_features(readings, window)
        label = f"{window:g}h"
        summary[label] = {
            "peak_mean": round(max(r["mean"] for r in rows), 2),
            "trough_mean": round(min(r["mean"] for r in rows), 2),
            "max_slope": round(max(r["slope"] for r in rows), 4),
            "mean_variability": round(sum(r["variability"] for r in rows) / len(rows), 3),
        }
    return summary


def sustained_run(readings: Sequence[Tuple[float, float]],
                  above: float) -> Optional[Dict[str, float]]:
    """The longest unbroken stretch above a threshold — section 5.3's "sustained".

    A run breaks on a reading at or below the threshold, and also on a sampling
    gap wider than MAX_SAMPLE_GAP_HOURS: a fever cannot be claimed to have
    persisted across a period when the device was off the patient.

    Returns None when nothing forms a run, which includes the case of isolated
    crossings — a single sample above 100 bpm in an otherwise calm day is normal
    physiological variation, not sustained tachycardia.
    """
    best: Optional[Dict[str, float]] = None
    run: List[Tuple[float, float]] = []

    def close(current: List[Tuple[float, float]]) -> None:
        nonlocal best
        if len(current) < MIN_RUN_SAMPLES:
            return
        span = current[-1][0] - current[0][0]
        if span <= 0:
            return
        if best is None or span > best["hours"]:
            best = {"hours": round(span, 2),
                    "start": round(current[0][0], 2),
                    "end": round(current[-1][0], 2),
                    "peak": round(max(v for _, v in current), 2),
                    "samples": len(current)}

    for hour, value in readings:
        if value > above:
            if run and hour - run[-1][0] > MAX_SAMPLE_GAP_HOURS:
                close(run)
                run = []
            run.append((hour, value))
        else:
            close(run)
            run = []
    close(run)
    return best


# ===========================================================================
# 3. THE GAUSSIAN PROCESS — doc section 5.1
#
# A GP does not fit a curve through the points. It describes a whole family of
# plausible curves consistent with what has been seen, and reports where the
# next reading should fall and how tightly. That interval is the deviation test:
# a value outside it is one the patient's own trajectory does not explain.
#
# Two properties earn it here over ordinary regression:
#   - irregular spacing is native. Labs every 2-3 days need no interpolation.
#   - it reports uncertainty. Four readings give a wide interval, twelve give a
#     tight one, and the flag threshold adjusts itself accordingly.
# ===========================================================================

def matern32(a: np.ndarray, b: np.ndarray, length_scale: float) -> np.ndarray:
    """Matern 3/2 covariance between two sets of times.

    Says how much two days inform each other: 1.0 for the same day, decaying
    smoothly to 0 as they move apart. Matern 3/2 rather than the smoother RBF
    because biological trajectories are continuous but not smooth — they kink,
    and an RBF would over-smooth those kinks away.
    """
    r = np.abs(a - b)
    s = np.sqrt(3.0) * r / length_scale
    return (1.0 + s) * np.exp(-s)


def gp_predict(
    train_days: Sequence[float],
    train_values: Sequence[float],
    query_day: float,
    length_scale: float = GP_LENGTH_SCALE_DAYS,
    noise_fraction: float = GP_NOISE_FRACTION,
) -> Tuple[float, float]:
    """Predict one signal on one day from earlier readings.

    Returns (expected value, standard deviation).

    Values are standardised first so one set of hyperparameters works for CRP in
    mg/L and prealbumin in mg/L alike, then converted back at the end.
    """
    t = np.asarray(train_days, dtype=float)
    y = np.asarray(train_values, dtype=float)

    centre = float(y.mean())
    spread = float(y.std())
    if spread < 1e-9:          # a perfectly flat history has no scale of its own
        spread = 1.0
    z = (y - centre) / spread

    # Covariance among the training days, plus noise on the diagonal. The small
    # jitter keeps the matrix invertible when two readings nearly coincide.
    K = matern32(t[:, None], t[None, :], length_scale)
    K += (noise_fraction ** 2 + 1e-8) * np.eye(len(t))

    # Covariance between the day we are predicting and each training day.
    k_star = matern32(np.array([[query_day]]), t[None, :], length_scale).ravel()

    try:
        alpha = np.linalg.solve(K, z)
        v = np.linalg.solve(K, k_star)
    except np.linalg.LinAlgError:
        return centre, spread

    mean_z = float(k_star @ alpha)
    var_z = max(1.0 - float(k_star @ v), 1e-9)   # 1.0 is k(t*,t*) for this kernel

    mean = mean_z * spread + centre
    sd = float(np.sqrt(var_z + noise_fraction ** 2) * spread)
    return mean, sd


def gp_trajectory(points: Sequence[Tuple[int, float]]) -> List[Dict[str, Any]]:
    """Walk the signal forward, judging each day from the days before it.

    Causal by construction: to judge day t the GP is fitted on days 0..t-1 and
    never sees day t itself. A breach is therefore something a clinician could
    have been told that morning, not hindsight.
    """
    out: List[Dict[str, Any]] = []
    for i, (day, observed) in enumerate(points):
        history = points[:i]
        if len(history) < GP_MIN_HISTORY:
            out.append({"day": day, "observed": round(observed, 2),
                        "status": "insufficient_history"})
            continue

        mean, sd = gp_predict([d for d, _ in history], [v for _, v in history], float(day))
        low, high = mean - GP_CREDIBLE_Z * sd, mean + GP_CREDIBLE_Z * sd
        if GP_NON_NEGATIVE:
            low = max(0.0, low)
        out.append({
            "day": day,
            "observed": round(observed, 2),
            "expected": round(mean, 2),
            "credible_interval": [round(low, 2), round(high, 2)],
            "status": "above" if observed > high else "below" if observed < low else "within",
        })
    return out


# ===========================================================================
# 4. THE SEVEN DEVIATIONS — doc section 5.3
#    Each returns a list of findings. A finding names the day, what was seen,
#    what was expected, and how it was detected.
# ===========================================================================

def _collapse(findings: List[Dict[str, Any]], max_gap: int = 3) -> List[Dict[str, Any]]:
    """Fold a run of consecutive daily findings into one episode.

    A five-day plateau is one clinical event, not five. Reporting each sliding
    window separately buries the physician in restatements of the same problem.
    The episode is reported on the day it began, carrying how long it ran.

    max_gap allows for sparse labs: prealbumin drawn every third day still forms
    one episode rather than three.
    """
    if not findings:
        return []
    ordered = sorted(findings, key=lambda f: f["day"])
    episodes: List[Dict[str, Any]] = []
    current = dict(ordered[0])
    last_day = current["day"]
    span = 1

    for item in ordered[1:]:
        if item["day"] - last_day <= max_gap:
            span += 1
            last_day = item["day"]
        else:
            current["days_in_episode"] = span
            current["episode_last_day"] = last_day
            episodes.append(current)
            current, last_day, span = dict(item), item["day"], 1

    current["days_in_episode"] = span
    current["episode_last_day"] = last_day
    episodes.append(current)
    return episodes


def _finding(name: str, day: int, finding: str, detected_by: str,
             approximated: bool = False, **extra: Any) -> Dict[str, Any]:
    spec = DEVIATIONS[name]
    return {
        "deviation": name,
        "signal": spec["signal"],
        "day": day,
        "finding": finding,
        "severity": spec["severity"],
        "detected_by": detected_by,
        "approximated": approximated,
        "clinical_interpretation": spec["interpretation"],
        "recommended_action": spec["action"],
        "source": spec["source"],
        **extra,
    }


def detect_inflammation_not_resolving(crp: Sequence[Tuple[int, float]],
                                      trajectory: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CRP plateau or re-elevation after Day 3.

    Two detectors, because they catch different failures. The GP catches a sharp
    break — CRP was falling on a smooth curve and today's value is above what
    that curve allows. The plateau rule catches the slow version the GP will
    happily absorb: CRP that simply stops coming down.
    """
    found: List[Dict[str, Any]] = []

    for row in trajectory:
        # The floor gates the GP path too. On a flat series the credible interval
        # collapses to almost nothing, so a CRP of 8.6 against an expected 8.33
        # reads as a breach — arithmetically true, clinically meaningless.
        if row["observed"] < CRP_CLINICAL_FLOOR:
            continue
        if row["day"] > CRP_WATCH_FROM_DAY and row.get("status") == "above":
            found.append(_finding(
                "inflammation_not_resolving", row["day"],
                f"CRP {row['observed']} against an expected {row['expected']} "
                f"(95% CI {row['credible_interval'][0]}–{row['credible_interval'][1]}). "
                "Re-elevation above this patient's own trajectory.",
                "gaussian_process",
                observed=row["observed"], expected=row["expected"],
                credible_interval=row["credible_interval"]))

    flagged_days = {f["day"] for f in found}
    # Only readings that are actually abnormal can constitute a failure to resolve.
    late = [(d, v) for d, v in crp if d > CRP_WATCH_FROM_DAY and v >= CRP_CLINICAL_FLOOR]
    for i in range(CRP_PLATEAU_READINGS - 1, len(late)):
        window = late[i - CRP_PLATEAU_READINGS + 1: i + 1]
        # Measured across the whole window, not between consecutive readings. A
        # CRP falling 8% per draw is resolving; testing each step against a 10%
        # threshold would call that a plateau and raise a HIGH alert on a patient
        # who is plainly recovering.
        fall = (window[0][1] - window[-1][1]) / window[0][1] if window[0][1] else 0.0
        if fall <= CRP_DECLINE_PCT and window[-1][0] not in flagged_days:
            found.append(_finding(
                "inflammation_not_resolving", window[-1][0],
                f"CRP has not declined across {CRP_PLATEAU_READINGS} consecutive readings "
                f"({' → '.join(str(v) for _, v in window)}). Inflammation not resolving.",
                "plateau_rule",
                observed=window[-1][1]))
            flagged_days.add(window[-1][0])

    return _collapse(found)


def detect_protein_synthesis_impairment(prealbumin: Sequence[Tuple[int, float]],
                                        trajectory: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prealbumin flat or declining from Day 5 onward.

    Prealbumin should be *rising* during proliferation — it is the marker of the
    protein synthesis that builds collagen. Flat is already a finding here, which
    is why this is not simply a fall detector.
    """
    found: List[Dict[str, Any]] = []

    for row in trajectory:
        # Same gate: a dip below trajectory at a healthy prealbumin level is not
        # protein synthesis impairment.
        if row["observed"] > PREALBUMIN_CLINICAL_FLOOR:
            continue
        if row["day"] >= PREALBUMIN_WATCH_FROM_DAY and row.get("status") == "below":
            found.append(_finding(
                "protein_synthesis_impairment", row["day"],
                f"Prealbumin {row['observed']} against an expected {row['expected']} "
                f"(95% CI {row['credible_interval'][0]}–{row['credible_interval'][1]}). "
                "Below this patient's own trajectory during collagen synthesis.",
                "gaussian_process",
                observed=row["observed"], expected=row["expected"],
                credible_interval=row["credible_interval"]))

    flagged_days = {f["day"] for f in found}
    # Flat prealbumin at a healthy level is not impairment — there is nothing to recover.
    late = [(d, v) for d, v in prealbumin
            if d >= PREALBUMIN_WATCH_FROM_DAY and v <= PREALBUMIN_CLINICAL_FLOOR]
    for i in range(PREALBUMIN_FLAT_READINGS - 1, len(late)):
        window = late[i - PREALBUMIN_FLAT_READINGS + 1: i + 1]
        # Same reasoning as CRP: measured across the window. Prealbumin climbing
        # steadily but gently is a recovering patient, not a stalled one.
        rise = (window[-1][1] - window[0][1]) / window[0][1] if window[0][1] else 0.0
        if rise <= PREALBUMIN_RISE_PCT and window[-1][0] not in flagged_days:
            found.append(_finding(
                "protein_synthesis_impairment", window[-1][0],
                f"Prealbumin flat or falling across {PREALBUMIN_FLAT_READINGS} readings "
                f"({' → '.join(str(v) for _, v in window)}). Protein synthesis is not "
                "accelerating as expected.",
                "flat_rule",
                observed=window[-1][1]))
            flagged_days.add(window[-1][0])

    return _collapse(found)


def detect_phase_transition_delay(layer1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """HMM posterior stuck in Inflammation past Day 7.

    Reads Layer 1's output directly. Reported once, on the first day it is true,
    rather than every day after — a delay is one clinical event, not a daily one.
    """
    for row in layer1.get("daily", []):
        if row["day"] > PHASE_DELAY_AFTER_DAY and row["phase"] == "INFLAMMATION":
            return [_finding(
                "phase_transition_delay", row["day"],
                f"Still estimated in inflammation on day {row['day']} "
                f"({row['confidence']:.0%} confidence), past the expected "
                f"day-{PHASE_DELAY_AFTER_DAY} transition window.",
                "layer1_posterior",
                observed=row["confidence"])]
    return []


def detect_oxygenation_concern(days: Sequence[Dict[str, Any]],
                               layer1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SpO2 consistently 93-95% during proliferation (section 5.3).

    "Consistently" is read against the 12-hour rolling mean from the feature
    engine, not against individual samples. A pulse oximeter dips momentarily
    whenever a patient moves; what matters for wound-bed oxygenation is whether
    delivery sits low for hours.

    Phase-conditioned: the same SpO2 is only this finding while collagen is being
    laid down, because that is when oxygen delivery limits synthesis.
    """
    phase_by_day = {row["day"]: row["phase"] for row in layer1.get("daily", [])}
    low, high = SPO2_CONCERN_RANGE
    run: List[int] = []
    found: List[Dict[str, Any]] = []

    for day in days:
        number = int(day.get("day", 0))
        readings = stream(day, "spo2")
        if not readings or phase_by_day.get(number) != "PROLIFERATION":
            run = []
            continue

        rolling = rolling_features(readings, 12.0)
        in_band = [r["mean"] for r in rolling if low <= r["mean"] < high]
        consistent = len(in_band) >= SPO2_CONSISTENT_FRACTION * len(rolling)

        if consistent:
            run.append(number)
            if len(run) == SPO2_CONSECUTIVE_DAYS:
                found.append(_finding(
                    "oxygenation_concern", number,
                    f"12-hour rolling mean SpO2 in the {low:.0f}-{high:.0f}% band for "
                    f"{len(in_band)/len(rolling):.0%} of day {number} and the day before "
                    f"(mean {sum(in_band)/len(in_band):.1f}%), during proliferation.",
                    "rolling_mean", observed=round(sum(in_band) / len(in_band), 1),
                    samples=len(readings)))
        else:
            run = []
    return _collapse(found)


def detect_glucose_dysregulation(days: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Glucose > 180 mg/dL on 2+ consecutive readings.

    The one hour-free rule in section 5.3 — it counts readings, and a daily
    glucose is a reading. No approximation is involved.
    """
    run: List[Tuple[int, float]] = []
    found: List[Dict[str, Any]] = []

    for day in days:
        labs = day.get("labs") if isinstance(day.get("labs"), dict) else {}
        glucose = labs.get("glucose")
        if glucose is not None and float(glucose) > GLUCOSE_HIGH:
            run.append((int(day.get("day", 0)), float(glucose)))
            if len(run) == GLUCOSE_CONSECUTIVE:
                found.append(_finding(
                    "glucose_dysregulation", run[-1][0],
                    f"Glucose above {GLUCOSE_HIGH:.0f} mg/dL on {GLUCOSE_CONSECUTIVE} "
                    f"consecutive readings ({', '.join(str(v) for _, v in run)}).",
                    "threshold_rule", observed=run[-1][1]))
        else:
            run = []
    return _collapse(found)


def _sustained_deviation(days: Sequence[Dict[str, Any]], *, name: str, signal: str,
                         threshold: float, required_hours: float, from_day: int,
                         unit: str) -> List[Dict[str, Any]]:
    """Shared engine for the two deviations section 5.3 specifies in hours.

    With a continuous stream the question is simply answered: find the longest
    unbroken stretch above threshold and compare it to the criterion. There is no
    inference and no approximation, so there is no severity ladder — the finding
    either meets section 5.3 or it does not.

    Isolated crossings fire nothing. In a 96-reading day, normal physiological
    variation puts individual samples over any threshold; "sustained" means an
    unbroken run, and a lone sample forms no run at all.
    """
    spec = DEVIATIONS[name]
    found: List[Dict[str, Any]] = []

    for day in days:
        number = int(day.get("day", 0))
        if number <= from_day:
            continue
        readings = stream(day, signal)
        if not readings:
            continue

        run = sustained_run(readings, threshold)
        if run is None or run["hours"] < required_hours:
            continue

        found.append(_finding(
            name, number,
            f"{spec['signal']} above {threshold:g}{unit} continuously for "
            f"{run['hours']:.1f} hours ({_clock(run['start'])}-{_clock(run['end'])}, "
            f"peak {run['peak']:g}{unit}, {run['samples']} readings), meeting the "
            f"{required_hours:g}-hour criterion.",
            "sustained_run", observed=run["peak"], measured_hours=run["hours"],
            window=[_clock(run["start"]), _clock(run["end"])],
            samples_in_run=run["samples"]))

    return _collapse(found)


def _clock(hour: float) -> str:
    return f"{int(hour):02d}:{int(round((hour % 1) * 60)):02d}"


def detect_unexplained_tachycardia(days: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """HR > 100 sustained > 4 hours, after Day 2 (section 5.3)."""
    return _sustained_deviation(days, name="unexplained_tachycardia", signal="hr",
                                threshold=HR_HIGH, required_hours=HR_SUSTAINED_HOURS,
                                from_day=HR_WATCH_FROM_DAY, unit=" bpm")


def detect_fever_pattern(days: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Temp > 38.5 sustained > 6 hours, after Day 2 (section 5.3).

    The only CRITICAL deviation in the framework.
    """
    return _sustained_deviation(days, name="fever_pattern", signal="temp",
                                threshold=TEMP_HIGH, required_hours=TEMP_SUSTAINED_HOURS,
                                from_day=TEMP_WATCH_FROM_DAY, unit="°C")


# ===========================================================================
# 5. THE DETECTOR — timeline + Layer 1 result in, healing status out
# ===========================================================================

def require_upstream(result: Any, name: str, key: str, producer: str) -> None:
    """Check an upstream layer's result is present and the right shape.

    Exists because `"daily" in layer1` passes for {"daily": None}, and the failure
    then surfaces as "'NoneType' object is not iterable" from inside a loop — a
    message that tells the caller nothing about which argument was wrong.
    """
    if not isinstance(result, dict):
        raise ValueError(f"{name} result must be an object — pass the output of "
                         f"{producer}(), got {type(result).__name__}")
    if not isinstance(result.get(key), list):
        raise ValueError(f"{name} result is missing a valid '{key}' list — pass the "
                         f"output of {producer}()")


def longest_assessable_span(day: Dict[str, Any], signal: str) -> float:
    """Longest stretch, in hours, that this day's sampling could evidence.

    A run breaks on a gap wider than MAX_SAMPLE_GAP_HOURS, so readings spaced
    further apart than that can never form a sustained run no matter what they
    say. This measures the ceiling imposed by SAMPLING ALONE, ignoring the
    values entirely.
    """
    readings = stream(day, signal)
    if len(readings) < MIN_RUN_SAMPLES:
        return 0.0
    best = span = 0.0
    for (prev_h, _), (hour, _) in zip(readings, readings[1:]):
        gap = hour - prev_h
        span = span + gap if gap <= MAX_SAMPLE_GAP_HOURS else 0.0
        best = max(best, span)
    return round(best, 2)


def _sustained_check_coverage(days: Sequence[Dict[str, Any]], signal: str,
                              required_hours: float) -> Tuple[bool, str, float]:
    """Could a sustained-run check on `signal` have fired on ANY day?

    Presence of readings is not enough, and treating it as enough is how a
    ward taking observations every six hours got the same output as a patient
    with no fever: `days_with_data` was 11, the check reported "clear", and the
    physician read absence of a flag as absence of fever. Sampling that cannot
    span the required window makes the finding UNDETECTABLE, which is a
    different statement from "not found" and has to be reported as one.
    """
    spans = [longest_assessable_span(d, signal) for d in days]
    best = max(spans) if spans else 0.0
    if best >= required_hours:
        return True, "", best
    if not any(stream(d, signal) for d in days):
        return False, f"no {signal} stream — wearable produced no readings", best
    return False, (
        f"{signal} sampling too sparse — the longest unbroken window any day "
        f"could evidence is {best:.1f} h, and this check needs "
        f"{required_hours:.0f} h. Readings more than "
        f"{MAX_SAMPLE_GAP_HOURS:.0f} h apart cannot form a sustained run."
    ), best


def _coverage(days: Sequence[Dict[str, Any]], layer1: Dict[str, Any],
              trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Whether each of the seven checks could run at all, and on what evidence.

    This is the part that stops silence being ambiguous. Without it, a patient
    with no thermometer produces exactly the same output as a patient with no
    fever — and fever is the one CRITICAL deviation in the framework.
    """
    last_day = int(days[-1].get("day", 0))
    # Stream coverage per signal: how many readings arrived, and on how many
    # days the device produced anything at all. An unworn wearable is the reason
    # a check cannot run, and the reason has to reach the physician.
    coverage_by_signal: Dict[str, Dict[str, int]] = {}
    for signal in STREAM_SIGNALS:
        per_day = [len(stream(d, signal)) for d in days]
        coverage_by_signal[signal] = {
            "total_readings": sum(per_day),
            "days_with_data": sum(1 for n in per_day if n),
            "days_total": len(per_day),
        }

    def lab_days(key: str) -> int:
        return len(series(days, "labs", key))

    hr_ok, hr_why, hr_span = _sustained_check_coverage(days, "hr", HR_SUSTAINED_HOURS)
    temp_ok, temp_why, temp_span = _sustained_check_coverage(days, "temp", TEMP_SUSTAINED_HOURS)

    def judged(signal: str) -> int:
        return sum(1 for r in trajectories.get(signal, [])
                   if r.get("status") != "insufficient_history")

    out: Dict[str, Dict[str, Any]] = {}

    for name, needs, ok, why in (
        ("inflammation_not_resolving", "CRP",
         lab_days("crp") >= 3,
         f"only {lab_days('crp')} CRP readings — at least 3 are needed"),
        ("protein_synthesis_impairment", "Prealbumin",
         lab_days("prealbumin") >= 3,
         f"only {lab_days('prealbumin')} prealbumin readings — at least 3 are needed"),
        ("phase_transition_delay", "Layer 1 phase estimate",
         bool(layer1.get("daily")) and last_day > PHASE_DELAY_AFTER_DAY,
         f"stay ends on day {last_day}, before the day-{PHASE_DELAY_AFTER_DAY} window"),
        ("oxygenation_concern", "SpO2",
         coverage_by_signal["spo2"]["days_with_data"] > 0,
         "no SpO2 stream — wearable produced no readings"),
        ("glucose_dysregulation", "Glucose",
         lab_days("glucose") >= GLUCOSE_CONSECUTIVE,
         f"only {lab_days('glucose')} glucose readings — at least {GLUCOSE_CONSECUTIVE} are needed"),
        ("unexplained_tachycardia", "Heart rate", hr_ok, hr_why),
        ("fever_pattern", "Temperature", temp_ok, temp_why),
    ):
        signal_key = {"oxygenation_concern": "spo2", "unexplained_tachycardia": "hr",
                      "fever_pattern": "temp"}.get(name)
        out[name] = {
            "requires": needs,
            "assessable": bool(ok),
            "reason": None if ok else why,
            "stream": coverage_by_signal.get(signal_key) if signal_key else None,
            "severity_if_detected": DEVIATIONS[name]["severity"],
        }
        if name == "unexplained_tachycardia":
            out[name]["longest_assessable_span_hours"] = hr_span
            out[name]["required_hours"] = HR_SUSTAINED_HOURS
        elif name == "fever_pattern":
            out[name]["longest_assessable_span_hours"] = temp_span
            out[name]["required_hours"] = TEMP_SUSTAINED_HOURS

    # The GP needs more history than the rules do; note when only rules could run.
    for name, signal in (("inflammation_not_resolving", "crp"),
                         ("protein_synthesis_impairment", "prealbumin")):
        if out[name]["assessable"] and judged(signal) == 0:
            out[name]["trajectory_model"] = (
                f"not fitted — fewer than {GP_MIN_HISTORY} prior readings; "
                "rule-based checks only")
        elif out[name]["assessable"]:
            out[name]["trajectory_model"] = f"fitted on {judged(signal)} judged readings"

    return out


def detect_deviations(timeline: Dict[str, Any], layer1: Dict[str, Any]) -> Dict[str, Any]:
    """Run all seven deviation checks and consolidate them.

    Takes Layer 1's result rather than computing it, so the orchestrator owns the
    chaining and both layers stay independently testable.
    """
    days = validate_days(timeline)
    require_upstream(layer1, "layer1", "daily", "estimate_phase")

    trajectories: Dict[str, List[Dict[str, Any]]] = {}
    for signal in GP_SIGNALS:
        trajectories[signal] = gp_trajectory(series(days, "labs", signal))

    # Section 8's feature engine: rolling statistics over the wearable stream.
    features = [
        {"day": int(d.get("day", 0)),
         **{signal: feature_summary(stream(d, signal)) for signal in STREAM_SIGNALS}}
        for d in days
    ]

    deviations: List[Dict[str, Any]] = []
    deviations += detect_inflammation_not_resolving(series(days, "labs", "crp"), trajectories["crp"])
    deviations += detect_protein_synthesis_impairment(
        series(days, "labs", "prealbumin"), trajectories["prealbumin"])
    deviations += detect_phase_transition_delay(layer1)
    deviations += detect_oxygenation_concern(days, layer1)
    deviations += detect_glucose_dysregulation(days)
    deviations += detect_unexplained_tachycardia(days)
    deviations += detect_fever_pattern(days)

    # Worst severity first, then most recent — a physician reads the top of the list.
    deviations.sort(key=lambda d: (SEVERITY_ORDER.index(d["severity"]), d["day"]), reverse=True)

    # Section 7.1 shows a healing_status field but never defines how it is derived,
    # so this rule is a choice rather than a reading of the document.
    #
    # It requires corroboration before declaring a patient delayed. The GP flags at
    # a 95% credible interval, which by construction misclassifies about one
    # reading in twenty — so a single isolated HIGH is as likely to be chance as
    # signal. Two independent findings, or anything CRITICAL, is not.
    severities = [d["severity"] for d in deviations]
    high_count = severities.count("HIGH")
    if "CRITICAL" in severities:
        status = "CRITICAL"
    elif high_count >= 2:
        status = "DELAYED"
    elif high_count == 1 or "MODERATE" in severities:
        status = "AT_RISK"
    else:
        status = "ON_TRACK"

    escalate: List[str] = []
    for d in deviations:
        for who in DEVIATIONS[d["deviation"]]["escalate_to"]:
            if who not in escalate:
                escalate.append(who)

    coverage = _coverage(days, layer1, trajectories)
    detected_names = {d["deviation"] for d in deviations}
    for name, entry in coverage.items():
        entry["status"] = ("detected" if name in detected_names
                           else "clear" if entry["assessable"] else "not_assessed")

    not_assessed = [n for n, e in coverage.items() if e["status"] == "not_assessed"]
    critical_gaps = [n for n in not_assessed
                     if DEVIATIONS[n]["severity"] == "CRITICAL"]

    limitations = [NO_IMAGING_LIMITATION, WITHIN_PATIENT_LIMITATION]

    for name in critical_gaps:
        limitations.insert(0, (
            f"NOT ASSESSED — {name.replace('_', ' ')} ({DEVIATIONS[name]['signal']}): "
            f"{coverage[name]['reason']}. This is a CRITICAL check and it did not run. "
            "Absence of this finding must not be read as absence of the condition."))

    other_gaps = [n for n in not_assessed if n not in critical_gaps]
    if other_gaps:
        limitations.append(
            "Not assessed: " + "; ".join(
                f"{n.replace('_', ' ')} ({coverage[n]['reason']})" for n in other_gaps)
            + ". These checks did not run and are not reported as clear.")

    thin_stream = [sig for sig, cov in
                   ((s2, coverage[n]["stream"]) for s2, n in
                    (("temperature", "fever_pattern"),
                     ("heart rate", "unexplained_tachycardia"),
                     ("SpO2", "oxygenation_concern")))
                   if cov and 0 < cov["days_with_data"] < cov["days_total"]]
    if thin_stream:
        limitations.append(
            f"Wearable coverage is incomplete for {', '.join(thin_stream)} — the device "
            "produced no readings on some days. Sustained-duration criteria could not be "
            "evaluated on those days.")

    return {
        "patient_ref": timeline.get("patient_ref"),
        "days_post_op": int(days[-1].get("day", 0)),
        "current_phase": layer1.get("current_phase"),
        "healing_status": status,
        "deviations": deviations,
        "deviation_count": len(deviations),
        "escalation_required": bool(deviations),
        "escalate_to": escalate,
        "gp_trajectories": trajectories,
        "assessment": coverage,
        "checks_not_assessed": not_assessed,
        "critical_checks_not_assessed": critical_gaps,
        "feature_engine": features,
        "limitations": limitations,
        "model_version": MODEL_VERSION,
    }


# ===========================================================================
# 6. TERMINAL OUTPUT
# ===========================================================================

STATUS_MARK = {"ON_TRACK": "", "AT_RISK": "  (!)", "DELAYED": "  (!!)", "CRITICAL": "  (!!!)"}


def render(result: Dict[str, Any]) -> str:
    lines = ["=" * 78,
             " POST-OP PHASE 1 - LAYER 2 : DEVIATION DETECTOR",
             "=" * 78,
             f" Patient        {result.get('patient_ref') or '-'}",
             f" Day post-op    {result['days_post_op']}",
             f" Phase (L1)     {result.get('current_phase') or '-'}",
             "",
             f" HEALING STATUS   {result['healing_status']}{STATUS_MARK[result['healing_status']]}",
             ""]

    if result["deviations"]:
        lines.append(f" DEVIATIONS   ({result['deviation_count']} found, worst first)")
        for d in result["deviations"]:
            tag = " [approximated]" if d["approximated"] else ""
            lines += ["",
                      f"   {d['severity']:<9} day {d['day']:<3} {d['signal']}{tag}",
                      f"     {d['finding']}",
                      f"     -> {d['recommended_action']}",
                      f"     detected by: {d['detected_by']}   |   {d['source']}"]
    else:
        lines.append(" DEVIATIONS   none detected")

    lines += ["", " GP TRAJECTORY  (expected vs observed, fitted on prior days only)"]
    for signal, rows in result["gp_trajectories"].items():
        judged = [r for r in rows if r.get("status") != "insufficient_history"]
        if not judged:
            lines.append(f"   {signal:<12} not enough readings to fit")
            continue
        lines.append(f"   {signal}")
        for r in rows:
            if r.get("status") == "insufficient_history":
                lines.append(f"     day {r['day']:>3}   {r['observed']:>8}   (building history)")
            else:
                flag = {"above": "  <-- ABOVE", "below": "  <-- BELOW", "within": ""}[r["status"]]
                ci = r["credible_interval"]
                lines.append(f"     day {r['day']:>3}   {r['observed']:>8}   "
                             f"expected {r['expected']:>8}   CI [{ci[0]}, {ci[1]}]{flag}")

    lines += ["", " CHECK COVERAGE   (what ran, on what evidence)"]
    for name, entry in result["assessment"].items():
        mark = {"detected": "FLAGGED ", "clear": "clear   ",
                "not_assessed": "NOT RUN "}[entry["status"]]
        cov = entry.get("stream")
        detail = (f"{cov['total_readings']} readings over "
                  f"{cov['days_with_data']}/{cov['days_total']} days") if cov else ""
        if entry["status"] == "not_assessed":
            detail = entry["reason"]
        lines.append(f"   {mark} {name:<30} {detail}")

    if result["escalate_to"]:
        lines += ["", " ESCALATE TO   " + ", ".join(result["escalate_to"])]

    lines += ["", " LIMITATIONS"]
    lines += [f"   * {text}" for text in result["limitations"]]
    lines += ["", f" {result['model_version']}"]
    return "\n".join(lines)


# ===========================================================================
# 7. SERVICE — stateless, same shape as Layer 1
# ===========================================================================

try:
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    class DetectRequest(BaseModel):
        model_config = ConfigDict(extra="allow")
        patient_ref: Optional[str] = None
        days: List[Dict[str, Any]] = Field(default_factory=list)
        clinician_phase_overrides: List[Dict[str, Any]] = Field(default_factory=list)
        layer1: Optional[Dict[str, Any]] = None

    app = FastAPI(title="Post-op Phase 1 - Layer 2", version=MODEL_VERSION)

    def _token() -> Optional[str]:
        for name in ("POSTOP_PHASE1_SERVICE_API_KEY", "FOODHAK_API_TOKEN", "LANGGRAPH_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "postop-phase1-layer2",
                "model_version": MODEL_VERSION,
                "deviations": list(DEVIATIONS),
                "gp_signals": list(GP_SIGNALS),
                "auth_required": _token() is not None}

    @app.post("/postop/phase1/detect-deviations")
    async def detect(
        request: DetectRequest,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        expected = _token()
        if expected:
            received = x_api_key
            if authorization and authorization.lower().startswith("bearer "):
                received = authorization[7:].strip()
            if received != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing service token")
        payload = request.model_dump()
        try:
            # Layer 1 output may be supplied; if not, run it here so the endpoint
            # is usable on its own. The orchestrator will always supply it.
            layer1 = payload.get("layer1") or estimate_phase(payload)
            return detect_deviations(payload, layer1)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

except ImportError as exc:  # the CLI must still work without FastAPI installed
    _MISSING_DEPENDENCY = (
        f"The HTTP service is disabled: {exc}. "
        "Install the service dependencies with:  pip install -r requirements.txt"
    )
    print(f"layer2_deviation_detector: {_MISSING_DEPENDENCY}\n"
          "The command-line tool still works.", file=sys.stderr)

    async def app(scope, receive, send):  # type: ignore[misc]
        """Stand-in so an ASGI server reports the real problem rather than
        'NoneType is not callable'."""
        if scope.get("type") != "http":
            return
        body = json.dumps({"detail": _MISSING_DEPENDENCY}).encode()
        await send({"type": "http.response.start", "status": 503,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


# ===========================================================================
# 8. CLI
# ===========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect deviations from expected healing trajectory.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--timeline", type=Path, help="Path to a timeline JSON file")
    source.add_argument("--scenario", choices=SCENARIOS, help="Generate a synthetic patient")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Raw JSON instead of the table")
    args = parser.parse_args(argv)

    if args.timeline:
        timeline = json.loads(args.timeline.read_text())
    else:
        timeline = generate_timeline(args.scenario or "normal", args.days, args.seed)
        print("note: Layer 1's synthetic generator produces daily aggregates, not the\n"
              "      wearable streams doc section 3 describes, so the three stream-backed\n"
              "      checks will report NOT RUN. Use --timeline with stream data to\n"
              "      exercise them.\n", file=sys.stderr)

    try:
        layer1 = estimate_phase(timeline)
        result = detect_deviations(timeline, layer1)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
