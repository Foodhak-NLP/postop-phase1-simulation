"""
Post-op Phase 1 — Layer 1: Healing Phase Estimator.

Implements section 4 of Phase_1_Wound_Recovery.docx.

    "given the observable signals available, which healing phase is this
     patient currently in?"

    HAEMOSTASIS  ->  INFLAMMATION  ->  PROLIFERATION  ->  REMODELLING
       0-3 h           0-5 d             5-21 d            21+ d

The phase is hidden. There is no wound imaging in this system, so it must be
inferred from indirect daily signals — which is what a Hidden Markov Model is
built for (doc section 4.2).

The model is two probability tables and one recursion:

    A[i][j]  = P(phase tomorrow = j | phase today = i)      section 1
    B[k][i]  = P(observation k | phase i)                    section 1
    forward recursion combines them, day by day              section 3

Run it:

    python3 layer1_phase_estimator.py --scenario normal
    python3 layer1_phase_estimator.py --timeline patient.json --json
    uvicorn layer1_phase_estimator:app --port 8010 --reload
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_VERSION = "phase1-hmm-v1"

# =========================================================================
# 1. THE MODEL
#    Two tables. Everything clinical lives here; nothing below this section
#    contains a medical number.
# =========================================================================

PHASES: Tuple[str, ...] = ("HAEMOSTASIS", "INFLAMMATION", "PROLIFERATION", "REMODELLING")
PHASE_INDEX = {name: i for i, name in enumerate(PHASES)}
H, I, P, R = 0, 1, 2, 3

# --- A: transition matrix, per day -----------------------------------------
#
# Derived from the phase durations in doc section 2. If a phase lasts N days on
# average, the chance of leaving it on any given day is 1/N:
#
#   INFLAMMATION   0-5 days   -> 1/5  = 0.20 leave, 0.80 stay
#   PROLIFERATION  5-21 days  -> 1/16 = 0.06 leave, 0.94 stay
#   REMODELLING    21+ days   -> no end inside the stay, so near-absorbing
#   HAEMOSTASIS    0-3 hours  -> always gone by day 1
#
# Backward transitions are allowed at low probability: a surgical site
# infection genuinely re-enters the inflammatory phase, and Layer 3 reads this
# phase to set protein and zinc targets. The proliferation row is 0.94 minus
# the 0.03 regression, so it sums to 1.
#
# Nothing returns to HAEMOSTASIS — healing does not restart clot formation.
TRANSITION_MATRIX = np.array(
    [
        [0.00, 1.00, 0.00, 0.00],
        [0.00, 0.80, 0.20, 0.00],
        [0.00, 0.03, 0.91, 0.06],
        [0.00, 0.00, 0.02, 0.98],
    ]
)

# --- B: emission tables ----------------------------------------------------
#
# One table per signal. Rows are phases, columns are the symbols below, and
# every row sums to 1. Read a cell as: "in this phase, how often do we see
# this?" These are literature priors (doc section 4.3), not fitted values.
SYMBOLS: Dict[str, Tuple[str, ...]] = {
    "crp_velocity": ("rising", "plateau", "declining"),
    "wbc_pattern": ("normal", "neutrophilia", "elevated"),
    "prealbumin_trend": ("falling", "flat", "rising"),
    "albumin_trend": ("falling", "flat", "rising"),
    "temp_pattern": ("afebrile", "low_grade", "febrile"),
    "hr_trend": ("normal", "elevated", "sustained_tachycardia"),
    "pain_trajectory": ("improving", "static", "escalating"),
}

EMISSIONS: Dict[str, np.ndarray] = {
    # Rising CRP is the inflammation signature; actively declining CRP is the
    # proliferation signature; flat-and-normal is remodelling.
    "crp_velocity": np.array([
        [0.60, 0.30, 0.10],
        [0.70, 0.22, 0.08],
        [0.08, 0.20, 0.72],
        [0.05, 0.60, 0.35],
    ]),
    "wbc_pattern": np.array([
        [0.30, 0.45, 0.25],
        [0.15, 0.55, 0.30],
        [0.60, 0.30, 0.10],
        [0.85, 0.12, 0.03],
    ]),
    "prealbumin_trend": np.array([
        [0.50, 0.40, 0.10],
        [0.55, 0.35, 0.10],
        [0.15, 0.30, 0.55],
        [0.10, 0.45, 0.45],
    ]),
    "albumin_trend": np.array([
        [0.55, 0.35, 0.10],
        [0.50, 0.38, 0.12],
        [0.20, 0.45, 0.35],
        [0.12, 0.48, 0.40],
    ]),
    "temp_pattern": np.array([
        [0.55, 0.35, 0.10],
        [0.35, 0.45, 0.20],
        [0.75, 0.20, 0.05],
        [0.90, 0.08, 0.02],
    ]),
    "hr_trend": np.array([
        [0.30, 0.50, 0.20],
        [0.35, 0.45, 0.20],
        [0.70, 0.25, 0.05],
        [0.88, 0.10, 0.02],
    ]),
    "pain_trajectory": np.array([
        [0.20, 0.50, 0.30],
        [0.35, 0.45, 0.20],
        [0.70, 0.25, 0.05],
        [0.85, 0.13, 0.02],
    ]),
}

# Shown alongside every estimate so a reviewing clinician can trace a flag back
# to a published standard (doc section 12).
SOURCES: Dict[str, str] = {
    "crp_velocity": "CRP velocity is key phase indicator — doc section 3",
    "wbc_pattern": "Neutrophilia pattern indicates phase — doc section 3",
    "prealbumin_trend": "Best available nutrition response marker — ASPEN 2016",
    "albumin_trend": "Protein synthesis capacity, slow marker — ASPEN 2016",
    "temp_pattern": "Sustained fever >38.5 = complication — doc section 5.3",
    "hr_trend": "HR >100 sustained = complication flag — doc section 5.3",
    "pain_trajectory": "Escalating pain = complication flag — doc section 3",
}

# --- thresholds: raw reading -> symbol --------------------------------------
CRP_TREND_PCT = 0.10          # +/- vs last drawn CRP
PROTEIN_TREND_PCT = 0.05      # +/- vs last drawn albumin / prealbumin
PAIN_DELTA = 1.0              # pain score points vs previous day
WBC_NORMAL = (4.0, 11.0)      # x10^9/L
WBC_NEUTROPHILIA_MAX = 15.0
TEMP_AFEBRILE_MAX = 37.5      # degrees C
TEMP_FEBRILE_MIN = 38.5
HR_NORMAL_MAX = 90.0          # bpm
HR_TACHYCARDIA_MIN = 100.0

GLUCOSE_CONTROLLED_MAX = 140.0   # mg/dL
GLUCOSE_UNCONTROLLED_MIN = 180.0
SPO2_OK_MIN = 95.0               # %
SPO2_LOW_MAX = 93.0
ZINC_DEFICIENT_MAX = 70.0        # ug/dL

LOW_CONFIDENCE = 0.60

NO_IMAGING = (
    "No wound imaging available. Phase estimate is inferred from indirect signal "
    "trajectory. Clinical wound assessment recommended to corroborate."
)


# =========================================================================
# 2. ENCODING — raw daily readings into the symbols table B understands
# =========================================================================

# Doc section 3: wearables deliver HR, temperature and SpO2 as continuous
# streams. Layer 1 needs one figure per day per signal, so it reduces the stream
# the way the signal's clinical meaning requires: the day's *peak* temperature,
# because a fever spike matters and averaging hides it, and the *mean* heart rate
# and SpO2, because those are about sustained level rather than excursion.
STREAM_REDUCERS: Dict[str, Tuple[str, str]] = {
    "temp_max": ("temp", "max"),
    "hr_mean": ("hr", "mean"),
    "spo2_mean": ("spo2", "mean"),
    # Doc section 3: pain score is self-reported 4-6 hourly, not once a day. The
    # day's mean is the right reduction for a *trajectory* symbol — a single
    # reading would make the trend an artefact of which hour it was taken.
    "pain_score_mean": ("pain", "mean"),
}


def _from_stream(block: Dict[str, Any], key: str) -> Optional[float]:
    """Reduce a wearable stream to the single daily figure `key` names."""
    mapping = STREAM_REDUCERS.get(key)
    if not mapping:
        return None
    signal, how = mapping
    raw = block.get(signal)
    if not isinstance(raw, list) or not raw:
        return None
    values = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get("v") if "v" in item else item.get("value")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return max(values) if how == "max" else sum(values) / len(values)


def _get(day: Dict[str, Any], group: str, key: str) -> Optional[float]:
    """One reading from a day record, or None if it was not measured.

    Accepts either a daily figure or a wearable stream, so the same timeline
    feeds Layer 1 and Layer 2 unchanged.
    """
    block = day.get(group)
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if value is None or value == "":
        return _from_stream(block, key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _last_drawn(days: Sequence[Dict[str, Any]], index: int, group: str, key: str) -> Optional[float]:
    """The most recent earlier reading of this signal.

    Labs are sparse — prealbumin is drawn every 2-3 days (doc section 3). A
    trend must compare against the last day it was actually drawn, not against
    yesterday, or the trend disappears whenever a day is skipped.
    """
    for j in range(index - 1, -1, -1):
        value = _get(days[j], group, key)
        if value is not None:
            return value
    return None


def _trend(current: Optional[float], previous: Optional[float], threshold: float,
           up: str, flat: str, down: str) -> Optional[str]:
    if current is None or previous is None or previous == 0:
        return None
    change = (current - previous) / abs(previous)
    return up if change > threshold else (down if change < -threshold else flat)


def encode_day(days: Sequence[Dict[str, Any]], index: int) -> Dict[str, Optional[str]]:
    """Encode one post-op day into symbols.

    A signal that was not measured yields None and simply drops out of the
    calculation — no imputation. The model learns less from that day rather
    than learning something invented.
    """
    day = days[index]
    out: Dict[str, Optional[str]] = {}

    out["crp_velocity"] = _trend(
        _get(day, "labs", "crp"), _last_drawn(days, index, "labs", "crp"),
        CRP_TREND_PCT, "rising", "plateau", "declining")

    wbc = _get(day, "labs", "wbc")
    if wbc is None:
        out["wbc_pattern"] = None
    elif wbc < WBC_NORMAL[0]:
        # Leukopenia is abnormal, not normal — a low WBC post-op is a sepsis
        # signal. This branch must come first or 2.1 reads as neutrophilia.
        out["wbc_pattern"] = "elevated"
    elif wbc <= WBC_NORMAL[1]:
        out["wbc_pattern"] = "normal"
    elif wbc <= WBC_NEUTROPHILIA_MAX:
        out["wbc_pattern"] = "neutrophilia"
    else:
        out["wbc_pattern"] = "elevated"

    out["prealbumin_trend"] = _trend(
        _get(day, "labs", "prealbumin"), _last_drawn(days, index, "labs", "prealbumin"),
        PROTEIN_TREND_PCT, "rising", "flat", "falling")
    out["albumin_trend"] = _trend(
        _get(day, "labs", "albumin"), _last_drawn(days, index, "labs", "albumin"),
        PROTEIN_TREND_PCT, "rising", "flat", "falling")

    temp = _get(day, "vitals", "temp_max")
    out["temp_pattern"] = None if temp is None else (
        "afebrile" if temp < TEMP_AFEBRILE_MAX
        else "low_grade" if temp < TEMP_FEBRILE_MIN
        else "febrile")

    hr = _get(day, "vitals", "hr_mean")
    out["hr_trend"] = None if hr is None else (
        "normal" if hr < HR_NORMAL_MAX
        else "elevated" if hr < HR_TACHYCARDIA_MIN
        else "sustained_tachycardia")

    pain = _get(day, "self_report", "pain_score_mean")
    before = _last_drawn(days, index, "self_report", "pain_score_mean")
    if pain is None or before is None:
        out["pain_trajectory"] = None
    elif pain <= before - PAIN_DELTA:
        out["pain_trajectory"] = "improving"
    elif pain >= before + PAIN_DELTA:
        out["pain_trajectory"] = "escalating"
    else:
        out["pain_trajectory"] = "static"

    return out


def encode_passthrough(day: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Signals Layer 1 carries but never uses. See the passthrough note above."""
    glucose = _get(day, "labs", "glucose")
    spo2 = _get(day, "vitals", "spo2_mean")
    zinc = _get(day, "labs", "zinc")
    report = day.get("self_report") if isinstance(day.get("self_report"), dict) else {}

    return {
        "glucose_level": None if glucose is None else (
            "controlled" if glucose < GLUCOSE_CONTROLLED_MAX
            else "elevated" if glucose <= GLUCOSE_UNCONTROLLED_MIN
            else "uncontrolled"),
        "spo2_level": None if spo2 is None else (
            "ok" if spo2 >= SPO2_OK_MIN
            else "borderline" if spo2 >= SPO2_LOW_MAX
            else "low"),
        "serum_zinc": None if zinc is None else (
            "deficient" if zinc < ZINC_DEFICIENT_MAX else "replete"),
        "appetite": report.get("appetite"),
        "nausea": report.get("nausea"),
    }

_LOG0 = -1e300

# =========================================================================
# 3. THE HMM — doc section 4.1
#
# Everything is in log space, because multiplying thirty days of probabilities
# together underflows to zero in floating point. Logs turn those products into
# sums.
#
# Two passes over the days:
#
#   forward  -> P(phase | everything up to today).  The live daily estimate.
#   viterbi  -> the single most likely whole path.  Gives transition days.
#
# The doc says "P(phase | all observations to date) via Viterbi". Those are two
# different algorithms: Viterbi returns a best path and no per-day probability,
# so it cannot produce a confidence score. The forward recursion is what that
# line actually needs, and Viterbi is kept for the transition timestamps.
# =========================================================================

def _log(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, _LOG0)
    np.log(x, out=out, where=x > 0)
    return out

def _logsumexp(v: np.ndarray) -> float:
    peak = float(np.max(v))
    return peak if peak <= _LOG0 else peak + float(np.log(np.sum(np.exp(v - peak))))

def _normalise(v: np.ndarray) -> np.ndarray:
    total = _logsumexp(v)
    return np.full(len(PHASES), 1 / len(PHASES)) if total <= _LOG0 else np.exp(v - total)

def day_likelihood(symbols: Dict[str, Optional[str]]) -> np.ndarray:
    """How well does each of the four phases explain this day?
    Signals are treated as independent given the phase, so their probabilities
    multiply — which in log space means they add. That independence assumption
    is what keeps B a readable per-signal table.

    A day with nothing recorded returns all zeros, so belief simply carries
    forward on A alone. That is correct, not a failure.
    """
    total = np.zeros(len(PHASES))
    for channel, symbol in symbols.items():
        if symbol is None or channel not in EMISSIONS:
            continue
        column = SYMBOLS[channel].index(symbol)
        total += _log(EMISSIONS[channel][:, column])
    return total


def _pin(vector: np.ndarray, phase: Optional[int]) -> np.ndarray:
    """Force a day to a known phase.

    Day 0 uses this to pin HAEMOSTASIS — a 0-3 hour phase cannot be identified
    from daily labs, so we assert it rather than pretend to infer it. Clinician
    overrides use the same mechanism, so an override propagates forward exactly
    the way real evidence does.
    """
    if phase is None:
        return vector
    forced = np.full(len(PHASES), _LOG0)
    forced[phase] = 0.0
    return forced


def initial_distribution(first_day: int) -> np.ndarray:
    """Log prior over the four phases on a timeline's first recorded day.

    A timeline does not always start at day 0 — a patient may be enrolled
    mid-stay. Pinning the first *record* to haemostasis would then tell the model
    a day-22 patient is three hours out of theatre, and it would spend the rest of
    the stay catching up.

    Instead the prior is the transition matrix run forward from haemostasis by the
    number of elapsed days, with no evidence. That is exactly what the model would
    believe about an unobserved patient on that day, which is the right starting
    point when the earlier days were genuinely not recorded.
    """
    belief = np.zeros(len(PHASES))
    belief[H] = 1.0
    for _ in range(max(0, int(first_day))):
        belief = belief @ TRANSITION_MATRIX
    return _log(belief)


def forward(likelihoods: Sequence[np.ndarray], pinned: Dict[int, int],
            first_day: int = 0) -> np.ndarray:
    """Daily posterior: P(phase on day t | all observations up to day t).

    For each day: where could I have come from (previous belief x A), then what
    did I see today (x B). Only ever looks backwards, which is what makes it an
    honest live estimate — it is what a clinician could have known that day.
    """
    log_a = _log(TRANSITION_MATRIX)
    n, steps = len(PHASES), len(likelihoods)

    alpha = np.full((steps, n), _LOG0)
    start = initial_distribution(first_day)
    alpha[0] = _pin(start + likelihoods[0], pinned.get(0))

    for t in range(1, steps):
        for j in range(n):
            alpha[t][j] = _logsumexp(alpha[t - 1] + log_a[:, j])
        alpha[t] = _pin(alpha[t] + likelihoods[t], pinned.get(t))

    return np.vstack([_normalise(alpha[t]) for t in range(steps)])


def viterbi(likelihoods: Sequence[np.ndarray], pinned: Dict[int, int],
            first_day: int = 0) -> List[int]:
    """The single most likely sequence of phases across the whole stay.

    Same shape as forward, but takes the max instead of summing, and remembers
    which phase the max came from so the path can be walked back. Used for
    transition days: taking each day's argmax independently could produce a
    path that jumps around impossibly.
    """
    log_a = _log(TRANSITION_MATRIX)
    n, steps = len(PHASES), len(likelihoods)

    score = np.full((steps, n), _LOG0)
    came_from = np.zeros((steps, n), dtype=int)
    start = initial_distribution(first_day)
    score[0] = _pin(start + likelihoods[0], pinned.get(0))

    for t in range(1, steps):
        for j in range(n):
            options = score[t - 1] + log_a[:, j]
            came_from[t][j] = int(np.argmax(options))
            score[t][j] = float(np.max(options))
        score[t] = _pin(score[t] + likelihoods[t], pinned.get(t))

    path = [0] * steps
    path[-1] = int(np.argmax(score[-1]))
    for t in range(steps - 2, -1, -1):
        path[t] = int(came_from[t + 1][path[t + 1]])
    return path



# =========================================================================
# 4. THE ESTIMATOR — timeline in, one phase out
# =========================================================================

def _why(symbols: Dict[str, Optional[str]], phase: int) -> List[Dict[str, Any]]:
    """The evidence behind a day's answer.

    For each observed signal, reports the literal cell from table B — its
    probability in the estimated phase — so a clinician can look the number up
    in the document.

    Ranked by how much more likely the observation is in this phase than in the
    next most likely one. A high probability alone does not mean a signal drove
    the answer: "afebrile" is 0.75 in proliferation but 0.90 in remodelling, so
    it separates nothing. "CRP declining" is 0.72 against 0.08 in inflammation,
    and that ratio is what actually decided the phase.
    """
    rows = []
    for channel, symbol in symbols.items():
        if symbol is None or channel not in EMISSIONS:
            continue
        column = SYMBOLS[channel].index(symbol)
        probabilities = EMISSIONS[channel][:, column]
        others = [probabilities[k] for k in range(len(PHASES)) if k != phase]
        best_rival = max(max(others), 1e-6)
        rows.append({
            "signal": channel,
            "observed": symbol,
            "probability_in_phase": round(float(probabilities[phase]), 3),
            "times_more_likely_than_next_phase": round(float(probabilities[phase] / best_rival), 2),
            "source": SOURCES[channel],
        })
    rows.sort(key=lambda r: r["times_more_likely_than_next_phase"], reverse=True)
    return rows


def validate_days(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate and normalise the day records before any inference runs.

    Every layer starts here, because a malformed timeline that reaches the maths
    fails somewhere deep with a message that names a numpy operation rather than
    the field the caller got wrong. Rejecting it at the boundary with a specific
    reason is the difference between a fixable 422 and a support ticket.

    Rejects, rather than quietly repairing: duplicate day numbers (which day's
    labs would win?), negative days (nothing precedes surgery), and non-integer
    day numbers. Out-of-order records ARE repaired, by sorting — the order a
    caller serialises records in carries no clinical meaning.
    """
    days = timeline.get("days")
    if not isinstance(days, list) or not days:
        raise ValueError("timeline.days must be a non-empty list of day records")

    normalised: List[Dict[str, Any]] = []
    seen: Dict[int, int] = {}
    for position, record in enumerate(days):
        if not isinstance(record, dict):
            raise ValueError(f"timeline.days[{position}] must be an object, got "
                             f"{type(record).__name__}")
        raw = record.get("day")
        if raw is None:
            raise ValueError(f"timeline.days[{position}] is missing its 'day' number")
        try:
            number = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"timeline.days[{position}].day must be a whole number, "
                             f"got {raw!r}")
        if number < 0:
            raise ValueError(f"timeline.days[{position}].day is {number}; post-op days "
                             "start at 0")
        if number in seen:
            raise ValueError(f"duplicate record for day {number} (positions "
                             f"{seen[number]} and {position})")
        seen[number] = position
        normalised.append({**record, "day": number})

    return sorted(normalised, key=lambda d: d["day"])


def estimate_phase(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate which healing phase a patient is in.

    Returns one phase — the most likely of the four — with the confidence
    behind it, the daily trajectory, the transitions, and the evidence.
    """
    days = validate_days(timeline)
    numbers = [int(d["day"]) for d in days]

    symbols = [encode_day(days, i) for i in range(len(days))]
    extra = [encode_passthrough(d) for d in days]
    likelihoods = [day_likelihood(s) for s in symbols]

    # Haemostasis is pinned only when day 0 is actually present. A timeline that
    # begins mid-stay gets its prior from initial_distribution() instead.
    pinned: Dict[int, int] = {0: H} if numbers[0] == 0 else {}
    applied = []
    for item in timeline.get("clinician_phase_overrides") or []:
        phase_name = str(item.get("phase", "")).upper()
        if phase_name not in PHASE_INDEX:
            raise ValueError(f"Unknown phase in override: {item.get('phase')!r}")
        day_number = int(item.get("day"))
        if day_number in numbers:
            pinned[numbers.index(day_number)] = PHASE_INDEX[phase_name]
            applied.append({"day": day_number, "phase": phase_name, "source": "clinician_override"})

    posterior = forward(likelihoods, pinned, first_day=numbers[0])
    path = viterbi(likelihoods, pinned, first_day=numbers[0])

    daily = []
    for i, number in enumerate(numbers):
        best = int(np.argmax(posterior[i]))
        daily.append({
            "day": number,
            "phase": PHASES[best],
            "confidence": round(float(posterior[i][best]), 4),
            "posterior": {PHASES[k]: round(float(posterior[i][k]), 4) for k in range(len(PHASES))},
            "signals": {k: v for k, v in symbols[i].items() if v is not None},
            "why": _why(symbols[i], best),
            "passthrough": {k: v for k, v in extra[i].items() if v is not None},
        })

    transitions = [
        {"day": numbers[i], "from": PHASES[path[i - 1]], "to": PHASES[path[i]],
         "regression": path[i] < path[i - 1]}
        for i in range(1, len(path)) if path[i] != path[i - 1]
    ]

    dwell: Dict[str, int] = {}
    for phase in path:
        dwell[PHASES[phase]] = dwell.get(PHASES[phase], 0) + 1

    final = posterior[-1]
    current = int(np.argmax(final))
    confidence = float(final[current])

    observed = sum(len(d["signals"]) for d in daily)
    possible = len(days) * len(EMISSIONS)

    limitations = [NO_IMAGING]
    if confidence < LOW_CONFIDENCE:
        limitations.append(
            f"Phase confidence {confidence:.0%} is below the {LOW_CONFIDENCE:.0%} "
            "threshold. Treat this estimate as provisional.")
    if any(t["regression"] for t in transitions):
        limitations.append(
            "A backward phase transition was detected, consistent with renewed "
            "inflammation. Corroborate against clinical wound assessment.")

    # The clinical profile (surgery type, ASA grade, weight, ...) lives in
    # timeline.patient — Layers 2-4 need the rest of it too (doc section 5.1's
    # cohort conditioning variables). A flat top-level surgery_type is still
    # accepted for backward compatibility with older callers and the synthetic
    # generator below, which has no patient block at all.
    patient = timeline.get("patient") if isinstance(timeline.get("patient"), dict) else {}
    surgery_type = patient.get("surgery_type") or timeline.get("surgery_type")

    return {
        "patient_ref": timeline.get("patient_ref"),
        "surgery_type": surgery_type,
        "days_post_op": numbers[-1],
        "current_phase": PHASES[current],
        "phase_confidence": round(confidence, 4),
        "low_confidence": confidence < LOW_CONFIDENCE,
        "phase_posterior": {PHASES[k]: round(float(final[k]), 4) for k in range(len(PHASES))},
        "daily": daily,
        "viterbi_path": [PHASES[p] for p in path],
        "phase_transitions": transitions,
        "dwell_days": dwell,
        "signals_observed": {"observed": observed, "possible": possible,
                             "pct": round(observed / possible, 4) if possible else 0.0},
        "overrides_applied": applied,
        "limitations": limitations,
        "model_version": MODEL_VERSION,
    }

# =========================================================================
# 5. SYNTHETIC PATIENTS — how this gets tested with no real data
#
# Each scenario declares a true phase for every day. Symbols are then sampled
# from table B for that true phase, and rendered as raw values that encode back
# to the same symbol. The estimator sees only the raw values.
#
# Note what this does and does not prove: patients are drawn from the same
# tables the estimator reads, so accuracy here measures whether the inference
# is correct — not whether the tables match real biology.
# =========================================================================

SCENARIOS = ("normal", "delayed", "ssi_regression")


def true_phases(scenario: str, days: int) -> List[int]:
    """The ground truth a synthetic patient is generated from."""
    def phase_on(day: int) -> int:
        if day == 0:
            return H
        if scenario == "delayed":
            # Still inflammatory well past the expected day-5 exit.
            return I if day <= 11 else (P if day <= 26 else R)
        if scenario == "ssi_regression":
            # Normal course, then infection at day 8 drives it backwards.
            if day <= 5:
                return I
            if day <= 7:
                return P
            if day <= 14:
                return I
            return P if day <= 27 else R
        return I if day <= 5 else (P if day <= 20 else R)
    return [phase_on(d) for d in range(days)]


def generate_timeline(scenario: str = "normal", days: int = 30, seed: int = 7,
                      lab_dropout: float = 0.0) -> Dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    rng = np.random.default_rng(seed)
    truth = true_phases(scenario, days)
    crp, prealbumin, albumin = 8.0, 180.0, 38.0
    records: List[Dict[str, Any]] = []

    for day in range(days):
        phase = truth[day]

        def sample(channel: str) -> str:
            return str(rng.choice(SYMBOLS[channel], p=EMISSIONS[channel][phase]))

        crp = float(np.clip(crp * {"rising": 1.45, "plateau": 1.0, "declining": 0.72}[sample("crp_velocity")], 1, 400))
        prealbumin = float(np.clip(prealbumin * {"rising": 1.12, "flat": 1.0, "falling": 0.90}[sample("prealbumin_trend")], 40, 400))
        albumin = float(np.clip(albumin * {"rising": 1.09, "flat": 1.0, "falling": 0.92}[sample("albumin_trend")], 15, 55))
        wbc = {"normal": 8.0, "neutrophilia": 13.0, "elevated": 18.5}[sample("wbc_pattern")]
        temp = {"afebrile": 36.9, "low_grade": 38.0, "febrile": 38.9}[sample("temp_pattern")]
        hr = {"normal": 78.0, "elevated": 95.0, "sustained_tachycardia": 108.0}[sample("hr_trend")]

        previous_pain = records[-1]["self_report"]["pain_score_mean"] if records else 7.0
        shift = {"improving": -1.5, "static": 0.0, "escalating": 1.5}[sample("pain_trajectory")] if records else 0.0
        pain = float(np.clip(previous_pain + shift, 0, 10))

        labs: Dict[str, Any] = {
            "crp": round(crp, 1), "wbc": wbc,
            "prealbumin": round(prealbumin, 1), "albumin": round(albumin, 1),
            "glucose": round(float(np.clip(rng.normal(135, 25), 70, 260)), 1),
        }
        if day % 7 == 0:
            labs["zinc"] = round(float(np.clip(rng.normal(72, 12), 40, 120)), 1)
        # Labs are sparse in reality; vitals and self-report are not.
        if lab_dropout and day > 0 and rng.random() < lab_dropout:
            labs = {"glucose": labs["glucose"]}

        records.append({
            "day": day,
            "labs": labs,
            "vitals": {"hr_mean": hr, "temp_max": temp,
                       "spo2_mean": round(float(np.clip(rng.normal(96.5, 1.6), 88, 100)), 1)},
            "self_report": {"pain_score_mean": round(pain, 1),
                            "appetite": "poor" if pain > 6 else "fair",
                            "nausea": bool(pain > 7)},
        })

    return {
        "patient_ref": f"SYN-{scenario.upper()}-{seed}",
        "surgery_type": "Open abdominal — bowel resection",
        "scenario": scenario,
        "ground_truth_phases": [PHASES[p] for p in truth],
        "days": records,
    }


# =========================================================================
# 6. TERMINAL OUTPUT
# =========================================================================

def render(result: Dict[str, Any], width: int = 20) -> str:
    lines = ["=" * 74,
             " POST-OP PHASE 1 - LAYER 1 : HEALING PHASE ESTIMATOR",
             "=" * 74,
             f" Patient      {result.get('patient_ref') or '-'}",
             f" Surgery      {result.get('surgery_type') or '-'}",
             f" Day post-op  {result['days_post_op']}",
             ""]

    flag = "   ** LOW CONFIDENCE **" if result["low_confidence"] else ""
    lines += [f" CURRENT PHASE   {result['current_phase']}"
              f"   (confidence {result['phase_confidence']:.0%}){flag}", ""]

    for name, probability in result["phase_posterior"].items():
        bar = "#" * int(round(probability * width))
        lines.append(f"   {name:<15}{bar:<{width}} {probability:>6.1%}")
    lines += ["", " DAILY TRAJECTORY", f" {'Day':>3}  {'Phase':<14} {'Conf':>5}  Signals"]

    for row in result["daily"]:
        if row["day"] == 0:
            detail = "pinned - haemostasis is 0-3h, not inferred from daily labs"
        elif row["why"]:
            detail = ", ".join(f"{w['signal']}={w['observed']}" for w in row["why"][:3])
        else:
            detail = "nothing recorded - belief carried forward"
        lines.append(f" {row['day']:>3}  {row['phase']:<14} {row['confidence']:>5.0%}  {detail}")

    lines += ["", " PHASE TRANSITIONS  (most likely path)"]
    if result["phase_transitions"]:
        for t in result["phase_transitions"]:
            mark = "   <-- REGRESSION" if t["regression"] else ""
            lines.append(f"   day {t['day']:>3}   {t['from']} -> {t['to']}{mark}")
    else:
        lines.append("   none detected")

    lines += ["", " DWELL",
              "   " + "   ".join(f"{k} {v}d" for k, v in result["dwell_days"].items())]

    if result["overrides_applied"]:
        lines += ["", " CLINICIAN OVERRIDES"]
        lines += [f"   day {o['day']:>3}   forced to {o['phase']}" for o in result["overrides_applied"]]

    lines += ["", " LIMITATIONS"]
    lines += [f"   * {text}" for text in result["limitations"]]
    signals = result["signals_observed"]
    lines += ["", f" {signals['observed']}/{signals['possible']} signal readings "
                  f"({signals['pct']:.0%})   {result['model_version']}"]
    return "\n".join(lines)

# =========================================================================
# 7. SERVICE — stateless. Timeline in the request, phase in the response.
#    No LangGraph; the request shape leaves room for one later.
# =========================================================================

try:
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    class EstimatePhaseRequest(BaseModel):
        model_config = ConfigDict(extra="allow")
        patient_ref: Optional[str] = None
        patient: Dict[str, Any] = Field(default_factory=dict)
        surgery_type: Optional[str] = None  # accepted flat too — see estimate_phase()
        days: List[Dict[str, Any]] = Field(default_factory=list)
        clinician_phase_overrides: List[Dict[str, Any]] = Field(default_factory=list)

    app = FastAPI(title="Post-op Phase 1 - Layer 1", version=MODEL_VERSION)

    def _token() -> Optional[str]:
        for name in ("POSTOP_PHASE1_SERVICE_API_KEY", "FOODHAK_API_TOKEN", "LANGGRAPH_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "postop-phase1-layer1",
                "model_version": MODEL_VERSION, "phases": list(PHASES),
                "signals": list(EMISSIONS), "auth_required": _token() is not None}

    @app.post("/postop/phase1/estimate-phase")
    async def estimate(
        request: EstimatePhaseRequest,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        # Auth engages only when a token is configured: open for local terminal
        # work, locked down in staging.
        expected = _token()
        if expected:
            received = x_api_key
            if authorization and authorization.lower().startswith("bearer "):
                received = authorization[7:].strip()
            if received != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing service token")
        try:
            return estimate_phase(request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

except ImportError as exc:  # the CLI must still work without FastAPI installed
    _MISSING_DEPENDENCY = (
        f"The HTTP service is disabled: {exc}. "
        "Install the service dependencies with:  pip install -r requirements.txt"
    )
    print(f"layer1_phase_estimator: {_MISSING_DEPENDENCY}\n"
          "The command-line tool still works.", file=sys.stderr)

    async def app(scope, receive, send):  # type: ignore[misc]
        """Stand-in so an ASGI server reports the real problem.

        Without this, `app` was None and uvicorn served it, producing
        "TypeError: 'NoneType' object is not callable" on every request —
        which says nothing about the actual cause.
        """
        if scope.get("type") != "http":
            return
        body = json.dumps({"detail": _MISSING_DEPENDENCY}).encode()
        await send({"type": "http.response.start", "status": 503,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


# =========================================================================
# 8. CLI
# =========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate the wound-healing phase from a daily signal timeline.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--timeline", type=Path, help="Path to a timeline JSON file")
    source.add_argument("--scenario", choices=SCENARIOS, help="Generate a synthetic patient")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lab-dropout", type=float, default=0.0,
                        help="Fraction of days with labs withheld, 0.0-1.0")
    parser.add_argument("--json", action="store_true", help="Raw JSON instead of the table")
    args = parser.parse_args(argv)

    timeline = (json.loads(args.timeline.read_text()) if args.timeline
                else generate_timeline(args.scenario or "normal", args.days, args.seed, args.lab_dropout))

    try:
        result = estimate_phase(timeline)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(render(result))

    truth = timeline.get("ground_truth_phases")
    if truth:
        predicted = [row["phase"] for row in result["daily"]]
        hits = sum(1 for a, b in zip(truth, predicted) if a == b)
        print(f"\n GROUND TRUTH (synthetic)  {hits}/{len(truth)} days recovered ({hits / len(truth):.0%})")
        print(f"   true       {' '.join(p[:4] for p in truth)}")
        print(f"   estimated  {' '.join(p[:4] for p in predicted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
