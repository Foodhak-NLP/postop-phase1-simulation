"""
Post-op Phase 1 — Layer 3: Nutritional Gap Engine.

Implements section 6 of Phase_1_Wound_Recovery.docx.

Layer 1 says which phase. Layer 2 says whether healing is on track. Layer 3 asks
the question those two set up: is this patient actually being fed what their
current phase requires?

    phase (Layer 1) + intake log + deviations (Layer 2)
        -> per-nutrient gaps, a sufficiency score, and interaction flags

Deliberately NOT a machine learning model. Section 6 is explicit about why:
clinical evidence for wound-healing nutrition is strong and well established in
the ASPEN and ESPEN guidelines, so introducing ML would add uncertainty where
none is needed and create a regulatory burden with no clinical benefit. Every
number below traces to a published guideline, and every flag cites it.

Run it:

    python3 layer3_nutrition_gap.py --timeline example_patient.json
    uvicorn layer3_nutrition_gap:app --port 8012 --reload
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layer1_phase_estimator import PHASES, estimate_phase, validate_days
from layer2_deviation_detector import detect_deviations, require_upstream

MODEL_VERSION = "phase1-nutrition-v1"


# ===========================================================================
# 1. CONFIG — section 6.1's targets, verbatim
#
# Each entry carries the lower and upper bound of the published range per phase,
# and the guideline it comes from. The *lower* bound is the target used for gap
# scoring: section 7.1's worked example shows protein target 1.8 g/kg during
# proliferation, which is the bottom of that phase's 1.8-2.0 range, and its
# 39% gap against an actual 1.1 g/kg confirms (1.8 - 1.1) / 1.8.
#
# A phase with no numeric target for a nutrient (arginine in remodelling,
# "not required") is not scored — absent guidance is not a gap of zero.
# ===========================================================================

NUTRIENT_TARGETS: Dict[str, Dict[str, Any]] = {
    "protein_g": {
        "label": "Protein", "unit": "g/kg/day", "per_kg": True,
        "INFLAMMATION": (1.5, 2.0), "PROLIFERATION": (1.8, 2.0), "REMODELLING": (1.2, 1.5),
        "source": "ASPEN 2016, ESPEN 2017",
    },
    "calories_kcal": {
        "label": "Total calories", "unit": "kcal/kg/day", "per_kg": True,
        "INFLAMMATION": (25.0, 30.0), "PROLIFERATION": (30.0, 35.0), "REMODELLING": (25.0, 30.0),
        "source": "ASPEN 2016",
    },
    "vitamin_c_mg": {
        "label": "Vitamin C", "unit": "mg/day", "per_kg": False,
        "INFLAMMATION": (200.0, 500.0), "PROLIFERATION": (500.0, 1000.0), "REMODELLING": (100.0, 200.0),
        "source": "Posthauer 2015",
    },
    "zinc_mg": {
        "label": "Zinc", "unit": "mg/day", "per_kg": False,
        # "25-50 mg/day if deficient" — so the target only applies when serum
        # zinc says the patient is deficient. Supplementing a replete patient is
        # not the goal, and excess zinc is itself harmful.
        "conditional_on": "zinc_deficient",
        "INFLAMMATION": (25.0, 50.0), "PROLIFERATION": (25.0, 50.0), "REMODELLING": None,
        "source": "EPUAP 2019",
    },
    "arginine_g": {
        "label": "Arginine", "unit": "g/day", "per_kg": False,
        "INFLAMMATION": None, "PROLIFERATION": (4.5, 14.0), "REMODELLING": None,
        "source": "ESPEN 2017",
    },
    "vitamin_a_iu": {
        "label": "Vitamin A", "unit": "IU/day", "per_kg": False,
        # Inflammation target applies only "if at risk"; proliferation is
        # unconditional; remodelling is "not routinely required".
        "conditional_on_phase": {"INFLAMMATION": "vitamin_a_at_risk"},
        "INFLAMMATION": (10000.0, 10000.0), "PROLIFERATION": (10000.0, 10000.0), "REMODELLING": None,
        "source": "NPUAP 2019",
    },
    "fluid_ml": {
        "label": "Hydration", "unit": "ml/kg/day", "per_kg": True,
        "INFLAMMATION": (30.0, 35.0), "PROLIFERATION": (30.0, 35.0), "REMODELLING": (30.0, 30.0),
        "source": "ASPEN 2016",
    },
}

# --- section 6.2 gap thresholds ---------------------------------------------
GAP_HIGH = 0.30           # "if gap(N) > 0.30: FLAG as HIGH priority"
GAP_MODERATE = 0.15       # "if gap(N) > 0.15: FLAG as MODERATE priority"

# "if gap(N) < 0: FLAG as EXCESS (rare but relevant for zinc/Vit A)". Only these
# two are flagged for excess, because only these two are toxic in surplus at the
# doses used in wound care.
EXCESS_RELEVANT = ("zinc_mg", "vitamin_a_iu")
EXCESS_TOLERANCE = 0.25   # how far above the *upper* bound counts as excess

# --- section 6.2 NSS escalation ---------------------------------------------
NSS_ESCALATE = 0.70       # "NSS < 0.70: Escalate to dietitian immediately"
NSS_REVIEW = 0.85         # "NSS < 0.85: Flag for dietitian review at next round"

# NSS weights. Section 6.2 says only "Weights reflect relative criticality per
# phase. Protein weight highest in proliferation. Vitamin K weight highest in
# haemostasis." The exact numbers are not published, so these are ours, chosen to
# honour those two statements, and each phase's weights sum to 1.
#
# Vitamin K has no target row in section 6.1 and haemostasis lasts 0-3 hours, so
# there is no 24-hour intake to score on day 0 anyway. The gap engine therefore
# does not run during haemostasis, which is also clinically right: a patient
# hours out of theatre is not being fed.
NSS_WEIGHTS: Dict[str, Dict[str, float]] = {
    "INFLAMMATION": {"protein_g": 0.25, "calories_kcal": 0.20, "vitamin_c_mg": 0.20,
                     "zinc_mg": 0.15, "arginine_g": 0.05, "vitamin_a_iu": 0.05,
                     "fluid_ml": 0.10},
    "PROLIFERATION": {"protein_g": 0.30, "calories_kcal": 0.20, "vitamin_c_mg": 0.20,
                      "zinc_mg": 0.10, "arginine_g": 0.10, "vitamin_a_iu": 0.05,
                      "fluid_ml": 0.05},
    "REMODELLING": {"protein_g": 0.35, "calories_kcal": 0.25, "vitamin_c_mg": 0.20,
                    "zinc_mg": 0.05, "arginine_g": 0.05, "vitamin_a_iu": 0.05,
                    "fluid_ml": 0.05},
}

ZINC_DEFICIENT_BELOW = 70.0    # ug/dL, the same reference Layer 1 uses

# --- section 6.3 interaction flags ------------------------------------------
#
# Beyond individual nutrient gaps, the engine checks for clinically significant
# interactions between nutritional status and Layer 2's deviations. These are the
# findings neither layer can make alone: a protein gap is a dietetic issue, and
# non-resolving CRP is a clinical one, but the two together say the deficiency is
# probably *causing* the delay.
INTERACTIONS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "protein_gap_with_unresolving_inflammation",
        "deviation": "inflammation_not_resolving",
        "nutrient": "protein_g", "gap_above": 0.30,
        "interpretation": ("Insufficient substrate for immune function — protein "
                           "deficiency likely contributing to inflammation persistence"),
        "alert": ("Protein intake {gap:.0%} below phase target. CRP not resolving. "
                  "Protein supplementation review recommended today."),
        "severity": "HIGH",
    },
    {
        "id": "vitamin_c_gap_with_phase_delay",
        "deviation": "phase_transition_delay",
        "nutrient": "vitamin_c_mg", "gap_above": 0.0,
        "interpretation": ("Collagen synthesis substrate insufficient — Vitamin C "
                           "deficiency may be limiting fibroblast activity"),
        "alert": ("Vitamin C intake below proliferation threshold. Phase transition "
                  "to proliferation delayed beyond expected window."),
        "severity": "HIGH",
    },
    {
        "id": "carbohydrate_load_with_glucose_dysregulation",
        "deviation": "glucose_dysregulation",
        "nutrient": None, "carbohydrate_share_above": 0.50,
        "interpretation": ("Enteral formula carbohydrate composition may be "
                           "contributing to glycaemic instability"),
        "alert": ("Glucose above threshold on consecutive readings. Enteral "
                  "carbohydrate load review with pharmacy/dietitian recommended."),
        "severity": "HIGH",
    },
    {
        "id": "caloric_gap_with_flat_prealbumin",
        "deviation": "protein_synthesis_impairment",
        "nutrient": "calories_kcal", "gap_above": 0.20,
        "interpretation": ("Insufficient energy intake causing protein to be "
                           "catabolised for energy rather than used for tissue synthesis"),
        "alert": ("Caloric intake {gap:.0%} below target. Prealbumin not rising. "
                  "Protein being used for energy — caloric increase indicated."),
        "severity": "HIGH",
    },
)

# Carbohydrate share of total energy above which the load counts as "high".
# 4 kcal per gram of carbohydrate is the standard Atwater factor.
KCAL_PER_G_CARBOHYDRATE = 4.0

NO_IMAGING_LIMITATION = (
    "No wound imaging available. Nutritional targets are set from the inferred "
    "healing phase, not from direct wound assessment."
)


# ===========================================================================
# 2. TARGET RESOLUTION — phase and patient into a number
# ===========================================================================

def resolve_target(nutrient: str, phase: str, patient: Dict[str, Any],
                   zinc_deficient: Optional[bool]) -> Optional[Dict[str, Any]]:
    """The numeric target for one nutrient in one phase, or None if none applies.

    Exists because section 6.1's table is not a lookup — three of its seven rows
    are conditional. Zinc applies only if the patient is deficient, Vitamin A in
    inflammation only "if at risk", and arginine only during proliferation.
    Resolving that in one place keeps the conditionality auditable instead of
    scattering `if` statements through the scoring code.

    Returns None where the guideline gives no target, which the caller must treat
    as "not scored" rather than "target zero".
    """
    spec = NUTRIENT_TARGETS[nutrient]
    band = spec.get(phase)
    if band is None:
        return None

    # Conditional on a patient fact that holds in every phase (zinc deficiency).
    condition = spec.get("conditional_on")
    if condition == "zinc_deficient" and not zinc_deficient:
        return None

    # Conditional only in specific phases (Vitamin A "if at risk" in inflammation).
    phase_condition = (spec.get("conditional_on_phase") or {}).get(phase)
    if phase_condition and not patient.get(phase_condition):
        return None

    low, high = band
    weight = patient.get("weight_kg")
    if spec["per_kg"]:
        # Per-kilogram targets are meaningless without a body weight, and
        # inventing one would silently fabricate a clinical target.
        if not weight:
            return None
        return {"target": low * float(weight), "upper": high * float(weight),
                "per_kg_low": low, "per_kg_high": high, "unit": spec["unit"],
                "absolute_unit": spec["unit"].split("/")[0]}
    return {"target": low, "upper": high, "unit": spec["unit"],
            "absolute_unit": spec["unit"].split("/")[0]}


def zinc_status(days: Sequence[Dict[str, Any]], up_to_day: int) -> Optional[bool]:
    """Whether the patient is zinc deficient, from the most recent serum zinc.

    Separate from target resolution because serum zinc is drawn on admission then
    weekly (section 3), so the answer on any given day comes from a lab that may
    be days old. Carrying the last known value forward is correct here — zinc
    status changes slowly — but it has to be a deliberate choice, not an accident
    of lookup order.
    """
    latest = None
    for day in days:
        if int(day.get("day", 0)) > up_to_day:
            break
        labs = day.get("labs") if isinstance(day.get("labs"), dict) else {}
        if labs.get("zinc") not in (None, ""):
            latest = float(labs["zinc"])
    return None if latest is None else latest < ZINC_DEFICIENT_BELOW


# ===========================================================================
# 3. GAP SCORING — section 6.2, verbatim
#
#   gap(N) = (target(N) - actual(N)) / target(N)
#   gap > 0.30 -> HIGH,  gap > 0.15 -> MODERATE,  gap < 0 -> EXCESS
# ===========================================================================

def score_nutrient(nutrient: str, intake: Optional[float],
                   target_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Score one nutrient's 24-hour intake against its phase target.

    A single nutrient scored in isolation, because that is how a dietitian reads
    it — "protein is 39% short" is actionable, a composite number is not. The NSS
    is built from these afterwards rather than instead of them.

    Intake of None means the dietitian did not log it, which is different from
    logging zero. Zero intake is a 100% gap and a genuine finding; unlogged is
    unknown and must not be scored as starvation.
    """
    target, upper = target_spec["target"], target_spec["upper"]

    if intake is None:
        return {"nutrient": nutrient, "label": NUTRIENT_TARGETS[nutrient]["label"],
                "status": "not_logged", "target": round(target, 1),
                "unit": target_spec["absolute_unit"],
                "source": NUTRIENT_TARGETS[nutrient]["source"]}

    gap = (target - intake) / target if target else 0.0

    # Excess is only meaningful for the nutrients that are toxic in surplus.
    if nutrient in EXCESS_RELEVANT and intake > upper * (1 + EXCESS_TOLERANCE):
        priority = "EXCESS"
    elif gap > GAP_HIGH:
        priority = "HIGH"
    elif gap > GAP_MODERATE:
        priority = "MODERATE"
    else:
        priority = "ADEQUATE"

    return {
        "nutrient": nutrient,
        "label": NUTRIENT_TARGETS[nutrient]["label"],
        "status": "scored",
        "target": round(target, 1),
        "target_range": [round(target, 1), round(upper, 1)],
        "actual": round(intake, 1),
        "unit": target_spec["absolute_unit"],
        "gap_pct": round(gap * 100, 1),
        "priority": priority,
        # Clipped to [0, 1]: a nutrient cannot contribute more than "fully met" to
        # the score, so a large surplus of one cannot mask a deficit in another.
        "sufficiency": round(max(0.0, min(1.0, 1.0 - gap)), 4),
        "source": NUTRIENT_TARGETS[nutrient]["source"],
    }


def nutritional_sufficiency_score(scored: Sequence[Dict[str, Any]],
                                  phase: str) -> Optional[float]:
    """Section 6.2's NSS: the weighted mean of (1 - gap) across all nutrients.

    Weights are renormalised across only the nutrients that were actually scored.
    Without that, a patient whose arginine has no target in their phase would be
    penalised for a nutrient the guideline never asked for — the score would drift
    downward for reasons that have nothing to do with their care.

    Returns None when nothing could be scored, so the caller reports "no score"
    rather than a misleading zero.
    """
    weights = NSS_WEIGHTS.get(phase, {})
    usable = [row for row in scored if row["status"] == "scored"]
    total_weight = sum(weights.get(row["nutrient"], 0.0) for row in usable)
    if not usable or total_weight <= 0:
        return None
    weighted = sum(row["sufficiency"] * weights.get(row["nutrient"], 0.0) for row in usable)
    return round(weighted / total_weight, 4)


def nss_action(nss: Optional[float]) -> Dict[str, Any]:
    """Translate the NSS into section 6.2's escalation ladder.

    Kept separate from the score itself so the thresholds can be reviewed by a
    clinician without touching the arithmetic that produces the number.
    """
    if nss is None:
        return {"escalation": "none", "detail": "No nutrient could be scored."}
    if nss < NSS_ESCALATE:
        return {"escalation": "immediate",
                "detail": f"NSS {nss:.2f} is below {NSS_ESCALATE:.2f} — escalate to dietitian immediately.",
                "escalate_to": ["Dietitian"]}
    if nss < NSS_REVIEW:
        return {"escalation": "next_round",
                "detail": f"NSS {nss:.2f} is below {NSS_REVIEW:.2f} — flag for dietitian review at next round.",
                "escalate_to": ["Dietitian"]}
    return {"escalation": "none",
            "detail": f"NSS {nss:.2f} is at or above {NSS_REVIEW:.2f} — logged as adequate, no alert."}


# ===========================================================================
# 4. INTERACTION FLAGS — section 6.3
#
# The findings neither layer can make alone. A protein gap is a dietetic issue;
# non-resolving CRP is a clinical one. Together they say the deficiency is
# probably *causing* the delay, which is a different and more urgent statement.
# ===========================================================================

def carbohydrate_share(nutrition: Dict[str, Any], calories: Optional[float]) -> Optional[float]:
    """Carbohydrate as a fraction of total energy intake.

    Section 6.3's glucose interaction turns on "high carbohydrate load noted",
    which is a proportion rather than a gram count — 200 g of carbohydrate means
    something different in a 1200 kcal feed than in a 2600 kcal one.
    """
    carbs = nutrition.get("carbohydrate_g")
    if carbs in (None, "") or not calories:
        return None
    return (float(carbs) * KCAL_PER_G_CARBOHYDRATE) / float(calories)


def detect_interactions(scored: Sequence[Dict[str, Any]],
                        deviations: Sequence[Dict[str, Any]],
                        nutrition: Dict[str, Any],
                        calories: Optional[float]) -> List[Dict[str, Any]]:
    """Cross-reference Layer 2's deviations against this day's nutrient gaps.

    Runs after individual scoring rather than during it, because an interaction
    is a statement about the *pair*. Firing it inside the per-nutrient loop would
    lose the fact that both halves have to be true at once.
    """
    by_nutrient = {row["nutrient"]: row for row in scored if row["status"] == "scored"}
    present = {d["deviation"] for d in deviations}
    found: List[Dict[str, Any]] = []

    for rule in INTERACTIONS:
        if rule["deviation"] not in present:
            continue

        if rule.get("nutrient"):
            row = by_nutrient.get(rule["nutrient"])
            if not row or row["gap_pct"] / 100.0 <= rule["gap_above"]:
                continue
            gap = row["gap_pct"] / 100.0
            evidence = {"nutrient": row["label"], "gap_pct": row["gap_pct"],
                        "target": row["target"], "actual": row["actual"], "unit": row["unit"]}
        else:
            share = carbohydrate_share(nutrition, calories)
            if share is None or share <= rule["carbohydrate_share_above"]:
                continue
            gap = share
            evidence = {"carbohydrate_share_pct": round(share * 100, 1)}

        found.append({
            "id": rule["id"],
            "severity": rule["severity"],
            "paired_deviation": rule["deviation"],
            "clinical_interpretation": rule["interpretation"],
            "alert_text": rule["alert"].format(gap=gap),
            "evidence": evidence,
            "source": "doc section 6.3",
        })
    return found


# ===========================================================================
# 5. THE ENGINE — phase + intake + deviations in, gaps and NSS out
# ===========================================================================

def score_nutrition(timeline: Dict[str, Any], layer1: Dict[str, Any],
                    layer2: Dict[str, Any]) -> Dict[str, Any]:
    """Score every day that has an intake log, and surface the most recent.

    Scores the whole stay rather than only today because section 8's daily
    summary needs the trend — one bad day is a missed meal, four in a row is a
    plan that is not working. Takes Layer 1 and Layer 2 results rather than
    computing them, so the orchestrator owns the chaining and each layer stays
    independently testable.
    """
    days = validate_days(timeline)
    require_upstream(layer1, "layer1", "daily", "estimate_phase")
    require_upstream(layer2, "layer2", "deviations", "detect_deviations")
    patient = timeline.get("patient") if isinstance(timeline.get("patient"), dict) else {}
    phase_by_day = {row["day"]: row["phase"] for row in layer1["daily"]}

    def active_deviations(up_to_day: int) -> List[Dict[str, Any]]:
        """Deviations in force on a given day.

        A deviation is a standing clinical condition, not a one-day event. Layer 2
        collapses runs into episodes reported on the day they began, so CRP that
        stopped resolving on day 8 is still not resolving on day 10 — matching
        strictly by day would miss exactly the pairings section 6.3 exists to
        catch, since the nutrient gap and the deviation rarely surface together.
        """
        return [d for d in layer2["deviations"] if d["day"] <= up_to_day]

    daily: List[Dict[str, Any]] = []
    for day in days:
        number = int(day.get("day", 0))
        phase = phase_by_day.get(number)
        nutrition = day.get("nutrition") if isinstance(day.get("nutrition"), dict) else None

        # Haemostasis is day 0 and lasts hours; there is no prior 24 hours of
        # intake to score, and a patient just out of theatre is not being fed.
        if phase in (None, "HAEMOSTASIS") or not nutrition:
            daily.append({"day": number, "phase": phase, "status": "not_scored",
                          "reason": "no intake logged" if phase not in (None, "HAEMOSTASIS")
                                    else "haemostasis — no 24h intake window"})
            continue

        deficient = zinc_status(days, number)
        scored: List[Dict[str, Any]] = []
        for nutrient in NUTRIENT_TARGETS:
            spec = resolve_target(nutrient, phase, patient, deficient)
            if spec is None:
                continue
            value = nutrition.get(nutrient)
            scored.append(score_nutrient(nutrient, None if value in (None, "") else float(value), spec))

        nss = nutritional_sufficiency_score(scored, phase)
        calories = nutrition.get("calories_kcal")
        interactions = detect_interactions(
            scored, active_deviations(number), nutrition,
            float(calories) if calories not in (None, "") else None)

        daily.append({
            "day": number, "phase": phase, "status": "scored",
            "zinc_deficient": deficient,
            "nutrients": scored,
            "nss": nss,
            "nss_action": nss_action(nss),
            "gaps": [r for r in scored if r.get("priority") in ("HIGH", "MODERATE", "EXCESS")],
            "interactions": interactions,
        })

    latest = next((d for d in reversed(daily) if d["status"] == "scored"), None)

    limitations = [NO_IMAGING_LIMITATION]
    if not patient.get("weight_kg"):
        limitations.append(
            "No body weight supplied, so the per-kilogram targets (protein, "
            "calories, hydration) could not be resolved and were not scored.")
    unlogged = sorted({r["label"] for d in daily if d["status"] == "scored"
                       for r in d["nutrients"] if r["status"] == "not_logged"})
    if unlogged:
        limitations.append(
            f"Not logged by the dietitian on one or more days: {', '.join(unlogged)}. "
            "These were left unscored rather than treated as zero intake.")

    escalate: List[str] = []
    if latest:
        escalate = list(latest["nss_action"].get("escalate_to", []))
        if latest["interactions"] and "Dietitian" not in escalate:
            escalate.append("Dietitian")
        if latest["interactions"] and "Physician" not in escalate:
            escalate.append("Physician")

    return {
        "patient_ref": timeline.get("patient_ref"),
        "days_post_op": int(days[-1].get("day", 0)),
        "current_phase": layer1.get("current_phase"),
        "nss": latest["nss"] if latest else None,
        "nss_action": latest["nss_action"] if latest else nss_action(None),
        "gaps": latest["gaps"] if latest else [],
        "interactions": latest["interactions"] if latest else [],
        "escalation_required": bool(latest and (latest["nss_action"]["escalation"] != "none"
                                                or latest["interactions"])),
        "escalate_to": escalate,
        "daily": daily,
        "limitations": limitations,
        "model_version": MODEL_VERSION,
    }


# ===========================================================================
# 6. TERMINAL OUTPUT
# ===========================================================================

def render(result: Dict[str, Any]) -> str:
    """Render the score as a dietitian would want to read it.

    Separate from the scoring so the result stays pure data — the service returns
    JSON, the terminal prints a table, and neither shapes the other.
    """
    lines = ["=" * 78,
             " POST-OP PHASE 1 - LAYER 3 : NUTRITIONAL GAP ENGINE",
             "=" * 78,
             f" Patient        {result.get('patient_ref') or '-'}",
             f" Day post-op    {result['days_post_op']}",
             f" Phase (L1)     {result.get('current_phase') or '-'}",
             ""]

    nss = result["nss"]
    lines.append(f" NUTRITIONAL SUFFICIENCY SCORE   {nss if nss is not None else 'n/a'}")
    lines.append(f"   {result['nss_action']['detail']}")
    lines.append("")

    latest = next((d for d in reversed(result["daily"]) if d["status"] == "scored"), None)
    if latest:
        lines.append(f" NUTRIENTS  (day {latest['day']}, {latest['phase']} targets)")
        lines.append(f"   {'Nutrient':<16}{'Target':>10}{'Actual':>10}{'Gap':>8}   Priority")
        for row in latest["nutrients"]:
            if row["status"] == "not_logged":
                lines.append(f"   {row['label']:<16}{row['target']:>10}{'—':>10}{'—':>8}   not logged")
            else:
                lines.append(f"   {row['label']:<16}{row['target']:>10}{row['actual']:>10}"
                             f"{row['gap_pct']:>7.0f}%   {row['priority']}")
    lines.append("")

    if result["interactions"]:
        lines.append(" INTERACTION FLAGS   (section 6.3 — nutrition x deviation)")
        for item in result["interactions"]:
            lines += [f"   {item['severity']}  {item['id']}",
                      f"     {item['alert_text']}",
                      f"     {item['clinical_interpretation']}"]
    else:
        lines.append(" INTERACTION FLAGS   none")

    if result["escalate_to"]:
        lines += ["", " ESCALATE TO   " + ", ".join(result["escalate_to"])]

    lines += ["", " LIMITATIONS"]
    lines += [f"   * {text}" for text in result["limitations"]]
    lines += ["", f" {result['model_version']}"]
    return "\n".join(lines)


# ===========================================================================
# 7. SERVICE — stateless, same shape as Layers 1 and 2
# ===========================================================================

try:
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    class ScoreRequest(BaseModel):
        """The HTTP contract.

        layer1 and layer2 are optional so the endpoint is usable standalone for a
        demo; the orchestrator always supplies them so the upstream layers run
        once rather than three times.
        """
        model_config = ConfigDict(extra="allow")
        patient_ref: Optional[str] = None
        patient: Dict[str, Any] = Field(default_factory=dict)
        days: List[Dict[str, Any]] = Field(default_factory=list)
        clinician_phase_overrides: List[Dict[str, Any]] = Field(default_factory=list)
        layer1: Optional[Dict[str, Any]] = None
        layer2: Optional[Dict[str, Any]] = None

    app = FastAPI(title="Post-op Phase 1 - Layer 3", version=MODEL_VERSION)

    def _token() -> Optional[str]:
        """Read the service token from the environment.

        Three fallback names, matching the convention already used by
        Postop-CQL-Service, so deployment config stays uniform across services.
        """
        for name in ("POSTOP_PHASE1_SERVICE_API_KEY", "FOODHAK_API_TOKEN", "LANGGRAPH_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Liveness, and self-documentation — it reports which nutrients and
        thresholds the running build actually uses, so a deployment can be
        verified without reading the source."""
        return {"status": "ok", "service": "postop-phase1-layer3",
                "model_version": MODEL_VERSION,
                "nutrients": list(NUTRIENT_TARGETS),
                "interactions": [r["id"] for r in INTERACTIONS],
                "nss_thresholds": {"escalate": NSS_ESCALATE, "review": NSS_REVIEW},
                "auth_required": _token() is not None}

    @app.post("/postop/phase1/score-nutrition")
    async def score(
        request: ScoreRequest,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """Deliberately thin: auth and error translation only.

        All logic stays in score_nutrition() so HTTP concerns never leak into the
        clinical rules, and the rules stay testable without a server.
        """
        expected = _token()
        if expected:
            received = x_api_key
            if authorization and authorization.lower().startswith("bearer "):
                received = authorization[7:].strip()
            if received != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing service token")
        payload = request.model_dump()
        try:
            layer1 = payload.get("layer1") or estimate_phase(payload)
            layer2 = payload.get("layer2") or detect_deviations(payload, layer1)
            return score_nutrition(payload, layer1, layer2)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

except ImportError as exc:  # the CLI must still work without FastAPI installed
    _MISSING_DEPENDENCY = (
        f"The HTTP service is disabled: {exc}. "
        "Install the service dependencies with:  pip install -r requirements.txt"
    )
    print(f"layer3_nutrition_gap: {_MISSING_DEPENDENCY}\n"
          "The command-line tool still works.", file=sys.stderr)

    async def app(scope, receive, send):  # type: ignore[misc]
        """Stand-in so an ASGI server reports the real problem rather than
        'NoneType is not callable', which says nothing about the cause."""
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
    """Command-line entry.

    Argument parsing and file I/O kept out of the engine, so score_nutrition()
    never needs to know whether it was called from a terminal, a web request or
    a test.
    """
    parser = argparse.ArgumentParser(
        description="Score nutritional intake against phase-specific targets.")
    parser.add_argument("--timeline", type=Path, required=True,
                        help="Path to a timeline JSON file with a nutrition log")
    parser.add_argument("--json", action="store_true", help="Raw JSON instead of the table")
    args = parser.parse_args(argv)

    timeline = json.loads(args.timeline.read_text())
    try:
        layer1 = estimate_phase(timeline)
        layer2 = detect_deviations(timeline, layer1)
        result = score_nutrition(timeline, layer1, layer2)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
