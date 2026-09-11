"""
Phase 1 macro profile — the healing-phase counterpart to Macro-Details.xlsm.

Phase 2 resolves macro targets from a sheet keyed on
    action_space + macro_profile_version + sex + age_band + weight_band + height_band
Phase 1 needs the same lookup with `healing_phase` in place of `action_space`.

WHAT THIS SHEET DOES NOT DO
---------------------------
It does not define calories, protein, fluid, vitamin C, zinc, arginine or
vitamin A. Layer 3 (layer3_nutrition_gap.py) already establishes those from
ASPEN/ESPEN for the current healing phase, scaled by actual body weight. Putting
them here as well would create two sources of truth that disagree — and the
sheet's weight *bands* are coarser than Layer 3's per-kg arithmetic, so the
sheet would be the worse of the two.

This sheet supplies only what Layer 3 leaves open:

    * how the non-protein energy splits between carbohydrate and fat
    * fibre
    * the safety caps: saturated fat, added sugar, cholesterol, sodium
    * the micronutrients Layer 3 does not cover: calcium, iron, potassium,
      vitamin D, vitamin E

Carbohydrate and fat are stored as a SHARE OF NON-PROTEIN ENERGY, not as grams.
Grams are derived at resolve time from Layer 3's calorie and protein targets, so
the three can never contradict each other:

    protein_kcal    = layer3.protein_g * 4
    non_protein     = layer3.calories_kcal - protein_kcal
    carbohydrate_g  = non_protein * carb_share / 4
    fat_g           = non_protein * (1 - carb_share) / 9

PROVENANCE
----------
Every number below is tagged. `DRI/DGA` values are published population
reference intakes, copied from the Baselines tab of Macro-Details.xlsm so the
two workbooks agree. `DERIVED` values are ours — a documented lever applied to
the baseline, not a published wound-care figure. No lever here is presented as
evidence when it is judgement; the dietitian review column exists for exactly
that reason.

Usage:
    python3 macro_profile_phase1.py --build          # write xlsx + csv
    python3 macro_profile_phase1.py --resolve fixtures/patient_ssi.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

MACRO_PROFILE_VERSION = "phase1_v1"

# Layer 3 owns these; the sheet must never define them.
LAYER3_OWNED = (
    "calories_kcal",
    "protein_g",
    "fluid_ml",
    "vitamin_c_mg",
    "zinc_mg",
    "arginine_g",
    "vitamin_a_iu",
)

PHASES = ("HAEMOSTASIS", "INFLAMMATION", "PROLIFERATION", "REMODELLING")

# ===========================================================================
# 1. BASELINES — DRI / Dietary Guidelines, general healthy adult
#    Copied verbatim from Macro-Details.xlsm -> 'Baselines' so Phase 1 and
#    Phase 2 start from identical population references.
# ===========================================================================

AGE_BANDS = ("<19", "19-30", "31-50", "51-70", "71+")

# sex -> age_band -> baseline. Source: DRI/DGA (Macro-Details.xlsm Baselines).
BASELINES: Dict[str, Dict[str, Dict[str, float]]] = {
    "Male": {
        "<19":   {"calories": 2600.0, "protein": 117.0, "carbs": 325.0, "fat": 92.0,
                  "sodium_max": 2300.0, "sat_fat_max": 29.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 65.0, "calcium": 1300.0, "iron": 11.0, "potassium": 3400.0},
        "19-30": {"calories": 2600.0, "protein": 117.0, "carbs": 325.0, "fat": 92.0,
                  "sodium_max": 2300.0, "sat_fat_max": 29.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 65.0, "calcium": 1000.0, "iron": 8.0, "potassium": 3400.0},
        "31-50": {"calories": 2400.0, "protein": 108.0, "carbs": 300.0, "fat": 85.0,
                  "sodium_max": 2300.0, "sat_fat_max": 27.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 60.0, "calcium": 1000.0, "iron": 8.0, "potassium": 3400.0},
        "51-70": {"calories": 2200.0, "protein": 110.0, "carbs": 264.0, "fat": 78.0,
                  "sodium_max": 2300.0, "sat_fat_max": 24.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 55.0, "calcium": 1000.0, "iron": 8.0, "potassium": 3400.0},
        "71+":   {"calories": 2000.0, "protein": 110.0, "carbs": 230.0, "fat": 71.0,
                  "sodium_max": 2300.0, "sat_fat_max": 22.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 50.0, "calcium": 1200.0, "iron": 8.0, "potassium": 3400.0},
    },
    "Female": {
        "<19":   {"calories": 2000.0, "protein": 90.0, "carbs": 250.0, "fat": 71.0,
                  "sodium_max": 2300.0, "sat_fat_max": 22.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 50.0, "calcium": 1300.0, "iron": 15.0, "potassium": 2600.0},
        "19-30": {"calories": 2000.0, "protein": 90.0, "carbs": 250.0, "fat": 71.0,
                  "sodium_max": 2300.0, "sat_fat_max": 22.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 50.0, "calcium": 1000.0, "iron": 18.0, "potassium": 2600.0},
        "31-50": {"calories": 1900.0, "protein": 86.0, "carbs": 238.0, "fat": 68.0,
                  "sodium_max": 2300.0, "sat_fat_max": 21.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 48.0, "calcium": 1000.0, "iron": 18.0, "potassium": 2600.0},
        "51-70": {"calories": 1800.0, "protein": 90.0, "carbs": 216.0, "fat": 64.0,
                  "sodium_max": 2300.0, "sat_fat_max": 20.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 45.0, "calcium": 1200.0, "iron": 8.0, "potassium": 2600.0},
        "71+":   {"calories": 1700.0, "protein": 94.0, "carbs": 196.0, "fat": 60.0,
                  "sodium_max": 2300.0, "sat_fat_max": 19.0, "cholesterol_max": 300.0,
                  "added_sugar_max": 42.0, "calcium": 1200.0, "iron": 8.0, "potassium": 2600.0},
    },
}

# Vitamin D / E are not in the Phase 2 Baselines tab; taken from the same DRI set
# the rest of that tab uses.  Source: DRI.
VITAMIN_D_IU = {"<19": 600.0, "19-30": 600.0, "31-50": 600.0, "51-70": 600.0, "71+": 800.0}
VITAMIN_E_MG = 15.0


def baseline_carb_share(sex: str, age_band: str) -> float:
    """Carbohydrate's share of NON-PROTEIN energy in the population baseline.

    Derived arithmetically from the Baselines row itself, so the phase levers
    below are expressed as deltas against a real published split rather than an
    invented starting point. Male 19-30 works out at 0.611.
    """
    b = BASELINES[sex][age_band]
    carb_kcal = b["carbs"] * 4.0
    fat_kcal = b["fat"] * 9.0
    return carb_kcal / (carb_kcal + fat_kcal)


# ===========================================================================
# 2. HEALING-PHASE LEVERS
# ===========================================================================
# Each lever states its own provenance. `DERIVED` means we chose it; it is a
# defensible starting point for dietitian review, not a published wound-care
# number. Section references are to Phase_1_Wound_Recovery.docx.

PHASE_PROFILES: Dict[str, Dict[str, Any]] = {
    "HAEMOSTASIS": {
        # PROVISIONED but NOT SCORED — the two are separate questions and the
        # evidence answers them differently.
        #
        # What to OFFER is settled. ESPEN's surgery guideline: "Oral intake,
        # including clear liquids, shall be initiated within hours after surgery
        # in most patients" (Grade A, 100% consensus), and oral intake "shall be
        # continued after surgery without interruption" (Grade A). Delayed oral
        # intake has not proven beneficial even after colorectal resection. The
        # ERAS colorectal guideline agrees: intake resumed within hours, strong
        # recommendation. Offering nothing on day 0 is the option NOT supported.
        #
        # What to SCORE against is not settled. The systematic review of early
        # oral feeding after elective bowel surgery reports tolerance of only
        # 55-86%, vomiting in one trial at 48% vs 33%, and NO data on energy or
        # protein actually achieved in the first 24-48 h. Scoring adequacy here
        # would mark a patient who managed a few sips as severely deficient and
        # escalate them to a dietitian, when the guidelines only ever asked us
        # to offer. So Layer 3 still returns not_scored and NSS stays null.
        "scored": False,
        "provisioned": True,
        "form": "clear_fluids",
        # ERAS gives the one concrete early-postoperative figure: oral
        # nutritional supplements of at least 500 kcal/day alongside regular
        # food for the first 3-5 days. Used here as a PROVISION anchor (what to
        # put in front of the patient), never as an adequacy target.
        "provision_kcal_per_day": 500.0,
        "provision_kcal_source": "ERAS colorectal 2025 — ONS >= 500 kcal/day, days 1-5",
        # Clear fluids only, per clinical direction for day 0 after open
        # abdominal surgery. Layer 1 pins the WHOLE calendar day to haemostasis,
        # so these are offered from hour one and must be safe at hour one —
        # which is what rules out the low-residue solids the trial evidence
        # supports from POD 1 onward.
        "texture": "clear fluids only — no solids, no residue, no dairy",
        "serving_note": "offer as tolerated in small volumes; not a prescription",
        "rationale": (
            "Day 0 is provisioned with clear fluids because ESPEN and ERAS both "
            "direct that oral intake begin within hours of surgery, but it is "
            "not scored because no evidence establishes what a patient should "
            "achieve in the first 24 h."
        ),
        "provenance": ("Phase_1_Wound_Recovery.docx section 6.1; "
                       "ESPEN clinical nutrition in surgery; "
                       "ERAS colorectal 2025"),
    },
    "INFLAMMATION": {
        "scored": True,
        # Hyperglycaemia impairs neutrophil function and collagen deposition, and
        # section 5.3 already treats glucose > 180 as a deviation. Shifting a
        # little energy from carbohydrate to fat, and tightening the added-sugar
        # cap, is the dietary lever consistent with that.
        "carb_share_delta": -0.03,
        "added_sugar_cap_multiplier": 0.60,
        "sat_fat_cap_multiplier": 0.85,
        "sodium_cap_multiplier": 1.00,
        "fibre_g": 25.0,
        "micro_multipliers": {"potassium": 1.10, "calcium": 1.00, "iron": 1.00,
                              "vitamin_d": 1.00, "vitamin_e": 1.00},
        "rationale": (
            "Hypermetabolic, insulin-resistant phase. Energy is shifted marginally "
            "away from carbohydrate and the added-sugar cap is tightened, because "
            "hyperglycaemia impairs healing and section 5.3 flags glucose > 180 "
            "as a deviation. Potassium is raised for drain/ileus losses."
        ),
        "provenance": "carb/sugar direction: ASPEN 2016 + section 5.3 (DERIVED magnitudes)",
    },
    "PROLIFERATION": {
        "scored": True,
        # Collagen synthesis peaks here. Adequate non-protein energy spares
        # protein for tissue synthesis rather than oxidation - the same logic
        # Layer 3's own interaction rule uses when calories are short and
        # prealbumin is not rising.
        "carb_share_delta": +0.03,
        "added_sugar_cap_multiplier": 0.75,
        "sat_fat_cap_multiplier": 0.90,
        "sodium_cap_multiplier": 1.00,
        "fibre_g": 30.0,
        "micro_multipliers": {"potassium": 1.00, "calcium": 1.00, "iron": 1.25,
                              "vitamin_d": 1.00, "vitamin_e": 1.00},
        "rationale": (
            "Collagen synthesis peak. Carbohydrate share is raised so non-protein "
            "energy spares protein for tissue synthesis. Iron is raised because it "
            "is the cofactor for prolyl/lysyl hydroxylase in collagen cross-linking."
        ),
        "provenance": "iron/collagen: ESPEN 2017; protein-sparing: ASPEN 2016 (DERIVED magnitudes)",
    },
    "REMODELLING": {
        "scored": True,
        "carb_share_delta": 0.0,
        "added_sugar_cap_multiplier": 1.00,
        "sat_fat_cap_multiplier": 1.00,
        "sodium_cap_multiplier": 1.00,
        "fibre_g": 30.0,
        "micro_multipliers": {"potassium": 1.00, "calcium": 1.00, "iron": 1.00,
                              "vitamin_d": 1.00, "vitamin_e": 1.00},
        "rationale": (
            "Wound strength is maturing and demand returns toward baseline. No "
            "lever is applied; the population reference is the target."
        ),
        "provenance": "DRI/DGA baseline, no adjustment",
    },
}


# ===========================================================================
# 3. NO INTENSITY DIMENSION — deliberately
# ===========================================================================
# Phase 2's sheet has step_down/maintain/step_up, driven by weekly ADHERENCE:
# a behavioural signal about whether the patient is following the plan, which
# genuinely warrants tightening or easing execution.
#
# Phase 1 has no adherence signal. Its nearest equivalent, Layer 3's NSS, is a
# nutritional ADEQUACY score — a low NSS means the patient is undernourished.
# Tightening discretionary caps in that state restricts accessible energy from
# someone who is already not eating enough, which is backwards. The shortfall is
# also already fully expressed, per nutrient and per cent, in
# `nutrient_priorities`; encoding it again as cap multipliers would say the same
# thing worse.
#
# So the caps here are phase-based safety ceilings and nothing moves them. NSS
# still travels with the target for context, but it is not a sheet key.

# ===========================================================================
# 4. BANDS — identical vocabulary to Macro-Details.xlsm 'Full grid'
# ===========================================================================

WEIGHT_BANDS = ("30-50", "50-65", "65-80", "80-100", "100-120", "120+")
HEIGHT_BANDS = ("140-155", "155-165", "165-175", "175-185", "185-200")


def _band_contains(band: str, value: float) -> bool:
    """Inclusive range match, mirroring recipe_pool_repository._band_contains."""
    text = str(band or "").replace("–", "-").replace("—", "-").replace(" ", "").casefold()
    if text in {"any", "all"}:
        return True
    if text.endswith("+"):
        try:
            return float(value) >= float(text[:-1])
        except ValueError:
            return False
    if "-" in text:
        lo, _, hi = text.partition("-")
        try:
            return float(lo) <= float(value) <= float(hi)
        except ValueError:
            return False
    return False


def band_for(bands: Tuple[str, ...], value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    for band in bands:
        if _band_contains(band, value):
            return band
    return None


def age_band_for(age: Optional[float]) -> Optional[str]:
    if age is None:
        return None
    if age < 19:
        return "<19"
    for band in ("19-30", "31-50", "51-70"):
        lo, hi = band.split("-")
        if float(lo) <= age <= float(hi):
            return band
    return "71+"


def normalise_sex(value: Any) -> Optional[str]:
    text = str(value or "").strip().casefold()
    if text in {"m", "male"}:
        return "Male"
    if text in {"f", "female"}:
        return "Female"
    return None


# ===========================================================================
# 5. THE PROFILE ROW — what the sheet actually stores
# ===========================================================================


def build_profile_row(
    *,
    phase: str,
    sex: str,
    age_band: str,
) -> Optional[Dict[str, Any]]:
    """One sheet row: ratios and caps only, never calories or protein."""
    profile = PHASE_PROFILES[phase]
    if not profile.get("scored"):
        return None

    base = BASELINES[sex][age_band]

    share = baseline_carb_share(sex, age_band) + profile["carb_share_delta"]
    share = min(max(share, 0.45), 0.75)  # keep the split physiologically sane
    micro = profile["micro_multipliers"]

    return {
        "macro_profile_version": MACRO_PROFILE_VERSION,
        "healing_phase": phase,
        "sex": sex,
        "age_band": age_band,
        # --- energy distribution (grams derived from Layer 3 at resolve time) ---
        "carb_share_of_non_protein_energy": round(share, 4),
        "fat_share_of_non_protein_energy": round(1.0 - share, 4),
        # --- absolute targets the sheet owns outright ---
        "fibre_g": round(profile["fibre_g"], 1),
        "saturated_fat_g_max": round(base["sat_fat_max"] * profile["sat_fat_cap_multiplier"], 1),
        "added_sugar_g_max": round(base["added_sugar_max"] * profile["added_sugar_cap_multiplier"], 1),
        "cholesterol_mg_max": round(base["cholesterol_max"], 0),
        "sodium_mg_max": round(base["sodium_max"] * profile["sodium_cap_multiplier"], 0),
        "calcium_mg": round(base["calcium"] * micro["calcium"], 0),
        "iron_mg": round(base["iron"] * micro["iron"], 1),
        "potassium_mg": round(base["potassium"] * micro["potassium"], 0),
        "vitamin_d_iu": round(VITAMIN_D_IU[age_band] * micro["vitamin_d"], 0),
        "vitamin_e_mg": round(VITAMIN_E_MG * micro["vitamin_e"], 1),
        "provenance": profile["provenance"],
        "rationale": profile["rationale"],
    }


def build_grid() -> List[Dict[str, Any]]:
    """Every (phase x sex x age x weight x height x intensity) combination.

    Weight and height bands do not change any value in this sheet — Layer 3
    already applies body weight per-kg, so re-applying it here would double-count.
    They are still emitted because the Phase 2 recipe_pool schema keys on them and
    the loader matches on all six columns.
    """
    rows: List[Dict[str, Any]] = []
    for phase in PHASES:
        if not PHASE_PROFILES[phase].get("scored"):
            continue
        for sex in ("Male", "Female"):
            for age_band in AGE_BANDS:
                row = build_profile_row(phase=phase, sex=sex, age_band=age_band)
                if not row:
                    continue
                for weight_band in WEIGHT_BANDS:
                    for height_band in HEIGHT_BANDS:
                        rows.append({**row,
                                     "weight_band_kg": weight_band,
                                     "height_band_cm": height_band})
    return rows


# ===========================================================================
# 6. RESOLVE — Layer 3 targets + sheet row -> one complete macro target
# ===========================================================================


# Nutrients the recipe generator cannot optimise against, because no ingredient
# dataset we hold carries them. An arginine gap in proliferation has no proxy at
# all — it can only be closed by supplementation — so it is stated rather than
# left to fail silently downstream.
UNOPTIMISABLE_NUTRIENTS = ("zinc_mg", "arginine_g", "vitamin_a_iu", "fibre_g", "fluid_ml")

# Layer 3's own priority vocabulary, ordered worst-first.
_PRIORITY_RANK = {"HIGH": 0, "MODERATE": 1, "ADEQUATE": 2}


def NUTRIENT_TARGETS_FOR_PHASE(phase: str, weight_kg: float) -> Dict[str, float]:
    """Layer 3's own targets for a phase and body weight, without running Layer 3.

    Used by the offline pool-population job, which has no patient timeline to
    score — only a demographic combination. It imports layer3_nutrition_gap's
    NUTRIENT_TARGETS table directly so the offline job and the live pipeline can
    never disagree about what a phase requires.

    Conditional nutrients (zinc, arginine, vitamin A) are omitted: they depend on
    per-patient state (serum zinc, at-risk flag) that a demographic combination
    does not carry, and Layer 3 leaves them unscored in exactly that situation.
    """
    import layer3_nutrition_gap as L3

    out: Dict[str, float] = {}
    for nutrient, spec in L3.NUTRIENT_TARGETS.items():
        if spec.get("conditional_on") or spec.get("conditional_on_phase"):
            continue
        band = spec.get(phase)
        if not band:
            continue
        low = band[0] if isinstance(band, (tuple, list)) else band
        out[nutrient] = round(float(low) * float(weight_kg), 1) if spec.get("per_kg") else float(low)
    return out


def build_nutrient_priorities(phase1_nutrition_targets: Dict[str, Any]) -> Dict[str, Any]:
    """What the patient is short of, worst first.

    This is the part of Phase 1 the recipe generator genuinely cannot derive: it
    is Layer 3's clinical read of intake against the phase target, not a lookup.
    The generator owns ingredient choice — it is the only component that knows
    portion sizes — so Phase 1 passes the assessment, not a shortlist.
    """
    targets = (phase1_nutrition_targets or {}).get("targets") or {}

    ranked: List[Dict[str, Any]] = []
    for nutrient, row in targets.items():
        if not isinstance(row, dict):
            continue
        priority = str(row.get("priority") or "").upper()
        ranked.append({
            "nutrient": nutrient,
            "label": row.get("label"),
            "priority": priority or None,
            "target": row.get("target"),
            "actual": row.get("actual"),
            "unit": row.get("unit"),
            "gap_pct": row.get("gap_pct"),
            "source": row.get("source"),
            "optimisable": nutrient not in UNOPTIMISABLE_NUTRIENTS,
        })

    ranked.sort(key=lambda r: (
        _PRIORITY_RANK.get(r["priority"] or "", 3),
        -(r["gap_pct"] or 0.0),
    ))

    shortfalls = [r for r in ranked if (r["gap_pct"] or 0.0) > 0]
    return {
        "schema_version": "phase1_nutrient_priorities_v1",
        "source": "layer3_nutrition_gap",
        "nss": (phase1_nutrition_targets or {}).get("nss"),
        "nss_escalation": (phase1_nutrition_targets or {}).get("nss_escalation"),
        "priorities": ranked,
        "shortfalls": [r["nutrient"] for r in shortfalls],
        "high_priority": [r["nutrient"] for r in ranked if r["priority"] == "HIGH"],
        # Stated, never silently dropped.
        "not_optimisable": [r["nutrient"] for r in ranked if not r["optimisable"]],
        "guidance": (
            "Ingredient selection belongs to the recipe generator, which knows "
            "portion sizes. These are the nutrients to prioritise and by how much."
        ),
    }


def resolve_phase1_macro_target(
    *,
    phase1_nutrition_targets: Dict[str, Any],
    sex: Any,
    age: Optional[float],
    weight_kg: Optional[float] = None,
    height_cm: Optional[float] = None,
) -> Dict[str, Any]:
    """Combine Layer 3's targets with the sheet into a full macro target.

    Layer 3 is authoritative for everything it scored. The sheet fills the rest.
    Carbohydrate and fat grams are computed from Layer 3's own calorie and
    protein figures, so the three can never disagree.
    """
    out: Dict[str, Any] = {
        "schema_version": "phase1_resolved_macro_target_v1",
        "macro_profile_version": MACRO_PROFILE_VERSION,
        "source": "layer3_nutrition_gap + phase1_macro_profile",
    }

    targets = (phase1_nutrition_targets or {}).get("targets") or {}
    phase = (phase1_nutrition_targets or {}).get("current_phase")
    # Carried for context only — NSS no longer keys the sheet. See section 3.
    escalation = (phase1_nutrition_targets or {}).get("nss_escalation")

    resolved_sex = normalise_sex(sex)
    age_band = age_band_for(age)

    out["healing_phase"] = phase
    out["nss_escalation"] = escalation
    out["sex"] = resolved_sex
    out["age_band"] = age_band
    out["weight_band_kg"] = band_for(WEIGHT_BANDS, weight_kg)
    out["height_band_cm"] = band_for(HEIGHT_BANDS, height_cm)

    # --- everything Layer 3 established, copied verbatim -------------------
    layer3_block: Dict[str, Any] = {}
    for nutrient, row in targets.items():
        if isinstance(row, dict) and row.get("target") is not None:
            layer3_block[nutrient] = row["target"]
    out["from_layer3"] = layer3_block

    blockers: List[str] = []
    if phase not in PHASE_PROFILES:
        blockers.append(f"unknown healing_phase {phase!r}")
    elif not PHASE_PROFILES[phase].get("scored"):
        blockers.append(f"{phase} is not scored for nutrition (section 6.1)")
    if resolved_sex is None:
        blockers.append("sex missing or unrecognised")
    if age_band is None:
        blockers.append("age missing")

    if blockers:
        out["status"] = "unresolved"
        out["blocked_by"] = blockers
        out["targets"] = dict(layer3_block)
        # Layer 3's assessment is still valid even when the sheet lookup is not.
        out["nutrient_priorities"] = build_nutrient_priorities(phase1_nutrition_targets)
        return out

    row = build_profile_row(phase=phase, sex=resolved_sex, age_band=age_band)
    assert row is not None  # guarded by the blocker check above

    merged: Dict[str, Any] = dict(layer3_block)

    calories = layer3_block.get("calories_kcal")
    protein = layer3_block.get("protein_g")
    if calories is not None and protein is not None:
        non_protein = max(float(calories) - float(protein) * 4.0, 0.0)
        carb_share = row["carb_share_of_non_protein_energy"]
        merged["carbohydrate_g"] = round(non_protein * carb_share / 4.0, 1)
        merged["fat_g"] = round(non_protein * (1.0 - carb_share) / 9.0, 1)
        out["energy_split"] = {
            "non_protein_kcal": round(non_protein, 1),
            "carb_share_of_non_protein_energy": carb_share,
            "derived_from": "layer3 calories_kcal and protein_g",
        }
    else:
        out.setdefault("warnings", []).append(
            "carbohydrate_g and fat_g not derived: Layer 3 did not score both "
            "calories_kcal and protein_g for this patient"
        )

    for key in ("fibre_g", "saturated_fat_g_max", "added_sugar_g_max", "cholesterol_mg_max",
                "sodium_mg_max", "calcium_mg", "iron_mg", "potassium_mg",
                "vitamin_d_iu", "vitamin_e_mg"):
        merged[key] = row[key]

    out["status"] = "resolved"
    out["nutrient_priorities"] = build_nutrient_priorities(phase1_nutrition_targets)
    out["targets"] = merged
    out["from_sheet"] = [k for k in merged if k not in layer3_block]
    out["provenance"] = row["provenance"]
    # Documented, per decision 4: LangGraph state cannot currently supply these.
    out["not_tracked_by_langgraph_state"] = ["zinc_mg", "arginine_g", "vitamin_a_iu", "fluid_ml"]
    return out


# ===========================================================================
# 7. EXPORT
# ===========================================================================

CSV_COLUMNS = [
    "macro_profile_version", "healing_phase", "sex", "age_band",
    "weight_band_kg", "height_band_cm",
    "carb_share_of_non_protein_energy", "fat_share_of_non_protein_energy",
    "fibre_g", "saturated_fat_g_max", "added_sugar_g_max", "cholesterol_mg_max",
    "sodium_mg_max", "calcium_mg", "iron_mg", "potassium_mg",
    "vitamin_d_iu", "vitamin_e_mg", "provenance",
]


def export_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_workbook(rows: List[Dict[str, Any]], path: str) -> None:
    """Dietitian-review workbook, mirroring Macro-Details.xlsm's review layout."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        raise SystemExit("openpyxl is required to write the workbook: pip install openpyxl")

    wb = openpyxl.Workbook()
    head_font = Font(bold=True)
    yellow = PatternFill("solid", fgColor="FFF2CC")
    grey = PatternFill("solid", fgColor="EEEEEE")

    intro = wb.active
    intro.title = "Read me"
    for r, line in enumerate([
        ("Phase 1 macro profile — healing-phase target matrix",),
        (f"macro_profile_version: {MACRO_PROFILE_VERSION}",),
        ("",),
        ("WHAT THIS SHEET DOES NOT DEFINE",),
        ("Calories, protein, fluid, vitamin C, zinc, arginine and vitamin A are set by Layer 3",),
        ("(the nutritional gap engine) from ASPEN/ESPEN for the current healing phase, scaled by",),
        ("the patient's actual body weight. They are deliberately absent here so there is exactly",),
        ("one source of truth for them.",),
        ("",),
        ("HOW CARBOHYDRATE AND FAT WORK",),
        ("Stored as a share of NON-PROTEIN energy, not as grams. Grams are computed at resolve",),
        ("time from Layer 3's own calorie and protein targets, so the three cannot contradict.",),
        ("",),
        ("NO INTENSITY DIMENSION",),
        ("Phase 2 varies targets by weekly adherence (step_down/maintain/step_up). Phase 1 has no",),
        ("adherence signal. Its nearest equivalent, NSS, measures nutritional ADEQUACY — a low NSS",),
        ("means the patient is undernourished, and tightening caps then would restrict energy from",),
        ("someone already not eating enough. The shortfall is carried per-nutrient in",),
        ("nutrient_priorities instead. Caps here are phase-based safety ceilings; nothing moves them.",),
        ("",),
        ("HOW TO REVIEW",),
        ("1. Work phase by phase on the 'Profile grid' tab.",),
        ("2. In the yellow Approve? column pick Approve / Revise / Reject and add notes.",),
        ("3. Values marked DERIVED in the provenance column are ours, not published wound-care",),
        ("   figures. Those are the ones that most need your judgement.",),
        ("4. HAEMOSTASIS generates no rows: Layer 3 does not score nutrition in a 0-3 h phase.",),
    ], start=1):
        intro.cell(row=r, column=1, value=line[0])
    intro.cell(row=1, column=1).font = Font(bold=True, size=14)
    intro.column_dimensions["A"].width = 100

    # --- levers tab ---------------------------------------------------------
    lev = wb.create_sheet("Phase levers")
    lev.append(["Healing phase", "Scored?", "Carb share delta", "Added-sugar cap x",
                "Sat-fat cap x", "Fibre (g)", "Iron x", "Potassium x", "Provenance", "Rationale"])
    for c in lev[1]:
        c.font = head_font
        c.fill = grey
    for phase in PHASES:
        p = PHASE_PROFILES[phase]
        if not p.get("scored"):
            lev.append([phase, "no", "", "", "", "", "", "", p["provenance"], p["rationale"]])
            continue
        m = p["micro_multipliers"]
        lev.append([phase, "yes", p["carb_share_delta"], p["added_sugar_cap_multiplier"],
                    p["sat_fat_cap_multiplier"], p["fibre_g"], m["iron"], m["potassium"],
                    p["provenance"], p["rationale"]])
    for col, w in zip("ABCDEFGHIJ", (16, 8, 16, 18, 14, 10, 8, 12, 46, 80)):
        lev.column_dimensions[col].width = w

    # --- the grid -----------------------------------------------------------
    grid = wb.create_sheet("Profile grid")
    headers = CSV_COLUMNS + ["Approve?", "Dietitian notes"]
    grid.append(headers)
    for c in grid[1]:
        c.font = head_font
        c.fill = grey
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        grid.append([row.get(k) for k in CSV_COLUMNS] + ["", ""])
    for idx in (len(CSV_COLUMNS) + 1, len(CSV_COLUMNS) + 2):
        letter = grid.cell(row=1, column=idx).column_letter
        grid.column_dimensions[letter].width = 26
        for r in range(2, grid.max_row + 1):
            grid.cell(row=r, column=idx).fill = yellow
    grid.freeze_panes = "A2"

    wb.save(path)


# ===========================================================================
# 8. CLI
# ===========================================================================


def _resolve_from_timeline(path: str) -> Dict[str, Any]:
    """Run a Phase 1 timeline through Layer 3, then resolve the macro target."""
    sys.path.insert(0, ".")
    from layer1_phase_estimator import estimate_phase
    from layer2_deviation_detector import detect_deviations
    from layer3_nutrition_gap import score_nutrition

    timeline = json.load(open(path))
    l1 = estimate_phase(timeline)
    l2 = detect_deviations(timeline, layer1=l1)
    l3 = score_nutrition(timeline, layer1=l1, layer2=l2)

    # Same shape the LangGraph node stores in phase1_nutrition_targets.
    daily = l3.get("daily") or []
    latest = next((d for d in reversed(daily) if isinstance(d.get("nutrients"), list)), {})
    targets = {
        r["nutrient"]: r
        for r in (latest.get("nutrients") or [])
        if isinstance(r, dict) and r.get("status") == "scored" and r.get("target") is not None
    }
    block = {
        "current_phase": l3.get("current_phase"),
        "nss": l3.get("nss"),
        "nss_escalation": (l3.get("nss_action") or {}).get("escalation"),
        "targets": targets,
    }
    patient = timeline.get("patient") or {}
    return resolve_phase1_macro_target(
        phase1_nutrition_targets=block,
        sex=patient.get("sex") or patient.get("gender"),
        age=patient.get("age"),
        weight_kg=patient.get("weight_kg"),
        height_cm=patient.get("height_cm"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 healing-phase macro profile.")
    ap.add_argument("--build", action="store_true", help="Write the xlsx and csv.")
    ap.add_argument("--xlsx", default="Phase1-Macro-Profile.xlsx")
    ap.add_argument("--csv", default="phase1_macro_profile.csv")
    ap.add_argument("--resolve", metavar="TIMELINE", help="Resolve a macro target for a timeline.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.resolve:
        result = _resolve_from_timeline(args.resolve)
        print(json.dumps(result, indent=2))
        return

    rows = build_grid()
    if args.build:
        export_csv(rows, args.csv)
        export_workbook(rows, args.xlsx)
        print(f"{len(rows)} rows")
        print(f"  {args.csv}")
        print(f"  {args.xlsx}")
    else:
        print(f"{len(rows)} profile rows (use --build to write files)")
        print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
