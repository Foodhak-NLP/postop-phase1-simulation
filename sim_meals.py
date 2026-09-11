"""Intake as a consequence of meals logged, not as a percentage.

Phase 2 scores adherence — did the patient eat what was prescribed. Phase 1 has
no prescription to adhere to: section 6.2 scores the previous 24 hours of intake
against a phase target, and the input is whatever the dietitian wrote down. So
the control here is **meals logged**, and everything else follows from it.

Three consequences, all of them the point:

  * fewer meals logged means a smaller intake, which Layer 3 scores as a gap and
    the NSS falls;
  * a day with nothing logged has no `nutrition` block at all, so Layer 3 leaves
    it unscored and the NSS is *unknown* — which is a different statement from
    "the patient ate nothing";
  * before oral intake starts there is nothing to log, and that is not a failure
    of adherence. It is the surgery.

Meal shares are `MEAL_SPLITS` from `populate_phase1_pool.py`, so the split the
recipe pool generates against and the split the intake is built from cannot
drift apart.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import sim_paths  # noqa: F401
from layer3_nutrition_gap import NUTRIENT_TARGETS
from populate_phase1_pool import (HAEMOSTASIS_MEAL_SPLITS, MEAL_SPLITS)

MEALS = ("breakfast", "lunch", "dinner")

# The nutrients section 6.1 scores. Micronutrients ride on the meals like the
# macros do — a patient who logs one meal did not get a day's vitamin C from it.
NUTRIENTS = ("protein_g", "calories_kcal", "vitamin_c_mg", "zinc_mg",
             "arginine_g", "vitamin_a_iu", "fluid_ml")

# The denominator. Proliferation is the most demanding phase for protein,
# calories and vitamin C, so a full three meals against it is a genuine ceiling
# rather than one that moves with the phase — and a run stays comparable with
# itself when the phase changes underneath it.
REFERENCE_PHASE = "PROLIFERATION"

CARBOHYDRATE_SHARE = 0.42        # of calories; section 6.3's interaction check
KCAL_PER_G_CARBOHYDRATE = 4.0


def reference_target(nutrient: str, weight_kg: float) -> Optional[float]:
    """A full day's target for one nutrient — what three meals would deliver."""
    spec = NUTRIENT_TARGETS[nutrient]
    band = spec.get(REFERENCE_PHASE)
    if band is None:
        return None
    lower = float(band[0])
    return round(lower * weight_kg, 1) if spec["per_kg"] else round(lower, 1)


def reference_targets(weight_kg: float) -> Dict[str, Optional[float]]:
    return {n: reference_target(n, weight_kg) for n in NUTRIENTS}


def meal_shares(provisioned_only: bool = False) -> Dict[str, float]:
    """20/40/40, or equal thirds while the patient is on provisioned intake."""
    return dict(HAEMOSTASIS_MEAL_SPLITS if provisioned_only else MEAL_SPLITS)


def logged_share(meals: List[str], provisioned_only: bool = False) -> float:
    """How much of a day's target the meals actually logged add up to."""
    shares = meal_shares(provisioned_only)
    return sum(shares.get(meal, 0.0) for meal in meals)


def meals_for_day(rng: np.random.Generator, mean_meals: float,
                  jitter: float = 0.0) -> List[str]:
    """Which meals the dietitian logged today.

    Breakfast is dropped first and dinner last. That is not arbitrary: a
    post-operative patient with poor appetite skips the morning meal, and the
    evening meal is the one a ward round is most likely to have recorded. It
    also means the share lost per missed meal is the smallest one first, which
    is the conservative direction for a demo.
    """
    target = float(np.clip(mean_meals, 0.0, 3.0))
    # Variance vanishes at both ends. Asking for none must give none — a spread
    # that can still produce a meal makes "nothing logged" untestable and hides
    # the unlogged-versus-zero distinction this module exists to preserve. The
    # same holds at three: a fully logged patient is fully logged. `jitter`
    # scales that envelope, so a pattern can be steady or genuinely erratic.
    spread = jitter * min(target, 3.0 - target) / 1.5
    count = int(np.clip(round(rng.normal(target, spread)) if spread else
                        round(target), 0, 3))
    # Keep dinner, then lunch, then breakfast.
    return list(MEALS[3 - count:]) if count else []


def intake(weight_kg: float, meals: List[str],
           provisioned_only: bool = False) -> Optional[Dict[str, Any]]:
    """One day's `nutrition` block, or None when nothing was logged.

    None rather than a block of zeros, deliberately. Section 6.2 treats an
    absent nutrient as unknown and a zero as a 100% gap, and those are different
    findings. A day the dietitian did not write up is the first, not the second.
    """
    if not meals:
        return None
    share = logged_share(meals, provisioned_only)
    log: Dict[str, Any] = {}
    for nutrient in NUTRIENTS:
        target = reference_target(nutrient, weight_kg)
        if target is None:
            continue
        log[nutrient] = round(target * share, 1)
    if "calories_kcal" in log:
        log["carbohydrate_g"] = round(
            log["calories_kcal"] * CARBOHYDRATE_SHARE / KCAL_PER_G_CARBOHYDRATE, 1)
    return log


def describe(meals: List[str], provisioned_only: bool = False) -> str:
    if not meals:
        return "nothing logged"
    share = logged_share(meals, provisioned_only)
    return (f"{len(meals)} of 3 — {', '.join(m[:1].upper() + m[1:] for m in meals)}"
            f" ({share:.0%} of the day's target)")
