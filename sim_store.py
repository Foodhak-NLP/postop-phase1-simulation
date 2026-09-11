"""Collect a whole stay — every day's insights, recommendations and recipes.

The app shows one day at a time. This assembles all of them into one record so
the stay can be exported, diffed, or handed to someone who was not at the
screen-share.

Recipes come from `recipe_pool` through the same `select_phase1_recipes` call
the orchestrator makes. They depend only on the healing phase and the patient's
demographic band, so a six-week stay needs at most four queries rather than
forty-two — the phase is the cache key.

**Nothing here writes to the RDS or to LangGraph.** It reads the pool and keeps
the result locally. Writing recommendations back into the staging store is a
side effect on someone else's data and needs to be asked for explicitly, not
slipped into an export.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, List, Optional

import sim_insights
import sim_paths  # noqa: F401
import sim_recipes


def recipes_by_phase(arc: Dict[str, Any], *, age: float, sex: str,
                     height_cm: float, weight_kg: float,
                     diets: Optional[List[str]] = None,
                     allergens: Optional[List[str]] = None,
                     recipes_per_meal: int = 3,
                     user_id: str = "phase1-sim",
                     date: Optional[str] = None) -> Dict[str, Any]:
    """One pool query per distinct healing phase in the stay."""
    date = date or dt.date.today().isoformat()
    out: Dict[str, Any] = {}
    for record in arc["days"]:
        phase = record["phase"]
        if phase in out:
            continue
        out[phase] = sim_recipes.select(
            bundle=record["bundle"], age=age, sex=sex, height_cm=height_cm,
            weight_kg=weight_kg, diets=list(diets or []),
            allergens=list(allergens or []), user_id=user_id, date=date,
            recipes_per_meal=recipes_per_meal)
    return out


def collect(arc: Dict[str, Any], *, date_of: Optional[Callable[[int], dt.date]] = None,
            recipes: Optional[Dict[str, Any]] = None,
            patient: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole stay: one entry per day, plus what produced it."""
    recipes = recipes or {}
    days: List[Dict[str, Any]] = []

    for record in arc["days"]:
        bundle = record["bundle"]
        view = sim_insights.overall(bundle, meals_logged=record["meals_logged"])
        selection = recipes.get(record["phase"]) or {}
        by_meal = sim_recipes.by_meal(selection) if selection else {}

        days.append({
            "day": record["day"],
            "date": (date_of(record["day"]).isoformat() if date_of else None),
            "phase": record["phase"],
            "phase_confidence": record["confidence"],
            "healing_status": record["healing_status"],
            "nss": record["nss"],
            "meals_logged": record["meals_logged"],
            "deviations": record["deviations"],
            "checks_not_assessed": record["not_assessed"],
            "escalate_to": record["escalate_to"],
            "phase2_unlocked": record["phase2_unlocked"],
            "gate_blocking": record["gate_blocking"],
            "insights": {name: part["insight"]
                         for name, part in view["layers"].items()},
            "recommendations": view["recommendation"],
            "layer4_narrative": bundle["layer4_alert"].get("narrative"),
            "layer4_source": bundle["layer4_alert"].get("source"),
            "recipes": {
                "status": selection.get("status"),
                "healing_phase": selection.get("healing_phase"),
                "reason": selection.get("reason"),
                # By value, so the export stands alone when the pool is not
                # reachable from wherever it is read next.
                "meals": {meal: [{"title": r.get("food_title"),
                                  "recipe_id": r.get("recipe_id"),
                                  "why": r.get("why_this_works_for_you"),
                                  "recipe": r}
                                 for r in items]
                          for meal, items in by_meal.items()},
            },
        })

    return {
        "schema_version": "phase1_simulation_stay_v1",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "patient": patient or {},
        "summary": {
            "days": len(days),
            "phase_first_seen": arc["phase_first_seen"],
            "first_flagged": arc["first_flagged"],
            "gate_opened_on": arc["gate_opened_on"],
            "gate_stayed_open": arc["gate_stayed_open"],
        },
        "recipe_sources": {phase: {"status": sel.get("status"),
                                   "reason": sel.get("reason"),
                                   "returned": (sel.get("availability") or {})
                                   .get("available_total", 0)}
                           for phase, sel in recipes.items()},
        "days": days,
    }
