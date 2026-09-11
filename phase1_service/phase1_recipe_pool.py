"""
Phase 1 recipe selection from the pre-generated recipe pool.

Flow, when the Phase 1 route is hit:

    LangGraph state  ->  user demographics + restrictions
    Phase 1 output   ->  healing phase + NSS escalation
                          |
                          v
              recipe_pool  (pre-generated)
                          |
                          v
              9 recipes (3 meals x 3 variants)

This mirrors what postop_daily_recommendation_service_v1 does for Phase 2, with
two differences:

  * the pool partition is keyed on the HEALING PHASE rather than the CQL action
    space, and on macro_profile_version 'phase1_v1';
  * a shortage is never fatal. Phase 1's job is the clinical assessment and the
    physician alert. If the pool is thin, the caller still gets the assessment,
    with the shortage stated explicitly.

NO SCHEMA CHANGE IS REQUIRED. recipe_pool's query already filters on
`action_space` and `macro_profile_version`; Phase 1 rows are tagged with the
healing phase in `action_space` and 'phase1_v1' as the profile version, so they
partition cleanly away from Phase 2 rows. If a dedicated `healing_phase` column
is added later, only POOL_PARTITION_COLUMN below needs to change.
"""

from __future__ import annotations

import logging
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MACRO_PROFILE_VERSION = "phase1_v1"

# breakfast/lunch/dinner x 3 variants = the 9 recipes per day.
MEALS = ("breakfast", "lunch", "dinner")
DEFAULT_RECIPES_PER_MEAL = 3

# Phase 1 phases that have a recipe partition. Haemostasis has no nutrition
# target (Layer 3 does not score a 0-3 h phase), so it has no recipes either.
# HAEMOSTASIS is provisioned but not scored: ESPEN and ERAS both direct that
# oral intake start within hours of surgery, so day 0 gets clear fluids, while
# Layer 3 still returns not_scored because no evidence sets a day-0 target.
POOL_PHASES = ("HAEMOSTASIS", "INFLAMMATION", "PROLIFERATION", "REMODELLING")

# Diet and allergen tags must match the recipe_pool vocabulary EXACTLY (the SQL
# compares lower(trim(...)) with no alias expansion). The pool stores canonical
# Title Case forms — "Gluten Free", "Cashew Nuts" — so normalising to snake_case
# here would match nothing and silently return an empty pool. The canonical
# tables live in recipe_presentation, Phase 1's own copy of them.


class Phase1RecipePoolError(RuntimeError):
    """Raised only for configuration faults, never for an empty pool."""


# ---------------------------------------------------------------------------
# Repository access
# ---------------------------------------------------------------------------

try:                                    # staging defaults, env still wins
    from phase1_config import apply_defaults
    apply_defaults()
except Exception:                       # standalone use without the config file
    pass

LANGGRAPH_BASE_URL = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:8000")
LANGGRAPH_STATE_PATH = "/langgraph/users/{user_id}/state/{date}"


def fetch_langgraph_state(user_id: str, date: str,
                          timeout_sec: int = 30) -> Dict[str, Any]:
    """Read one date's LangGraph state, the way Phase 2's service does.

    Recipe selection needs the patient's body and dietary restrictions, and
    those live in LangGraph, not in the clinical payload. Phase 2's
    daily-recommendation service solves this by fetching state over HTTP and
    keeping pool access on its own side; Phase 1 now does the same, which is
    why the LangGraph app needs no recipe-pool configuration at all.

    Never raises: a state read that fails must not cost the physician their
    clinical assessment.
    """
    # bootstrap_if_missing is the documented contract for external services:
    # without it the route answers 404 whenever the thread has no base profile
    # yet, which is precisely the case on a patient's first Phase 1 call of the
    # day — and a 404 there means zero recipes with no visible cause.
    url = (LANGGRAPH_BASE_URL.rstrip("/")
           + LANGGRAPH_STATE_PATH.format(user_id=user_id, date=date)
           + "?bootstrap_if_missing=true")
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body if isinstance(body, dict) else {}
    except Exception as exc:
        logger.warning("[PHASE1 POOL] could not read state for %s @ %s: %s",
                       user_id, date, exc)
        return {}


def _load_repository():
    """Phase 1's own recipe_pool_repository.

    This used to import the daily-recommendation service's copy via
    RECIPE_POOL_MODULE_PATH — an absolute path into a sibling checkout. That
    works on a developer laptop and cannot work in a container, where the
    Dockerfile's `COPY . .` brings in this service's files and nothing else.
    The failure would also have been quiet: recipe selection is non-fatal here,
    so every patient would have received `recipes: unavailable` with the real
    cause visible only in a warning log.

    The module is now vendored beside this one. test_vendored_parity.py fails
    if it drifts from the daily-recommendation service's copy while both are on
    disk, so divergence is caught in CI rather than in production.
    """
    try:
        import recipe_pool_repository  # type: ignore
        return recipe_pool_repository
    except ImportError as exc:                       # pragma: no cover
        raise Phase1RecipePoolError(
            "recipe_pool_repository.py is missing from the Phase 1 service "
            f"directory. Underlying error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 1. User details, from LangGraph state
# ---------------------------------------------------------------------------

def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_gender(value: Any) -> Optional[str]:
    text = str(value or "").strip().casefold()
    if text in {"m", "male"}:
        return "male"
    if text in {"f", "female"}:
        return "female"
    return None


# Corrections applied on top of Phase 2's alias tables.
#
# recipe_pool tags dairy-containing recipes "Dairy" and never "Milk", and the pool
# query compares lower(trim(...)) with no alias expansion. Mapping "milk" to
# "Milk" therefore matches nothing and leaves every dairy recipe eligible for a
# milk-allergic patient — an allergen exclusion failing open.
#
# The upstream Phase 2 table has since been corrected, so this overlay is now a
# defence in depth rather than a live workaround: it keeps Phase 1 safe if the
# shared table regresses or if the fallback tables are used. Entries here should
# only ever map a user tag onto a spelling recipe_pool actually stores.
_POOL_VOCABULARY_CORRECTIONS = {
    "milk": "Dairy",
}


def _alias_tables() -> Tuple[Dict[str, str], Dict[str, str]]:
    """The canonical diet and allergen spellings, plus Phase 1's overlay."""
    from recipe_presentation import _ALLERGEN_ALIASES, _DIET_ALIASES
    return _DIET_ALIASES, {**_ALLERGEN_ALIASES, **_POOL_VOCABULARY_CORRECTIONS}


def _normalise_tags(values: Any, aliases: Dict[str, str]) -> List[str]:
    """Map to the pool's canonical spelling, keeping unknown tags verbatim.

    An unrecognised tag is passed through rather than dropped: dropping it would
    quietly relax a dietary restriction, which is the unsafe direction.
    """
    if not isinstance(values, (list, tuple, set)):
        values = [values] if values else []
    out: List[str] = []
    seen = set()
    for raw in values:
        if isinstance(raw, dict):
            raw = raw.get("name") or raw.get("title") or raw.get("type")
        text = str(raw or "").strip()
        if not text:
            continue
        canonical = aliases.get(text.casefold(), text)
        key = canonical.casefold()
        if key not in seen:
            seen.add(key)
            out.append(canonical)
    return out


def resolve_pool_profile(
    state: Dict[str, Any],
    resolved_weight_kg: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Pull the recipe-pool filters out of LangGraph state.

    Returns (profile, missing_fields). Unlike the Phase 2 equivalent this does
    NOT raise on missing fields — Phase 1's clinical output must still be
    delivered when a profile is incomplete. The caller decides what to do with
    the missing list.

    Reads the same `user_profile` keys Phase 2 reads, so one profile serves both.

    `resolved_weight_kg` is the body LangGraph already decided on for this run
    — manual tracker first, then the profile. It is passed in rather than
    re-derived here so the weight BAND this function picks cannot disagree with
    the weight Layer 3 scaled its per-kg targets from. Deriving it twice is how
    a 55 kg patient was once scored at 55 kg and handed recipes generated at
    78 kg. Falls back to the profile when running standalone, where nothing
    upstream has resolved anything.
    """
    profile = state.get("user_profile") or state.get("profile") or {}
    if not isinstance(profile, dict):
        profile = {}

    age = _coerce_float(profile.get("User Age"))
    height_cm = _coerce_float(profile.get("User Height"))
    weight_kg = (_coerce_float(resolved_weight_kg)
                 if resolved_weight_kg is not None
                 else _coerce_float(profile.get("User Weight")))
    gender = _normalise_gender(profile.get("User Sex") or profile.get("User Gender"))
    diet_aliases, allergen_aliases = _alias_tables()

    missing = [
        name for name, value in (
            ("User Age", age), ("User Sex", gender),
            ("User Height", height_cm), ("User Weight", weight_kg),
        ) if value is None
    ]

    resolved = {
        "age": age,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "dietary_restrictions": _normalise_tags(
            profile.get("User Dietary Restrictions") or [], diet_aliases),
        "allergens": _normalise_tags(
            profile.get("User Allergens") or [], allergen_aliases),
    }
    return resolved, missing


# ---------------------------------------------------------------------------
# 2. Pool partition, from the Phase 1 output
# ---------------------------------------------------------------------------

def resolve_pool_partition(phase1_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Which slice of the pool this patient's recipes come from today."""
    summary = phase1_bundle.get("summary") if isinstance(phase1_bundle, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    layer3 = phase1_bundle.get("layer3_nutrition") if isinstance(phase1_bundle, dict) else {}
    layer3 = layer3 if isinstance(layer3, dict) else {}

    phase = str(summary.get("current_phase") or "").upper() or None
    escalation = (layer3.get("nss_action") or {}).get("escalation") if isinstance(
        layer3.get("nss_action"), dict) else None

    return {
        "healing_phase": phase,
        "macro_profile_version": MACRO_PROFILE_VERSION,
        "nss_escalation": escalation or "none",
        "eligible": phase in POOL_PHASES,
        "reason": None if phase in POOL_PHASES else (
            f"{phase} has no recipe partition"
            if phase else "healing phase missing from Phase 1 output"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Fetch
# ---------------------------------------------------------------------------

def _group_by_meal(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {meal: [] for meal in MEALS}
    for row in rows:
        meal = str(row.get("meal_type") or "").strip().casefold()
        if meal in grouped:
            grouped[meal].append(row)
    return grouped


def _recipe_shaping():
    """Key-ordering and JSON decoding, so both phases store one recipe shape."""
    from recipe_presentation import _decode_recipe, order_recipe_keys
    return order_recipe_keys, _decode_recipe


def build_recommended_recipes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Shape the pooled rows the way Phase 2 stores them in LangGraph.

    Phase 2's convention is `{meal: {"recommended_meals": [envelope]}}` where an
    envelope is `{action_space, macro_profile_version, recipe}` — the full recipe
    by value, key-ordered for display. Matching it means an app rendering a
    patient's day uses ONE code path whether they are in Phase 1 or Phase 2,
    which is the whole point of mirroring rather than inventing a second shape.

    Height and weight are deliberately absent: Phase 2 omits them from the
    LangGraph envelope too (they live in user_profile and the selection block)
    and only adds them to the custom-meal-plan API payload.
    """
    order_recipe_keys, decode_recipe = _recipe_shaping()
    grouped: Dict[str, Any] = {meal: {"recommended_meals": []} for meal in MEALS}
    for row in rows:
        meal = str(row.get("meal_type") or "").strip().casefold()
        if meal not in grouped:
            continue
        grouped[meal]["recommended_meals"].append({
            "action_space": str(row.get("action_space") or ""),
            "macro_profile_version": str(row.get("macro_profile_version") or ""),
            "recipe": order_recipe_keys(decode_recipe(row.get("recipe"))),
        })
    return grouped


def summarise_availability(
    rows: List[Dict[str, Any]], recipes_per_meal: int
) -> Dict[str, Any]:
    """Report a shortage; never reject. Mirrors Phase 2's _validate_pool_rows."""
    grouped = _group_by_meal(rows)
    counts = {meal: len(items) for meal, items in grouped.items()}
    shortages = {
        meal: {"required": recipes_per_meal, "available": count,
               "missing": max(0, recipes_per_meal - count)}
        for meal, count in counts.items() if count < recipes_per_meal
    }
    required_total = recipes_per_meal * len(MEALS)
    available_total = sum(counts.values())
    return {
        "complete": not shortages,
        "meal_counts": counts,
        "shortages": shortages,
        "required_total": required_total,
        "available_total": available_total,
        "missing_total": max(0, required_total - available_total),
    }


def select_phase1_recipes(
    *,
    state: Dict[str, Any],
    phase1_bundle: Dict[str, Any],
    user_id: str,
    date: str,
    recipes_per_meal: int = DEFAULT_RECIPES_PER_MEAL,
    excluded_recipe_ids: Optional[List[str]] = None,
    repeat_after_days: int = 15,
) -> Dict[str, Any]:
    """Select the 9 recipes for one patient-day.

    Never raises for a data reason. Every failure mode — incomplete profile,
    ineligible phase, empty pool, repository misconfiguration — comes back as a
    structured result with `status` explaining why, so the Phase 1 clinical
    output is delivered regardless.
    """
    partition = resolve_pool_partition(phase1_bundle)
    # The orchestrator echoes back the patient block LangGraph handed it, and
    # LangGraph has already written the resolved body weight into it. Take that
    # number rather than reading the profile again — see resolve_pool_profile.
    bundle_patient = (phase1_bundle.get("patient")
                      if isinstance(phase1_bundle, dict) else None) or {}
    profile, missing = resolve_pool_profile(
        state, _coerce_float(bundle_patient.get("weight_kg")))

    base: Dict[str, Any] = {
        "schema_version": "phase1_recipe_selection_v2_recipe_pool",
        "healing_phase": partition["healing_phase"],
        "macro_profile_version": partition["macro_profile_version"],
        "nss_escalation": partition["nss_escalation"],
        "recipes_per_meal": recipes_per_meal,
        "meals": list(MEALS),
        "requested_total": recipes_per_meal * len(MEALS),
    }

    if not partition["eligible"]:
        return {**base, "status": "skipped", "reason": partition["reason"],
                "days": [], "availability": None}

    if missing:
        return {**base, "status": "profile_incomplete",
                "reason": "LangGraph user_profile is missing: " + ", ".join(missing),
                "missing_profile_fields": missing, "days": [], "availability": None}

    try:
        repo = _load_repository()
    except Phase1RecipePoolError as exc:
        logger.warning("[PHASE1 POOL] repository unavailable: %s", exc)
        return {**base, "status": "unavailable", "reason": str(exc),
                "days": [], "availability": None}

    def _fetch(excluded: List[str]):
        return repo.fetch_daily_recipe_rows(
            # Phase 1 rows are tagged with the healing phase in action_space and
            # partitioned away from Phase 2 by macro_profile_version.
            action_space=partition["healing_phase"],
            macro_profile_version=partition["macro_profile_version"],
            gender=profile["gender"],
            age=profile["age"],
            weight_kg=profile["weight_kg"],
            height_cm=profile["height_cm"],
            meal_types=list(MEALS),
            required_dietary_restrictions=profile["dietary_restrictions"],
            excluded_allergens=profile["allergens"],
            excluded_recipe_ids=sorted(excluded),
            user_id=user_id,
            recommendation_date=date,
            recipes_per_meal=recipes_per_meal,
        )

    excluded = sorted(excluded_recipe_ids or [])
    repeat_relaxed = False
    try:
        rows = _fetch(excluded) or []
        # Variety must never cost the patient a meal. The 15-day no-repeat rule
        # is a quality preference; being handed nothing to eat is a clinical
        # failure. On a 6-week stay the pool is exhausted long before the window
        # closes — proliferation holds 3 dinners against 15 days of exclusions —
        # so when the rule starves a meal, the rule yields, and says that it did.
        if excluded and not summarise_availability(rows, recipes_per_meal)["complete"]:
            relaxed = _fetch([]) or []
            if len(relaxed) > len(rows):
                logger.info(
                    "[PHASE1 POOL] %s @ %s | 15-day no-repeat left %d/%d recipes; "
                    "relaxed to %d. The pool is too thin for the exclusion window.",
                    user_id, date, len(rows), recipes_per_meal * len(MEALS), len(relaxed))
                rows, repeat_relaxed = relaxed, True
    except Exception as exc:  # repository raises its own error type
        logger.warning("[PHASE1 POOL] fetch failed for %s @ %s: %s", user_id, date, exc)
        return {**base, "status": "fetch_failed", "reason": str(exc),
                "days": [], "availability": None}
    availability = summarise_availability(rows, recipes_per_meal)

    logger.info(
        "[PHASE1 POOL] %s @ %s | phase=%s | %s/%s recipes | complete=%s",
        user_id, date, partition["healing_phase"],
        availability["available_total"], availability["required_total"],
        availability["complete"],
    )

    return {
        **base,
        "status": "ok" if availability["complete"] else "partial",
        "profile_used": {
            "gender": profile["gender"], "age": profile["age"],
            "weight_kg": profile["weight_kg"], "height_cm": profile["height_cm"],
            "dietary_restrictions": profile["dietary_restrictions"],
            "allergens": profile["allergens"],
        },
        # Phase 2's shape. `days` holds a single entry because Phase 1 runs once
        # per patient-day, but keeping the array means a consumer can walk
        # days[].recommended_recipes identically for either phase.
        "source": "ai_rds_recipe_pool",
        "date": date,
        "summary": ("Approved profile-compatible recipes selected from the AI-RDS "
                    "recipe pool for the current healing phase."),
        "days": [{
            "day": 1,
            "date": date,
            "healing_phase": partition["healing_phase"],
            "recommended_recipes": build_recommended_recipes(rows),
        }],
        "selection": {
            "source": "ai_rds_recipe_pool",
            "selected_count": len(rows),
            "selected_recipe_ids": [str(r.get("recipe_id")) for r in rows
                                    if r.get("recipe_id")],
        },
        "recipe_ids": [str(r.get("recipe_id")) for r in rows if r.get("recipe_id")],
        "availability": availability,
        # Says plainly when a patient is seeing a dish again inside the 15-day
        # window, and why — a thin pool, not a scheduling accident.
        "repeat_window_days": repeat_after_days,
        "recently_served_excluded": len(excluded),
        "repeat_window_relaxed": repeat_relaxed,
    }
