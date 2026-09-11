"""
Recipe presentation helpers — Phase 1's own copy.

Phase 1 stores its 9 recipes in the same envelope shape Phase 2 uses, so it
needs the same key ordering, the same JSON decoding, and the same diet/allergen
alias tables. These are COPIED from postop_daily_recommendation_service_v1
rather than imported across repositories.

Why copied: importing meant RECIPE_POOL_MODULE_PATH pointing at a sibling
checkout. That works on a laptop and fails in a container, where only this
service's own files are present — and it fails silently, because Phase 1 never
lets a recipe problem break the clinical output. Every patient would have got
`recipes: unavailable` with no error anywhere.

Keep in sync with the daily-recommendation service; test_vendored_parity.py
fails if the two disagree while both are on disk, so drift is caught in CI
rather than by a patient.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict

logger = logging.getLogger(__name__)


RECIPE_DISPLAY_KEY_ORDER = [
    "summary",
    "food_title",
    "mealType",
    "servings",
    "ready_in_minutes",
    "ingredients",
    "Calories",
    "Protein",
    "Carbohydrates",
    "Total Fat",
    "Sodium",
    "Saturated Fat",
    "Cholesterol",
    "Sugar",
    "Calcium",
    "Iron",
    "Potassium",
    "Vitamin C",
    "Vitamin E",
    "Vitamin D",
    "cautions",
    "diet_labels",
    "instructions",
    "main_ingredients",
    "short_description",
    "long_description",
    "vitamin_mineral_claims",
    "why_this_works_for_you",
    "scientific_evidence_support",
]


_DIET_ALIASES = {
    "gluten free": "Gluten Free",
    "gluten-free": "Gluten Free",
    "pescetarian": "Pescetarian",
    "pescatarian": "Pescetarian",
    "ketogenic": "Ketogenic",
    "keto": "Ketogenic",
    "low fodmap": "Low FODMAP",
    "low-fodmap": "Low FODMAP",
    "vegetarian": "Vegetarian",
    "vegan": "Vegan",
    "paleo": "Paleo",
}


_ALLERGEN_ALIASES = {
    "almond": "Almond",
    "almonds": "Almond",
    "bass": "Bass",
    "cashew": "Cashew Nuts",
    "cashews": "Cashew Nuts",
    "cashew nuts": "Cashew Nuts",
    "celery": "Celery",
    "cereal": "Cereals",
    "cereals": "Cereals",
    "crustacean": "Crustaceans",
    "crustaceans": "Crustaceans",
    "dairy": "Dairy",
    "egg": "Eggs",
    "eggs": "Eggs",
    "fish": "Fish",
    "flour": "Flour",
    "gluten": "Gluten",
    "hazelnut": "Hazelnuts",
    "hazelnuts": "Hazelnuts",
    # recipe_pool tags dairy-containing recipes "Dairy" and never "Milk", and the
    # pool query compares lower(trim(...)) with no alias expansion. Mapping "milk"
    # to "Milk" therefore matched nothing and left every dairy recipe eligible for a
    # milk-allergic patient — an allergen exclusion failing open.
    "milk": "Dairy",
    "mustard": "Mustard",
    "nut": "Nut",
    "nuts": "Nut",
    "tree nut": "Nut",
    "tree nuts": "Nut",
    "oat": "Oats",
    "oats": "Oats",
    "peanut": "Peanuts",
    "peanuts": "Peanuts",
    "salmon": "Salmon",
    "sesame": "Sesame",
    "soy": "Soy",
    "soya": "Soya",
    "sulphite": "Sulphites",
    "sulphites": "Sulphites",
    "sulfite": "Sulphites",
    "sulfites": "Sulphites",
    "trout": "Trout",
    "walnut": "Walnuts",
    "walnuts": "Walnuts",
}


def order_recipe_keys(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new recipe dictionary using the preferred response/display order.

    Unknown or newly introduced fields are retained after the known fields.
    This is only for predictable API/LangGraph presentation. Consumers should
    still access JSON fields by key rather than by position.
    """
    if not isinstance(recipe, dict):
        return {}

    ordered: Dict[str, Any] = {}

    for key in RECIPE_DISPLAY_KEY_ORDER:
        if key in recipe:
            ordered[key] = recipe[key]

    for key, value in recipe.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


def _decode_recipe(value: Any) -> Dict[str, Any]:
    """Decode a recipe_pool row's recipe column.

    Deliberately DIVERGES from Phase 2, which raises HTTPException(502) on bad
    JSON. Phase 1's contract is that a recipe problem never costs the physician
    the clinical assessment, so a malformed row is dropped with a warning and
    the remaining recipes are still returned. The decoded dict is deep-copied
    for the same reason Phase 2 copies it: callers reorder and annotate it, and
    the row must not be mutated underneath a shared cache.
    """
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            logger.warning("[PHASE1 POOL] unparseable recipe JSON: %s", exc)
            return {}
        if isinstance(parsed, dict):
            return parsed
    logger.warning("[PHASE1 POOL] recipe column was %s, not an object",
                   type(value).__name__)
    return {}
