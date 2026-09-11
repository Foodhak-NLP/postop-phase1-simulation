"""The day's nine recipes, from the live pool.

`select_phase1_recipes` reads a LangGraph state document for the patient's body
and restrictions. There is no LangGraph here, so this builds that document from
the sidebar and calls the same function the orchestrator calls — the query, the
phase partition and the allergen exclusion are all the service's.

There is deliberately no offline substitute. The pool holds approved recipes;
composing a stand-in when it is unreachable would put unapproved food next to
approved food with nothing on screen to tell them apart. When the pool cannot
answer, the status it returned is what the tab shows.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sim_paths  # noqa: F401
from phase1_recipe_pool import MEALS, select_phase1_recipes

# The vocabularies recipe_presentation aliases into pool tags. Anything outside
# these lists is passed through unchanged and will simply match nothing.
DIETS = ("Gluten Free", "Pescetarian", "Ketogenic", "Low FODMAP",
         "Vegetarian", "Vegan", "Paleo")
ALLERGENS = ("Dairy", "Eggs", "Fish", "Gluten", "Peanuts", "Nut", "Almond",
             "Cashew Nuts", "Hazelnuts", "Walnuts", "Sesame", "Soy", "Celery",
             "Mustard", "Crustaceans", "Sulphites", "Oats", "Salmon")


def langgraph_state(*, age: float, sex: str, height_cm: float, weight_kg: float,
                    diets: List[str], allergens: List[str]) -> Dict[str, Any]:
    """The `user_profile` keys `resolve_pool_profile` reads. Same keys as Phase 2."""
    return {
        "user_profile": {
            "User Age": age,
            "User Sex": sex,
            "User Height": height_cm,
            "User Weight": weight_kg,
            "User Dietary Restrictions": list(diets),
            "User Allergens": list(allergens),
        }
    }


CACHE_DIR = Path(__file__).resolve().parent / ".recipe_cache"


def _cache_key(*, phase: Optional[str], age: float, sex: str, height_cm: float,
               weight_kg: float, diets: List[str], allergens: List[str],
               recipes_per_meal: int) -> str:
    """What actually changes the answer. The date and the user id do not: the
    pool query is keyed on the healing phase and the demographic band."""
    raw = json.dumps({
        "phase": phase, "age": age, "sex": sex, "height_cm": height_cm,
        "weight_kg": weight_kg, "diets": sorted(diets),
        "allergens": sorted(allergens), "per_meal": recipes_per_meal,
    }, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cached(key: str) -> Optional[Dict[str, Any]]:
    path = CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:                                   # noqa: BLE001
        return None


def store(key: str, selection: Dict[str, Any]) -> None:
    """Keep what the pool returned, so a demo survives losing the database."""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(json.dumps({
            "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
            "selection": selection,
        }, indent=1))
    except Exception:                                   # noqa: BLE001
        pass                                            # a cache is a courtesy


def cache_contents() -> List[Dict[str, Any]]:
    """What is on disk, for the tab to report."""
    if not CACHE_DIR.is_dir():
        return []
    out = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        try:
            blob = json.loads(path.read_text())
        except Exception:                               # noqa: BLE001
            continue
        selection = blob.get("selection") or {}
        if not (selection.get("availability") or {}).get("available_total"):
            continue
        out.append({
            "phase": selection.get("healing_phase"),
            "fetched_at": blob.get("fetched_at"),
            "recipes": (selection.get("availability") or {})
            .get("available_total", 0),
            "key": path.stem,
        })
    return out


def select(*, bundle: Dict[str, Any], age: float, sex: str, height_cm: float,
           weight_kg: float, diets: List[str], allergens: List[str],
           user_id: str, date: str, recipes_per_meal: int = 3,
           allow_cache: bool = True) -> Dict[str, Any]:
    """Never raises. Every failure comes back as a `status` the tab can show.

    A live answer is stored on the way out; an unreachable database falls back
    to the last stored one for the same phase and band, labelled as such.
    """
    phase = (bundle.get("summary") or {}).get("current_phase")
    key = _cache_key(phase=phase, age=age, sex=sex, height_cm=height_cm,
                     weight_kg=weight_kg, diets=diets, allergens=allergens,
                     recipes_per_meal=int(recipes_per_meal))
    state = langgraph_state(age=age, sex=sex, height_cm=height_cm,
                            weight_kg=weight_kg, diets=diets, allergens=allergens)
    try:
        selection = select_phase1_recipes(
            state=state, phase1_bundle=bundle, user_id=user_id, date=date,
            recipes_per_meal=int(recipes_per_meal))
    except Exception as exc:                            # noqa: BLE001
        selection = {"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                     "days": [], "availability": None, "healing_phase": phase}

    returned = (selection.get("availability") or {}).get("available_total", 0)
    if selection.get("status") in ("ok", "partial"):
        # Only keep an answer that has something in it. Caching an empty
        # partial would later be served as "cached" with no recipes, which
        # reads as "the pool has none for this band" when the truth is that
        # nobody has asked the pool since it came back.
        if returned:
            store(key, selection)
        return {**selection, "served_from": "recipe_pool", "cache_key": key}

    if allow_cache:
        blob = cached(key)
        if blob and blob.get("selection"):
            return {**blob["selection"], "status": "cached",
                    "served_from": "local cache",
                    "fetched_at": blob.get("fetched_at"),
                    "live_reason": selection.get("reason"),
                    "cache_key": key}
    return {**selection, "served_from": "nothing", "cache_key": key}


def by_meal(selection: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten `days[0].recommended_recipes` into {meal: [recipe, ...]}."""
    days = selection.get("days") or []
    recommended = (days[0].get("recommended_recipes") if days else {}) or {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for meal in MEALS:
        envelopes = (recommended.get(meal) or {}).get("recommended_meals") or []
        out[meal] = [e.get("recipe") or {} for e in envelopes]
    return out


NUMERIC_FIELDS = ("Calories", "Protein", "Carbohydrates", "Total Fat",
                  "Saturated Fat", "Sodium", "Sugar", "Vitamin C", "Vitamin D",
                  "Vitamin E", "Iron", "Calcium", "Potassium", "Cholesterol")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "amount", "quantity"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
    if isinstance(value, str):
        cleaned = "".join(c for c in value if c.isdigit() or c in ".-")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def macro_row(recipe: Dict[str, Any], meal: str) -> Dict[str, Any]:
    """One recipe as a table row — title, meal, and whatever macros it carries."""
    row: Dict[str, Any] = {
        "Meal": meal.title(),
        "Recipe": recipe.get("food_title") or "(untitled)",
        "Servings": recipe.get("servings"),
        "Ready in": recipe.get("ready_in_minutes"),
    }
    for field in NUMERIC_FIELDS:
        value = _number(recipe.get(field))
        if value is not None:
            row[field] = round(value, 1)
    return row


# ---------------------------------------------------------------------------
# Pool coverage
# ---------------------------------------------------------------------------
# The Phase 1 pool is generated per demographic band, and only some bands have
# been generated so far. Without this, a patient in an ungenerated band gets an
# empty tab and no way to tell an empty band from a broken connection.

def pool_coverage() -> Dict[str, Any]:
    """Which (phase, sex, age, weight, height) bands actually hold recipes."""
    import os

    from recipe_pool_repository import (_postgres_connection_settings,
                                        _validated_table_name)
    try:
        dsn, settings = _postgres_connection_settings()
        import psycopg
        from psycopg.rows import dict_row

        table = _validated_table_name()
        timeout = int(os.getenv("RECIPE_POOL_DB_CONNECT_TIMEOUT_SEC", "15"))
        connect = (psycopg.connect(dsn, connect_timeout=timeout, row_factory=dict_row)
                   if dsn else
                   psycopg.connect(**settings, connect_timeout=timeout, row_factory=dict_row))
        with connect as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT action_space, gender, age_band, weight_band_kg, "  # noqa: S608
                f"height_band_cm, meal_type, COUNT(*) AS n FROM {table} "
                f"WHERE macro_profile_version = %s "
                f"GROUP BY 1, 2, 3, 4, 5, 6 ORDER BY 1, 2, 3, 4, 5, 6",
                ("phase1_v1",),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return {"status": "ok", "rows": rows}
    except Exception as exc:                            # noqa: BLE001
        return {"status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}", "rows": []}
