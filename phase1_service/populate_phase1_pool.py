"""
Populate recipe_pool with Phase 1 recipes.

Offline job. For a (healing_phase, sex, age_band, weight_band, height_band)
combination it resolves the daily macro target, splits it across the three meals
using the same 20:40:40 rule Phase 2 uses, calls the recipe generator once per
meal per variant, and writes the approved records into recipe_pool.

    phase1_macro_profile + Layer 3 target
             |  20:40:40
             v
    generator :8007  x9
             |
             v
        recipe_pool   (action_space = <HEALING_PHASE>, macro_profile_version = phase1_v1)

Phase 1 rows are tagged with the healing phase in `action_space` and 'phase1_v1'
as the profile version, so they partition cleanly away from Phase 2's rows.

    python3 populate_phase1_pool.py --phase INFLAMMATION --sex Male --age 64 \
        --weight 78 --height 176 --variants 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

import macro_profile_phase1 as MP

MEALS = ("breakfast", "lunch", "dinner")
MEAL_SPLITS = {"breakfast": 0.20, "lunch": 0.40, "dinner": 0.40}

# The generator matches on a range, so a tolerance band is required. +/-10% of
# the meal target is the same width Phase 2's own command examples use.
TOLERANCE = 0.10

GENERATOR_URL = os.getenv("RECIPE_GENERATOR_URL", "http://127.0.0.1:8007/v1/recipes")
MACRO_PROFILE_VERSION = os.getenv("PHASE1_MACRO_PROFILE_VERSION", "phase1_v1")


LANGGRAPH_BASE_URL = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:8000")

# Haemostasis is PROVISIONED, not scored: ESPEN and ERAS both direct that oral
# intake begin within hours of surgery, but no evidence sets a day-0 adequacy
# target, so there is no Layer 3 target to split. The generator is given the
# ERAS provision figure instead, as clear fluids only.
HAEMOSTASIS_DAILY_KCAL = 500.0
HAEMOSTASIS_MEAL_SPLITS = {"breakfast": 1 / 3, "lunch": 1 / 3, "dinner": 1 / 3}


def langgraph_profile(user_id: str, timeout: int = 30) -> Dict[str, Any]:
    """Body and restrictions for one patient, from their LangGraph profile."""
    from datetime import date as _date
    url = (f"{LANGGRAPH_BASE_URL.rstrip('/')}/langgraph/users/{user_id}"
           f"/state/{_date.today().isoformat()}?bootstrap_if_missing=true")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        state = json.loads(response.read().decode("utf-8"))
    profile = state.get("user_profile") or {}
    restrictions = profile.get("User Dietary Restrictions") or []
    if isinstance(restrictions, str):
        restrictions = [r.strip() for r in restrictions.split(",") if r.strip()]
    allergens = profile.get("User Allergens") or []
    if isinstance(allergens, str):
        allergens = [a.strip() for a in allergens.split(",") if a.strip()]
    sex = str(profile.get("User Sex") or "").strip().casefold()
    # The manual tracker outranks the profile, the same precedence the run
    # route applies — the pool must band the patient at the weight Layer 3
    # scored them at, or the two halves disagree by construction.
    weight = profile.get("User Weight")
    tracker = ((state.get("manual_trackers") or {}).get("weight") or {})
    if str(tracker.get("unit") or "kg").strip().casefold() in ("kg", "") and tracker.get("value"):
        weight = tracker["value"]
    return {
        "sex": "Female" if sex.startswith("f") else "Male",
        "age": float(profile.get("User Age") or 0) or None,
        "weight": float(weight) if weight else None,
        "height": float(profile.get("User Height") or 0) or None,
        "diets": list(restrictions),
        "allergens": list(allergens),
    }


def haemostasis_daily_target(*, sex: str, age: float, weight_kg: float,
                             height_cm: float) -> Dict[str, Any]:
    """Clear-fluid provision for day 0, in the shape the caller expects.

    Only ENERGY is constrained. Pinning protein, carbohydrate or fat on a
    clear-fluid day would push the generator towards foods that are not clear
    fluids, and no guideline gives day-0 macro splits to pin them to. The
    bands are still resolved the normal way, because the pool partition is an
    exact match on them regardless of which phase the row belongs to.
    """
    bands = {
        "age_band": MP.age_band_for(age),
        "weight_band_kg": MP.band_for(MP.WEIGHT_BANDS, weight_kg),
        "height_band_cm": MP.band_for(MP.HEIGHT_BANDS, height_cm),
    }
    missing = [k for k, v in bands.items() if not v]
    if missing:
        return {"status": "unresolved", "blocked_by": f"no band for {', '.join(missing)}"}
    return {
        "status": "resolved",
        **bands,
        "targets": {"calories_kcal": HAEMOSTASIS_DAILY_KCAL},
        "provisioned_only": True,
        "form": "clear_fluids",
        "source": "ERAS colorectal 2025 — ONS >= 500 kcal/day, days 1-5",
    }


def _band(lo: float, hi: float) -> Dict[str, float]:
    return {"minimum": round(lo, 1), "maximum": round(hi, 1)}


def meal_targets(daily: Dict[str, Any], meal: str,
                 provisioned_only: bool = False) -> Dict[str, Any]:
    """Split a daily macro target into one meal's target for the generator."""
    # Day 0 splits evenly rather than 20:40:40. The normal split assumes a
    # main meal; clear fluids on the day of surgery are taken as small volumes
    # through the day, and a 200 kcal "dinner" is not that.
    splits = HAEMOSTASIS_MEAL_SPLITS if provisioned_only else MEAL_SPLITS
    share = splits.get(meal, 1.0 / 3.0)

    if provisioned_only:
        # The generator requires all four macro bands, so "energy only" is not
        # an option — the bands have to say what a clear fluid IS.
        #
        # Fat is the defining constraint: a clear fluid is fat-free, and fat is
        # what makes a liquid opalescent, so it is pinned near zero. Energy
        # therefore comes almost entirely from carbohydrate, which is also what
        # the ERAS pre-operative carbohydrate drinks are. Protein is allowed a
        # generous ceiling rather than a target, because clear whey supplements
        # exist and are useful here, but nothing requires one.
        #
        # These are PROVISION bands for generating clear-fluid options, not a
        # macro target: Layer 3 still does not score day 0.
        kcal = float(daily.get("calories_kcal") or 0.0) * share
        protein_max = round(min(12.0, kcal * 0.20 / 4.0), 1)
        fat_max = 1.5
        carb_lo = max(0.0, (kcal - protein_max * 4 - fat_max * 9) / 4.0)
        carb_hi = kcal / 4.0
        return {
            "energy_kcal": _band(kcal * (1 - TOLERANCE), kcal * (1 + TOLERANCE)),
            "protein_g": _band(0.0, protein_max),
            "carbohydrates_g": _band(carb_lo * 0.9, carb_hi * 1.1),
            "fat_g": _band(0.0, fat_max),
            "sugar_g_max": round(carb_hi, 1),      # clear fluids are sugars
            "saturated_fat_g_max": 0.5,
            "cholesterol_mg_max": 5.0,
            "sodium_mg_max": 500.0,                 # broths are salty by nature
        }

    def scaled(name: str) -> Optional[float]:
        value = daily.get(name)
        try:
            return float(value) * share
        except (TypeError, ValueError):
            return None

    out: Dict[str, Any] = {}
    for target_name, gen_name in (
        ("calories_kcal", "energy_kcal"),
        ("protein_g", "protein_g"),
        ("carbohydrate_g", "carbohydrates_g"),
        ("fat_g", "fat_g"),
    ):
        value = scaled(target_name)
        if value is not None:
            out[gen_name] = _band(value * (1 - TOLERANCE), value * (1 + TOLERANCE))

    # Ceilings scale with the meal too; they are maxima, not ranges.
    for target_name, gen_name in (
        ("sodium_mg_max", "sodium_mg_max"),
        ("added_sugar_g_max", "sugar_g_max"),
        ("cholesterol_mg_max", "cholesterol_mg_max"),
        ("saturated_fat_g_max", "saturated_fat_g_max"),
    ):
        value = scaled(target_name)
        if value is not None:
            out[gen_name] = round(value, 1)
    return out


def build_daily_target(
    *, phase: str, sex: str, age: float, weight_kg: float, height_cm: float,
    nss_escalation: str = "none",
) -> Dict[str, Any]:
    """Layer 3 targets for this phase/weight, completed by the macro profile."""
    if phase == "HAEMOSTASIS":
        return haemostasis_daily_target(sex=sex, age=age, weight_kg=weight_kg,
                                        height_cm=height_cm)
    layer3_targets: Dict[str, Any] = {}
    for nutrient, spec in MP.NUTRIENT_TARGETS_FOR_PHASE(phase, weight_kg).items():
        layer3_targets[nutrient] = {"target": spec}
    return MP.resolve_phase1_macro_target(
        phase1_nutrition_targets={
            "current_phase": phase,
            "nss_escalation": nss_escalation,
            "targets": layer3_targets,
        },
        sex=sex, age=age, weight_kg=weight_kg, height_cm=height_cm,
    )


def call_generator(payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    request = urllib.request.Request(
        GENERATOR_URL + "?include_meta=true&include_pool_record=true",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:400]
        raise RuntimeError(f"generator HTTP {exc.code}: {detail}") from exc


def persist(record: Dict[str, Any], recipe: Dict[str, Any], *, phase: str,
            meal: str, profile: Dict[str, Any]) -> str:
    import psycopg

    recipe_id = str(record.get("recipe_id") or uuid.uuid4())
    row = {
        "recipe_id": recipe_id,
        "action_space": phase,
        "meal_type": meal,
        "macro_profile_version": MACRO_PROFILE_VERSION,
        "gender": profile["sex"],
        "age_band": profile["age_band"],
        "weight_band_kg": profile["weight_band_kg"],
        "height_band_cm": profile["height_band_cm"],
        "dietary_restrictions": record.get("dietary_restrictions") or [],
        "allergens": record.get("allergens") or [],
        "recipe": json.dumps(recipe),
        "status": "approved",
    }
    kw = dict(host=os.environ["APP_DB_HOST"], port=int(os.environ["APP_DB_PORT"]),
              dbname=os.environ["APP_DB_NAME"], user=os.environ["APP_DB_USER"],
              password=os.environ["APP_DB_PASS"],
              sslmode=os.environ.get("APP_DB_SSLMODE", "require"), connect_timeout=20)
    with psycopg.connect(**kw) as cn:
        with cn.cursor() as cur:
            cur.execute("""
                insert into recipe_pool (recipe_id, action_space, meal_type,
                    macro_profile_version, gender, age_band, weight_band_kg,
                    height_band_cm, dietary_restrictions, allergens, recipe, status)
                values (%(recipe_id)s, %(action_space)s, %(meal_type)s,
                    %(macro_profile_version)s, %(gender)s, %(age_band)s, %(weight_band_kg)s,
                    %(height_band_cm)s, %(dietary_restrictions)s, %(allergens)s,
                    %(recipe)s::jsonb, %(status)s)
                on conflict (recipe_id) do nothing
            """, row)
        cn.commit()
    return recipe_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Phase 1 recipes into recipe_pool.")
    ap.add_argument("--phase", required=True,
                    choices=("HAEMOSTASIS", "INFLAMMATION", "PROLIFERATION", "REMODELLING"))
    # --from-langgraph fills sex/age/weight/height/diets/allergens from the
    # patient's own profile. Typing them by hand is how a pool gets built for
    # a body and a diet that no real patient has, and the pool partition is an
    # EXACT match — a wrong band produces zero rows for that patient, silently.
    ap.add_argument("--from-langgraph", metavar="USER_ID",
                    help="pull the body and restrictions from this user's LangGraph profile")
    ap.add_argument("--sex", choices=("Male", "Female"))
    ap.add_argument("--age", type=float)
    ap.add_argument("--weight", type=float)
    ap.add_argument("--height", type=float)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--diets", default="")
    ap.add_argument("--allergens", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.from_langgraph:
        pulled = langgraph_profile(args.from_langgraph)
        for field in ("sex", "age", "weight", "height"):
            if getattr(args, field) is None:
                setattr(args, field, pulled[field])
        # The patient's own restrictions, unless the operator overrode them.
        # Generating a pool that ignores them produces rows the pool query can
        # never return for that patient: the dietary filter is a superset match
        # and the allergen filter an exclusion, so a mismatch yields zero rows.
        if not args.diets and pulled["diets"]:
            args.diets = ",".join(pulled["diets"])
        if not args.allergens and pulled["allergens"]:
            args.allergens = ",".join(pulled["allergens"])
        print(f"profile from LangGraph: {args.sex} age={args.age} "
              f"wt={args.weight} ht={args.height} "
              f"diets={args.diets or '-'} allergens={args.allergens or '-'}")

    for field in ("sex", "age", "weight", "height"):
        if getattr(args, field) is None:
            raise SystemExit(f"--{field} is required unless --from-langgraph is given")

    daily = build_daily_target(phase=args.phase, sex=args.sex, age=args.age,
                               weight_kg=args.weight, height_cm=args.height)
    if daily.get("status") != "resolved":
        raise SystemExit(f"macro target unresolved: {daily.get('blocked_by')}")

    profile = {
        "sex": args.sex, "age_band": daily["age_band"],
        "weight_band_kg": daily["weight_band_kg"], "height_band_cm": daily["height_band_cm"],
    }
    diets = [d.strip() for d in args.diets.split(",") if d.strip()]
    allergens = [a.strip() for a in args.allergens.split(",") if a.strip()]

    print(f"phase={args.phase} {args.sex} {profile['age_band']} "
          f"wt={profile['weight_band_kg']} ht={profile['height_band_cm']}")
    print("daily target:", {k: daily["targets"].get(k) for k in
                            ("calories_kcal", "protein_g", "carbohydrate_g", "fat_g")})

    written: List[str] = []
    for meal in MEALS:
        mt = meal_targets(daily["targets"], meal,
                          provisioned_only=bool(daily.get("provisioned_only")))
        print(f"\n{meal}: {json.dumps(mt)}")
        for variant in range(1, args.variants + 1):
            payload = {
                "action_space": args.phase,
                "macro_profile_version": MACRO_PROFILE_VERSION,
                "meal_type": meal,
                "servings": 1,
                "gender": args.sex,
                "age_band": profile["age_band"],
                "weight_band_kg": profile["weight_band_kg"],
                "height_band_cm": profile["height_band_cm"],
                "targets": mt,
            }
            if diets:
                payload["required_dietary_restrictions"] = diets
            if allergens:
                payload["excluded_allergens"] = allergens
            if args.dry_run:
                print(f"  [dry-run] {meal} v{variant}")
                continue
            try:
                result = call_generator(payload)
            except Exception as exc:
                print(f"  v{variant} GENERATOR FAILED: {exc}")
                continue
            recipe = result.get("recipe") or result
            record = result.get("pool_record") or {}
            # The generator persists to recipe_pool itself (RECIPE_POOL_WRITE_MODE).
            # Only fall back to writing here if it returned a record it did not store.
            rid = str(record.get("recipe_id") or "")
            if not rid:
                rid = persist(record, recipe, phase=args.phase, meal=meal, profile=profile)
            written.append(rid)
            title = (recipe.get("food_title") or recipe.get("title") or "?")
            print(f"  v{variant} -> {str(title)[:52]}  ({rid[:8]})")

    print(f"\nwrote {len(written)} recipes to recipe_pool")


if __name__ == "__main__":
    main()
