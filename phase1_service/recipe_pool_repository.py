"""Read pre-generated post-op recipes from the AI-RDS recipe pool.

The selector applies the exact CQL action space and macro profile, meal type,
user gender, age/weight/height band containment, dietary compatibility, allergen
exclusion, and recent-recipe exclusion. It returns up to N recipes per meal.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List
from urllib import error, request


class RecipePoolReadError(RuntimeError):
    pass


def _postgres_connection_settings() -> tuple[str | None, Dict[str, Any]]:
    """Resolve PostgreSQL settings from a DSN or APP_DB_* variables."""
    dsn = os.getenv("RECIPE_POOL_DATABASE_URL", "").strip() or None
    if dsn:
        return dsn, {}

    required_names = (
        "APP_DB_HOST",
        "APP_DB_PORT",
        "APP_DB_NAME",
        "APP_DB_USER",
        "APP_DB_PASS",
    )
    values = {name: os.getenv(name, "").strip() for name in required_names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RecipePoolReadError(
            "PostgreSQL is not configured. Set RECIPE_POOL_DATABASE_URL or: "
            + ", ".join(required_names)
            + ". Missing: "
            + ", ".join(missing)
        )

    settings: Dict[str, Any] = {
        "host": values["APP_DB_HOST"],
        "port": int(values["APP_DB_PORT"]),
        "dbname": values["APP_DB_NAME"],
        "user": values["APP_DB_USER"],
        "password": values["APP_DB_PASS"],
    }
    sslmode = os.getenv("APP_DB_SSLMODE", "").strip()
    if sslmode:
        settings["sslmode"] = sslmode
    return None, settings


def _postgres_is_configured() -> bool:
    if os.getenv("RECIPE_POOL_DATABASE_URL", "").strip():
        return True
    return all(
        os.getenv(name, "").strip()
        for name in (
            "APP_DB_HOST",
            "APP_DB_PORT",
            "APP_DB_NAME",
            "APP_DB_USER",
            "APP_DB_PASS",
        )
    )


def _validated_table_name() -> str:
    table = os.getenv("RECIPE_POOL_TABLE", "recipe_pool").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise RecipePoolReadError("RECIPE_POOL_TABLE contains unsafe characters")
    return table


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    out: List[str] = []
    seen = set()
    for item in values:
        if isinstance(item, dict):
            item = item.get("name") or item.get("title") or item.get("type")
        text = str(item or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _fetch_postgres(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    dsn, connection_settings = _postgres_connection_settings()
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise RecipePoolReadError(
            "Direct PostgreSQL mode requires: pip install 'psycopg[binary]'"
        ) from exc

    table = _validated_table_name()
    sql = f"""
        WITH base AS (
            SELECT
                recipe_id,
                action_space,
                meal_type,
                macro_profile_version,
                gender,
                age_band,
                weight_band_kg,
                height_band_cm,
                dietary_restrictions,
                allergens,
                recipe,
                status,
                created_at,
                regexp_replace(
                    translate(coalesce(age_band, ''), '–—', '--'), '\\s', '', 'g'
                ) AS age_band_clean,
                regexp_replace(
                    translate(coalesce(weight_band_kg, ''), '–—', '--'), '\\s', '', 'g'
                ) AS weight_band_clean,
                regexp_replace(
                    translate(coalesce(height_band_cm, ''), '–—', '--'), '\\s', '', 'g'
                ) AS height_band_clean
            FROM {table}
            WHERE status = 'approved'
              AND action_space = %(action_space)s
              AND macro_profile_version = %(macro_profile_version)s
              AND meal_type = ANY(%(meal_types)s)
              AND (
                    lower(trim(coalesce(gender, ''))) = lower(%(gender)s)
                    OR lower(trim(coalesce(gender, ''))) IN ('any', 'all')
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM unnest(%(required_dietary_restrictions)s::text[]) AS required(tag)
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM unnest(
                            coalesce(dietary_restrictions, ARRAY[]::text[])
                        ) AS actual(tag)
                        WHERE lower(trim(actual.tag)) = lower(trim(required.tag))
                    )
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM unnest(coalesce(allergens, ARRAY[]::text[])) AS actual(tag)
                    JOIN unnest(%(excluded_allergens)s::text[]) AS blocked(tag)
                      ON lower(trim(actual.tag)) = lower(trim(blocked.tag))
              )
              AND NOT (recipe_id::text = ANY(%(excluded_recipe_ids)s::text[]))
        ), eligible AS (
            SELECT *
            FROM base
            WHERE CASE
                WHEN lower(age_band_clean) IN ('any', 'all') THEN TRUE
                WHEN age_band_clean ~ '^[0-9]+([.][0-9]+)?-[0-9]+([.][0-9]+)?$' THEN
                    %(age)s::numeric BETWEEN
                        split_part(age_band_clean, '-', 1)::numeric AND
                        split_part(age_band_clean, '-', 2)::numeric
                WHEN age_band_clean ~ '^[0-9]+([.][0-9]+)?[+]$' THEN
                    %(age)s::numeric >= replace(age_band_clean, '+', '')::numeric
                ELSE FALSE
            END
              AND CASE
                WHEN lower(weight_band_clean) IN ('any', 'all') THEN TRUE
                WHEN weight_band_clean ~ '^[0-9]+([.][0-9]+)?-[0-9]+([.][0-9]+)?$' THEN
                    %(weight_kg)s::numeric BETWEEN
                        split_part(weight_band_clean, '-', 1)::numeric AND
                        split_part(weight_band_clean, '-', 2)::numeric
                WHEN weight_band_clean ~ '^[0-9]+([.][0-9]+)?[+]$' THEN
                    %(weight_kg)s::numeric >= replace(weight_band_clean, '+', '')::numeric
                ELSE FALSE
            END
              AND CASE
                WHEN lower(height_band_clean) IN ('any', 'all') THEN TRUE
                WHEN height_band_clean ~ '^[0-9]+([.][0-9]+)?-[0-9]+([.][0-9]+)?$' THEN
                    %(height_cm)s::numeric BETWEEN
                        split_part(height_band_clean, '-', 1)::numeric AND
                        split_part(height_band_clean, '-', 2)::numeric
                WHEN height_band_clean ~ '^[0-9]+([.][0-9]+)?[+]$' THEN
                    %(height_cm)s::numeric >= replace(height_band_clean, '+', '')::numeric
                ELSE FALSE
            END
        ), ranked AS (
            SELECT
                eligible.*,
                ROW_NUMBER() OVER (
                    PARTITION BY meal_type
                    ORDER BY md5(
                        recipe_id::text
                        || ':'
                        || %(user_id)s::text
                        || ':'
                        || %(recommendation_date)s::text
                    )
                ) AS meal_rank
            FROM eligible
        )
        SELECT
            recipe_id,
            action_space,
            meal_type,
            macro_profile_version,
            gender,
            age_band,
            weight_band_kg,
            height_band_cm,
            dietary_restrictions,
            allergens,
            recipe,
            status,
            created_at
        FROM ranked
        WHERE meal_rank <= %(recipes_per_meal)s
        ORDER BY CASE meal_type
                    WHEN 'breakfast' THEN 1
                    WHEN 'lunch' THEN 2
                    WHEN 'dinner' THEN 3
                    ELSE 4
                 END,
                 meal_rank;
    """
    params = dict(filters)
    params["excluded_recipe_ids"] = _as_text_list(filters.get("excluded_recipe_ids"))
    params["required_dietary_restrictions"] = _as_text_list(
        filters.get("required_dietary_restrictions")
    )
    params["excluded_allergens"] = _as_text_list(filters.get("excluded_allergens"))
    timeout = int(os.getenv("RECIPE_POOL_DB_CONNECT_TIMEOUT_SEC", "10"))
    connect_args = {
        "connect_timeout": timeout,
        "row_factory": dict_row,
        **connection_settings,
    }
    connection_context = (
        psycopg.connect(dsn, **connect_args)
        if dsn
        else psycopg.connect(**connect_args)
    )
    with connection_context as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
    return [dict(row) for row in rows]


def _fetch_api(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = os.getenv("RECIPE_POOL_READ_API_URL", "").strip()
    if not url:
        raise RecipePoolReadError("RECIPE_POOL_READ_API_URL is not configured")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.getenv("RECIPE_POOL_READ_API_KEY", "").strip()
    if token:
        header = os.getenv("RECIPE_POOL_READ_AUTH_HEADER", "Authorization").strip()
        scheme = os.getenv("RECIPE_POOL_READ_AUTH_SCHEME", "Bearer").strip()
        headers[header] = f"{scheme} {token}".strip()
    req = request.Request(
        url,
        data=json.dumps(filters, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(
            req, timeout=int(os.getenv("RECIPE_POOL_READ_TIMEOUT_SEC", "30"))
        ) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RecipePoolReadError(
            f"Recipe-pool API returned HTTP {exc.code}: {detail[:1200]}"
        ) from exc
    except error.URLError as exc:
        raise RecipePoolReadError(f"Recipe-pool API is unreachable: {exc}") from exc
    rows = body.get("recipes") if isinstance(body, dict) else body
    if not isinstance(rows, list):
        raise RecipePoolReadError("Recipe-pool API response must contain a recipes array")
    return [row for row in rows if isinstance(row, dict)]


def _clean_band(value: Any) -> str:
    return str(value or "").replace("–", "-").replace("—", "-").replace(" ", "")


def _band_contains(band: Any, value: float) -> bool:
    text = _clean_band(band).casefold()
    if text in {"any", "all"}:
        return True
    match = re.fullmatch(r"([0-9]+(?:[.][0-9]+)?)-([0-9]+(?:[.][0-9]+)?)", text)
    if match:
        return float(match.group(1)) <= float(value) <= float(match.group(2))
    match = re.fullmatch(r"([0-9]+(?:[.][0-9]+)?)[+]", text)
    if match:
        return float(value) >= float(match.group(1))
    return False


def _fetch_json(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = os.getenv("RECIPE_POOL_PLACEHOLDER_JSON", "").strip()
    if not path:
        raise RecipePoolReadError("RECIPE_POOL_PLACEHOLDER_JSON is not configured")
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise RecipePoolReadError("Placeholder JSON must contain an array")

    required_diets = {
        x.casefold() for x in _as_text_list(filters.get("required_dietary_restrictions"))
    }
    blocked = {x.casefold() for x in _as_text_list(filters.get("excluded_allergens"))}
    excluded_ids = set(_as_text_list(filters.get("excluded_recipe_ids")))
    eligible: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_diets = {
            str(x).casefold() for x in _as_text_list(row.get("dietary_restrictions"))
        }
        row_allergens = {
            str(x).casefold() for x in _as_text_list(row.get("allergens"))
        }
        row_gender = str(row.get("gender") or "").casefold()
        if str(row.get("status", "")).casefold() != "approved":
            continue
        if row.get("action_space") != filters["action_space"]:
            continue
        if row.get("macro_profile_version") != filters["macro_profile_version"]:
            continue
        if row_gender not in {str(filters["gender"]).casefold(), "any", "all"}:
            continue
        if not _band_contains(row.get("age_band"), float(filters["age"])):
            continue
        if not _band_contains(row.get("weight_band_kg"), float(filters["weight_kg"])):
            continue
        if not _band_contains(row.get("height_band_cm"), float(filters["height_cm"])):
            continue
        if row.get("meal_type") not in filters["meal_types"]:
            continue
        if required_diets and not required_diets.issubset(row_diets):
            continue
        if blocked.intersection(row_allergens):
            continue
        if str(row.get("recipe_id")) in excluded_ids:
            continue
        eligible.append(dict(row))

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seed = f"{filters['user_id']}:{filters['recommendation_date']}"
    for row in sorted(
        eligible,
        key=lambda item: hashlib.md5(
            f"{item.get('recipe_id')}:{seed}".encode("utf-8")
        ).hexdigest(),
    ):
        meal = str(row.get("meal_type"))
        if len(grouped[meal]) < int(filters["recipes_per_meal"]):
            grouped[meal].append(row)
    return [row for meal in filters["meal_types"] for row in grouped.get(meal, [])]


def fetch_daily_recipe_rows(**filters: Any) -> List[Dict[str, Any]]:
    mode = os.getenv("RECIPE_POOL_READ_MODE", "postgres").strip().lower()
    if mode == "postgres":
        return _fetch_postgres(filters)
    if mode == "api":
        return _fetch_api(filters)
    if mode == "json":
        return _fetch_json(filters)
    raise RecipePoolReadError(f"Unsupported RECIPE_POOL_READ_MODE: {mode}")


def recipe_pool_healthcheck() -> Dict[str, Any]:
    mode = os.getenv("RECIPE_POOL_READ_MODE", "postgres").strip().lower()
    if mode == "postgres":
        return {
            "mode": mode,
            "configured": _postgres_is_configured(),
            "table": os.getenv("RECIPE_POOL_TABLE", "recipe_pool"),
            "host": os.getenv("APP_DB_HOST", "") or None,
            "database": os.getenv("APP_DB_NAME", "") or None,
        }
    if mode == "api":
        return {
            "mode": mode,
            "configured": bool(os.getenv("RECIPE_POOL_READ_API_URL", "").strip()),
        }
    if mode == "json":
        return {
            "mode": mode,
            "configured": bool(os.getenv("RECIPE_POOL_PLACEHOLDER_JSON", "").strip()),
        }
    return {"mode": mode, "configured": False}
