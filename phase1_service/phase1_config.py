"""
Phase 1 service configuration — deployment copy.

The same module the service uses, with **one difference**: it holds no
credentials. The service's own copy spells the staging database out in code so
it starts with a bare uvicorn command; that is reasonable on a laptop and
unacceptable in a repository, so here every secret is read from the environment
and there are no defaults to fall back to.

Set them however the host prefers:

    export RECIPE_POOL_DATABASE_URL='<postgres url for the recipe pool>'
    export ANTHROPIC_API_KEY='<your Anthropic API key>'

or, on Streamlit Cloud, in the app's **Secrets** panel:

    RECIPE_POOL_DATABASE_URL = "<postgres url for the recipe pool>"
    ANTHROPIC_API_KEY = "<your Anthropic API key>"

Neither is required to run. Without the database the recipes come from the
local cache in `.recipe_cache/`, labelled with when they were fetched; without
the key Layer 4 falls back to its deterministic template. Both are the
documented fallbacks, not failures.
"""

from __future__ import annotations

import os
from pathlib import Path

# Non-secret settings. These are the same values the service ships with and
# none of them identify anything.
RECIPE_POOL_READ_MODE = "postgres"
RECIPE_POOL_TABLE = "recipe_pool"
RECIPE_POOL_DB_CONNECT_TIMEOUT_SEC = "6"
LANGGRAPH_BASE_URL = "http://127.0.0.1:8000"
PHASE1_MACRO_PROFILE_VERSION = "phase1_v1"

DEFAULTS = {
    "RECIPE_POOL_READ_MODE": RECIPE_POOL_READ_MODE,
    "RECIPE_POOL_TABLE": RECIPE_POOL_TABLE,
    "RECIPE_POOL_DB_CONNECT_TIMEOUT_SEC": RECIPE_POOL_DB_CONNECT_TIMEOUT_SEC,
    "LANGGRAPH_BASE_URL": LANGGRAPH_BASE_URL,
    "PHASE1_MACRO_PROFILE_VERSION": PHASE1_MACRO_PROFILE_VERSION,
    # RECIPE_POOL_DATABASE_URL is deliberately absent. A default here would be
    # a credential in a repository, which is the thing this file exists to
    # avoid.
}

# A .env beside this file still works for local use and is git-ignored.
ENV_FILE = Path(__file__).resolve().parent / ".env"

_APPLIED = False


def _load_env_file(path: Path) -> int:
    """Read KEY=VALUE lines from .env into os.environ, not overriding real ones."""
    if not path.is_file():
        return 0
    loaded = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.getenv(key, "").strip():
            os.environ[key] = value
            loaded += 1
    return loaded


def _from_streamlit_secrets() -> None:
    """Lift secrets out of Streamlit's own store into the environment.

    `recipe_pool_repository` and Layer 4 both read `os.environ` directly, so a
    value living only in `st.secrets` would never reach them.
    """
    try:
        import streamlit as st

        for key in ("RECIPE_POOL_DATABASE_URL", "ANTHROPIC_API_KEY"):
            value = st.secrets.get(key)
            if value and not os.getenv(key, "").strip():
                os.environ[key] = str(value)
    except Exception:                                   # noqa: BLE001
        pass          # not running under Streamlit, or no secrets configured


def apply_defaults() -> None:
    """Publish the non-secret defaults into os.environ, secrets from elsewhere."""
    global _APPLIED
    if _APPLIED:
        return
    _load_env_file(ENV_FILE)
    _from_streamlit_secrets()
    for key, value in DEFAULTS.items():
        if not os.getenv(key, "").strip() and value:
            os.environ[key] = str(value)
    _APPLIED = True
