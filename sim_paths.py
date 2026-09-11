"""Make the Phase 1 service importable, without reimplementing any of it.

The simulator has no clinical logic of its own. Every threshold, matrix and
rule it shows comes from the Phase 1 service, imported at run time.

It looks in two places, in order:

1. **`../Postop-Phase1/Staging`** — the real service, when this is a checkout
   sitting beside it. Always preferred, so a threshold changed there shows up
   here on the next reload and the two can never silently disagree.
2. **`phase1_service/`** — a vendored copy, for a deployment that has no
   sibling checkout (Streamlit Cloud sees only this repository).

The vendored copy is the same code with one file changed: its `phase1_config`
holds no credentials. `test_vendored_parity.py` diffs the two whenever both are
present, so the copy cannot drift unnoticed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent / "Postop-Phase1" / "Staging"
VENDORED = HERE / "phase1_service"

# Which one is actually in use, for the UI to state plainly.
SOURCE = "service" if (SERVICE / "layer1_phase_estimator.py").is_file() else "vendored"
STAGING = SERVICE if SOURCE == "service" else VENDORED


def install() -> Path:
    """Put the service on the path and apply its env defaults. Idempotent."""
    if not (STAGING / "layer1_phase_estimator.py").is_file():
        raise RuntimeError(
            f"Phase 1 was not found at {SERVICE} or {VENDORED}. The simulator "
            "imports the real layers rather than reproducing them, so it "
            "cannot run without one of them.")
    path = str(STAGING)
    if path not in sys.path:
        sys.path.insert(0, path)

    # Credentials and the LangGraph base URL. The service's own config fills
    # them from a .env beside it; the vendored config reads the environment and
    # Streamlit secrets, and carries no defaults for anything secret.
    try:
        import phase1_config

        phase1_config.apply_defaults()
    except Exception:                                   # noqa: BLE001
        pass                        # only the recipe pool needs any of this

    # The staging database is not always reachable — VPN, an allowlist, or a
    # host that simply cannot see it. The repository's own default waits 15 s
    # before saying so, and a demo that stalls that long on every interaction
    # is worse than one that says "unavailable" quickly.
    os.environ.setdefault("RECIPE_POOL_DB_CONNECT_TIMEOUT_SEC", "6")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return STAGING


install()
