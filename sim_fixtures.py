"""The real patients, loaded from the service's own fixtures directory.

**Not used by the app.** The app runs on generated patients only. This exists so
`test_replay.py` can check the generator against data the team actually curated
— without that anchor, the simulation would only ever be self-consistent.

These are the files `phase1_daily_recommendation.py --timeline` runs and the
ones `demo.sh` posts to the endpoints, read from where the service keeps them
rather than copied. A copy would drift the moment someone regenerated them, and
a demo running on a stale copy of the data is worse than no demo.

Nothing here modifies a timeline. `as_of` slices it, which is the one operation
a day-by-day replay needs: post-op day D must be judged on days 0..D and
nothing later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sim_paths

FIXTURE_DIR = sim_paths.STAGING / "fixtures"

# What each patient is for. Taken from the service README's own table so the
# demo describes them the way the team already does.
BLURBS: Dict[str, Dict[str, str]] = {
    "patient_six_week": {
        "label": "Six-week recovery — the full arc",
        "blurb": "All four healing phases in one patient, and the section 10 "
                 "gate opening as each sign-off lands. The end-to-end story.",
    },
    "patient_uncomplicated": {
        "label": "Uncomplicated recovery",
        "blurb": "The negative control. If anything fires here, every positive "
                 "finding elsewhere is worthless.",
    },
    "patient_ssi": {
        "label": "Surgical site infection",
        "blurb": "The flagship case. CRP rebounds from day 8, a fever runs most "
                 "of the day, glucose climbs — all three layers speak.",
    },
    "patient_malnourished": {
        "label": "Healing well, badly underfed",
        "blurb": "Zero deviations and a normal phase trajectory on an intake "
                 "that cannot build collagen. Layer 3 raises the alarm alone.",
    },
    "patient_diabetic": {
        "label": "Diabetic, poor glycaemic control",
        "blurb": "Section 5.3's glucose path, with section 8's 200 mg/dL "
                 "immediate-escalation path firing above it.",
    },
    "patient_sparse_wearable": {
        "label": "Wearable worn intermittently",
        "blurb": "The device came off. Exercises the not-assessed path: a check "
                 "that could not run is reported as not run, never as clear.",
    },
    "patient_ready_for_phase2": {
        "label": "Ready for Phase 2",
        "blurb": "Starts on day 22. Every section 10 criterion satisfiable, "
                 "including the three only a human can sign.",
    },
    "edge_single_day": {
        "label": "Edge — a single day",
        "blurb": "Day 0 only. No trend exists yet and the GP has no history; "
                 "nothing downstream may crash.",
    },
    "edge_labs_only": {
        "label": "Edge — labs only",
        "blurb": "CRP and nothing else. No wearable, no intake log, no "
                 "sign-off — most checks cannot run and must say so.",
    },
}

# Best first: the six-week arc is the demo, the rest are the cases it does not
# cover. Anything not named here sorts after, alphabetically.
ORDER = ("patient_six_week", "patient_ssi", "patient_uncomplicated",
         "patient_malnourished", "patient_diabetic", "patient_sparse_wearable",
         "patient_ready_for_phase2", "edge_single_day", "edge_labs_only")


def _sort_key(name: str) -> tuple:
    return (ORDER.index(name), name) if name in ORDER else (len(ORDER), name)


def available() -> List[Dict[str, Any]]:
    """Every fixture on disk, described. Reads them; does not run them."""
    if not FIXTURE_DIR.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in FIXTURE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:                                # noqa: BLE001
            continue
        days = data.get("days") or []
        if not days:
            continue
        numbers = [int(d.get("day", 0)) for d in days]
        patient = data.get("patient") or {}
        meta = BLURBS.get(path.stem, {})
        out.append({
            "name": path.stem,
            "path": str(path),
            "label": meta.get("label", path.stem.replace("_", " ").capitalize()),
            "blurb": meta.get("blurb", ""),
            "patient_ref": data.get("patient_ref"),
            "surgery_type": patient.get("surgery_type"),
            "weight_kg": patient.get("weight_kg"),
            "vitamin_a_at_risk": patient.get("vitamin_a_at_risk"),
            "day_count": len(days),
            "first_day": min(numbers),
            "last_day": max(numbers),
            "size_kb": path.stat().st_size // 1024,
            "has_nutrition": sum(1 for d in days if d.get("nutrition")),
            "has_vitals": sum(1 for d in days if d.get("vitals")),
        })
    out.sort(key=lambda r: _sort_key(r["name"]))
    return out


def load(name: str) -> Dict[str, Any]:
    """One fixture, by stem."""
    path = FIXTURE_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No fixture {name!r} in {FIXTURE_DIR}")
    return json.loads(path.read_text())


# Slicing lives in sim_replay, which is what needs it. Re-exported so the tests
# that check the prefix property can reach it from either name.
from sim_replay import as_of, day_numbers  # noqa: E402,F401


def patient_profile(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """Body and demographics, with the fixture's own values kept as-is.

    The fixtures carry `surgery_type`, `weight_kg` and the Vitamin A flag —
    everything Layers 1-3 need. They carry no age, sex or height, because no
    layer reads those: only the recipe pool's demographic band does. The caller
    supplies them and this says plainly which came from the file.
    """
    patient = timeline.get("patient") or {}
    return {
        "surgery_type": patient.get("surgery_type"),
        "weight_kg": patient.get("weight_kg"),
        "vitamin_a_at_risk": patient.get("vitamin_a_at_risk"),
        "age": patient.get("age"),
        "sex": patient.get("sex") or patient.get("gender"),
        "height_cm": patient.get("height_cm"),
        "from_fixture": sorted(k for k in patient if patient.get(k) is not None),
    }
