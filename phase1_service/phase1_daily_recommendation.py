"""
Post-op Phase 1 — Orchestrator.

Implements sections 8 and 10 of Phase_1_Wound_Recovery.docx: the full system
flow, and the hard gate into Phase 2.

    Layer 1  HMM            which healing phase
    Layer 2  GP + rules     is healing on track
    Layer 3  ASPEN/ESPEN    is the patient being fed for that phase
    Layer 4  LLM            say it to the physician

Section 8 defines five distinct cadences, and they are not the same job run at
different speeds. The daily summary consolidates; the threshold-breach path
escalates immediately and *does not wait for it*. Collapsing those two would
mean a septic patient waited until 06:00 for a page.

Run it:

    python3 phase1_daily_recommendation.py --timeline example_patient.json
    python3 phase1_daily_recommendation.py --timeline example_patient.json --json
    uvicorn phase1_daily_recommendation:app --port 8020 --reload
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# BEFORE the layer imports. Configuration has to be in os.environ by the time
# any layer reads it, and it must not depend on which module happens to be
# imported first: Layer 4 checks ANTHROPIC_API_KEY, and /health reports it, so
# loading config lazily made the service claim the LLM was unconfigured until
# the first request had already run.
try:
    from phase1_config import apply_defaults
    apply_defaults()
except Exception:                       # standalone use without the config file
    pass

from layer1_phase_estimator import estimate_phase, validate_days
from layer2_deviation_detector import (
    detect_deviations, feature_summary, stream, sustained_run,
    series, HR_HIGH, TEMP_HIGH, TEMP_SUSTAINED_HOURS, HR_SUSTAINED_HOURS,
)
from layer3_nutrition_gap import score_nutrition
from layer4_physician_alert import generate_alert

LOG = logging.getLogger(__name__)

MODEL_VERSION = "phase1-orchestrator-v1"


# ===========================================================================
# 1. CONFIG — section 8's real-time escalation thresholds, section 10's gate
# ===========================================================================

# Section 8, ON THRESHOLD BREACH: "sustained fever, glucose > 200, HR > 110
# sustained, SpO2 < 93". These are deliberately HIGHER than section 5.3's
# deviation thresholds (glucose 180, HR 100) — 5.3 flags a concern for the ward
# round, 8 pages someone now. Keeping both sets is the point, not a duplication.
BREACH_GLUCOSE = 200.0
BREACH_HR = 110.0
BREACH_SPO2 = 93.0
BREACH_HR_SUSTAINED_HOURS = HR_SUSTAINED_HOURS
BREACH_TEMP_SUSTAINED_HOURS = TEMP_SUSTAINED_HOURS

DAILY_SUMMARY_TIME = "06:00"        # section 8: "DAILY (06:00 ward round preparation)"

# --- section 10, the Phase 1 -> Phase 2 transition gate ---------------------
#
# Every criterion must pass. Three are marked mandatory-human in the document and
# cannot be satisfied by any computation — the gate exists precisely so that the
# Phase 2 metabolic policy, which may recommend a caloric deficit, cannot
# activate while wound healing is still consuming protein.
GATE_CRP_BELOW = 10.0               # mg/L, "and declining"
GATE_PREALBUMIN_ABOVE = 150.0       # mg/L, "or returning to baseline"
GATE_FASTING_GLUCOSE_BELOW = 126.0  # mg/dL, "on 2 consecutive readings"
GATE_FASTING_GLUCOSE_READINGS = 2
GATE_REMODELLING_POSTERIOR = 0.85   # "HMM posterior > 85% probability of Remodelling"


# ===========================================================================
# 2. CONTINUOUS — section 8, every 15 minutes
# ===========================================================================

# Sign-off fields, and the order they are reported in.
CLINICAL_SIGNOFF_FIELDS = ("wound_closure_confirmed",
                           "full_oral_diet_tolerated",
                           "physician_discharge_signoff")


def latest_clinical(days: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The most recent recorded value of each sign-off, not just today's block.

    A sign-off is a fact about the PATIENT, not an observation about a calendar
    day. Reading only days[-1] meant a physician could confirm wound closure on
    day 9 and the gate would silently re-block on day 10 simply because the
    caller did not repeat the block — turning a clerical omission into a
    clinical regression, with `observed: null` the only clue.

    Resolved per FIELD rather than per block, so a day that records only the
    dietitian's diet assessment does not erase the physician's earlier wound
    confirmation.

    Deliberately NOT a latch. Wound dehiscence and post-operative ileus are real,
    so an explicit `false` on a later day must be able to withdraw an earlier
    `true`. Only an ABSENT field inherits; a present one always wins.
    """
    resolved: Dict[str, Any] = {}
    for day in days:                     # already day-ordered by the caller
        block = day.get("clinical")
        if not isinstance(block, dict):
            continue
        for field in CLINICAL_SIGNOFF_FIELDS:
            if field in block:
                resolved[field] = block[field]
    return resolved


def run_continuous(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ingest the wearable stream and compute the rolling feature statistics.

    Section 8's first cadence. Separated from the layers because it is the only
    stage that touches raw samples: everything downstream reads either the
    features it produces or the sparse labs. Running it once here means the
    rolling windows are computed once rather than inside each detector.
    """
    days = sorted(timeline.get("days", []), key=lambda d: int(d.get("day", 0)))
    return [
        {"day": int(d.get("day", 0)),
         "hr": feature_summary(stream(d, "hr")),
         "temp": feature_summary(stream(d, "temp")),
         "spo2": feature_summary(stream(d, "spo2"))}
        for d in days
    ]


# ===========================================================================
# 3. ON THRESHOLD BREACH — section 8, any time
# ===========================================================================

def check_threshold_breaches(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Real-time critical escalations that do not wait for the daily summary.

    A separate pass rather than a filter over Layer 2's findings, because these
    are a different clinical question. Layer 2 asks "is the trajectory
    deviating"; this asks "is this patient in danger right now". The thresholds
    are higher and the response is immediate, and a system that only produced
    them at 06:00 would be dangerous.
    """
    days = sorted(timeline.get("days", []), key=lambda d: int(d.get("day", 0)))
    breaches: List[Dict[str, Any]] = []

    for day in days:
        number = int(day.get("day", 0))

        fever = sustained_run(stream(day, "temp"), TEMP_HIGH)
        if fever and fever["hours"] >= BREACH_TEMP_SUSTAINED_HOURS:
            breaches.append({
                "day": number, "trigger": "sustained_fever",
                "detail": f"Temperature above {TEMP_HIGH}°C for {fever['hours']:.1f} hours "
                          f"({fever['start']:.2f}h-{fever['end']:.2f}h), peak {fever['peak']}°C.",
                "escalate_to": ["Physician"], "source": "doc section 8"})

        tachy = sustained_run(stream(day, "hr"), BREACH_HR)
        if tachy and tachy["hours"] >= BREACH_HR_SUSTAINED_HOURS:
            breaches.append({
                "day": number, "trigger": "sustained_tachycardia",
                "detail": f"Heart rate above {BREACH_HR:.0f} bpm for {tachy['hours']:.1f} hours, "
                          f"peak {tachy['peak']:.0f} bpm.",
                "escalate_to": ["Physician"], "source": "doc section 8"})

        low_spo2 = [v for _, v in stream(day, "spo2") if v < BREACH_SPO2]
        if low_spo2:
            breaches.append({
                "day": number, "trigger": "hypoxaemia",
                "detail": f"SpO2 below {BREACH_SPO2:.0f}% on {len(low_spo2)} readings, "
                          f"lowest {min(low_spo2):.1f}%.",
                "escalate_to": ["Physician"], "source": "doc section 8"})

        labs = day.get("labs") if isinstance(day.get("labs"), dict) else {}
        glucose = labs.get("glucose")
        if glucose not in (None, "") and float(glucose) > BREACH_GLUCOSE:
            breaches.append({
                "day": number, "trigger": "severe_hyperglycaemia",
                "detail": f"Glucose {float(glucose):.0f} mg/dL, above the "
                          f"{BREACH_GLUCOSE:.0f} mg/dL immediate-escalation threshold.",
                "escalate_to": ["Physician", "Pharmacy"], "source": "doc section 8"})

    return breaches


# ===========================================================================
# 4. THE TRANSITION GATE — section 10
# ===========================================================================

def evaluate_transition_gate(timeline: Dict[str, Any],
                             layer1: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all seven Phase 1 -> Phase 2 criteria.

    Implemented as a hard gate returning per-criterion detail rather than a
    boolean, because the interesting output is *which* criterion is blocking. The
    three physician- and dietitian-confirmed criteria can never be satisfied by
    computation: they read explicit clinical sign-off fields, and their absence
    is a block, not a pass. That asymmetry is the whole safety property — Phase 2
    optimises for biomarkers and may recommend a caloric deficit, which is
    actively harmful while collagen is still being laid down.
    """
    days = sorted(timeline.get("days", []), key=lambda d: int(d.get("day", 0)))
    clinical = latest_clinical(days)

    crp = series(days, "labs", "crp")
    prealbumin = series(days, "labs", "prealbumin")
    fasting = [v for _, v in series(days, "labs", "fasting_glucose")]

    crp_ok = bool(crp) and crp[-1][1] < GATE_CRP_BELOW and (
        len(crp) < 2 or crp[-1][1] <= crp[-2][1])
    prealbumin_ok = bool(prealbumin) and prealbumin[-1][1] > GATE_PREALBUMIN_ABOVE
    glucose_ok = (len(fasting) >= GATE_FASTING_GLUCOSE_READINGS
                  and all(v < GATE_FASTING_GLUCOSE_BELOW
                          for v in fasting[-GATE_FASTING_GLUCOSE_READINGS:]))
    remodelling = layer1.get("phase_posterior", {}).get("REMODELLING", 0.0)
    phase_ok = remodelling > GATE_REMODELLING_POSTERIOR

    criteria = [
        {"criterion": "Wound closure", "signal": "Clinical assessment",
         "threshold": "Physician confirmation of primary intention healing",
         "confirmed_by": "Physician — mandatory",
         "met": bool(clinical.get("wound_closure_confirmed")),
         "observed": clinical.get("wound_closure_confirmed", None)},
        {"criterion": "Inflammation resolved", "signal": "CRP",
         "threshold": f"CRP < {GATE_CRP_BELOW:g} mg/L and declining",
         "confirmed_by": "Automated + physician review",
         "met": crp_ok, "observed": crp[-1][1] if crp else None},
        {"criterion": "Nutritional status stable", "signal": "Prealbumin",
         "threshold": f"Prealbumin > {GATE_PREALBUMIN_ABOVE:g} mg/L or returning to baseline",
         "confirmed_by": "Automated",
         "met": prealbumin_ok, "observed": prealbumin[-1][1] if prealbumin else None},
        {"criterion": "Diet fully advanced", "signal": "Dietitian oral intake assessment",
         "threshold": "Full oral diet tolerated without restriction",
         "confirmed_by": "Dietitian — mandatory",
         "met": bool(clinical.get("full_oral_diet_tolerated")),
         "observed": clinical.get("full_oral_diet_tolerated", None)},
        {"criterion": "Glucose controlled", "signal": "Fasting glucose",
         "threshold": f"Fasting glucose < {GATE_FASTING_GLUCOSE_BELOW:g} mg/dL on "
                      f"{GATE_FASTING_GLUCOSE_READINGS} consecutive readings",
         "confirmed_by": "Automated",
         "met": glucose_ok, "observed": fasting[-GATE_FASTING_GLUCOSE_READINGS:] or None},
        {"criterion": "HMM phase confirmation", "signal": "Phase estimator",
         "threshold": f"HMM posterior > {GATE_REMODELLING_POSTERIOR:.0%} probability of Remodelling",
         "confirmed_by": "Automated",
         "met": phase_ok, "observed": remodelling},
        {"criterion": "Physician discharge sign-off", "signal": "Clinical notes",
         "threshold": "Attending physician confirms wound healing satisfactory",
         "confirmed_by": "Physician — mandatory",
         "met": bool(clinical.get("physician_discharge_signoff")),
         "observed": clinical.get("physician_discharge_signoff", None)},
    ]

    blocking = [c["criterion"] for c in criteria if not c["met"]]
    return {
        "phase2_unlocked": not blocking,
        "criteria": criteria,
        "blocking": blocking,
        "rationale": ("All section 10 criteria met — Phase 2 may activate."
                      if not blocking else
                      "Phase 2 remains locked. The Phase 2 policy optimises for "
                      "biomarkers and may recommend caloric restriction, which is "
                      "harmful during active wound healing."),
        "source": "doc section 10",
    }


# ===========================================================================
# 5. THE FULL FLOW — section 8, end to end
# ===========================================================================

def run_phase1(timeline: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
    """Run every stage of section 8's flow in order and return one bundle.

    Exists so that the four layers run exactly once each, in the right order,
    with each receiving the previous results rather than recomputing them. Called
    standalone, Layer 3 would run Layers 1 and 2 itself and Layer 4 would run all
    three — the same HMM fitted three times over. This is also the single place
    the chaining is defined, so LangGraph can later wrap this one function rather
    than reproducing the sequence.
    """
    # Validate once, here, so a malformed timeline is rejected before any layer
    # runs rather than failing partway through a four-stage pipeline.
    validate_days(timeline)

    # CONTINUOUS — wearable stream and rolling features (section 8, every 15 min)
    features = run_continuous(timeline)

    # ON THRESHOLD BREACH — computed before the summary, because these escalate
    # immediately and must never be gated behind the 06:00 consolidation.
    breaches = check_threshold_breaches(timeline)

    # PERIODIC — on each lab result: HMM posterior, GP deviations, gap scoring
    layer1 = estimate_phase(timeline)
    layer2 = detect_deviations(timeline, layer1)
    layer3 = score_nutrition(timeline, layer1, layer2)

    # DAILY 06:00 — consolidate and draft the physician alert
    layer4 = generate_alert(timeline, layer1, layer2, layer3, use_llm=use_llm)

    gate = evaluate_transition_gate(timeline, layer1)

    escalate = sorted(set(layer2.get("escalate_to", []))
                      | set(layer3.get("escalate_to", []))
                      | {who for b in breaches for who in b["escalate_to"]})

    return {
        "patient_ref": timeline.get("patient_ref"),
        "patient": timeline.get("patient", {}),
        "days_post_op": layer1["days_post_op"],
        "summary": {
            "current_phase": layer1["current_phase"],
            "phase_confidence": layer1["phase_confidence"],
            "healing_status": layer2["healing_status"],
            "nss": layer3["nss"],
            "deviation_count": layer2["deviation_count"],
            "immediate_breaches": len(breaches),
            "phase2_unlocked": gate["phase2_unlocked"],
            "escalate_to": escalate,
        },
        "continuous": {"summary_time": DAILY_SUMMARY_TIME, "feature_engine": features},
        "immediate_escalations": breaches,
        "layer1_phase": layer1,
        "layer2_deviations": layer2,
        "layer3_nutrition": layer3,
        "layer4_alert": layer4,
        "transition_gate": gate,
        "model_version": MODEL_VERSION,
    }


# ===========================================================================
# 5b. RECIPE SELECTION
# ===========================================================================
# Phase 1 selects its own recipes, the same way postop_daily_recommendation_
# service_v1 does for Phase 2: the service reads LangGraph state for the
# patient's body and restrictions, queries recipe_pool, and returns the result
# inside its own bundle. The LangGraph app therefore needs no pool credentials,
# no RECIPE_POOL_MODULE_PATH, and no second database — it just stores what
# comes back. It also means this dependency lives in the container that
# actually uses it, rather than a laptop path that would not exist in ECS.

def _select_recipes(payload: Dict[str, Any],
                    bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the day's 9 recipes. Never fatal — a thin pool is not a failure."""
    if not payload.get("include_recipes", True):
        return {"status": "disabled", "reason": "include_recipes=false"}
    user_id, date = payload.get("user_id"), payload.get("date")
    if not user_id or not date:
        return {"status": "skipped",
                "reason": "no user_id/date supplied; recipe selection needs the "
                          "LangGraph profile for body and dietary restrictions"}
    try:
        from phase1_recipe_pool import fetch_langgraph_state, select_phase1_recipes
        state = fetch_langgraph_state(user_id, date)
        # Recently-served recipe ids, resolved upstream by LangGraph's history
        # walk. Absent when the service is driven directly, in which case the
        # pool simply has nothing to exclude.
        excluded = payload.get("excluded_recipe_ids")
        return select_phase1_recipes(state=state, phase1_bundle=bundle,
                                     user_id=user_id, date=date,
                                     excluded_recipe_ids=excluded)
    except Exception as exc:
        LOG.warning("recipe selection failed for %s @ %s: %s", user_id, date, exc)
        return {"status": "error", "reason": str(exc), "days": []}


# ===========================================================================
# 6. TERMINAL OUTPUT
# ===========================================================================

def render(result: Dict[str, Any]) -> str:
    """One screen a clinician could actually read on a ward round.

    Ordered by urgency rather than by layer: the immediate escalations come
    first, because section 8 says they do not wait for the summary, and a
    rendering that buried them under the phase estimate would undo that.
    """
    s = result["summary"]
    lines = ["=" * 78,
             " POST-OP PHASE 1 — FULL SYSTEM FLOW",
             "=" * 78,
             f" Patient        {result.get('patient_ref') or '-'}",
             f" Surgery        {(result.get('patient') or {}).get('surgery_type') or '-'}",
             f" Day post-op    {result['days_post_op']}",
             ""]

    if result["immediate_escalations"]:
        lines.append(" ** IMMEDIATE ESCALATION — does not wait for the daily summary **")
        for b in result["immediate_escalations"]:
            lines.append(f"    day {b['day']:<3} {b['trigger']}: {b['detail']}")
            lines.append(f"           -> {', '.join(b['escalate_to'])}")
        lines.append("")

    lines += [" SUMMARY",
              f"   Layer 1  phase              {s['current_phase']} ({s['phase_confidence']:.0%})",
              f"   Layer 2  healing status     {s['healing_status']}  ({s['deviation_count']} deviations)",
              f"   Layer 3  sufficiency (NSS)  {s['nss'] if s['nss'] is not None else 'n/a'}",
              f"   Layer 4  alert              {result['layer4_alert']['priority'] if result['layer4_alert']['alert_generated'] else 'none generated'}",
              f"   Escalate to                 {', '.join(s['escalate_to']) or '—'}",
              ""]

    if result["layer4_alert"]["alert_generated"]:
        lines += [" PHYSICIAN ALERT", ""]
        lines += ["   " + line for line in result["layer4_alert"]["narrative"].splitlines()]
        lines += ["", f"   (composed by {result['layer4_alert']['source']})", ""]

    gate = result["transition_gate"]
    lines.append(f" PHASE 1 -> PHASE 2 GATE   {'UNLOCKED' if gate['phase2_unlocked'] else 'LOCKED'}")
    for c in gate["criteria"]:
        mark = "PASS" if c["met"] else "BLOCK"
        observed = "—" if c["observed"] is None else c["observed"]
        lines.append(f"   {mark:<6} {c['criterion']:<30} {c['threshold']}")
        lines.append(f"          observed: {observed}   ({c['confirmed_by']})")
    lines += ["", f"   {gate['rationale']}", "", f" {result['model_version']}"]
    return "\n".join(lines)


# ===========================================================================
# 7. SERVICE
# ===========================================================================

try:
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    class Phase1Request(BaseModel):
        """The whole Phase 1 contract in one object.

        This is the shape LangGraph will eventually post: patient profile, the
        day records, optional clinician overrides. No per-layer results, because
        the orchestrator computes all of them.
        """
        model_config = ConfigDict(extra="allow")
        patient_ref: Optional[str] = None
        patient: Dict[str, Any] = Field(default_factory=dict)
        days: List[Dict[str, Any]] = Field(default_factory=list)
        clinician_phase_overrides: List[Dict[str, Any]] = Field(default_factory=list)
        use_llm: bool = True
        # Supplied by the LangGraph wrapper. With them this service also selects
        # the day's 9 recipes, mirroring how the daily-recommendation service
        # does its own pool selection for Phase 2. Without them the clinical
        # assessment still runs — the orchestrator stays usable standalone.
        user_id: Optional[str] = None
        date: Optional[str] = None
        include_recipes: bool = True

    app = FastAPI(title="Post-op Phase 1 - Orchestrator", version=MODEL_VERSION)

    def _token() -> Optional[str]:
        """Same token fallbacks as every other service in this repo."""
        for name in ("POSTOP_PHASE1_SERVICE_API_KEY", "FOODHAK_API_TOKEN", "LANGGRAPH_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    def _llm_key_present() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    def _llm_sdk_installed() -> bool:
        import importlib.util
        return importlib.util.find_spec("anthropic") is not None

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Reports the whole pipeline's composition, so one call verifies which
        layers a deployment is actually running."""
        return {"status": "ok", "service": "postop-phase1-orchestrator",
                "model_version": MODEL_VERSION,
                "layers": ["layer1_phase_estimator", "layer2_deviation_detector",
                           "layer3_nutrition_gap", "layer4_physician_alert"],
                "flow": ["continuous_15min", "threshold_breach", "periodic_on_lab",
                         f"daily_{DAILY_SUMMARY_TIME}", "transition_gate"],
                # Split, for the same reason Layer 4 splits it: a key set with
                # the SDK missing used to report "configured" and then silently
                # fall back to the deterministic alert.
                "llm_configured": _llm_key_present() and _llm_sdk_installed(),
                "llm_api_key_present": _llm_key_present(),
                "llm_sdk_installed": _llm_sdk_installed(),
                "auth_required": _token() is not None}

    @app.post("/postop/phase1/run")
    async def run(
        request: Phase1Request,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """The single end-to-end endpoint. Auth and error translation only."""
        expected = _token()
        if expected:
            received = x_api_key
            if authorization and authorization.lower().startswith("bearer "):
                received = authorization[7:].strip()
            if received != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing service token")
        try:
            payload = request.model_dump()
            bundle = run_phase1(payload, use_llm=payload.get("use_llm", True))
            bundle["recommended_recipes"] = _select_recipes(payload, bundle)
            return bundle
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/postop/phase1/transition-gate")
    async def gate(
        request: Phase1Request,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """The gate on its own.

        Separate from /run because Phase 2 needs to ask "may I start?" without
        paying for the full pipeline, and because that question will be asked by
        a different caller than the one drafting ward-round alerts.
        """
        expected = _token()
        if expected:
            received = x_api_key
            if authorization and authorization.lower().startswith("bearer "):
                received = authorization[7:].strip()
            if received != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing service token")
        try:
            payload = request.model_dump()
            return evaluate_transition_gate(payload, estimate_phase(payload))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

except ImportError as exc:  # the CLI must still work without FastAPI installed
    _MISSING_DEPENDENCY = (
        f"The HTTP service is disabled: {exc}. "
        "Install the service dependencies with:  pip install -r requirements.txt"
    )
    print(f"phase1_daily_recommendation: {_MISSING_DEPENDENCY}\nThe command-line tool still works.",
          file=sys.stderr)

    async def app(scope, receive, send):  # type: ignore[misc]
        """Stand-in so an ASGI server reports the real cause."""
        if scope.get("type") != "http":
            return
        body = json.dumps({"detail": _MISSING_DEPENDENCY}).encode()
        await send({"type": "http.response.start", "status": 503,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


# ===========================================================================
# 8. CLI
# ===========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the whole flow from a timeline file."""
    parser = argparse.ArgumentParser(description="Run the full Phase 1 flow.")
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--no-llm", action="store_true",
                        help="Force the deterministic alert even if a key is set")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    timeline = json.loads(args.timeline.read_text())
    try:
        result = run_phase1(timeline, use_llm=not args.no_llm)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
