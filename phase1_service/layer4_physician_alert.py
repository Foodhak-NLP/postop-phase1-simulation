"""
Post-op Phase 1 — Layer 4: Physician Alert Layer.

Implements section 7 of Phase_1_Wound_Recovery.docx.

    "The LLM receives a structured JSON payload from Layer 3 and translates it
     into a prioritised, explainable clinical alert. It performs no clinical
     reasoning and cannot modify any value produced by Layers 1-3. Its only
     function is language."

That sentence is the entire design constraint, and section 7 lists five things
the LLM is explicitly forbidden from doing. Those are not prompt instructions
here — a prompt is a request, not a guarantee. They are enforced in code, after
generation, against the structured payload. A narrative that breaks one is
rejected and the deterministic text is used instead.

The alert is also generated without an LLM at all when no API key is present, so
the system never depends on a network call to tell a physician their patient is
septic.

Run it:

    python3 layer4_physician_alert.py --timeline example_patient.json
    ANTHROPIC_API_KEY=... python3 layer4_physician_alert.py --timeline example_patient.json
    uvicorn layer4_physician_alert:app --port 8013 --reload
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from layer1_phase_estimator import estimate_phase
from layer2_deviation_detector import detect_deviations
from layer3_nutrition_gap import score_nutrition

MODEL_VERSION = "phase1-alert-v1"

LLM_MODEL = "claude-sonnet-5"
LLM_MAX_TOKENS = 1200

# Section 7: "What the LLM is explicitly forbidden from doing".
FORBIDDEN = (
    "Modifying any numerical value output by Layers 1-3",
    "Suggesting a specific nutritional intervention not flagged by the Gap Engine",
    "Estimating wound status from self-report or vital signs directly",
    "Generating an alert when the structured payload contains no flags",
    "Providing reassurance when the deviation detector has flagged a concern",
)

# Words that would constitute reassurance. Forbidden outright whenever Layer 2
# has flagged anything, because a physician skimming an alert must not read
# comfort into a page that exists because something is wrong.
REASSURANCE_PATTERNS = (
    r"\bno cause for concern\b", r"\breassur\w*", r"\bnothing to worry\b",
    r"\bappears? (?:to be )?(?:normal|fine|well)\b", r"\bnot concerning\b",
    r"\bunremarkable\b", r"\bstable and improving\b",
)

NO_IMAGING_STATEMENT = (
    "No wound imaging available. Phase estimate is inferred from signal "
    "trajectory. Clinical wound assessment recommended to corroborate."
)


# ===========================================================================
# 1. THE STRUCTURED PAYLOAD — section 7.1
#
# Built here rather than inside the LLM call so that exactly one object is both
# the model's input and the validator's reference. If they could diverge, the
# guardrail below would be checking the narrative against something other than
# what produced it.
# ===========================================================================

def build_payload(layer1: Dict[str, Any], layer2: Dict[str, Any],
                  layer3: Dict[str, Any], patient_ref: Optional[str],
                  surgery_type: Optional[str]) -> Dict[str, Any]:
    """Assemble section 7.1's payload from the three upstream layers.

    Field names follow the document's example exactly, because this object is the
    contract between the clinical layers and the language layer — and, later,
    between Phase 1 and whatever consumes its alerts.
    """
    return {
        "patient_id": patient_ref,
        "day_post_op": layer1.get("days_post_op"),
        "surgery_type": surgery_type,
        "current_phase": layer1.get("current_phase"),
        "phase_confidence": layer1.get("phase_confidence"),
        "healing_status": layer2.get("healing_status"),
        "deviations": [
            {"signal": d["signal"], "finding": d["finding"], "severity": d["severity"]}
            for d in layer2.get("deviations", [])
        ],
        "nutritional_gaps": [
            {"nutrient": g["label"], "target": f"{g['target']}{g['unit']}",
             "actual": f"{g['actual']}{g['unit']}", "gap_pct": g["gap_pct"],
             "priority": g["priority"]}
            for g in layer3.get("gaps", [])
        ],
        "interaction_flags": [
            {"finding": i["alert_text"], "interpretation": i["clinical_interpretation"],
             "severity": i["severity"]}
            for i in layer3.get("interactions", [])
        ],
        "nss": layer3.get("nss"),
        "checks_not_assessed": layer2.get("checks_not_assessed", []),
        "escalation_required": bool(layer2.get("escalation_required")
                                    or layer3.get("escalation_required")),
        "escalate_to": sorted(set(layer2.get("escalate_to", []))
                              | set(layer3.get("escalate_to", []))),
    }


# ===========================================================================
# 2. THE GUARDRAILS — section 7's forbidden list, enforced in code
# ===========================================================================

def payload_numbers(payload: Dict[str, Any]) -> Set[float]:
    """Every number the layers actually produced, as raw floats.

    The reference set for the first and most important prohibition. Built by
    walking the payload rather than listing fields by hand, so a field added
    later is protected automatically instead of silently becoming unguarded.

    Kept as raw floats rather than formatted strings: rounding here once cost a
    real bug, where a confidence of 0.81 was rounded to 0.8 before the percentage
    expansion and the validator then rejected the phrase "81%" that the system's
    own alert had written. Tolerance belongs at comparison time, not in storage.
    """
    found: Set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            found.add(float(node))
        elif isinstance(node, str):
            for token in re.findall(r"-?\d+\.?\d*", node):
                try:
                    found.add(float(token))
                except ValueError:
                    continue

    walk(payload)
    return found


def number_is_allowed(token: float, allowed: Set[float]) -> bool:
    """Whether a number in the prose traces to one in the payload.

    Accepts the value itself, the value as a percentage, and the value as a
    fraction, because prose legitimately writes 0.81 as "81%". Tolerance is 1%
    relative or 0.05 absolute, whichever is larger, so ordinary rounding is
    permitted while a genuinely invented figure is not.
    """
    for value in allowed:
        for candidate in (value, value * 100.0, value / 100.0):
            if abs(candidate - token) <= max(0.05, abs(candidate) * 0.01):
                return True
    return False


def validate_narrative(text: str, payload: Dict[str, Any]) -> List[str]:
    """Check a generated narrative against section 7's forbidden list.

    Returns the violations found. Runs after generation rather than relying on
    the prompt, because a prompt is a request and this is a safety property: the
    physician must be able to trust that every number on screen came from a
    clinical layer, not from a language model.
    """
    violations: List[str] = []
    allowed = payload_numbers(payload)

    # 1. Modifying any numerical value output by Layers 1-3.
    invented = [token for token in re.findall(r"-?\d+\.?\d*", text)
                if not number_is_allowed(float(token), allowed)]
    # Day numbers and small integers appear naturally in prose ("Day 7", "3 of 7").
    invented = [t for t in invented if not (float(t).is_integer() and abs(float(t)) <= 31)]
    if invented:
        violations.append(
            f"Narrative contains numbers absent from the structured payload: "
            f"{', '.join(sorted(set(invented)))}")

    # 5. Providing reassurance when the deviation detector has flagged a concern.
    if payload["deviations"]:
        for pattern in REASSURANCE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                violations.append(
                    f"Narrative offers reassurance ('{pattern}') while deviations are flagged")
                break

    # 3. Estimating wound status directly. The absence of imaging must be stated.
    if "wound imaging" not in text.lower():
        violations.append("Narrative omits the mandatory no-wound-imaging statement")

    return violations


# ===========================================================================
# 3. THE DETERMINISTIC ALERT — the default, not the fallback
#
# Written first and used whenever no LLM is configured or the LLM output fails
# validation. A physician alert must not depend on a network call, and every
# sentence here is assembled from payload fields, so it cannot invent anything.
# ===========================================================================

def deterministic_narrative(payload: Dict[str, Any]) -> str:
    """Compose the alert from the payload alone, in the shape of section 7.2."""
    if not payload["deviations"] and not payload["nutritional_gaps"]:
        # Forbidden item 4: no alert when the payload contains no flags.
        return ""

    worst = "CRITICAL" if any(d["severity"] == "CRITICAL" for d in payload["deviations"]) else \
            "HIGH" if any(d["severity"] == "HIGH" for d in payload["deviations"]) else "REVIEW"
    who = " + ".join(payload["escalate_to"]) or "Physician"

    lines = [
        f"ALERT — Day {payload['day_post_op']} Post-Op | {payload['patient_id']} | "
        f"{who} Review Required | {worst}",
        "",
        f"Healing status: {payload['healing_status']}. Patient estimated in "
        f"{payload['current_phase']} phase (confidence "
        f"{payload['phase_confidence']:.0%})."
        if payload["phase_confidence"] is not None else
        f"Healing status: {payload['healing_status']}.",
    ]

    if payload["deviations"]:
        lines.append("")
        for d in payload["deviations"]:
            lines.append(f"[{d['severity']}] {d['signal']}: {d['finding']}")

    if payload["nutritional_gaps"]:
        lines.append("")
        for g in payload["nutritional_gaps"]:
            lines.append(
                f"[{g['priority']}] {g['nutrient']} intake is {g['gap_pct']:.0f}% below the "
                f"{payload['current_phase'].lower()} phase target "
                f"(actual {g['actual']} vs target {g['target']}).")

    if payload["interaction_flags"]:
        lines.append("")
        for i in payload["interaction_flags"]:
            lines.append(f"[{i['severity']}] {i['finding']} {i['interpretation']}.")

    if payload["nss"] is not None:
        lines += ["", f"Overall Nutritional Sufficiency Score: {payload['nss']}."]

    if payload["checks_not_assessed"]:
        lines += ["", "Checks that could not be run on the available data: "
                      + ", ".join(c.replace("_", " ") for c in payload["checks_not_assessed"])
                      + ". Absence of these findings is not evidence of their absence."]

    lines += ["", f"Note: {NO_IMAGING_STATEMENT}"]
    return "\n".join(lines)


# ===========================================================================
# 4. THE LLM PATH — language only
# ===========================================================================

LLM_SYSTEM_PROMPT = """You translate a structured clinical payload into a physician alert.

You perform NO clinical reasoning. You are forbidden from:
  1. Modifying, rounding differently, or inventing ANY numerical value.
  2. Suggesting a nutritional intervention that the payload does not already flag.
  3. Estimating wound status from vital signs or self-report.
  4. Producing an alert when the payload contains no flags.
  5. Offering reassurance when deviations are present.

Use only facts in the payload. Every number in your output must appear in the
payload. Always include the statement that no wound imaging is available.
Write in clinical prose: priority header, healing status, the findings in
severity order, the nutritional gaps, then the recommended actions."""


def llm_narrative(payload: Dict[str, Any]) -> Optional[str]:
    """Ask Claude to phrase the alert, or return None if unavailable.

    Isolated in its own function with every failure returning None, so that a
    missing key, a network error or an API change degrades to the deterministic
    text rather than producing no alert at all.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": json.dumps(payload, indent=2)}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()
    except Exception:
        return None


# ===========================================================================
# 5. THE ENTRY POINT
# ===========================================================================

def daily_summary_narrative(payload: Dict[str, Any]) -> str:
    """A routine daily note for a day with nothing flagged.

    Section 7's forbidden item 4 bars an ALERT when the payload carries no
    flags, and that stays enforced — manufacturing alarm on a quiet day is how
    a ward learns to ignore alerts. But "no alert" was being delivered as no
    text at all, which reads to a clinician as though the day was not assessed.

    So a quiet day now gets a SUMMARY instead: same facts, no escalation, and
    explicit about what could not be checked. It states the unassessed checks
    rather than implying everything was verified, which is the honest form of
    reassurance and the only one the guardrails permit.
    """
    # phase_confidence is None whenever Layer 1 could not estimate — a
    # single-day timeline, or one with no usable signals. Formatting it
    # unconditionally crashed the summary for exactly the sparse cases this
    # summary exists to describe.
    confidence = payload.get("phase_confidence")
    confidence_text = (f" (confidence {confidence:.0%})"
                       if isinstance(confidence, (int, float)) else "")
    lines = [
        f"DAILY SUMMARY - Day {payload['day_post_op']} Post-Op | "
        f"{payload['patient_id']} | No escalation required",
        "",
        f"Healing status: {payload['healing_status']}. Patient estimated in "
        f"{payload['current_phase']} phase{confidence_text}.",
        "",
        "No deviations were detected and no nutritional gaps were flagged.",
    ]

    if payload.get("nss") is None:
        lines += [
            "",
            "Nutrition was NOT scored today. No intake was recorded, either in "
            "the request or from meals logged in the app, so the gap engine had "
            "nothing to compare against targets. This is an absence of data, "
            "not evidence of adequate intake.",
        ]
    else:
        lines += ["", f"Normalised Sufficiency Score: {payload['nss']}."]

    not_assessed = payload.get("checks_not_assessed") or []
    if not_assessed:
        lines += ["", "Not assessed today:"]
        lines += [f"- {name.replace('_', ' ').capitalize()}" for name in not_assessed]

    lines += ["", NO_IMAGING_STATEMENT]
    return "\n".join(lines)


def generate_alert(timeline: Dict[str, Any], layer1: Dict[str, Any],
                   layer2: Dict[str, Any], layer3: Dict[str, Any],
                   use_llm: bool = True) -> Dict[str, Any]:
    """Produce the physician alert, with the guardrails applied.

    The order matters: build the payload, try the LLM, validate what it returned,
    and fall back if it broke a rule. The deterministic text is always computed,
    so there is never a path where a validation failure leaves the physician with
    nothing.
    """
    patient = timeline.get("patient") if isinstance(timeline.get("patient"), dict) else {}
    payload = build_payload(
        layer1, layer2, layer3,
        timeline.get("patient_ref"),
        patient.get("surgery_type") or timeline.get("surgery_type"))

    baseline = deterministic_narrative(payload)

    # Forbidden item 4 still holds: no flags, no ALERT. What changed is that a
    # quiet day is no longer silent — it returns a daily summary instead, so
    # "nothing was wrong" and "nothing was assessed" stop looking identical.
    # alert_generated stays False, so nothing downstream escalates on it.
    if not baseline:
        return {"alert_generated": False,
                "summary_generated": True,
                "narrative_kind": "daily_summary",
                "priority": "ROUTINE",
                "escalate_to": [],
                "reason": "No deviations and no nutritional gaps — section 7 forbids "
                          "an alert when the payload contains no flags, so a routine "
                          "daily summary was written instead.",
                "structured_payload": payload,
                "narrative": daily_summary_narrative(payload),
                "source": "deterministic",
                "guardrail_violations": [],
                "guardrails_enforced": list(FORBIDDEN),
                "model_version": MODEL_VERSION}

    narrative, source, violations = baseline, "deterministic", []
    if use_llm:
        candidate = llm_narrative(payload)
        if candidate:
            violations = validate_narrative(candidate, payload)
            if violations:
                source = "deterministic_after_guardrail_rejection"
            else:
                narrative, source = candidate, "llm"

    return {
        "alert_generated": True,
        "summary_generated": True,
        "narrative_kind": "alert",
        "priority": ("CRITICAL" if any(d["severity"] == "CRITICAL" for d in payload["deviations"])
                     else "HIGH" if any(d["severity"] == "HIGH" for d in payload["deviations"])
                     else "REVIEW"),
        "escalate_to": payload["escalate_to"],
        "narrative": narrative,
        "source": source,
        "guardrail_violations": violations,
        "guardrails_enforced": list(FORBIDDEN),
        "structured_payload": payload,
        "model_version": MODEL_VERSION,
    }


# ===========================================================================
# 6. TERMINAL OUTPUT
# ===========================================================================

def render(result: Dict[str, Any]) -> str:
    """Show the alert as the ward round would see it, plus how it was produced.

    The provenance line matters as much as the text: a reviewer needs to know
    whether a physician read machine-composed prose or a template, and whether
    the language model was rejected for breaking a rule.
    """
    lines = ["=" * 78,
             " POST-OP PHASE 1 - LAYER 4 : PHYSICIAN ALERT",
             "=" * 78]

    if not result["alert_generated"]:
        lines += ["", " NO ALERT GENERATED", f"   {result['reason']}", ""]
        lines.append(f" {result['model_version']}")
        return "\n".join(lines)

    lines += ["", result["narrative"], "",
              "-" * 78,
              f" priority      {result['priority']}",
              f" escalate to   {', '.join(result['escalate_to']) or '—'}",
              f" composed by   {result['source']}"]

    if result["guardrail_violations"]:
        lines.append(" GUARDRAIL REJECTIONS  (LLM output discarded)")
        for v in result["guardrail_violations"]:
            lines.append(f"   - {v}")

    lines += ["", " GUARDRAILS ENFORCED  (doc section 7)"]
    lines += [f"   - {rule}" for rule in result["guardrails_enforced"]]
    lines += ["", f" {result['model_version']}"]
    return "\n".join(lines)


# ===========================================================================
# 7. SERVICE
# ===========================================================================

try:
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    class AlertRequest(BaseModel):
        """Upstream results are optional so the endpoint works standalone for a
        demo; the orchestrator supplies them so each layer runs exactly once."""
        model_config = ConfigDict(extra="allow")
        patient_ref: Optional[str] = None
        patient: Dict[str, Any] = Field(default_factory=dict)
        days: List[Dict[str, Any]] = Field(default_factory=list)
        clinician_phase_overrides: List[Dict[str, Any]] = Field(default_factory=list)
        layer1: Optional[Dict[str, Any]] = None
        layer2: Optional[Dict[str, Any]] = None
        layer3: Optional[Dict[str, Any]] = None
        use_llm: bool = True

    app = FastAPI(title="Post-op Phase 1 - Layer 4", version=MODEL_VERSION)

    def _token() -> Optional[str]:
        """Service token, with the same fallback names as the other layers so
        deployment configuration stays uniform."""
        for name in ("POSTOP_PHASE1_SERVICE_API_KEY", "FOODHAK_API_TOKEN", "LANGGRAPH_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Reports whether an LLM is actually configured, so a demo does not
        silently show template text believing it came from the model."""
        # Both halves matter. The key alone was reported before, but the
        # `anthropic` import is lazy and its ImportError is swallowed, so a box
        # with the key set and the package missing answered "llm_configured:
        # true" while serving template text to every patient. Report the two
        # separately so a deploy check can tell those cases apart.
        import importlib.util
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_sdk = importlib.util.find_spec("anthropic") is not None
        return {"status": "ok", "service": "postop-phase1-layer4",
                "model_version": MODEL_VERSION,
                "llm_configured": has_key and has_sdk,
                "llm_api_key_present": has_key,
                "llm_sdk_installed": has_sdk,
                "llm_model": LLM_MODEL,
                "guardrails": list(FORBIDDEN),
                "auth_required": _token() is not None}

    @app.post("/postop/phase1/physician-alert")
    async def alert(
        request: AlertRequest,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """Thin wrapper: auth, chaining, error translation. No language or
        clinical logic lives here."""
        expected = _token()
        if expected:
            received = x_api_key
            if authorization and authorization.lower().startswith("bearer "):
                received = authorization[7:].strip()
            if received != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing service token")
        payload = request.model_dump()
        try:
            layer1 = payload.get("layer1") or estimate_phase(payload)
            layer2 = payload.get("layer2") or detect_deviations(payload, layer1)
            layer3 = payload.get("layer3") or score_nutrition(payload, layer1, layer2)
            return generate_alert(payload, layer1, layer2, layer3, use_llm=payload.get("use_llm", True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

except ImportError as exc:  # the CLI must still work without FastAPI installed
    _MISSING_DEPENDENCY = (
        f"The HTTP service is disabled: {exc}. "
        "Install the service dependencies with:  pip install -r requirements.txt"
    )
    print(f"layer4_physician_alert: {_MISSING_DEPENDENCY}\n"
          "The command-line tool still works.", file=sys.stderr)

    async def app(scope, receive, send):  # type: ignore[misc]
        """Stand-in so an ASGI server reports the real cause rather than
        'NoneType is not callable'."""
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
    """Command-line entry, keeping file I/O out of the alert logic."""
    parser = argparse.ArgumentParser(description="Compose the physician alert.")
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--no-llm", action="store_true",
                        help="Force the deterministic alert even if a key is set")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    timeline = json.loads(args.timeline.read_text())
    try:
        layer1 = estimate_phase(timeline)
        layer2 = detect_deviations(timeline, layer1)
        layer3 = score_nutrition(timeline, layer1, layer2)
        result = generate_alert(timeline, layer1, layer2, layer3, use_llm=not args.no_llm)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
