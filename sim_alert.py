"""Let the model phrase the quiet-day note as well as the alert.

Layer 4 already writes alerts with `claude-sonnet-5`. What it does *not* do is
call the model on a day with nothing flagged: it returns a deterministic daily
summary and stops. That is a reasonable service default — a routine note must
never depend on a network call — but it means a well patient's Layer 4 tab
never shows the model at all.

So on those days the simulator asks the model to phrase the same summary, from
the same payload, and puts it through the **same guardrails the service applies
to an alert**:

  * no number that is not already in the payload,
  * no intervention the layers did not flag,
  * no claim about the wound itself,
  * the no-imaging statement present.

The fifth prohibition — no reassurance while a deviation is flagged — is not in
play here by construction: this path only runs when nothing is flagged.

If the wording breaks any rule it is discarded and the service's own
deterministic summary is shown, exactly as an alert would be. `alert_generated`
is never touched, so nothing downstream escalates on a quiet day.
"""
from __future__ import annotations

from typing import Any, Dict

import sim_paths  # noqa: F401
from layer4_physician_alert import (LLM_MODEL, llm_narrative,
                                    validate_narrative)

SUMMARY_INSTRUCTION = (
    "\n\nThis patient has NOTHING flagged today. Do not write an alert and do "
    "not imply urgency. Write a short routine ward-round note in the same "
    "format, stating the healing phase, that no deviations were detected, the "
    "nutritional score, and — explicitly — any check that could not be "
    "assessed. Use only the numbers given."
)


def phrase_summary(layer4: Dict[str, Any]) -> Dict[str, Any]:
    """Re-phrase a routine daily note with the model, or leave it alone.

    Takes and returns Layer 4's own result dict, so a caller can drop this in
    without knowing whether it did anything.
    """
    if layer4.get("alert_generated") or not layer4.get("structured_payload"):
        return layer4
    if layer4.get("narrative_kind") != "daily_summary":
        return layer4

    payload = dict(layer4["structured_payload"])
    # The instruction rides in the payload rather than the system prompt: the
    # prompt belongs to the service and this must not reach past its own call.
    payload["note"] = SUMMARY_INSTRUCTION.strip()

    candidate = llm_narrative(payload)
    if not candidate:
        return layer4

    violations = validate_narrative(candidate, layer4["structured_payload"])
    if violations:
        return {**layer4,
                "source": "deterministic_after_guardrail_rejection",
                "guardrail_violations": violations,
                "summary_phrased_by": "template (model wording rejected)"}

    return {**layer4,
            "narrative": candidate,
            "source": "llm",
            "guardrail_violations": [],
            "summary_phrased_by": LLM_MODEL}
