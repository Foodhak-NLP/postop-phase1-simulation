"""One insight and one recommendation per layer, for one day.

Every string here is assembled from that layer's own output. Nothing is
inferred across layers, no number is recomputed, and no recommendation is
written that the layer did not already make — Layer 2's come from each
deviation's `recommended_action`, Layer 3's from `nss_action` and section 6.3's
interactions, Layer 4's from its own escalation list. Where a layer has nothing
to recommend, that is what it says.

The point is that a reader can put a finger on any sentence and find the field
it came from. A summariser that paraphrased would break that, and a summariser
that added clinical advice of its own would be doing the thing section 7 exists
to stop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import sim_paths  # noqa: F401
from layer2_deviation_detector import SEVERITY_ORDER

NO_ACTION = "No action from this layer today."


def _fmt_pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0%}"


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


# ---------------------------------------------------------------------------
# Layer 1 — which phase
# ---------------------------------------------------------------------------

def layer1(bundle: Dict[str, Any]) -> Dict[str, Any]:
    l1 = bundle["layer1_phase"]
    day = bundle["days_post_op"]
    phase = l1["current_phase"].title()
    confidence = l1["phase_confidence"]

    today = next((d for d in reversed(l1["daily"]) if d["day"] == day), None)
    drivers = []
    if today:
        ranked = sorted(today.get("why") or [],
                        key=lambda w: w.get("probability_in_phase") or 0,
                        reverse=True)
        drivers = [f"{w['signal'].replace('_', ' ')} {w['observed']}"
                   for w in ranked[:3]]

    insight = f"Day {day}: **{phase}**, {_fmt_pct(confidence)} confidence."
    if drivers:
        insight += " Driven by " + ", ".join(drivers) + "."
    regression = [t for t in l1["phase_transitions"] if t["regression"]]
    if regression:
        insight += (f" A backward transition on day {regression[-1]['day']} — "
                    "the model has re-entered an earlier phase.")

    actions: List[str] = []
    if l1.get("low_confidence"):
        actions.append("Treat the stage as provisional — there is not enough "
                       "agreement between today's readings to be confident.")
    if regression:
        actions.append("The wound appears to have gone backwards a stage. "
                       "Check it directly — the feeding targets follow from "
                       "this stage.")
    actions.append("This cannot see the wound. The stage is worked out from "
                   "bloods and vitals, so confirm it by looking at the wound.")

    return {
        "insight": insight,
        "recommendation": actions,
        "source": "Layer 1",
        "metrics": {
            "Phase": phase,
            "Confidence": _fmt_pct(confidence),
            "Signals used": f"{l1['signals_observed']['observed']} of "
                            f"{l1['signals_observed']['possible']}",
        },
    }


# ---------------------------------------------------------------------------
# Layer 2 — on track?
# ---------------------------------------------------------------------------

def layer2(bundle: Dict[str, Any]) -> Dict[str, Any]:
    l2 = bundle["layer2_deviations"]
    status = l2["healing_status"].replace("_", " ").title()
    deviations = l2["deviations"]
    not_assessed = l2.get("checks_not_assessed") or []

    if deviations:
        worst = max(deviations,
                    key=lambda d: SEVERITY_ORDER.index(d["severity"]))
        insight = (f"**{status}** — {len(deviations)} deviation"
                   f"{'s' if len(deviations) != 1 else ''} flagged. "
                   f"Worst is {worst['severity'].lower()}: "
                   f"{worst['deviation'].replace('_', ' ')} on day "
                   f"{worst['day']} — {worst['finding']}")
    else:
        insight = (f"**{status}** — no deviation on any check that ran.")

    if not_assessed:
        insight += (f" {len(not_assessed)} check"
                    f"{'s' if len(not_assessed) != 1 else ''} could not run "
                    "and is not reported as clear: "
                    + ", ".join(n.replace("_", " ") for n in not_assessed) + ".")

    actions = _dedupe([d["recommended_action"] for d in deviations])
    if not actions:
        actions = [NO_ACTION]
    if l2.get("critical_checks_not_assessed"):
        actions.insert(0, "An important check could not run — "
                          + ", ".join(n.replace("_", " ") for n in
                                      l2["critical_checks_not_assessed"])
                          + ". Nothing was found because nothing was measured, "
                            "so check it at the bedside.")

    return {
        "insight": insight,
        "recommendation": actions,
        "source": "Layer 2",
        "metrics": {
            "Healing status": status,
            "Deviations": len(deviations),
            "Not assessed": len(not_assessed),
        },
    }


# ---------------------------------------------------------------------------
# Layer 3 — fed correctly?
# ---------------------------------------------------------------------------

def layer3(bundle: Dict[str, Any], meals_logged: Optional[int] = None) -> Dict[str, Any]:
    l3 = bundle["layer3_nutrition"]
    nss = l3["nss"]
    action = l3.get("nss_action") or {}
    gaps = l3.get("gaps") or []
    interactions = l3.get("interactions") or []

    if nss is None:
        insight = ("**No intake was logged**, so nothing could be scored. "
                   "This is unknown, not zero — section 6.2 scores the previous "
                   "24 hours and an unlogged nutrient is left unscored rather "
                   "than counted as starvation.")
    else:
        insight = f"**NSS {nss:.2f}**"
        if meals_logged is not None:
            insight += f" on {meals_logged} of 3 meals logged"
        if gaps:
            worst = sorted(gaps, key=lambda g: -(g.get("gap_pct") or 0))[:3]
            insight += ". Short on " + ", ".join(
                f"{g['label'].lower()} {g['gap_pct']:.0f}%" for g in worst) + "."
        else:
            insight += " — every scored nutrient met its phase target."

    actions: List[str] = []
    if action.get("detail"):
        actions.append(action["detail"])
    actions += [i.get("alert_text", "") for i in interactions]
    actions = _dedupe([a for a in actions if a])
    if not actions:
        actions = [NO_ACTION]

    return {
        "insight": insight,
        "recommendation": actions,
        "source": "Layer 3",
        "metrics": {
            "NSS": "—" if nss is None else f"{nss:.2f}",
            "Escalation": str(action.get("escalation", "none")).replace("_", " "),
            "Nutrients short": len(gaps),
        },
    }


# ---------------------------------------------------------------------------
# Layer 4 — tell the physician
# ---------------------------------------------------------------------------

def layer4(bundle: Dict[str, Any]) -> Dict[str, Any]:
    l4 = bundle["layer4_alert"]
    kind = l4.get("narrative_kind") or (
        "alert" if l4.get("alert_generated") else "nothing")
    escalate = l4.get("escalate_to") or []

    if l4.get("alert_generated"):
        insight = (f"**{l4['priority']} alert** raised. "
                   + (f"Escalates to {', '.join(escalate)}."
                      if escalate else "No escalation target named."))
    else:
        insight = ("**Routine note** — nothing needed attention today, so no "
                   "alert was raised. A note is written anyway, so that a "
                   "quiet day and an unassessed day do not look the same.")

    if escalate:
        actions = [f"Page {who}." for who in escalate]
    else:
        actions = ["No escalation. File the summary with the ward round."]

    from layer4_physician_alert import LLM_MODEL
    written_by = {"llm": LLM_MODEL,
                  "deterministic": "template",
                  "deterministic_after_guardrail_rejection": "template"
                  }.get(l4.get("source", ""), l4.get("source", ""))
    provenance = {
        "llm": f"Phrased by {LLM_MODEL} and checked before being shown.",
        "deterministic": (
            "Written from a fixed template. On a day with nothing to report "
            "the model is not called at all."
            if not l4.get("alert_generated") else
            "Written from a fixed template — the model was not available."),
        "deterministic_after_guardrail_rejection": (
            "The model's wording broke one of the rules and was discarded. "
            "What is shown below is the standard text."),
    }.get(l4.get("source", ""), "")

    return {
        "insight": insight,
        "recommendation": actions,
        "source": "Layer 4",
        "narrative": l4.get("narrative"),
        "provenance": provenance,
        "metrics": {
            "Output": kind.replace("_", " "),
            "Priority": l4.get("priority", "—"),
            "Written by": written_by,
        },
    }


# ---------------------------------------------------------------------------
# The day, in one place
# ---------------------------------------------------------------------------

def overall(bundle: Dict[str, Any],
            meals_logged: Optional[int] = None) -> Dict[str, Any]:
    """The four layers' answers as one line each, plus what to do about them."""
    parts = [layer1(bundle), layer2(bundle),
             layer3(bundle, meals_logged), layer4(bundle)]
    gate = bundle["transition_gate"]

    recommendations = _dedupe(
        [a for part in parts[1:4] for a in part["recommendation"]
         if a != NO_ACTION])
    if not recommendations:
        recommendations = ["Continue the current plan. Nothing this system "
                           "measured requires a change today."]

    return {
        "layers": {"Layer 1": parts[0], "Layer 2": parts[1],
                   "Layer 3": parts[2], "Layer 4": parts[3]},
        "recommendation": recommendations,
        # Phase 2 optimises for biomarkers and can recommend eating less,
        # which is harmful while a wound is still being built. So the gate is
        # reported as what is still outstanding, not as a score.
        "gate": ("Ready for Phase 2 — everything needed has been met and "
                 "signed off."
                 if gate["phase2_unlocked"] else
                 "Not ready for Phase 2 yet. Still outstanding: "
                 f"{', '.join(gate['blocking'])}."),
    }
