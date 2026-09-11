"""Run Phase 1 once per post-op day, the way production will call it.

A single run over a whole timeline answers "where is this patient now". It
cannot show the thing a demo needs to show: that the answer *changed*, and on
which morning. The section 10 gate stepping from six blockers to none, a
deviation appearing on day 8, the HMM crossing into Remodelling — those are all
statements about a sequence of runs, not about one.

So this replays: for each day D the layers see days 0..D and nothing later,
which is exactly what `run_daily_simulation.py` sends to LangGraph one calendar
day at a time. No state is carried between days; Phase 1 is stateless per call
and the prefix is the whole input.

The cost is quadratic — day 41 refits the HMM over 42 days — and a six-week
patient takes about half a minute. That is why the caller caches it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import sim_paths  # noqa: F401
from phase1_daily_recommendation import run_phase1


def day_numbers(timeline: Dict[str, Any]) -> List[int]:
    return [int(d.get("day", 0)) for d in timeline.get("days", [])]


def as_of(timeline: Dict[str, Any], day: int) -> Dict[str, Any]:
    """The timeline as it stood on the morning of post-op day `day`.

    Everything after `day` is dropped rather than blanked. A field not yet
    recorded must be absent, because absent is what the layers treat as "not
    measured" — blanking it to null would be a different claim. It is a prefix,
    never a filter: a replay that leaked a later day would make every finding
    look prescient, and that is the hardest kind of wrong to notice.
    """
    days = [d for d in timeline.get("days", []) if int(d.get("day", 0)) <= day]
    return {**timeline, "days": days}


def _meals_logged(timeline: Dict[str, Any], day: int) -> Optional[int]:
    """How many meals the dietitian logged that day.

    A generated patient records this directly. A fixture does not — it carries a
    24-hour total with no meal structure — so all that can be said there is
    whether anything was logged at all, and None is the honest answer for how
    many. Guessing three from the presence of a block would put a number on
    screen that no file contains.
    """
    sim = timeline.get("_simulation") or {}
    meals = sim.get("meals_by_day") or {}
    if meals:
        logged = meals.get(day, meals.get(str(day)))
        return len(logged) if logged is not None else None
    return None


def _record(day: int, bundle: Dict[str, Any],
            timeline: Dict[str, Any]) -> Dict[str, Any]:
    """The one-line-per-day view. The full bundle rides along for the tabs."""
    summary = bundle["summary"]
    gate = bundle["transition_gate"]
    layer2 = bundle["layer2_deviations"]
    layer4 = bundle["layer4_alert"]
    return {
        "day": day,
        # The timeline as it stood that morning, so a tab rendering this day
        # cannot accidentally read a later one.
        "timeline": timeline,
        "meals_logged": _meals_logged(timeline, day),
        "phase": summary["current_phase"],
        "confidence": summary["phase_confidence"],
        "posterior": bundle["layer1_phase"]["phase_posterior"],
        "healing_status": summary["healing_status"],
        "nss": summary["nss"],
        "deviation_count": summary["deviation_count"],
        "deviations": [d["deviation"] for d in layer2["deviations"]],
        "worst_severity": (layer2["deviations"][0]["severity"]
                           if layer2["deviations"] else None),
        "immediate_breaches": summary["immediate_breaches"],
        "not_assessed": list(layer2.get("checks_not_assessed") or []),
        "escalate_to": list(summary["escalate_to"]),
        "phase2_unlocked": gate["phase2_unlocked"],
        "gate_blocking": list(gate["blocking"]),
        "gate_blocking_count": len(gate["blocking"]),
        # Layer 4 now writes a routine summary on a quiet day rather than
        # nothing, so "alert" and "summary" are different states and the arc
        # has to keep them apart.
        "narrative_kind": layer4.get("narrative_kind")
                          or ("alert" if layer4.get("alert_generated") else None),
        "alert_priority": layer4.get("priority"),
        "bundle": bundle,
    }


def replay(timeline: Dict[str, Any], *, use_llm: bool = False,
           progress: Optional[Callable[[float, str], None]] = None,
           since: Optional[int] = None,
           reuse: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every post-op day in the timeline, judged as of that morning.

    `since` and `reuse` make an edit cheap. Day D is judged on days 0..D, so
    changing what day K recorded cannot alter any day before K — those records
    are lifted from `reuse` unchanged and only D >= K is recomputed. Editing the
    last day of a six-week stay then costs one run instead of forty-two.

    It is an optimisation, not a shortcut: the days that are recomputed are
    recomputed in full, from a timeline that already carries the edit.
    """
    numbers = day_numbers(timeline)
    days: List[Dict[str, Any]] = []
    total = len(numbers) or 1

    kept = {r["day"]: r for r in (reuse or {}).get("days", [])} if reuse else {}
    for index, day in enumerate(numbers):
        if since is not None and day < since and day in kept:
            days.append(kept[day])
            continue
        if progress:
            progress((index + 1) / total, f"Day {day} of {numbers[-1]}")
        sliced = as_of(timeline, day)
        days.append(_record(day, run_phase1(sliced, use_llm=use_llm), sliced))

    # The morning each thing first became true. A deviation stays in the list on
    # every later day, so "when did this start" is the first day it appears —
    # which is the question a physician actually asks.
    first_flagged: Dict[str, int] = {}
    for record in days:
        for name in record["deviations"]:
            first_flagged.setdefault(name, record["day"])

    phase_first_seen: Dict[str, int] = {}
    for record in days:
        phase_first_seen.setdefault(record["phase"], record["day"])

    # Sticky by observation, not by rule: once every criterion holds the gate
    # stays open on this patient. Reported as the first day it opened, and
    # separately whether it ever closed again, because a gate that reopens and
    # recloses is a finding rather than a detail.
    opened = [r["day"] for r in days if r["phase2_unlocked"]]
    after_open = [r for r in days if opened and r["day"] >= opened[0]]

    return {
        "patient_ref": timeline.get("patient_ref"),
        "day_numbers": numbers,
        "days": days,
        "first_flagged": first_flagged,
        "phase_first_seen": phase_first_seen,
        "gate_opened_on": opened[0] if opened else None,
        "gate_stayed_open": bool(opened) and all(r["phase2_unlocked"]
                                                 for r in after_open),
        "used_llm": bool(use_llm),
    }


def by_day(arc: Dict[str, Any], day: int) -> Dict[str, Any]:
    """One day's record, or the last one at or before it."""
    candidates = [r for r in arc["days"] if r["day"] <= day]
    return candidates[-1] if candidates else arc["days"][0]


def arc_frame(arc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per day, for the timeline table and charts."""
    return [{
        "Day": r["day"],
        "Phase": r["phase"].title(),
        "Confidence": r["confidence"],
        "Status": r["healing_status"].replace("_", " ").title(),
        "NSS": r["nss"],
        "Deviations": r["deviation_count"],
        "Worst": r["worst_severity"] or "",
        "Breaches": r["immediate_breaches"],
        "Not assessed": len(r["not_assessed"]),
        "Gate blockers": r["gate_blocking_count"],
        "Phase 2": "open" if r["phase2_unlocked"] else "locked",
        "Layer 4": (r["narrative_kind"] or "").replace("_", " "),
        "Meals logged": r["meals_logged"],
    } for r in arc["days"]]


def posterior_frame(arc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The HMM posterior on each morning — what the model believed that day.

    Deliberately not Layer 1's own `daily` block. That is one run's smoothed
    view of the whole stay; this is the sequence of end-of-day beliefs, which is
    what a clinician was actually told each morning.
    """
    return [{"Day": r["day"], "Phase": phase, "Posterior": value}
            for r in arc["days"] for phase, value in r["posterior"].items()]


def events(arc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The mornings something changed. The demo's narrative spine."""
    out: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None

    for record in arc["days"]:
        day = record["day"]
        if previous is None:
            out.append({"day": day, "event": "Monitoring starts",
                        "detail": f"First assessment — {record['phase'].title()} "
                                  f"at {record['confidence']:.0%} confidence"
                                  if record["confidence"] is not None else
                                  "First assessment"})
        else:
            if record["phase"] != previous["phase"]:
                out.append({
                    "day": day, "event": "Phase change",
                    "detail": f"{previous['phase'].title()} → "
                              f"{record['phase'].title()}"})
            if record["healing_status"] != previous["healing_status"]:
                out.append({
                    "day": day, "event": "Healing status",
                    "detail": f"{previous['healing_status'].replace('_', ' ').title()} "
                              f"→ {record['healing_status'].replace('_', ' ').title()}"})
            new = [d for d in record["deviations"] if d not in previous["deviations"]]
            for name in new:
                out.append({"day": day, "event": "Deviation flagged",
                            "detail": name.replace("_", " ").capitalize()})
            cleared = [d for d in previous["deviations"] if d not in record["deviations"]]
            for name in cleared:
                out.append({"day": day, "event": "Deviation cleared",
                            "detail": name.replace("_", " ").capitalize()})
            if record["gate_blocking_count"] != previous["gate_blocking_count"]:
                resolved = [c for c in previous["gate_blocking"]
                            if c not in record["gate_blocking"]]
                regained = [c for c in record["gate_blocking"]
                            if c not in previous["gate_blocking"]]
                if resolved:
                    out.append({"day": day, "event": "Gate criterion met",
                                "detail": ", ".join(resolved)
                                          + f" — {record['gate_blocking_count']} left"})
                if regained:
                    out.append({"day": day, "event": "Gate criterion lost",
                                "detail": ", ".join(regained)})
            if record["phase2_unlocked"] and not previous["phase2_unlocked"]:
                out.append({"day": day, "event": "Phase 2 unlocked",
                            "detail": "All seven section 10 criteria met"})
            if previous["phase2_unlocked"] and not record["phase2_unlocked"]:
                out.append({"day": day, "event": "Phase 2 re-locked",
                            "detail": ", ".join(record["gate_blocking"])})
            if record["narrative_kind"] != previous["narrative_kind"]:
                out.append({"day": day, "event": "Layer 4",
                            "detail": f"{(previous['narrative_kind'] or 'nothing')} "
                                      f"→ {record['narrative_kind'] or 'nothing'}"})
        previous = record
    return out
