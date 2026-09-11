"""End-to-end Phase 1 simulation, from the terminal.

Every surgery, six weeks each, one full Phase 1 run per post-op day — the same
code path the app drives, without the app. Use it for a screen-share, a
recording, or to diff two runs.

    ../Postop-Phase1/Staging/phase1/bin/python run_simulation.py
    ... run_simulation.py --surgery cabg --weeks 4 --meals 1.5 --days
    ... run_simulation.py --complication ssi --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import sim_insights
import sim_paths  # noqa: F401
import sim_replay
import sim_timeline as T

RULE = "=" * 78


def _fmt(value: Any, width: int = 0) -> str:
    text = "—" if value is None else str(value)
    return text.ljust(width) if width else text


def run_one(surgery: str, *, weeks: int, recovery: str, intake: str,
            diabetic: bool, coverage: float, seed: int,
            day_count: Optional[int] = None) -> Dict[str, Any]:
    timeline = T.build(surgery=surgery, weeks=weeks, days=day_count,
                       recovery=recovery, intake_pattern=intake,
                       diabetic=diabetic, wearable_coverage=coverage,
                       seed=seed)
    return sim_replay.replay(timeline, use_llm=False)


def report_patient(surgery: str, arc: Dict[str, Any], *, show_days: bool) -> None:
    spec = T.SURGERIES[surgery]
    last = arc["days"][-1]
    print()
    print(RULE)
    print(f" {spec['label'].upper()}")
    print(RULE)
    print(f" {spec['note']}")
    print()
    print(f"  Days simulated       {len(arc['days'])}")
    print(f"  Phase reached        {last['phase'].title()} "
          f"({_fmt(last['confidence'])})")
    print(f"  Healing status       {last['healing_status'].replace('_', ' ')}")
    print(f"  Deviations           {len(arc['first_flagged'])} distinct, "
          f"{sum(1 for r in arc['days'] if r['deviation_count'])} days flagged")
    print(f"  Final NSS            {_fmt(last['nss'])}")
    print(f"  Phase 2 gate         "
          + (f"opened on day {arc['gate_opened_on']}"
             if arc["gate_opened_on"] is not None
             else f"never opened — {last['gate_blocking_count']} outstanding"))

    print()
    print("  Phases entered:")
    for phase, day in arc["phase_first_seen"].items():
        print(f"    day {day:>2}   {phase.title()}")

    print()
    print("  The mornings something changed:")
    for item in sim_replay.events(arc):
        print(f"    day {item['day']:>2}   {item['event']:<22} {item['detail']}")

    if show_days:
        print()
        print("  " + "day  phase          conf   status     nss     meals  dev  gate")
        for row in arc["days"]:
            print(f"  {row['day']:>3}  {row['phase'].title():<14} "
                  f"{_fmt(row['confidence']):<6} "
                  f"{row['healing_status'].replace('_', ' '):<10} "
                  f"{_fmt(row['nss']):<7} {_fmt(row['meals_logged']):<6} "
                  f"{row['deviation_count']:>3}  "
                  f"{'open' if row['phase2_unlocked'] else row['gate_blocking_count']}")

    print()
    print("  The last day, layer by layer:")
    view = sim_insights.overall(last["bundle"], meals_logged=last["meals_logged"])
    for name, part in view["layers"].items():
        print(f"    {name} — {_plain(part['insight'])}")
    print()
    print("  Recommended today:")
    for action in view["recommendation"]:
        print(f"    * {_plain(action)}")
    print(f"  {_plain(view['gate'])}")


def _plain(text: str) -> str:
    """Markdown emphasis is noise in a terminal."""
    return text.replace("**", "").replace("`", "")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--surgery", choices=list(T.SURGERY_ORDER),
                        help="One surgery. Default: all of them.")
    parser.add_argument("--weeks", type=int, default=T.MAX_WEEKS,
                        help=f"1-{T.MAX_WEEKS} (default {T.MAX_WEEKS}).")
    parser.add_argument("--days", dest="day_count", type=int,
                        help=f"Days instead of whole weeks, 1-{T.MAX_DAYS}.")
    parser.add_argument("--recovery", choices=list(T.RECOVERY_ORDER),
                        default="textbook",
                        help="How the recovery goes — the axis the layers see.")
    parser.add_argument("--intake", choices=list(T.INTAKE_ORDER), default="all",
                        help="How well the patient eats over the stay.")
    parser.add_argument("--diabetic", action="store_true")
    parser.add_argument("--coverage", type=float, default=1.0,
                        help="Wearable coverage, 0-1 (default 1.0).")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--per-day", dest="per_day", action="store_true",
                        help="Print every day, not just the events.")
    parser.add_argument("--json", metavar="PATH",
                        help="Also write the full stay here — every day's "
                             "insights, recommendations and recipes.")
    args = parser.parse_args(argv)

    surgeries = [args.surgery] if args.surgery else list(T.SURGERY_ORDER)

    print(RULE)
    print(" POST-OP PHASE 1 — END-TO-END SIMULATION")
    print(RULE)
    span = min(args.day_count or args.weeks * 7, T.MAX_DAYS)
    print(f" {len(surgeries)} patient(s) x {span} days = "
          f"{len(surgeries) * span} runs of all four layers "
          "and the section 10 gate.")
    print(f" Recovery: {T.RECOVERY_PATTERNS[args.recovery]['label']}")
    print(f" Intake:   {T.INTAKE_PATTERNS[args.intake]['label']}"
          + ("  |  diabetic" if args.diabetic else ""))
    print()
    print(" No layer reads surgery_type. Every difference between these")
    print(" patients is a difference in the signals the simulator produced.")

    arcs: Dict[str, Dict[str, Any]] = {}
    for surgery in surgeries:
        arcs[surgery] = run_one(surgery, weeks=args.weeks,
                                day_count=args.day_count,
                                recovery=args.recovery, intake=args.intake,
                                diabetic=args.diabetic,
                                coverage=args.coverage, seed=args.seed)
        report_patient(surgery, arcs[surgery], show_days=args.per_day)

    if len(surgeries) > 1:
        print()
        print(RULE)
        print(" COHORT")
        print(RULE)
        print(f" {'surgery':<34} {'phase':<14} {'nss':<7} {'dev':>3}  gate")
        for surgery, arc in arcs.items():
            last = arc["days"][-1]
            print(f" {T.SURGERIES[surgery]['label']:<34} "
                  f"{last['phase'].title():<14} {_fmt(last['nss']):<7} "
                  f"{len(arc['first_flagged']):>3}  "
                  + (f"day {arc['gate_opened_on']}"
                     if arc["gate_opened_on"] is not None else "locked"))

    if args.json:
        import sim_store
        payload = {}
        for surgery, arc in arcs.items():
            print(f" Collecting {T.SURGERIES[surgery]['label']}…")
            payload[surgery] = {
                **sim_store.collect(
                    arc,
                    patient={"surgery": T.SURGERIES[surgery]["label"],
                             "weight_kg": T.SURGERIES[surgery]["weight_kg"]},
                    recipes=sim_store.recipes_by_phase(
                        arc, age=45, sex="Male", height_cm=178,
                        weight_kg=T.SURGERIES[surgery]["weight_kg"])),
                "events": sim_replay.events(arc),
            }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=1)
        print(f"\n Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
