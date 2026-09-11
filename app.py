"""
Post-op Phase 1 — wound recovery, end to end.

A patient — a real fixture or a generated one — pushed through the real Layer
1-4 code and section 10's gate, one post-op day at a time. Every day gives an
insight and a recommendation from each layer.

Run:  ../Postop-Phase1/Staging/phase1/bin/streamlit run app.py
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import sim_insights
import sim_meals
import sim_paths  # noqa: F401  (import path — must precede the layer imports)
import sim_alert
import sim_recipes
import sim_replay
import sim_timeline as T
from layer1_phase_estimator import (LOW_CONFIDENCE, PHASES,
                                    TRANSITION_MATRIX)
from layer2_deviation_detector import (DEVIATIONS, GP_CREDIBLE_Z,
                                       GP_MIN_HISTORY, GP_SIGNALS)
from layer3_nutrition_gap import (GAP_HIGH, GAP_MODERATE, NSS_ESCALATE,
                                  NSS_REVIEW, NUTRIENT_TARGETS)
from layer4_physician_alert import FORBIDDEN, LLM_MODEL
from phase1_daily_recommendation import (GATE_CRP_BELOW,
                                         GATE_FASTING_GLUCOSE_BELOW,
                                         GATE_PREALBUMIN_ABOVE,
                                         GATE_REMODELLING_POSTERIOR, run_phase1)

st.set_page_config(page_title="Post-op Phase 1", page_icon="🩹", layout="wide")

DAY_AXIS = alt.Axis(tickMinStep=1, format="d")
DEFAULT_USER_ID = "4d1c2fc8-77ad-4205-8226-cf58a175e910"
PHASE_COLOURS = ["#B08968", "#C1443C", "#2E7D5B", "#3D6DA8"]


def _key() -> str:
    """The Anthropic key for this session: pasted, then secrets, then env."""
    if st.session_state.get("anthropic_key"):
        return str(st.session_state["anthropic_key"])
    try:
        found = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        found = None
    return str(found or os.environ.get("ANTHROPIC_API_KEY") or "")


@contextlib.contextmanager
def _anthropic_env(key: str):
    """Lend Layer 4 the key for one call, then take it back.

    Layer 4 reads `os.environ["ANTHROPIC_API_KEY"]` — it is service code and the
    simulator does not get to change that. One process serves every visitor of a
    hosted app, so the key is restored immediately afterwards.
    """
    if not key:
        yield
        return
    previous = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = previous


LAYER_BLURB = {
    "Layer 1": "**Which stage of healing is this wound in?** Wounds heal in "
               "four stages, and each one needs different nutrition. This "
               "works out the current stage from the daily bloods and vitals, "
               "and says how sure it is.  \n"
               "**The result:** a stage, and a confidence in it. Everything "
               "below depends on getting this right.",
    "Layer 2": "**Is this patient healing as expected?** It learns each "
               "patient's own normal course over the first few days, then "
               "flags anything that departs from it — plus fixed checks for "
               "fever, fast heart rate, high glucose and low oxygen.  \n"
               "**The result:** a status — on track, at risk, delayed or "
               "critical — and what was found, on which day.",
    "Layer 3": "**Is this patient being fed enough to heal?** Wound healing "
               "needs far more protein than normal, and it is the one thing "
               "here a ward can actually change. It compares what was eaten "
               "against what this stage of healing needs.  \n"
               "**The result:** a score out of 1 (the NSS) and which "
               "nutrients are short.",
    "Layer 4": "**What should the physician be told?** It turns the three "
               "layers above into a short written alert for the ward round, "
               "or a routine note when nothing needs attention.  \n"
               "**The result:** the alert, who to escalate to, and how "
               "urgent it is.",
}


def _insight_block(part: Dict[str, Any], layer: Optional[str] = None) -> None:
    """A layer's insight and its recommendation, side by side."""
    if layer and layer in LAYER_BLURB:
        st.caption(LAYER_BLURB[layer])
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Insight**")
        st.markdown(part["insight"])
    with right:
        st.markdown("**Recommendation**")
        for action in part["recommendation"]:
            st.markdown(f"- {action}")
    cols = st.columns(len(part["metrics"]))
    for col, (label, value) in zip(cols, part["metrics"].items()):
        col.metric(label, value)
    st.divider()


# ===========================================================================
# SIDEBAR
# ===========================================================================
st.sidebar.title("Post-op Phase 1")

st.sidebar.divider()


@st.cache_data(show_spinner=False, max_entries=16)
def _timeline_for(**kwargs: Any) -> Dict[str, Any]:
    return T.build(**kwargs)


@st.cache_data(show_spinner="Replaying the stay, one Phase 1 run per day…",
               max_entries=16)
def _replay_sim(**kwargs: Any) -> Dict[str, Any]:
    return sim_replay.replay(_timeline_for(**kwargs), use_llm=False)


@st.cache_data(show_spinner="Re-running from the edited day…", max_entries=16)
def _replay_edited(scenario: tuple, edits: str, _timeline: Dict[str, Any],
                   _base: Dict[str, Any]) -> Dict[str, Any]:
    """The stay with edits applied, recomputed only from the earliest one.

    Keyed on the scenario and the edits as JSON; the timeline and the base arc
    ride along underscore-prefixed because Streamlit skips those when building
    the key, and hashing a 42-day timeline to look up a cache entry would cost
    more than the lookup saves.
    """
    overrides = {int(day): value for day, value in json.loads(edits).items()}
    return sim_replay.replay(T.apply_overrides(_timeline, overrides),
                             since=min(overrides), reuse=_base)


@st.cache_data(show_spinner=False, max_entries=64)
def _llm_day(scenario: tuple, day: int, _timeline: Dict[str, Any],
             _key_value: str) -> Dict[str, Any]:
    """One day re-run with Layer 4's model.

    Keyed on the scenario and the day rather than on the timeline itself: a
    timeline is a large nested dict and hashing one per day to look up a cache
    entry costs more than the lookup saves.
    """
    with _anthropic_env(_key_value):
        out = run_phase1(_timeline, use_llm=True)
        # The service does not call the model on a day with nothing flagged —
        # a routine note must not depend on a network call. Here it can, so
        # the model phrases that note too, through the same guardrails.
        out["layer4_alert"] = sim_alert.phrase_summary(out["layer4_alert"])
        return out


# ---------------------------------------------------------------------------
# The patient — everything the sidebar holds
# ---------------------------------------------------------------------------
st.sidebar.subheader("Patient")
surgery = st.sidebar.selectbox(
    "Surgery type", T.SURGERY_ORDER, key="surgery",
    format_func=lambda s: T.SURGERIES[s]["label"])
_spec = T.SURGERIES[surgery]
st.sidebar.caption(_spec["note"])

# The surgery carries a typical body. Written into session_state when the
# operation changes so the fields below show it, and editable afterwards —
# weight is the one that matters, because Layer 3's protein and calorie targets
# are per kilogram.
if st.session_state.get("_surgery_was") != surgery:
    st.session_state["_surgery_was"] = surgery
    st.session_state["weight_kg"] = float(_spec["weight_kg"])

b1, b2 = st.sidebar.columns(2)
age = b1.number_input("Age", 18, 95, 45, key="age")
sex = b2.selectbox("Sex", ["Male", "Female"], key="sex")
height_cm = b1.number_input("Height (cm)", 140, 210, 178, key="height_cm")
weight_kg = b2.number_input("Weight (kg)", 35.0, 200.0, step=0.5,
                            key="weight_kg")
st.sidebar.caption(
    "Weight sets Layer 3's per-kilogram targets. Age, sex and height are read "
    "only by the recipe pool, to pick a demographic band.")

st.sidebar.subheader("Wound healing")
recovery = st.sidebar.selectbox(
    "How it goes", T.RECOVERY_ORDER, key="recovery",
    format_func=lambda r: T.RECOVERY_PATTERNS[r]["label"])
st.sidebar.caption(T.RECOVERY_PATTERNS[recovery]["note"])

intake_pattern = st.sidebar.selectbox(
    "Meals actually eaten", T.INTAKE_ORDER, key="intake_pattern",
    format_func=lambda i: T.INTAKE_PATTERNS[i]["label"],
    help="What the patient eats over the six weeks. The Nutritional "
         "Sufficiency Score on the Layer 3 tab is worked out from this — it is "
         "not set directly.")

st.sidebar.subheader("Diet and conditions")
allergens = st.sidebar.multiselect("Allergens", sim_recipes.ALLERGENS,
                                   key="allergens")
diets = st.sidebar.multiselect("Dietary restrictions", sim_recipes.DIETS,
                               key="diets")

conditions = st.sidebar.multiselect(
    "Existing conditions",
    ["Diabetes", "Hypertension", "Chronic kidney disease", "COPD",
     "Rheumatoid arthritis", "Obesity", "Anaemia", "Peripheral vascular disease"],
    key="conditions", accept_new_options=True,
    help="Type anything not listed and press enter to add it.")

# Only diabetes changes anything the layers can see. Phase 1's inputs carry no
# blood pressure at all, so hypertension — and anything typed in — is recorded
# as context and nothing more. Saying so is better than implying an effect that
# is not there.
diabetic = "Diabetes" in conditions
if conditions:
    _inert = [c for c in conditions if c != "Diabetes"]
    st.sidebar.caption(
        ("Diabetes raises this patient's glucose, which the layers do read."
         if "Diabetes" in conditions else "")
        + (("  \n" if "Diabetes" in conditions else "")
           + f"{', '.join(_inert)} recorded for context. Phase 1 monitors "
             "wound healing and takes no blood pressure reading, so nothing "
             "downstream changes." if _inert else ""))

user_id = st.sidebar.text_input(
    "Foodhak user id", value=DEFAULT_USER_ID, key="user_id",
    help="Identifies the patient, the way the real payload does. It is what "
         "the recipe pool is queried with and what Layer 4's note names. No "
         "layer computes with it.").strip() or DEFAULT_USER_ID



# ---------------------------------------------------------------------------
# The stay
# ---------------------------------------------------------------------------
# Two placeholders, claimed now and filled later. The heading depends on the
# day being reviewed, which depends on a replay that cannot run until the
# window is known — but the heading still has to appear above the window on
# screen, and Streamlit lays out in call order.
_heading_slot = st.container()

# Phase 1 covers wound healing, which this models for six weeks. There is no
# window to choose: the stay is the whole of it, and the calendar counts post-op
# days rather than dates, so the only thing a date adds is a label.
days_total = T.MAX_DAYS

scenario = (surgery, days_total, recovery, intake_pattern, diabetic,
            float(age), str(sex), float(height_cm), float(weight_kg), user_id)
# A widget value belongs to the patient it was typed against. Changing the
# patient underneath it would turn stale numbers into phantom edits.
if st.session_state.get("_scenario_was") != scenario:
    st.session_state["_scenario_was"] = scenario
    for _stale in [k for k in st.session_state
                   if k.split("_")[0] in ("lab", "tp", "fh", "hp", "th", "sp",
                                          "pain", "ap", "na", "ml")]:
        del st.session_state[_stale]
# The patient is identified by the Foodhak user id, the way the real payload
# does it — `patient_ref` is what Layer 4 prints and what the layers echo back.
_build_args = dict(surgery=surgery, days=days_total, recovery=recovery,
                   intake_pattern=intake_pattern, diabetic=diabetic,
                   age=float(age), sex=str(sex).lower(),
                   height_cm=float(height_cm), weight_kg=float(weight_kg),
                   patient_ref=user_id, seed=5)
base_timeline = _timeline_for(**_build_args)
day_numbers_of_base = [int(d["day"]) for d in base_timeline["days"]]
base_arc = _replay_sim(**_build_args)

# The day's inputs are ordinary widgets, not a form: changing one should
# re-run on its own. That means the edit has to be read out of session_state
# *here*, before the replay, because Streamlit runs the script top to bottom
# and the widgets themselves are not drawn until further down.
def _edit_for(day: int) -> Optional[Dict[str, Any]]:
    """What the widgets currently say for `day`, if it differs from the
    generated patient. Returns None when the day is untouched, so a day that
    was merely looked at is not recorded as edited."""
    if f"tp_{day}" not in st.session_state:
        return None                       # never drawn, so nothing to read
    base = T.day_inputs(base_timeline, day)
    get = st.session_state.get
    candidate = {
        "labs": {name: get(f"lab_{name}_{day}") for name in T.EDITABLE_LABS},
        "mean_temp": get(f"tp_{day}"), "fever_hours": get(f"fh_{day}"),
        "mean_hr": get(f"hp_{day}"), "tachy_hours": get(f"th_{day}"),
        "mean_spo2": get(f"sp_{day}"), "pain": get(f"pain_{day}"),
        "appetite": get(f"ap_{day}"), "nausea": get(f"na_{day}"),
        "meals": get(f"ml_{day}"), "clinical": base["clinical"],
    }

    def _same(left: Any, right: Any) -> bool:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)) < 0.05
        return left == right

    base_pain = round(sum(base["pain_series"]) / len(base["pain_series"]))
    unchanged = (
        all(_same(candidate["labs"][k], base["labs"].get(k))
            for k in T.EDITABLE_LABS)
        and all(_same(candidate[k], base[k]) for k in
                ("mean_temp", "fever_hours", "mean_hr", "tachy_hours",
                 "mean_spo2", "appetite", "nausea", "meals"))
        and _same(candidate["pain"], base_pain))
    return None if unchanged else candidate


overrides = {day: edit for day in day_numbers_of_base
             if (edit := _edit_for(day)) is not None}
st.session_state.overrides = overrides
arc = (_replay_edited(scenario, json.dumps(overrides, sort_keys=True),
                      base_timeline, base_arc)
       if overrides else base_arc)

# --- the day being reviewed ------------------------------------------------
# Chosen from the calendar below: it needs seven columns of width, and putting
# it above the tabs means the day can be changed while looking at any layer.
day_numbers = arc["day_numbers"]
if st.session_state.get("picked_day") not in day_numbers:
    st.session_state.picked_day = day_numbers[-1]
picked_day = int(st.session_state.picked_day)

record = sim_replay.by_day(arc, picked_day)
timeline = record["timeline"]
bundle = record["bundle"]


# The model is NOT called on the way in. Phrasing one day's note takes about
# eight seconds against roughly seven tenths of a second for all four layers
# and the gate — so calling it automatically made every click on the calendar
# feel broken. It is asked for explicitly on the Layer 4 tab instead, and the
# answer is kept for as long as the day and the patient stay the same.
_written = st.session_state.setdefault("written_days", {})
_written_key = (scenario, picked_day)
if _written_key in _written:
    bundle = _written[_written_key]

summary = bundle["summary"]
layer1, layer2 = bundle["layer1_phase"], bundle["layer2_deviations"]
layer3, layer4 = bundle["layer3_nutrition"], bundle["layer4_alert"]
gate = bundle["transition_gate"]

meals_today = record.get("meals_logged")
day_view = sim_insights.overall(bundle, meals_logged=meals_today)

_heading_slot.title(_spec["label"])
_heading_slot.caption(
    f"{weight_kg:g} kg"
    + (f" · {', '.join(conditions)}" if conditions else "")
    + f"  \n**Post-op day {picked_day} of {day_numbers[-1]}.** Everything "
      "below is Phase 1 as it stood that morning.")


# ===========================================================================
# INTRO
# ===========================================================================
# Collapsed: it is orientation for a first look, not something to scroll past
# on every visit.
with st.expander("What Phase 1 does, and the four stages of healing"):
    st.markdown(
        "A wound heals in four stages, and **each stage needs different "
        "feeding**. Phase 1 watches a patient through those stages and answers "
        "four questions every morning:")
    q = st.columns(4)
    for col, (number, question, layer) in zip(q, (
            ("1", "Which stage is this wound in?", "Layer 1"),
            ("2", "Is it going the way it should?", "Layer 2"),
            ("3", "Is the patient being fed enough to heal?", "Layer 3"),
            ("4", "What should the physician be told?", "Layer 4"))):
        col.markdown(f"**{number}. {question}**  \n:gray[{layer}]")

    # Durations are read out of the model's own transition matrix rather than
    # typed here: a phase the model leaves with probability p lasts 1/p days,
    # so if someone retunes the matrix this table follows it.
    st.markdown("**The four stages**")
    _blurbs = {
        "HAEMOSTASIS": "Bleeding stops and a clot forms. Hours, not days — so "
                       "it is pinned to the day of surgery rather than "
                       "inferred, because daily bloods cannot resolve it.",
        "INFLAMMATION": "Immune cells clear the wound. CRP climbs, peaks, and "
                        "must then start falling.",
        # No markdown in here: a dataframe renders cell text literally, so
        # asterisks would show up as asterisks.
        "PROLIFERATION": "New tissue is built. The hungriest stage — protein "
                         "and vitamin C targets are at their highest.",
        "REMODELLING": "The scar reorganises and strengthens. Targets fall "
                       "back towards normal.",
    }
    st.dataframe(pd.DataFrame([{
        "Stage": name.title(),
        "Typically lasts": ("hours" if TRANSITION_MATRIX[i][i] == 0 else
                            f"about {1 / (1 - TRANSITION_MATRIX[i][i]):.0f} days"),
        "What is happening": _blurbs[name],
    } for i, name in enumerate(PHASES)]), hide_index=True, width="stretch")
    st.caption(
        "The system monitors and flags; the physician decides. It cannot see "
        "the wound — everything here is inferred from bloods, wearables and "
        "what was eaten — so every stage estimate asks to be confirmed by "
        "looking at it.")


# ===========================================================================
# CALENDAR, AND THE DAY'S INPUTS BESIDE IT
# ===========================================================================
_by_number = {r["day"]: r for r in arc["days"]}
_edits = st.session_state.get("overrides", {})


def _mark(record: Dict[str, Any]) -> str:
    """One glyph per day. Severity first — a critical day must not look like a
    quiet one at a glance."""
    if record["deviation_count"] == 0:
        return "·"
    return "!" if record["worst_severity"] == "CRITICAL" else "•"


# Streamlit stamps `st-key-<key>` onto a keyed widget's wrapper, which is the
# only reliable handle for styling one group of buttons and not every button on
# the page. The cells are squared off and the row gap closed so the grid reads
# as a calendar rather than as forty-two separate buttons.
# Streamlit stamps `st-key-<key>` onto a keyed widget's wrapper, which is the
# only reliable handle for styling one group of widgets and not every widget on
# the page. Both blocks are shrunk to fit above the fold without scrolling:
# forty-two cells and four blocks of inputs at default sizing needs 750px, and
# the tabs underneath are the point of the screen.
st.markdown("""
<style>
[class*="st-key-cal_"] button {
    height: 1.7rem;
    min-height: 1.7rem;
    padding: 0 !important;
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
}
[class*="st-key-cal_"] { margin-bottom: -0.95rem; }

/* The day's inputs: smaller type, and no +/- steppers. The steppers cost a
   third of each field's width for a control nobody uses when the value is
   typed. */
.st-key-day_inputs [data-testid="stNumberInput"] input,
.st-key-day_inputs [data-baseweb="select"] { font-size: 0.78rem; }
.st-key-day_inputs [data-testid="stNumberInput"] input { padding: 0.18rem 0.4rem; }
.st-key-day_inputs [data-testid="stNumberInputStepUp"],
.st-key-day_inputs [data-testid="stNumberInputStepDown"] { display: none; }
.st-key-day_inputs label p { font-size: 0.68rem; margin-bottom: 0 !important; }
.st-key-day_inputs [data-testid="stMarkdownContainer"] p { margin-bottom: 0.1rem; }
/* Breathing room above each group heading, so labs / vitals / self_report read
   as three blocks rather than one run of fields. */
.st-key-day_inputs [data-testid="stMarkdownContainer"] p > strong { line-height: 2; }
.st-key-day_inputs [data-testid="stVerticalBlock"] { gap: 0.15rem; }
.st-key-day_inputs [data-baseweb="select"] > div { min-height: 1.9rem; }
.st-key-day_inputs [data-testid="stCheckbox"] p { font-size: 0.72rem; }
.st-key-day_inputs button[kind="secondaryFormSubmit"],
.st-key-day_inputs button[kind="primaryFormSubmit"] { padding: 0.2rem 0.5rem; }
</style>
""", unsafe_allow_html=True)

cal_col, input_col = st.columns([1, 1], gap="medium")

# --- left: the calendar ----------------------------------------------------
with cal_col:
    st.markdown("**Post-op calendar**")
    with st.container(border=True):
        # Seven to a row, starting at day 0. Weekday columns only earn their
        # place when the dates are on screen, and they are not: what matters
        # here is the post-op day, so the grid is simply weeks of seven with no
        # blank cells to skip over.
        cells: List[Optional[int]] = list(day_numbers)
        while len(cells) % 7:
            cells.append(None)

        for start in range(0, len(cells), 7):
            row = st.columns(7)
            for col, number in zip(row, cells[start:start + 7]):
                if number is None:
                    col.write("")
                    continue
                record = _by_number[number]
                # The label stays at three characters. A fourth (an edit
                # marker) overflows a 49px cell and renders as an ellipsis,
                # which loses the severity glyph — and severity is the thing
                # that must survive at a glance. Edited days are named in the
                # caption below and in each cell's tooltip instead.
                if col.button(
                        f"{_mark(record)}{number}",
                        key=f"cal_{number}", width="stretch",
                        type="primary" if number == picked_day else "secondary",
                        help=f"Post-op day {number} · "
                             f"{record['phase'].title()} · "
                             f"{record['healing_status'].replace('_', ' ').lower()}"
                             + (" · edited" if number in _edits else "")):
                    st.session_state.picked_day = number
                    st.rerun()
        st.caption("Week by week from the operation  ·  "
                   "`·` clear  `•` deviation  `!` critical")
        st.divider()

        _today_record = _by_number[picked_day]
        _status = _today_record["healing_status"]
        st.markdown(
            f"**Post-op day {picked_day}**  \n"
            f"{_today_record['phase'].title()} · "
            + (":green" if _status == "ON_TRACK"
               else ":red" if _status == "CRITICAL" else ":orange")
            + f"[**{_status.replace('_', ' ').title()}**]"
            + (f" · {_today_record['deviation_count']} flagged"
               if _today_record["deviation_count"] else "")
            + (f"  \nNSS {_today_record['nss']:.2f}"
               if _today_record["nss"] is not None else "  \nNSS not scored")
            + (f" · {_today_record['meals_logged']} of 3 meals logged"
               if _today_record["meals_logged"] is not None else ""))

        if _edits:
            st.caption("Edited: " + ", ".join(f"day {int(d)}"
                                              for d in sorted(int(x) for x in _edits)))
            undo = st.columns(2)
            if picked_day in _edits and undo[0].button("Undo this day",
                                                       width="stretch"):
                st.session_state.overrides.pop(picked_day, None)
                st.rerun()
            if undo[1].button("Undo all", width="stretch"):
                st.session_state.overrides = {}
                st.rerun()

# --- right: what that day recorded, editable -------------------------------
with input_col:
    _inputs_box = st.container(border=True, key="day_inputs")

    current = T.day_inputs(timeline, picked_day)
    with _inputs_box:
        st.markdown(f"**Inputs for post-op day {picked_day}**")
        st.markdown("**labs**")
        _LABS = {"crp": "CRP", "wbc": "WBC", "glucose": "Glucose",
                 "fasting_glucose": "Fasting gluc.", "prealbumin": "Prealbumin",
                 "albumin": "Albumin", "zinc": "Zinc"}
        labs: Dict[str, Any] = {}
        lab_cols = st.columns(4)
        for index, (name, label) in enumerate(_LABS.items()):
            labs[name] = lab_cols[index % 4].number_input(
                label, value=current["labs"].get(name), step=1.0, format="%.1f",
                key=f"lab_{name}_{picked_day}")

        # A blank field is a blood that was not taken today, not a missing
        # value. Bloods are not all drawn daily — so say when each empty one
        # was last taken, otherwise an empty box reads as a bug.
        _missing = [name for name in _LABS if current["labs"].get(name) is None]
        if _missing:
            _last = []
            for name in _missing:
                prior = [(d["day"], (d.get("labs") or {})[name])
                         for d in timeline["days"]
                         if d["day"] < picked_day and name in (d.get("labs") or {})]
                _last.append(
                    f"{_LABS[name]} last taken day {prior[-1][0]} "
                    f"({prior[-1][1]:g})" if prior
                    else f"{_LABS[name]} not taken yet")
            st.caption(
                "Not taken today — " + " · ".join(_last)
                + ". Blank means not measured, not zero.",
                help="Blood is not drawn daily for everything. A result "
                     "that was never taken is left blank rather than "
                     "filled in, so nothing is ever scored against a "
                     "number nobody measured. Type one in to add it.")

        st.markdown("**vitals** — daily means, and hours held above")
        v = st.columns(5)
        mean_temp = v[0].number_input("Mean temp", 34.0, 43.0,
                                      float(current["mean_temp"] or 37.0), 0.1,
                                      key=f"tp_{picked_day}")
        fever_hours = v[1].number_input(
            "h > 38.5", 0.0, 24.0, float(current["fever_hours"]), 0.5,
            key=f"fh_{picked_day}",
            help="A fever counts once it has run for 6 hours without a break. "
                 "An average alone cannot tell a steady low temperature from a "
                 "short spike, so the duration is asked for separately.")
        mean_hr = v[2].number_input("Mean HR", 30.0, 200.0,
                                    float(current["mean_hr"] or 80.0), 1.0,
                                    key=f"hp_{picked_day}")
        tachy_hours = v[3].number_input(
            "h > 100", 0.0, 24.0, float(current["tachy_hours"]), 0.5,
            key=f"th_{picked_day}",
            help="A fast heart rate counts once it has run for 4 hours without a break.")
        mean_spo2 = v[4].number_input("Mean SpO2", 70.0, 100.0,
                                      float(current["mean_spo2"] or 97.0), 0.5,
                                      key=f"sp_{picked_day}")

        st.markdown("**self_report**")
        r = st.columns(4)
        pain_mean = r[3].number_input(
            "Pain (mean)", 0, 10, int(round(
                sum(current["pain_series"]) / len(current["pain_series"]))),
            key=f"pain_{picked_day}",
            help="Recorded four-hourly through the day; this is the "
                 "average of those readings, and editing it sets all four.")
        appetite = r[0].selectbox(
            "Appetite", T.APPETITE_CHOICES, key=f"ap_{picked_day}",
            index=T.APPETITE_CHOICES.index(current["appetite"])
            if current["appetite"] in T.APPETITE_CHOICES else 1)
        meals = r[1].number_input(
            "Meals logged", 0, 3, int(current["meals"]), key=f"ml_{picked_day}",
            help="Drives `nutrition`. Zero logs nothing at all, which leaves "
                 "the NSS unknown rather than zero.")
        nausea = r[2].checkbox("Nausea", value=current["nausea"],
                               key=f"na_{picked_day}")

        st.caption(
            "Changing any field re-runs the four layers from this day "
            "onward. Earlier days cannot move: each day is judged only "
            "on the days up to it.")



tab = dict(zip(
    (_L := ["Today", "Layer 1 · Phase", "Layer 2 · Deviations",
            "Layer 3 · Nutrition", "Layer 4 · Alert"]),
    st.tabs(_L)))


# ===========================================================================
# TODAY
# ===========================================================================
with tab["Today"]:
    # Meals logged only exists for a generated patient. A fixture carries a
    # 24-hour total with no meal structure, so the column would be a permanent
    # em-dash — better to not offer it than to offer it empty.
    tiles = [("Phase", summary["current_phase"].title(), None),
             ("Healing", summary["healing_status"].replace("_", " ").title(),
              f"{summary['deviation_count']} deviations"),
             ("NSS", "—" if summary["nss"] is None
              else f"{summary['nss']:.2f}", None)]
    _nss_help = ("Nutritional Sufficiency Score, 0 to 1 — how much of what "
                 "this patient needs to heal they actually got. Explained in "
                 "full on the Layer 3 tab.")
    if meals_today is not None:
        tiles.append(("Meals logged", f"{meals_today} of 3", None))
    for col, (label, value, delta) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, delta, delta_color="off",
                   help=_nss_help if label == "NSS" else None)

    (st.success if gate["phase2_unlocked"] else st.info)(day_view["gate"])

    st.subheader("What to do today")
    st.caption("Every line comes from one of the four layers. Nothing here is "
               "written by this app.")
    for action in day_view["recommendation"]:
        st.markdown(f"- {action}")

    st.subheader("What each layer found")
    for name, part in day_view["layers"].items():
        st.markdown(f"**{name}** — {part['insight']}")




# ===========================================================================
# LAYER 1 — which stage of healing
# ===========================================================================
with tab["Layer 1 · Phase"]:
    _insight_block(day_view["layers"]["Layer 1"], "Layer 1")

    st.markdown("**What today's readings said**")
    st.caption("Each reading is turned into a plain description, and the stage "
               "that best explains all of them together is the one reported.")
    today = next((d for d in reversed(layer1["daily"])
                  if d["day"] == int(picked_day)), None)
    if today and today["signals"]:
        _why = {w["signal"]: w for w in (today.get("why") or [])}
        st.dataframe(pd.DataFrame([{
            "Reading": name.replace("_", " ").replace("velocity", "trend")
                           .replace("pattern", "").strip().capitalize(),
            "Says": str(value).replace("_", " "),
            f"Typical of {today['phase'].title()}?":
                ("yes" if (_why.get(name, {}).get("probability_in_phase") or 0) >= 0.30
                 else "not especially"),
        } for name, value in today["signals"].items()]),
            hide_index=True, width="stretch")
    else:
        st.caption("Nothing was measured today, so the estimate is carried "
                   "forward from yesterday.")

    st.markdown("**How the stage changed over the stay**")
    st.caption("The estimate on each morning, using only what was known by "
               "then. The dashed line is the day being reviewed.")
    post = pd.DataFrame(sim_replay.posterior_frame(arc))
    rule = alt.Chart(pd.DataFrame({"Day": [int(picked_day)]})).mark_rule(
        color="#111", strokeWidth=2, strokeDash=[3, 3]).encode(x="Day:Q")
    span = alt.Scale(domain=[day_numbers[0], day_numbers[-1]], nice=False)
    st.altair_chart(
        (alt.Chart(post).mark_area(interpolate="monotone").encode(
            x=alt.X("Day:Q", title="Post-op day", axis=DAY_AXIS, scale=span),
            y=alt.Y("Posterior:Q", stack="normalize", title="How likely",
                    axis=alt.Axis(format="%")),
            color=alt.Color("Phase:N", sort=list(PHASES),
                            scale=alt.Scale(domain=list(PHASES),
                                            range=PHASE_COLOURS),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Day", "Phase", alt.Tooltip("Posterior:Q", format=".0%")])
         + rule).properties(height=240), width="stretch")

    if layer1["phase_transitions"]:
        st.caption("Stage changes: " + " · ".join(
            f"day {t['day']} {t['from'].title()} → {t['to'].title()}"
            + (" (went backwards)" if t["regression"] else "")
            for t in layer1["phase_transitions"]))


# ===========================================================================
# LAYER 2 — is healing going to plan
# ===========================================================================
with tab["Layer 2 · Deviations"]:
    _insight_block(day_view["layers"]["Layer 2"], "Layer 2")

    # The seven checks are what this layer *is*. They were buried at the
    # bottom behind a chart nobody could read; they belong first.
    st.markdown("**The seven things it checks, every day**")
    st.caption("Each one has a rule. A check with nothing to work with is "
               "reported as **not assessed** — never as clear, because a "
               "patient with no thermometer would otherwise look exactly like "
               "a patient with no fever.")

    _PLAIN = {
        "inflammation_not_resolving": "Inflammation not settling",
        "protein_synthesis_impairment": "Not building protein",
        "phase_transition_delay": "Stuck in one stage",
        "oxygenation_concern": "Low oxygen to the wound",
        "glucose_dysregulation": "Blood sugar too high",
        "unexplained_tachycardia": "Heart rate too fast",
        "fever_pattern": "Fever",
    }
    _found = {d["deviation"]: d for d in layer2["deviations"]}
    _STATUS = {"detected": "⚠️ found", "clear": "✓ clear",
               "not_assessed": "— not assessed"}

    def _rule(name: str) -> str:
        """The rule as the service states it, minus its internal shorthand."""
        text = DEVIATIONS[name]["source"].split("—", 1)[-1].strip()
        for jargon, plain in (
                ("HMM posterior stuck in Inflammation",
                 "still in the inflammation stage"),
                ("Prealbumin flat or declining", "prealbumin not rising"),
                ("CRP plateau or re-elevation", "CRP stops falling, or rises again"),
                ("SpO2 consistently", "oxygen saturation stays at"),
                ("sustained", "held for")):
            text = text.replace(jargon, plain)
        return text[0].upper() + text[1:]

    st.dataframe(pd.DataFrame([{
        "Check": _PLAIN.get(name, name.replace("_", " ").capitalize()),
        "Looks at": DEVIATIONS[name]["signal"],
        "Rule": _rule(name),
        "Result": _STATUS.get(entry["status"], entry["status"]),
        # A blank rather than "None": this column is only filled when something
        # was found, and pandas renders a missing integer as the word None.
        "On day": str(_found[name]["day"]) if name in _found else "",
        "Why it matters": DEVIATIONS[name]["interpretation"],
    } for name, entry in layer2["assessment"].items()]),
        hide_index=True, width="stretch")

    _unassessed = {name: entry for name, entry in layer2["assessment"].items()
                   if entry["status"] == "not_assessed"}
    if _unassessed:
        st.caption("Could not run today — " + " · ".join(
            f"**{_PLAIN.get(name, name)}**: {entry.get('reason') or 'no data'}"
            for name, entry in _unassessed.items()))

    if layer2["deviations"]:
        st.markdown("**What was found**")
        for dev in layer2["deviations"]:
            with st.container(border=True):
                head, sev = st.columns([4, 1])
                head.markdown(f"**{_PLAIN.get(dev['deviation'], dev['deviation'])}**"
                              f" · first seen on day {dev['day']}")
                sev.markdown(
                    f":{'red' if dev['severity'] == 'CRITICAL' else 'orange'}"
                    f"[**{dev['severity'].lower()}**]")
                st.markdown(dev["finding"])
                st.caption(f"{dev['clinical_interpretation']} — "
                           f"**{dev['recommended_action']}**")
    else:
        st.success("Nothing was found on any check that could run.")

    # The chart is supporting evidence for two of the seven checks, not the
    # headline. Behind an expander, with the reading spelled out.
    _judged = {signal: [r for r in layer2["gp_trajectories"].get(signal, [])
                        if r.get("status") != "insufficient_history"]
               for signal in GP_SIGNALS}
    if any(_judged.values()):
        with st.expander("How 'not settling' and 'not building protein' are judged"):
            st.caption(
                "Two of the seven checks compare a blood result against this "
                "patient's **own** earlier days rather than against a fixed "
                "number — because a CRP of 40 is reassuring in someone who was "
                "at 90 yesterday and alarming in someone who was at 12. The "
                "shaded band is where the next reading was expected to land. A "
                "dot outside it is a day that broke the patient's own pattern.")
            for signal in GP_SIGNALS:
                judged = _judged[signal]
                label = {"crp": "CRP — inflammation",
                         "prealbumin": "Prealbumin — protein building"}.get(
                             signal, signal.title())
                if not judged:
                    st.caption(f"*{label}* — not enough earlier readings yet.")
                    continue
                outside = [r for r in judged if r["status"] != "within"]
                st.markdown(f"**{label}**")
                st.caption(
                    (f"{len(outside)} day(s) outside the expected range: "
                     + ", ".join(f"day {r['day']} expected {r['expected']:g}, "
                                 f"got {r['observed']:g}" for r in outside[:3]))
                    if outside else
                    "Every reading landed where it was expected.")
                df = pd.DataFrame([{
                    "Day": r["day"], "Measured": r["observed"],
                    "Expected": r["expected"],
                    "low": r["credible_interval"][0],
                    "high": r["credible_interval"][1],
                    "Result": "as expected" if r["status"] == "within"
                    else "higher than expected" if r["status"] == "above"
                    else "lower than expected"} for r in judged])
                st.altair_chart(
                    (alt.Chart(df).mark_area(opacity=0.18, color="#3D6DA8").encode(
                        x=alt.X("Day:Q", title="Post-op day", axis=DAY_AXIS),
                        y=alt.Y("low:Q", title=label.split(" —")[0]),
                        y2="high:Q")
                     + alt.Chart(df).mark_line(strokeDash=[5, 3],
                                               color="#3D6DA8").encode(
                         x="Day:Q", y="Expected:Q")
                     + alt.Chart(df).mark_point(filled=True, size=110).encode(
                         x="Day:Q", y="Measured:Q",
                         color=alt.Color("Result:N", scale=alt.Scale(
                             domain=["as expected", "higher than expected",
                                     "lower than expected"],
                             range=["#2E7D5B", "#C1443C", "#D98324"]),
                             legend=alt.Legend(orient="bottom", title=None)),
                         tooltip=["Day", "Measured", "Expected", "Result"])
                     ).properties(height=200), width="stretch")


# ===========================================================================
# LAYER 3
# ===========================================================================
with tab["Layer 3 · Nutrition"]:
    _insight_block(day_view["layers"]["Layer 3"], "Layer 3")

    with st.expander("What is the NSS?"):
        st.markdown(
            "**A single number for *is this patient being fed well enough to "
            "heal*.** It runs from 0 to 1.\n\n"
            "Healing a wound is building tissue, and that needs materials — "
            "protein above all, plus vitamin C, zinc and enough calories that "
            "the protein is not simply burnt for energy. How much is needed "
            "changes with the stage of healing, which is why Layer 1 comes "
            "first.\n\n"
            "For each nutrient the patient needs today, this works out how "
            "much of the target they actually got. Those are then combined "
            "into one score, weighted so that the nutrients that matter most "
            "at this stage count for most.\n\n"
            f"- **1.00** — everything needed was met.\n"
            f"- **below {NSS_REVIEW:g}** — flag it for the dietitian at the "
            "next ward round.\n"
            f"- **below {NSS_ESCALATE:g}** — tell the dietitian today.\n\n"
            "Two things it deliberately does not do. A nutrient this patient "
            "does not need right now is left out of the score rather than "
            "counted as met — so a score cannot be flattered by a nutrient "
            "nobody was aiming for. And a day nothing was written down scores "
            "*nothing at all* rather than zero, because not knowing what "
            "someone ate is not the same as knowing they ate nothing.\n\n"
            "It is the one number on this screen a ward can move directly: "
            "the others are measurements, this one is a consequence of what "
            "was put in front of the patient.")

    scored = [d for d in layer3["daily"] if d["status"] == "scored"]
    if not scored:
        st.info("Nothing was eaten and logged in this window, so there is "
                "nothing to score. Scoring looks at the previous 24 hours, so "
                "the day of the operation never has a score.")
    else:
        latest = scored[-1]
        if meals_today is not None:
            logged = (timeline.get("_simulation") or {}).get("meals_by_day", {})
            names = logged.get(int(picked_day), logged.get(str(picked_day)))
            if names is not None:
                st.caption("Logged today: " + sim_meals.describe(list(names)))

        rows = list(latest["nutrients"])
        listed = {r["nutrient"] for r in rows}
        for nutrient, spec in NUTRIENT_TARGETS.items():
            if nutrient in listed:
                continue
            if spec.get(latest["phase"]) is None:
                reason = f"not needed during {latest['phase'].lower()}"
            elif spec.get("conditional_on") == "zinc_deficient":
                reason = "this patient's zinc is not low, so none is needed"
            elif (spec.get("conditional_on_phase") or {}).get(latest["phase"]):
                reason = "only needed if the patient is at risk of a shortage"
            else:
                reason = "not needed at this stage"
            rows.append({"nutrient": nutrient, "label": spec["label"],
                         "status": "not scored", "source": reason})

        st.dataframe(pd.DataFrame([{
            "Nutrient": r["label"],
            "Status": r["status"].replace("_", " "),
            "Target": r.get("target"), "Actual": r.get("actual"),
            "Unit": r.get("unit"), "Gap %": r.get("gap_pct"),
            "Priority": r.get("priority", ""),
            "Source": r.get("source"),
        } for r in rows]), hide_index=True, width="stretch")
        st.caption(
            f"More than {GAP_HIGH:.0%} short is **high** priority, more than "
            f"{GAP_MODERATE:.0%} is **moderate**. **Not scored** means this "
            "patient does not need that nutrient right now — the reason is in "
            "the last column — and it is left out of the score rather than "
            "counted as a shortfall.")

        if latest["interactions"]:
            st.markdown("**Worth putting together**")
            st.caption("Findings that only mean something in combination. A "
                       "protein shortfall is a feeding problem and inflammation "
                       "that will not settle is a clinical one — together they "
                       "suggest the shortfall is holding the healing back.")
            for item in latest["interactions"]:
                st.warning(f"**{item.get('severity')}** · "
                           f"{item.get('alert_text', '')}  \n"
                           f":gray[{item.get('clinical_interpretation', '')}]")



    st.divider()
    st.subheader("Recipes for this phase")
    st.caption("Approved recipes for this stage of healing, filtered to the "
               "patient's age, sex, height, weight, diet and allergens.")

    @st.cache_data(show_spinner="Querying the recipe pool…", max_entries=32)
    def _pool(phase: Optional[str], pool_age: float, pool_sex: str,
              pool_height: float, pool_weight: float, pool_diets: tuple,
              pool_allergens: tuple, pool_user: str,
              _bundle: Dict[str, Any]) -> Dict[str, Any]:
        # Only `_bundle` is underscore-prefixed: Streamlit skips those when it
        # builds the cache key, which is right for the bundle (the phase already
        # stands for it) and would be a silent bug for anything else.
        return sim_recipes.select(
            bundle=_bundle, age=pool_age, sex=pool_sex, height_cm=pool_height,
            weight_kg=pool_weight, diets=list(pool_diets),
            allergens=list(pool_allergens), user_id=pool_user,
            date=dt.date.today().isoformat(), recipes_per_meal=3)

    if st.button("Re-query the pool"):
        _pool.clear()
    selection = _pool(summary["current_phase"], float(age), str(sex),
                      float(height_cm), float(weight_kg), tuple(diets),
                      tuple(allergens), user_id, bundle)

    status = selection.get("status")
    availability = selection.get("availability") or {}
    c = st.columns(3)
    c[0].metric("Source", str(selection.get("served_from") or "—"))
    c[1].metric("Phase", str(selection.get("healing_phase") or "—").title())
    c[2].metric("Recipes", availability.get("available_total", 0))

    if status == "cached":
        st.info(
            f"The database is unreachable, so these are the rows it last "
            f"returned for this phase and band, fetched "
            f"{selection.get('fetched_at', 'earlier')}. Every one came from "
            f"`recipe_pool` — nothing here is composed.  \n"
            f":gray[Live attempt: {selection.get('live_reason')}]")

    _oral_from = (timeline.get("_simulation") or {}).get("oral_from_day", 1)
    if picked_day < _oral_from:
        st.info(
            f"Nothing to suggest on day {picked_day}. Oral intake starts on "
            f"day {_oral_from} after this operation, so there is no meal to "
            "plan — which is also why nothing was scored for nutrition today.")
    elif status not in ("ok", "partial"):
        st.error(f"**{status}** — {selection.get('reason')}")
        st.caption("Nothing is invented to fill the gap — an unapproved recipe "
                   "sitting beside approved ones, with nothing to tell them "
                   "apart, is the one thing worth avoiding here.")
    elif status == "partial":
        st.warning("Fewer recipes than requested for this band — shortage "
                   "reported rather than padded.")

    by_meal = sim_recipes.by_meal(selection)

    # Which meals the dietitian logged today, and — of the three offered — which
    # one the patient actually had. The pool offers options; the intake log only
    # records that the meal happened, so the choice is drawn deterministically
    # from the day and the meal. It is the simulator's, not the service's, and
    # it is the one thing on this tab that is.
    _logged = set((timeline.get("_simulation") or {})
                  .get("meals_by_day", {}).get(int(picked_day)) or [])

    def _eaten_index(meal: str, options: int) -> Optional[int]:
        if meal not in _logged or options <= 0:
            return None
        seed = (int(picked_day) * 31 + sum(ord(c) for c in meal)) % options
        return seed

    if any(by_meal.values()):
        st.caption(
            f"Three options per meal. **{len(_logged)} of 3 meals were logged "
            f"today** — a ✓ marks the one the patient had. A meal with no tick "
            "was not eaten, and that is what the nutrition score is built from.")

        for column, (meal, items) in zip(st.columns(len(by_meal)),
                                         by_meal.items()):
            with column:
                eaten = _eaten_index(meal, len(items))
                st.markdown(f"##### {meal.title()}")
                st.caption("✓ eaten" if eaten is not None
                           else ":red[not logged today]")
                if not items:
                    st.caption("No recipe for this band.")
                    continue

                for index, recipe in enumerate(items):
                    was_eaten = index == eaten
                    row = sim_recipes.macro_row(recipe, meal)
                    title = recipe.get("food_title", "(untitled)")
                    with st.container(border=True):
                        st.markdown(f"**✓ {title}**" if was_eaten
                                    else f":gray[{title}]")
                        carbs = row.get("Carbohydrates")
                        macros = " · ".join(
                            f"{k} {row[k]:g}" for k in
                            ("Calories", "Protein", "Total Fat") if k in row)
                        st.markdown(
                            (f"**Carbs {carbs:g} g**" if carbs is not None
                             else "")
                            + (f"  \n:gray[{macros}]" if macros else ""))

                        if recipe.get("short_description") or recipe.get("summary"):
                            with st.expander("Description"):
                                st.write(recipe.get("short_description")
                                         or recipe.get("summary"))
                                if recipe.get("why_this_works_for_you"):
                                    st.info(recipe["why_this_works_for_you"])

                        ingredients = recipe.get("ingredients")
                        if isinstance(ingredients, list) and ingredients:
                            with st.expander("Ingredients"):
                                st.dataframe(pd.DataFrame([{
                                    "Ingredient": i.get("name"),
                                    "Qty": i.get("quantity"),
                                    "Unit": i.get("unit")}
                                    for i in ingredients
                                    if isinstance(i, dict)]),
                                    hide_index=True, width="stretch")

                        steps = recipe.get("instructions")
                        if isinstance(steps, list) and steps:
                            with st.expander("Instructions"):
                                for step in steps:
                                    st.markdown(str(step))

                        for field, colour in (("vitamin_mineral_claims", "green"),
                                              ("diet_labels", "blue"),
                                              ("cautions", "red")):
                            tags = recipe.get(field) or []
                            if isinstance(tags, list) and tags:
                                st.markdown(" ".join(
                                    f":{colour}-badge[{t}]" for t in tags
                                    if isinstance(t, str)))


    _held = sim_recipes.cache_contents()
    if _held:
        st.caption("Held locally, so a lost database does not empty this tab: "
                   + " · ".join(f"{row['phase'].title()} {row['recipes']} "
                                f"({row['fetched_at'][:10]})"
                                for row in _held if row["phase"]))

    if not any(by_meal.values()):
        with st.expander("Which bands the pool has been generated for"):
            coverage_info = sim_recipes.pool_coverage()
            if coverage_info["status"] != "ok":
                st.caption(coverage_info["reason"])
            else:
                st.dataframe(pd.DataFrame([{
                    "Phase": r["action_space"].title(), "Sex": r["gender"],
                    "Age": r["age_band"], "Weight kg": r["weight_band_kg"],
                    "Height cm": r["height_band_cm"], "Meal": r["meal_type"],
                    "Recipes": r["n"]} for r in coverage_info["rows"]]),
                    hide_index=True, width="stretch")



# ===========================================================================
# LAYER 4
# ===========================================================================
with tab["Layer 4 · Alert"]:
    _insight_block(day_view["layers"]["Layer 4"], "Layer 4")

    # The note is only shown once the model has written it. The deterministic
    # template is the service's own default and it is what a ward would get if
    # the network were down — but showing it here, before anyone asked, put a
    # block of machine-assembled text on screen that reads like the finished
    # article and is not. With no key reachable there is nothing else to offer,
    # so it is shown then, labelled as the fallback it is.
    if _written_key in _written or not _key():
        with st.container(border=True):
            st.markdown(layer4.get("narrative") or "_Nothing to report today._")

        st.markdown("**How it was written**")
        if layer4.get("source") == "llm":
            st.caption(
                f"`{LLM_MODEL}` phrased it, then it was checked automatically "
                "before being shown — it may not use a number the layers did "
                "not produce, recommend anything they did not flag, say "
                "anything about the wound itself, or reassure while something "
                "is flagged. It passed every check.")
        elif layer4.get("source") == "deterministic_after_guardrail_rejection":
            st.caption(
                f"`{LLM_MODEL}` phrased it, the wording broke one of those "
                "checks, and it was thrown away. What is shown above is the "
                "standard text.")
            for violation in layer4["guardrail_violations"]:
                st.caption(f"· {violation}")
        else:
            st.caption(
                "No API key was found, so this is the fixed template the "
                "service falls back to. Set `ANTHROPIC_API_KEY` or add it to "
                "`.streamlit/secrets.toml` and the model will write it.")
    else:
        left, right = st.columns([1, 3])
        if left.button("Generate insights", type="primary", width="stretch"):
            with st.spinner(f"{LLM_MODEL} is writing this day's note…"):
                _written[_written_key] = _llm_day(
                    scenario, picked_day, timeline, _key())
            st.rerun()
        right.caption(
            f"`{LLM_MODEL}` writes the note for this day from the three layers "
            "above. It takes a few seconds, so it is asked for rather than run "
            "on every click.")