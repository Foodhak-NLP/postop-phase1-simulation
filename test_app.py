"""Render the app for every source and assert nothing raised.

Streamlit hides the inactive tabs rather than skipping them, so one render
executes all six. A table Arrow cannot serialise is caught from the logs,
because Streamlit neither raises nor shows an error element for it.
"""
from __future__ import annotations

import logging
import os
import sys

import sim_paths  # noqa: F401
import sim_timeline as T
from streamlit.testing.v1 import AppTest

# Layer 4 is always on in the app, and `sim_paths` loads a key out of the
# service's .env. These tests check that the tabs render, not that the model
# writes well, so take the key away: otherwise every alerting day in the suite
# is a live API call, and a rendering test that costs money and network is one
# nobody runs. The deterministic narrative exercises the same UI path.
os.environ.pop("ANTHROPIC_API_KEY", None)

FAILURES: list[str] = []


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


# The stay is fixed at the six weeks Phase 1 covers — there is no window to
# choose, so every render is a 42-day replay. Streamlit's cache is process-wide,
# so scenarios repeat cheaply within a run.
def render(**state) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=900)
    for key, value in state.items():
        at.session_state[key] = value
    capture, root = Capture(), logging.getLogger()
    root.addHandler(capture)
    try:
        at.run()
    finally:
        root.removeHandler(capture)
    at._captured = capture.records                      # noqa: SLF001
    return at


def assert_clean(at: AppTest, label: str) -> None:
    check(f"{label} renders", not at.exception,
          "; ".join(str(e.value) for e in at.exception))
    noisy = [line for line in getattr(at, "_captured", [])
             if "Arrow" in line or "Traceback" in line or "Serialization" in line]
    check(f"{label} serialises every table", not noisy,
          " | ".join(line.splitlines()[0] for line in noisy[:2]))


print("\nevery surgery")
check("three surgeries, not a catalogue", len(T.SURGERY_ORDER) == 3,
      str(T.SURGERY_ORDER))
for surgery in T.SURGERY_ORDER:
    assert_clean(render(surgery=surgery), surgery)

print("\nrecovery patterns")
for name in T.RECOVERY_ORDER:
    assert_clean(render(recovery=name), f"recovery/{name}")

print("\nintake patterns")
for name in T.INTAKE_ORDER:
    assert_clean(render(intake_pattern=name), f"intake/{name}")

print("\nconditions")
for label, state in (
    ("diabetes", {"conditions": ["Diabetes"]}),
    ("hypertension only", {"conditions": ["Hypertension"]}),
    ("a typed-in condition", {"conditions": ["Gout"]}),
):
    assert_clean(render(**state), label)

print("\nscrubbing to any day")
for day in (0, 1, 8, 13):
    assert_clean(render(surgery="open_bowel_resection", picked_day=day),
                 f"day {day}")
# One full six-week render, because the calendar spans three month rows there
# and the arc charts get their widest input.
at42 = render(days=42, surgery="open_bowel_resection", picked_day=41)
assert_clean(at42, "six weeks, last day")
_six = [int("".join(c for c in b.label if c.isdigit()))
        for b in at42.button if (b.key or "").startswith("cal_")]
check("a six-week calendar runs 0 to 41 without restarting",
      _six == list(range(42)), str(_six[28:36]))

print("\nstructure")
at = render(surgery="bk_amputation")
labels = [t.label for t in at.tabs]
check("five tabs: a day view and one per layer",
      labels == ["Today", "Layer 1 · Phase", "Layer 2 · Deviations",
                 "Layer 3 · Nutrition", "Layer 4 · Alert"],
      str(labels))
# The four blurbs are rendered as captions, one per layer tab. Streamlit
# renders hidden tabs too, so all four are present in a single run.
_captions = " ".join(c.value for c in at.caption)
check("every layer tab says what its layer does",
      _captions.count("**The result:**") == 4,
      str(_captions.count("**The result:**")))
check("and each blurb is the right one for its layer",
      all(phrase in _captions for phrase in
          ("Which stage of healing", "Is this patient healing as expected",
           "Is this patient being fed enough", "What should the physician")),
      "a layer blurb is missing or reworded")
check("the layer tabs carry no document section references",
      "section 5.3" not in _captions.lower()
      and "section 6" not in _captions.lower()
      and "section 7" not in _captions.lower(),
      "a section reference survived")
check("the seven-criteria table is gone from Today",
      not any("seven criteria" in m.value.lower() for m in at.markdown),
      "the gate table is still rendered")
check("the limitations expander is gone",
      not any("cannot see" in e.label.lower() for e in at.expander),
      str([e.label for e in at.expander]))
check("the export block is gone",
      not any("Prepare export" in b.label for b in at.button),
      "export survived")
check("the NSS is explained where it is used",
      any("What is the NSS" in e.label for e in at.expander),
      str([e.label for e in at.expander]))
check("the app opens straight onto a generated patient — no source to pick",
      not at.radio, str([r.label for r in at.radio]))
_sidebar = ([w.label for w in at.sidebar.selectbox]
            + [w.label for w in at.sidebar.number_input]
            + [w.label for w in at.sidebar.multiselect]
            + [w.label for w in at.sidebar.text_input])
check("the sidebar holds the patient, the course and how they eat",
      sorted(_sidebar) == sorted([
          "Surgery type", "Age", "Sex", "Height (cm)", "Weight (kg)",
          "Allergens", "Dietary restrictions", "Existing conditions",
          "Foodhak user id", "How it goes", "Meals actually eaten"]),
      str(sorted(_sidebar)))
check("the sidebar sections are patient, healing, diet",
      [h.value for h in at.sidebar.subheader]
      == ["Patient", "Wound healing", "Diet and conditions"],
      str([h.value for h in at.sidebar.subheader]))
check("the sidebar carries no essay about the NSS",
      not any("worked out from what was eaten" in c.value
              for c in at.sidebar.caption),
      "the NSS explanation is still in the sidebar")
check("there is no date field to fill in",
      not at.date_input, str([d.label for d in at.date_input]))
_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
_months = ("Jan ", "Feb ", "Mar ", "Apr ", "Jun ", "Jul ", "Aug ", "Sep ",
           "Oct ", "Nov ", "Dec ")
check("nothing on screen is dated — the unit is the post-op day",
      not any(month in _text for month in _months),
      next((m for m in _months if m in _text), ""))
_cell_labels = [b.label for b in at.button if (b.key or "").startswith("cal_")]
check("calendar cells count post-op days, not days of the month",
      [int("".join(c for c in label if c.isdigit())) for label in _cell_labels]
      == list(range(42)), str(_cell_labels[:6]))
check("no model toggle — Layer 4 always writes with the model",
      not any("writes with" in t.label for t in at.toggle),
      str([t.label for t in at.toggle]))
cells = [b for b in at.button if (b.key or "").startswith("cal_")]
check("a day is picked from the calendar grid", bool(cells),
      str([b.key for b in at.button][:8]))
check("one calendar cell per post-op day", len(cells) == 42, str(len(cells)))

_labels = [n.label for n in at.number_input]
check("the day's inputs are the payload's own fields",
      all(name in _labels for name in
          ("CRP", "WBC", "Glucose", "Fasting gluc.", "Prealbumin", "Albumin",
           "Zinc", "Mean temp", "Mean HR", "Mean SpO2", "Meals logged")),
      str(_labels))
check("vitals are means, not a time series",
      not any("Peak" in label for label in _labels), str(_labels))
check("pain is one mean value, not four timestamps",
      "Pain (mean)" in _labels
      and not any(clock in _labels
                  for clock in ("06:00", "12:00", "18:00", "00:00")),
      str(_labels))
check("there is no Apply button — a changed field re-runs on its own",
      not any("Apply" in b.label for b in at.button),
      str([b.label for b in at.button if not (b.key or "").startswith("cal_")]))

print("\nthe intro")
at = render()
check("the intro is there, and closed until it is wanted",
      any("four stages of healing" in e.label for e in at.expander),
      str([e.label for e in at.expander]))
check("the four questions are inside it",
      all(q in " ".join(m.value for m in at.markdown) for q in
          ("Which stage is this wound in?", "Is it going the way it should?",
           "Is the patient being fed enough to heal?",
           "What should the physician be told?")),
      "a question is missing")
check("the calendar and the inputs each have a heading",
      {"**Post-op calendar**"} <= {m.value for m in at.markdown}
      and any(m.value.startswith("**Inputs for post-op day")
              for m in at.markdown),
      "a block is unlabelled")

print("\nrecipes show what was eaten")
at = render(intake_pattern="all", picked_day=20)
_eaten = [m.value for m in at.markdown if m.value.startswith("**✓ ")]
check("one recipe per meal is marked as eaten", len(_eaten) == 3, str(_eaten))
check("the others are shown but not marked",
      any(m.value.startswith(":gray[") for m in at.markdown))
check("carbohydrate is the macro pulled out",
      any(m.value.startswith("**Carbs ") for m in at.markdown),
      "no carbs line")
check("ingredients, description and method are each behind a dropdown",
      {"Description", "Ingredients", "Instructions"}
      <= {e.label for e in at.expander}, str(sorted({e.label for e in at.expander})))

at = render(intake_pattern="poor", picked_day=20)
_marked = [m.value for m in at.markdown if m.value.startswith("**✓ ")]
check("a patient who eats one meal has one recipe marked",
      len(_marked) <= 1, str(_marked))
check("and the meals they skipped say so",
      any("not logged today" in c.value for c in at.caption),
      "a skipped meal is not called out")

print("\nthe model is on demand, not on every click")
at = render(picked_day=20)
assert_clean(at, "a day before generating")
check("nothing is generated until it is asked for",
      not dict(at.session_state["written_days"]),
      str(list(dict(at.session_state["written_days"]))))
# This suite runs with the key removed, so it exercises the fallback branch:
# with nothing to ask, the template is shown rather than an empty tab, and it
# says why. The other branch — a key present, so nothing until the button is
# pressed — costs an API call per render and is checked by hand plus by
# `test_replay`'s sim_alert cases.
check("with no key the template is shown rather than an empty tab",
      any("DAILY SUMMARY" in m.value or "ALERT" in m.value
          for m in at.markdown),
      "nothing at all is on the Layer 4 tab")
check("and it says it is the fallback",
      any("No API key was found" in c.value for c in at.caption),
      "the fallback is unlabelled")
check("there is a button to ask for it",
      any(b.label == "Generate insights" for b in at.button)
      or not os.environ.get("ANTHROPIC_API_KEY"),
      str([b.label for b in at.button if not (b.key or "").startswith("cal_")]))

print("\nediting a day through the app")
# Overrides are derived from the widgets rather than stored by a button, so the
# way to edit one here is the way a user does it: change the field and re-run.
at = render(surgery="open_bowel_resection", picked_day=8)
assert_clean(at, "before the edit")
check("looking at a day does not count as editing it",
      not dict(at.session_state["overrides"]),
      str(dict(at.session_state["overrides"])))
check("and nothing is flagged yet",
      not any("Critical" in m.value for m in at.markdown))

at.number_input(key="tp_8").set_value(38.4).run()
at.number_input(key="fh_8").set_value(9.0).run()
assert_clean(at, "after the edit")
check("changing a field records the edit without a button press",
      list(dict(at.session_state["overrides"])) == [8],
      str(list(dict(at.session_state["overrides"]))))
check("and the edit reaches the layers",
      any("Critical" in m.value or "critical" in m.value for m in at.markdown),
      "healing status did not change")

print()
if FAILURES:
    print(f"{len(FAILURES)} failed")
    for line in FAILURES:
        print(f"  - {line}")
    sys.exit(1)
print("all passed")
