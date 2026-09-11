"""The vendored service must stay identical to the real one.

`phase1_service/` exists so the app can be deployed somewhere that cannot see
`Postop-Phase1/Staging`. A copy that drifts from the original is worse than no
copy at all: the deployed app would show numbers the service no longer
produces, and nothing would say so.

So every vendored file is compared byte for byte, with exactly one exception —
`phase1_config.py`, which is deliberately different because the original holds
credentials and this one must not. That file is checked the other way round:
that it contains no credential at all.

Skips when the real service is not present, because then there is nothing to
compare against.
"""
from __future__ import annotations

import ast
import re
import sys

import sim_paths

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


if sim_paths.SOURCE != "service":
    print("\nskip — the real service is not beside this checkout, so there is "
          "nothing to compare the vendored copy against.")
    sys.exit(0)

print("\nvendored copy against the real service")
vendored = sorted(p.name for p in sim_paths.VENDORED.glob("*.py"))
check("the vendored directory is not empty", bool(vendored))

for name in vendored:
    original = sim_paths.SERVICE / name
    copy = sim_paths.VENDORED / name
    if name == "phase1_config.py":
        continue
    check(f"{name} still exists in the service", original.is_file(),
          f"{original} is gone — the vendored copy is now the only one")
    if original.is_file():
        check(f"{name} is byte-identical",
              original.read_bytes() == copy.read_bytes(),
              "the service has moved on; re-vendor it")

print("\nthe one file that is meant to differ")
config = (sim_paths.VENDORED / "phase1_config.py").read_text()
check("the vendored config differs from the service's",
      config != (sim_paths.SERVICE / "phase1_config.py").read_text())

# A password, a host, or a key that got pasted back in. The docstring names the
# variables to set but shows no specimen value, so there is nothing to exempt:
# anything matching below is a real credential and must not be here.
PLACEHOLDERS = ()
SECRETS = (
    (r"sk-ant-[A-Za-z0-9_\-]{12,}", "an Anthropic key"),
    (r"[A-Za-z0-9.\-]+\.rds\.amazonaws\.com", "a database host"),
    (r"postgresql://[^\s\"']*:[^\s\"'@]+@[^\s\"']+", "a database URL with a password"),
    (r"""DB_PASS\s*=\s*["'][^"']+["']""", "a hardcoded password"),
)
for pattern, what in SECRETS:
    found = next((m for m in re.finditer(pattern, config)
                  if not any(ph in m.group(0) for ph in PLACEHOLDERS)), None)
    check(f"the vendored config carries no {what}", not found,
          found.group(0)[:28] + "…" if found else "")

# Parsed rather than string-matched: the comment inside DEFAULTS says the key
# is deliberately absent, and a substring search cannot tell a comment from a
# value.
_defaults = next(
    (node.value for node in ast.walk(ast.parse(config))
     if isinstance(node, ast.Assign)
     and any(getattr(t, "id", None) == "DEFAULTS" for t in node.targets)), None)
_keys = [k.value for k in _defaults.keys] if isinstance(_defaults, ast.Dict) else []
check("DEFAULTS was found and parsed", bool(_keys), str(_keys))
check("and it has no database URL to fall back to",
      "RECIPE_POOL_DATABASE_URL" not in _keys, str(_keys))

print()
if FAILURES:
    print(f"{len(FAILURES)} failed")
    for line in FAILURES:
        print(f"  - {line}")
    sys.exit(1)
print("all passed")
