"""The form-layer shape of every material, captured and checked.

Phase 1 of the energy/residual unification re-keys form blocks per component so
the block metadata and the lowered fields agree, and the parallel
``residual_fields`` / ``residual_expressions`` / ``jacobian_action_blocks``
representation can be retired.  That representation has 49 consumers across
eight files against 19 for the form-order accessors it replaces, so the change
needs to arrive as a reviewable diff rather than as a claim that nothing moved.

This is the full sweep.  It lives here rather than in the unit tests because
lowering every material takes tens of minutes -- the 2-form of a hyperelastic
material in three dimensions is a large symbolic differentiation -- which is
the same reason ``codegen_snapshot`` and ``reproducibility`` are tools.  The
fast subset, the three materials that carry the three distinct shapes, stays in
``tests/test_form_keying_baseline.py`` so the distinction cannot regress
unnoticed between sweeps.

    python -m codegen.framework.tools.form_keying capture
    python -m codegen.framework.tools.form_keying check

``capture`` rewrites the baseline and is a deliberate act: the diff it produces
is the statement of exactly which materials changed shape.
"""

import argparse
import json
import os
import sys

from codegen.framework.tests.test_form_keying_baseline import (
    BASELINE_PATH,
    collect,
)


def _load():
    if not os.path.exists(BASELINE_PATH):
        return {}
    with open(BASELINE_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _write(snapshot):
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _summarise(snapshot):
    lines = []
    for material, dims in sorted(snapshot.items()):
        for dim, equations in sorted(dims.items()):
            for equation, record in sorted(equations.items()):
                blocks = (record.get("orders", {}).get("ONE") or {}).get("blocks")
                rows = [row for row, _ in blocks] if blocks else []
                fields = record.get("residual_fields") or []
                agree = "" if sorted(rows) == sorted(fields) else "   <-- keyed differently"
                lines.append(
                    "  %-36s dim=%s %-8s 1-form %-22s fields %s%s"
                    % (material, dim, record.get("kind"), str(rows), fields, agree)
                )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("capture", "check"))
    args = parser.parse_args(argv)

    measured = collect()
    if args.action == "capture":
        _write(measured)
        print("captured the form-layer shape of %d materials" % len(measured))
        print("\n".join(_summarise(measured)))
        return 0

    recorded = _load()
    if recorded == measured:
        print("the form layer matches the baseline for %d materials" % len(measured))
        return 0
    moved = sorted(
        name
        for name in set(recorded) | set(measured)
        if recorded.get(name) != measured.get(name)
    )
    print("the form layer moved for %d material(s):" % len(moved))
    for name in moved:
        print("    %s" % name)
    print("\nrun 'capture' to record it, and review the diff")
    return 1


if __name__ == "__main__":
    sys.exit(main())
