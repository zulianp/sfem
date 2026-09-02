"""A plan type nobody reads is a decision nobody moved.

The planning layer describes a kernel: its phases, its data streams, its block
structure, its geometry, and how an element matrix scatters into each supported
matrix format.  If an emitter does not read a plan, then whatever that plan
describes is still being decided inside the emitter, and the plan is at best
documentation that can drift from the code it claims to describe.

This module tracks exactly that number.  The thirteen plan types named in
``KERNELS.md`` as carrying kernel structure are listed here; every one that no
emitter or backend reads is pinned.  Entries may be removed as decisions move
up, and never added -- so the list is a measure of how much of the emission
layer's job is still misplaced, and it can only shrink.

The check is deliberately shallow: does the type's name appear anywhere in an
emitter or backend module.  That over-counts consumption if anything -- a
mention is not the same as being driven by it -- which is the safe direction
for a ratchet.
"""

import os
import unittest


FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSUMER_PACKAGES = ("emitters", "backends")

#: The plan types KERNELS.md names as describing kernel structure.
STRUCTURAL_PLAN_TYPES = (
    "LocalPhasePlan",
    "MeshPhasePlan",
    "DataStreamPlan",
    "BlockPlan",
    "GeometryPlan",
    "LocalKernelPlan",
    "MeshKernelPlan",
    "CRSAssemblyPlan",
    "BSRAssemblyPlan",
    "DIAAssemblyPlan",
    "COOAssemblyPlan",
    "PatchAssemblyPlan",
    "BlockDiagSymAssemblyPlan",
)

#: Types no emitter or backend reads yet.  Each one is a decision still made
#: inside the emission layer:
#:
#:   the seven structural plans   loop phases, streams, blocks, geometry and
#:                                kernel naming are re-derived by the emitters
#:   the six assembly plans       CRS/BSR/DIA/COO/patch scatter is emitted from
#:                                string templates selected by substring tests
#:
#: Removing an entry means that decision now comes from the plan.  See open
#: points 1 and 3 in ARCHITECTURE.html.
UNCONSUMED_PLAN_TYPES = frozenset(STRUCTURAL_PLAN_TYPES)


def _consumer_sources():
    for package in CONSUMER_PACKAGES:
        directory = os.path.join(FRAMEWORK_ROOT, package)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            with open(path, encoding="utf-8") as handle:
                yield "%s/%s" % (package, name), handle.read()


def consumed_plan_types():
    """Which structural plan types are mentioned by an emitter or backend."""
    sources = list(_consumer_sources())
    consumed = {}
    for plan_type in STRUCTURAL_PLAN_TYPES:
        readers = [name for name, source in sources if plan_type in source]
        if readers:
            consumed[plan_type] = readers
    return consumed


class PlansAreConsumedTest(unittest.TestCase):
    def test_no_plan_type_stops_being_consumed(self):
        """A plan that had a reader must keep one."""
        consumed = consumed_plan_types()
        regressed = sorted(
            plan_type
            for plan_type in STRUCTURAL_PLAN_TYPES
            if plan_type not in UNCONSUMED_PLAN_TYPES and plan_type not in consumed
        )
        self.assertEqual(
            regressed,
            [],
            "plan type(s) lost their only consumer, so the decision they carry "
            "moved back into the emission layer: %s" % ", ".join(regressed),
        )

    def test_unconsumed_list_shrinks_and_never_goes_stale(self):
        """A plan that gains a reader must leave the list."""
        consumed = consumed_plan_types()
        stale = sorted(set(UNCONSUMED_PLAN_TYPES) & set(consumed))
        lines = [
            "%s (now read by %s)" % (plan_type, ", ".join(consumed[plan_type]))
            for plan_type in stale
        ]
        self.assertEqual(
            stale,
            [],
            "these plan types now have a consumer; remove them from "
            "UNCONSUMED_PLAN_TYPES:\n  " + "\n  ".join(lines),
        )

    def test_reports_how_much_of_the_emitters_job_is_still_misplaced(self):
        """Not a threshold -- a visible count that should fall over time."""
        unconsumed = len(UNCONSUMED_PLAN_TYPES)
        self.assertLessEqual(
            unconsumed,
            len(STRUCTURAL_PLAN_TYPES),
            "UNCONSUMED_PLAN_TYPES lists a type that is not a structural plan",
        )
        self.assertEqual(
            unconsumed,
            13,
            "the number of unread structural plans changed to %d; update this "
            "expectation deliberately, and say why in the commit" % unconsumed,
        )


if __name__ == "__main__":
    unittest.main()
