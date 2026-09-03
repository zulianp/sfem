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

The check counts a plan as consumed when an emitter or backend mentions either
the type itself or one of the planning-layer factories that returns it.  A
factory counts because consuming ``local_kernel_plan_for(...)`` is consuming the
plan -- the first version of this test missed exactly that and under-reported.
It over-counts if anything, since a mention is not the same as being driven by
it, which is the safe direction for a ratchet.
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
#:
#: Removed so far:
#:   LocalKernelPlan, MeshKernelPlan -- which files a kernel produces and what
#:   they are called, including the rule that a prefix already naming its
#:   element does not get a second one.  The emitters had rebuilt both names
#:   from format strings.
#:
#:   CRSAssemblyPlan -- the row-pointer, column-index and value stream names
#:   and the reduction policy of the CRS scatter.  The emitter had them as
#:   string literals; it now spells what the plan defines.  The other five
#:   assembly plans follow the same shape and are the obvious next step.
UNCONSUMED_PLAN_TYPES = frozenset(
    set(STRUCTURAL_PLAN_TYPES)
    - {
        "LocalKernelPlan",
        "MeshKernelPlan",
        "CRSAssemblyPlan",
        # The BSR scatter now spells its stream names and reduction from
        # BSRAssemblyPlan instead of hardcoding them.  BSR is the format
        # vector problems use, so it is the assembly plan worth connecting
        # first.
        "BSRAssemblyPlan",
        # The residual emitter no longer invents the strings "affine" and
        # "isoparametric" -- 81 hand-written literals -- and checks the
        # geometry plans it is handed against the modes it spells, so a plan
        # that changed its mind stops generation instead of being ignored.
        "GeometryPlan",
        # Which streams cross a local kernel's boundary, and in what order,
        # is now decided by plans.streams.local_kernel_stream_plans.  About
        # seventy lines of conditionals left the emitter, which keeps only
        # _declare_stream -- how C writes a declaration down.  This is the
        # first of these connections that moved logic rather than re-routing
        # a name.
        "DataStreamPlan",
    }
)


#: Planning-layer factories that return one of the structural plans.  Consuming
#: a factory is consuming the plan it returns.
PLAN_FACTORIES = {
    "LocalKernelPlan": ("local_kernel_plan_for",),
    "MeshKernelPlan": ("mesh_kernel_plan_for_element", "mesh_kernel_plan_from_context"),
}


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
        tokens = (plan_type,) + PLAN_FACTORIES.get(plan_type, ())
        readers = [
            name
            for name, source in sources
            if any(token in source for token in tokens)
        ]
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
            7,
            "the number of unread structural plans changed to %d; update this "
            "expectation deliberately, and say why in the commit" % unconsumed,
        )


if __name__ == "__main__":
    unittest.main()
