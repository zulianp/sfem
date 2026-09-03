"""The phase and block structure of a residual kernel.

A residual kernel has a fixed shape: it gathers, computes geometry, calls the
local kernel once per block, and scatters.  Each local call evaluates the
trial functions, transforms to the reference element, evaluates the material
and contracts against the test functions.  That shape is a planning decision --
it follows from what a residual kernel *is*, not from how any of it is
spelled.

These factories used to live in ``pipeline/driver.py``.  Building a plan is
not orchestration, and putting them there meant the emission layer could not
reach them even in principle: the driver sits above emission, so an emitter
importing from it would be a downward edge.  Moving them here makes the
structure available to everything that should be subordinate to it, the
driver included.
"""

from codegen.framework.symbolic.forms import FormOrder
from codegen.framework.plans.generation import (
    BlockPlan,
    LocalPhase,
    LocalPhasePlan,
    MeshPhase,
    MeshPhasePlan,
)


def residual_local_phase_plans():
    """The four phases every residual local kernel runs, in order."""
    return (
        LocalPhasePlan(LocalPhase.EVALUATE_TRIAL),
        LocalPhasePlan(LocalPhase.TRANSFORM_REFERENCE),
        LocalPhasePlan(LocalPhase.EVALUATE_MATERIAL),
        LocalPhasePlan(LocalPhase.CONTRACT_TEST),
    )


def residual_local_phases():
    return tuple(plan.phase for plan in residual_local_phase_plans())


def residual_mesh_phase_plans(blocks):
    """The four mesh phases, with the local call carrying its blocks."""
    return (
        MeshPhasePlan(MeshPhase.GATHER),
        MeshPhasePlan(MeshPhase.GEOMETRY),
        MeshPhasePlan(MeshPhase.LOCAL_CALL, blocks=tuple(blocks)),
        MeshPhasePlan(MeshPhase.SCATTER),
    )


def residual_mesh_phases():
    return tuple(plan.phase for plan in residual_mesh_phase_plans(()))


def block_plan_from_form_block(block):
    """The plan for one Jacobian block, from the lowered form block.

    ``name`` here is the authority for what the block is called; emitters that
    name a kernel or a diagnostic after a block take it from this rather than
    reaching into the form.
    """
    return BlockPlan(
        block.name,
        block.row_field,
        block.column_field or "",
        block.order,
        local_phase_plans=residual_local_phase_plans(),
    )


def block_plans_from_form_collection(collection):
    return tuple(block_plan_from_form_block(block) for block in collection.blocks)


def jacobian_block_plan(block):
    """The plan for one Jacobian block of a residual system.

    A second factory rather than a tolerant one, because these are genuinely
    two different things.  ``block_plan_from_form_block`` takes a lowered form
    block, which carries its own order; a ``ResidualJacobianBlock`` is always
    second order and derives its name from the two fields it couples.  Making
    one factory accept both by probing for attributes would hide that.
    """
    return BlockPlan(
        block.name,
        block.row_field,
        block.column_field,
        FormOrder.TWO,
        local_phase_plans=residual_local_phase_plans(),
    )
