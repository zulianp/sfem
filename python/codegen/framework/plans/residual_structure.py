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

from codegen.framework.forms.forms import FormOrder
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


def publishes_scalar_jacobian_action(system, dependencies):
    """Whether this system publishes a packed scalar Jacobian-action kernel.

    Two conditions, and both are about what the kernel *is* rather than how it
    is spelled: the system carries one field, so the action is scalar, and the
    form reads a direction, so there is an action to take at all.  A coupled
    system's blocks are handled by the block kernels instead, and a form with no
    direction has no Jacobian action to publish.

    `emitters/residual_codegen.py` spelled the pair as `len(system.fields) != 1
    or not dependencies.direction` at the head of both packed jacobian-action
    sources, each returning an empty list.  That is `plans.dependencies`'
    `publishes_kernel` one level up -- a form that contributes nothing publishes
    no kernel -- and it belongs beside the rest of a residual kernel's structure
    rather than being restated wherever a kernel begins.

    Note that a third site, `_simplex_metric_scalar_affine_fast_path_body`,
    looks like a third copy and is not: it pairs the single-field clause with
    `rule.n_qp != 1`, which is a question about the quadrature rule.  Only the
    first clause is shared, and sharing one clause is not sharing a question.

    This answers False for everything currently generated.  Instrumenting both
    call sites across a full regeneration recorded 148 entries and not one True:
    every residual material still in the tree -- navier_stokes, two_phase_flow,
    mooney_rivlin_kelvin_voigt_newmark -- reaches this emitter through paths
    where the field count is not one, and the single-field residuals that would
    have taken it, laplace among them, are written as energies now and never
    arrive here at all.

    That is a scope reduction, not a decision against the capability, and the
    two sources it guards are kept for the same reason: a single-field residual
    would want them back.  It is recorded because it is the shape OP 26 found in
    the Laplacian special cases -- a generator left reachable but never reached,
    where a byte-identical regeneration says nothing about whether its body
    still works.
    """
    return len(system.fields) == 1 and bool(dependencies.direction)


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
