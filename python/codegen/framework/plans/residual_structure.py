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
from codegen.framework.plans.dependencies import publishes_kernel
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


#: The forms a residual unit can publish, in the order it has always published
#: them.
RESIDUAL_FORMS = ("residual", "jacobian_action")


def unit_exists_for_the_merit(unit_name):
    """Whether this unit is the one carrying the material's whole residual.

    Its reason to exist is the residual merit: one kernel contracting one form
    over a node's complete value.  Everything else a residual unit would
    normally publish is something the material's own units already publish, so
    this predicate is the left-hand side of three refusals below -- the forms,
    the block kernels and the projected apply.

    One predicate rather than the same comparison written at each of them.  The
    comparison was written once, at the forms, and the other two paths reached
    emission by different names and emitted a full second copy of the material
    unnoticed.
    """
    from codegen.framework.forms.equations import TOTAL_RESIDUAL_UNIT_NAME

    return str(unit_name) == TOTAL_RESIDUAL_UNIT_NAME


def published_residual_forms(form_dependencies, unit_name=""):
    """Which of `RESIDUAL_FORMS` a unit publishes.

    Two conditions, and neither is emission's to decide.

    A form whose coefficients are all structurally zero contributes nothing and
    is absent from the sequence, so nothing below emits, declares or dispatches
    to it -- see `plans.dependencies.publishes_kernel`.

    And a unit carrying the material's *whole* residual publishes neither form.
    It exists so that one kernel can contract the merit over one form; the
    residual and the Jacobian action it would otherwise emit are what the
    material's own units already emit, and a second set computing the same
    numbers is a redundant path with nothing to keep the two in step.  For a
    single-unit material they would be the same arithmetic twice; for a
    multi-unit one they are a second spelling of the sum.
    """
    if unit_exists_for_the_merit(unit_name):
        return ()
    return tuple(
        form
        for form in RESIDUAL_FORMS
        if form in form_dependencies and publishes_kernel(form_dependencies[form])
    )


def published_block_kernels(unit_name, blocks):
    """Which of a coupled unit's blocks are published as kernels of their own.

    A coupled system emits a kernel per block so an assembly can fill one block
    at a time.  A unit carrying the material's whole residual for the merit
    publishes neither the residual nor the Jacobian action -- see
    `published_residual_forms` -- so it has no blocks to publish either: a
    block kernel *is* one of those forms, restricted to a block.

    Stated here rather than left to the refusal above, because the two are not
    reached by the same name.  A unit's blocks become units of their own, named
    after the block -- `total_form_1_p_c`, `total_form_2_p_c_p_w` -- and those
    names are not `TOTAL_RESIDUAL_UNIT_NAME`, so they walked past the check and
    emitted a full second set of kernels for every coupled material.  For
    two_phase_flow that was 123 files standing behind one merit kernel.
    """
    if unit_exists_for_the_merit(unit_name):
        return ()
    return tuple(blocks)


def publishes_inexact_apply(unit_name, material_publishes_inexact_apply):
    """Whether this unit publishes the projected (inexact) Jacobian action.

    The projected apply is a *variant* of the Jacobian action, so a unit that
    publishes no Jacobian action publishes no variant of one.  The material's
    flag says the material has a lowering for it; it does not say that each of
    the material's units has a form for the variant to be a variant of, and
    applying it per material rather than per unit is what emitted a projected
    tangent for a Jacobian nobody publishes -- twelve files per vector
    material, standing behind a merit kernel that does not use them.
    """
    if unit_exists_for_the_merit(unit_name):
        return False
    return bool(material_publishes_inexact_apply)


def published_local_kernel_files(published_forms, unit_name=""):
    """Whether this unit emits an element-local kernel header: one, or none.

    A sequence rather than a predicate, so emission walks it instead of testing
    it.  A unit that publishes no form has no element-local kernel to put in a
    header, and an empty header is not the answer -- a form that contributes
    nothing publishes no kernel rather than an empty one, and the backend
    contract checks that the header it is handed is a real templated kernel.

    Keyed on the unit rather than on whether a matrix format was requested.
    The material's format plan reaches every one of its units, so a unit that
    assembles nothing would otherwise be handed one and emit a header for an
    assembly it does not do.  The unit carrying the whole residual for the
    merit is exactly that case: its kernel is the patch merit kernel, which
    lives in the per-element source.
    """
    if not published_forms:
        return ()
    return ("local",)


def published_mesh_operator_files(published_forms, patch_merit_kernels=()):
    """Whether this unit emits a per-element mesh operator source: one, or none.

    A unit that publishes no form and no patch merit kernel has nothing to put
    in one.  That is not hypothetical: the unit carrying the material's whole
    residual publishes no form by design, and on a device target it publishes no
    patch kernel either, because that kernel is host-shaped.  Emitting a source
    whose only content is its own includes leaves the CUDA backend rejecting a
    translation unit with no kernels in it -- correctly.
    """
    if published_forms or patch_merit_kernels:
        return ("operator",)
    return ()


def published_patch_merit_kernels(element_type, unit_name="", mixed=False,
                                  has_parallel_region=True):
    """The node-centric merit kernels this unit publishes: one, or none.

    A sequence rather than a predicate, for the reason `published_forms` is one:
    the emitter walks what this returns instead of testing it, so a unit with no
    such kernel simply contributes nothing rather than being skipped by a branch
    in emission.

    There is at most one, and whether it exists is a question about the element.
    The arrangement rests on presenting a chosen node at local slot 0 by a
    permutation that does not reverse the element, and only the affine simplices
    have one.  A mixed-order unit has none either: its fields do not share a
    shape count, so there is no single local slot to bring to the front.
    """
    from codegen.framework.fem.patch_orientation import supports_patch_orientation
    from codegen.framework.forms.equations import TOTAL_RESIDUAL_UNIT_NAME

    if mixed or not supports_patch_orientation(element_type):
        return ()
    # Only the unit carrying the material's whole residual.  A block unit
    # numbers its coefficients by the block's row rather than from zero, so a
    # kernel contracting `grad_coeff0_*` does not even compile against one --
    # and a unit that is only part of the residual has no merit to contract in
    # the first place.
    if str(unit_name) != TOTAL_RESIDUAL_UNIT_NAME:
        return ()
    # And only where the target has a host parallel region to put it in.  The
    # kernel is host-shaped throughout -- one thread per patch, thread-private
    # staging buffers between its two loops, an atomic to combine the per-thread
    # step totals -- so it is not a kernel a device target can spell
    # differently, it is a kernel a device target does not have.  Emitting it
    # anyway puts OpenMP pragmas in a `.cu`, which the generator refuses.
    if not has_parallel_region:
        return ()
    return ("merit_patch",)


def patch_merit_staged_roles(dependencies):
    """Which field roles a sampled patch kernel carries between its two loops.

    `current` is always there, and it is staged together with the direction
    because the two are what the trial step combines: loop 2 forms
    `current + alpha * direction`.

    `previous` is staged alone, and the asymmetry is the point.  A history is
    fixed for the whole time step, so it does not combine with alpha and has no
    direction partner; loop 2 reads it as it was interpolated.  A rate-dependent
    material such as Kelvin-Voigt needs it, and a kernel that staged only the
    state would reference `u_old` symbols nothing defines -- which is what the
    compiler caught the first time this kernel was emitted for real.
    """
    roles = ["current"]
    if dependencies.previous:
        roles.append("previous")
    return tuple(roles)


#: Which pair of the dependency record's flags decides a role's quantities.
#: `current` speaks for the direction too: loop 2 forms `current + alpha *
#: direction`, so whatever one of the pair is interpolated, the other must be.
_STAGED_QUANTITY_FLAGS = {
    "current": ("current_value", "current_gradient"),
    "previous": ("previous_value", "previous_gradient"),
}


def patch_merit_staged_quantities(dependencies, role="current"):
    """Which of a field's quantities a sampled patch kernel carries between its
    two loops, for one role.

    Loop 1 interpolates them and loop 2 combines each with `alpha`, so the two
    have to agree about which exist -- a quantity staged and not combined is a
    dead buffer, and one combined but not staged is a symbol nothing defines.
    Answering it once here is what keeps them in step, and keeps the emitter
    spelling a sequence rather than testing the form.

    Per role, because the roles do not read the same quantities.  Asking the
    current state's flags on the history's behalf staged a previous gradient
    for every form that reads a current one, and two_phase_flow -- which reads
    the previous *value* and no previous gradient -- interpolated six dead
    buffers per quadrature point to prove it.

    The order is the order the buffers are declared and the combinations are
    emitted, so it is part of the kernel's layout rather than incidental.
    """
    value_flag, gradient_flag = _STAGED_QUANTITY_FLAGS[role]
    quantities = []
    if getattr(dependencies, value_flag):
        quantities.append("value")
    if getattr(dependencies, gradient_flag):
        quantities.append("gradient")
    return tuple(quantities)


def residual_step_dependent_phases():
    """Which local phases a sampled merit repeats for every trial step.

    The state enters the arithmetic at the transform -- that is where
    `grad(x + alpha h)` is formed -- so the transform and everything after it
    depends on the step length, while the trial-function accumulation before it
    does not.  Naming the boundary here is what lets the emitter open its step
    loop without deciding what belongs inside it.

    This is also the statement of what the sampling buys and what it cannot.
    `EVALUATE_TRIAL` is the sum over shape functions, hoisted once whatever the
    line search asks for; `EVALUATE_MATERIAL` is the constitutive law and is
    inside, because it is not affine in the state and repeats per step by
    construction.
    """
    return (
        LocalPhase.TRANSFORM_REFERENCE,
        LocalPhase.EVALUATE_MATERIAL,
        LocalPhase.CONTRACT_TEST,
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
    mooney_rivlin_kelvin_voigt -- reaches this emitter through paths
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


def published_jacobian_blocks(system, action_dependencies):
    """The Jacobian blocks this system actually publishes a kernel for.

    `CoupledResidualSystem.jacobian_blocks` enumerates the full cross product of
    the fields, because that is the shape a Jacobian has.  Whether any of them
    is *emitted* is a different question, and `plans.dependencies.publishes_kernel`
    already answers it -- a form that contracts nothing onto the test functions
    has no kernel, "and the local block, the element entry point, the mesh
    kernels, their diagnostics record and their C ABI entries all follow from
    that".

    The diagnostics half of that sentence had no implementation.  A residual
    that is linear in the test function reads no trial field at all -- a body
    force `-rho * g . v` is the example -- so its Jacobian is identically zero,
    no block kernel is emitted, and the lowering drops the blocks from the form
    metadata the diagnostics plan is built from.  The emitter went on asking for
    a record per block anyway, and generation refused with "diagnostics plan is
    missing entries" before writing a file.

    One helper rather than the same test at each site, because the two sites
    have to agree: the list the plan is asked to carry and the list the source
    advertises are the same list, or validating one against the other says
    nothing.
    """
    if not publishes_kernel(action_dependencies):
        return ()
    return tuple(system.jacobian_blocks())


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
