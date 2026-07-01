from dataclasses import dataclass


@dataclass(frozen=True)
class EnergySoAKernelEmissionPlan:
    unit: object
    context: object
    forms: tuple
    prefix: str
    local_prefix: str
    local_kernel: object
    mesh_kernel: object
    emission_plan: object
    reference_data_plan: object = None
    diagnostics_plan: object = None
    local_signatures: tuple = ()
    mesh_signature: object = None


def energy_soa_kernel_emission_plan(unit, context):
    from .diagnostics_plan import kernel_diagnostics_plan_from_plan
    from .emission_plan import emission_plan_from_unit_context
    from .generation_plan import mesh_kernel_plan_from_context
    from .kernel_signature import (
        local_kernel_signatures_from_plan,
        mesh_kernel_signature_from_plan,
    )
    from .reference_data_plan import reference_data_plan_from_emission_plan

    prefix = unit.name
    local_kernel = unit.local_kernel_plan(context, prefix)
    mesh_kernel = mesh_kernel_plan_from_context(unit, context, prefix)
    element_plan = emission_plan_from_unit_context(unit, context)
    _validate_element_emission_plan(unit.name, element_plan)
    reference_data_plan = reference_data_plan_from_emission_plan(
        unit,
        context,
        element_plan,
        mesh_kernel.name,
    )
    local_signatures = local_kernel_signatures_from_plan(
        unit,
        element_plan,
        local_kernel.name,
        "energy_soa",
    )
    mesh_signature = mesh_kernel_signature_from_plan(
        unit,
        element_plan,
        mesh_kernel.name,
        "energy_soa",
    )
    diagnostics_plan = kernel_diagnostics_plan_from_plan(
        unit,
        element_plan,
        mesh_kernel.name,
        "energy_soa",
        reference_data_plan,
        mesh_signature,
        local_signatures,
    )
    return EnergySoAKernelEmissionPlan(
        unit=unit,
        context=context,
        forms=_energy_kernel_forms(unit),
        prefix=mesh_kernel.name,
        local_prefix=local_kernel.name,
        local_kernel=local_kernel,
        mesh_kernel=mesh_kernel,
        emission_plan=element_plan,
        reference_data_plan=reference_data_plan,
        diagnostics_plan=diagnostics_plan,
        local_signatures=local_signatures,
        mesh_signature=mesh_signature,
    )


def _energy_kernel_forms(unit):
    kernel_forms = tuple(
        expression_plan.source
        for expression_plan in unit.expression_plans
        if expression_plan.source is not None
    )
    if not kernel_forms:
        raise ValueError("energy kernel plan '%s' has no expression-plan kernel forms" % unit.name)
    return kernel_forms


def _validate_element_emission_plan(kernel_name, element_plan):
    _validate_geometry_specialization(
        kernel_name,
        element_plan.affine_geometry,
        element_plan.affine_specialization,
    )
    _validate_geometry_specialization(
        kernel_name,
        element_plan.isoparametric_geometry,
        element_plan.isoparametric_specialization,
    )


def _validate_geometry_specialization(kernel_name, geometry, specialization):
    rule = specialization.quadrature_rule
    if geometry.node.n_shape != rule.n_shape or geometry.node.n_qp != rule.n_qp:
        raise ValueError(
            "kernel plan '%s' geometry mode '%s' has (%d shapes, %d qp), "
            "but specialization has (%d shapes, %d qp)"
            % (
                kernel_name,
                geometry.mode.value,
                geometry.node.n_shape,
                geometry.node.n_qp,
                rule.n_shape,
                rule.n_qp,
            )
        )
