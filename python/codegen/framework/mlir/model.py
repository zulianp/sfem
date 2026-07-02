from dataclasses import dataclass

import sympy as sp


def pad_to_vector_width(count, vector_width):
    if vector_width <= 0:
        raise ValueError("vector_width must be positive")
    return int(vector_width) * int((int(count) + int(vector_width) - 1) // int(vector_width))


@dataclass(frozen=True)
class MLIRLoweringSpec:
    scalar_type: str = "f32"
    index_type: str = "index"
    function_suffix: str = "mlir_apply_threaded"

    def __post_init__(self):
        if self.scalar_type != "f32":
            raise ValueError("initial MLIR EBE lowering supports f32 only")
        if self.index_type != "index":
            raise ValueError("initial MLIR EBE lowering supports MLIR index topology only")
        if not self.function_suffix:
            raise ValueError("function_suffix must be non-empty")


@dataclass(frozen=True)
class MLIRKernelModel:
    material_name: str
    element_type: str
    element_label: str
    kernel_name: str
    kernel_kind: str
    mesh_kernel_name: str
    local_apply_name: str
    dim: int
    vector_size: int
    n_shape: int
    n_qp: int
    quadrature_order: int
    affine_n_qp: int
    affine_quadrature_order: int
    mesh_phases: tuple
    expression_names: tuple
    parameters: tuple
    mesh_arguments: tuple
    local_apply_arguments: tuple
    basis_family: str
    basis_layout: str
    reference_shape_size: int
    reference_gradient_size: int
    quadrature_weights: tuple
    reference_gradients: tuple
    apply_material_expressions: tuple
    isoparametric_geometry: dict
    affine_geometry: dict

    def __post_init__(self):
        object.__setattr__(self, "material_name", str(self.material_name))
        object.__setattr__(self, "element_type", str(self.element_type).upper())
        object.__setattr__(self, "element_label", str(self.element_label).lower())
        object.__setattr__(self, "kernel_name", str(self.kernel_name))
        object.__setattr__(self, "kernel_kind", str(self.kernel_kind))
        object.__setattr__(self, "mesh_kernel_name", str(self.mesh_kernel_name))
        object.__setattr__(self, "local_apply_name", str(self.local_apply_name))
        object.__setattr__(self, "dim", int(self.dim))
        object.__setattr__(self, "vector_size", int(self.vector_size))
        object.__setattr__(self, "n_shape", int(self.n_shape))
        object.__setattr__(self, "n_qp", int(self.n_qp))
        object.__setattr__(self, "quadrature_order", int(self.quadrature_order))
        object.__setattr__(self, "affine_n_qp", int(self.affine_n_qp))
        object.__setattr__(self, "affine_quadrature_order", int(self.affine_quadrature_order))
        object.__setattr__(self, "mesh_phases", tuple(self.mesh_phases))
        object.__setattr__(self, "expression_names", tuple(self.expression_names))
        object.__setattr__(self, "parameters", tuple(str(p) for p in self.parameters))
        object.__setattr__(self, "mesh_arguments", tuple(self.mesh_arguments))
        object.__setattr__(self, "local_apply_arguments", tuple(self.local_apply_arguments))
        object.__setattr__(self, "basis_family", str(self.basis_family))
        object.__setattr__(self, "basis_layout", str(self.basis_layout))
        object.__setattr__(self, "reference_shape_size", int(self.reference_shape_size))
        object.__setattr__(self, "reference_gradient_size", int(self.reference_gradient_size))
        object.__setattr__(self, "quadrature_weights", tuple(float(v) for v in self.quadrature_weights))
        object.__setattr__(self, "reference_gradients", tuple(float(v) for v in self.reference_gradients))
        object.__setattr__(self, "apply_material_expressions", tuple(self.apply_material_expressions))
        if self.dim <= 0 or self.vector_size <= 0 or self.n_shape <= 0 or self.n_qp <= 0:
            raise ValueError("model dimension, vector size, shape count, and quadrature count must be positive")
        if "apply" not in self.expression_names:
            raise ValueError("MLIR EBE lowering requires an apply expression plan")
        if len(self.reference_gradients) != self.n_qp * self.n_shape * self.dim:
            raise ValueError("reference gradient size does not match quadrature/shape/dim model")
        if len(self.quadrature_weights) != self.n_qp:
            raise ValueError("quadrature weight count does not match quadrature model")
        if len(self.apply_material_expressions) != self.dim * self.dim:
            raise ValueError("apply material expression count does not match model dimension")

    @property
    def n_field_components(self):
        return self.dim

    @property
    def scratch_components(self):
        return self.n_shape * self.n_field_components

    @property
    def padded_scratch_components(self):
        return pad_to_vector_width(self.scratch_components, self.vector_size)

    @property
    def function_name(self):
        return f"{self.mesh_kernel_name}_mlir_apply_threaded"

    def to_dict(self):
        return {
            "material_name": self.material_name,
            "element_type": self.element_type,
            "element_label": self.element_label,
            "kernel_name": self.kernel_name,
            "kernel_kind": self.kernel_kind,
            "mesh_kernel_name": self.mesh_kernel_name,
            "local_apply_name": self.local_apply_name,
            "dim": self.dim,
            "vector_size": self.vector_size,
            "n_shape": self.n_shape,
            "n_qp": self.n_qp,
            "quadrature_order": self.quadrature_order,
            "affine_n_qp": self.affine_n_qp,
            "affine_quadrature_order": self.affine_quadrature_order,
            "mesh_phases": list(self.mesh_phases),
            "expression_names": list(self.expression_names),
            "parameters": list(self.parameters),
            "basis_family": self.basis_family,
            "basis_layout": self.basis_layout,
            "reference_shape_size": self.reference_shape_size,
            "reference_gradient_size": self.reference_gradient_size,
            "quadrature_weights": list(self.quadrature_weights),
            "reference_gradients": list(self.reference_gradients),
            "apply_material_expressions": [str(expr) for expr in self.apply_material_expressions],
            "isoparametric_geometry": dict(self.isoparametric_geometry),
            "affine_geometry": dict(self.affine_geometry),
        }

def linear_elasticity_mlir_model(element="TET4", vector_size=8, quadrature_order=None):
    from codegen.framework.materials.linear_elasticity import material

    return mlir_model_from_material(
        material,
        element=element,
        vector_size=vector_size,
        quadrature_order=quadrature_order,
    )


def mlir_model_from_material(material, *, element, vector_size=8, quadrature_order=None):
    from sfem import gen
    from codegen.framework.energy_codegen import (
        _weak_form_deformation_gradient_substitutions,
        _weak_form_material_expression,
    )
    from codegen.framework.energy_plan import energy_soa_kernel_emission_plan

    if getattr(material, "name", "") != "linear_elasticity":
        raise ValueError("initial MLIR EBE lowering is wired to the linear_elasticity model")

    user_input = gen.UserInputStage.create(
        material,
        (element,),
        int(vector_size),
        quadrature_order,
    )
    form_evaluation = gen._evaluate_forms(user_input)
    codegen_plan = gen.SpecializedFormManipulationStage(
        user_input,
        form_evaluation,
    ).run()
    context = user_input.element_contexts[0]
    units = tuple(codegen_plan.emission_kernels_for_context(context))
    if len(units) != 1:
        raise ValueError("expected exactly one linear elasticity energy kernel for context")
    unit = units[0]
    emission = energy_soa_kernel_emission_plan(unit, context)
    local_apply = None
    for signature in emission.local_signatures:
        if signature.name.endswith("_apply"):
            local_apply = signature
            break
    if local_apply is None:
        raise ValueError("linear elasticity energy kernel has no apply local signature")
    apply_plan = None
    for expression_plan in unit.expression_plans:
        if expression_plan.name == "apply":
            apply_plan = expression_plan
            break
    if apply_plan is None:
        raise ValueError("linear elasticity energy kernel has no apply expression plan")

    rule = context.specialization.quadrature_rule
    affine_rule = context.affine_specialization.quadrature_rule
    basis = emission.emission_plan.basis_plans[0]
    iso_geometry = _geometry_dict(emission.emission_plan.isoparametric_geometry.node)
    affine_geometry = _geometry_dict(emission.emission_plan.affine_geometry.node)
    trial_gradient = tuple(
        sp.symbols("trial_grad%d" % i)
        for i in range(unit.dim * unit.dim)
    )
    deformation_substitutions = _weak_form_deformation_gradient_substitutions(
        apply_plan.weak_form,
        "grad_u",
        scalar_temporaries=True,
    )
    apply_material = tuple(
        _weak_form_material_expression(
            apply_plan.weak_form,
            "apply",
            deformation_substitutions,
            trial_gradient,
        )
    )
    return MLIRKernelModel(
        material_name=material.name,
        element_type=context.element_type,
        element_label=context.label,
        kernel_name=unit.name,
        kernel_kind=unit.kind.value,
        mesh_kernel_name=emission.mesh_kernel.name,
        local_apply_name=local_apply.name,
        dim=unit.dim,
        vector_size=context.specialization.vector_size,
        n_shape=rule.n_shape,
        n_qp=rule.n_qp,
        quadrature_order=rule.order,
        affine_n_qp=affine_rule.n_qp,
        affine_quadrature_order=affine_rule.order,
        mesh_phases=tuple(phase.phase.value for phase in unit.mesh_phase_plans),
        expression_names=tuple(plan.name for plan in unit.expression_plans),
        parameters=tuple(
            argument.name
            for argument in emission.mesh_signature.arguments
            if argument.role == "parameter"
        ),
        mesh_arguments=tuple(argument.to_dict() for argument in emission.mesh_signature.arguments),
        local_apply_arguments=tuple(argument.to_dict() for argument in local_apply.arguments),
        basis_family=basis.family.value,
        basis_layout=basis.data_layout.value,
        reference_shape_size=basis.reference_shape_size,
        reference_gradient_size=basis.reference_gradient_size,
        quadrature_weights=tuple(rule.weights),
        reference_gradients=tuple(rule.reference_gradients),
        apply_material_expressions=apply_material,
        isoparametric_geometry=iso_geometry,
        affine_geometry=affine_geometry,
    )

def _geometry_dict(node):
    return {
        "mode": node.mode.value,
        "element_type": node.element_type,
        "dim": node.dim,
        "n_shape": node.n_shape,
        "n_qp": node.n_qp,
        "input_layout": node.input_layout.value,
        "evaluation": node.evaluation.value,
        "jacobian_scope": node.jacobian_scope,
        "geometry_points_per_element": node.geometry_points_per_element,
        "geometry_stream_count": node.geometry_stream_count,
    }
