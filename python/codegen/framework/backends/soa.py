"""The SoA backend boundary: one traversal, whichever target is bound.

Everything between a planned code-generation unit and its emitted files --
which plans to build, in what order, and which emitter consumes each -- is the
same for every target, and this module owns it.  What differs is four things,
and they are the four hooks a subclass fills in: which unit targets it accepts,
what it requires of its own target, which energy emitter it builds, and what it
checks about the files that come back.

`backends/openmp.py` and `backends/cuda.py` are those subclasses.  Before this
split the traversal lived in the OpenMP backend and the CUDA one had no way to
reach it, so it accepted `energy_soa` units and raised on everything else --
not because the residual family could not be lowered, but because the code that
lowers it was behind a class the CUDA backend could not name.
"""

from dataclasses import dataclass

import sympy as sp

from codegen.framework.forms.forms import FormOrder
from codegen.framework.plans.generation import KernelTarget, MeshPhase, LocalPhase, mesh_kernel_plan_from_context
from codegen.framework.plans.emission import emission_plan_for_element, emission_plan_from_unit_context
from codegen.framework.plans.kernel_signature import (
    local_kernel_signatures_from_plan,
    local_kernel_suffix_from_plan,
    mesh_kernel_signature_from_plan,
)
from codegen.framework.plans.diagnostics import kernel_diagnostics_plan_from_plan
from codegen.framework.plans.energy import energy_soa_kernel_emission_plan
from codegen.framework.plans.reference_data import reference_data_plan_from_emission_plan
from codegen.framework.emitters.boundary_codegen import generate_boundary_residual_sfem_files
from codegen.framework.emitters.artifacts import GeneratedKernelFile
from codegen.framework.emitters.inexact_apply_codegen import inexact_apply_files
from codegen.framework.plans.residual_model import (
    diagonal_block_emission_model,
    residual_emission_model,
)
from codegen.framework.forms.residual import WeakResidualCoefficients
from codegen.framework.emitters.residual_codegen import (
    generate_coupled_residual_sfem_files,
    generate_mixed_residual_sfem_files,
)
from codegen.framework.emitters.matrix_formats import emit_matrix_format_metadata_files
from codegen.framework.targets import use_target


@dataclass(frozen=True)
class SoAEmission:
    files: tuple

    def __iter__(self):
        return iter(self.files)


@dataclass(frozen=True)
class SoATraversal:
    kind: str
    unit: object
    context: object
    prefix: str
    local_prefix: str = ""
    local_name: str = ""
    operator_prefix: str = ""
    operator_name: str = ""
    emission_plan: object = None
    system: object = None
    compatible_element: object = None
    residual_coeffs: tuple = ()
    action_coeffs: tuple = ()
    field_element_types: object = None
    boundary_expression_plan: object = None
    local_signatures: tuple = ()
    mesh_signature: object = None
    reference_data_plan: object = None
    diagnostics_plan: object = None
    energy_plan: object = None
    matrix_format_plan: object = None


@dataclass(frozen=True)
class SoABackend:
    """The SoA backend boundary for planned code-generation units."""

    supports_op_wrapper: bool = True
    target: object = None
    emitter: object = None

    #: The four hooks.  A subclass answers all of them; the base answers none,
    #: because every default here would be one target's answer wearing the name
    #: of all of them.
    def _shared_emitter(self):
        """The energy emitter for this target."""
        raise NotImplementedError

    def mesh_source_extension(self):
        """What this target's mesh translation unit is called."""
        raise NotImplementedError

    def local_header_extension(self):
        """What this target's element-local header is called."""
        raise NotImplementedError

    def _require_backend_target(self):
        """What this backend requires of the target it was constructed with."""
        raise NotImplementedError

    def _require_unit_target(self, unit):
        """Which unit targets this backend will emit."""
        raise NotImplementedError

    def _validate_emitted(self, files, traversal):
        """What must be true of the files that came back."""
        raise NotImplementedError

    def shared_primitive_files(self):
        """The target's material-independent headers, bound to this target.

        Asked of the backend for the same reason `emit_inexact` is: the target
        is what decides how they are spelled, and the caller should not have to
        build an emitter to find out.
        """
        with use_target(self.target):
            return tuple(self._shared_emitter().shared_primitive_files())

    def emit_inexact(self, material, unit, context):
        """The inexact-apply family, through the same boundary as everything else.

        It had no backend.  `_emit_codegen_unit` picks a backend for the target
        and lets it bind; this family called its emitter directly from the
        driver, so it ran under whatever target happened to be ambient and
        emitted the CPU shape whatever was asked for.  Routing it here is what
        makes the workflow one workflow: the target binds, the emitter asks it,
        and the backend checks what came out.
        """
        self._require_backend_target()
        with use_target(self.target):
            return tuple(
                GeneratedKernelFile(path, source)
                for path, source in inexact_apply_files(material, unit, context)
            )

    def emit(self, unit, context):
        self._require_backend_target()
        self._require_unit_target(unit)
        unit.validate_for_context(context)
        # The backend's target reaches the emitters here.  Before this binding
        # existed it reached only the check above and the energy emitter: the
        # residual emitters constructed their own OpenMPTarget, so this
        # backend's target had no effect on them.  No output changes today,
        # because AVX512Target and the ARM targets spell every qualifier and
        # pragma exactly as OpenMPTarget does -- the wiring was dead rather
        # than wrong.  It stops being dead the moment one of them differs.
        with use_target(self.target):
            traversal = self._traversal(unit, context)
            files = tuple(self._emit_traversal_files(traversal))
        self._validate_emitted(files, traversal)
        return SoAEmission(files)

    def local_signatures(self, unit, context):
        unit.validate_for_context(context)
        return self._traversal(unit, context).local_signatures

    def mesh_signature(self, unit, context):
        unit.validate_for_context(context)
        return self._traversal(unit, context).mesh_signature

    def reference_data_plan(self, unit, context):
        unit.validate_for_context(context)
        return self._traversal(unit, context).reference_data_plan

    def diagnostics_plan(self, unit, context):
        unit.validate_for_context(context)
        return self._traversal(unit, context).diagnostics_plan

    @staticmethod
    def _diagnostics_enabled(unit):
        payload = getattr(unit, "payload", None) or {}
        return bool(payload.get("diagnostics", True))

    def _traversal(self, unit, context):
        kind = _kind_value(unit.kind)
        if kind == "energy_soa":
            return self._energy_traversal(unit, context)
        if kind == "residual_soa":
            return self._residual_traversal(unit, context)
        if kind == "boundary_residual_soa":
            return self._boundary_residual_traversal(unit, context)
        raise ValueError("unsupported SoA kernel kind '%s'" % kind)

    def _energy_traversal(self, unit, context):
        self._validate_energy_plan(unit)
        energy_plan = energy_soa_kernel_emission_plan(unit, context)
        return SoATraversal(
            "energy_soa",
            unit,
            context,
            unit.name,
            local_prefix=energy_plan.local_prefix,
            local_name=energy_plan.local_kernel.header,
            operator_prefix=energy_plan.mesh_kernel.name,
            operator_name=energy_plan.mesh_kernel.source,
            emission_plan=energy_plan.emission_plan,
            local_signatures=energy_plan.local_signatures,
            mesh_signature=energy_plan.mesh_signature,
            reference_data_plan=energy_plan.reference_data_plan,
            diagnostics_plan=energy_plan.diagnostics_plan,
            energy_plan=energy_plan,
            matrix_format_plan=unit.matrix_format_plan,
        )

    def _residual_traversal(self, unit, context):
        self._validate_residual_plan(unit)
        collection = unit.form_collection
        system = residual_emission_model(collection)
        residual_coeffs = _coefficients_for_unit(unit, collection, FormOrder.ONE)
        action_coeffs = _coefficients_for_unit(unit, collection, FormOrder.TWO)
        residual_plan = _expression_plan_for_order(unit, FormOrder.ONE)
        action_plan = _expression_plan_for_order(unit, FormOrder.TWO)
        _validate_coefficient_dependencies(
            unit.name,
            residual_plan.dependencies,
            residual_coeffs,
        )
        _validate_coefficient_dependencies(
            unit.name,
            action_plan.dependencies,
            action_coeffs,
        )
        prefix = _generated_prefix(unit)
        if _is_diagonal_two_form_block(unit) and context.is_mixed_order:
            model = _diagonal_block_model(unit, collection, context)
            mesh_kernel = mesh_kernel_plan_from_context(
                unit,
                context,
                prefix,
                element_label=model.element_type,
                source_extension=self.mesh_source_extension(),
            )
            kind = "mixed_residual_soa"
            local_kernel = unit.local_kernel_plan(
                context,
                prefix,
                local_kernel_suffix_from_plan(unit, context, kind),
                self.local_header_extension(),
            )
            field_element_types = {
                field.field_name: model.element_type for field in model.system.fields
            }
            reference_data_plan = reference_data_plan_from_emission_plan(
                unit,
                context,
                model.emission_plan,
                prefix,
                field_element_types,
            )
            local_signatures = local_kernel_signatures_from_plan(
                unit,
                model.emission_plan,
                local_kernel.name,
                kind,
            )
            mesh_signature = mesh_kernel_signature_from_plan(
                unit,
                model.emission_plan,
                mesh_kernel.name,
                kind,
            )
            diagnostics_plan = (
                kernel_diagnostics_plan_from_plan(
                    unit,
                    model.emission_plan,
                    mesh_kernel.name,
                    kind,
                    reference_data_plan,
                    mesh_signature,
                    local_signatures,
                )
                if self._diagnostics_enabled(unit)
                else None
            )
            return SoATraversal(
                kind,
                unit,
                context,
                prefix,
                local_prefix=local_kernel.name,
                local_name=local_kernel.header,
                operator_prefix=mesh_kernel.name,
                operator_name=mesh_kernel.source,
                emission_plan=model.emission_plan,
                system=model.system,
                compatible_element=_CompatibleElementLabel(
                    model.element_type,
                    model.element_type,
                ),
                residual_coeffs=model.residual_coeffs,
                action_coeffs=model.action_coeffs,
                field_element_types=field_element_types,
                local_signatures=local_signatures,
                mesh_signature=mesh_signature,
                reference_data_plan=reference_data_plan,
                diagnostics_plan=diagnostics_plan,
                matrix_format_plan=unit.matrix_format_plan,
            )
        mesh_kernel = mesh_kernel_plan_from_context(
            unit, context, prefix, source_extension=self.mesh_source_extension()
        )
        if context.is_mixed_order:
            kind = "mixed_residual_soa"
            emission_plan = _validated_emission_plan(unit, context)
            local_kernel = unit.local_kernel_plan(
                context,
                prefix,
                local_kernel_suffix_from_plan(unit, context, kind),
                self.local_header_extension(),
            )
            field_element_types = _field_element_types_for_context(
                collection.fields,
                context,
            )
            reference_data_plan = reference_data_plan_from_emission_plan(
                unit,
                context,
                emission_plan,
                prefix,
                field_element_types,
            )
            local_signatures = local_kernel_signatures_from_plan(
                unit,
                emission_plan,
                local_kernel.name,
                kind,
            )
            mesh_signature = mesh_kernel_signature_from_plan(
                unit,
                emission_plan,
                mesh_kernel.name,
                kind,
            )
            diagnostics_plan = (
                kernel_diagnostics_plan_from_plan(
                    unit,
                    emission_plan,
                    mesh_kernel.name,
                    kind,
                    reference_data_plan,
                    mesh_signature,
                    local_signatures,
                )
                if self._diagnostics_enabled(unit)
                else None
            )
            return SoATraversal(
                kind,
                unit,
                context,
                prefix,
                local_prefix=local_kernel.name,
                local_name=local_kernel.header,
                operator_prefix=mesh_kernel.name,
                operator_name=mesh_kernel.source,
                emission_plan=emission_plan,
                system=system,
                compatible_element=context.compatible_element,
                residual_coeffs=residual_coeffs,
                action_coeffs=action_coeffs,
                field_element_types=field_element_types,
                local_signatures=local_signatures,
                mesh_signature=mesh_signature,
                reference_data_plan=reference_data_plan,
                diagnostics_plan=diagnostics_plan,
                matrix_format_plan=unit.matrix_format_plan,
            )
        kind = "residual_soa"
        emission_plan = _validated_emission_plan(unit, context)
        local_kernel = unit.local_kernel_plan(
            context,
            prefix,
            local_kernel_suffix_from_plan(unit, context, kind),
            self.local_header_extension(),
        )
        reference_data_plan = reference_data_plan_from_emission_plan(
            unit,
            context,
            emission_plan,
            mesh_kernel.name,
        )
        local_signatures = local_kernel_signatures_from_plan(
            unit,
            emission_plan,
            local_kernel.name,
            kind,
        )
        mesh_signature = mesh_kernel_signature_from_plan(
            unit,
            emission_plan,
            mesh_kernel.name,
            kind,
        )
        diagnostics_plan = (
            kernel_diagnostics_plan_from_plan(
                unit,
                emission_plan,
                mesh_kernel.name,
                kind,
                reference_data_plan,
                mesh_signature,
                local_signatures,
            )
            if self._diagnostics_enabled(unit)
            else None
        )
        return SoATraversal(
            kind,
            unit,
            context,
            prefix,
            local_prefix=local_kernel.name,
            local_name=local_kernel.header,
            operator_prefix=mesh_kernel.name,
            operator_name=mesh_kernel.source,
            emission_plan=emission_plan,
            system=system,
            residual_coeffs=residual_coeffs,
            action_coeffs=action_coeffs,
            local_signatures=local_signatures,
            mesh_signature=mesh_signature,
            reference_data_plan=reference_data_plan,
            diagnostics_plan=diagnostics_plan,
            matrix_format_plan=unit.matrix_format_plan,
        )

    def _boundary_residual_traversal(self, unit, context):
        self._validate_boundary_residual_plan(unit)
        prefix = _generated_prefix(unit)
        mesh_kernel = mesh_kernel_plan_from_context(
            unit, context, prefix, source_extension=self.mesh_source_extension()
        )
        emission_plan = _validated_emission_plan(unit, context)
        kind = "boundary_residual_soa"
        reference_data_plan = reference_data_plan_from_emission_plan(
            unit,
            context,
            emission_plan,
            mesh_kernel.name,
        )
        local_signatures = local_kernel_signatures_from_plan(
            unit,
            emission_plan,
            "",
            kind,
        )
        mesh_signature = mesh_kernel_signature_from_plan(
            unit,
            emission_plan,
            mesh_kernel.name,
            kind,
        )
        diagnostics_plan = (
            kernel_diagnostics_plan_from_plan(
                unit,
                emission_plan,
                mesh_kernel.name,
                kind,
                reference_data_plan,
                mesh_signature,
                local_signatures,
            )
            if self._diagnostics_enabled(unit)
            else None
        )
        return SoATraversal(
            kind,
            unit,
            context,
            prefix,
            operator_prefix=mesh_kernel.name,
            operator_name=mesh_kernel.source,
            emission_plan=emission_plan,
            boundary_expression_plan=_expression_plan_for_order(unit, FormOrder.ONE),
            local_signatures=local_signatures,
            mesh_signature=mesh_signature,
            reference_data_plan=reference_data_plan,
            diagnostics_plan=diagnostics_plan,
            matrix_format_plan=unit.matrix_format_plan,
        )

    def _emit_traversal_files(self, traversal):
        files = ()
        if traversal.kind == "energy_soa":
            emitter = self.emitter or self._shared_emitter()
            files = tuple(emitter.emit_plan(traversal.energy_plan))
        elif traversal.kind == "residual_soa":
            files = tuple(generate_coupled_residual_sfem_files(
                traversal.system,
                prefix=traversal.prefix,
                emission_plan=traversal.emission_plan,
                residual_coeffs=traversal.residual_coeffs,
                action_coeffs=traversal.action_coeffs,
                local_prefix=traversal.local_prefix,
                local_name=traversal.local_name,
                operator_prefix=traversal.operator_prefix,
                operator_name=traversal.operator_name,
                reference_data_plan=traversal.reference_data_plan,
                diagnostics_plan=traversal.diagnostics_plan,
                matrix_format_plan=traversal.matrix_format_plan,
            ))
        elif traversal.kind == "mixed_residual_soa":
            files = tuple(generate_mixed_residual_sfem_files(
                traversal.system,
                prefix=traversal.prefix,
                compatible_element=traversal.compatible_element,
                emission_plan=traversal.emission_plan,
                residual_coeffs=traversal.residual_coeffs,
                action_coeffs=traversal.action_coeffs,
                field_element_types=traversal.field_element_types,
                local_prefix=traversal.local_prefix,
                local_name=traversal.local_name,
                operator_prefix=traversal.operator_prefix,
                operator_name=traversal.operator_name,
                reference_data_plan=traversal.reference_data_plan,
                diagnostics_plan=traversal.diagnostics_plan,
            ))
        elif traversal.kind == "boundary_residual_soa":
            files = tuple(generate_boundary_residual_sfem_files(
                traversal.unit.form_collection,
                prefix=traversal.operator_prefix,
                emission_plan=traversal.emission_plan,
                expression_plan=traversal.boundary_expression_plan,
                reference_data_plan=traversal.reference_data_plan,
                diagnostics_plan=traversal.diagnostics_plan,
            ))
        else:
            raise ValueError("unsupported SoA traversal kind '%s'" % traversal.kind)
        return files + tuple(
            emit_matrix_format_metadata_files(
                traversal.operator_prefix,
                traversal.matrix_format_plan,
            )
        )

    @staticmethod
    def _validate_energy_plan(unit):
        _require_form_metadata(unit, tuple(form.order for form in unit.form_collection.forms))
        _require_geometry_modes(unit, ("affine", "isoparametric"))
        for form in unit.form_collection.forms:
            _validate_expression_dependencies(
                unit.name,
                form.order,
                unit.form_collection.form_metadata(form.order).dependencies,
                (form.expression,),
            )
        _require_mesh_phases(
            unit,
            (
                MeshPhase.GEOMETRY,
                MeshPhase.LOCAL_CALL,
                MeshPhase.SCATTER,
            ),
        )

    @staticmethod
    def _validate_residual_plan(unit):
        _require_form_metadata(unit, (FormOrder.ONE, FormOrder.TWO))
        _require_geometry_modes(unit, ("affine", "isoparametric"))
        _require_mesh_phases(
            unit,
            (
                MeshPhase.GATHER,
                MeshPhase.GEOMETRY,
                MeshPhase.LOCAL_CALL,
                MeshPhase.SCATTER,
            ),
        )
        for block in unit.blocks:
            _require_local_phases(
                block,
                (
                    LocalPhase.EVALUATE_TRIAL,
                    LocalPhase.TRANSFORM_REFERENCE,
                    LocalPhase.EVALUATE_MATERIAL,
                    LocalPhase.CONTRACT_TEST,
                ),
            )

    @staticmethod
    def _validate_boundary_residual_plan(unit):
        _require_form_metadata(unit, (FormOrder.ONE,))
        _require_geometry_modes(unit, ("affine", "isoparametric"))
        _require_mesh_phases(
            unit,
            (
                MeshPhase.GATHER,
                MeshPhase.GEOMETRY,
                MeshPhase.LOCAL_CALL,
                MeshPhase.SCATTER,
            ),
        )


def _kind_value(kind):
    return getattr(kind, "value", str(kind))


def _generated_prefix(unit):
    return unit.name


def _expression_plan_for_order(unit, order):
    order = FormOrder(order)
    for expression_plan in unit.expression_plans:
        if expression_plan.form_order is order:
            return expression_plan
    raise ValueError(
        "kernel plan '%s' has no expression plan for %s"
        % (unit.name, order.name)
    )


def _require_form_metadata(unit, orders):
    for order in orders:
        try:
            unit.form_collection.form_metadata(order)
        except ValueError as exc:
            raise ValueError(
                "kernel plan '%s' is missing FormMetadata for %s"
                % (unit.name, FormOrder(order).name)
            ) from exc


def _require_geometry_modes(unit, modes):
    available = {geometry.mode.value for geometry in _geometry_phase(unit).geometries}
    missing = tuple(mode for mode in modes if mode not in available)
    if missing:
        raise ValueError(
            "kernel plan '%s' is missing geometry phase modes: %s"
            % (unit.name, ", ".join(missing))
        )


def _validated_emission_plan(unit, context):
    emission_plan = emission_plan_from_unit_context(unit, context)
    _validate_geometry_specialization(
        unit.name,
        emission_plan.affine_geometry,
        emission_plan.affine_specialization,
    )
    _validate_geometry_specialization(
        unit.name,
        emission_plan.isoparametric_geometry,
        emission_plan.isoparametric_specialization,
    )
    return emission_plan


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


def _geometry_phase(unit):
    for phase in unit.mesh_phase_plans:
        if phase.phase is MeshPhase.GEOMETRY:
            return phase
    raise ValueError("kernel plan '%s' has no geometry phase" % unit.name)


def _validate_coefficient_dependencies(kernel_name, dependencies, coefficients):
    expressions = []
    for coefficient in coefficients:
        expressions.append(coefficient.value)
        expressions.extend(tuple(coefficient.gradient))
    _validate_expression_dependencies(
        kernel_name,
        None,
        dependencies,
        expressions,
    )


def _validate_expression_dependencies(kernel_name, order, dependencies, expressions):
    declared = set(_dependency_symbols(dependencies))
    required = set()
    for expression in expressions:
        required.update(sp.sympify(expression).free_symbols)
    missing = tuple(sorted(required.difference(declared), key=str))
    if missing:
        order_name = "" if order is None else " %s" % FormOrder(order).name
        raise ValueError(
            "kernel plan '%s'%s requests undeclared FormMetadata inputs: %s"
            % (kernel_name, order_name, ", ".join(map(str, missing)))
        )


def _dependency_symbols(dependencies):
    symbols = getattr(dependencies, "symbols", None)
    if symbols is not None:
        return tuple(symbols)
    ret = []
    for attr in (
        "current_symbols",
        "previous_symbols",
        "direction_symbols",
        "geometry_symbols",
        "parameters",
    ):
        ret.extend(tuple(getattr(dependencies, attr, ())))
    return tuple(dict.fromkeys(ret))


def _require_mesh_phases(unit, expected):
    actual = tuple(phase.phase for phase in unit.mesh_phase_plans)
    if actual != tuple(expected):
        raise ValueError(
            "kernel '%s' mesh phase plan %s does not match SoA contract %s"
            % (
                unit.name,
                tuple(phase.value for phase in actual),
                tuple(phase.value for phase in expected),
            )
        )


def _require_local_phases(block, expected):
    actual = tuple(phase.phase for phase in block.local_phase_plans)
    if actual != tuple(expected):
        raise ValueError(
            "block '%s' local phase plan %s does not match SoA contract %s"
            % (
                block.name,
                tuple(phase.value for phase in actual),
                tuple(phase.value for phase in expected),
            )
        )


def _field_element_types_for_context(equation_fields, context):
    ret = {}
    for field, element_type in context.fem_policy.field_element_types_for(equation_fields):
        ret[field.name] = element_type
    return ret


@dataclass(frozen=True)
class _DiagonalBlockModel:
    system: object
    element_type: str
    specialization: object
    affine_specialization: object
    emission_plan: object
    residual_coeffs: tuple
    action_coeffs: tuple


@dataclass(frozen=True)
class _CompatibleElementLabel:
    name: str
    cell_element_type: str


def _is_diagonal_two_form_block(unit):
    return (
        unit.is_block
        and unit.block is not None
        and unit.block.form_order is FormOrder.TWO
        and unit.block.column_field
        and unit.block.row_field == unit.block.column_field
    )


def _diagonal_block_model(unit, collection, context):
    field = _collection_field(collection, unit.block.row_field)
    field_element_types = {
        equation_field.name: element_type
        for equation_field, element_type in context.fem_policy.field_element_types_for(collection.fields)
    }
    element_type = field_element_types[field.name]
    emission_plan = emission_plan_for_element(
        element_type,
        context.specialization.vector_size,
        context.specialization.quadrature_rule.order,
        affine_quadrature_order=context.affine_specialization.quadrature_rule.order,
    )
    specialization = emission_plan.isoparametric_specialization
    affine_specialization = emission_plan.affine_specialization
    system, _ = diagonal_block_emission_model(
        collection,
        field,
        None,
    )
    block = collection.block(FormOrder.TWO, unit.block.row_field, unit.block.column_field)
    return _DiagonalBlockModel(
        system,
        element_type,
        specialization,
        affine_specialization,
        emission_plan,
        _zero_coefficients(system),
        block.coefficients,
    )


def _collection_field(collection, name):
    name = str(name)
    for field in collection.fields:
        if field.name == name:
            return field
    raise ValueError("field '%s' is not in form collection" % name)


def _component_field_names(field):
    components = int(getattr(field, "components", 1))
    if components == 1:
        return (field.name,)
    return tuple("%s%d" % (field.name, component) for component in range(components))


def _coefficients_for_unit(unit, collection, order):
    order = FormOrder(order)
    expression_plan = _expression_plan_for_order(unit, order)
    if not unit.is_block:
        return expression_plan.coefficients
    if unit.block.form_order is not order:
        return _zero_coefficients_for_collection(collection)
    if not expression_plan.coefficients:
        raise ValueError(
            "block kernel '%s' has no coefficient sets for '%s'"
            % (unit.name, unit.block.name)
        )
    return _selected_row_coefficients_for_collection(collection, expression_plan.coefficients)


def _collection_dim(collection):
    """Spatial dimension of a lowered residual collection."""
    for field in collection.residual_fields:
        return field.dim
    raise ValueError(
        "form collection '%s' carries no lowered residual fields" % collection.equation_name
    )


def _zero_coefficients_for_collection(collection):
    dim = _collection_dim(collection)
    return tuple(
        WeakResidualCoefficients(
            field.name,
            sp.S.Zero,
            tuple(sp.S.Zero for _ in range(dim)),
        )
        for field in collection.residual_fields
    )


def _selected_row_coefficients_for_collection(collection, coefficients):
    selected = list(_zero_coefficients_for_collection(collection))
    by_name = {field.name: index for index, field in enumerate(collection.residual_fields)}
    for coefficient in coefficients:
        try:
            selected[by_name[coefficient.row_field]] = coefficient
        except KeyError:
            raise ValueError(
                "row field '%s' is not in residual system" % coefficient.row_field
            )
    return tuple(selected)


def _zero_coefficients(system):
    return tuple(
        WeakResidualCoefficients(
            field.name,
            sp.S.Zero,
            tuple(sp.S.Zero for _ in range(system.dim)),
        )
        for field in system.fields
    )


