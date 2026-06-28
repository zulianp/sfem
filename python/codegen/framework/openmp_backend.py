from dataclasses import dataclass

import sympy as sp

from .forms import FormOrder
from .generation_plan import (
    KernelTarget,
    MeshKernelPlan,
    MeshPhase,
    LocalPhase,
)
from .symbolic import (
    generate_sfem_soa_cpp_files_for_element,
)
from .fem import sfem_soa_element_specialization
from .residual import CoupledResidualSystem
from .residual_codegen import (
    WeakResidualCoefficients,
    generate_coupled_residual_sfem_files,
    generate_mixed_residual_sfem_files,
)


@dataclass(frozen=True)
class OpenMPSoAEmission:
    files: tuple

    def __iter__(self):
        return iter(self.files)


@dataclass(frozen=True)
class OpenMPSoABackend:
    """Single OpenMP/SoA backend boundary for planned code-generation units."""

    def emit(self, unit, context):
        unit.validate_for_context(context)
        kind = _kind_value(unit.kind)
        if kind == "energy_soa":
            return self._emit_energy_plan(unit, context)
        if kind == "residual_soa":
            return self._emit_residual_plan(unit, context)
        raise ValueError("unsupported OpenMP SoA kernel kind '%s'" % kind)

    def _emit_energy_plan(self, unit, context):
        self._validate_energy_plan(unit)
        payload = unit.payload
        prefix = _generated_prefix(unit)
        local_kernel = unit.local_kernel_plan(context, prefix)
        mesh_kernel = MeshKernelPlan(prefix, context.element_type)
        return self.emit_energy(
            payload.kernel_forms,
            prefix=mesh_kernel.name,
            local_prefix=local_kernel.name,
            specialization=context.specialization,
            affine_specialization=context.affine_specialization,
        )

    def _emit_residual_plan(self, unit, context):
        self._validate_residual_plan(unit)
        collection = unit.form_collection
        system = collection.source
        residual_coeffs = _coefficients_for_unit(unit, collection, FormOrder.ONE)
        action_coeffs = _coefficients_for_unit(unit, collection, FormOrder.TWO)
        prefix = _generated_prefix(unit)
        if _is_diagonal_two_form_block(unit) and context.is_mixed_order:
            model = _diagonal_block_model(unit, collection, context)
            local_kernel = unit.local_kernel_plan(context, prefix)
            operator_prefix = "%s_%s" % (prefix, model.element_type.lower())
            return self.emit_mixed_residual(
                model.system,
                prefix=prefix,
                compatible_element=_CompatibleElementLabel(
                    model.element_type,
                    model.element_type,
                ),
                vector_size=model.specialization.vector_size,
                quadrature_order=model.specialization.quadrature_rule.order,
                residual_coeffs=model.residual_coeffs,
                action_coeffs=model.action_coeffs,
                field_element_types={
                    field.field_name: model.element_type for field in model.system.fields
                },
                local_prefix=local_kernel.name,
                local_name=local_kernel.header,
                operator_prefix=operator_prefix,
                operator_name="%s_operator.cpp" % operator_prefix,
            )
        local_kernel = unit.local_kernel_plan(
            context,
            prefix,
            "_mixed" if context.is_mixed_order else "",
        )
        mesh_kernel = unit.mesh_kernel_plan(context, prefix)
        if context.is_mixed_order:
            return self.emit_mixed_residual(
                system,
                prefix=prefix,
                compatible_element=context.compatible_element,
                vector_size=context.specialization.vector_size,
                quadrature_order=context.specialization.quadrature_rule.order,
                residual_coeffs=residual_coeffs,
                action_coeffs=action_coeffs,
                field_element_types=_field_element_types_for_context(
                    collection.fields,
                    context,
                ),
                local_prefix=local_kernel.name,
                local_name=local_kernel.header,
                operator_prefix=mesh_kernel.name,
                operator_name=mesh_kernel.source,
            )
        return self.emit_residual(
            system,
            prefix=prefix,
            element_type=context.element_type,
            vector_size=context.specialization.vector_size,
            quadrature_order=context.specialization.quadrature_rule.order,
            specialization=context.specialization,
            affine_specialization=context.affine_specialization,
            residual_coeffs=residual_coeffs,
            action_coeffs=action_coeffs,
            local_prefix=local_kernel.name,
            local_name=local_kernel.header,
            operator_prefix=mesh_kernel.name,
            operator_name=mesh_kernel.source,
        )

    def emit_energy(
        self,
        kernel_forms,
        *,
        prefix,
        local_prefix,
        specialization,
        affine_specialization=None,
    ):
        files = tuple(
            generate_sfem_soa_cpp_files_for_element(
                kernel_forms,
                prefix=prefix,
                local_prefix=local_prefix,
                specialization=specialization,
                affine_specialization=affine_specialization,
            )
        )
        self._validate_common_source_contract(files, local_prefix)
        return OpenMPSoAEmission(files)

    def emit_residual(
        self,
        system,
        *,
        prefix,
        element_type,
        vector_size,
        quadrature_order,
        specialization,
        affine_specialization,
        residual_coeffs,
        action_coeffs,
        local_prefix,
        local_name,
        operator_prefix,
        operator_name,
    ):
        files = tuple(
            generate_coupled_residual_sfem_files(
                system,
                prefix=prefix,
                element_type=element_type,
                vector_size=vector_size,
                quadrature_order=quadrature_order,
                specialization=specialization,
                affine_specialization=affine_specialization,
                residual_coeffs=residual_coeffs,
                action_coeffs=action_coeffs,
                local_prefix=local_prefix,
                local_name=local_name,
                operator_prefix=operator_prefix,
                operator_name=operator_name,
            )
        )
        self._validate_common_source_contract(files, local_prefix)
        return OpenMPSoAEmission(files)

    def emit_mixed_residual(
        self,
        system,
        *,
        prefix,
        compatible_element,
        vector_size,
        quadrature_order,
        residual_coeffs,
        action_coeffs,
        field_element_types,
        local_prefix,
        local_name,
        operator_prefix,
        operator_name,
    ):
        files = tuple(
            generate_mixed_residual_sfem_files(
                system,
                prefix=prefix,
                compatible_element=compatible_element,
                vector_size=vector_size,
                quadrature_order=quadrature_order,
                residual_coeffs=residual_coeffs,
                action_coeffs=action_coeffs,
                field_element_types=field_element_types,
                local_prefix=local_prefix,
                local_name=local_name,
                operator_prefix=operator_prefix,
                operator_name=operator_name,
            )
        )
        self._validate_common_source_contract(files, local_prefix)
        return OpenMPSoAEmission(files)

    @staticmethod
    def _validate_common_source_contract(files, local_prefix):
        source_by_path = {file.path: file.source for file in files}
        local_name = "%s_local.hpp" % local_prefix
        local_source = source_by_path.get(local_name)
        if local_source is None:
            raise RuntimeError("OpenMP SoA backend did not emit '%s'" % local_name)
        operator_sources = tuple(
            file for file in files if file.path.endswith("_operator.cpp")
        )
        if not operator_sources:
            raise RuntimeError("OpenMP SoA backend did not emit a mesh operator")
        if "template <typename scalar_t, int N_QP" not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' is not templated on N_QP" % local_name
            )
        if "int VECTOR_SIZE" not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' is not templated on VECTOR_SIZE"
                % local_name
            )
        block_name = "%s_" % local_prefix
        if block_name not in local_source:
            raise RuntimeError(
                "OpenMP SoA local kernel '%s' does not use local prefix '%s'"
                % (local_name, local_prefix)
            )
        include = '#include "%s"' % local_name
        for operator in operator_sources:
            if include not in operator.source:
                raise RuntimeError(
                    "OpenMP SoA operator '%s' does not include '%s'"
                    % (operator.path, local_name)
                )

    @staticmethod
    def _validate_energy_plan(unit):
        _require_openmp(unit)
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
        _require_openmp(unit)
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


def _kind_value(kind):
    return getattr(kind, "value", str(kind))


def _generated_prefix(unit):
    return unit.name


def _require_openmp(unit):
    if unit.target is not KernelTarget.OPENMP:
        raise ValueError(
            "OpenMP SoA backend cannot emit target '%s'"
            % getattr(unit.target, "value", unit.target)
        )


def _require_mesh_phases(unit, expected):
    actual = tuple(phase.phase for phase in unit.mesh_phase_plans)
    if actual != tuple(expected):
        raise ValueError(
            "kernel '%s' mesh phase plan %s does not match OpenMP SoA contract %s"
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
            "block '%s' local phase plan %s does not match OpenMP SoA contract %s"
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
    specialization = sfem_soa_element_specialization(
        element_type,
        context.specialization.vector_size,
        context.specialization.quadrature_rule.order,
    )
    affine_specialization = sfem_soa_element_specialization(
        element_type,
        context.affine_specialization.vector_size,
        context.affine_specialization.quadrature_rule.order,
    )
    system = CoupledResidualSystem(collection.source.dim)
    if collection.source.parameters:
        system.add_parameters(*collection.source.parameters)
    for component, component_name in enumerate(_component_field_names(field)):
        lowered = system.add_field(
            component_name,
            field_name=field.name,
            component=component,
            components=field.components,
        )
        system.add_residual(lowered, sp.S.Zero)
    block = collection.block(FormOrder.TWO, unit.block.row_field, unit.block.column_field)
    return _DiagonalBlockModel(
        system,
        element_type,
        specialization,
        affine_specialization,
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
    metadata = collection.form_metadata(order)
    if not unit.is_block:
        return metadata.coefficients
    if unit.block.form_order is not order:
        return _zero_coefficients(collection.source)
    column = unit.block.column_field or None
    block = collection.block(order, unit.block.row_field, column)
    if not block.coefficients:
        raise ValueError(
            "block kernel '%s' has no coefficient sets for '%s'"
            % (unit.name, block.name)
        )
    return _selected_row_coefficients(collection.source, block.coefficients)


def _zero_coefficients(system):
    return tuple(
        WeakResidualCoefficients(
            field.name,
            sp.S.Zero,
            tuple(sp.S.Zero for _ in range(system.dim)),
        )
        for field in system.fields
    )


def _selected_row_coefficients(system, coefficients):
    ret = list(_zero_coefficients(system))
    by_name = {field.name: index for index, field in enumerate(system.fields)}
    for coefficient in coefficients:
        try:
            ret[by_name[coefficient.row_field]] = coefficient
        except KeyError:
            raise ValueError(
                "row field '%s' is not in residual system" % coefficient.row_field
            )
    return tuple(ret)
