from dataclasses import dataclass, replace
from enum import Enum

from codegen.framework.fem.reference import sfem_field_n_shape


MATRIX_FORMAT_PLAN_SCHEMA_VERSION = 3

SIMPLEX_AFFINE_DIA_ELEMENTS = frozenset(("TRI3", "TET4"))
TENSOR_PRODUCT_DIA_ELEMENTS = frozenset(
    (
        "QUAD4",
        "PROTEUS_QUAD4",
        "HEX8",
        "HEX27",
        "PROTEUS_HEX8",
        "PROTEUS_HEX27",
        "PROTEUS_HEX64",
        "PROTEUS_HEX125",
        "PROTEUS_HEX729",
    )
)


class MatrixFormat(Enum):
    CRS = "crs"
    BSR = "bsr"
    DIA = "dia"
    COO = "coo"
    PATCH = "patch"
    BLOCK_DIAG_SYM = "block_diag_sym"


class MatrixMeshLayout(Enum):
    STANDARD = "standard"
    PACKED = "packed"


class PackedAssemblyPass(Enum):
    NONE = "none"
    ONE_PASS = "one_pass"
    TWO_PASS = "two_pass"


@dataclass(frozen=True)
class CRSAssemblyPlan:
    row_pointer: str = "rowptr"
    column_index: str = "colidx"
    value_stream: str = "values"
    element_connectivity: str = "elements"
    mesh_access: str = "standard_block_elements"
    pack_index_type: str = "idx_t"
    pack_partition: str = "none"
    packed_node_partition: str = "none"
    value_mapping: str = "identity"
    block_offsets: str = "element_dof_offsets"
    accumulation_policy: str = "add_scatter"
    structural_compatibility: str = "requires_full_graph"
    reduction_policy: str = "atomic_add"
    row_dofs_per_element: int = 0
    column_dofs_per_element: int = 0
    entries_per_element: int = 0

    def to_dict(self):
        return {
            "kind": "crs",
            "row_pointer": self.row_pointer,
            "column_index": self.column_index,
            "value_stream": self.value_stream,
            "element_connectivity": self.element_connectivity,
            "mesh_access": self.mesh_access,
            "pack_index_type": self.pack_index_type,
            "pack_partition": self.pack_partition,
            "packed_node_partition": self.packed_node_partition,
            "value_mapping": self.value_mapping,
            "block_offsets": self.block_offsets,
            "accumulation_policy": self.accumulation_policy,
            "structural_compatibility": self.structural_compatibility,
            "reduction_policy": self.reduction_policy,
            "row_dofs_per_element": self.row_dofs_per_element,
            "column_dofs_per_element": self.column_dofs_per_element,
            "entries_per_element": self.entries_per_element,
        }


@dataclass(frozen=True)
class BSRAssemblyPlan:
    row_pointer: str = "rowptr"
    column_index: str = "colidx"
    value_stream: str = "values"
    element_connectivity: str = "elements"
    mesh_access: str = "standard_block_elements"
    pack_index_type: str = "idx_t"
    pack_partition: str = "none"
    packed_node_partition: str = "none"
    value_mapping: str = "identity"
    block_value_layout: str = "node_major_row_component_column_component"
    component_ordering: str = "node_major_component_contiguous"
    accumulation_policy: str = "add_scatter"
    structural_compatibility: str = "requires_node_block_graph"
    reduction_policy: str = "atomic_add"
    row_block_size: int = 1
    column_block_size: int = 1
    block_size: int = 1
    block_rows_per_element: int = 0
    block_columns_per_element: int = 0
    block_entries_per_element: int = 0
    compatible_block_size: bool = True

    def to_dict(self):
        return {
            "kind": "bsr",
            "row_pointer": self.row_pointer,
            "column_index": self.column_index,
            "value_stream": self.value_stream,
            "element_connectivity": self.element_connectivity,
            "mesh_access": self.mesh_access,
            "pack_index_type": self.pack_index_type,
            "pack_partition": self.pack_partition,
            "packed_node_partition": self.packed_node_partition,
            "value_mapping": self.value_mapping,
            "block_value_layout": self.block_value_layout,
            "component_ordering": self.component_ordering,
            "accumulation_policy": self.accumulation_policy,
            "structural_compatibility": self.structural_compatibility,
            "reduction_policy": self.reduction_policy,
            "row_block_size": self.row_block_size,
            "column_block_size": self.column_block_size,
            "block_size": self.block_size,
            "block_rows_per_element": self.block_rows_per_element,
            "block_columns_per_element": self.block_columns_per_element,
            "block_entries_per_element": self.block_entries_per_element,
            "compatible_block_size": self.compatible_block_size,
        }


@dataclass(frozen=True)
class DIAAssemblyPlan:
    diagonal_offsets: str = "diagonal_offsets"
    value_stream: str = "values"
    element_connectivity: str = "elements"
    mesh_access: str = "standard_block_elements"
    pack_index_type: str = "idx_t"
    pack_partition: str = "none"
    packed_node_partition: str = "none"
    value_mapping: str = "identity"
    stride: str = "nnodes"
    value_layout: str = "diagonal_node_block_row_major"
    stencil_compatibility: str = "requires_stable_diagonal_structure"
    accumulation_policy: str = "fill_diagonal_values"
    structural_compatibility: str = "runtime_validated_diagonal_offsets"
    reduction_policy: str = "atomic_add"
    row_dofs_per_element: int = 0
    values_per_element: int = 0

    def to_dict(self):
        return {
            "kind": "dia",
            "diagonal_offsets": self.diagonal_offsets,
            "value_stream": self.value_stream,
            "element_connectivity": self.element_connectivity,
            "mesh_access": self.mesh_access,
            "pack_index_type": self.pack_index_type,
            "pack_partition": self.pack_partition,
            "packed_node_partition": self.packed_node_partition,
            "value_mapping": self.value_mapping,
            "stride": self.stride,
            "value_layout": self.value_layout,
            "stencil_compatibility": self.stencil_compatibility,
            "accumulation_policy": self.accumulation_policy,
            "structural_compatibility": self.structural_compatibility,
            "reduction_policy": self.reduction_policy,
            "row_dofs_per_element": self.row_dofs_per_element,
            "values_per_element": self.values_per_element,
        }


@dataclass(frozen=True)
class COOAssemblyPlan:
    row_index_stream: str = "rowidx"
    column_index_stream: str = "colidx"
    value_stream: str = "values"
    element_connectivity: str = "elements"
    mesh_access: str = "standard_block_elements"
    pack_index_type: str = "idx_t"
    pack_partition: str = "none"
    packed_node_partition: str = "none"
    value_mapping: str = "identity"
    duplicate_policy: str = "deterministic_element_order_external_reduction"
    sort_policy: str = "external_stable_sort_or_existing_sfem_coo_reduce"
    reduction_phase: str = "non_hot_setup_phase"
    accumulation_policy: str = "emit_triplets"
    structural_compatibility: str = "allows_duplicates"
    entries_per_element: int = 0

    def to_dict(self):
        return {
            "kind": "coo",
            "row_index_stream": self.row_index_stream,
            "column_index_stream": self.column_index_stream,
            "value_stream": self.value_stream,
            "element_connectivity": self.element_connectivity,
            "mesh_access": self.mesh_access,
            "pack_index_type": self.pack_index_type,
            "pack_partition": self.pack_partition,
            "packed_node_partition": self.packed_node_partition,
            "value_mapping": self.value_mapping,
            "duplicate_policy": self.duplicate_policy,
            "sort_policy": self.sort_policy,
            "reduction_phase": self.reduction_phase,
            "accumulation_policy": self.accumulation_policy,
            "structural_compatibility": self.structural_compatibility,
            "entries_per_element": self.entries_per_element,
        }


@dataclass(frozen=True)
class PatchAssemblyPlan:
    patch_graph: str = "rowptr_colidx"
    value_stream: str = "values"
    element_connectivity: str = "elements"
    mesh_access: str = "standard_block_elements"
    pack_index_type: str = "idx_t"
    pack_partition: str = "none"
    packed_node_partition: str = "none"
    value_mapping: str = "identity"
    patch_value_layout: str = "node_block_crs_block_row_major"
    node_index_filter: bool = False
    accumulation_policy: str = "add_scatter"
    structural_compatibility: str = "requires_full_graph"
    reduction_policy: str = "atomic_add"
    row_dofs_per_patch: int = 0
    column_dofs_per_patch: int = 0
    entries_per_patch: int = 0

    def to_dict(self):
        return {
            "kind": "patch",
            "patch_graph": self.patch_graph,
            "value_stream": self.value_stream,
            "element_connectivity": self.element_connectivity,
            "mesh_access": self.mesh_access,
            "pack_index_type": self.pack_index_type,
            "pack_partition": self.pack_partition,
            "packed_node_partition": self.packed_node_partition,
            "value_mapping": self.value_mapping,
            "patch_value_layout": self.patch_value_layout,
            "node_index_filter": self.node_index_filter,
            "accumulation_policy": self.accumulation_policy,
            "structural_compatibility": self.structural_compatibility,
            "reduction_policy": self.reduction_policy,
            "row_dofs_per_patch": self.row_dofs_per_patch,
            "column_dofs_per_patch": self.column_dofs_per_patch,
            "entries_per_patch": self.entries_per_patch,
        }


@dataclass(frozen=True)
class BlockDiagSymAssemblyPlan:
    value_stream: str = "values"
    element_connectivity: str = "elements"
    mesh_access: str = "standard_block_elements"
    pack_index_type: str = "idx_t"
    pack_partition: str = "none"
    packed_node_partition: str = "none"
    value_mapping: str = "identity"
    value_layout: str = "node_major_upper_symmetric_aos"
    accumulation_policy: str = "add_node_diagonal_blocks"
    structural_compatibility: str = "requires_square_vector_block"
    reduction_policy: str = "atomic_add"
    block_size: int = 1
    symmetric_entries_per_node: int = 1
    value_writes_per_element: int = 0

    def to_dict(self):
        return {
            "kind": "block_diag_sym",
            "value_stream": self.value_stream,
            "element_connectivity": self.element_connectivity,
            "mesh_access": self.mesh_access,
            "pack_index_type": self.pack_index_type,
            "pack_partition": self.pack_partition,
            "packed_node_partition": self.packed_node_partition,
            "value_mapping": self.value_mapping,
            "value_layout": self.value_layout,
            "accumulation_policy": self.accumulation_policy,
            "structural_compatibility": self.structural_compatibility,
            "reduction_policy": self.reduction_policy,
            "block_size": self.block_size,
            "symmetric_entries_per_node": self.symmetric_entries_per_node,
            "value_writes_per_element": self.value_writes_per_element,
        }


@dataclass(frozen=True)
class MatrixAssemblyVariantPlan:
    matrix_format: MatrixFormat
    mesh_layout: MatrixMeshLayout = MatrixMeshLayout.STANDARD
    packed_pass: PackedAssemblyPass = PackedAssemblyPass.NONE
    node_index_filter: bool = False
    row_dofs_per_element: int = 0
    column_dofs_per_element: int = 0
    entries_per_element: int = 0
    index_reads_per_element: int = 0
    value_writes_per_element: int = 0
    expected_flops_per_element: int = 0
    expected_bytes_per_element: int = 0
    assembly_plan: object = None

    def __post_init__(self):
        matrix_format = MatrixFormat(self.matrix_format)
        mesh_layout = MatrixMeshLayout(self.mesh_layout)
        packed_pass = PackedAssemblyPass(self.packed_pass)
        node_index_filter = bool(self.node_index_filter)
        if mesh_layout is MatrixMeshLayout.STANDARD and packed_pass is not PackedAssemblyPass.NONE:
            raise ValueError("standard matrix assembly cannot request packed passes")
        if mesh_layout is MatrixMeshLayout.PACKED and packed_pass is PackedAssemblyPass.NONE:
            raise ValueError("packed matrix assembly requires one_pass or two_pass")
        for name in (
            "row_dofs_per_element",
            "column_dofs_per_element",
            "entries_per_element",
            "index_reads_per_element",
            "value_writes_per_element",
            "expected_flops_per_element",
            "expected_bytes_per_element",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError("%s must be non-negative" % name)
        object.__setattr__(self, "matrix_format", matrix_format)
        object.__setattr__(self, "mesh_layout", mesh_layout)
        object.__setattr__(self, "packed_pass", packed_pass)
        object.__setattr__(self, "node_index_filter", node_index_filter)

    @property
    def name(self):
        parts = [self.matrix_format.value, self.mesh_layout.value]
        if self.packed_pass is not PackedAssemblyPass.NONE:
            parts.append(self.packed_pass.value)
        if self.node_index_filter:
            parts.append("indexed")
        return "_".join(parts)

    @property
    def is_packed(self):
        return self.mesh_layout is MatrixMeshLayout.PACKED

    def to_dict(self):
        return {
            "name": self.name,
            "format": self.matrix_format.value,
            "mesh_layout": self.mesh_layout.value,
            "packed_pass": self.packed_pass.value,
            "node_index_filter": self.node_index_filter,
            "row_dofs_per_element": self.row_dofs_per_element,
            "column_dofs_per_element": self.column_dofs_per_element,
            "entries_per_element": self.entries_per_element,
            "index_reads_per_element": self.index_reads_per_element,
            "value_writes_per_element": self.value_writes_per_element,
            "expected_flops_per_element": self.expected_flops_per_element,
            "expected_bytes_per_element": self.expected_bytes_per_element,
            "assembly_plan": None
            if self.assembly_plan is None
            else self.assembly_plan.to_dict(),
        }


@dataclass(frozen=True)
class MatrixFormatPlan:
    variants: tuple = ()

    def __post_init__(self):
        variants = tuple(self.variants)
        seen = set()
        for variant in variants:
            if not isinstance(variant, MatrixAssemblyVariantPlan):
                raise TypeError("matrix-format variants must be MatrixAssemblyVariantPlan objects")
            if variant.name in seen:
                raise ValueError("duplicate matrix-format variant '%s'" % variant.name)
            seen.add(variant.name)
        object.__setattr__(self, "variants", variants)

    @property
    def is_empty(self):
        return not self.variants

    @property
    def formats(self):
        return tuple(dict.fromkeys(variant.matrix_format for variant in self.variants))

    def to_dict(self):
        return {
            "schema": "sfem.matrix_format_plan",
            "schema_version": MATRIX_FORMAT_PLAN_SCHEMA_VERSION,
            "variants": [variant.to_dict() for variant in self.variants],
        }


def matrix_format_plan_from_request(
    formats=(),
    mesh_layouts=("standard",),
    packed_passes=("one_pass", "two_pass"),
    patch_node_index_filter=False,
):
    formats = _normalize_formats(formats)
    if not formats:
        return MatrixFormatPlan(())
    mesh_layouts = _normalize_mesh_layouts(mesh_layouts)
    packed_passes = _normalize_packed_passes(packed_passes)
    variants = []
    for matrix_format in formats:
        for mesh_layout in mesh_layouts:
            if mesh_layout is MatrixMeshLayout.PACKED:
                for packed_pass in packed_passes:
                    variants.append(
                        MatrixAssemblyVariantPlan(
                            matrix_format,
                            mesh_layout,
                            packed_pass,
                            node_index_filter=_uses_node_index_filter(
                                matrix_format,
                                patch_node_index_filter,
                            ),
                        )
                    )
            else:
                variants.append(
                    MatrixAssemblyVariantPlan(
                        matrix_format,
                        mesh_layout,
                        PackedAssemblyPass.NONE,
                        node_index_filter=_uses_node_index_filter(
                            matrix_format,
                            patch_node_index_filter,
                        ),
                    )
                )
    return MatrixFormatPlan(tuple(variants))


def specialize_matrix_format_plan(plan, unit, context):
    if plan is None or plan.is_empty:
        return MatrixFormatPlan(())
    row_layouts, column_layouts = _element_dof_layouts(unit, context)
    row_dofs = sum(layout["dofs"] for layout in row_layouts)
    column_dofs = sum(layout["dofs"] for layout in column_layouts)
    entries = row_dofs * column_dofs
    index_reads = row_dofs + column_dofs
    flops = max(1, 2 * entries)
    return MatrixFormatPlan(
        tuple(
            replace(
                variant,
                row_dofs_per_element=row_dofs,
                column_dofs_per_element=column_dofs,
                entries_per_element=entries,
                index_reads_per_element=index_reads,
                value_writes_per_element=_value_writes_per_element(
                    variant,
                    row_layouts,
                    column_layouts,
                    row_dofs,
                    entries,
                ),
                expected_flops_per_element=_expected_flops_per_element(variant, flops),
                expected_bytes_per_element=_expected_bytes_per_element(
                    variant,
                    row_layouts,
                    column_layouts,
                    row_dofs,
                    entries,
                    index_reads,
                ),
                assembly_plan=_assembly_plan_for_variant(
                    variant,
                    row_layouts,
                    column_layouts,
                    row_dofs,
                    column_dofs,
                    entries,
                ),
            )
            for variant in plan.variants
        )
    )


def _normalize_formats(values):
    raw = _as_tokens(values)
    if not raw:
        return ()
    if "all" in raw:
        raw = tuple(format_.value for format_ in MatrixFormat)
    return tuple(dict.fromkeys(MatrixFormat(value) for value in raw))


def _normalize_mesh_layouts(values):
    raw = _as_tokens(values)
    if not raw:
        return (MatrixMeshLayout.STANDARD,)
    if "all" in raw:
        raw = ("standard", "packed")
    return tuple(dict.fromkeys(MatrixMeshLayout(value) for value in raw))


def _normalize_packed_passes(values):
    raw = _as_tokens(values)
    if not raw:
        return (PackedAssemblyPass.ONE_PASS, PackedAssemblyPass.TWO_PASS)
    if "all" in raw:
        raw = ("one_pass", "two_pass")
    passes = tuple(dict.fromkeys(PackedAssemblyPass(value) for value in raw))
    if any(pass_ is PackedAssemblyPass.NONE for pass_ in passes):
        raise ValueError("packed pass selection cannot include none")
    return passes


def _as_tokens(values):
    if values is None:
        return ()
    if isinstance(values, str):
        values = values.split(",")
    tokens = []
    for value in values:
        if isinstance(value, Enum):
            token = value.value
        else:
            token = str(value)
        for part in token.split(","):
            part = part.strip().lower()
            if part:
                tokens.append(part)
    return tuple(tokens)


def _uses_node_index_filter(matrix_format, patch_node_index_filter):
    return False


def _element_dof_layouts(unit, context):
    row_fields, column_fields = _matrix_fields(unit)
    fields = tuple(unit.form_collection.fields)
    if getattr(context.fem_policy, "is_mixed_order", False):
        _validate_mixed_order_field_mappings(context, fields)
    layouts_by_field = {}
    for field, element_type in context.fem_policy.field_element_types_for(fields):
        components = int(getattr(field, "components", 1))
        n_shape = sfem_field_n_shape(element_type)
        layouts_by_field[field.name] = {
            "field": field.name,
            "element_type": str(element_type),
            "n_shape": n_shape,
            "components": components,
            "dofs": n_shape * components,
        }
    row_layouts = tuple(layouts_by_field[name] for name in row_fields)
    column_layouts = tuple(layouts_by_field[name] for name in column_fields)
    return row_layouts, column_layouts


def _validate_mixed_order_field_mappings(context, fields):
    compatibility = context.fem_policy.compatibility
    mapped_families = {
        str(family)
        for family, _ in getattr(compatibility, "field_element_types", ())
    }
    missing = []
    for field in fields:
        family = str(getattr(field, "family", "") or getattr(field, "name", ""))
        if family not in mapped_families:
            missing.append(family)
    if missing:
        raise ValueError(
            "missing matrix field-element mapping for mixed-order field families: %s"
            % ", ".join(dict.fromkeys(missing))
        )


def _matrix_fields(unit):
    fields = tuple(unit.form_collection.fields)
    if getattr(unit, "is_block", False) and unit.block is not None:
        row = (unit.block.row_field,)
        column = (unit.block.column_field or unit.block.row_field,)
        return row, column
    names = tuple(field.name for field in fields)
    return names, names


def _value_writes_per_element(variant, row_layouts, column_layouts, row_dofs, entries):
    if variant.matrix_format is MatrixFormat.DIA:
        return max(1, row_dofs)
    if variant.matrix_format is MatrixFormat.BLOCK_DIAG_SYM:
        block_size = _component_block_size(row_layouts)
        if (
            block_size != _component_block_size(column_layouts)
            or not _single_matching_field_layout(row_layouts, column_layouts)
        ):
            return 0
        return max(1, (row_dofs // block_size) * block_size * (block_size + 1) // 2)
    return entries


def _expected_flops_per_element(variant, flops):
    if variant.mesh_layout is MatrixMeshLayout.PACKED and variant.packed_pass is PackedAssemblyPass.TWO_PASS:
        return flops + max(1, flops // 8)
    return flops


def _expected_bytes_per_element(variant, row_layouts, column_layouts, row_dofs, entries, index_reads):
    scalar_bytes = 8
    index_bytes = 4
    pass_multiplier = 2 if variant.packed_pass is PackedAssemblyPass.TWO_PASS else 1
    if variant.matrix_format is MatrixFormat.DIA:
        output_entries = max(1, row_dofs)
    elif variant.matrix_format is MatrixFormat.BLOCK_DIAG_SYM:
        output_entries = _value_writes_per_element(
            variant,
            row_layouts,
            column_layouts,
            row_dofs,
            entries,
        )
    else:
        output_entries = entries
    return pass_multiplier * (
        output_entries * scalar_bytes
        + index_reads * index_bytes
        + entries * scalar_bytes
    )


def _assembly_plan_for_variant(
    variant,
    row_layouts,
    column_layouts,
    row_dofs,
    column_dofs,
    entries,
):
    mesh_contract = _mesh_contract_for_variant(variant)
    if variant.matrix_format is MatrixFormat.CRS:
        return CRSAssemblyPlan(
            **mesh_contract,
            row_dofs_per_element=row_dofs,
            column_dofs_per_element=column_dofs,
            entries_per_element=entries,
        )
    if variant.matrix_format is MatrixFormat.BSR:
        row_block_size = _component_block_size(row_layouts)
        column_block_size = _component_block_size(column_layouts)
        compatible = row_block_size == column_block_size
        block_size = row_block_size if compatible else 0
        return BSRAssemblyPlan(
            **mesh_contract,
            row_block_size=row_block_size,
            column_block_size=column_block_size,
            block_size=block_size,
            block_rows_per_element=_block_count(row_dofs, row_block_size),
            block_columns_per_element=_block_count(column_dofs, column_block_size),
            block_entries_per_element=(
                _block_count(row_dofs, row_block_size)
                * _block_count(column_dofs, column_block_size)
            ),
            compatible_block_size=compatible,
        )
    if variant.matrix_format is MatrixFormat.DIA:
        dia_contract = _dia_structure_contract(row_layouts, column_layouts)
        return DIAAssemblyPlan(
            **mesh_contract,
            **dia_contract,
            row_dofs_per_element=row_dofs,
            values_per_element=max(1, row_dofs),
        )
    if variant.matrix_format is MatrixFormat.COO:
        return COOAssemblyPlan(**mesh_contract, entries_per_element=entries)
    if variant.matrix_format is MatrixFormat.PATCH:
        return PatchAssemblyPlan(
            **mesh_contract,
            node_index_filter=variant.node_index_filter,
            row_dofs_per_patch=row_dofs,
            column_dofs_per_patch=column_dofs,
            entries_per_patch=entries,
        )
    if variant.matrix_format is MatrixFormat.BLOCK_DIAG_SYM:
        block_size = _component_block_size(row_layouts)
        square_vector_block = (
            block_size == _component_block_size(column_layouts)
            and _single_matching_field_layout(row_layouts, column_layouts)
        )
        if not square_vector_block:
            return BlockDiagSymAssemblyPlan(
                **mesh_contract,
                block_size=0,
                symmetric_entries_per_node=0,
                value_writes_per_element=0,
                structural_compatibility="unsupported_mixed_or_asymmetric_block_diag_sym",
                reduction_policy="not_emitted",
            )
        symmetric_entries = block_size * (block_size + 1) // 2
        return BlockDiagSymAssemblyPlan(
            **mesh_contract,
            block_size=block_size,
            symmetric_entries_per_node=symmetric_entries,
            value_writes_per_element=_block_count(row_dofs, block_size) * symmetric_entries,
        )
    raise ValueError("unsupported matrix format '%s'" % variant.matrix_format.value)


def _mesh_contract_for_variant(variant):
    if variant.mesh_layout is MatrixMeshLayout.PACKED:
        return {
            "element_connectivity": "packed->elements(block)->data()",
            "mesh_access": "FunctionSpace::PackedMesh",
            "pack_index_type": "FunctionSpace::PackedIdxType",
            "pack_partition": "n_packs/n_elements_per_pack/max_nodes_per_pack",
            "packed_node_partition": "owned_nodes_ptr/n_shared/ghost_ptr/ghost_idx",
            "value_mapping": "PackedMesh::map_to_packed/map_to_unpacked",
        }
    return {
        "element_connectivity": "elements",
        "mesh_access": "standard_block_elements",
        "pack_index_type": "idx_t",
        "pack_partition": "none",
        "packed_node_partition": "none",
        "value_mapping": "identity",
    }


def _component_block_size(layouts):
    components = tuple(layout["components"] for layout in layouts)
    if not components:
        return 1
    block_size = components[0]
    for components_i in components[1:]:
        if components_i != block_size:
            return 1
    return max(1, block_size)


def _block_count(dofs, block_size):
    block_size = max(1, int(block_size))
    if dofs % block_size != 0:
        return dofs
    return dofs // block_size


def _dia_structure_contract(row_layouts, column_layouts):
    if not _single_matching_field_layout(row_layouts, column_layouts):
        return {
            "stencil_compatibility": "unsupported_mixed_or_asymmetric_diagonal_structure",
            "structural_compatibility": "unsupported_mixed_or_asymmetric_diagonal_structure",
            "reduction_policy": "not_emitted",
        }

    element_type = row_layouts[0]["element_type"].upper()
    if element_type in SIMPLEX_AFFINE_DIA_ELEMENTS:
        return {
            "stencil_compatibility": "stable_simplex_affine_diagonal_offsets",
            "structural_compatibility": "stable_simplex_affine_diagonal_offsets",
        }
    if element_type in TENSOR_PRODUCT_DIA_ELEMENTS:
        return {
            "stencil_compatibility": "stable_tensor_product_diagonal_offsets",
            "structural_compatibility": "stable_tensor_product_diagonal_offsets",
        }
    return {
        "stencil_compatibility": "runtime_validated_diagonal_offsets",
        "structural_compatibility": "runtime_validated_diagonal_offsets",
    }


def _single_matching_field_layout(row_layouts, column_layouts):
    if len(row_layouts) != 1 or len(column_layouts) != 1:
        return False
    row = row_layouts[0]
    column = column_layouts[0]
    return (
        row["field"] == column["field"]
        and row["element_type"] == column["element_type"]
        and row["components"] == column["components"]
    )
