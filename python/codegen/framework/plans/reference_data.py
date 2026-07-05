from dataclasses import dataclass

from codegen.framework.fem.basis import basis_plan_for_element_at_cell_rule
from codegen.framework.fem.reference import sfem_simplex_grad_ref_name
from codegen.framework.fem.geometry import GeometryMode


@dataclass(frozen=True)
class ReferenceBasisDataPlan:
    role: str
    element_type: str
    accessor_prefix: str
    family: str
    n_shape: int
    n_qp: int
    n_shape_1d: int = 0
    n_qp_1d: int = 0
    shape_accessor: str = ""
    gradient_accessors: tuple = ()

    def __post_init__(self):
        role = str(self.role)
        element_type = str(self.element_type).upper()
        accessor_prefix = str(self.accessor_prefix)
        family = str(self.family)
        n_shape = int(self.n_shape)
        n_qp = int(self.n_qp)
        n_shape_1d = int(self.n_shape_1d)
        n_qp_1d = int(self.n_qp_1d)
        shape_accessor = str(self.shape_accessor)
        gradient_accessors = tuple(str(name) for name in self.gradient_accessors)
        if not role:
            raise ValueError("reference basis plan requires a role")
        if family not in ("simplex", "tensor_product"):
            raise ValueError("reference basis family must be simplex or tensor_product")
        if n_shape <= 0 or n_qp <= 0:
            raise ValueError("reference basis n_shape and n_qp must be positive")
        if family == "tensor_product" and (n_shape_1d <= 0 or n_qp_1d <= 0):
            raise ValueError("tensor-product reference basis requires 1D sizes")
        if not shape_accessor:
            raise ValueError("reference basis plan requires a shape accessor")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "accessor_prefix", accessor_prefix)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "n_qp", n_qp)
        object.__setattr__(self, "n_shape_1d", n_shape_1d)
        object.__setattr__(self, "n_qp_1d", n_qp_1d)
        object.__setattr__(self, "shape_accessor", shape_accessor)
        object.__setattr__(self, "gradient_accessors", gradient_accessors)

    @property
    def is_tensor_product(self):
        return self.family == "tensor_product"

    @property
    def accessors(self):
        return (self.shape_accessor,) + self.gradient_accessors

    def accessor_call(self, struct_name, accessor, scalar_type="scalar_t"):
        return "sfem::codegen::%s<%s>::%s()" % (
            str(struct_name),
            str(scalar_type),
            str(accessor),
        )

    def to_dict(self):
        return {
            "role": self.role,
            "element_type": self.element_type,
            "accessor_prefix": self.accessor_prefix,
            "family": self.family,
            "n_shape": self.n_shape,
            "n_qp": self.n_qp,
            "n_shape_1d": self.n_shape_1d,
            "n_qp_1d": self.n_qp_1d,
            "shape_accessor": self.shape_accessor,
            "gradient_accessors": list(self.gradient_accessors),
        }


@dataclass(frozen=True)
class ReferenceDataSetPlan:
    stage: str
    struct_name: str
    geometry_mode: GeometryMode
    family: str
    cell_element_type: str
    n_qp: int
    n_shape: int
    weight_accessor: str
    basis_entries: tuple
    field_element_types: tuple = ()
    n_qp_1d: int = 0

    def __post_init__(self):
        stage = str(self.stage)
        struct_name = str(self.struct_name)
        geometry_mode = GeometryMode(self.geometry_mode)
        family = str(self.family)
        cell_element_type = str(self.cell_element_type).upper()
        n_qp = int(self.n_qp)
        n_shape = int(self.n_shape)
        weight_accessor = str(self.weight_accessor)
        basis_entries = tuple(self.basis_entries)
        field_element_types = tuple(
            (str(field), str(element).upper())
            for field, element in self.field_element_types
        )
        n_qp_1d = int(self.n_qp_1d)
        if stage not in ("affine", "isoparametric"):
            raise ValueError("reference-data stage must be affine or isoparametric")
        if not struct_name:
            raise ValueError("reference-data plan requires a struct name")
        if family not in ("simplex", "tensor_product"):
            raise ValueError("reference-data family must be simplex or tensor_product")
        if n_qp <= 0 or n_shape <= 0:
            raise ValueError("reference-data n_qp and n_shape must be positive")
        if family == "tensor_product" and n_qp_1d <= 0:
            raise ValueError("tensor-product reference data requires n_qp_1d")
        if not weight_accessor:
            raise ValueError("reference-data plan requires a weight accessor")
        for basis in basis_entries:
            if not isinstance(basis, ReferenceBasisDataPlan):
                raise TypeError("basis_entries must contain ReferenceBasisDataPlan objects")
            if basis.family != family:
                raise ValueError("basis family does not match reference-data family")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "struct_name", struct_name)
        object.__setattr__(self, "geometry_mode", geometry_mode)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "cell_element_type", cell_element_type)
        object.__setattr__(self, "n_qp", n_qp)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "weight_accessor", weight_accessor)
        object.__setattr__(self, "basis_entries", basis_entries)
        object.__setattr__(self, "field_element_types", field_element_types)
        object.__setattr__(self, "n_qp_1d", n_qp_1d)

    @property
    def is_tensor_product(self):
        return self.family == "tensor_product"

    @property
    def accessors(self):
        ret = [self.weight_accessor]
        for basis in self.basis_entries:
            ret.extend(basis.accessors)
        return tuple(dict.fromkeys(ret))

    @property
    def unique_element_types(self):
        return tuple(dict.fromkeys(basis.element_type for basis in self.basis_entries))

    @property
    def is_mixed_order(self):
        return len(self.unique_element_types) > 1

    def accessor_call(self, accessor, scalar_type="scalar_t"):
        return "sfem::codegen::%s<%s>::%s()" % (
            self.struct_name,
            str(scalar_type),
            str(accessor),
        )

    def to_dict(self):
        return {
            "stage": self.stage,
            "struct_name": self.struct_name,
            "geometry_mode": self.geometry_mode.value,
            "family": self.family,
            "cell_element_type": self.cell_element_type,
            "n_qp": self.n_qp,
            "n_shape": self.n_shape,
            "n_qp_1d": self.n_qp_1d,
            "weight_accessor": self.weight_accessor,
            "accessors": list(self.accessors),
            "field_element_types": list(self.field_element_types),
            "basis_entries": [basis.to_dict() for basis in self.basis_entries],
        }


@dataclass(frozen=True)
class ReferenceDataPlan:
    prefix: str
    element_label: str
    affine: ReferenceDataSetPlan
    isoparametric: ReferenceDataSetPlan

    def __post_init__(self):
        prefix = str(self.prefix)
        element_label = str(self.element_label).lower()
        if not prefix or not element_label:
            raise ValueError("reference-data plan requires prefix and element label")
        if self.affine.stage != "affine":
            raise ValueError("reference-data plan affine dataset has wrong stage")
        if self.isoparametric.stage != "isoparametric":
            raise ValueError("reference-data plan isoparametric dataset has wrong stage")
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "element_label", element_label)

    @property
    def datasets(self):
        return (self.affine, self.isoparametric)

    @property
    def is_mixed_order(self):
        return any(dataset.is_mixed_order for dataset in self.datasets)

    @property
    def families(self):
        return tuple(dict.fromkeys(dataset.family for dataset in self.datasets))

    def dataset(self, stage):
        stage = str(stage)
        for dataset in self.datasets:
            if dataset.stage == stage:
                return dataset
        raise ValueError("reference-data stage '%s' is not available" % stage)

    def to_dict(self):
        return {
            "prefix": self.prefix,
            "element_label": self.element_label,
            "is_mixed_order": self.is_mixed_order,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }


def reference_data_plan_from_emission_plan(
    unit,
    context,
    emission_plan,
    prefix=None,
    field_element_types=None,
):
    prefix = unit.name if prefix is None else str(prefix)
    fields = _fields_for_unit(unit)
    field_types = _field_element_types(fields, context, emission_plan, field_element_types)
    return ReferenceDataPlan(
        prefix,
        emission_plan.label,
        _reference_dataset_plan(
            prefix,
            "affine",
            GeometryMode.AFFINE,
            emission_plan.affine_specialization.quadrature_rule,
            emission_plan.basis_family,
            field_types,
        ),
        _reference_dataset_plan(
            prefix,
            "isoparametric",
            GeometryMode.ISOPARAMETRIC,
            emission_plan.isoparametric_specialization.quadrature_rule,
            emission_plan.basis_family,
            field_types,
        ),
    )


def validate_reference_data_plan(
    plan,
    prefix,
    affine_rule,
    isoparametric_rule,
    family,
):
    if not isinstance(plan, ReferenceDataPlan):
        raise TypeError("reference_data_plan must be a ReferenceDataPlan")
    if plan.prefix != str(prefix):
        raise ValueError(
            "reference-data plan prefix '%s' does not match kernel prefix '%s'"
            % (plan.prefix, prefix)
        )
    _validate_reference_dataset(plan.affine, "affine", affine_rule, family)
    _validate_reference_dataset(
        plan.isoparametric,
        "isoparametric",
        isoparametric_rule,
        family,
    )
    return plan


def _validate_reference_dataset(dataset, stage, rule, family):
    if dataset.stage != stage:
        raise ValueError("reference-data dataset has wrong stage '%s'" % dataset.stage)
    if dataset.family != str(family):
        raise ValueError(
            "reference-data dataset family '%s' does not match '%s'"
            % (dataset.family, family)
        )
    if dataset.cell_element_type != rule.element_type:
        raise ValueError(
            "reference-data dataset cell element '%s' does not match '%s'"
            % (dataset.cell_element_type, rule.element_type)
        )
    if dataset.n_qp != rule.n_qp or dataset.n_shape != rule.n_shape:
        raise ValueError(
            "reference-data dataset has (%d qp, %d shape), expected (%d qp, %d shape)"
            % (dataset.n_qp, dataset.n_shape, rule.n_qp, rule.n_shape)
        )
    if dataset.is_tensor_product and dataset.n_qp_1d != rule.tensor_product_n_qp_1d:
        raise ValueError(
            "reference-data tensor-product N_QP_1D %d does not match %d"
            % (dataset.n_qp_1d, rule.tensor_product_n_qp_1d)
        )


def _reference_dataset_plan(prefix, stage, mode, cell_rule, family, field_types):
    basis_entries = _basis_entries(cell_rule, family, field_types)
    return ReferenceDataSetPlan(
        stage,
        "%s_%s_reference_data" % (prefix, stage),
        mode,
        family,
        cell_rule.element_type,
        cell_rule.n_qp,
        cell_rule.n_shape,
        "q_weight_1d" if family == "tensor_product" else "q_weight",
        basis_entries,
        field_types,
        cell_rule.tensor_product_n_qp_1d if family == "tensor_product" else 0,
    )


def _basis_entries(cell_rule, family, field_types):
    element_types = [cell_rule.element_type]
    element_types.extend(element for _, element in field_types)
    element_types = tuple(dict.fromkeys(str(element).upper() for element in element_types))
    prefix_accessors = len(element_types) > 1
    ret = []
    for index, element_type in enumerate(element_types):
        role = "cell" if index == 0 else "field:%s" % element_type.lower()
        basis = basis_plan_for_element_at_cell_rule(element_type, cell_rule, role)
        if family == "tensor_product" and not basis.is_tensor_product:
            raise ValueError(
                "tensor-product reference-data plan cannot use simplex basis '%s'"
                % element_type
            )
        if family == "simplex" and not basis.is_simplex:
            raise ValueError(
                "simplex reference-data plan cannot use tensor-product basis '%s'"
                % element_type
            )
        accessor_prefix = element_type.lower() if prefix_accessors else ""
        ret.append(_basis_entry_from_basis(basis, family, accessor_prefix))
    return tuple(ret)


def _basis_entry_from_basis(basis, family, accessor_prefix):
    if family == "tensor_product":
        shape = _prefixed_name(accessor_prefix, "shape_1d")
        grads = (_prefixed_name(accessor_prefix, "grad_1d"),)
    else:
        grad_prefix = _prefixed_name(accessor_prefix, "grad_ref")
        shape = _prefixed_name(accessor_prefix, "shape")
        grads = tuple(sfem_simplex_grad_ref_name(grad_prefix, d) for d in range(basis.dim))
    return ReferenceBasisDataPlan(
        basis.role,
        basis.element_type,
        accessor_prefix,
        family,
        basis.n_shape,
        basis.n_qp,
        basis.n_shape_1d,
        basis.n_qp_1d,
        shape,
        grads,
    )


def _prefixed_name(prefix, suffix):
    prefix = str(prefix)
    suffix = str(suffix)
    if not prefix:
        return suffix
    return "%s_%s" % (prefix, suffix)


def _fields_for_unit(unit):
    fields = tuple(getattr(unit.form_collection, "fields", ()))
    block = getattr(unit, "block", None)
    if block is None:
        return fields
    names = [block.row_field]
    if getattr(block, "column_field", ""):
        names.append(block.column_field)
    names = tuple(dict.fromkeys(names))
    by_name = {field.name: field for field in fields}
    return tuple(by_name[name] for name in names if name in by_name)


def _field_element_types(fields, context, emission_plan, field_element_types):
    fields = tuple(fields)
    if not fields:
        return ()
    provided = _field_element_type_map(field_element_types)
    ret = []
    for field in fields:
        element_type = provided.get(field.name)
        if element_type is None:
            try:
                element_type = context.fem_policy.element_for_field(field)
            except AttributeError:
                element_type = emission_plan.element_type
        ret.append((field.name, element_type))
    return tuple(ret)


def _field_element_type_map(field_element_types):
    if field_element_types is None:
        return {}
    if isinstance(field_element_types, dict):
        return {str(name): str(element).upper() for name, element in field_element_types.items()}
    return {
        str(getattr(field, "name", field)): str(element).upper()
        for field, element in field_element_types
    }
