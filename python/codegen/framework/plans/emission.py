from dataclasses import dataclass

from codegen.framework.fem.basis import BasisFamily, basis_plans_for_fem_policy
from codegen.framework.fem.reference import sfem_fem_policy
from codegen.framework.fem.geometry import (
    GeometryMode,
    affine_geometry_plan,
    isoparametric_geometry_plan,
)


@dataclass(frozen=True)
class ElementEmissionPlan:
    element_type: str
    label: str
    family: str
    vector_size: int
    affine_geometry: object
    isoparametric_geometry: object
    basis_plans: tuple
    affine_specialization: object
    isoparametric_specialization: object

    def __post_init__(self):
        element_type = str(self.element_type).upper()
        label = str(self.label).lower()
        family = str(self.family)
        vector_size = int(self.vector_size)
        basis_plans = tuple(self.basis_plans)
        if family not in ("simplex", "tensor_product"):
            raise ValueError("emission plan family must be simplex or tensor_product")
        if vector_size <= 0:
            raise ValueError("emission plan vector size must be positive")
        if self.affine_geometry.mode is not GeometryMode.AFFINE:
            raise ValueError("emission plan requires affine geometry")
        if self.isoparametric_geometry.mode is not GeometryMode.ISOPARAMETRIC:
            raise ValueError("emission plan requires isoparametric geometry")
        if not basis_plans:
            raise ValueError("emission plan requires basis plans")
        for basis in basis_plans:
            if basis.family is BasisFamily.TENSOR_PRODUCT and family != "tensor_product":
                raise ValueError("tensor-product basis requires tensor-product family")
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "vector_size", vector_size)
        object.__setattr__(self, "basis_plans", basis_plans)

    @property
    def quadrature_order(self):
        return self.isoparametric_specialization.quadrature_rule.order

    @property
    def affine_quadrature_order(self):
        return self.affine_specialization.quadrature_rule.order

    @property
    def uses_tensor_product_geometry(self):
        return self.isoparametric_geometry.uses_sum_factorization

    @property
    def uses_tensor_product_basis(self):
        return any(basis.uses_sum_factorization for basis in self.basis_plans)

    @property
    def basis_family(self):
        return "tensor_product" if self.uses_tensor_product_basis else "simplex"

    @property
    def geometry_family(self):
        return "tensor_product" if self.uses_tensor_product_geometry else "simplex"


def emission_plan_from_unit_context(unit, context):
    affine = _geometry_for_mode(unit, GeometryMode.AFFINE)
    isoparametric = _geometry_for_mode(unit, GeometryMode.ISOPARAMETRIC)
    basis_plans = tuple(
        basis
        for block in unit.blocks
        for basis in block.basis_plans
    )
    if not basis_plans:
        basis_plans = tuple(context.basis_plans)
    return ElementEmissionPlan(
        context.element_type,
        context.label,
        context.family,
        context.specialization.vector_size,
        affine,
        isoparametric,
        tuple(dict.fromkeys(basis_plans)),
        context.affine_specialization,
        context.specialization,
    )


def emission_plan_for_element(
    element,
    vector_size,
    isoparametric_quadrature_order=None,
    *,
    isoparametric_integration_case="standard",
    affine_quadrature_order=None,
    affine_integration_case="standard",
):
    policy = sfem_fem_policy(
        element,
        vector_size,
        isoparametric_quadrature_order,
        integration_case=isoparametric_integration_case,
    )
    affine_policy = sfem_fem_policy(
        element,
        vector_size,
        affine_quadrature_order,
        integration_case=affine_integration_case,
    )
    return ElementEmissionPlan(
        policy.cell_element_type,
        policy.label,
        policy.family,
        policy.specialization.vector_size,
        affine_geometry_plan(affine_policy),
        isoparametric_geometry_plan(policy),
        basis_plans_for_fem_policy(policy),
        affine_policy.specialization,
        policy.specialization,
    )


def _geometry_for_mode(unit, mode):
    mode = GeometryMode(mode)
    for phase in unit.mesh_phase_plans:
        for geometry in phase.geometries:
            if geometry.mode is mode:
                return geometry
    raise ValueError(
        "kernel plan '%s' has no %s geometry phase data"
        % (unit.name, mode.value)
    )
