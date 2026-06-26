from dataclasses import dataclass
from enum import Enum

from .tensor_product import (
    TensorProductSumFactorizationPlan,
    tensor_product_geometry_jacobian_plan_from_sizes,
)


class GeometryMode(Enum):
    AFFINE = "affine"
    ISOPARAMETRIC = "isoparametric"


class GeometryInputLayout(Enum):
    ADJUGATE_DETERMINANT_SOA = "adjugate_determinant_soa"
    COORDINATE_AOS = "coordinate_aos"


class GeometryEvaluation(Enum):
    ROUTE_PRECOMPUTED_AFFINE = "route_precomputed_affine"
    SIMPLEX_REFERENCE = "simplex_reference"
    TENSOR_PRODUCT_SUM_FACTOR = "tensor_product_sum_factor"


@dataclass(frozen=True)
class GeometryPlanNode:
    mode: GeometryMode
    element_type: str
    dim: int
    n_shape: int
    n_qp: int
    input_layout: GeometryInputLayout
    evaluation: GeometryEvaluation
    jacobian_scope: str
    geometry_points_per_element: int
    geometry_stream_count: int
    sum_factorization_plan: object = None

    def __post_init__(self):
        mode = GeometryMode(self.mode)
        input_layout = GeometryInputLayout(self.input_layout)
        evaluation = GeometryEvaluation(self.evaluation)
        dim = int(self.dim)
        n_shape = int(self.n_shape)
        n_qp = int(self.n_qp)
        points = int(self.geometry_points_per_element)
        streams = int(self.geometry_stream_count)
        sum_factorization_plan = self.sum_factorization_plan
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "element_type", str(self.element_type).upper())
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "n_qp", n_qp)
        object.__setattr__(self, "input_layout", input_layout)
        object.__setattr__(self, "evaluation", evaluation)
        object.__setattr__(self, "jacobian_scope", str(self.jacobian_scope))
        object.__setattr__(self, "geometry_points_per_element", points)
        object.__setattr__(self, "geometry_stream_count", streams)
        object.__setattr__(self, "sum_factorization_plan", sum_factorization_plan)
        if dim <= 0:
            raise ValueError("geometry plan dimension must be positive")
        if n_shape <= 0 or n_qp <= 0:
            raise ValueError("geometry plan shape and quadrature counts must be positive")
        if points <= 0 or streams <= 0:
            raise ValueError("geometry plan point and stream counts must be positive")
        if self.jacobian_scope not in ("element", "quadrature_point"):
            raise ValueError("geometry plan jacobian_scope must be 'element' or 'quadrature_point'")
        if evaluation is GeometryEvaluation.TENSOR_PRODUCT_SUM_FACTOR:
            if not isinstance(sum_factorization_plan, TensorProductSumFactorizationPlan):
                raise TypeError("tensor-product geometry plans require a TensorProductSumFactorizationPlan")
            if not sum_factorization_plan.evaluates_geometry_jacobian:
                raise ValueError("tensor-product geometry plan must evaluate the geometry Jacobian")
        elif sum_factorization_plan is not None:
            raise ValueError("non tensor-product geometry plans cannot own a sum-factorization plan")

    @property
    def name(self):
        return self.mode.value

    @property
    def is_affine(self):
        return self.mode is GeometryMode.AFFINE

    @property
    def is_isoparametric(self):
        return self.mode is GeometryMode.ISOPARAMETRIC

    @property
    def uses_sum_factorization(self):
        return self.evaluation is GeometryEvaluation.TENSOR_PRODUCT_SUM_FACTOR

    @property
    def requires_coordinates(self):
        return self.input_layout is GeometryInputLayout.COORDINATE_AOS

    @property
    def requires_adjugate_determinant_streams(self):
        return self.input_layout is GeometryInputLayout.ADJUGATE_DETERMINANT_SOA


def affine_geometry_plan(fem_policy):
    rule = fem_policy.quadrature_rule
    return GeometryPlanNode(
        GeometryMode.AFFINE,
        rule.element_type,
        rule.dim,
        rule.n_shape,
        rule.n_qp,
        GeometryInputLayout.ADJUGATE_DETERMINANT_SOA,
        GeometryEvaluation.ROUTE_PRECOMPUTED_AFFINE,
        "element",
        1,
        rule.dim * rule.dim + 1,
    )


def isoparametric_geometry_plan(fem_policy):
    rule = fem_policy.quadrature_rule
    tensor_product = fem_policy.family == "tensor_product"
    sum_factorization_plan = None
    if tensor_product:
        sum_factorization_plan = tensor_product_geometry_jacobian_plan_from_sizes(
            rule.dim,
            rule.n_shape,
            rule.n_qp,
            rule.tensor_product_n_shape_1d,
            rule.tensor_product_n_qp_1d,
        )
    return GeometryPlanNode(
        GeometryMode.ISOPARAMETRIC,
        rule.element_type,
        rule.dim,
        rule.n_shape,
        rule.n_qp,
        GeometryInputLayout.COORDINATE_AOS,
        GeometryEvaluation.TENSOR_PRODUCT_SUM_FACTOR
        if tensor_product
        else GeometryEvaluation.SIMPLEX_REFERENCE,
        "quadrature_point",
        rule.n_qp,
        rule.dim * rule.dim + 1,
        sum_factorization_plan,
    )


def geometry_plans_for_fem_policy(fem_policy, modes=None):
    modes = (GeometryMode.AFFINE, GeometryMode.ISOPARAMETRIC) if modes is None else tuple(modes)
    plans = []
    for mode in modes:
        mode = GeometryMode(mode)
        if mode is GeometryMode.AFFINE:
            plans.append(affine_geometry_plan(fem_policy))
        elif mode is GeometryMode.ISOPARAMETRIC:
            plans.append(isoparametric_geometry_plan(fem_policy))
        else:
            raise ValueError("unsupported geometry mode '%s'" % mode.value)
    return tuple(plans)
