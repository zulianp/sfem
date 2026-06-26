from dataclasses import dataclass
from enum import Enum

from .fem import sfem_element_quadrature_rule
from .tensor_product import (
    TensorProductOperation,
    TensorProductSumFactorizationPlan,
    tensor_product_sum_factorization_plan,
)


class BasisFamily(Enum):
    SIMPLEX = "simplex"
    TENSOR_PRODUCT = "tensor_product"


class BasisEvaluation(Enum):
    DIRECT_REFERENCE = "direct_reference"
    TENSOR_PRODUCT_SUM_FACTOR = "tensor_product_sum_factor"


class BasisDataLayout(Enum):
    QP_SHAPE = "qp_shape"
    TENSOR_PRODUCT_1D = "tensor_product_1d"


@dataclass(frozen=True)
class BasisPlanNode:
    role: str
    element_type: str
    cell_element_type: str
    dim: int
    n_shape: int
    n_qp: int
    family: BasisFamily
    evaluation: BasisEvaluation
    data_layout: BasisDataLayout
    n_shape_1d: int = 0
    n_qp_1d: int = 0
    reference_shape_size: int = 0
    reference_gradient_size: int = 0
    sum_factorization_plans: tuple = ()

    def __post_init__(self):
        family = BasisFamily(self.family)
        evaluation = BasisEvaluation(self.evaluation)
        data_layout = BasisDataLayout(self.data_layout)
        dim = int(self.dim)
        n_shape = int(self.n_shape)
        n_qp = int(self.n_qp)
        n_shape_1d = int(self.n_shape_1d)
        n_qp_1d = int(self.n_qp_1d)
        reference_shape_size = int(self.reference_shape_size)
        reference_gradient_size = int(self.reference_gradient_size)
        sum_factorization_plans = tuple(self.sum_factorization_plans)
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "element_type", str(self.element_type).upper())
        object.__setattr__(self, "cell_element_type", str(self.cell_element_type).upper())
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "n_qp", n_qp)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "evaluation", evaluation)
        object.__setattr__(self, "data_layout", data_layout)
        object.__setattr__(self, "n_shape_1d", n_shape_1d)
        object.__setattr__(self, "n_qp_1d", n_qp_1d)
        object.__setattr__(self, "reference_shape_size", reference_shape_size)
        object.__setattr__(self, "reference_gradient_size", reference_gradient_size)
        object.__setattr__(self, "sum_factorization_plans", sum_factorization_plans)
        if not self.role:
            raise ValueError("basis plan requires a role")
        if dim <= 0 or n_shape <= 0 or n_qp <= 0:
            raise ValueError("basis plan dimension, shape count, and quadrature count must be positive")
        if family is BasisFamily.TENSOR_PRODUCT:
            if evaluation is not BasisEvaluation.TENSOR_PRODUCT_SUM_FACTOR:
                raise ValueError("tensor-product basis plans must use sum-factor evaluation")
            if data_layout is not BasisDataLayout.TENSOR_PRODUCT_1D:
                raise ValueError("tensor-product basis plans must use 1D data layout")
            if n_shape_1d <= 0 or n_qp_1d <= 0:
                raise ValueError("tensor-product basis plans require positive 1D sizes")
            if not sum_factorization_plans:
                raise ValueError("tensor-product basis plans require sum-factorization plans")
            for plan in sum_factorization_plans:
                if not isinstance(plan, TensorProductSumFactorizationPlan):
                    raise TypeError("sum_factorization_plans must contain TensorProductSumFactorizationPlan objects")
        else:
            if evaluation is not BasisEvaluation.DIRECT_REFERENCE:
                raise ValueError("simplex basis plans must use direct reference evaluation")
            if data_layout is not BasisDataLayout.QP_SHAPE:
                raise ValueError("simplex basis plans must use qp-shape data layout")
            if sum_factorization_plans:
                raise ValueError("simplex basis plans cannot own sum-factorization plans")

    @property
    def is_simplex(self):
        return self.family is BasisFamily.SIMPLEX

    @property
    def is_tensor_product(self):
        return self.family is BasisFamily.TENSOR_PRODUCT

    @property
    def uses_sum_factorization(self):
        return self.evaluation is BasisEvaluation.TENSOR_PRODUCT_SUM_FACTOR

    @property
    def scatter_n_shape(self):
        return self.n_shape

    @property
    def field_evaluation_sum_factorization(self):
        return self._sum_factorization_for(TensorProductOperation.FIELD_VALUE)

    @property
    def test_contraction_sum_factorization(self):
        return self._sum_factorization_for(TensorProductOperation.TEST_VALUE_CONTRACTION)

    def _sum_factorization_for(self, operation):
        operation = TensorProductOperation(operation)
        for plan in self.sum_factorization_plans:
            if operation in plan.operations:
                return plan
        raise ValueError("sum-factorization operation '%s' is not available" % operation.value)


def basis_plan_for_quadrature_rule(rule, role="cell"):
    return basis_plan_for_element_at_cell_rule(rule.element_type, rule, role)


def basis_plan_for_element_at_cell_rule(element_type, cell_rule, role):
    element_type = str(element_type).upper()
    element_rule = _field_rule_for_cell_rule(element_type, cell_rule)
    if element_rule.dim != cell_rule.dim:
        raise ValueError(
            "basis element '%s' has dimension %d but cell element '%s' has dimension %d"
            % (element_type, element_rule.dim, cell_rule.element_type, cell_rule.dim)
        )

    if element_rule.is_tensor_product and cell_rule.is_tensor_product:
        n_shape_1d = element_rule.tensor_product_n_shape_1d
        n_qp_1d = cell_rule.tensor_product_n_qp_1d
        return BasisPlanNode(
            role,
            element_type,
            cell_rule.element_type,
            cell_rule.dim,
            element_rule.n_shape,
            cell_rule.n_qp,
            BasisFamily.TENSOR_PRODUCT,
            BasisEvaluation.TENSOR_PRODUCT_SUM_FACTOR,
            BasisDataLayout.TENSOR_PRODUCT_1D,
            n_shape_1d,
            n_qp_1d,
            n_shape_1d * n_qp_1d,
            n_shape_1d * n_qp_1d,
            (
                tensor_product_sum_factorization_plan(
                    cell_rule.dim,
                    element_rule.n_shape,
                    cell_rule.n_qp,
                    n_shape_1d,
                    n_qp_1d,
                    (
                        TensorProductOperation.FIELD_VALUE,
                        TensorProductOperation.FIELD_GRADIENT,
                    ),
                ),
                tensor_product_sum_factorization_plan(
                    cell_rule.dim,
                    element_rule.n_shape,
                    cell_rule.n_qp,
                    n_shape_1d,
                    n_qp_1d,
                    (
                        TensorProductOperation.TEST_VALUE_CONTRACTION,
                        TensorProductOperation.TEST_GRADIENT_CONTRACTION,
                    ),
                ),
            ),
        )

    return BasisPlanNode(
        role,
        element_type,
        cell_rule.element_type,
        cell_rule.dim,
        element_rule.n_shape,
        cell_rule.n_qp,
        BasisFamily.SIMPLEX,
        BasisEvaluation.DIRECT_REFERENCE,
        BasisDataLayout.QP_SHAPE,
        0,
        0,
        cell_rule.n_qp * element_rule.n_shape,
        cell_rule.n_qp * element_rule.n_shape * cell_rule.dim,
    )


def basis_plans_for_fem_policy(fem_policy):
    return (basis_plan_for_quadrature_rule(fem_policy.quadrature_rule, "cell"),)


def field_basis_plan_for_fem_policy(fem_policy, field):
    element_type = fem_policy.element_for_field(field)
    role = "field:%s" % field.name
    return basis_plan_for_element_at_cell_rule(
        element_type,
        fem_policy.quadrature_rule,
        role,
    )


def field_basis_plans_for_fem_policy(fem_policy, fields):
    return tuple(field_basis_plan_for_fem_policy(fem_policy, field) for field in fields)


def _field_rule_for_cell_rule(element_type, cell_rule):
    if element_type in ("QUAD4", "HEX8", "HEX27"):
        return sfem_element_quadrature_rule(element_type, cell_rule.order)
    return sfem_element_quadrature_rule(element_type)
