from dataclasses import dataclass
from enum import Enum


class TensorProductOperation(Enum):
    FIELD_VALUE = "field_value"
    FIELD_GRADIENT = "field_gradient"
    GEOMETRY_JACOBIAN = "geometry_jacobian"
    TEST_VALUE_CONTRACTION = "test_value_contraction"
    TEST_GRADIENT_CONTRACTION = "test_gradient_contraction"


class TensorProductDataLayout(Enum):
    BASIS_1D = "basis_1d"
    SHAPE_ORDERED_STREAMS = "shape_ordered_streams"
    QP_VECTOR_SOA = "qp_vector_soa"


@dataclass(frozen=True)
class TensorProductSumFactorizationPlan:
    dim: int
    n_shape: int
    n_qp: int
    n_shape_1d: int
    n_qp_1d: int
    operations: tuple
    input_layout: TensorProductDataLayout = TensorProductDataLayout.SHAPE_ORDERED_STREAMS
    basis_layout: TensorProductDataLayout = TensorProductDataLayout.BASIS_1D
    output_layout: TensorProductDataLayout = TensorProductDataLayout.QP_VECTOR_SOA

    def __post_init__(self):
        dim = int(self.dim)
        n_shape = int(self.n_shape)
        n_qp = int(self.n_qp)
        n_shape_1d = int(self.n_shape_1d)
        n_qp_1d = int(self.n_qp_1d)
        operations = tuple(TensorProductOperation(operation) for operation in self.operations)
        input_layout = TensorProductDataLayout(self.input_layout)
        basis_layout = TensorProductDataLayout(self.basis_layout)
        output_layout = TensorProductDataLayout(self.output_layout)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "n_qp", n_qp)
        object.__setattr__(self, "n_shape_1d", n_shape_1d)
        object.__setattr__(self, "n_qp_1d", n_qp_1d)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "input_layout", input_layout)
        object.__setattr__(self, "basis_layout", basis_layout)
        object.__setattr__(self, "output_layout", output_layout)
        if dim not in (2, 3):
            raise ValueError("tensor-product sum factorization supports dimensions 2 and 3")
        if n_shape <= 0 or n_qp <= 0 or n_shape_1d <= 0 or n_qp_1d <= 0:
            raise ValueError("tensor-product sizes must be positive")
        if n_shape != n_shape_1d ** dim:
            raise ValueError("tensor-product n_shape must equal n_shape_1d ** dim")
        if n_qp != n_qp_1d ** dim:
            raise ValueError("tensor-product n_qp must equal n_qp_1d ** dim")
        if not operations:
            raise ValueError("tensor-product sum factorization requires at least one operation")

    @property
    def evaluates_values(self):
        return TensorProductOperation.FIELD_VALUE in self.operations

    @property
    def evaluates_gradients(self):
        return TensorProductOperation.FIELD_GRADIENT in self.operations

    @property
    def evaluates_geometry_jacobian(self):
        return TensorProductOperation.GEOMETRY_JACOBIAN in self.operations

    @property
    def contracts_tests(self):
        return (
            TensorProductOperation.TEST_VALUE_CONTRACTION in self.operations
            or TensorProductOperation.TEST_GRADIENT_CONTRACTION in self.operations
        )

    @property
    def uses_1d_basis(self):
        return self.basis_layout is TensorProductDataLayout.BASIS_1D


def tensor_product_sum_factorization_plan(
    dim,
    n_shape,
    n_qp,
    n_shape_1d,
    n_qp_1d,
    operations,
):
    return TensorProductSumFactorizationPlan(
        dim,
        n_shape,
        n_qp,
        n_shape_1d,
        n_qp_1d,
        tuple(operations),
    )


def tensor_product_field_evaluation_plan(basis_plan, include_gradient=True):
    operations = [TensorProductOperation.FIELD_VALUE]
    if include_gradient:
        operations.append(TensorProductOperation.FIELD_GRADIENT)
    return tensor_product_sum_factorization_plan(
        basis_plan.dim,
        basis_plan.n_shape,
        basis_plan.n_qp,
        basis_plan.n_shape_1d,
        basis_plan.n_qp_1d,
        operations,
    )


def tensor_product_geometry_jacobian_plan(geometry_plan, n_shape_1d, n_qp_1d):
    return tensor_product_sum_factorization_plan(
        geometry_plan.dim,
        geometry_plan.n_shape,
        geometry_plan.n_qp,
        n_shape_1d,
        n_qp_1d,
        (TensorProductOperation.GEOMETRY_JACOBIAN,),
    )


def tensor_product_geometry_jacobian_plan_from_sizes(dim, n_shape, n_qp, n_shape_1d, n_qp_1d):
    return tensor_product_sum_factorization_plan(
        dim,
        n_shape,
        n_qp,
        n_shape_1d,
        n_qp_1d,
        (TensorProductOperation.GEOMETRY_JACOBIAN,),
    )


def tensor_product_test_contraction_plan(basis_plan, include_gradient=True):
    operations = [TensorProductOperation.TEST_VALUE_CONTRACTION]
    if include_gradient:
        operations.append(TensorProductOperation.TEST_GRADIENT_CONTRACTION)
    return tensor_product_sum_factorization_plan(
        basis_plan.dim,
        basis_plan.n_shape,
        basis_plan.n_qp,
        basis_plan.n_shape_1d,
        basis_plan.n_qp_1d,
        operations,
    )


def tensor_product_cartesian_shape_order(dim, n_shape):
    if dim == 2 and n_shape == 4:
        return (0, 1, 3, 2)
    if dim == 3 and n_shape == 8:
        return (0, 1, 3, 2, 4, 5, 7, 6)
    if dim == 3 and n_shape == 27:
        return (
            0, 8, 1,
            11, 24, 9,
            3, 10, 2,
            16, 20, 17,
            23, 26, 21,
            19, 22, 18,
            4, 12, 5,
            15, 25, 13,
            7, 14, 6,
        )
    return tuple(range(n_shape))


def streams_in_shape_order(streams, n_components, shape_order):
    if len(streams) != n_components * len(shape_order):
        raise ValueError("stream count must be component count * number of shapes")
    return tuple(
        streams[shape * n_components + component]
        for shape in shape_order
        for component in range(n_components)
    )
