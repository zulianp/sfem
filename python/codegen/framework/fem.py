from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class SfemSoAArrayInput:
    name: str
    size: int
    scalar_type: str = "scalar_t"
    layout: str = "element_stream"
    n_qp: int = 1
    n_shape: int = 1
    components: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "n_qp", int(self.n_qp))
        object.__setattr__(self, "n_shape", int(self.n_shape))
        components = self.size if self.components is None else int(self.components)
        object.__setattr__(self, "components", components)
        if self.size <= 0:
            raise ValueError("SoA array input size must be positive")
        if self.n_qp <= 0:
            raise ValueError("SoA array input n_qp must be positive")
        if self.n_shape <= 0:
            raise ValueError("SoA array input n_shape must be positive")
        if self.components <= 0:
            raise ValueError("SoA array input components must be positive")
        if self.layout not in ("element_stream", "reference_qp_shape"):
            raise ValueError("SoA array input layout must be 'element_stream' or 'reference_qp_shape'")
        if self.layout == "reference_qp_shape":
            expected_size = self.n_qp * self.n_shape * self.components
            if self.size != expected_size:
                raise ValueError(
                    "reference_qp_shape input size must equal n_qp * n_shape * components"
                )

    @property
    def is_reference_qp_shape(self):
        return self.layout == "reference_qp_shape"

    @property
    def local_size(self):
        if self.is_reference_qp_shape:
            return self.n_shape * self.components
        return self.size


def sfem_soa_array_input(name, size, scalar_type="scalar_t"):
    return SfemSoAArrayInput(name, size, scalar_type)


def sfem_soa_reference_input(name, n_qp, n_shape, components, scalar_type="scalar_t"):
    return SfemSoAArrayInput(
        name,
        int(n_qp) * int(n_shape) * int(components),
        scalar_type,
        "reference_qp_shape",
        n_qp,
        n_shape,
        components,
    )


@dataclass(frozen=True)
class SfemElementQuadratureRule:
    element_type: str
    dim: int
    n_shape: int
    weights: Tuple[float, ...]
    reference_gradients: Tuple[float, ...]
    order: int = 1
    tensor_product_shape_values_1d: Tuple[float, ...] = ()
    tensor_product_shape_gradients_1d: Tuple[float, ...] = ()
    tensor_product_weights_1d: Tuple[float, ...] = ()
    tensor_product_dim: int = 0

    def __post_init__(self):
        element_type = str(self.element_type).upper()
        dim = int(self.dim)
        n_shape = int(self.n_shape)
        weights = tuple(float(value) for value in self.weights)
        reference_gradients = tuple(float(value) for value in self.reference_gradients)
        tensor_product_shape_values_1d = tuple(
            float(value) for value in self.tensor_product_shape_values_1d
        )
        tensor_product_shape_gradients_1d = tuple(
            float(value) for value in self.tensor_product_shape_gradients_1d
        )
        tensor_product_weights_1d = tuple(
            float(value) for value in self.tensor_product_weights_1d
        )
        tensor_product_dim = int(self.tensor_product_dim)
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "reference_gradients", reference_gradients)
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(
            self,
            "tensor_product_shape_values_1d",
            tensor_product_shape_values_1d,
        )
        object.__setattr__(
            self,
            "tensor_product_shape_gradients_1d",
            tensor_product_shape_gradients_1d,
        )
        object.__setattr__(self, "tensor_product_weights_1d", tensor_product_weights_1d)
        object.__setattr__(self, "tensor_product_dim", tensor_product_dim)
        if dim <= 0:
            raise ValueError("element quadrature dim must be positive")
        if n_shape <= 0:
            raise ValueError("element quadrature n_shape must be positive")
        if not weights:
            raise ValueError("element quadrature rule must have at least one point")
        expected = len(weights) * n_shape * dim
        if len(reference_gradients) != expected:
            raise ValueError(
                "reference_gradients must have N_QP * N_SHAPE * dim entries"
            )
        has_tensor_product_data = (
            bool(tensor_product_shape_values_1d)
            or bool(tensor_product_shape_gradients_1d)
            or bool(tensor_product_weights_1d)
            or tensor_product_dim != 0
        )
        if has_tensor_product_data:
            if tensor_product_dim != dim:
                raise ValueError("tensor_product_dim must match dim")
            if not tensor_product_weights_1d:
                raise ValueError("tensor-product quadrature must provide 1D weights")
            n_qp_1d = len(tensor_product_weights_1d)
            expected_1d = n_qp_1d * self.tensor_product_n_shape_1d
            if len(tensor_product_shape_values_1d) != expected_1d:
                raise ValueError("tensor-product shape values must be N_QP_1D * N_SHAPE_1D")
            if len(tensor_product_shape_gradients_1d) != expected_1d:
                raise ValueError("tensor-product shape gradients must be N_QP_1D * N_SHAPE_1D")
            if n_shape != self.tensor_product_n_shape_1d ** dim:
                raise ValueError("tensor-product n_shape does not match 1D shape count")
            if len(weights) != n_qp_1d ** dim:
                raise ValueError("tensor-product weights do not match 1D quadrature count")

    @property
    def n_qp(self):
        return len(self.weights)

    @property
    def is_tensor_product(self):
        return bool(self.tensor_product_weights_1d)

    @property
    def tensor_product_n_qp_1d(self):
        return len(self.tensor_product_weights_1d)

    @property
    def tensor_product_n_shape_1d(self):
        if not self.tensor_product_weights_1d:
            return 0
        return len(self.tensor_product_shape_values_1d) // len(self.tensor_product_weights_1d)


@dataclass(frozen=True)
class SfemSoAElementSpecialization:
    quadrature_rule: SfemElementQuadratureRule
    vector_size: int = 16

    def __post_init__(self):
        object.__setattr__(self, "vector_size", int(self.vector_size))
        if self.vector_size <= 0:
            raise ValueError("vector_size must be positive")

    @property
    def element_type(self):
        return self.quadrature_rule.element_type

    @property
    def dim(self):
        return self.quadrature_rule.dim

    @property
    def n_shape(self):
        return self.quadrature_rule.n_shape

    @property
    def n_qp(self):
        return self.quadrature_rule.n_qp

    def reference_shape_gradient_input(self, name="grad_ref"):
        return sfem_soa_reference_input(name, self.n_qp, self.n_shape, self.dim)

    def adjugate_geometry_inputs(
        self,
        grad_ref_name="grad_ref",
        adjugate_name="jacobian_adjugate",
        determinant_name="jacobian_determinant",
    ):
        try:
            from .symbolic import sfem_soa_adjugate_geometry_inputs
        except ImportError:
            from symbolic import sfem_soa_adjugate_geometry_inputs

        return sfem_soa_adjugate_geometry_inputs(
            self,
            grad_ref_name=grad_ref_name,
            adjugate_name=adjugate_name,
            determinant_name=determinant_name,
        )


@dataclass(frozen=True)
class SfemCompatibleElement:
    name: str
    cell_element_type: str
    field_element_types: tuple

    def __post_init__(self):
        name = str(self.name).upper()
        cell_element_type = str(self.cell_element_type).upper()
        fields = tuple((str(field), str(element).upper()) for field, element in self.field_element_types)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "cell_element_type", cell_element_type)
        object.__setattr__(self, "field_element_types", fields)
        if not name or not cell_element_type:
            raise ValueError("compatible element requires a name and cell element")
        if not fields:
            raise ValueError("compatible element requires at least one field family")

    @property
    def is_mixed_order(self):
        return any(element != self.cell_element_type for _, element in self.field_element_types)

    def element_for_field(self, field_name):
        field_name = str(field_name)
        for field, element in self.field_element_types:
            if field == field_name:
                return element
        return self.cell_element_type


def sfem_element_quadrature_rule(element_type, order=None):
    element_type = str(element_type).upper()
    if order is None:
        order = (
            3
            if element_type == "HEX27"
            else 2
            if element_type in ("QUAD4", "HEX8", "TRI6", "TET10")
            else 1
        )
    order = int(order)

    if element_type == "TRI3":
        weights = (0.5,)
        gradients = (-1.0, -1.0, 1.0, 0.0, 0.0, 1.0)
        return SfemElementQuadratureRule(element_type, 2, 3, weights, gradients, order)

    if element_type == "TET4":
        weights = (1.0 / 6.0,)
        gradients = (
            -1.0,
            -1.0,
            -1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        return SfemElementQuadratureRule(element_type, 3, 4, weights, gradients, order)

    if element_type == "TRI6":
        if order != 2:
            raise ValueError("TRI6 currently supports quadrature order 2")
        points = (
            (1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0),
        )
        weights = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
        gradients = []
        for x, y in points:
            gradients.extend(_tri6_reference_gradients(x, y))
        return SfemElementQuadratureRule(element_type, 2, 6, weights, gradients, order)

    if element_type == "TET10":
        if order != 2:
            raise ValueError("TET10 currently supports quadrature order 2")
        a = 0.5854101966249685
        b = 0.1381966011250105
        points = (
            (b, b, b),
            (a, b, b),
            (b, a, b),
            (b, b, a),
        )
        weights = (1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0)
        gradients = []
        for x, y, z in points:
            gradients.extend(_tet10_reference_gradients(x, y, z))
        return SfemElementQuadratureRule(element_type, 3, 10, weights, gradients, order)

    if element_type == "QUAD4":
        points, weights_1d = _sfem_unit_interval_gauss_rule(order)
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q1_1d_shapes(points)
        gradients = []
        weights = []
        for y, wy in zip(points, weights_1d):
            for x, wx in zip(points, weights_1d):
                weights.append(wx * wy)
                gradients.extend(
                    (
                        -(1.0 - y),
                        -(1.0 - x),
                        1.0 - y,
                        -x,
                        y,
                        x,
                        -y,
                        1.0 - x,
                    )
                )
        return SfemElementQuadratureRule(
            element_type,
            2,
            4,
            weights,
            gradients,
            order,
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            2,
        )

    if element_type == "HEX8":
        points, weights_1d = _sfem_unit_interval_gauss_rule(order)
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q1_1d_shapes(points)
        gradients = []
        weights = []
        for z, wz in zip(points, weights_1d):
            for y, wy in zip(points, weights_1d):
                for x, wx in zip(points, weights_1d):
                    weights.append(wx * wy * wz)
                    xm = 1.0 - x
                    ym = 1.0 - y
                    zm = 1.0 - z
                    gradients.extend(
                        (
                            -ym * zm,
                            -xm * zm,
                            -xm * ym,
                            ym * zm,
                            -x * zm,
                            -x * ym,
                            y * zm,
                            x * zm,
                            -x * y,
                            -y * zm,
                            xm * zm,
                            -xm * y,
                            -ym * z,
                            -xm * z,
                            xm * ym,
                            ym * z,
                            -x * z,
                            x * ym,
                            y * z,
                            x * z,
                            x * y,
                            -y * z,
                            xm * z,
                            xm * y,
                        )
                    )
        return SfemElementQuadratureRule(
            element_type,
            3,
            8,
            weights,
            gradients,
            order,
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            3,
        )

    if element_type == "HEX27":
        if order != 3:
            raise ValueError("HEX27 currently supports quadrature order 3")
        points, weights_1d = _sfem_unit_interval_gauss_rule(order)
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q2_1d_shapes(points)
        gradients, weights = _sfem_tensor_product_hex_gradients_and_weights(
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            3,
        )
        return SfemElementQuadratureRule(
            element_type,
            3,
            27,
            weights,
            gradients,
            order,
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            3,
        )

    raise ValueError("unsupported element type '%s'" % element_type)


def sfem_supported_element_types():
    return ("TRI3", "TRI6", "QUAD4", "TET4", "TET10", "HEX8", "HEX27")


def sfem_taylor_hood_element_types():
    return (
        SfemCompatibleElement(
            "TRI6_TRI3",
            "TRI6",
            (("displacement", "TRI6"), ("pressure", "TRI3")),
        ),
        SfemCompatibleElement(
            "TET10_TET4",
            "TET10",
            (("displacement", "TET10"), ("pressure", "TET4")),
        ),
        SfemCompatibleElement(
            "HEX27_HEX8",
            "HEX27",
            (("displacement", "HEX27"), ("pressure", "HEX8")),
        ),
    )


def sfem_soa_element_specializations(element_types=None, vector_size=16, quadrature_order=None):
    element_types = sfem_supported_element_types() if element_types is None else tuple(element_types)
    return tuple(
        sfem_soa_element_specialization(element_type, vector_size, quadrature_order)
        for element_type in element_types
    )


def sfem_soa_element_specialization(element_type, vector_size=16, quadrature_order=None):
    return SfemSoAElementSpecialization(
        sfem_element_quadrature_rule(element_type, quadrature_order),
        vector_size,
    )


def _tri6_reference_gradients(x, y):
    return (
        -3.0 + 4.0 * x + 4.0 * y,
        -3.0 + 4.0 * x + 4.0 * y,
        4.0 * x - 1.0,
        0.0,
        0.0,
        4.0 * y - 1.0,
        4.0 - 8.0 * x - 4.0 * y,
        -4.0 * x,
        4.0 * y,
        4.0 * x,
        -4.0 * y,
        4.0 - 4.0 * x - 8.0 * y,
    )


def _tet10_reference_gradients(x, y, z):
    dx = (
        4.0 * x + 4.0 * y + 4.0 * z - 3.0,
        4.0 * x - 1.0,
        0.0,
        0.0,
        -8.0 * x - 4.0 * y - 4.0 * z + 4.0,
        4.0 * y,
        -4.0 * y,
        -4.0 * z,
        4.0 * z,
        0.0,
    )
    dy = (
        4.0 * x + 4.0 * y + 4.0 * z - 3.0,
        0.0,
        4.0 * y - 1.0,
        0.0,
        -4.0 * x,
        4.0 * x,
        -8.0 * y - 4.0 * x - 4.0 * z + 4.0,
        -4.0 * z,
        0.0,
        4.0 * z,
    )
    dz = (
        4.0 * x + 4.0 * y + 4.0 * z - 3.0,
        0.0,
        0.0,
        4.0 * z - 1.0,
        -4.0 * x,
        0.0,
        -4.0 * y,
        -8.0 * z - 4.0 * x - 4.0 * y + 4.0,
        4.0 * x,
        4.0 * y,
    )
    gradients = []
    for i in range(10):
        gradients.extend((dx[i], dy[i], dz[i]))
    return tuple(gradients)


def _sfem_unit_interval_gauss_rule(order):
    if order == 1:
        return (0.5,), (1.0,)
    if order == 2:
        offset = 0.5 / math.sqrt(3.0)
        return (0.5 - offset, 0.5 + offset), (0.5, 0.5)
    if order == 3:
        offset = 0.5 * math.sqrt(3.0 / 5.0)
        return (0.5 - offset, 0.5, 0.5 + offset), (5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0)
    raise ValueError("unsupported quadrature order %d" % order)


def _sfem_lagrange_q1_1d_shapes(points):
    shape_values = []
    shape_gradients = []
    for x in points:
        shape_values.extend((1.0 - x, x))
        shape_gradients.extend((-1.0, 1.0))
    return tuple(shape_values), tuple(shape_gradients)


def _sfem_lagrange_q2_1d_shapes(points):
    shape_values = []
    shape_gradients = []
    for x in points:
        shape_values.extend(
            (
                2.0 * x * x - 3.0 * x + 1.0,
                4.0 * x - 4.0 * x * x,
                2.0 * x * x - x,
            )
        )
        shape_gradients.extend((4.0 * x - 3.0, 4.0 - 8.0 * x, 4.0 * x - 1.0))
    return tuple(shape_values), tuple(shape_gradients)


def _sfem_tensor_product_hex_gradients_and_weights(
    shape_values_1d,
    shape_gradients_1d,
    weights_1d,
    n_shape_1d,
):
    n_qp_1d = len(weights_1d)
    gradients = []
    weights = []
    for qz in range(n_qp_1d):
        for qy in range(n_qp_1d):
            for qx in range(n_qp_1d):
                weights.append(weights_1d[qx] * weights_1d[qy] * weights_1d[qz])
                qp_gradients = [None] * (n_shape_1d * n_shape_1d * n_shape_1d)
                for sz in range(n_shape_1d):
                    for sy in range(n_shape_1d):
                        for sx in range(n_shape_1d):
                            sxv = shape_values_1d[qx * n_shape_1d + sx]
                            syv = shape_values_1d[qy * n_shape_1d + sy]
                            szv = shape_values_1d[qz * n_shape_1d + sz]
                            dx = shape_gradients_1d[qx * n_shape_1d + sx] * syv * szv
                            dy = sxv * shape_gradients_1d[qy * n_shape_1d + sy] * szv
                            dz = sxv * syv * shape_gradients_1d[qz * n_shape_1d + sz]
                            shape = _sfem_tensor_hex_shape_index(
                                n_shape_1d,
                                sx,
                                sy,
                                sz,
                            )
                            qp_gradients[shape] = (dx, dy, dz)
                for gradient in qp_gradients:
                    gradients.extend(gradient)
    return tuple(gradients), tuple(weights)


def _sfem_tensor_hex_shape_index(n_shape_1d, sx, sy, sz):
    if n_shape_1d == 2:
        return (sx if sy == 0 else (3 if sx == 0 else 2)) + 4 * sz
    if n_shape_1d == 3:
        cartesian_to_hex27 = (
            0,
            8,
            1,
            11,
            24,
            9,
            3,
            10,
            2,
            16,
            20,
            17,
            23,
            26,
            21,
            19,
            22,
            18,
            4,
            12,
            5,
            15,
            25,
            13,
            7,
            14,
            6,
        )
        return cartesian_to_hex27[sx + 3 * (sy + 3 * sz)]
    raise ValueError("unsupported tensor-product hex order")
