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
class SfemReferenceData:
    name: str
    values: Tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        if not self.name:
            raise ValueError("reference data requires a name")


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


def _has_mixed_order_fields(cell_element_type, field_element_types):
    return any(element != cell_element_type for _, element in field_element_types)


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
        return _has_mixed_order_fields(
            self.cell_element_type,
            self.field_element_types,
        )

    def element_for_field(self, field_name):
        field_name = str(field_name)
        for field, element in self.field_element_types:
            if field == field_name:
                return element
        return self.cell_element_type


@dataclass(frozen=True)
class SfemElementBasisPolicy:
    element_type: str
    dim: int
    n_shape: int
    family: str
    degree: int
    cell: str
    is_tensor_product: bool

    def __post_init__(self):
        element_type = str(self.element_type).upper()
        dim = int(self.dim)
        n_shape = int(self.n_shape)
        family = str(self.family)
        degree = int(self.degree)
        cell = str(self.cell)
        is_tensor_product = bool(self.is_tensor_product)
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "is_tensor_product", is_tensor_product)
        if family not in ("simplex", "tensor_product"):
            raise ValueError("basis family must be 'simplex' or 'tensor_product'")
        if dim <= 0 or n_shape <= 0 or degree <= 0:
            raise ValueError("basis policy dim, n_shape, and degree must be positive")


@dataclass(frozen=True)
class SfemFieldFamilyCompatibilityPolicy:
    cell_element_type: str
    field_element_types: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "cell_element_type", str(self.cell_element_type).upper())
        object.__setattr__(
            self,
            "field_element_types",
            tuple(
                (str(family), str(element).upper())
                for family, element in self.field_element_types
            ),
        )

    @classmethod
    def from_element(cls, element):
        if isinstance(element, SfemCompatibleElement):
            return cls(element.cell_element_type, element.field_element_types)
        return cls(str(element).upper(), ())

    @property
    def is_mixed_order(self):
        return _has_mixed_order_fields(
            self.cell_element_type,
            self.field_element_types,
        )

    def element_for_family(self, family):
        family = str(family)
        for field_family, element in self.field_element_types:
            if field_family == family:
                return element
        return self.cell_element_type

    def element_for_field(self, field):
        family = getattr(field, "family", "") or getattr(field, "name", "")
        return self.element_for_family(family)

    def field_element_types_for(self, fields):
        return tuple((field, self.element_for_field(field)) for field in fields)


@dataclass(frozen=True)
class SfemFEMPolicy:
    element: object
    label: str
    cell_element_type: str
    basis: SfemElementBasisPolicy
    quadrature_rule: SfemElementQuadratureRule
    specialization: SfemSoAElementSpecialization
    compatibility: SfemFieldFamilyCompatibilityPolicy
    compatible_element: object = None

    @property
    def family(self):
        return self.basis.family

    @property
    def dim(self):
        return self.basis.dim

    @property
    def is_mixed_order(self):
        return self.compatibility.is_mixed_order

    def element_for_family(self, family):
        return self.compatibility.element_for_family(family)

    def element_for_field(self, field):
        return self.compatibility.element_for_field(field)

    def field_element_types_for(self, fields):
        return self.compatibility.field_element_types_for(fields)


def sfem_fem_policy(
    element,
    vector_size=16,
    quadrature_order=None,
    integration_case="standard",
):
    compatible = element if isinstance(element, SfemCompatibleElement) else None
    cell_element_type = compatible.cell_element_type if compatible else str(element).upper()
    if (
        compatible is not None
        and compatible.is_mixed_order
        and str(integration_case) == "standard"
    ):
        integration_case = "isoparametric_mixed"
    if quadrature_order is None:
        quadrature_order = sfem_default_quadrature_order(
            cell_element_type,
            integration_case=integration_case,
        )
    quadrature_rule = sfem_element_quadrature_rule(cell_element_type, quadrature_order)
    specialization = SfemSoAElementSpecialization(quadrature_rule, vector_size)
    return SfemFEMPolicy(
        element,
        compatible.name.lower() if compatible else cell_element_type.lower(),
        cell_element_type,
        _sfem_basis_policy(quadrature_rule),
        quadrature_rule,
        specialization,
        SfemFieldFamilyCompatibilityPolicy.from_element(element),
        compatible,
    )


def _sfem_basis_policy(quadrature_rule):
    element_type = quadrature_rule.element_type
    degree = 2 if element_type in ("TRI6", "TET10", "HEX27") else 1
    return SfemElementBasisPolicy(
        element_type,
        quadrature_rule.dim,
        quadrature_rule.n_shape,
        "tensor_product" if quadrature_rule.is_tensor_product else "simplex",
        degree,
        _sfem_cell_name(element_type),
        quadrature_rule.is_tensor_product,
    )


def _sfem_cell_name(element_type):
    element_type = str(element_type).upper()
    if element_type in ("TRI3", "TRI6"):
        return "triangle"
    if element_type in ("TET4", "TET10"):
        return "tetrahedron"
    if element_type == "QUAD4":
        return "quadrilateral"
    if element_type in ("HEX8", "HEX27"):
        return "hexahedron"
    raise ValueError("unsupported element type '%s'" % element_type)


def sfem_normalize_integration_case(integration_case="standard"):
    integration_case = str(integration_case)
    aliases = {
        "": "",
        "isoparametric_energy": "energy",
        "curved_energy": "energy",
        "isoparametric_value_residual": "value_residual",
        "affine_value_residual": "value_residual",
        "isoparametric_value_linear_residual": "value_linear_residual",
        "affine_value_linear_residual": "value_linear_residual",
        "isoparametric_standard": "standard",
        "affine_standard": "standard",
    }
    return aliases.get(integration_case, integration_case)


def sfem_default_quadrature_order(element_type, integration_case="standard"):
    element_type = str(element_type).upper()
    integration_case = sfem_normalize_integration_case(integration_case)
    if integration_case == "affine_energy":
        integration_case = "standard"
    if integration_case == "affine_mixed":
        if element_type in ("TRI6", "TET10"):
            return 2
        if element_type == "HEX27":
            return 3
        integration_case = "standard"
    if integration_case == "isoparametric_mixed":
        if element_type in ("TRI6", "TET10", "HEX27"):
            return 4
    if integration_case == "value_linear_residual":
        if element_type in ("TRI3", "TET4", "QUAD4", "HEX8"):
            return 2
        if element_type in ("TRI6", "TET10"):
            return 4
        if element_type == "HEX27":
            return 3
    if integration_case == "value_residual":
        if element_type in ("TRI3", "TET4", "TRI6", "TET10", "QUAD4", "HEX8", "HEX27"):
            return 4
    if integration_case == "energy":
        if element_type in ("TRI6", "TET10", "HEX27"):
            return 4
    if element_type == "HEX27":
        return 3
    if element_type in ("QUAD4", "HEX8", "TRI6", "TET10"):
        return 2
    return 1


def sfem_element_quadrature_rule(element_type, order=None):
    element_type = str(element_type).upper()
    if order is None:
        order = sfem_default_quadrature_order(element_type)
    order = int(order)

    if element_type == "TRI3":
        if order == 1:
            weights = (0.5,)
            gradients = (-1.0, -1.0, 1.0, 0.0, 0.0, 1.0)
        else:
            _, weights = _sfem_triangle_quadrature_rule(order)
            gradients = (-1.0, -1.0, 1.0, 0.0, 0.0, 1.0) * len(weights)
        return SfemElementQuadratureRule(element_type, 2, 3, weights, gradients, order)

    if element_type == "TET4":
        tet4_gradients = (
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
        if order == 1:
            weights = (1.0 / 6.0,)
            gradients = tet4_gradients
        else:
            _, weights = _sfem_tetrahedron_quadrature_rule(order)
            gradients = tet4_gradients * len(weights)
        return SfemElementQuadratureRule(element_type, 3, 4, weights, gradients, order)

    if element_type == "TRI6":
        points, weights = _sfem_triangle_quadrature_rule(order)
        gradients = []
        for x, y in points:
            gradients.extend(_tri6_reference_gradients(x, y))
        return SfemElementQuadratureRule(element_type, 2, 6, weights, gradients, order)

    if element_type == "TET10":
        points, weights = _sfem_tetrahedron_quadrature_rule(order)
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
            (("displacement", "TRI6"), ("velocity", "TRI6"), ("pressure", "TRI3")),
        ),
        SfemCompatibleElement(
            "TET10_TET4",
            "TET10",
            (("displacement", "TET10"), ("velocity", "TET10"), ("pressure", "TET4")),
        ),
        SfemCompatibleElement(
            "HEX27_HEX8",
            "HEX27",
            (("displacement", "HEX27"), ("velocity", "HEX27"), ("pressure", "HEX8")),
        ),
    )


def sfem_taylor_hood_element_types_for_dim(dim):
    dim = int(dim)
    if dim == 2:
        return tuple(
            element
            for element in sfem_taylor_hood_element_types()
            if element.cell_element_type == "TRI6"
        )
    if dim == 3:
        return tuple(
            element
            for element in sfem_taylor_hood_element_types()
            if element.cell_element_type in ("TET10", "HEX27")
        )
    return ()


def sfem_detect_taylor_hood_element_types(fields):
    fields = tuple(fields)
    if not _fields_request_taylor_hood(fields):
        return ()
    dim = _fields_geometric_dimension(fields)
    return sfem_taylor_hood_element_types_for_dim(dim)


def _fields_request_taylor_hood(fields):
    has_high_order_vector = False
    has_pressure = False
    for field in fields:
        family = getattr(field, "family", "")
        space = getattr(field, "metadata", {}).get("space")
        if space is None:
            continue
        element = getattr(space, "element", None)
        if element is None or str(getattr(element, "family", "")) != "Lagrange":
            continue
        degree = int(getattr(element, "degree", -1))
        value_shape = tuple(getattr(element, "value_shape", ()))
        components = int(getattr(field, "components", 1))
        if family in ("displacement", "velocity") and degree == 2 and components > 1:
            if value_shape in (("geometric",), (components,)):
                has_high_order_vector = True
        elif family == "pressure" and degree == 1 and components == 1:
            if not value_shape:
                has_pressure = True
    return has_high_order_vector and has_pressure


def _fields_geometric_dimension(fields):
    dim = None
    for field in fields:
        metadata = getattr(field, "metadata", {})
        field_dim = metadata.get("dim")
        if field_dim is None and int(getattr(field, "components", 1)) > 1:
            field_dim = int(field.components)
        if field_dim is None:
            continue
        field_dim = int(field_dim)
        if dim is None:
            dim = field_dim
        elif dim != field_dim:
            raise ValueError("mixed finite-element fields have inconsistent dimensions")
    if dim is None:
        raise ValueError("mixed finite-element fields do not define a geometric dimension")
    return dim


def sfem_detect_compatible_element_types(fields):
    return sfem_detect_taylor_hood_element_types(fields)


def sfem_soa_element_specializations(
    element_types=None,
    vector_size=16,
    quadrature_order=None,
    integration_case="standard",
):
    element_types = sfem_supported_element_types() if element_types is None else tuple(element_types)
    return tuple(
        sfem_soa_element_specialization(
            element_type,
            vector_size,
            quadrature_order,
            integration_case=integration_case,
        )
        for element_type in element_types
    )


def sfem_soa_element_specialization(
    element_type,
    vector_size=16,
    quadrature_order=None,
    integration_case="standard",
):
    if quadrature_order is None:
        quadrature_order = sfem_default_quadrature_order(
            element_type,
            integration_case=integration_case,
        )
    return SfemSoAElementSpecialization(
        sfem_element_quadrature_rule(element_type, quadrature_order),
        vector_size,
    )


def sfem_simplex_grad_ref_name(prefix, component):
    names = ("x", "y", "z")
    component = int(component)
    if component < 0 or component >= len(names):
        raise ValueError("unsupported simplex gradient component %d" % component)
    return "%s_%s" % (prefix, names[component])


def sfem_split_reference_gradient_data(prefix, gradients, n_qp, n_shape, dim):
    components = []
    for d in range(dim):
        values = []
        for q in range(n_qp):
            for shape in range(n_shape):
                values.append(gradients[(q * n_shape + shape) * dim + d])
        components.append(SfemReferenceData(sfem_simplex_grad_ref_name(prefix, d), values))
    return tuple(components)


def sfem_reference_data(rule):
    if rule.is_tensor_product:
        return (
            SfemReferenceData("shape_1d", rule.tensor_product_shape_values_1d),
            SfemReferenceData("grad_1d", rule.tensor_product_shape_gradients_1d),
            SfemReferenceData("q_weight_1d", rule.tensor_product_weights_1d),
        )
    shape, gradients = sfem_shape_data_for_element_at_cell_rule(rule.element_type, rule)
    return (
        (SfemReferenceData("shape", shape),)
        + sfem_split_reference_gradient_data(
            "grad_ref",
            gradients,
            rule.n_qp,
            rule.n_shape,
            rule.dim,
        )
        + (SfemReferenceData("q_weight", rule.weights),)
    )


def sfem_mesh_reference_data(rule):
    return sfem_reference_data(rule)


def sfem_tensor_product_field_reference_data(element_type, cell_rule, prefix):
    element_type = str(element_type).upper()
    if not cell_rule.is_tensor_product:
        raise ValueError("cell rule must be tensor-product")
    points, _ = _sfem_unit_interval_gauss_rule(cell_rule.order)
    if element_type in ("QUAD4", "HEX8"):
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q1_1d_shapes(points)
    elif element_type == "HEX27":
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q2_1d_shapes(points)
    else:
        raise ValueError("unsupported tensor-product field element '%s'" % element_type)
    return (
        SfemReferenceData("%s_shape_1d" % prefix, shape_values_1d),
        SfemReferenceData("%s_grad_1d" % prefix, shape_gradients_1d),
    )


def sfem_simplex_field_reference_data(element_type, cell_rule, prefix):
    element_type = str(element_type).upper()
    if cell_rule.is_tensor_product:
        raise ValueError("cell rule must be simplex")
    shape, gradients = sfem_shape_data_for_element_at_cell_rule(element_type, cell_rule)
    n_shape = len(shape) // cell_rule.n_qp
    return (
        (SfemReferenceData("%s_shape" % prefix, shape),)
        + sfem_split_reference_gradient_data(
            "%s_grad_ref" % prefix,
            gradients,
            cell_rule.n_qp,
            n_shape,
            cell_rule.dim,
        )
    )


def sfem_field_n_shape(element_type, quadrature_order=None):
    return sfem_element_quadrature_rule(element_type, quadrature_order).n_shape


def sfem_shape_data_for_element_at_cell_rule(element_type, cell_rule):
    element_type = str(element_type).upper()
    points = sfem_cell_rule_points(cell_rule)
    if element_type == "TRI3":
        shape = []
        gradients = []
        for x, y in points:
            shape.extend((1.0 - x - y, x, y))
            gradients.extend((-1.0, -1.0, 1.0, 0.0, 0.0, 1.0))
        return tuple(shape), tuple(gradients)
    if element_type == "TRI6":
        shape = []
        gradients = []
        for x, y in points:
            l0 = 1.0 - x - y
            shape.extend(
                (
                    l0 * (2.0 * l0 - 1.0),
                    x * (2.0 * x - 1.0),
                    y * (2.0 * y - 1.0),
                    4.0 * x * l0,
                    4.0 * x * y,
                    4.0 * y * l0,
                )
            )
            gradients.extend(_tri6_reference_gradients(x, y))
        return tuple(shape), tuple(gradients)
    if element_type == "TET4":
        shape = []
        gradients = []
        for x, y, z in points:
            shape.extend((1.0 - x - y - z, x, y, z))
            gradients.extend(
                (
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
            )
        return tuple(shape), tuple(gradients)
    if element_type == "TET10":
        shape = []
        gradients = []
        for x, y, z in points:
            l0 = 1.0 - x - y - z
            shape.extend(
                (
                    l0 * (2.0 * l0 - 1.0),
                    x * (2.0 * x - 1.0),
                    y * (2.0 * y - 1.0),
                    z * (2.0 * z - 1.0),
                    4.0 * x * l0,
                    4.0 * x * y,
                    4.0 * y * l0,
                    4.0 * z * l0,
                    4.0 * x * z,
                    4.0 * y * z,
                )
            )
            gradients.extend(_tet10_reference_gradients(x, y, z))
        return tuple(shape), tuple(gradients)
    if element_type in ("HEX8", "HEX27"):
        order = 1 if element_type == "HEX8" else 2
        return _sfem_hex_lagrange_shape_gradients(points, order)
    raise ValueError("unsupported residual field element '%s'" % element_type)


def sfem_cell_rule_points(rule):
    if rule.element_type == "TRI3":
        if rule.order == 1:
            return ((1.0 / 3.0, 1.0 / 3.0),)
        points, _ = _sfem_triangle_quadrature_rule(rule.order)
        return points
    if rule.element_type == "TRI6":
        points, _ = _sfem_triangle_quadrature_rule(rule.order)
        return points
    if rule.element_type == "TET4":
        if rule.order == 1:
            return ((0.25, 0.25, 0.25),)
        points, _ = _sfem_tetrahedron_quadrature_rule(rule.order)
        return points
    if rule.element_type == "TET10":
        points, _ = _sfem_tetrahedron_quadrature_rule(rule.order)
        return points
    if rule.element_type in ("HEX8", "HEX27"):
        points, _ = _sfem_unit_interval_gauss_rule(rule.order)
        return tuple((x, y, z) for z in points for y in points for x in points)
    raise ValueError("unsupported cell rule '%s'" % rule.element_type)


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


def _sfem_triangle_quadrature_rule(order):
    if order == 2:
        return (
            (
                (1.0 / 6.0, 1.0 / 6.0),
                (2.0 / 3.0, 1.0 / 6.0),
                (1.0 / 6.0, 2.0 / 3.0),
            ),
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        )
    if order == 3:
        a = 1.0 / 3.0
        b = 0.2
        c = 0.6
        return (
            ((a, a), (b, b), (c, b), (b, c)),
            (-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0),
        )
    if order == 4:
        a = 0.4459484909159649
        b = 0.1081030181680702
        c = 0.09157621350977074
        d = 0.8168475729804585
        w0 = 0.1116907948390057
        w1 = 0.0549758718276610
        return (
            ((a, a), (b, a), (a, b), (c, c), (d, c), (c, d)),
            (w0, w0, w0, w1, w1, w1),
        )
    raise ValueError("triangle quadrature currently supports orders 2, 3, and 4")


def _sfem_tetrahedron_quadrature_rule(order):
    if order == 2:
        a = 0.5854101966249685
        b = 0.1381966011250105
        return (
            ((b, b, b), (a, b, b), (b, a, b), (b, b, a)),
            (1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0),
        )
    if order == 4:
        a = 0.7857142857142857
        b = 0.07142857142857142
        c = 0.3994035761667992
        d = 0.1005964238332008
        w0 = -0.013155555555555556
        w1 = 0.007622222222222222
        w2 = 0.024888888888888887
        return (
            (
                (0.25, 0.25, 0.25),
                (b, b, b),
                (a, b, b),
                (b, a, b),
                (b, b, a),
                (c, c, d),
                (c, d, c),
                (d, c, c),
                (c, d, d),
                (d, c, d),
                (d, d, c),
            ),
            (w0, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2),
        )
    raise ValueError("tetrahedron quadrature currently supports orders 2 and 4")


def _sfem_unit_interval_gauss_rule(order):
    if order == 1:
        return (0.5,), (1.0,)
    if order == 2:
        offset = 0.5 / math.sqrt(3.0)
        return (0.5 - offset, 0.5 + offset), (0.5, 0.5)
    if order == 3:
        offset = 0.5 * math.sqrt(3.0 / 5.0)
        return (0.5 - offset, 0.5, 0.5 + offset), (5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0)
    if order == 4:
        outer = math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
        inner = math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
        w_outer = (18.0 - math.sqrt(30.0)) / 72.0
        w_inner = (18.0 + math.sqrt(30.0)) / 72.0
        return (
            0.5 * (1.0 - outer),
            0.5 * (1.0 - inner),
            0.5 * (1.0 + inner),
            0.5 * (1.0 + outer),
        ), (w_outer, w_inner, w_inner, w_outer)
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


def _sfem_lagrange_1d_at(x, order):
    if order == 1:
        return (1.0 - x, x), (-1.0, 1.0)
    if order == 2:
        return (
            2.0 * x * x - 3.0 * x + 1.0,
            4.0 * x - 4.0 * x * x,
            2.0 * x * x - x,
        ), (
            4.0 * x - 3.0,
            4.0 - 8.0 * x,
            4.0 * x - 1.0,
        )
    raise ValueError("unsupported 1D Lagrange order %d" % order)


def _sfem_hex_lagrange_shape_gradients(points, order):
    n = order + 1
    shape = []
    gradients = []
    for x, y, z in points:
        values_x, grads_x = _sfem_lagrange_1d_at(x, order)
        values_y, grads_y = _sfem_lagrange_1d_at(y, order)
        values_z, grads_z = _sfem_lagrange_1d_at(z, order)
        shape_q = [None] * (n * n * n)
        gradients_q = [None] * (n * n * n)
        for sz in range(n):
            for sy in range(n):
                for sx in range(n):
                    idx = sfem_tensor_hex_shape_index(n, sx, sy, sz)
                    vx = values_x[sx]
                    vy = values_y[sy]
                    vz = values_z[sz]
                    shape_q[idx] = vx * vy * vz
                    gradients_q[idx] = (
                        grads_x[sx] * vy * vz,
                        vx * grads_y[sy] * vz,
                        vx * vy * grads_z[sz],
                    )
        shape.extend(shape_q)
        for item in gradients_q:
            gradients.extend(item)
    return tuple(shape), tuple(gradients)


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
                            shape = sfem_tensor_hex_shape_index(
                                n_shape_1d,
                                sx,
                                sy,
                                sz,
                            )
                            qp_gradients[shape] = (dx, dy, dz)
                for gradient in qp_gradients:
                    gradients.extend(gradient)
    return tuple(gradients), tuple(weights)


def sfem_tensor_hex_shape_index(n_shape_1d, sx, sy, sz):
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
