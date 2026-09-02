"""How element data is ordered and indexed for a kernel.

Which shape function maps to which stream, where a field's components sit in a
packed lane buffer, how tensor-product elements reorder their nodes into
Cartesian order, and what offset a given field/component/shape triplet lands at.
These are layout decisions: they determine the memory access pattern of every
generated loop and therefore whether the loads are unit stride.

They are pure index arithmetic -- no expression is inspected and no text is
produced -- and they were living in ``emitters/residual_codegen.py``, mixed in
with the code that prints the loops they describe. Deciding a layout is
planning; spelling the resulting subscript is emission.

The mixed-order helpers here carry the Taylor-Hood cases, where the cell
element and a field's element differ and the per-field shape counts diverge.
"""

from codegen.framework.fem.reference import (
    sfem_field_n_shape,
    sfem_is_tensor_product_hex_element,
    sfem_tensor_product_hex_uses_cartesian_ordering,
    sfem_tensor_product_quad_uses_cartesian_ordering,
)
from codegen.framework.fem.tensor_product import tensor_product_cartesian_shape_order


def _identity_order(order):
    return tuple(order) == tuple(range(len(order)))


def _linear_index_offset(values):
    values = tuple(values)
    if not values:
        return 0
    offset = values[0]
    if values == tuple(offset + i for i in range(len(values))):
        return offset
    return None


def _tensor_product_coordinate_shape_order(dim, n_shape, element_type):
    if sfem_tensor_product_hex_uses_cartesian_ordering(element_type) or sfem_tensor_product_quad_uses_cartesian_ordering(element_type):
        return tuple(range(n_shape))
    return tensor_product_cartesian_shape_order(dim, n_shape)


def _single_field_shape_order(n_shape, n_fields, field_stream_order):
    return tuple(field_stream_order[shape * n_fields] // n_fields for shape in range(n_shape))


def _stream_to_tensor_order(field_stream_order):
    ordered = [0] * len(field_stream_order)
    for tensor_stream, mesh_stream in enumerate(field_stream_order):
        ordered[mesh_stream] = tensor_stream
    return tuple(ordered)


def _field_element_type(field_or_name, cell_rule, field_element_types):
    field_name = _residual_parent_field_name(field_or_name)
    return str(field_element_types.get(field_name, cell_rule.element_type)).upper()


def _field_n_shape(field, cell_rule, field_element_types):
    return _field_n_shape_by_name(
        _residual_parent_field_name(field),
        cell_rule,
        field_element_types,
    )


def _field_n_shape_by_name(field_name, cell_rule, field_element_types):
    element_type = _field_element_type(field_name, cell_rule, field_element_types)
    return sfem_field_n_shape(
        element_type,
        cell_rule.order
        if element_type in ("QUAD4", "PROTEUS_QUAD4") or sfem_is_tensor_product_hex_element(element_type)
        else None,
    )


def _is_tensor_product_family(rule, basis_family=None):
    if basis_family is None:
        raise ValueError("basis family must be provided by the emission plan")
    return str(basis_family) == "tensor_product"


def _residual_parent_field_name(field_or_name):
    return str(getattr(field_or_name, "field_name", field_or_name))


def _mixed_field_shape_orders(
    layout,
    cell_rule,
    field_element_types,
    basis_family,
):
    if not _is_tensor_product_family(cell_rule, basis_family):
        return tuple(tuple(range(layout.n_shape(field_index))) for field_index in range(len(layout.fields)))

    field_element_types = {} if field_element_types is None else field_element_types
    orders = []
    for field_index, field in enumerate(layout.fields):
        n_shape = layout.n_shape(field_index)
        element_type = _field_element_type(
            _residual_parent_field_name(field),
            cell_rule,
            field_element_types,
        )
        orders.append(
            tuple(range(n_shape))
            if sfem_tensor_product_hex_uses_cartesian_ordering(element_type)
            else tensor_product_cartesian_shape_order(cell_rule.dim, n_shape)
        )
    return tuple(orders)


def _mixed_tensor_product_field_stream_order(
    layout,
    cell_rule,
    field_element_types,
    basis_family,
):
    if not _is_tensor_product_family(cell_rule, basis_family):
        return tuple(range(layout.total_streams))

    field_element_types = {} if field_element_types is None else field_element_types
    order = []
    for field_index, field in enumerate(layout.fields):
        n_shape = layout.n_shape(field_index)
        element_type = _field_element_type(
            _residual_parent_field_name(field),
            cell_rule,
            field_element_types,
        )
        shape_order = (
            tuple(range(n_shape))
            if sfem_tensor_product_hex_uses_cartesian_ordering(element_type)
            else tensor_product_cartesian_shape_order(cell_rule.dim, n_shape)
        )
        order.extend(layout.stream_index(field_index, shape) for shape in shape_order)
    return tuple(order)


def _mixed_stream_shape_offsets(layout):
    offsets = []
    for field_index, _ in enumerate(layout.fields):
        offsets.extend(range(layout.n_shape(field_index)))
    return tuple(offsets)


def _mixed_triplet_stream_indices(layout, group_names):
    selected = set(group_names)
    indices = []
    for field_index, field in enumerate(layout.fields):
        if _residual_parent_field_name(field) not in selected:
            continue
        indices.extend(
            layout.stream_index(field_index, local_shape)
            for local_shape in range(layout.n_shape(field_index))
        )
    return tuple(indices)


def _compatible_matrix_stream_indices(field_indices, n_shape):
    streams = []
    for field_index in field_indices:
        streams.extend(field_index * n_shape + shape for shape in range(n_shape))
    return tuple(streams)


def _compatible_stream_component_offsets(n_fields, n_shape):
    offsets = []
    for field_index in range(n_fields):
        offsets.extend(field_index for _ in range(n_shape))
    return tuple(offsets)


def _compatible_stream_shape_offsets(n_fields, n_shape):
    offsets = []
    for _ in range(n_fields):
        offsets.extend(range(n_shape))
    return tuple(offsets)
