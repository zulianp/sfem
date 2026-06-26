try:
    from .tensor_product import (
        streams_in_shape_order,
        tensor_product_cartesian_shape_order,
        tensor_product_geometry_jacobian_plan_from_sizes,
    )
except ImportError:
    from tensor_product import (
        streams_in_shape_order,
        tensor_product_cartesian_shape_order,
        tensor_product_geometry_jacobian_plan_from_sizes,
    )


def isoparametric_adjugate_lines(
    dim,
    indent,
    index,
    adjugate_target,
    determinant_target,
):
    adj = lambda component: adjugate_target(component, index)
    det = determinant_target(index)
    if dim == 1:
        return [
            "%s%s = 1;" % (indent, adj(0)),
            "%s%s = J00;" % (indent, det),
        ]
    if dim == 2:
        return [
            "%s%s = J11;" % (indent, adj(0)),
            "%s%s = -J01;" % (indent, adj(1)),
            "%s%s = -J10;" % (indent, adj(2)),
            "%s%s = J00;" % (indent, adj(3)),
            "%s%s = J00 * J11 - J01 * J10;" % (indent, det),
        ]
    if dim == 3:
        return [
            "%s%s = J11 * J22 - J12 * J21;" % (indent, adj(0)),
            "%s%s = J02 * J21 - J01 * J22;" % (indent, adj(1)),
            "%s%s = J01 * J12 - J02 * J11;" % (indent, adj(2)),
            "%s%s = J12 * J20 - J10 * J22;" % (indent, adj(3)),
            "%s%s = J00 * J22 - J02 * J20;" % (indent, adj(4)),
            "%s%s = J02 * J10 - J00 * J12;" % (indent, adj(5)),
            "%s%s = J10 * J21 - J11 * J20;" % (indent, adj(6)),
            "%s%s = J01 * J20 - J00 * J21;" % (indent, adj(7)),
            "%s%s = J00 * J11 - J01 * J10;" % (indent, adj(8)),
            (
                "%s%s = J00 * (J11 * J22 - J12 * J21)"
                " - J01 * (J10 * J22 - J12 * J20)"
                " + J02 * (J10 * J21 - J11 * J20);"
            )
            % (indent, det),
        ]
    raise ValueError("isoparametric geometry supports dimensions 1, 2, and 3")


def tensor_product_isoparametric_geometry_lines(
    *,
    dim,
    n_shape,
    n_qp=None,
    coordinate_streams,
    evaluator_lines,
    gradient_name="coordinate_grad_ref",
    stream_array_name="block_coordinate_streams",
    indent="        ",
    adjugate_target,
    determinant_target,
):
    if dim not in (2, 3):
        raise ValueError("tensor-product geometry supports dimensions 2 and 3")
    if len(coordinate_streams) != dim * n_shape:
        raise ValueError("coordinate stream count must be dim * n_shape")
    n_shape_1d = round(n_shape ** (1.0 / dim))
    if n_shape_1d ** dim != n_shape:
        raise ValueError("tensor-product geometry n_shape must be a perfect tensor power")
    n_qp = n_shape if n_qp is None else int(n_qp)
    n_qp_1d = round(n_qp ** (1.0 / dim))
    if n_qp_1d ** dim != n_qp:
        raise ValueError("tensor-product geometry n_qp must be a perfect tensor power")
    sum_factorization = tensor_product_geometry_jacobian_plan_from_sizes(
        dim,
        n_shape,
        n_qp,
        n_shape_1d,
        n_qp_1d,
    )

    lines = [
        "%sconst scalar_t *const %s[DIM * N_SHAPE] = {%s};"
        % (indent, stream_array_name, ", ".join(coordinate_streams)),
        "%sscalar_t %s[DIM * N_QP * DIM * VECTOR_SIZE];"
        % (indent, gradient_name),
    ]
    lines.extend(
        evaluator_lines(
            stream_array_name,
            gradient_name,
            indent,
        )
    )
    if not sum_factorization.evaluates_geometry_jacobian:
        raise ValueError("tensor-product geometry requires a Jacobian sum-factorization plan")
    lines.extend(
        [
            "",
            "%sfor (int q = 0; q < N_QP; ++q) {" % indent,
            "#pragma omp simd",
            "%s    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {" % indent,
        ]
    )
    body_indent = indent + "        "
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "%sconst scalar_t J%d%d = %s[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + lane];"
                % (body_indent, row, col, gradient_name, row, col)
            )
    lines.extend(
        isoparametric_adjugate_lines(
            dim,
            body_indent,
            "q * VECTOR_SIZE + lane",
            adjugate_target,
            determinant_target,
        )
    )
    lines.extend(["%s    }" % indent, "%s}" % indent])
    return lines
