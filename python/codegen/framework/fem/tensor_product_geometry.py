from codegen.framework.backends.targets import OpenMPTarget
from codegen.framework.fem.tensor_product import (
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_geometry_jacobian_plan_from_sizes,
)


def _default_target():
    return OpenMPTarget()


def _target_simd_lines(simd_lines=None):
    if simd_lines is not None:
        return tuple(simd_lines)
    pragma = _default_target().vectorize_pragma()
    return () if pragma is None else (pragma,)


def _target_work_item_index(work_item_index=None):
    return _default_target().work_item_index() if work_item_index is None else str(work_item_index)


def _work_item_loop_lines(indent, *, work_item_index=None, simd_lines=None, single_work_item=False):
    if single_work_item:
        return ("%s{" % indent,)
    work_item = _target_work_item_index(work_item_index)
    return tuple("%s%s" % (indent, line) for line in _target_simd_lines(simd_lines)) + (
        "%sfor (int %s = 0; %s < nelems; ++%s) {"
        % (indent, work_item, work_item, work_item),
    )


def _restrict_define_line(restrict_definition):
    restrict_definition = str(restrict_definition)
    if restrict_definition:
        return "#define SFEM_RESTRICT %s" % restrict_definition
    return "#define SFEM_RESTRICT"


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


def sfem_geometry_kernels_header_source(
    *,
    inline_qualifier=None,
    inline_definition="inline",
    define_sfem_inline=True,
    restrict_definition="",
    work_item_index=None,
    simd_lines=None,
    single_work_item=False,
    header_guard_suffix="HPP",
):
    inline_qualifier = _default_target().inline_qualifier() if inline_qualifier is None else inline_qualifier
    work_item = _target_work_item_index(work_item_index)
    inline_block = (
        list(_default_target().inline_definition_lines(inline_definition)) + [""]
        if define_sfem_inline
        else []
    )
    work_loop = _work_item_loop_lines(
        "            ",
        work_item_index=work_item,
        simd_lines=simd_lines,
        single_work_item=single_work_item,
    )
    return "\n".join(
        [
            "#ifndef SFEM_CODEGEN_GEOMETRY_KERNELS_%s" % header_guard_suffix,
            "#define SFEM_CODEGEN_GEOMETRY_KERNELS_%s" % header_guard_suffix,
            "",
            "#include <stddef.h>",
            "",
            *inline_block,
            "#ifndef SFEM_RESTRICT",
            _restrict_define_line(restrict_definition),
            "#endif",
            "",
            "namespace sfem {",
            "namespace codegen {",
            "",
            "template <typename scalar_t, int DIM, int N_QP, int VECTOR_SIZE>",
            "struct GeometryJacobianAdjugateDeterminant;",
            "",
            "template <typename scalar_t>",
            "static %s void geometry_jacobian_adjugate_and_determinant_2(" % inline_qualifier,
            "        const scalar_t J00,",
            "        const scalar_t J01,",
            "        const scalar_t J10,",
            "        const scalar_t J11,",
            "        scalar_t *const *const SFEM_RESTRICT adjugate,",
            "        scalar_t *const SFEM_RESTRICT determinant,",
            "        const ptrdiff_t offset) {",
            "    adjugate[0][offset] = J11;",
            "    adjugate[1][offset] = -J01;",
            "    adjugate[2][offset] = -J10;",
            "    adjugate[3][offset] = J00;",
            "    determinant[offset] = J00 * J11 - J01 * J10;",
            "}",
            "",
            "template <typename scalar_t>",
            "static %s void geometry_jacobian_adjugate_and_determinant_3(" % inline_qualifier,
            "        const scalar_t J00,",
            "        const scalar_t J01,",
            "        const scalar_t J02,",
            "        const scalar_t J10,",
            "        const scalar_t J11,",
            "        const scalar_t J12,",
            "        const scalar_t J20,",
            "        const scalar_t J21,",
            "        const scalar_t J22,",
            "        scalar_t *const *const SFEM_RESTRICT adjugate,",
            "        scalar_t *const SFEM_RESTRICT determinant,",
            "        const ptrdiff_t offset) {",
            "    adjugate[0][offset] = J11 * J22 - J12 * J21;",
            "    adjugate[1][offset] = J02 * J21 - J01 * J22;",
            "    adjugate[2][offset] = J01 * J12 - J02 * J11;",
            "    adjugate[3][offset] = J12 * J20 - J10 * J22;",
            "    adjugate[4][offset] = J00 * J22 - J02 * J20;",
            "    adjugate[5][offset] = J02 * J10 - J00 * J12;",
            "    adjugate[6][offset] = J10 * J21 - J11 * J20;",
            "    adjugate[7][offset] = J01 * J20 - J00 * J21;",
            "    adjugate[8][offset] = J00 * J11 - J01 * J10;",
            "    determinant[offset] = J00 * (J11 * J22 - J12 * J21)",
            "            - J01 * (J10 * J22 - J12 * J20)",
            "            + J02 * (J10 * J21 - J11 * J20);",
            "}",
            "",
            "template <typename scalar_t, int N_QP, int VECTOR_SIZE>",
            "struct GeometryJacobianAdjugateDeterminant<scalar_t, 2, N_QP, VECTOR_SIZE> {",
            "    static %s void eval(" % inline_qualifier,
            "            const int nelems,",
            "            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "            scalar_t *const *const SFEM_RESTRICT adjugate,",
            "            scalar_t *const SFEM_RESTRICT determinant) {",
            "        for (int q = 0; q < N_QP; ++q) {",
            *work_loop,
            "                const ptrdiff_t offset = q * VECTOR_SIZE + %s;" % work_item,
            "                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + %s];" % work_item,
            "                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(",
            "                        J00, J01, J10, J11, adjugate, determinant, offset);",
            "            }",
            "        }",
            "    }",
            "};",
            "",
            "template <typename scalar_t, int N_QP, int VECTOR_SIZE>",
            "struct GeometryJacobianAdjugateDeterminant<scalar_t, 3, N_QP, VECTOR_SIZE> {",
            "    static %s void eval(" % inline_qualifier,
            "            const int nelems,",
            "            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "            scalar_t *const *const SFEM_RESTRICT adjugate,",
            "            scalar_t *const SFEM_RESTRICT determinant) {",
            "        for (int q = 0; q < N_QP; ++q) {",
            *work_loop,
            "                const ptrdiff_t offset = q * VECTOR_SIZE + %s;" % work_item,
            "                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + %s];" % work_item,
            "                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + %s];" % work_item,
            "                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(",
            "                        J00, J01, J02, J10, J11, J12, J20, J21, J22,",
            "                        adjugate, determinant, offset);",
            "            }",
            "        }",
            "    }",
            "};",
            "",
            "template <typename scalar_t, int DIM, int N_QP, int VECTOR_SIZE>",
            "static %s void geometry_jacobian_adjugate_and_determinant(" % inline_qualifier,
            "        const int nelems,",
            "        const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,",
            "        scalar_t *const *const SFEM_RESTRICT adjugate,",
            "        scalar_t *const SFEM_RESTRICT determinant) {",
            "    GeometryJacobianAdjugateDeterminant<scalar_t, DIM, N_QP, VECTOR_SIZE>::eval(",
            "            nelems, coordinate_grad_ref, adjugate, determinant);",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
            "#endif",
            "",
        ]
    )


def isoparametric_adjugate_stream_array_lines(
    *,
    dim,
    indent,
    stream_array_name,
    adjugate_streams,
):
    return [
        "%sscalar_t *%s[DIM * DIM] = {%s};"
        % (indent, stream_array_name, ", ".join(adjugate_streams))
    ]


def isoparametric_adjugate_call_lines(
    *,
    dim,
    indent,
    index,
    stream_array_name,
    determinant_stream,
):
    if dim == 2:
        return [
            "%sgeometry_jacobian_adjugate_and_determinant_2<scalar_t>(" % indent,
            "%s        J00, J01, J10, J11, %s, %s, %s);"
            % (indent, stream_array_name, determinant_stream, index),
        ]
    if dim == 3:
        return [
            "%sgeometry_jacobian_adjugate_and_determinant_3<scalar_t>(" % indent,
            "%s        J00, J01, J02, J10, J11, J12, J20, J21, J22,"
            % indent,
            "%s        %s, %s, %s);"
            % (indent, stream_array_name, determinant_stream, index),
        ]
    raise ValueError("isoparametric geometry supports dimensions 2 and 3")


def coordinate_stream_array_lines(
    coordinate_streams,
    *,
    stream_array_name="block_coordinate_streams",
    indent="        ",
):
    if isinstance(coordinate_streams, str):
        return [
            "%sconst scalar_t *%s[DIM * N_SHAPE];" % (indent, stream_array_name),
            "%sfor (int stream = 0; stream < DIM * N_SHAPE; ++stream) {" % indent,
            "%s    %s[stream] = %s[stream];"
            % (indent, stream_array_name, coordinate_streams),
            "%s}" % indent,
        ]

    return [
        "%sconst scalar_t *const %s[DIM * N_SHAPE] = {%s};"
        % (indent, stream_array_name, ", ".join(coordinate_streams))
    ]


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
    adjugate_streams=None,
    determinant_stream=None,
):
    if dim not in (2, 3):
        raise ValueError("tensor-product geometry supports dimensions 2 and 3")
    if not isinstance(coordinate_streams, str) and len(coordinate_streams) != dim * n_shape:
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

    lines = coordinate_stream_array_lines(
        coordinate_streams,
        stream_array_name=stream_array_name,
        indent=indent,
    )
    lines.extend([
        "%sscalar_t %s[DIM * N_QP * DIM * VECTOR_SIZE];"
        % (indent, gradient_name),
    ])
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
        tensor_product_adjugate_determinant_lines(
            dim=dim,
            gradient_name=gradient_name,
            indent=indent,
            adjugate_target=adjugate_target,
            determinant_target=determinant_target,
            adjugate_streams=adjugate_streams,
            determinant_stream=determinant_stream,
            include_lane_loop=True,
        )
    )
    return lines


def tensor_product_ordered_streams(streams, n_components, dim, n_shape, shape_order=None):
    shape_order = (
        tensor_product_cartesian_shape_order(dim, n_shape)
        if shape_order is None
        else tuple(shape_order)
    )
    return streams_in_shape_order(tuple(streams), n_components, shape_order)


def tensor_product_ordered_coordinate_streams(
    dim,
    n_shape,
    coordinate_streams,
    wrapper=None,
    shape_order=None,
):
    wrapper = (lambda stream: stream) if wrapper is None else wrapper
    return tuple(
        wrapper(stream)
        for stream in tensor_product_ordered_streams(
            coordinate_streams,
            dim,
            dim,
            n_shape,
            shape_order=shape_order,
        )
    )


def tensor_product_evaluated_isoparametric_geometry_lines(
    *,
    dim,
    n_shape,
    n_qp,
    local_prefix,
    coordinate_streams,
    indent="        ",
    gradient_name="coordinate_grad_ref",
    stream_array_name="block_coordinate_streams",
    shape_name="shape_1d",
    grad_name="grad_1d",
    adjugate_target,
    determinant_target,
    adjugate_streams=None,
    determinant_stream=None,
):
    def evaluator_lines(streams, gradient, evaluator_indent):
        return [
            "%sscalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];"
            % evaluator_indent,
            "%stensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>("
            % evaluator_indent,
            "%s        nelems, %s, %s, %s,"
            % (evaluator_indent, shape_name, grad_name, streams),
            "%s        coordinate_value, %s);" % (evaluator_indent, gradient),
        ]

    return tensor_product_isoparametric_geometry_lines(
        dim=dim,
        n_shape=n_shape,
        n_qp=n_qp,
        coordinate_streams=coordinate_streams,
        evaluator_lines=evaluator_lines,
        gradient_name=gradient_name,
        stream_array_name=stream_array_name,
        indent=indent,
        adjugate_target=adjugate_target,
        determinant_target=determinant_target,
        adjugate_streams=adjugate_streams,
        determinant_stream=determinant_stream,
    )


def tensor_product_coordinate_gradient_lines(
    *,
    dim,
    local_prefix,
    coordinate_streams,
    indent="        ",
    gradient_name="coordinate_grad_ref",
    stream_array_name="block_coordinate_streams",
    shape_name="shape_1d",
    grad_name="grad_1d",
):
    lines = coordinate_stream_array_lines(
        coordinate_streams,
        stream_array_name=stream_array_name,
        indent=indent,
    )
    lines.extend([
        "%sscalar_t %s[DIM * N_QP * DIM * VECTOR_SIZE];"
        % (indent, gradient_name),
    ])
    for component in range(dim):
        lines.extend(
            [
                "%stensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, %d>("
                % (indent, dim),
                "%s        nelems, %s, %s, %s, %d,"
                % (indent, shape_name, grad_name, stream_array_name, component),
                "%s        %s + %d * N_QP * DIM * VECTOR_SIZE);"
                % (indent, gradient_name, component),
            ]
        )
    return lines


def tensor_product_current_q_isoparametric_geometry_lines(
    *,
    dim,
    gradient_name="coordinate_grad_ref",
    indent="        ",
    adjugate_target,
    determinant_target,
    output_index=None,
    work_item_index=None,
    simd_lines=None,
    single_work_item=False,
):
    work_item = _target_work_item_index(work_item_index)
    output_index = work_item if output_index is None else output_index
    lines = list(
        _work_item_loop_lines(
            indent,
            work_item_index=work_item,
            simd_lines=simd_lines,
            single_work_item=single_work_item,
        )
    )
    body_indent = indent + "    "
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "%sconst scalar_t J%d%d = %s[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + %s];"
                % (body_indent, row, col, gradient_name, row, col, work_item)
            )
    lines.extend(
        isoparametric_adjugate_lines(
            dim,
            body_indent,
            output_index,
            adjugate_target,
            determinant_target,
        )
    )
    lines.append("%s}" % indent)
    return lines


def tensor_product_gradient_isoparametric_geometry_lines(
    *,
    dim,
    n_shape,
    n_qp,
    local_prefix,
    coordinate_streams,
    indent="        ",
    gradient_name="coordinate_grad_ref",
    stream_array_name="block_coordinate_streams",
    shape_name="shape_1d",
    grad_name="grad_1d",
    adjugate_target,
    determinant_target,
    adjugate_streams=None,
    determinant_stream=None,
):
    n_shape_1d = round(n_shape ** (1.0 / dim))
    if n_shape_1d ** dim != n_shape:
        raise ValueError("tensor-product geometry n_shape must be a perfect tensor power")
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
    if not sum_factorization.evaluates_geometry_jacobian:
        raise ValueError("tensor-product geometry requires a Jacobian sum-factorization plan")

    lines = tensor_product_coordinate_gradient_lines(
        dim=dim,
        local_prefix=local_prefix,
        coordinate_streams=coordinate_streams,
        indent=indent,
        gradient_name=gradient_name,
        stream_array_name=stream_array_name,
        shape_name=shape_name,
        grad_name=grad_name,
    )
    lines.extend(
        tensor_product_adjugate_determinant_lines(
            dim=dim,
            gradient_name=gradient_name,
            indent=indent,
            adjugate_target=adjugate_target,
            determinant_target=determinant_target,
            adjugate_streams=adjugate_streams,
            determinant_stream=determinant_stream,
            include_lane_loop=False,
        )
    )
    return lines


def tensor_product_adjugate_determinant_lines(
    *,
    dim,
    gradient_name,
    indent,
    adjugate_target,
    determinant_target,
    adjugate_streams=None,
    determinant_stream=None,
    include_lane_loop,
    work_item_index=None,
    simd_lines=None,
    single_work_item=False,
):
    if adjugate_streams is not None and determinant_stream is not None:
        return [
            "",
            "%sscalar_t *%s_adjugate_streams[DIM * DIM] = {%s};"
            % (indent, gradient_name, ", ".join(adjugate_streams)),
            "%sgeometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>("
            % indent,
            "%s        nelems, %s, %s_adjugate_streams, %s);"
            % (indent, gradient_name, gradient_name, determinant_stream),
        ]

    lines = ["", "%sfor (int q = 0; q < N_QP; ++q) {" % indent]
    work_item = _target_work_item_index(work_item_index)
    if include_lane_loop:
        lines.extend(
            _work_item_loop_lines(
                indent + "    ",
                work_item_index=work_item,
                simd_lines=simd_lines,
                single_work_item=single_work_item,
            )
        )
        body_indent = indent + "        "
        for row in range(dim):
            for col in range(dim):
                lines.append(
                    "%sconst scalar_t J%d%d = %s[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + %s];"
                    % (body_indent, row, col, gradient_name, row, col, work_item)
                )
        lines.extend(
            isoparametric_adjugate_lines(
                dim,
                body_indent,
                "q * VECTOR_SIZE + %s" % work_item,
                adjugate_target,
                determinant_target,
            )
        )
        lines.append("%s    }" % indent)
    else:
        lines.extend(
            tensor_product_current_q_isoparametric_geometry_lines(
                dim=dim,
                gradient_name=gradient_name,
                indent=indent + "    ",
                adjugate_target=adjugate_target,
                determinant_target=determinant_target,
                output_index="q * VECTOR_SIZE + %s" % work_item,
                work_item_index=work_item,
                simd_lines=simd_lines,
                single_work_item=single_work_item,
            )
        )
    lines.append("%s}" % indent)
    return lines
