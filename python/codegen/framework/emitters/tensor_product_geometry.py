from codegen.framework.emitters.cprinter import c_group, c_product, c_sum, kernel_status_macro_lines
from codegen.framework.emitters.tensor_product_kernels import (
    micro_kernel,
    micro_kernel_count,
    micro_kernel_template,
)
from codegen.framework.plans.conventions import restrict_prelude
from codegen.framework.targets import current_target
from codegen.framework.fem.tensor_product import (
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_geometry_jacobian_plan_from_sizes,
)


def _default_target():
    """The target this emission prints for -- bound by the backend, not chosen here."""
    return current_target()


#: Which evaluator a contiguous stream layout calls; the name the target then
#: renders, blocked or scalar, comes from `micro_kernel`.
_TENSOR_EVALUATE_KERNEL = {
    True: "tensor_evaluate_contiguous",
    False: "tensor_evaluate",
}


def _width_factors():
    """The work-item width, as template/product factors -- none where there is none."""
    width = _default_target().kernel_vector_width()
    return (width,) if width is not None else ()


def _width_parameters():
    """The same, spelled as template parameters."""
    return tuple("int %s" % name for name in _width_factors())


def _width_suffix():
    """` * VS` where a work item is a lane of a block, and nothing where it is one element."""
    return "".join(" * %s" % name for name in _width_factors())


def _block_offset(outer):
    """`q * VS + lane`, or `q` alone where a work item is the element."""
    return _default_target().work_item_block_offset(outer)


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
        "%sfor (int %s = 0; %s < ne; ++%s) {"
        % (indent, work_item, work_item, work_item),
    )


def _restrict_define_line(restrict_definition):
    restrict_definition = str(restrict_definition)
    if restrict_definition:
        return "#define RSTR %s" % restrict_definition
    return "#define RSTR"


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


def geometry_kernels_header_source_for(target):
    """`geometry_kernels`, spelled for one target.

    Every argument below is one of the target's own answers, so there is one
    place that turns a target into this header.  There used to be three: the
    OpenMP source builder passed its answers, `emitters/energy.py` carried a
    125-line hand-written copy for CUDA, and the residual emitter called the
    builder with its CPU defaults -- which under a device target produced
    `for (int 0 = 0; 0 < ne; ++0)`, since the default work item is a lane name
    and the device's is the literal `0`.  Two of those wrote
    `geometry_kernels.cuh` with different contents.
    """
    inline_qualifier = target.inline_qualifier()
    vectorize_pragma = target.vectorize_pragma()
    policy = target.loop_lowering_policy()
    return sfem_geometry_kernels_header_source(
        inline_qualifier=inline_qualifier,
        define_sfem_inline=inline_qualifier == "SFEM_INLINE",
        restrict_definition=target.restrict_definition(),
        work_item_index=target.work_item_index(),
        simd_lines=() if vectorize_pragma is None else (vectorize_pragma,),
        single_work_item=not policy.emits_lane_loop,
        header_guard_suffix=target.header_guard_suffix(),
    )


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
        "      ",
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
            *restrict_prelude(restrict_definition or ""),
            "",
            *kernel_status_macro_lines(),
            "namespace sfem {",
            "namespace codegen {",
            "",
            "template <%s>" % ", ".join(("typename s_t", "int ND", "int NQ") + _width_parameters()),
            "struct GeometryJacobianAdjugateDeterminant;",
            "",
            "template <typename s_t>",
            "static %s void geometry_jacobian_adjugate_and_determinant_2(" % inline_qualifier,
            "    const s_t J00,",
            "    const s_t J01,",
            "    const s_t J10,",
            "    const s_t J11,",
            "    s_t *const *const RSTR adjugate,",
            "    s_t *const RSTR determinant,",
            "    const ptrdiff_t offset) {",
            "  adjugate[0][offset] = J11;",
            "  adjugate[1][offset] = -J01;",
            "  adjugate[2][offset] = -J10;",
            "  adjugate[3][offset] = J00;",
            "  determinant[offset] = J00 * J11 - J01 * J10;",
            "}",
            "",
            "template <typename s_t>",
            "static %s void geometry_jacobian_adjugate_and_determinant_3(" % inline_qualifier,
            "    const s_t J00,",
            "    const s_t J01,",
            "    const s_t J02,",
            "    const s_t J10,",
            "    const s_t J11,",
            "    const s_t J12,",
            "    const s_t J20,",
            "    const s_t J21,",
            "    const s_t J22,",
            "    s_t *const *const RSTR adjugate,",
            "    s_t *const RSTR determinant,",
            "    const ptrdiff_t offset) {",
            "  adjugate[0][offset] = J11 * J22 - J12 * J21;",
            "  adjugate[1][offset] = J02 * J21 - J01 * J22;",
            "  adjugate[2][offset] = J01 * J12 - J02 * J11;",
            "  adjugate[3][offset] = J12 * J20 - J10 * J22;",
            "  adjugate[4][offset] = J00 * J22 - J02 * J20;",
            "  adjugate[5][offset] = J02 * J10 - J00 * J12;",
            "  adjugate[6][offset] = J10 * J21 - J11 * J20;",
            "  adjugate[7][offset] = J01 * J20 - J00 * J21;",
            "  adjugate[8][offset] = J00 * J11 - J01 * J10;",
            "  determinant[offset] = J00 * (J11 * J22 - J12 * J21)",
            "      - J01 * (J10 * J22 - J12 * J20)",
            "      + J02 * (J10 * J21 - J11 * J20);",
            "}",
            "",
            "template <%s>" % ", ".join(("typename s_t", "int NQ") + _width_parameters()),
            "struct GeometryJacobianAdjugateDeterminant<%s> {" % ", ".join(("s_t", "2", "NQ") + _width_factors()),
            "  static %s void eval(" % inline_qualifier,
            "      const int ne,",
            "      const s_t *const RSTR coordinate_grad_ref,",
            "      s_t *const *const RSTR adjugate,",
            "      s_t *const RSTR determinant) {",
            "    for (int q = 0; q < NQ; ++q) {",
            "      const s_t *const RSTR J00_q = &coordinate_grad_ref[((0 * NQ + q) * 2 + 0)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J01_q = &coordinate_grad_ref[((0 * NQ + q) * 2 + 1)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J10_q = &coordinate_grad_ref[((1 * NQ + q) * 2 + 0)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J11_q = &coordinate_grad_ref[((1 * NQ + q) * 2 + 1)%s];"
            % _width_suffix(),
            *work_loop,
            "        const ptrdiff_t offset = %s;" % _block_offset("q"),
            "        const s_t J00 = J00_q[%s];" % work_item,
            "        const s_t J01 = J01_q[%s];" % work_item,
            "        const s_t J10 = J10_q[%s];" % work_item,
            "        const s_t J11 = J11_q[%s];" % work_item,
            "        geometry_jacobian_adjugate_and_determinant_2<s_t>(",
            "            J00, J01, J10, J11, adjugate, determinant, offset);",
            "      }",
            "    }",
            "  }",
            "};",
            "",
            "template <%s>" % ", ".join(("typename s_t", "int NQ") + _width_parameters()),
            "struct GeometryJacobianAdjugateDeterminant<%s> {" % ", ".join(("s_t", "3", "NQ") + _width_factors()),
            "  static %s void eval(" % inline_qualifier,
            "      const int ne,",
            "      const s_t *const RSTR coordinate_grad_ref,",
            "      s_t *const *const RSTR adjugate,",
            "      s_t *const RSTR determinant) {",
            "    for (int q = 0; q < NQ; ++q) {",
            "      const s_t *const RSTR J00_q = &coordinate_grad_ref[((0 * NQ + q) * 3 + 0)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J01_q = &coordinate_grad_ref[((0 * NQ + q) * 3 + 1)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J02_q = &coordinate_grad_ref[((0 * NQ + q) * 3 + 2)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J10_q = &coordinate_grad_ref[((1 * NQ + q) * 3 + 0)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J11_q = &coordinate_grad_ref[((1 * NQ + q) * 3 + 1)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J12_q = &coordinate_grad_ref[((1 * NQ + q) * 3 + 2)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J20_q = &coordinate_grad_ref[((2 * NQ + q) * 3 + 0)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J21_q = &coordinate_grad_ref[((2 * NQ + q) * 3 + 1)%s];"
            % _width_suffix(),
            "      const s_t *const RSTR J22_q = &coordinate_grad_ref[((2 * NQ + q) * 3 + 2)%s];"
            % _width_suffix(),
            *work_loop,
            "        const ptrdiff_t offset = %s;" % _block_offset("q"),
            "        const s_t J00 = J00_q[%s];" % work_item,
            "        const s_t J01 = J01_q[%s];" % work_item,
            "        const s_t J02 = J02_q[%s];" % work_item,
            "        const s_t J10 = J10_q[%s];" % work_item,
            "        const s_t J11 = J11_q[%s];" % work_item,
            "        const s_t J12 = J12_q[%s];" % work_item,
            "        const s_t J20 = J20_q[%s];" % work_item,
            "        const s_t J21 = J21_q[%s];" % work_item,
            "        const s_t J22 = J22_q[%s];" % work_item,
            "        geometry_jacobian_adjugate_and_determinant_3<s_t>(",
            "            J00, J01, J02, J10, J11, J12, J20, J21, J22,",
            "            adjugate, determinant, offset);",
            "      }",
            "    }",
            "  }",
            "};",
            "",
            "template <%s>" % ", ".join(("typename s_t", "int ND", "int NQ") + _width_parameters()),
            "static %s void geometry_jacobian_adjugate_and_determinant(" % inline_qualifier,
            "    const int ne,",
            "    const s_t *const RSTR coordinate_grad_ref,",
            "    s_t *const *const RSTR adjugate,",
            "    s_t *const RSTR determinant) {",
            "  GeometryJacobianAdjugateDeterminant<%s>::eval(" % ", ".join(("s_t", "ND", "NQ") + _width_factors()),
            "      ne, coordinate_grad_ref, adjugate, determinant);",
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
    dim_name="ND",
):
    return [
        "%ss_t *%s[%s * %s] = {%s};"
        % (indent, stream_array_name, dim_name, dim_name, ", ".join(adjugate_streams))
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
            "%sgeometry_jacobian_adjugate_and_determinant_2<s_t>(" % indent,
            "%s    J00, J01, J10, J11, %s, %s, %s);"
            % (indent, stream_array_name, determinant_stream, index),
        ]
    if dim == 3:
        return [
            "%sgeometry_jacobian_adjugate_and_determinant_3<s_t>(" % indent,
            "%s    J00, J01, J02, J10, J11, J12, J20, J21, J22,"
            % indent,
            "%s    %s, %s, %s);"
            % (indent, stream_array_name, determinant_stream, index),
        ]
    raise ValueError("isoparametric geometry supports dimensions 2 and 3")


def coordinate_stream_array_lines(
    coordinate_streams,
    *,
    stream_array_name="coordinate_streams",
    indent="    ",
):
    if isinstance(coordinate_streams, str):
        return [
            "%sconst s_t *%s[ND * NS];" % (indent, stream_array_name),
            "%sfor (int stream = 0; stream < ND * NS; ++stream) {" % indent,
            "%s  %s[stream] = %s[stream];"
            % (indent, stream_array_name, coordinate_streams),
            "%s}" % indent,
        ]

    return [
        "%sconst s_t *const %s[ND * NS] = {%s};"
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
    stream_array_name="coordinate_streams",
    indent="    ",
    adjugate_target,
    determinant_target,
    adjugate_streams=None,
    determinant_stream=None,
    contiguous_coordinate_streams=False,
    dim_name="ND",
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

    if contiguous_coordinate_streams:
        if not isinstance(coordinate_streams, str):
            raise ValueError("contiguous coordinate streams require a storage name")
        lines = []
        evaluator_streams = coordinate_streams
    else:
        lines = coordinate_stream_array_lines(
            coordinate_streams,
            stream_array_name=stream_array_name,
            indent=indent,
        )
        evaluator_streams = stream_array_name
    lines.extend([
        "%ss_t %s[%s];"
        % (
            indent,
            gradient_name,
            c_product(dim_name, "NQ", dim_name, *_width_factors()),
        ),
    ])
    lines.extend(
        evaluator_lines(
            evaluator_streams,
            gradient_name,
            indent,
        )
    )
    if not sum_factorization.evaluates_geometry_jacobian:
        raise ValueError("tensor-product geometry requires a Jacobian sum-factorization plan")
    lines.extend(
        tensor_product_adjugate_determinant_lines(
            dim_name=dim_name,
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


def tensor_product_ordered_streams(streams, n_field_components, dim, n_shape, shape_order=None):
    shape_order = (
        tensor_product_cartesian_shape_order(dim, n_shape)
        if shape_order is None
        else tuple(shape_order)
    )
    return streams_in_shape_order(tuple(streams), n_field_components, shape_order)


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
    indent="    ",
    gradient_name="coordinate_grad_ref",
    stream_array_name="coordinate_streams",
    shape_name="shape_1d",
    grad_name="grad_1d",
    adjugate_target,
    determinant_target,
    adjugate_streams=None,
    determinant_stream=None,
    contiguous_coordinate_streams=False,
    dim_name="ND",
):
    def evaluator_lines(streams, gradient, evaluator_indent):
        tensor_evaluate = _TENSOR_EVALUATE_KERNEL[bool(contiguous_coordinate_streams)]
        return [
            "%ss_t coordinate_value[%s];"
            % (evaluator_indent, c_product("ND", "NQ", *_width_factors())),
            "%s%s%s("
            % (
                evaluator_indent,
                micro_kernel(tensor_evaluate),
                micro_kernel_template("ND", "ND"),
            ),
            "%s    %s%s, %s, %s,"
            % (evaluator_indent, micro_kernel_count(), shape_name, grad_name, streams),
            "%s    coordinate_value, %s);" % (evaluator_indent, gradient),
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
        contiguous_coordinate_streams=contiguous_coordinate_streams,
    )


def tensor_product_coordinate_gradient_lines(
    *,
    dim,
    local_prefix,
    coordinate_streams,
    indent="    ",
    gradient_name="coordinate_grad_ref",
    stream_array_name="coordinate_streams",
    shape_name="shape_1d",
    grad_name="grad_1d",
    contiguous_coordinate_streams=False,
    dim_name="ND",
):
    if contiguous_coordinate_streams:
        if not isinstance(coordinate_streams, str):
            raise ValueError("contiguous coordinate streams require a storage name")
        lines = []
        evaluator_streams = coordinate_streams
        tensor_gradient = "tensor_gradient_contiguous"
    else:
        lines = coordinate_stream_array_lines(
            coordinate_streams,
            stream_array_name=stream_array_name,
            indent=indent,
        )
        evaluator_streams = stream_array_name
        tensor_gradient = "tensor_gradient"
    lines.extend([
        "%ss_t %s[%s];"
        % (
            indent,
            gradient_name,
            c_product(dim_name, "NQ", dim_name, *_width_factors()),
        ),
    ])
    for component in range(dim):
        lines.extend(
            [
                "%s%s%s("
                % (indent, micro_kernel(tensor_gradient), micro_kernel_template(dim)),
                "%s    %s%s, %s, %s, %d,"
                % (
                    indent,
                    micro_kernel_count(),
                    shape_name,
                    grad_name,
                    evaluator_streams,
                    component,
                ),
                "%s    %s + %s);"
                % (
                    indent,
                    gradient_name,
                    c_product(component, "NQ", dim_name, *_width_factors()),
                ),
            ]
        )
    return lines


def tensor_product_current_q_isoparametric_geometry_lines(
    *,
    dim,
    gradient_name="coordinate_grad_ref",
    indent="    ",
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
    body_indent = indent + "  "
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "%sconst s_t J%d%d = %s[%s];"
                % (
                    body_indent,
                    row,
                    col,
                    gradient_name,
                    _block_offset(
                        c_group(
                            c_sum(
                                c_product(
                                    c_group(c_sum(c_product(row, "NQ"), "q")), "ND"
                                ),
                                col,
                            )
                        )
                    ),
                )
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
    indent="    ",
    gradient_name="coordinate_grad_ref",
    stream_array_name="coordinate_streams",
    shape_name="shape_1d",
    grad_name="grad_1d",
    adjugate_target,
    determinant_target,
    adjugate_streams=None,
    determinant_stream=None,
    contiguous_coordinate_streams=False,
    dim_name="ND",
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
        contiguous_coordinate_streams=contiguous_coordinate_streams,
        dim_name=dim_name,
    )
    lines.extend(
        tensor_product_adjugate_determinant_lines(
            dim_name=dim_name,
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
    dim_name="ND",
):
    if adjugate_streams is not None and determinant_stream is not None:
        return [
            "",
            "%ss_t *%s_adjugate_streams[%s * %s] = {%s};"
            % (indent, gradient_name, dim_name, dim_name, ", ".join(adjugate_streams)),
            "%sgeometry_jacobian_adjugate_and_determinant<%s>("
            % (indent, ", ".join(("s_t", dim_name, "NQ") + _width_factors())),
            "%s    ne, %s, %s_adjugate_streams, %s);"
            % (indent, gradient_name, gradient_name, determinant_stream),
        ]

    lines = ["", "%sfor (int q = 0; q < NQ; ++q) {" % indent]
    work_item = _target_work_item_index(work_item_index)
    if include_lane_loop:
        lines.extend(
            _work_item_loop_lines(
                indent + "  ",
                work_item_index=work_item,
                simd_lines=simd_lines,
                single_work_item=single_work_item,
            )
        )
        body_indent = indent + "    "
        for row in range(dim):
            for col in range(dim):
                lines.append(
                    "%sconst s_t J%d%d = %s[%s];"
                    % (
                        body_indent,
                        row,
                        col,
                        gradient_name,
                        _block_offset(
                            c_group(
                                c_sum(
                                    c_product(
                                        c_group(c_sum(c_product(row, "NQ"), "q")), "ND"
                                    ),
                                    col,
                                )
                            )
                        ),
                    )
                )
        lines.extend(
            isoparametric_adjugate_lines(
                dim,
                body_indent,
                _block_offset("q"),
                adjugate_target,
                determinant_target,
            )
        )
        lines.append("%s  }" % indent)
    else:
        lines.extend(
            tensor_product_current_q_isoparametric_geometry_lines(
                dim=dim,
                gradient_name=gradient_name,
                indent=indent + "  ",
                adjugate_target=adjugate_target,
                determinant_target=determinant_target,
                output_index=_block_offset("q"),
                work_item_index=work_item,
                simd_lines=simd_lines,
                single_work_item=single_work_item,
            )
        )
    lines.append("%s}" % indent)
    return lines
