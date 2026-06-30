import sympy as sp

try:
    from .symbolic import (
        ExpressionCost,
        ExpressionRole,
        GeneratedKernelFile,
        KernelExpressions,
        SfemElementQuadratureRule,
        SfemSoAElementSpecialization,
        _component_name,
        _cpp_argument_name,
        _cpp_macro_name,
        _sfem_ccode,
        _sfem_math_header_source,
        _validate_diagnostics_plan_names,
        isoparametric_adjugate_call_lines,
        isoparametric_adjugate_lines,
        isoparametric_adjugate_stream_array_lines,
        quadrature_reference_accessor,
        quadrature_reference_struct_lines,
        sfem_element_quadrature_rule,
        sfem_mesh_reference_data,
        sfem_soa_reference_input,
        sfem_tensor_product_hex_uses_cartesian_ordering,
        streams_in_shape_order,
        tensor_product_cartesian_shape_order,
        tensor_product_coordinate_gradient_lines,
        tensor_product_current_q_isoparametric_geometry_lines,
        tensor_product_gradient_isoparametric_geometry_lines,
        tensor_product_ordered_coordinate_streams,
        validate_reference_data_plan,
    )
except ImportError:
    from symbolic import (
        ExpressionCost,
        ExpressionRole,
        GeneratedKernelFile,
        KernelExpressions,
        SfemElementQuadratureRule,
        SfemSoAElementSpecialization,
        _component_name,
        _cpp_argument_name,
        _cpp_macro_name,
        _sfem_ccode,
        _sfem_math_header_source,
        _validate_diagnostics_plan_names,
        isoparametric_adjugate_call_lines,
        isoparametric_adjugate_lines,
        isoparametric_adjugate_stream_array_lines,
        quadrature_reference_accessor,
        quadrature_reference_struct_lines,
        sfem_element_quadrature_rule,
        sfem_mesh_reference_data,
        sfem_soa_reference_input,
        sfem_tensor_product_hex_uses_cartesian_ordering,
        streams_in_shape_order,
        tensor_product_cartesian_shape_order,
        tensor_product_coordinate_gradient_lines,
        tensor_product_current_q_isoparametric_geometry_lines,
        tensor_product_gradient_isoparametric_geometry_lines,
        tensor_product_ordered_coordinate_streams,
        validate_reference_data_plan,
    )


def _default_openmp_energy_source_builder():
    try:
        from .energy_emitters import OpenMPEnergySoASourceBuilder
    except ImportError:
        from energy_emitters import OpenMPEnergySoASourceBuilder

    return OpenMPEnergySoASourceBuilder()


def generate_sfem_soa_cpp_files(
    forms,
    *,
    prefix,
    dim,
    n_nodes,
    n_qp=1,
    vector_size=16,
    array_inputs=None,
    element_type=None,
    quadrature_order=None,
    quadrature_rule=None,
    affine_quadrature_rule=None,
    basis_family=None,
    geometry_family=None,
    local_prefix=None,
    reference_data_plan=None,
    diagnostics_plan=None,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    forms = tuple(forms)
    if quadrature_rule is None and element_type is not None:
        quadrature_rule = sfem_element_quadrature_rule(element_type, quadrature_order)
    if quadrature_rule is not None:
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        n_qp = quadrature_rule.n_qp
    if affine_quadrature_rule is None:
        affine_quadrature_rule = quadrature_rule
    if affine_quadrature_rule is not None:
        if affine_quadrature_rule.dim != dim:
            raise ValueError("affine and isoparametric quadrature rules must have the same dimension")
        if affine_quadrature_rule.n_shape != n_nodes:
            raise ValueError("affine and isoparametric quadrature rules must have the same shape count")
    n_qp = int(n_qp)
    array_inputs = tuple(
        array_inputs
        if array_inputs is not None
        else (sfem_soa_reference_input("grad_ref", n_qp, n_nodes, dim),)
    )
    if dim < 1 or dim > 3:
        raise ValueError("SoA backend currently supports dimensions 1, 2, and 3")
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if n_qp <= 0:
        raise ValueError("n_qp must be positive")
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    for array_input in _sfem_soa_reference_inputs(array_inputs):
        if array_input.n_qp != n_qp:
            raise ValueError(
                "reference input '%s' has n_qp=%d, expected %d"
                % (array_input.name, array_input.n_qp, n_qp)
            )
        if array_input.n_shape != n_nodes:
            raise ValueError(
                "reference input '%s' has n_shape=%d, expected %d"
                % (array_input.name, array_input.n_shape, n_nodes)
            )
    if quadrature_rule is not None:
        _validate_sfem_soa_quadrature_rule(quadrature_rule, dim, n_nodes, n_qp, array_inputs)
    if reference_data_plan is not None:
        validate_reference_data_plan(
            reference_data_plan,
            prefix,
            affine_quadrature_rule,
            quadrature_rule,
            basis_family,
        )
    if diagnostics_plan is not None:
        _validate_diagnostics_plan_names(
            diagnostics_plan,
            tuple(
                _sfem_soa_public_function_name(prefix, form.name, quadrature_rule)
                for form in forms
            ),
        )

    local_prefix = prefix if local_prefix is None else str(local_prefix)
    use_shared_weak_local = local_prefix != prefix
    local_name = "%s_local.hpp" % local_prefix
    math_name = "kernel_math.hpp"
    tensor_product_name = "tensor_product_kernels.hpp"
    geometry_name = "geometry_kernels.hpp"
    diagnostics_name = "kernel_diagnostics.hpp"
    operator_name = "%s_operator.%s" % (prefix, source_builder.operator_extension)
    files = [
        GeneratedKernelFile(
            math_name,
            _sfem_math_header_source(),
        ),
        GeneratedKernelFile(
            geometry_name,
            source_builder.geometry_header_source(),
        ),
        GeneratedKernelFile(
            diagnostics_name,
            "\n".join(_sfem_soa_diagnostics_header()),
        ),
    ]
    if source_builder.emits_tensor_product_header(basis_family):
        files.append(
            GeneratedKernelFile(
                tensor_product_name,
                source_builder.tensor_product_header_source(),
            )
        )
    files.extend(
        [
        GeneratedKernelFile(
            local_name,
            _sfem_soa_local_header(
                forms,
                local_prefix,
                dim,
                n_nodes,
                array_inputs,
                quadrature_rule,
                basis_family,
                use_shared_weak_local,
                math_name,
                tensor_product_name,
                source_builder,
            ),
        ),
            GeneratedKernelFile(
            operator_name,
            _sfem_soa_operator_source(
                forms,
                prefix,
                dim,
                n_nodes,
                n_qp,
                vector_size,
                local_prefix,
                local_name,
                geometry_name,
                diagnostics_name,
                array_inputs,
                quadrature_rule,
                affine_quadrature_rule,
                basis_family,
                geometry_family,
                use_shared_weak_local,
                source_builder,
            ),
        ),
        ]
    )
    return tuple(files)


def generate_sfem_soa_cpp_files_for_element(
    forms,
    *,
    prefix,
    emission_plan,
    array_inputs=None,
    local_prefix=None,
    reference_data_plan=None,
    diagnostics_plan=None,
    source_builder=None,
):
    if emission_plan is None:
        raise ValueError("energy code generation requires an ElementEmissionPlan")
    specialization = emission_plan.isoparametric_specialization
    affine_specialization = emission_plan.affine_specialization
    basis_family = emission_plan.basis_family
    geometry_family = emission_plan.geometry_family
    if isinstance(specialization, SfemElementQuadratureRule):
        specialization = SfemSoAElementSpecialization(specialization)
    if not isinstance(specialization, SfemSoAElementSpecialization):
        raise TypeError("specialization must be an SfemSoAElementSpecialization")
    if affine_specialization is None:
        affine_specialization = specialization
    if isinstance(affine_specialization, SfemElementQuadratureRule):
        affine_specialization = SfemSoAElementSpecialization(affine_specialization)
    if not isinstance(affine_specialization, SfemSoAElementSpecialization):
        raise TypeError("affine_specialization must be an SfemSoAElementSpecialization")
    array_inputs = (
        specialization.adjugate_geometry_inputs()
        if array_inputs is None
        else tuple(array_inputs)
    )
    return generate_sfem_soa_cpp_files(
        forms,
        prefix=prefix,
        dim=specialization.dim,
        n_nodes=specialization.n_shape,
        n_qp=specialization.n_qp,
        vector_size=specialization.vector_size,
        array_inputs=array_inputs,
        quadrature_rule=specialization.quadrature_rule,
        affine_quadrature_rule=affine_specialization.quadrature_rule,
        basis_family=basis_family,
        geometry_family=geometry_family,
        local_prefix=local_prefix,
        reference_data_plan=reference_data_plan,
        diagnostics_plan=diagnostics_plan,
        source_builder=source_builder,
    )


def _sfem_soa_local_header(
    forms,
    prefix,
    dim,
    n_nodes,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    use_shared_weak_local=False,
    math_name="kernel_math.hpp",
    tensor_product_name="tensor_product_kernels.hpp",
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    guard = "%s_LOCAL_HPP" % _cpp_macro_name(prefix)
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "#include <math.h>",
        "#include <stddef.h>",
        "#if defined(__has_include)",
        '#if __has_include("sfem_base.hpp")',
        '#include "sfem_base.hpp"',
        "#define SFEM_GENERATED_SCALAR_T",
        "#endif",
        "#endif",
        "",
        *source_builder.local_header_preamble_lines(
            math_name,
            tensor_product_name,
            basis_family,
        ),
        "",
        "#ifndef SFEM_GENERATED_SCALAR_T",
        "#define SFEM_GENERATED_SCALAR_T",
        "typedef double real_t;",
        "typedef ptrdiff_t idx_t;",
        "typedef double geom_t;",
        "#endif",
        "",
    ]
    lines = [line for line in lines if line != ""]
    lines.extend(["namespace sfem {", "namespace codegen {", ""])

    for form in forms:
        lines.extend(
            _sfem_soa_block_function(
                form,
                prefix,
                dim,
                n_nodes,
                array_inputs,
                quadrature_rule,
                basis_family,
                use_shared_weak_local,
                source_builder,
            )
        )
        lines.append("")

    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _sfem_soa_block_function(
    form,
    prefix,
    dim,
    n_nodes,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    use_shared_weak_local=False,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    name = "%s_%s_block" % (prefix, form.name)
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_reference_gradient_vectors = (
        not use_tensor_product_reference
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_stream_arrays = use_shared_weak_local and form.weak_form is not None
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )
    params = ["const ptrdiff_t nelems"]
    if form.weak_form is not None:
        params.append("const ptrdiff_t geometry_stride")
    else:
        params.append("const int q")
    params.extend(
        "const %s *const SFEM_RESTRICT %s" % (array_input.scalar_type, stream)
        for array_input in element_inputs
        for stream in _soa_array_stream_names(array_input)
    )
    if use_tensor_product_reference:
        params.extend(
            (
                "const scalar_t *const SFEM_RESTRICT shape_1d",
                "const scalar_t *const SFEM_RESTRICT grad_1d",
            )
        )
    elif use_reference_gradient_vectors:
        params.extend(_sfem_reference_gradient_vector_params(dim))
    else:
        params.extend(
            "const %s *const SFEM_RESTRICT %s"
            % (array_input.scalar_type, _sfem_soa_reference_param_name(array_input))
            for array_input in reference_inputs
        )
    if form.weak_form is not None:
        if use_tensor_product_reference:
            params.append("const scalar_t *const SFEM_RESTRICT q_weight_1d")
        else:
            params.append("const scalar_t *const SFEM_RESTRICT q_weight")
    else:
        params.append("const scalar_t qw")
    params.extend(("const scalar_t mu", "const scalar_t lmbda"))
    if use_stream_arrays:
        params.extend(
            (
                "const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * %d]" % dim,
            )
        )
        if form.has_direction:
            params.extend(
                (
                    "const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * %d]"
                    % dim,
                )
            )
        if form.name == "objective":
            params.extend(("scalar_t *const SFEM_RESTRICT value",))
        else:
            params.extend(
                (
                    "scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * %d]"
                    % dim,
                )
            )
    else:
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s" % name
            for name in _field_stream_names("u", dim, n_nodes)
        )
        if form.has_direction:
            params.extend(
                "const scalar_t *const SFEM_RESTRICT %s" % name
                for name in _field_stream_names("h", dim, n_nodes)
            )
        params.extend(
            "scalar_t *const SFEM_RESTRICT %s" % name
            for name in _output_stream_names(form, dim, n_nodes)
        )

    lines = [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>",
        "static SFEM_INLINE void %s(" % name,
    ]
    for idx, param in enumerate(params):
        comma = "," if idx + 1 < len(params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static_assert(N_QP > 0, \"N_QP must be positive\");",
            "    static_assert(VECTOR_SIZE > 0, \"VECTOR_SIZE must be positive\");",
        ]
    )
    if not use_stream_arrays:
        lines.append(
            "    static_assert(N_SHAPE == %d, \"N_SHAPE does not match generated expression\");"
            % n_nodes
        )
    if use_tensor_product_reference:
        if use_stream_arrays:
            lines.extend(
                [
                    "    static constexpr int N_QP_1D = integer_root(N_QP, %d);"
                    % quadrature_rule.dim,
                    "    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, %d);"
                    % quadrature_rule.dim,
                    "    static_assert(ipow(N_QP_1D, %d) == N_QP, \"N_QP must be tensor-product compatible\");"
                    % quadrature_rule.dim,
                    "    static_assert(ipow(N_SHAPE_1D, %d) == N_SHAPE, \"N_SHAPE must be tensor-product compatible\");"
                    % quadrature_rule.dim,
                ]
            )
        else:
            lines.extend(
                [
                    "    static constexpr int N_QP_1D = %d;"
                    % quadrature_rule.tensor_product_n_qp_1d,
                    "    static constexpr int N_SHAPE_1D = %d;"
                    % quadrature_rule.tensor_product_n_shape_1d,
                ]
            )
        if form.weak_form is None:
            lines.extend(_tensor_product_q_index_lines(quadrature_rule.dim, "    "))
    if form.weak_form is not None and not use_stream_arrays:
        lines.append(
            "    const scalar_t *const weak_u_streams[N_SHAPE * %d] = {%s};"
            % (
                dim,
                ", ".join(
                    streams_in_shape_order(
                        _field_stream_names("u", dim, n_nodes),
                        dim,
                        stream_shape_order,
                    )
                ),
            )
        )
        if form.has_direction:
            lines.append(
                "    const scalar_t *const weak_h_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        streams_in_shape_order(
                            _field_stream_names("h", dim, n_nodes),
                            dim,
                            stream_shape_order,
                        )
                    ),
                )
            )
        if form.name != "objective":
            lines.append(
                "    scalar_t *const weak_out_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        streams_in_shape_order(
                            _output_stream_names(form, dim, n_nodes),
                            dim,
                            stream_shape_order,
                        )
                    ),
                )
            )
    if form.weak_form is not None and use_tensor_product_reference:
        _append_sfem_soa_tensor_weak_form_lines(
            lines,
            form,
            prefix,
            dim,
            use_stream_arrays,
            source_builder,
        )
        lines.append("}")
        return lines

    if form.weak_form is None:
        lines.extend(source_builder.simd_lines())
        lines.append("    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {")
    if form.weak_form is None:
        lines.append("        scalar_t u[N_SHAPE * %d];" % dim)
    for array_input in array_inputs:
        if form.weak_form is not None:
            continue
        if array_input.is_reference_qp_shape:
            array_decl = "%s %s[N_SHAPE * %d];" % (
                array_input.scalar_type,
                array_input.name,
                array_input.components,
            )
        else:
            array_decl = "%s %s[%d];" % (
                array_input.scalar_type,
                array_input.name,
                array_input.local_size,
            )
        lines.append("        %s" % array_decl)
    if form.has_direction and form.weak_form is None:
        lines.append("        scalar_t du[N_SHAPE * %d];" % dim)

    if form.weak_form is None:
        for array_input in element_inputs:
            for i, stream in enumerate(_soa_array_stream_names(array_input)):
                lines.append("        %s[%d] = %s[lane];" % (array_input.name, i, stream))
    if form.weak_form is not None:
        pass
    elif use_tensor_product_reference:
        _append_tensor_product_reference_gradient_lines(
            lines,
            reference_inputs[0].name,
            quadrature_rule,
        )
    elif use_reference_gradient_vectors:
        array_input = reference_inputs[0]
        for shape in range(array_input.n_shape):
            for component in range(array_input.components):
                local_idx = shape * array_input.components + component
                lines.append(
                    "        %s[%d] = %s[q * N_SHAPE + %d];"
                    % (
                        array_input.name,
                        local_idx,
                        _sfem_reference_gradient_vector_name(component),
                        shape,
                    )
                )
    else:
        for array_input in reference_inputs:
            source = _sfem_soa_reference_param_name(array_input)
            for shape in range(array_input.n_shape):
                for component in range(array_input.components):
                    local_idx = shape * array_input.components + component
                    lines.append(
                        "        %s[%d] = %s[(q * N_SHAPE + %d) * %d + %d];"
                        % (
                            array_input.name,
                            local_idx,
                            source,
                            shape,
                            array_input.components,
                            component,
                        )
                    )

    if form.weak_form is None:
        if use_stream_arrays:
            lines.extend(["        for (int shape = 0; shape < N_SHAPE; ++shape) {"])
            for d in range(dim):
                lines.append(
                    "            u[shape * %d + %d] = u_streams[shape * %d + %d][lane];"
                    % (dim, d, dim, d)
                )
                if form.has_direction:
                    lines.append(
                        "            du[shape * %d + %d] = h_streams[shape * %d + %d][lane];"
                        % (dim, d, dim, d)
                    )
            lines.append("        }")
        else:
            for node in range(n_nodes):
                for d in range(dim):
                    idx = node * dim + d
                    component = _component_name(d)
                    lines.append("        u[%d] = u%s%d[lane];" % (idx, component, node))
                    if form.has_direction:
                        lines.append("        du[%d] = h%s%d[lane];" % (idx, component, node))

    if form.weak_form is not None:
        _append_sfem_soa_weak_form_lines(
            lines,
            form,
            prefix,
            dim,
            n_nodes,
            reference_inputs,
            use_tensor_product_reference,
            quadrature_rule,
            use_stream_arrays,
            source_builder,
        )
        lines.append("}")
        return lines

    output_count = len(form.expression_graph.evaluation_plan.outputs)
    lines.append("        scalar_t element_vector[%d];" % max(1, output_count))
    _append_sfem_soa_statement_lines(lines, form.expression_graph, "element_vector")
    _append_sfem_soa_output_lines(lines, form, dim, n_nodes)
    lines.extend(["    }", "}"])
    return lines


def _append_sfem_soa_tensor_weak_form_lines(
    lines,
    form,
    prefix,
    dim,
    use_stream_arrays,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    weak_form = form.weak_form
    u_streams = "u_streams" if use_stream_arrays else "weak_u_streams"
    h_streams = "h_streams" if use_stream_arrays else "weak_h_streams"
    out_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
    block_extent = "N_QP * %d * VECTOR_SIZE" % (dim * dim)

    lines.append("    scalar_t grad_u_ref_q[%s];" % block_extent)
    if form.has_direction:
        lines.append("    scalar_t grad_h_ref_q[%s];" % block_extent)
    if form.name != "objective":
        lines.append("    scalar_t loperand_q[%s];" % block_extent)

    for row in range(dim):
        output_offset = "%d * N_QP * %d * VECTOR_SIZE" % (row, dim)
        lines.append(
            "    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, %d>(nelems, shape_1d, grad_1d, %s, %d, &grad_u_ref_q[%s]);"
            % (dim, u_streams, row, output_offset)
        )
        if form.has_direction:
            lines.append(
                "    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, %d>(nelems, shape_1d, grad_1d, %s, %d, &grad_h_ref_q[%s]);"
                % (dim, h_streams, row, output_offset)
            )

    lines.append("    for (int q = 0; q < N_QP; ++q) {")
    lines.extend(_tensor_product_q_index_lines(dim, "        "))
    lines.append(
        "        const scalar_t qw = %s;"
        % _tensor_product_quadrature_weight_expr(dim)
    )
    lines.extend(source_builder.simd_lines())
    lines.extend(
        [
            "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "            const ptrdiff_t geometry_offset = q * geometry_stride + lane;",
        ]
    )
    for component in range(dim * dim):
        lines.append(
            "            const scalar_t jacobian_adjugate_lane%d = jacobian_adjugate%d[geometry_offset];"
            % (component, component)
        )
    lines.append(
        "            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];"
    )
    lines.append("            scalar_t grad_u_ref[%d];" % (dim * dim))
    for row in range(dim):
        for col in range(dim):
            component = row * dim + col
            lines.append(
                "            grad_u_ref[%d] = grad_u_ref_q[((%d * N_QP + q) * %d + %d) * VECTOR_SIZE + lane];"
                % (component, row, dim, col)
            )
    if form.has_direction:
        lines.append("            scalar_t grad_h_ref[%d];" % (dim * dim))
        for row in range(dim):
            for col in range(dim):
                component = row * dim + col
                lines.append(
                    "            grad_h_ref[%d] = grad_h_ref_q[((%d * N_QP + q) * %d + %d) * VECTOR_SIZE + lane];"
                    % (component, row, dim, col)
                )

    def geometry_value(name, component):
        return "%s_lane%d" % (name, component)

    lines.append("            scalar_t grad_u[%d];" % (dim * dim))
    if form.has_direction:
        lines.append("            scalar_t trial_grad[%d];" % (dim * dim))

    lines.append(
        "            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;"
    )
    for row in range(dim):
        for col in range(dim):
            terms = [
                "grad_u_ref[%d] * %s"
                % (
                    row * dim + k,
                    geometry_value("jacobian_adjugate", k * dim + col),
                )
                for k in range(dim)
            ]
            lines.append(
                "            grad_u[%d] = (%s) * inv_jacobian_determinant;"
                % (row * dim + col, " + ".join(terms))
            )
            if form.has_direction:
                terms = [
                    "grad_h_ref[%d] * %s"
                    % (
                        row * dim + k,
                        geometry_value("jacobian_adjugate", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "            trial_grad[%d] = (%s) * inv_jacobian_determinant;"
                    % (row * dim + col, " + ".join(terms))
                )

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "grad_u",
    )
    if form.name == "objective":
        _append_cse_array_assignments(
            lines,
            [weak_form.energy_density.xreplace(deformation_gradient_substitutions)],
            ["value[lane] %s" % ("+=" if form.output_mode == "accumulate" else "=")],
            "weak_obj_tmp",
            scale="qw * jacobian_determinant_lane0",
        )
        lines.extend(["        }", "    }"])
        return

    material = (
        weak_form.linearized_first_piola(
            tuple(sp.symbols("trial_grad[%d]" % i) for i in range(dim * dim))
        )
        if form.name == "apply"
        else weak_form.first_piola()
    ).xreplace(deformation_gradient_substitutions)
    lines.append("            scalar_t loperand[%d];" % (dim * dim))
    _append_transformed_loperand_lines(
        lines,
        material,
        dim,
        "weak_mat_tmp",
        geometry_value,
    )
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "            loperand_q[((%d * N_QP + q) * %d + %d) * VECTOR_SIZE + lane] = loperand[%d];"
                % (row, dim, col, row * dim + col)
            )
    lines.extend(["        }", "    }"])
    for row in range(dim):
        lines.append(
            "    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, %d>(nelems, shape_1d, grad_1d, &loperand_q[%d * N_QP * %d * VECTOR_SIZE], %s, %d);"
            % (dim, row, dim, out_streams, row)
        )


def _append_sfem_soa_weak_form_lines(
    lines,
    form,
    prefix,
    dim,
    n_nodes,
    reference_inputs,
    use_tensor_product_reference,
    quadrature_rule,
    use_stream_arrays=False,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    weak_form = form.weak_form
    if weak_form.dim != dim:
        raise ValueError("weak form dim does not match SoA kernel dim")
    if form.name not in ("objective", "gradient", "apply"):
        raise ValueError("weak form kernel name must be objective, gradient, or apply")
    if form.name == "apply" and not form.has_direction:
        raise ValueError("weak form apply kernel requires has_direction=True")
    if len(reference_inputs) != 1 or reference_inputs[0].name != "grad_ref":
        raise ValueError("weak form kernels require one grad_ref reference input")
    if use_tensor_product_reference:
        raise AssertionError("tensor-product weak forms must use the SoA tensor weak-form source_builder")

    def reference_gradient(component):
        if use_tensor_product_reference:
            return _tensor_product_dynamic_reference_gradient_expr(dim, component)
        return "%s[q * N_SHAPE + shape]" % _sfem_reference_gradient_vector_name(component)

    def field_value(field, row):
        stream_prefix = "" if use_stream_arrays else "weak_"
        return "%s%s_streams[shape * %d + %d][lane]" % (
            stream_prefix,
            field,
            dim,
            row,
        )

    def geometry_value(name, component):
        return "%s_lane%d" % (name, component)

    lines.append("        for (int q = 0; q < N_QP; ++q) {")
    lines.extend(source_builder.simd_lines())
    lines.append("            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {")
    lines.append("            const ptrdiff_t geometry_offset = q * geometry_stride + lane;")
    for component in range(dim * dim):
        lines.append(
            "            const scalar_t jacobian_adjugate_lane%d = jacobian_adjugate%d[geometry_offset];"
            % (component, component)
        )
    lines.append(
        "            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];"
    )
    if use_tensor_product_reference:
        lines.extend(_tensor_product_q_index_lines(dim, "            "))
        lines.append(
            "            const scalar_t qw = %s;"
            % _tensor_product_quadrature_weight_expr(dim)
        )
        for row in range(dim):
            for col in range(dim):
                lines.append(
                    "            const scalar_t grad_u_ref%d = grad_u_ref_q[(%d * N_QP + q) * %d + %d];"
                    % (row * dim + col, row, dim, col)
                )
        if form.has_direction:
            for row in range(dim):
                for col in range(dim):
                    lines.append(
                        "            const scalar_t grad_h_ref%d = grad_h_ref_q[(%d * N_QP + q) * %d + %d];"
                        % (row * dim + col, row, dim, col)
                    )
    else:
        lines.append("            const scalar_t qw = q_weight[q];")
        for row in range(dim):
            for col in range(dim):
                idx = row * dim + col
                lines.append("            scalar_t grad_u_ref%d = scalar_t(0);" % idx)
                if form.has_direction:
                    lines.append("            scalar_t grad_h_ref%d = scalar_t(0);" % idx)
        lines.extend(["            for (int shape = 0; shape < N_SHAPE; ++shape) {"])
        for row in range(dim):
            for col in range(dim):
                lines.append(
                    "                grad_u_ref%d += %s * %s;"
                    % (row * dim + col, field_value("u", row), reference_gradient(col))
                )
                if form.has_direction:
                    lines.append(
                        "                grad_h_ref%d += %s * %s;"
                        % (row * dim + col, field_value("h", row), reference_gradient(col))
                    )
        lines.append("            }")

    lines.append(
        "        const scalar_t inv_jacobian_determinant = scalar_t(1) / %s;"
        % geometry_value("jacobian_determinant", 0)
    )
    for row in range(dim):
        for col in range(dim):
            terms = [
                "grad_u_ref%d * %s"
                % (
                    row * dim + k,
                    geometry_value("jacobian_adjugate", k * dim + col),
                )
                for k in range(dim)
            ]
            lines.append(
                "        const scalar_t grad_u%d = (%s) * inv_jacobian_determinant;"
                % (row * dim + col, " + ".join(terms))
            )
            if form.has_direction:
                terms = [
                    "grad_h_ref%d * %s"
                    % (
                        row * dim + k,
                        geometry_value("jacobian_adjugate", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "        const scalar_t trial_grad%d = (%s) * inv_jacobian_determinant;"
                    % (row * dim + col, " + ".join(terms))
                )

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "grad_u",
        scalar_temporaries=True,
    )

    if form.name == "objective":
        _append_cse_array_assignments(
            lines,
            [weak_form.energy_density.xreplace(deformation_gradient_substitutions)],
            ["value[lane] %s" % ("+=" if form.output_mode == "accumulate" else "=")],
            "weak_obj_tmp",
            scale="qw * %s" % geometry_value("jacobian_determinant", 0),
        )
        lines.extend(["            }", "        }"])
        return

    material = (
        weak_form.linearized_first_piola(
            tuple(sp.symbols("trial_grad%d" % i) for i in range(dim * dim))
        )
        if form.name == "apply"
        else weak_form.first_piola()
    ).xreplace(deformation_gradient_substitutions)
    _append_transformed_loperand_lines(
        lines,
        material,
        dim,
        "weak_mat_tmp",
        geometry_value,
        scalar_temporaries=True,
    )

    lines.extend(["            for (int shape = 0; shape < N_SHAPE; ++shape) {"])
    for row in range(dim):
        terms = [
            "loperand%d * %s" % (row * dim + col, reference_gradient(col))
            for col in range(dim)
        ]
        op = "+=" if form.output_mode == "accumulate" else "="
        output_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
        lines.append(
            "                %s[shape * %d + %d][lane] %s %s;"
            % (output_streams, dim, row, op, " + ".join(terms))
        )
    lines.extend(["            }", "            }", "        }"])


def _append_transformed_loperand_lines(
    lines,
    material,
    dim,
    temporary_prefix,
    geometry_value,
    scalar_temporaries=False,
):
    material_exprs = tuple(material)
    if scalar_temporaries:
        material_names = ["const scalar_t material%d =" % i for i in range(dim * dim)]
    else:
        material_names = ["material[%d] =" % i for i in range(dim * dim)]
        lines.append("        scalar_t material[%d];" % (dim * dim))
    _append_cse_array_assignments(lines, material_exprs, material_names, temporary_prefix)
    for row in range(dim):
        for col in range(dim):
            terms = [
                "%s * %s"
                % (
                    "material%d" % (row * dim + k)
                    if scalar_temporaries
                    else "material[%d]" % (row * dim + k),
                    geometry_value("jacobian_adjugate", col * dim + k),
                )
                for k in range(dim)
            ]
            if scalar_temporaries:
                lines.append(
                    "        const scalar_t loperand%d = qw * (%s);"
                    % (row * dim + col, " + ".join(terms))
                )
            else:
                lines.append(
                    "        loperand[%d] = qw * (%s);"
                    % (row * dim + col, " + ".join(terms))
                )


def _weak_form_deformation_gradient_substitutions(
    weak_form,
    gradient_name,
    scalar_temporaries=False,
):
    substitutions = {}
    for row in range(weak_form.dim):
        for col in range(weak_form.dim):
            idx = row * weak_form.dim + col
            if scalar_temporaries:
                value = sp.Symbol("%s%d" % (gradient_name, idx))
            else:
                value = sp.Symbol("%s[%d]" % (gradient_name, idx))
            if row == col:
                value = sp.Integer(1) + value
            substitutions[weak_form.deformation_gradient[idx]] = value
    return substitutions


def _append_cse_array_assignments(lines, expressions, targets, temporary_prefix, scale=None):
    temps, reduced = sp.cse(
        tuple(expressions),
        symbols=sp.numbered_symbols("%s" % temporary_prefix),
    )
    for symbol, expression in temps:
        lines.append("        const scalar_t %s = %s;" % (symbol, _sfem_ccode(expression)))
    for target, expression in zip(targets, reduced):
        if scale is not None:
            lines.append("        %s %s * (%s);" % (target, scale, _sfem_ccode(expression)))
        else:
            lines.append("        %s %s;" % (target, _sfem_ccode(expression)))


def _append_sfem_soa_statement_lines(lines, expression_graph, output_name):
    output_index = 0
    for statement in expression_graph.evaluation_plan.statements:
        expression = _sfem_ccode(statement.expression)
        if statement.kind == "intermediate":
            lines.append("        const scalar_t %s = %s;" % (statement.target, expression))
        else:
            lines.append("        %s[%d] = %s;" % (output_name, output_index, expression))
            output_index += 1


def _tensor_product_q_index_lines(dim, indent):
    if dim == 2:
        return (
            "%sconst int qx = q %% N_QP_1D;" % indent,
            "%sconst int qy = q / N_QP_1D;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int qx = q %% N_QP_1D;" % indent,
            "%sconst int qy = (q / N_QP_1D) %% N_QP_1D;" % indent,
            "%sconst int qz = q / (N_QP_1D * N_QP_1D);" % indent,
        )
    raise ValueError("tensor-product reference generation requires dim 2 or 3")


def _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes):
    if (
        quadrature_rule is not None
        and sfem_tensor_product_hex_uses_cartesian_ordering(quadrature_rule.element_type)
    ):
        return tuple(range(n_nodes))
    return tensor_product_cartesian_shape_order(dim, n_nodes)


def _use_tensor_product_reference(quadrature_rule, reference_inputs, basis_family=None):
    if quadrature_rule is None:
        return False
    if basis_family is None:
        raise ValueError("basis family must be provided by the emission plan")
    tensor_product = str(basis_family) == "tensor_product"
    return (
        tensor_product
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )


def _tensor_product_node_coords(quadrature_rule):
    dim = quadrature_rule.dim
    n_shape_1d = quadrature_rule.tensor_product_n_shape_1d
    cartesian_hex = (
        dim == 3
        and n_shape_1d == 2
        and sfem_tensor_product_hex_uses_cartesian_ordering(quadrature_rule.element_type)
    )
    if n_shape_1d == 2 and dim == 2:
        return ((0, 0), (1, 0), (1, 1), (0, 1))
    if n_shape_1d == 2 and dim == 3 and not cartesian_hex:
        return (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        )
    if dim == 2:
        return tuple(
            (sx, sy)
            for sy in range(n_shape_1d)
            for sx in range(n_shape_1d)
        )
    if dim == 3:
        return tuple(
            (sx, sy, sz)
            for sz in range(n_shape_1d)
            for sy in range(n_shape_1d)
            for sx in range(n_shape_1d)
        )
    raise ValueError("tensor-product node ordering requires dim 2 or 3")


def _tensor_product_shape_index_lines(quadrature_rule, indent):
    dim = quadrature_rule.dim
    n_shape_1d = quadrature_rule.tensor_product_n_shape_1d
    if n_shape_1d == 2 and dim == 2:
        return (
            "%sconst int sx = ((shape + 1) >> 1) & 1;" % indent,
            "%sconst int sy = shape >> 1;" % indent,
        )
    if (
        n_shape_1d == 2
        and dim == 3
        and not sfem_tensor_product_hex_uses_cartesian_ordering(quadrature_rule.element_type)
    ):
        return (
            "%sconst int sx = ((shape + 1) >> 1) & 1;" % indent,
            "%sconst int sy = (shape >> 1) & 1;" % indent,
            "%sconst int sz = shape >> 2;" % indent,
        )
    if dim == 2:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = shape / N_SHAPE_1D;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = (shape / N_SHAPE_1D) %% N_SHAPE_1D;" % indent,
            "%sconst int sz = shape / (N_SHAPE_1D * N_SHAPE_1D);" % indent,
        )
    raise ValueError("tensor-product shape indices require dim 2 or 3")


def _tensor_product_generic_shape_index_lines(dim, indent):
    if dim == 2:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = shape / N_SHAPE_1D;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = (shape / N_SHAPE_1D) %% N_SHAPE_1D;" % indent,
            "%sconst int sz = shape / (N_SHAPE_1D * N_SHAPE_1D);" % indent,
        )
    raise ValueError("tensor-product shape indices require dim 2 or 3")


def _sfem_reference_gradient_vector_name(component):
    return "grad_ref_%s" % _component_name(component)


def _sfem_reference_gradient_vector_params(dim):
    return tuple(
        "const scalar_t *const SFEM_RESTRICT %s"
        % _sfem_reference_gradient_vector_name(component)
        for component in range(dim)
    )


def _tensor_product_1d_factor(axis, node_axis, derivative_axis):
    qp_name = ("qx", "qy", "qz")[axis]
    table_name = "grad_1d" if axis == derivative_axis else "shape_1d"
    return "%s[%s * N_SHAPE_1D + %d]" % (table_name, qp_name, node_axis)


def _tensor_product_dynamic_reference_gradient_expr(
    dim,
    derivative_axis,
    shape_name="shape_1d",
    grad_name="grad_1d",
):
    factors = []
    for axis in range(dim):
        qp_name = ("qx", "qy", "qz")[axis]
        node_axis_name = ("sx", "sy", "sz")[axis]
        table_name = grad_name if axis == derivative_axis else shape_name
        factors.append("%s[%s * N_SHAPE_1D + %s]" % (table_name, qp_name, node_axis_name))
    return " * ".join(factors)


def _tensor_product_quadrature_weight_expr(dim, weight_name="q_weight_1d"):
    factors = ["%s[%s]" % (weight_name, name) for name in ("qx", "qy", "qz")[:dim]]
    return " * ".join(factors)


def _append_tensor_product_reference_gradient_lines(lines, name, quadrature_rule):
    for shape, coords in enumerate(_tensor_product_node_coords(quadrature_rule)):
        for component in range(quadrature_rule.dim):
            factors = [
                _tensor_product_1d_factor(axis, coords[axis], component)
                for axis in range(quadrature_rule.dim)
            ]
            lines.append(
                "        %s[%d] = %s;"
                % (name, shape * quadrature_rule.dim + component, " * ".join(factors))
            )


def _sfem_soa_isoparametric_geometry_lines(
    dim,
    n_nodes,
    quadrature_rule,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_inputs,
    q_major=False,
    reference_prefix="",
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    stream_array_name = "block_jacobian_adjugate_streams"
    lines = isoparametric_adjugate_stream_array_lines(
        dim=dim,
        indent="            ",
        stream_array_name=stream_array_name,
        adjugate_streams=tuple(
            "block_jacobian_adjugate%d" % component
            for component in range(dim * dim)
        ),
    )
    lines.extend(source_builder.simd_lines())
    lines.append("            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {")
    for row in range(dim):
        for col in range(dim):
            lines.append("                scalar_t J%d%d = scalar_t(0);" % (row, col))
    lines.append("                for (int shape = 0; shape < N_SHAPE; ++shape) {")
    if use_tensor_product_reference:
        lines.extend(_tensor_product_shape_index_lines(quadrature_rule, "                    "))
    for col in range(dim):
        lines.append(
            "                    const scalar_t g%d = %s;"
            % (
                col,
                _sfem_soa_isoparametric_reference_gradient_expr(
                    dim,
                    col,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    reference_inputs,
                    reference_prefix,
                ),
            )
        )
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "                    J%d%d += block_coordinate_streams[shape * %d + %d][lane] * g%d;"
                % (row, col, dim, row, col)
            )
    lines.append("                }")
    output_index = "q * VECTOR_SIZE + lane" if q_major else "lane"
    lines.extend(
        isoparametric_adjugate_call_lines(
            dim=dim,
            indent="                ",
            index=output_index,
            stream_array_name=stream_array_name,
            determinant_stream="block_jacobian_determinant0",
        )
    )
    lines.append("            }")
    return lines


def _sfem_soa_isoparametric_reference_gradient_expr(
    dim,
    component,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_inputs,
    reference_prefix="",
):
    if use_tensor_product_reference:
        return _tensor_product_dynamic_reference_gradient_expr(
            dim,
            component,
            "%sshape_1d" % reference_prefix,
            "%sgrad_1d" % reference_prefix,
        )
    if use_reference_gradient_vectors:
        return "%s%s[q * N_SHAPE + shape]" % (
            reference_prefix,
            _sfem_reference_gradient_vector_name(component),
        )
    if len(reference_inputs) == 1 and reference_inputs[0].name == "grad_ref":
        return "%s%s[(q * N_SHAPE + shape) * %d + %d]" % (
            reference_prefix,
            reference_inputs[0].name,
            dim,
            component,
        )
    raise ValueError("isoparametric geometry generation requires one grad_ref reference input")


def _sfem_soa_adjugate_assignment_lines(dim, indent, index="lane"):
    return isoparametric_adjugate_lines(
        dim,
        indent,
        index,
        lambda component, output_index: (
            "block_jacobian_adjugate%d[%s]" % (component, output_index)
        ),
        lambda output_index: (
            "block_jacobian_determinant0[%s]" % output_index
        ),
    )


def _append_sfem_soa_output_lines(lines, form, dim, n_nodes):
    output_count = len(form.expression_graph.evaluation_plan.outputs)
    if output_count == 1:
        op = "+=" if form.output_mode == "accumulate" else "="
        lines.append("        value[lane] %s element_vector[0];" % op)
        return

    if output_count != dim * n_nodes:
        raise ValueError(
            "SoA form '%s' has %d outputs, expected 1 or dim*n_nodes=%d"
            % (form.name, output_count, dim * n_nodes)
        )

    for node in range(n_nodes):
        for d in range(dim):
            stream = "out%s%d" % (_component_name(d), node)
            idx = node * dim + d
            op = "+=" if form.output_mode == "accumulate" else "="
            lines.append("        %s[lane] %s element_vector[%d];" % (stream, op, idx))


def _sfem_soa_operator_source(
    forms,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    local_name,
    geometry_name,
    diagnostics_name,
    array_inputs,
    quadrature_rule,
    affine_quadrature_rule,
    basis_family=None,
    geometry_family=None,
    use_shared_weak_local=False,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    lines = [
        *source_builder.operator_preamble_lines(local_name, geometry_name, diagnostics_name),
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "",
        "#ifndef MIN",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
        "#endif",
        "",
    ]
    lines = [line for line in lines if line != ""]
    lines.append("")
    if quadrature_rule is not None:
        lines.extend(["namespace sfem {", "namespace codegen {", ""])
        if affine_quadrature_rule is not None:
            lines.extend(
                quadrature_reference_struct_lines(
                    prefix,
                    "affine",
                    sfem_mesh_reference_data(affine_quadrature_rule),
                )
            )
        lines.extend(
            quadrature_reference_struct_lines(
                prefix,
                "isoparametric",
                sfem_mesh_reference_data(quadrature_rule),
            )
        )
        lines.extend(["", "} // namespace codegen", "} // namespace sfem"])
        lines.append("")

    for form in forms:
        lines.extend(
            _sfem_soa_diagnostics_lines(
                form,
                prefix,
                dim,
                n_nodes,
                n_qp,
                vector_size,
                array_inputs,
                quadrature_rule,
                basis_family,
            )
        )
        lines.append("")
        if quadrature_rule is not None and _sfem_soa_has_adjugate_geometry_inputs(array_inputs, dim):
            affine_rule = (
                affine_quadrature_rule
                if affine_quadrature_rule is not None
                else quadrature_rule
            )
            lines.append("")
            lines.extend(
                _sfem_soa_mesh_operator_function(
                    form,
                    prefix,
                    dim,
                    n_nodes,
                    affine_rule.n_qp,
                    vector_size,
                    local_prefix,
                    array_inputs,
                    affine_rule,
                    basis_family,
                    geometry_family,
                    use_shared_weak_local,
                    geometry_mode="affine",
                    source_builder=source_builder,
                )
            )
            if form.name == "objective" and source_builder.emit_objective_steps:
                lines.append("")
                lines.extend(
                    _sfem_soa_mesh_objective_steps_function(
                        form,
                        prefix,
                        dim,
                        n_nodes,
                        affine_rule.n_qp,
                        vector_size,
                        local_prefix,
                        array_inputs,
                        affine_rule,
                        basis_family,
                        geometry_family,
                        use_shared_weak_local,
                        geometry_mode="affine",
                        source_builder=source_builder,
                    )
                )
            lines.append("")
            lines.extend(
                _sfem_soa_mesh_operator_function(
                    form,
                    prefix,
                    dim,
                    n_nodes,
                    n_qp,
                    vector_size,
                    local_prefix,
                    array_inputs,
                    quadrature_rule,
                    basis_family,
                    geometry_family,
                    use_shared_weak_local,
                    geometry_mode="isoparametric",
                    source_builder=source_builder,
                )
            )
            if form.name == "objective" and source_builder.emit_objective_steps:
                lines.append("")
                lines.extend(
                    _sfem_soa_mesh_objective_steps_function(
                        form,
                        prefix,
                        dim,
                        n_nodes,
                        n_qp,
                        vector_size,
                        local_prefix,
                        array_inputs,
                        quadrature_rule,
                        basis_family,
                        geometry_family,
                        use_shared_weak_local,
                        geometry_mode="isoparametric",
                        source_builder=source_builder,
                    )
                )
        lines.append("")

    return "\n".join(lines)


def _sfem_soa_operator_function(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    geometry_family=None,
    use_shared_weak_local=False,
    isoparametric_geometry=False,
):
    function_name = (
        _sfem_soa_isoparametric_public_function_name(prefix, form.name, quadrature_rule)
        if isoparametric_geometry
        else _sfem_soa_public_function_name(prefix, form.name, quadrature_rule)
    )
    implementation_name = "%s_impl" % function_name
    block_name = "%s_%s_block" % (local_prefix, form.name)
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_tensor_product_geometry = (
        isoparametric_geometry and str(geometry_family) == "tensor_product"
    )
    use_reference_gradient_vectors = (
        not use_tensor_product_reference
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_stream_arrays = use_shared_weak_local and form.weak_form is not None
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )
    base_params = ["const ptrdiff_t nelements"]
    if isoparametric_geometry:
        base_params.extend(
            "const real_t *const SFEM_RESTRICT %s" % stream
            for stream in _coordinate_stream_names(dim, n_nodes)
        )
    else:
        base_params.extend(
            "const %s *const SFEM_RESTRICT %s" % (array_input.scalar_type, stream)
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    if use_tensor_product_reference:
        reference_params = (
            "const scalar_t *const SFEM_RESTRICT shape_1d",
            "const scalar_t *const SFEM_RESTRICT grad_1d",
        )
        quadrature_weight_param = "const scalar_t *const SFEM_RESTRICT q_weight_1d"
    elif use_reference_gradient_vectors:
        reference_params = _sfem_reference_gradient_vector_params(dim)
        quadrature_weight_param = "const scalar_t *const SFEM_RESTRICT q_weight"
    else:
        reference_params = tuple(
            "const %s *const SFEM_RESTRICT %s" % (array_input.scalar_type, array_input.name)
            for array_input in reference_inputs
        )
        quadrature_weight_param = "const scalar_t *const SFEM_RESTRICT q_weight"
    material_params = ["const scalar_t mu", "const scalar_t lmbda"]
    field_params = []
    field_params.extend(
        "const real_t *const SFEM_RESTRICT %s" % name
        for name in _field_stream_names("u", dim, n_nodes)
    )
    if form.has_direction:
        field_params.extend(
            "const real_t *const SFEM_RESTRICT %s" % name
            for name in _field_stream_names("h", dim, n_nodes)
        )
    output_params = tuple(
        "real_t *const SFEM_RESTRICT %s" % name
        for name in _output_stream_names(form, dim, n_nodes)
    )
    if isoparametric_geometry and quadrature_rule is None:
        raise ValueError("isoparametric SoA wrappers require an element quadrature rule")
    if quadrature_rule is None:
        impl_params = tuple(base_params) + reference_params + (
            "const scalar_t qw",
        ) + tuple(material_params) + tuple(field_params) + output_params
        wrapper_params = impl_params
    else:
        impl_params = tuple(base_params) + reference_params + (
            quadrature_weight_param,
        ) + tuple(material_params) + tuple(field_params) + output_params
        wrapper_params = tuple(base_params) + tuple(material_params) + tuple(field_params) + output_params
    wrapper_params = _sfem_soa_public_wrapper_params(wrapper_params)

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>",
        "static SFEM_INLINE int %s(" % implementation_name,
    ]
    for idx, param in enumerate(impl_params):
        comma = "," if idx + 1 < len(impl_params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static_assert(N_QP == %d, \"N_QP does not match generated geometry streams\");" % n_qp,
            "    static_assert(N_SHAPE == %d, \"N_SHAPE does not match generated expression\");" % n_nodes,
            "    static_assert(VECTOR_SIZE > 0, \"VECTOR_SIZE must be positive\");",
        ]
    )
    if use_tensor_product_reference:
        lines.extend(
            [
                "    static constexpr int N_QP_1D = %d;"
                % quadrature_rule.tensor_product_n_qp_1d,
                "    static constexpr int N_SHAPE_1D = %d;"
                % quadrature_rule.tensor_product_n_shape_1d,
            ]
        )
    lines.extend(
        [
            "",
            "#pragma omp parallel for schedule(static)",
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
        ]
    )

    compact_stream_buffers = use_stream_arrays
    block_streams = []
    if compact_stream_buffers:
        if isoparametric_geometry:
            lines.append("        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
        lines.append("        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if form.has_direction:
            lines.append("        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if form.name != "objective":
            lines.append("        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];")
        else:
            lines.append("        scalar_t block_value[VECTOR_SIZE];")
    elif isoparametric_geometry:
        for stream in _coordinate_stream_names(dim, n_nodes):
            block_streams.append(stream)
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
    if form.weak_form is None or isoparametric_geometry:
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                block_streams.append(stream)
                extent = "N_QP * VECTOR_SIZE" if form.weak_form is not None else "VECTOR_SIZE"
                lines.append("        %s block_%s[%s];" % (array_input.scalar_type, stream, extent))
    if not compact_stream_buffers:
        for stream in _field_stream_names("u", dim, n_nodes):
            block_streams.append(stream)
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
        if form.has_direction:
            for stream in _field_stream_names("h", dim, n_nodes):
                block_streams.append(stream)
                lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
        for stream in _output_stream_names(form, dim, n_nodes):
            block_streams.append(stream)
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)

    if compact_stream_buffers:
        if isoparametric_geometry:
            lines.append(
                "        const scalar_t *const coordinate_stream_inputs[N_SHAPE * DIM] = {%s};"
                % ", ".join(
                    stream
                    for stream in tensor_product_ordered_coordinate_streams(
                        dim,
                        n_nodes,
                        _coordinate_stream_names(dim, n_nodes),
                        shape_order=stream_shape_order,
                    )
                )
            )
            lines.extend(
                [
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "#pragma omp simd",
                    "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                    "                block_coordinate_data[stream][lane] = coordinate_stream_inputs[stream][evbegin + lane];",
                    "            }",
                    "        }",
                ]
            )
        lines.append(
            "        const scalar_t *const u_stream_inputs[N_SHAPE * DIM] = {%s};"
            % ", ".join(
                stream
                for stream in streams_in_shape_order(
                    _field_stream_names("u", dim, n_nodes),
                    dim,
                    stream_shape_order,
                )
            )
        )
        lines.extend(
            [
                "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                "#pragma omp simd",
                "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                "                block_u_data[stream][lane] = u_stream_inputs[stream][evbegin + lane];",
                "            }",
                "        }",
            ]
        )
        if form.has_direction:
            lines.append(
                "        const scalar_t *const h_stream_inputs[N_SHAPE * DIM] = {%s};"
                % ", ".join(
                    stream
                    for stream in streams_in_shape_order(
                        _field_stream_names("h", dim, n_nodes),
                        dim,
                        stream_shape_order,
                    )
                )
            )
            lines.extend(
                [
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "#pragma omp simd",
                    "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                    "                block_h_data[stream][lane] = h_stream_inputs[stream][evbegin + lane];",
                    "            }",
                    "        }",
                ]
            )
        if form.name == "objective":
            if form.output_mode == "accumulate":
                lines.extend(["#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {", "            block_value[lane] = value[evbegin + lane];", "        }"])
            else:
                lines.extend(["#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {", "            block_value[lane] = scalar_t(0);", "        }"])
        else:
            lines.append(
                "        scalar_t *const out_stream_outputs[N_SHAPE * DIM] = {%s};"
                % ", ".join(
                    stream
                    for stream in streams_in_shape_order(
                        _output_stream_names(form, dim, n_nodes),
                        dim,
                        stream_shape_order,
                    )
                )
            )
            if form.output_mode == "accumulate":
                lines.extend(
                    [
                        "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                        "#pragma omp simd",
                        "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                        "                block_out_data[stream][lane] = out_stream_outputs[stream][evbegin + lane];",
                        "            }",
                        "        }",
                    ]
                )
            else:
                lines.extend(
                    [
                        "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                        "#pragma omp simd",
                        "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                        "                block_out_data[stream][lane] = scalar_t(0);",
                        "            }",
                        "        }",
                    ]
                )
    else:
        lines.extend(["", "#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
        if isoparametric_geometry:
            for stream in _coordinate_stream_names(dim, n_nodes):
                lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
        for stream in _field_stream_names("u", dim, n_nodes):
            lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
        if form.has_direction:
            for stream in _field_stream_names("h", dim, n_nodes):
                lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
        for stream in _output_stream_names(form, dim, n_nodes):
            if form.output_mode == "accumulate":
                lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
            else:
                lines.append("            block_%s[lane] = scalar_t(0);" % stream)
        lines.append("        }")

    if use_stream_arrays:
        lines.append("")
        if compact_stream_buffers:
            lines.extend(
                [
                    "        const scalar_t *block_u_streams[N_SHAPE * DIM];",
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "            block_u_streams[stream] = block_u_data[stream];",
                    "        }",
                ]
            )
        else:
            lines.append(
                "        const scalar_t *const block_u_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        "block_%s" % stream
                        for stream in streams_in_shape_order(
                            _field_stream_names("u", dim, n_nodes),
                            dim,
                            stream_shape_order,
                        )
                    ),
                )
            )
        if form.has_direction:
            if compact_stream_buffers:
                lines.extend(
                    [
                        "        const scalar_t *block_h_streams[N_SHAPE * DIM];",
                        "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                        "            block_h_streams[stream] = block_h_data[stream];",
                        "        }",
                    ]
                )
            else:
                lines.append(
                    "        const scalar_t *const block_h_streams[N_SHAPE * %d] = {%s};"
                    % (
                        dim,
                        ", ".join(
                            "block_%s" % stream
                            for stream in streams_in_shape_order(
                                _field_stream_names("h", dim, n_nodes),
                                dim,
                                stream_shape_order,
                            )
                        ),
                    )
                )
        if form.name == "objective":
            pass
        else:
            if compact_stream_buffers:
                lines.extend(
                    [
                        "        scalar_t *block_out_streams[N_SHAPE * DIM];",
                        "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                        "            block_out_streams[stream] = block_out_data[stream];",
                        "        }",
                    ]
                )
            else:
                lines.append(
                    "        scalar_t *const block_out_streams[N_SHAPE * %d] = {%s};"
                    % (
                        dim,
                        ", ".join(
                            "block_%s" % stream
                            for stream in streams_in_shape_order(
                                _output_stream_names(form, dim, n_nodes),
                                dim,
                                stream_shape_order,
                            )
                        ),
                )
            )

    if isoparametric_geometry and not use_tensor_product_geometry:
        lines.append("")
        if compact_stream_buffers:
            lines.extend(
                [
                    "        const scalar_t *block_coordinate_streams[N_SHAPE * DIM];",
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "            block_coordinate_streams[stream] = block_coordinate_data[stream];",
                    "        }",
                ]
            )
        else:
            lines.append(
                "        const scalar_t *const block_coordinate_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        "block_%s" % stream
                        for stream in _coordinate_stream_names(dim, n_nodes)
                    ),
                )
            )

    if isoparametric_geometry and use_tensor_product_geometry:
        lines.append("")
        if form.weak_form is not None:
            lines.extend(
                tensor_product_gradient_isoparametric_geometry_lines(
                    dim=dim,
                    n_shape=n_nodes,
                    n_qp=quadrature_rule.n_qp,
                    local_prefix=local_prefix,
                    coordinate_streams=(
                        "block_coordinate_data"
                        if compact_stream_buffers
                        else tensor_product_ordered_coordinate_streams(
                            dim,
                            n_nodes,
                            _coordinate_stream_names(dim, n_nodes),
                            lambda stream: "block_%s" % stream,
                            shape_order=_tensor_product_stream_shape_order(
                                quadrature_rule,
                                dim,
                                n_nodes,
                            ),
                        )
                    ),
                    adjugate_target=lambda component, index: (
                        "block_jacobian_adjugate%d[%s]" % (component, index)
                    ),
                    determinant_target=lambda index: (
                        "block_jacobian_determinant0[%s]" % index
                    ),
                    adjugate_streams=tuple(
                        "block_jacobian_adjugate%d" % component
                        for component in range(dim * dim)
                    ),
                    determinant_stream="block_jacobian_determinant0",
                )
            )
        else:
            lines.extend(
                tensor_product_coordinate_gradient_lines(
                    dim=dim,
                    local_prefix=local_prefix,
                    coordinate_streams=(
                        "block_coordinate_data"
                        if compact_stream_buffers
                        else tensor_product_ordered_coordinate_streams(
                            dim,
                            n_nodes,
                            _coordinate_stream_names(dim, n_nodes),
                            lambda stream: "block_%s" % stream,
                            shape_order=_tensor_product_stream_shape_order(
                                quadrature_rule,
                                dim,
                                n_nodes,
                            ),
                        )
                    ),
                )
            )
            lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
            lines.extend(_tensor_product_q_index_lines(dim, "            "))
            lines.append(
                "            const scalar_t tensor_q_weight = %s;"
                % _tensor_product_quadrature_weight_expr(dim)
            )
            lines.extend(
                tensor_product_current_q_isoparametric_geometry_lines(
                    dim=dim,
                    indent="            ",
                    adjugate_target=lambda component, index: (
                        "block_jacobian_adjugate%d[%s]" % (component, index)
                    ),
                    determinant_target=lambda index: (
                        "block_jacobian_determinant0[%s]" % index
                    ),
                    output_index="lane",
                )
            )
    elif isoparametric_geometry:
        lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
        if use_tensor_product_reference:
            lines.extend(_tensor_product_q_index_lines(dim, "            "))
            if form.weak_form is None:
                lines.append(
                    "            const scalar_t tensor_q_weight = %s;"
                    % _tensor_product_quadrature_weight_expr(dim)
                )
        lines.extend(
            _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_inputs,
                form.weak_form is not None,
            )
        )
        if form.weak_form is not None:
            lines.append("        }")
    elif element_inputs and form.weak_form is None:
        lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
        if use_tensor_product_reference:
            lines.extend(_tensor_product_q_index_lines(dim, "            "))
            lines.append(
                "            const scalar_t tensor_q_weight = %s;"
                % _tensor_product_quadrature_weight_expr(dim)
            )
        lines.extend(["#pragma omp simd", "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                lines.append(
                    "                block_%s[lane] = %s[(ptrdiff_t)q * nelements + evbegin + lane];"
                    % (stream, stream)
                )
        lines.append("            }")
    elif form.weak_form is None:
        lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
        if use_tensor_product_reference:
            lines.extend(_tensor_product_q_index_lines(dim, "            "))
            lines.append(
                "            const scalar_t tensor_q_weight = %s;"
                % _tensor_product_quadrature_weight_expr(dim)
            )

    call_args = ["nelems"] if form.weak_form is not None else ["nelems", "q"]
    if form.weak_form is not None:
        call_args.append("VECTOR_SIZE" if isoparametric_geometry else "nelements")
    if form.weak_form is not None and not isoparametric_geometry:
        call_args.extend(
            "%s + evbegin" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        call_args.extend(
            "block_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    if use_tensor_product_reference:
        call_args.extend(("shape_1d", "grad_1d"))
    elif use_reference_gradient_vectors:
        call_args.extend(
            _sfem_reference_gradient_vector_name(component)
            for component in range(dim)
        )
    else:
        call_args.extend(array_input.name for array_input in reference_inputs)
    if form.weak_form is not None:
        call_args.append("q_weight_1d" if use_tensor_product_reference else "q_weight")
        call_args.extend(("mu", "lmbda"))
    elif quadrature_rule is None:
        call_args.extend(("qw", "mu", "lmbda"))
    elif use_tensor_product_reference:
        call_args.extend(("tensor_q_weight", "mu", "lmbda"))
    else:
        call_args.extend(("q_weight[q]", "mu", "lmbda"))
    if use_stream_arrays:
        call_args.append("block_u_streams")
        if form.has_direction:
            call_args.append("block_h_streams")
        if form.name == "objective":
            call_args.append("block_value")
        else:
            call_args.append("block_out_streams")
    else:
        call_args.extend("block_%s" % stream for stream in _field_stream_names("u", dim, n_nodes))
        if form.has_direction:
            call_args.extend("block_%s" % stream for stream in _field_stream_names("h", dim, n_nodes))
        call_args.extend("block_%s" % stream for stream in _output_stream_names(form, dim, n_nodes))
    call_indent = "        " if form.weak_form is not None else "            "
    lines.extend(
        [
            "",
            "%s%s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (call_indent, block_name, ", ".join(call_args)),
        ]
    )
    if form.weak_form is None:
        lines.append("        }")
    lines.append("")

    if compact_stream_buffers:
        if form.name == "objective":
            lines.extend(["#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {", "            value[evbegin + lane] = block_value[lane];", "        }"])
        else:
            lines.extend(
                [
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "#pragma omp simd",
                    "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                    "                out_stream_outputs[stream][evbegin + lane] = block_out_data[stream][lane];",
                    "            }",
                    "        }",
                ]
            )
    else:
        lines.extend(["#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
        for stream in _output_stream_names(form, dim, n_nodes):
            lines.append("            %s[evbegin + lane] = block_%s[lane];" % (stream, stream))
        lines.append("        }")
    lines.extend(
        [
            "    }",
            "",
            "    return SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )

    lines.append('extern "C" int %s(' % function_name)
    for idx, param in enumerate(wrapper_params):
        comma = "," if idx + 1 < len(wrapper_params) else ""
        lines.append("        %s%s" % (param, comma))
    if quadrature_rule is None:
        wrapper_args = tuple(_cpp_argument_name(param) for param in wrapper_params)
    else:
        wrapper_args = _sfem_soa_specialized_wrapper_arguments(
            prefix,
            quadrature_rule,
            wrapper_params,
            reference_inputs,
            use_reference_gradient_vectors,
            basis_family,
        )
    lines.extend(
        [
            ") {",
            "    return sfem::codegen::%s<real_t, %d, %d, %d>(%s);"
            % (
                implementation_name,
                n_qp,
                n_nodes,
                vector_size,
                ", ".join(wrapper_args),
            ),
            "}",
        ]
    )
    return lines


def _sfem_soa_mesh_operator_function(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    geometry_family=None,
    use_shared_weak_local=False,
    geometry_mode="affine",
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    effective_vector_size = source_builder.effective_vector_size(vector_size)
    if geometry_mode not in ("affine", "isoparametric"):
        raise ValueError("mesh geometry_mode must be 'affine' or 'isoparametric'")
    if quadrature_rule is None:
        raise ValueError("mesh SoA wrappers require an element quadrature rule")

    function_name = _sfem_soa_mesh_public_function_name(
        prefix,
        form.name,
        quadrature_rule,
        geometry_mode,
    )
    implementation_name = "%s_impl" % function_name
    block_name = "%s_%s_block" % (local_prefix, form.name)
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_tensor_product_geometry = (
        geometry_mode == "isoparametric"
        and str(geometry_family) == "tensor_product"
    )
    use_reference_gradient_vectors = (
        not use_tensor_product_reference
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_stream_arrays = use_shared_weak_local and form.weak_form is not None
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )

    base_params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
    ]
    if geometry_mode == "affine":
        base_params.extend(
            "const scalar_t *const SFEM_RESTRICT g_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        base_params.append("const geometry_t *const *const SFEM_RESTRICT points")

    material_params = ("const scalar_t mu", "const scalar_t lmbda")
    field_params = ["const ptrdiff_t u_stride"]
    field_params.extend(
        "const scalar_t *const SFEM_RESTRICT u%s" % _component_name(d)
        for d in range(dim)
    )
    if form.has_direction:
        field_params.append("const ptrdiff_t h_stride")
        field_params.extend(
            "const scalar_t *const SFEM_RESTRICT h%s" % _component_name(d)
            for d in range(dim)
        )
    if form.name == "objective":
        output_params = ("scalar_t *const SFEM_RESTRICT value",)
    else:
        output_params = tuple(["const ptrdiff_t out_stride"]) + tuple(
            "scalar_t *const SFEM_RESTRICT out%s" % _component_name(d)
            for d in range(dim)
        )

    impl_params = (
        tuple(base_params)
        + tuple(material_params)
        + tuple(field_params)
        + tuple(output_params)
    )
    wrapper_params = tuple(
        param.replace("geometry_t", "geom_t") for param in impl_params
    )

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        source_builder.mesh_template_line(geometry_mode),
        source_builder.mesh_function_line(implementation_name),
    ]
    for idx, param in enumerate(impl_params):
        comma = "," if idx + 1 < len(impl_params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % n_qp,
            "    static constexpr int N_SHAPE = %d;" % n_nodes,
            "    static constexpr int VECTOR_SIZE = %d;" % effective_vector_size,
            "    (void)nnodes;",
        ]
    )
    if geometry_mode == "isoparametric":
        for d in range(dim):
            lines.append(
                "    const geometry_t *const SFEM_RESTRICT %s = points[%d];"
                % (_component_name(d), d)
            )
    reference_prefix = "%s_" % geometry_mode
    tensor_shape_name = "%sshape_1d" % reference_prefix
    tensor_grad_name = "%sgrad_1d" % reference_prefix
    tensor_weight_name = "%sq_weight_1d" % reference_prefix
    scalar_weight_name = "%sq_weight" % reference_prefix
    lines.extend(
        _sfem_soa_mesh_reference_alias_lines(
            prefix,
            quadrature_rule,
            reference_inputs,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
            geometry_mode,
        )
    )
    if use_tensor_product_reference:
        lines.extend(
            [
                "    static constexpr int N_QP_1D = %d;"
                % quadrature_rule.tensor_product_n_qp_1d,
                "    static constexpr int N_SHAPE_1D = %d;"
                % quadrature_rule.tensor_product_n_shape_1d,
            ]
        )
    lines.append("")
    lines.extend(source_builder.parallel_for_lines())
    lines.extend(source_builder.mesh_loop_lines())
    lines.append("        idx_t ev[VECTOR_SIZE * N_SHAPE];")

    compact_stream_buffers = use_stream_arrays
    if compact_stream_buffers:
        lines.append("        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if form.has_direction:
            lines.append("        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if form.name != "objective":
            lines.append("        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];")
        else:
            lines.append("        scalar_t block_value[VECTOR_SIZE];")
        if geometry_mode == "isoparametric":
            lines.append("        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if tuple(stream_shape_order) != tuple(range(n_nodes)):
            lines.append(
                "        static constexpr int STREAM_SHAPE_ORDER[N_SHAPE] = {%s};"
                % ", ".join(str(shape) for shape in stream_shape_order)
            )
    elif geometry_mode == "isoparametric":
        for stream in _coordinate_stream_names(dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
    if geometry_mode == "isoparametric":
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                extent = "N_QP * VECTOR_SIZE" if form.weak_form is not None else "VECTOR_SIZE"
                lines.append("        scalar_t block_%s[%s];" % (stream, extent))
    if not compact_stream_buffers:
        for stream in _field_stream_names("u", dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
        if form.has_direction:
            for stream in _field_stream_names("h", dim, n_nodes):
                lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
        for stream in _output_stream_names(form, dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)

    lines.extend(
        [
            "",
            "        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {",
            "            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];",
            *tuple(source_builder.simd_lines()),
            "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];",
            "            }",
            "        }",
        ]
    )

    if geometry_mode == "isoparametric":
        if compact_stream_buffers:
            lines.append("        const geometry_t *const coordinate_components[DIM] = {%s};" % ", ".join(_component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "",
                    "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    "            const int stream_shape = %s;"
                    % (
                        "STREAM_SHAPE_ORDER[shape]"
                        if tuple(stream_shape_order) != tuple(range(n_nodes))
                        else "shape"
                    ),
                    "            for (int d = 0; d < DIM; ++d) {",
                    *source_builder.simd_lines(),
                    "                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                    "                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[lane * N_SHAPE + stream_shape]];",
                    "                }",
                    "            }",
                    "        }",
                ]
            )
        else:
            lines.extend([""])
            lines.extend(source_builder.simd_lines())
            lines.append("        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {")
            for shape in range(n_nodes):
                for d in range(dim):
                    stream = "%s%d" % (_component_name(d), shape)
                    lines.append(
                        "            block_%s[lane] = %s[ev[lane * N_SHAPE + %d]];"
                        % (stream, _component_name(d), shape)
                    )
            lines.append("        }")

    if compact_stream_buffers:
        lines.append("        const scalar_t *const u_components[DIM] = {%s};" % ", ".join("u%s" % _component_name(d) for d in range(dim)))
        if form.has_direction:
            lines.append("        const scalar_t *const h_components[DIM] = {%s};" % ", ".join("h%s" % _component_name(d) for d in range(dim)))
        lines.extend(
            [
                "",
                "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "            const int stream_shape = %s;"
                % (
                    "STREAM_SHAPE_ORDER[shape]"
                    if tuple(stream_shape_order) != tuple(range(n_nodes))
                    else "shape"
                ),
                "            for (int d = 0; d < DIM; ++d) {",
                *source_builder.simd_lines(),
                "                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                "                    const idx_t node = ev[lane * N_SHAPE + stream_shape];",
                "                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];",
            ]
        )
        if form.has_direction:
            lines.append("                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];")
        lines.extend(
            [
                "                }",
                "            }",
                "        }",
            ]
        )
        if form.name == "objective":
            lines.extend(source_builder.simd_lines())
            lines.extend(["        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {", "            block_value[lane] = scalar_t(0);", "        }"])
        else:
            lines.extend(
                [
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    *source_builder.simd_lines(),
                    "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                    "                block_out_data[stream][lane] = scalar_t(0);",
                    "            }",
                    "        }",
                ]
            )
    else:
        lines.extend([""])
        lines.extend(source_builder.simd_lines())
        lines.append("        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {")
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                lines.append(
                    "            block_u%s%d[lane] = u%s[ev[lane * N_SHAPE + %d] * u_stride];"
                    % (component, shape, component, shape)
                )
                if form.has_direction:
                    lines.append(
                        "            block_h%s%d[lane] = h%s[ev[lane * N_SHAPE + %d] * h_stride];"
                        % (component, shape, component, shape)
                    )
        for stream in _output_stream_names(form, dim, n_nodes):
            lines.append("            block_%s[lane] = scalar_t(0);" % stream)
        lines.append("        }")

    if use_stream_arrays:
        lines.append("")
        if compact_stream_buffers:
            lines.extend(
                [
                    "        const scalar_t *block_u_streams[N_SHAPE * DIM];",
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "            block_u_streams[stream] = block_u_data[stream];",
                    "        }",
                ]
            )
        else:
            lines.append(
                "        const scalar_t *const block_u_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        "block_%s" % stream
                        for stream in streams_in_shape_order(
                            _field_stream_names("u", dim, n_nodes),
                            dim,
                            stream_shape_order,
                        )
                    ),
                )
            )
        if form.has_direction:
            if compact_stream_buffers:
                lines.extend(
                    [
                        "        const scalar_t *block_h_streams[N_SHAPE * DIM];",
                        "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                        "            block_h_streams[stream] = block_h_data[stream];",
                        "        }",
                    ]
                )
            else:
                lines.append(
                    "        const scalar_t *const block_h_streams[N_SHAPE * %d] = {%s};"
                    % (
                        dim,
                        ", ".join(
                            "block_%s" % stream
                            for stream in streams_in_shape_order(
                                _field_stream_names("h", dim, n_nodes),
                                dim,
                                stream_shape_order,
                            )
                        ),
                    )
                )
        if form.name != "objective":
            if compact_stream_buffers:
                lines.extend(
                    [
                        "        scalar_t *block_out_streams[N_SHAPE * DIM];",
                        "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                        "            block_out_streams[stream] = block_out_data[stream];",
                        "        }",
                    ]
                )
            else:
                lines.append(
                    "        scalar_t *const block_out_streams[N_SHAPE * %d] = {%s};"
                    % (
                        dim,
                        ", ".join(
                            "block_%s" % stream
                            for stream in streams_in_shape_order(
                                _output_stream_names(form, dim, n_nodes),
                                dim,
                                stream_shape_order,
                            )
                        ),
                    )
                )

    if geometry_mode == "isoparametric" and not (
        form.weak_form is not None and use_tensor_product_reference
    ):
        lines.append("")
        if compact_stream_buffers:
            lines.extend(
                [
                    "        const scalar_t *block_coordinate_streams[N_SHAPE * DIM];",
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "            block_coordinate_streams[stream] = block_coordinate_data[stream];",
                    "        }",
                ]
            )
        else:
            lines.append(
                "        const scalar_t *const block_coordinate_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        "block_%s" % stream
                        for stream in _coordinate_stream_names(dim, n_nodes)
                    ),
                )
            )

    if form.weak_form is None:
        lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
        if use_tensor_product_reference:
            lines.extend(_tensor_product_q_index_lines(dim, "            "))
            lines.append(
                "            const scalar_t tensor_q_weight = %s;"
                % _tensor_product_quadrature_weight_expr(dim, tensor_weight_name)
            )

    if (
        geometry_mode == "isoparametric"
        and form.weak_form is not None
        and use_tensor_product_geometry
    ):
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_nodes,
                n_qp=quadrature_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams=(
                    "block_coordinate_data"
                    if compact_stream_buffers
                    else tensor_product_ordered_coordinate_streams(
                        dim,
                        n_nodes,
                        _coordinate_stream_names(dim, n_nodes),
                        lambda stream: "block_%s" % stream,
                        shape_order=_tensor_product_stream_shape_order(
                            quadrature_rule,
                            dim,
                            n_nodes,
                        ),
                    )
                ),
                adjugate_target=lambda component, index: (
                    "block_jacobian_adjugate%d[%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_jacobian_determinant0[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_jacobian_adjugate%d" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_jacobian_determinant0",
                shape_name=tensor_shape_name,
                grad_name=tensor_grad_name,
            )
        )
    elif geometry_mode == "isoparametric" and form.weak_form is not None:
        lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
        if use_tensor_product_geometry:
            lines.extend(_tensor_product_q_index_lines(dim, "            "))
        lines.extend(
            _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_geometry,
                use_reference_gradient_vectors,
                reference_inputs,
                q_major=form.weak_form is not None,
                reference_prefix=reference_prefix,
                source_builder=source_builder,
            )
        )
        lines.append("        }")
    elif geometry_mode == "isoparametric":
        lines.extend(
            _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_geometry,
                use_reference_gradient_vectors,
                reference_inputs,
                False,
                reference_prefix=reference_prefix,
                source_builder=source_builder,
            )
        )

    call_args = ["nelems"]
    if form.weak_form is not None:
        call_args.append("0" if geometry_mode == "affine" else "VECTOR_SIZE")
    else:
        call_args.append("q")
    if geometry_mode == "affine":
        call_args.extend(
            "g_%s + evbegin" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        call_args.extend(
            "block_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    if use_tensor_product_reference:
        call_args.extend((tensor_shape_name, tensor_grad_name))
    elif use_reference_gradient_vectors:
        call_args.extend(
            "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
            for component in range(dim)
        )
    else:
        call_args.extend("%s%s" % (reference_prefix, array_input.name) for array_input in reference_inputs)
    if form.weak_form is not None:
        call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
        call_args.extend(("mu", "lmbda"))
    elif use_tensor_product_reference:
        call_args.extend(("tensor_q_weight", "mu", "lmbda"))
    else:
        call_args.extend(("%s[q]" % scalar_weight_name, "mu", "lmbda"))
    if use_stream_arrays:
        call_args.append("block_u_streams")
        if form.has_direction:
            call_args.append("block_h_streams")
        if form.name == "objective":
            call_args.append("block_value")
        else:
            call_args.append("block_out_streams")
    else:
        call_args.extend("block_%s" % stream for stream in _field_stream_names("u", dim, n_nodes))
        if form.has_direction:
            call_args.extend("block_%s" % stream for stream in _field_stream_names("h", dim, n_nodes))
        call_args.extend("block_%s" % stream for stream in _output_stream_names(form, dim, n_nodes))
    call_indent = "        " if form.weak_form is not None else "            "
    lines.extend(
        [
            "",
            "%s%s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (call_indent, block_name, ", ".join(call_args)),
        ]
    )
    if form.weak_form is None:
        lines.append("        }")
    lines.append("")

    if form.name == "objective":
        lines.extend(source_builder.simd_lines())
        lines.append("        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {")
        lines.append("            value[evbegin + lane] += block_value[lane];")
        lines.append("        }")
    else:
        if compact_stream_buffers:
            lines.append("        scalar_t *const out_components[DIM] = {%s};" % ", ".join("out%s" % _component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    "            const int stream_shape = %s;"
                    % (
                        "STREAM_SHAPE_ORDER[shape]"
                        if tuple(stream_shape_order) != tuple(range(n_nodes))
                        else "shape"
                    ),
                    "            for (int d = 0; d < DIM; ++d) {",
                    "                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                    "                    const idx_t node = ev[lane * N_SHAPE + stream_shape] * out_stride;",
                    *source_builder.scatter_add_lines(
                        "out_components[d][node]",
                        "block_out_data[shape * DIM + d][lane]",
                        "                    ",
                    ),
                    "                }",
                    "            }",
                    "        }",
                ]
            )
        else:
            lines.append("        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {")
            for shape in range(n_nodes):
                for d in range(dim):
                    component = _component_name(d)
                    lines.extend(
                        list(
                            source_builder.scatter_add_lines(
                                "out%s[ev[lane * N_SHAPE + %d] * out_stride]"
                                % (component, shape),
                                "block_out%s%d[lane]" % (component, shape),
                                "            ",
                            )
                        )
                        + [""]
                    )
            lines.append("        }")

    lines.extend(
        [
            "    }",
            "",
            *source_builder.success_return_lines(),
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )

    wrapper_args = tuple(_cpp_argument_name(param) for param in wrapper_params)
    for public_name, scalar_type in (
        (function_name, "double"),
        ("%s_float" % function_name, "float"),
    ):
        concrete_params = _sfem_soa_concrete_scalar_params(wrapper_params, scalar_type)
        lines.append('extern "C" int %s(' % public_name)
        for idx, param in enumerate(concrete_params):
            comma = "," if idx + 1 < len(concrete_params) else ""
            lines.append("        %s%s" % (param, comma))
        lines.extend(
            [
                ") {",
            *source_builder.wrapper_call_lines(
                implementation_name,
                scalar_type,
                ", geom_t" if geometry_mode == "isoparametric" else "",
                wrapper_args,
            ),
            "}",
                "",
            ]
        )
    return lines


def _sfem_soa_mesh_objective_steps_function(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    geometry_family=None,
    use_shared_weak_local=False,
    geometry_mode="affine",
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    if form.name != "objective" or form.weak_form is None:
        return []
    if geometry_mode not in ("affine", "isoparametric"):
        raise ValueError("mesh geometry_mode must be 'affine' or 'isoparametric'")
    if quadrature_rule is None:
        raise ValueError("mesh objective_steps wrappers require an element quadrature rule")

    function_name = _sfem_soa_mesh_public_function_name(
        prefix,
        "objective_steps",
        quadrature_rule,
        geometry_mode,
    )
    implementation_name = "%s_impl" % function_name
    block_name = "%s_objective_block" % local_prefix
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_tensor_product_geometry = (
        geometry_mode == "isoparametric"
        and str(geometry_family) == "tensor_product"
    )
    use_reference_gradient_vectors = (
        not use_tensor_product_reference
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_stream_arrays = use_shared_weak_local
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )

    base_params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
    ]
    if geometry_mode == "affine":
        base_params.extend(
            "const scalar_t *const SFEM_RESTRICT g_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        base_params.append("const geometry_t *const *const SFEM_RESTRICT points")

    material_params = ("const scalar_t mu", "const scalar_t lmbda")
    field_params = ["const ptrdiff_t u_stride"]
    field_params.extend(
        "const scalar_t *const SFEM_RESTRICT u%s" % _component_name(d)
        for d in range(dim)
    )
    field_params.append("const ptrdiff_t h_stride")
    field_params.extend(
        "const scalar_t *const SFEM_RESTRICT h%s" % _component_name(d)
        for d in range(dim)
    )
    step_params = (
        "const int nsteps",
        "const scalar_t *const SFEM_RESTRICT steps",
    )
    output_params = ("scalar_t *const SFEM_RESTRICT value",)

    impl_params = (
        tuple(base_params)
        + tuple(material_params)
        + tuple(field_params)
        + tuple(step_params)
        + tuple(output_params)
    )
    wrapper_params = tuple(
        param.replace("geometry_t", "geom_t") for param in impl_params
    )

    reference_prefix = "%s_" % geometry_mode
    tensor_shape_name = "%sshape_1d" % reference_prefix
    tensor_grad_name = "%sgrad_1d" % reference_prefix
    tensor_weight_name = "%sq_weight_1d" % reference_prefix
    scalar_weight_name = "%sq_weight" % reference_prefix

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        source_builder.mesh_template_line(geometry_mode),
        source_builder.mesh_function_line(implementation_name),
    ]
    for idx, param in enumerate(impl_params):
        comma = "," if idx + 1 < len(impl_params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % n_qp,
            "    static constexpr int N_SHAPE = %d;" % n_nodes,
            "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
            "    (void)nnodes;",
        ]
    )
    if geometry_mode == "isoparametric":
        for d in range(dim):
            lines.append(
                "    const geometry_t *const SFEM_RESTRICT %s = points[%d];"
                % (_component_name(d), d)
            )
    lines.extend(
        _sfem_soa_mesh_reference_alias_lines(
            prefix,
            quadrature_rule,
            reference_inputs,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
            geometry_mode,
        )
    )
    if use_tensor_product_reference:
        lines.extend(
            [
                "    static constexpr int N_QP_1D = %d;"
                % quadrature_rule.tensor_product_n_qp_1d,
                "    static constexpr int N_SHAPE_1D = %d;"
                % quadrature_rule.tensor_product_n_shape_1d,
            ]
        )

    lines.extend(
        [
            "",
            *source_builder.parallel_for_lines(),
            *source_builder.mesh_loop_lines(),
            "        idx_t ev[VECTOR_SIZE * N_SHAPE];",
        ]
    )

    compact_stream_buffers = use_stream_arrays
    if compact_stream_buffers:
        lines.append("        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];")
        lines.append("        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];")
        lines.append("        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];")
        lines.append("        scalar_t block_value[VECTOR_SIZE];")
        if geometry_mode == "isoparametric":
            lines.append("        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if tuple(stream_shape_order) != tuple(range(n_nodes)):
            lines.append(
                "        static constexpr int STREAM_SHAPE_ORDER[N_SHAPE] = {%s};"
                % ", ".join(str(shape) for shape in stream_shape_order)
            )
    elif geometry_mode == "isoparametric":
        for stream in _coordinate_stream_names(dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
    if geometry_mode == "isoparametric":
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                extent = "N_QP * VECTOR_SIZE"
                lines.append("        scalar_t block_%s[%s];" % (stream, extent))
    if not compact_stream_buffers:
        for stream in _field_stream_names("u", dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
            lines.append("        scalar_t block_%s_base[VECTOR_SIZE];" % stream)
        for stream in _field_stream_names("h", dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
        lines.append("        scalar_t block_value[VECTOR_SIZE];")

    lines.extend(
        [
            "",
            "        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {",
            "            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];",
            *source_builder.simd_lines(),
            "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "                ev[lane * N_SHAPE + element_node] = element_shape[evbegin + lane];",
            "            }",
            "        }",
        ]
    )

    if geometry_mode == "isoparametric":
        if compact_stream_buffers:
            lines.append("        const geometry_t *const coordinate_components[DIM] = {%s};" % ", ".join(_component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "",
                    "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    "            const int stream_shape = %s;"
                    % (
                        "STREAM_SHAPE_ORDER[shape]"
                        if tuple(stream_shape_order) != tuple(range(n_nodes))
                        else "shape"
                    ),
                    "            for (int d = 0; d < DIM; ++d) {",
                    *source_builder.simd_lines(),
                    "                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                    "                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[lane * N_SHAPE + stream_shape]];",
                    "                }",
                    "            }",
                    "        }",
                ]
            )
        else:
            lines.extend(["", *source_builder.simd_lines(), "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
            for shape in range(n_nodes):
                for d in range(dim):
                    stream = "%s%d" % (_component_name(d), shape)
                    lines.append(
                        "            block_%s[lane] = %s[ev[lane * N_SHAPE + %d]];"
                        % (stream, _component_name(d), shape)
                    )
            lines.append("        }")

    if compact_stream_buffers:
        lines.append("")
        lines.append("        const scalar_t *const u_components[DIM] = {%s};" % ", ".join("u%s" % _component_name(d) for d in range(dim)))
        lines.append("        const scalar_t *const h_components[DIM] = {%s};" % ", ".join("h%s" % _component_name(d) for d in range(dim)))
        lines.extend(
            [
                "        const scalar_t *block_u_streams[N_SHAPE * DIM];",
                "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                "            block_u_streams[stream] = block_u_data[stream];",
                "        }",
            ]
        )
        lines.extend(
            [
                "",
                "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "            const int stream_shape = %s;"
                % (
                    "STREAM_SHAPE_ORDER[shape]"
                    if tuple(stream_shape_order) != tuple(range(n_nodes))
                    else "shape"
                ),
                "            for (int d = 0; d < DIM; ++d) {",
                *source_builder.simd_lines(),
                "                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                "                    const idx_t node = ev[lane * N_SHAPE + stream_shape];",
                "                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];",
                "                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];",
                "                }",
                "            }",
                "        }",
            ]
        )
    else:
        lines.append(
            "        const scalar_t *const block_u_streams[N_SHAPE * %d] = {%s};"
            % (
                dim,
                ", ".join(
                    "block_%s" % stream
                    for stream in streams_in_shape_order(
                        _field_stream_names("u", dim, n_nodes),
                        dim,
                        stream_shape_order,
                    )
                ),
            )
        )
        lines.extend(["", *source_builder.simd_lines(), "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                lines.append(
                    "            block_u%s%d_base[lane] = u%s[ev[lane * N_SHAPE + %d] * u_stride];"
                    % (component, shape, component, shape)
                )
                lines.append(
                    "            block_h%s%d[lane] = h%s[ev[lane * N_SHAPE + %d] * h_stride];"
                    % (component, shape, component, shape)
                )
        lines.append("        }")

    if geometry_mode == "isoparametric" and not use_tensor_product_geometry:
        lines.append("")
        if compact_stream_buffers:
            lines.extend(
                [
                    "        const scalar_t *block_coordinate_streams[N_SHAPE * DIM];",
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "            block_coordinate_streams[stream] = block_coordinate_data[stream];",
                    "        }",
                ]
            )
        else:
            lines.append(
                "        const scalar_t *const block_coordinate_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        "block_%s" % stream
                        for stream in _coordinate_stream_names(dim, n_nodes)
                    ),
                )
            )

    if geometry_mode == "isoparametric" and use_tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_nodes,
                n_qp=quadrature_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams=(
                    "block_coordinate_data"
                    if compact_stream_buffers
                    else tensor_product_ordered_coordinate_streams(
                        dim,
                        n_nodes,
                        _coordinate_stream_names(dim, n_nodes),
                        lambda stream: "block_%s" % stream,
                        shape_order=_tensor_product_stream_shape_order(
                            quadrature_rule,
                            dim,
                            n_nodes,
                        ),
                    )
                ),
                adjugate_target=lambda component, index: (
                    "block_jacobian_adjugate%d[%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_jacobian_determinant0[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_jacobian_adjugate%d" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_jacobian_determinant0",
                shape_name=tensor_shape_name,
                grad_name=tensor_grad_name,
            )
        )
    elif geometry_mode == "isoparametric":
        lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
        if use_tensor_product_geometry:
            lines.extend(_tensor_product_q_index_lines(dim, "            "))
        lines.extend(
            _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_geometry,
                use_reference_gradient_vectors,
                reference_inputs,
                q_major=True,
                reference_prefix=reference_prefix,
                source_builder=source_builder,
            )
        )
        lines.append("        }")

    call_args = ["nelems"]
    call_args.append("0" if geometry_mode == "affine" else "VECTOR_SIZE")
    if geometry_mode == "affine":
        call_args.extend(
            "g_%s + evbegin" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        call_args.extend(
            "block_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    if use_tensor_product_reference:
        call_args.extend((tensor_shape_name, tensor_grad_name))
    elif use_reference_gradient_vectors:
        call_args.extend(
            "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
            for component in range(dim)
        )
    else:
        call_args.extend("%s%s" % (reference_prefix, array_input.name) for array_input in reference_inputs)
    call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
    call_args.extend(("mu", "lmbda"))
    if use_stream_arrays:
        call_args.append("block_u_streams")
    else:
        call_args.extend("block_%s" % stream for stream in _field_stream_names("u", dim, n_nodes))
    call_args.append("block_value")

    lines.extend(
        [
            "",
            "        for (int step = 0; step < nsteps; ++step) {",
            "            const scalar_t alpha = steps[step];",
        ]
    )
    if compact_stream_buffers:
        lines.extend(
            [
                "            for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "                for (int d = 0; d < DIM; ++d) {",
                *source_builder.simd_lines(),
                "                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
                "                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];",
                "                    }",
                "                }",
                "            }",
            ]
        )
    else:
        lines.extend([*source_builder.simd_lines(), "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                lines.append(
                    "                block_u%s%d[lane] = block_u%s%d_base[lane] + alpha * block_h%s%d[lane];"
                    % (component, shape, component, shape, component, shape)
                )
        lines.append("            }")
    lines.extend(
        [
            *source_builder.simd_lines(),
            "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "                block_value[lane] = scalar_t(0);",
            "            }",
            "",
            "            %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_name, ", ".join(call_args)),
            "",
            *source_builder.simd_lines(),
            "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];",
            "            }",
            "        }",
            "    }",
            "",
            *source_builder.success_return_lines(),
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )

    wrapper_args = tuple(_cpp_argument_name(param) for param in wrapper_params)
    for public_name, scalar_type in (
        (function_name, "double"),
        ("%s_float" % function_name, "float"),
    ):
        concrete_params = _sfem_soa_concrete_scalar_params(wrapper_params, scalar_type)
        lines.append('extern "C" int %s(' % public_name)
        for idx, param in enumerate(concrete_params):
            comma = "," if idx + 1 < len(concrete_params) else ""
            lines.append("        %s%s" % (param, comma))
        lines.extend(
            [
                ") {",
                *source_builder.wrapper_call_lines(
                    implementation_name,
                    scalar_type,
                    ", geom_t" if geometry_mode == "isoparametric" else "",
                    wrapper_args,
                ),
                "}",
                "",
            ]
        )
    return lines


def _field_stream_names(prefix, dim, n_nodes):
    return tuple(
        "%s%s%d" % (prefix, _component_name(d), node)
        for node in range(n_nodes)
        for d in range(dim)
    )


def _coordinate_stream_names(dim, n_nodes):
    return tuple(
        "%s%d" % (_component_name(d), node)
        for node in range(n_nodes)
        for d in range(dim)
    )


def _soa_array_stream_names(array_input):
    return tuple("%s%d" % (array_input.name, i) for i in range(array_input.size))


def _sfem_soa_element_inputs(array_inputs):
    return tuple(array_input for array_input in array_inputs if not array_input.is_reference_qp_shape)


def _sfem_soa_reference_inputs(array_inputs):
    return tuple(array_input for array_input in array_inputs if array_input.is_reference_qp_shape)


def _sfem_soa_reference_param_name(array_input):
    return "%s_data" % array_input.name


def _sfem_soa_has_adjugate_geometry_inputs(array_inputs, dim):
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    names_and_sizes = {(array_input.name, array_input.size) for array_input in element_inputs}
    return (
        ("jacobian_adjugate", dim * dim) in names_and_sizes
        and ("jacobian_determinant", 1) in names_and_sizes
    )


def _sfem_soa_diagnostics_header():
    struct_name = _sfem_soa_diagnostics_struct_name()
    return [
        "#ifndef SFEM_CODEGEN_KERNEL_DIAGNOSTICS_HPP",
        "#define SFEM_CODEGEN_KERNEL_DIAGNOSTICS_HPP",
        "",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "",
        "#ifndef SFEM_INLINE",
        "#define SFEM_INLINE inline",
        "#endif",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
        "struct %s {" % struct_name,
        "    const char *kernel_name;",
        "    const char *element_type;",
        "    int dim;",
        "    int n_qp;",
        "    int n_shape;",
        "    int vector_size;",
        "    int quadrature_order;",
        "    long add_instructions_per_qp_lane;",
        "    long mul_instructions_per_qp_lane;",
        "    long div_instructions_per_qp_lane;",
        "    long sqrt_instructions_per_qp_lane;",
        "    long pow_instructions_per_qp_lane;",
        "    long exp_instructions_per_qp_lane;",
        "    long log_instructions_per_qp_lane;",
        "    long trig_instructions_per_qp_lane;",
        "    long load_instructions_per_qp_lane;",
        "    long store_instructions_per_qp_lane;",
        "    long flops_per_qp_lane;",
        "    long temporaries;",
        "    long estimated_registers;",
        "    int geometry_streams;",
        "    int reference_scalars;",
        "    int quadrature_weight_scalars;",
        "    int material_scalars;",
        "    int u_streams;",
        "    int h_streams;",
        "    int output_streams;",
        "    int output_reads_per_element;",
        "    int output_writes_per_element;",
        "    double add_cpi;",
        "    double mul_cpi;",
        "    double div_cpi;",
        "    double sqrt_cpi;",
        "    double pow_cpi;",
        "    double exp_cpi;",
        "    double log_cpi;",
        "    double trig_cpi;",
        "    double load_cpi;",
        "    double store_cpi;",
        "};",
        "",
        "static SFEM_INLINE double %s_total_flops(" % struct_name,
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements) {",
        "    const double n = nelements > 0 ? (double)nelements : 0.0;",
        "    return n * (double)d->n_qp * (double)d->flops_per_qp_lane;",
        "}",
        "",
        "static SFEM_INLINE size_t %s_total_bytes(" % struct_name,
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    (void)accumulator_bytes;",
        "    const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "    const size_t geometry_bytes = n * (size_t)d->n_qp * (size_t)d->geometry_streams * scalar_bytes;",
        "    const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "    const size_t output_bytes = n * (size_t)d->output_streams * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "    const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "    return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static SFEM_INLINE double %s_arithmetic_intensity(" % struct_name,
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const size_t bytes = %s_total_bytes(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "    return bytes ? %s_total_flops(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "static SFEM_INLINE void %s_print_rate(" % struct_name,
        "        const char *const name,",
        "        const %s *const d," % struct_name,
        "        const double elapsed,",
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t ndofs,",
        "        const int repeat,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const double seconds_per_call = repeat > 0 ? elapsed / (double)repeat : 0.0;",
        "    const double element_rate = seconds_per_call > 0.0 ? 1e-6 * (double)nelements / seconds_per_call : 0.0;",
        "    const double dof_rate = seconds_per_call > 0.0 ? 1e-6 * (double)ndofs / seconds_per_call : 0.0;",
        "    const double ai = %s_arithmetic_intensity(" % struct_name,
        "            d, nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "    const double gflops = seconds_per_call > 0.0",
        "            ? 1e-9 * %s_total_flops(d, nelements) / seconds_per_call" % struct_name,
        "            : 0.0;",
        '    printf("%-72s %12.6e %16.3f %13.3f %10.3f %13.3f\\n",',
        "           name ? name : d->kernel_name,",
        "           seconds_per_call, element_rate, dof_rate, ai, gflops);",
        "}",
        "",
        "} // namespace codegen",
        "} // namespace sfem",
        "",
        "#endif",
    ]


def _sfem_soa_diagnostic_print_wrapper_lines(
    function_name,
    variable_name,
    scalar_type,
):
    suffix = "" if scalar_type == "double" else "_float"
    public_name = "%s%s_print_rate" % (function_name, suffix)
    return [
        'extern "C" void %s(' % public_name,
        "        const double elapsed,",
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t ndofs,",
        "        const int repeat) {",
        "    sfem::codegen::KernelDiagnostics_print_rate(",
        '            "%s%s",' % (function_name, suffix),
        "            &sfem::codegen::%s," % variable_name,
        "            elapsed, nelements, ndofs, repeat,",
        "            sizeof(%s), sizeof(%s), sizeof(%s));"
        % (scalar_type, scalar_type, scalar_type),
        "}",
    ]


def _sfem_soa_diagnostics_lines(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    array_inputs,
    quadrature_rule,
    basis_family,
):
    public_name = _sfem_soa_public_function_name(prefix, form.name, quadrature_rule)
    struct_name = _sfem_soa_diagnostics_struct_name()
    variable_name = "%s_diagnostics_data" % public_name
    if form.expression_graph is not None:
        cost = form.expression_graph.cost
    elif form.weak_form is not None:
        diagnostic_graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, form.weak_form.diagnostic_expressions(form.has_direction))
            .build_graph(
                data_symbols=form.weak_form.deformation_gradient,
                temporary_prefix="weak_diag_tmp",
            )
        )
        cost = diagnostic_graph.cost
    else:
        cost = ExpressionCost()
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    geometry_streams = sum(array_input.size for array_input in element_inputs)
    if quadrature_rule is not None and str(basis_family) == "tensor_product":
        reference_scalars = (
            len(quadrature_rule.tensor_product_shape_values_1d)
            + len(quadrature_rule.tensor_product_shape_gradients_1d)
        )
        quadrature_weight_scalars = len(quadrature_rule.tensor_product_weights_1d)
    else:
        reference_scalars = sum(array_input.size for array_input in reference_inputs)
        quadrature_weight_scalars = n_qp if quadrature_rule is not None else 1
    output_streams = len(_output_stream_names(form, dim, n_nodes))
    output_reads = output_streams if form.output_mode == "accumulate" else 0
    output_writes = output_streams
    u_streams = dim * n_nodes
    h_streams = dim * n_nodes if form.has_direction else 0
    element_type = quadrature_rule.element_type if quadrature_rule is not None else "GENERIC"
    quadrature_order = quadrature_rule.order if quadrature_rule is not None else 0
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "static const %s %s = {" % (struct_name, variable_name),
        '    "%s",' % public_name,
        '    "%s",' % element_type,
        "    %d," % dim,
        "    %d," % n_qp,
        "    %d," % n_nodes,
        "    %d," % vector_size,
        "    %d," % quadrature_order,
        "    %d," % cost.adds,
        "    %d," % cost.muls,
        "    %d," % cost.divs,
        "    %d," % cost.sqrts,
        "    %d," % cost.pows,
        "    %d," % cost.exps,
        "    %d," % cost.logs,
        "    %d," % cost.trigs,
        "    %d," % cost.loads,
        "    %d," % cost.stores,
        "    %d," % cost.flops,
        "    %d," % cost.temporaries,
        "    %d," % cost.estimated_registers,
        "    %d," % geometry_streams,
        "    %d," % reference_scalars,
        "    %d," % quadrature_weight_scalars,
        "    2,",
        "    %d," % u_streams,
        "    %d," % h_streams,
        "    %d," % output_streams,
        "    %d," % output_reads,
        "    %d," % output_writes,
        "    1.0,",
        "    1.0,",
        "    8.0,",
        "    12.0,",
        "    16.0,",
        "    20.0,",
        "    20.0,",
        "    24.0,",
        "    1.0,",
        "    1.0",
        "};",
        "",
        "} // namespace codegen",
        "} // namespace sfem",
        "",
        'extern "C" const sfem::codegen::%s *%s_diagnostics(void) {' % (struct_name, public_name),
        "    return &sfem::codegen::%s;" % variable_name,
        "}",
        "",
        'extern "C" double %s_arithmetic_intensity(' % public_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    return sfem::codegen::%s_arithmetic_intensity(&sfem::codegen::%s, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % (struct_name, variable_name),
        "}",
    ]
    function_names = [public_name]
    if quadrature_rule is not None:
        function_names.extend(
            (
                _sfem_soa_mesh_public_function_name(
                    prefix,
                    form.name,
                    quadrature_rule,
                    "affine",
                ),
                _sfem_soa_mesh_public_function_name(
                    prefix,
                    form.name,
                    quadrature_rule,
                    "isoparametric",
                ),
            )
        )
    for function_name in function_names:
        for scalar_type in ("double", "float"):
            lines.append("")
            lines.extend(
                _sfem_soa_diagnostic_print_wrapper_lines(
                    function_name,
                    variable_name,
                    scalar_type,
                )
            )
    return lines


def _sfem_soa_diagnostics_struct_name():
    return "KernelDiagnostics"


def _validate_sfem_soa_quadrature_rule(quadrature_rule, dim, n_nodes, n_qp, array_inputs):
    if quadrature_rule.dim != dim:
        raise ValueError(
            "quadrature rule dimension %d does not match dim=%d"
            % (quadrature_rule.dim, dim)
        )
    if quadrature_rule.n_shape != n_nodes:
        raise ValueError(
            "quadrature rule n_shape %d does not match n_nodes=%d"
            % (quadrature_rule.n_shape, n_nodes)
        )
    if quadrature_rule.n_qp != n_qp:
        raise ValueError(
            "quadrature rule n_qp %d does not match n_qp=%d"
            % (quadrature_rule.n_qp, n_qp)
        )
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    if len(reference_inputs) != 1 or reference_inputs[0].name != "grad_ref":
        raise ValueError("element-specialized wrappers currently require one grad_ref reference input")
    grad_ref = reference_inputs[0]
    if grad_ref.components != dim or grad_ref.n_shape != n_nodes or grad_ref.n_qp != n_qp:
        raise ValueError("grad_ref reference input does not match quadrature rule")


def _sfem_soa_specialized_wrapper_arguments(
    prefix,
    quadrature_rule,
    wrapper_params,
    reference_inputs,
    use_reference_gradient_vectors=False,
    basis_family=None,
):
    arguments = [_cpp_argument_name(param) for param in wrapper_params]
    if basis_family is None:
        raise ValueError("basis family must be provided by the emission plan")
    if str(basis_family) == "tensor_product":
        offset = 1 + _sfem_soa_element_stream_count_from_params(wrapper_params)
        arguments.insert(
            offset,
            quadrature_reference_accessor(prefix, "isoparametric", "shape_1d", "real_t"),
        )
        arguments.insert(
            offset + 1,
            quadrature_reference_accessor(prefix, "isoparametric", "grad_1d", "real_t"),
        )
        arguments.insert(
            offset + 2,
            quadrature_reference_accessor(prefix, "isoparametric", "q_weight_1d", "real_t"),
        )
        return tuple(arguments)
    if use_reference_gradient_vectors:
        offset = 1 + _sfem_soa_element_stream_count_from_params(wrapper_params)
        for component in range(quadrature_rule.dim):
            arguments.insert(
                offset + component,
                quadrature_reference_accessor(
                    prefix,
                    "isoparametric",
                    _sfem_reference_gradient_vector_name(component),
                    "real_t",
                ),
            )
        arguments.insert(
            offset + quadrature_rule.dim,
            quadrature_reference_accessor(prefix, "isoparametric", "q_weight", "real_t"),
        )
        return tuple(arguments)
    for array_input in reference_inputs:
        arguments.insert(
            1 + _sfem_soa_element_stream_count_from_params(wrapper_params),
            quadrature_reference_accessor(prefix, "isoparametric", array_input.name, "real_t"),
        )
    arguments.insert(
        1 + _sfem_soa_element_stream_count_from_params(wrapper_params) + len(reference_inputs),
        quadrature_reference_accessor(prefix, "isoparametric", "q_weight", "real_t"),
    )
    return tuple(arguments)


def _sfem_soa_element_stream_count_from_params(params):
    count = 0
    for param in params[1:]:
        name = _cpp_argument_name(param)
        if name in ("mu", "lmbda"):
            break
        count += 1
    return count


def _sfem_soa_public_wrapper_params(params):
    return tuple(param.replace("scalar_t", "real_t") for param in params)


def _sfem_soa_concrete_scalar_params(params, scalar_type):
    return tuple(
        param.replace("scalar_t", scalar_type)
        .replace("real_t", scalar_type)
        for param in params
    )


def _sfem_soa_mesh_reference_alias_lines(
    prefix,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    geometry_mode,
):
    lines = []
    reference_prefix = "%s_" % geometry_mode
    if use_tensor_product_reference:
        for name in ("shape_1d", "grad_1d", "q_weight_1d"):
            lines.append(
                "    const scalar_t *const %s = %s;"
                % (
                    "%s%s" % (reference_prefix, name),
                    quadrature_reference_accessor(prefix, geometry_mode, name),
                )
            )
        return lines
    if use_reference_gradient_vectors:
        for component in range(quadrature_rule.dim):
            reference_name = _sfem_reference_gradient_vector_name(component)
            lines.append(
                "    const scalar_t *const %s%s = %s;"
                % (
                    reference_prefix,
                    reference_name,
                    quadrature_reference_accessor(prefix, geometry_mode, reference_name),
                )
            )
        lines.append(
            "    const scalar_t *const %sq_weight = %s;"
            % (
                reference_prefix,
                quadrature_reference_accessor(prefix, geometry_mode, "q_weight"),
            )
        )
        return lines
    for array_input in reference_inputs:
        if array_input.name != "grad_ref":
            raise ValueError("mesh reference aliases require grad_ref")
        lines.append(
            "    const scalar_t *const %s%s = %s;"
            % (
                reference_prefix,
                array_input.name,
                quadrature_reference_accessor(prefix, geometry_mode, array_input.name),
            )
        )
    lines.append(
        "    const scalar_t *const %sq_weight = %s;"
        % (
            reference_prefix,
            quadrature_reference_accessor(prefix, geometry_mode, "q_weight"),
        )
    )
    return lines


def _sfem_soa_public_function_name(prefix, form_name, quadrature_rule):
    if quadrature_rule is None:
        return "%s_%s_soa" % (prefix, form_name)
    return "%s_%s_%s_soa" % (
        prefix,
        quadrature_rule.element_type.lower(),
        form_name,
    )


def _sfem_soa_isoparametric_public_function_name(prefix, form_name, quadrature_rule):
    if quadrature_rule is None:
        return "%s_%s_isoparametric_soa" % (prefix, form_name)
    return "%s_%s_%s_isoparametric_soa" % (
        prefix,
        quadrature_rule.element_type.lower(),
        form_name,
    )


def _sfem_soa_mesh_public_function_name(prefix, form_name, quadrature_rule, geometry_mode):
    if quadrature_rule is None:
        return "%s_%s_%s_mesh_soa" % (prefix, form_name, geometry_mode)
    return "%s_%s_%s_%s_mesh_soa" % (
        prefix,
        quadrature_rule.element_type.lower(),
        form_name,
        geometry_mode,
    )


def _cpp_scalar_initializer_list(values, scalar_type="real_t"):
    return ", ".join(_cpp_scalar_literal(value, scalar_type) for value in values)


def _cpp_scalar_literal(value, scalar_type="real_t"):
    value = float(value)
    if value == 0.0:
        return "%s(0)" % scalar_type
    return "%s(%.17g)" % (scalar_type, value)


def _output_stream_names(form, dim, n_nodes):
    if form.weak_form is not None:
        if form.name == "objective":
            return ("value",)
        return tuple(
            "out%s%d" % (_component_name(d), node)
            for node in range(n_nodes)
            for d in range(dim)
        )
    output_count = len(form.expression_graph.evaluation_plan.outputs)
    if output_count == 1:
        return ("value",)
    return tuple(
        "out%s%d" % (_component_name(d), node)
        for node in range(n_nodes)
        for d in range(dim)
    )
