import sympy as sp

from codegen.framework.ir.kernel_ast import (
    LoopKind,
    LoopNode,
    ScatterNode,
    add_assign_increment,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
)
from codegen.framework.emitters.ast_printer import CLikeKernelASTPrinter, render_kernel_ast_lines
from codegen.framework.symbolic.core import (
    ExpressionCost,
    ExpressionRole,
    GeneratedKernelFile,
    KernelExpressions,
    SfemElementQuadratureRule,
    SfemSoAElementSpecialization,
    _component_name,
    _cpp_argument_name,
    _cpp_macro_name,
    _prune_dead_cse_intermediates,
    _sfem_ccode,
    _sfem_math_header_source,
    _validate_diagnostics_plan_names,
    isoparametric_adjugate_call_lines,
    isoparametric_adjugate_stream_array_lines,
    quadrature_reference_accessor,
    quadrature_reference_struct_lines,
    sfem_element_quadrature_rule,
    sfem_mesh_reference_data,
    sfem_soa_element_specialization,
    sfem_soa_reference_input,
    sfem_tensor_product_hex_uses_cartesian_ordering,
    sfem_tensor_product_quad_uses_cartesian_ordering,
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_coordinate_gradient_lines,
    tensor_product_current_q_isoparametric_geometry_lines,
    tensor_product_gradient_isoparametric_geometry_lines,
    validate_reference_data_plan,
)
from codegen.framework.plans.form_transformations import (
    constant_p1_simplex_reference_gradients,
)


def _default_openmp_energy_source_builder():
    from codegen.framework.emitters.energy import OpenMPEnergySoASourceBuilder

    return OpenMPEnergySoASourceBuilder()


def _sfem_packed_thread_scratch_header_source():
    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstddef>",
            "#include <cstdlib>",
            "",
            "#ifndef SFEM_RESTRICT",
            "#define SFEM_RESTRICT __restrict__",
            "#endif",
            "",
            "#ifndef SFEM_INLINE",
            "#define SFEM_INLINE inline",
            "#endif",
            "",
            "namespace sfem {",
            "namespace codegen {",
            "",
            "template <typename T>",
            "struct ThreadScratchBuffer {",
            "    T *data{nullptr};",
            "    size_t capacity{0};",
            "",
            "    ~ThreadScratchBuffer() { std::free(data); }",
            "",
            "    T *ensure(const size_t size) {",
            "        if (capacity < size) {",
            "            std::free(data);",
            "            data = static_cast<T *>(std::calloc(size, sizeof(T)));",
            "            capacity = data ? size : 0;",
            "        }",
            "        return data;",
            "    }",
            "};",
            "",
            "template <typename T>",
            "SFEM_INLINE T *thread_scratch(const int slot, const size_t size) {",
            "    static thread_local ThreadScratchBuffer<T> buffers[4];",
            "    return buffers[slot].ensure(size);",
            "}",
            "",
            "template <typename T>",
            "SFEM_INLINE void prealloc_thread_scratch(const int slot, const size_t size) {",
            "#pragma omp parallel",
            "    {",
            "        (void)thread_scratch<T>(slot, size);",
            "    }",
            "}",
            "",
            "}  // namespace codegen",
            "}  // namespace sfem",
            "",
        ]
    )


def _work_item_index(source_builder):
    if hasattr(source_builder, "work_item_index"):
        return source_builder.work_item_index()
    return "lane"


def _work_item_loop_lines(source_builder, indent):
    if hasattr(source_builder, "work_item_loop_lines"):
        target = getattr(source_builder, "target", None)
        if target is not None and hasattr(target, "loop_lowering_policy"):
            policy = target.loop_lowering_policy()
            if policy.emits_lane_loop:
                pragma = target.vectorize_pragma() if policy.vectorize_lane_loop else None
                printer = CLikeKernelASTPrinter(vectorize_pragma=pragma or "")
                lane_iterator = iterator(policy.lane_index, policy.lane_index_type)
                return tuple(
                    "%s%s" % (indent, line)
                    for line in render_kernel_ast_lines(
                        "work_item_loop_header",
                        (
                            LoopNode(
                                LoopKind.SIMD,
                                lane_iterator,
                                iteration_range(0, expr_ref("nelems", "tile_extent")),
                                pre_increment(lane_iterator),
                                vectorized=bool(pragma),
                            ),
                        ),
                        printer=printer,
                    )
                )
            return ("%s{" % indent,)
        return source_builder.work_item_loop_lines(indent)
    index = _work_item_index(source_builder)
    simd_lines = tuple(source_builder.simd_lines())
    lane_iterator = iterator(index, "int")
    return tuple(
        "%s%s"
        % (
            indent,
            line,
        )
        for line in render_kernel_ast_lines(
            "work_item_loop_header",
            (
                LoopNode(
                    LoopKind.SIMD,
                    lane_iterator,
                    iteration_range(0, expr_ref("nelems", "tile_extent")),
                    pre_increment(lane_iterator),
                    vectorized=bool(simd_lines),
                ),
            ),
            printer=CLikeKernelASTPrinter(
                vectorize_pragma=simd_lines[0] if simd_lines else ""
            ),
        )
    )


def _sfem_soa_affine_geometry_stream_lines(
    source_builder,
    element_inputs,
    indent,
    geometry_scalar_type="jacobian_t",
):
    lines = []
    for array_input in element_inputs:
        for stream in _soa_array_stream_names(array_input):
            lines.extend(
                [
                    "%sscalar_t block_%s_data[VECTOR_SIZE];" % (indent, stream),
                    "%sconst scalar_t *const block_%s = affine_geometry_stream<scalar_t, %s, VECTOR_SIZE>("
                    % (indent, stream, geometry_scalar_type),
                    "%s        nelems, g_%s + evbegin, block_%s_data, std::is_same<%s, scalar_t>());"
                    % (indent, stream, stream, geometry_scalar_type),
                ]
            )
    return lines


def _affine_geometry_stream_helper_lines(source_builder):
    inline_qualifier = source_builder.inline_qualifier()
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>",
        "%s const scalar_t *affine_geometry_stream(" % inline_qualifier,
        "        const int,",
        "        const jacobian_t *const SFEM_RESTRICT source,",
        "        scalar_t *const SFEM_RESTRICT,",
        "        std::true_type) {",
        "    return source;",
        "}",
        "",
        "template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>",
        "%s const scalar_t *affine_geometry_stream(" % inline_qualifier,
        "        const int nelems,",
        "        const jacobian_t *const SFEM_RESTRICT source,",
        "        scalar_t *const SFEM_RESTRICT converted,",
        "        std::false_type) {",
    ]
    if _emits_vector_lane_loop(source_builder):
        lines.extend("    %s" % line for line in source_builder.simd_lines())
        lines.extend(
            [
                "    for (int lane = 0; lane < nelems; ++lane) {",
                "        converted[lane] = scalar_t(source[lane]);",
                "    }",
            ]
        )
    else:
        index = _work_item_index(source_builder)
        lines.extend(
            [
                "    (void)nelems;",
                "    converted[%s] = scalar_t(source[%s]);" % (index, index),
            ]
        )
    lines.extend(
        [
            "    return converted;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
        ]
    )
    return lines


def _emits_vector_lane_loop(source_builder):
    target = getattr(source_builder, "target", None)
    if target is not None and hasattr(target, "loop_lowering_policy"):
        policy = target.loop_lowering_policy()
        return bool(policy.emits_lane_loop and policy.vectorize_lane_loop)
    return True


def _scatter_add_lines(source_builder, pointer, node_expr, value_expr, indent):
    work_item = _work_item_index(source_builder)
    if not _emits_vector_lane_loop(source_builder):
        return source_builder.scatter_add_lines(
            "%s[%s]" % (pointer, node_expr % work_item),
            value_expr % work_item,
            indent,
        )

    target = getattr(source_builder, "target", None)
    atomic_pragma = (
        target.atomic_update_pragma()
        if target is not None and hasattr(target, "atomic_update_pragma")
        else None
    )
    lines = [
        "%s{" % indent,
        *(
            "%s    %s" % (indent, line)
            for line in render_kernel_ast_lines(
                "scatter_loop_header",
                (
                    LoopNode(
                        LoopKind.SCATTER,
                        iterator("scatter", "int"),
                        iteration_range(0, expr_ref("nelems", "tile_extent")),
                        pre_increment(iterator("scatter", "int")),
                    ),
                ),
            )
        ),
    ]
    if atomic_pragma is not None:
        lines.extend(
            "%s        %s" % (indent, line)
            for line in render_kernel_ast_lines(
                "scatter_add",
                (
                    ScatterNode(
                        expr_ref("%s[%s]" % (pointer, node_expr % "scatter"), "scatter_target"),
                        expr_ref(value_expr % "scatter", "scatter_value"),
                        "+=",
                        atomic=True,
                    ),
                ),
                printer=CLikeKernelASTPrinter(atomic_update_pragma=atomic_pragma),
            )
        )
    else:
        lines.extend(
            source_builder.scatter_add_lines(
                "%s[%s]" % (pointer, node_expr % "scatter"),
                value_expr % "scatter",
                "%s        " % indent,
            )
        )
    lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    return tuple(lines)


def _work_item_name(source_builder, name, component):
    if hasattr(source_builder, "work_item_name"):
        return source_builder.work_item_name(name, component)
    return "%s_%s%d" % (name, _work_item_index(source_builder), component)


def _diagnostic_work_item(source_builder):
    if hasattr(source_builder, "diagnostic_work_item"):
        return source_builder.diagnostic_work_item()
    return _work_item_index(source_builder)


def _inline_qualifier(source_builder):
    if hasattr(source_builder, "inline_qualifier"):
        return source_builder.inline_qualifier()
    return "SFEM_INLINE"


def _defines_sfem_inline(source_builder):
    return _inline_qualifier(source_builder) == "SFEM_INLINE"


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
    matrix_format_plan=None,
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
    local_name = source_builder.header_name("%s_local" % local_prefix)
    math_name = source_builder.header_name("kernel_math")
    tensor_product_name = source_builder.header_name("tensor_product_kernels")
    geometry_name = source_builder.header_name("geometry_kernels")
    diagnostics_name = source_builder.header_name("kernel_diagnostics")
    hessian_name = source_builder.header_name("%s_hessian" % local_prefix)
    header_guard_suffix = source_builder.header_guard_suffix()
    operator_name = "%s_operator.%s" % (prefix, source_builder.operator_extension)
    emits_hessian_header = _sfem_soa_emits_hessian_header(
        forms,
        quadrature_rule,
        array_inputs,
        basis_family,
    )
    files = [
        GeneratedKernelFile(
            math_name,
            _sfem_math_header_source(
                header_guard_suffix,
                _inline_qualifier(source_builder),
                _defines_sfem_inline(source_builder),
            ),
        ),
        GeneratedKernelFile(
            geometry_name,
            source_builder.geometry_header_source(),
        ),
        GeneratedKernelFile(
            diagnostics_name,
            "\n".join(
                _sfem_soa_diagnostics_header(
                    _diagnostic_work_item(source_builder),
                    header_guard_suffix,
                    _inline_qualifier(source_builder),
                    _defines_sfem_inline(source_builder),
                )
            ),
        ),
    ]
    if getattr(source_builder, "operator_extension", "cpp") == "cpp":
        files.append(
            GeneratedKernelFile(
                "packed_thread_scratch.hpp",
                _sfem_packed_thread_scratch_header_source(),
            )
        )
    if source_builder.emits_tensor_product_header(basis_family):
        files.append(
            GeneratedKernelFile(
                tensor_product_name,
                source_builder.tensor_product_header_source(),
            )
        )
    if emits_hessian_header:
        files.append(
            GeneratedKernelFile(
                hessian_name,
                _sfem_soa_hessian_header(
                    forms,
                    local_prefix,
                    dim,
                    n_nodes,
                    array_inputs,
                    quadrature_rule,
                    basis_family,
                    math_name,
                    tensor_product_name,
                    source_builder,
                ),
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
                "%s_element.hpp" % prefix,
                _sfem_soa_element_api_header(
                    forms,
                    prefix,
                    dim,
                    n_nodes,
                    n_qp,
                    vector_size,
                    local_prefix,
                    local_name,
                    geometry_name,
                    array_inputs,
                    quadrature_rule,
                    basis_family,
                    use_shared_weak_local,
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
                hessian_name if emits_hessian_header else None,
                geometry_name,
                diagnostics_name,
                array_inputs,
                quadrature_rule,
                affine_quadrature_rule,
                basis_family,
                geometry_family,
                use_shared_weak_local,
                matrix_format_plan,
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
    matrix_format_plan=None,
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
        matrix_format_plan=matrix_format_plan,
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
    guard = "%s_LOCAL_%s" % (_cpp_macro_name(prefix), source_builder.header_guard_suffix())
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
        specialized = _constant_p1_specialized_local(
            prefix,
            quadrature_rule,
        )
        specialized_prefix = specialized[0] if specialized is not None else None
        specialized_rule = specialized[1] if specialized is not None else None
        if specialized_prefix is not None and form.weak_form is not None:
            lines.extend(
                _sfem_soa_block_function(
                    form,
                    prefix,
                    dim,
                    n_nodes,
                    array_inputs,
                    specialized_rule,
                    basis_family,
                    use_shared_weak_local,
                    source_builder,
                    function_name="%s_%s_block" % (specialized_prefix, form.name),
                    constant_p1_gradient_expansion=True,
                )
            )
            lines.append("")

    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _sfem_soa_emits_hessian_header(
    forms,
    quadrature_rule,
    array_inputs,
    basis_family,
):
    if quadrature_rule is None:
        return False
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    return any(
        _sfem_soa_direct_hessian_matrix_assembly_available(
            form,
            quadrature_rule,
            reference_inputs,
            basis_family,
        )
        for form in forms
    )


def _sfem_soa_direct_hessian_function_name(
    local_prefix,
    use_tensor_product_reference,
):
    family = "tensor_product" if use_tensor_product_reference else "reference"
    return "%s_direct_hessian_%s_element_matrix" % (local_prefix, family)


def _sfem_soa_hessian_header(
    forms,
    prefix,
    dim,
    n_nodes,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    math_name="kernel_math.hpp",
    tensor_product_name="tensor_product_kernels.hpp",
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    guard = "%s_HESSIAN_%s" % (
        _cpp_macro_name(prefix),
        source_builder.header_guard_suffix(),
    )
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

    emitted = set()
    for form in forms:
        if not _sfem_soa_direct_hessian_matrix_assembly_available(
            form,
            quadrature_rule,
            reference_inputs,
            basis_family,
        ):
            continue
        name = _sfem_soa_direct_hessian_function_name(
            prefix,
            use_tensor_product_reference,
        )
        if name in emitted:
            continue
        emitted.add(name)
        lines.extend(
            _sfem_soa_direct_hessian_element_matrix_function(
                form,
                name,
                dim,
                quadrature_rule,
                reference_inputs,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                source_builder,
            )
        )
        lines.append("")

    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _sfem_soa_direct_hessian_element_matrix_function(
    form,
    name,
    dim,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    source_builder,
):
    params = [
        *(
            "const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate%d" % component
            for component in range(dim * dim)
        ),
        "const scalar_t *const SFEM_RESTRICT block_jacobian_determinant0",
    ]
    if use_tensor_product_reference:
        params.extend(
            (
                "const scalar_t *const SFEM_RESTRICT shape_1d",
                "const scalar_t *const SFEM_RESTRICT grad_1d",
                "const scalar_t *const SFEM_RESTRICT q_weight_1d",
            )
        )
    elif use_reference_gradient_vectors:
        params.extend(_sfem_reference_gradient_vector_params(dim))
        params.append("const scalar_t *const SFEM_RESTRICT q_weight")
    else:
        params.extend(
            "const %s *const SFEM_RESTRICT %s"
            % (array_input.scalar_type, _sfem_soa_reference_param_name(array_input))
            for array_input in reference_inputs
        )
        params.append("const scalar_t *const SFEM_RESTRICT q_weight")
    params.extend(_form_material_parameter_declarations(form))
    params.append("scalar_t *const SFEM_RESTRICT element_matrix")

    lines = [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>",
        "static %s void %s(" % (_inline_qualifier(source_builder), name),
    ]
    for idx, param in enumerate(params):
        comma = "," if idx + 1 < len(params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static_assert(N_QP > 0, \"N_QP must be positive\");",
            "    static_assert(N_SHAPE > 0, \"N_SHAPE must be positive\");",
            "    static_assert(VECTOR_SIZE > 0, \"VECTOR_SIZE must be positive\");",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int NDOFS = DIM * N_SHAPE;",
        ]
    )
    if use_tensor_product_reference:
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
    lines.extend(
        _sfem_soa_direct_hessian_matrix_assembly_lines(
            form,
            dim,
            quadrature_rule,
            reference_inputs,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
            "",
            "    ",
            emit_tensor_product_static_constants=False,
        )
    )
    lines.append("}")
    return lines


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
    function_name=None,
    constant_p1_gradient_expansion=False,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
    name = function_name or "%s_%s_block" % (prefix, form.name)
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
    omit_reference_basis_inputs = (
        constant_p1_gradient_expansion
        and form.weak_form is not None
        and not use_tensor_product_reference
    )
    use_stream_arrays = use_shared_weak_local and form.weak_form is not None
    uses_current = _form_uses_current(form, default=True)
    uses_direction = _form_uses_direction(form, default=form.has_direction)
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )
    identity_stream_shape_order = tuple(stream_shape_order) == tuple(range(n_nodes))
    params = ["const int nelems"]
    if form.weak_form is not None:
        params.append("const ptrdiff_t geometry_stride")
    else:
        params.append("const int q")
    params.extend(
        "const %s *const SFEM_RESTRICT %s" % (array_input.scalar_type, stream)
        for array_input in element_inputs
        for stream in _soa_array_stream_names(array_input)
    )
    if omit_reference_basis_inputs:
        pass
    elif use_tensor_product_reference:
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
    params.extend(_form_material_parameter_declarations(form))
    if use_stream_arrays:
        if uses_current:
            params.extend(
                (
                    "const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * %d]" % dim,
                )
            )
        if uses_direction:
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
        if uses_current:
            params.extend(
                "const scalar_t *const SFEM_RESTRICT %s" % name
                for name in _field_stream_names("u", dim, n_nodes)
            )
        if uses_direction:
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
        "static %s void %s(" % (_inline_qualifier(source_builder), name),
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
    if form.weak_form is not None and not use_stream_arrays and uses_current:
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
    if form.weak_form is not None and not use_stream_arrays and uses_direction:
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
    if form.weak_form is not None and not use_stream_arrays and form.name != "objective":
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
        lines.extend(_work_item_loop_lines(source_builder, "    "))
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
                lines.append("        %s[%d] = %s[%s];" % (array_input.name, i, stream, work_item))
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
                    "            u[shape * %d + %d] = u_streams[shape * %d + %d][%s];"
                    % (dim, d, dim, d, work_item)
                )
                if form.has_direction:
                    lines.append(
                        "            du[shape * %d + %d] = h_streams[shape * %d + %d][%s];"
                        % (dim, d, dim, d, work_item)
                    )
            lines.append("        }")
        else:
            for node in range(n_nodes):
                for d in range(dim):
                    idx = node * dim + d
                    component = _component_name(d)
                    lines.append("        u[%d] = u%s%d[%s];" % (idx, component, node, work_item))
                    if form.has_direction:
                        lines.append("        du[%d] = h%s%d[%s];" % (idx, component, node, work_item))

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
            constant_p1_gradient_expansion=constant_p1_gradient_expansion,
        )
        lines.append("}")
        return lines

    output_count = len(form.expression_graph.evaluation_plan.outputs)
    lines.append("        scalar_t element_vector[%d];" % max(1, output_count))
    _append_sfem_soa_statement_lines(lines, form.expression_graph, "element_vector")
    _append_sfem_soa_output_lines(lines, form, dim, n_nodes, work_item)
    lines.extend(["    }", "}"])
    return lines


def _sfem_soa_direct_hessian_matrix_assembly_available(
    form,
    quadrature_rule,
    reference_inputs,
    basis_family,
):
    if form.name != "apply" or form.weak_form is None:
        return False
    if len(reference_inputs) != 1 or reference_inputs[0].name != "grad_ref":
        return False
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    return not _form_uses_current(form, default=True)


def _constant_p1_specialized_local(local_prefix, quadrature_rule):
    specialized_prefix = _constant_p1_specialized_local_prefix(
        local_prefix,
        quadrature_rule,
    )
    if specialized_prefix is not None:
        return specialized_prefix, quadrature_rule

    if quadrature_rule is None or not str(local_prefix).endswith("_simplex"):
        return None

    element_type = {2: "TRI3", 3: "TET4"}.get(int(getattr(quadrature_rule, "dim", 0)))
    if element_type is None:
        return None

    p1_specialization = sfem_soa_element_specialization(element_type)
    p1_prefix = _constant_p1_specialized_local_prefix(
        local_prefix,
        p1_specialization.quadrature_rule,
    )
    if p1_prefix is None:
        return None
    return p1_prefix, p1_specialization.quadrature_rule


def _constant_p1_specialized_local_prefix(local_prefix, quadrature_rule):
    if quadrature_rule is None:
        return None
    if constant_p1_simplex_reference_gradients(quadrature_rule) is None:
        return None
    element_type = str(getattr(quadrature_rule, "element_type", "")).lower()
    if element_type not in ("tri3", "tet4"):
        return None
    return "%s_%s" % (local_prefix, element_type)


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
    work_item = _work_item_index(source_builder)
    weak_form = form.weak_form
    uses_current = _form_uses_current(form, default=True)
    uses_direction = _form_uses_direction(form, default=form.has_direction)
    u_streams = "u_streams" if use_stream_arrays else "weak_u_streams"
    h_streams = "h_streams" if use_stream_arrays else "weak_h_streams"
    out_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
    block_extent = "N_QP * %d * VECTOR_SIZE" % (dim * dim)

    if uses_current:
        lines.append("    scalar_t grad_u_ref_q[%s];" % block_extent)
    if uses_direction:
        lines.append("    scalar_t grad_h_ref_q[%s];" % block_extent)
    if form.name != "objective":
        lines.append("    scalar_t loperand_q[%s];" % block_extent)

    for row in range(dim):
        output_offset = "%d * N_QP * %d * VECTOR_SIZE" % (row, dim)
        if uses_current:
            lines.append(
                "    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, %d>(nelems, shape_1d, grad_1d, %s, %d, &grad_u_ref_q[%s]);"
                % (dim, u_streams, row, output_offset)
            )
        if uses_direction:
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
    lines.extend(_work_item_loop_lines(source_builder, "        "))
    lines.extend(
        [
            "            const ptrdiff_t geometry_offset = q * geometry_stride + %s;" % work_item,
        ]
    )
    for component in range(dim * dim):
        lines.append(
            "            const scalar_t %s = jacobian_adjugate%d[geometry_offset];"
            % (_work_item_name(source_builder, "jacobian_adjugate", component), component)
        )
    lines.append(
        "            const scalar_t %s = jacobian_determinant0[geometry_offset];"
        % _work_item_name(source_builder, "jacobian_determinant", 0)
    )
    if uses_current:
        lines.append("            scalar_t grad_u_ref[%d];" % (dim * dim))
        for row in range(dim):
            for col in range(dim):
                component = row * dim + col
                lines.append(
                    "            grad_u_ref[%d] = grad_u_ref_q[((%d * N_QP + q) * %d + %d) * VECTOR_SIZE + %s];"
                    % (component, row, dim, col, work_item)
                )
    if uses_direction:
        lines.append("            scalar_t grad_h_ref[%d];" % (dim * dim))
        for row in range(dim):
            for col in range(dim):
                component = row * dim + col
                lines.append(
                    "            grad_h_ref[%d] = grad_h_ref_q[((%d * N_QP + q) * %d + %d) * VECTOR_SIZE + %s];"
                    % (component, row, dim, col, work_item)
                )

    def geometry_value(name, component):
        return _work_item_name(source_builder, name, component)

    if uses_current:
        lines.append("            scalar_t grad_u[%d];" % (dim * dim))
    if uses_direction:
        lines.append("            scalar_t trial_grad[%d];" % (dim * dim))

    lines.append(
        "            const scalar_t inv_jacobian_determinant = scalar_t(1) / %s;"
        % geometry_value("jacobian_determinant", 0)
    )
    for row in range(dim):
        for col in range(dim):
            if uses_current:
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
            if uses_direction:
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
            ["value[%s] %s" % (work_item, "+=" if form.output_mode == "accumulate" else "=")],
            "weak_obj_tmp",
            scale="qw * %s" % geometry_value("jacobian_determinant", 0),
        )
        lines.extend(["        }", "    }"])
        return

    material = _weak_form_material_expression(
        weak_form,
        form.name,
        deformation_gradient_substitutions,
        tuple(sp.symbols("trial_grad[%d]" % i) for i in range(dim * dim)),
    )
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
                "            loperand_q[((%d * N_QP + q) * %d + %d) * VECTOR_SIZE + %s] = loperand[%d];"
                % (row, dim, col, work_item, row * dim + col)
            )
    lines.extend(["        }", "    }"])
    for row in range(dim):
        lines.append(
            "    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, %d>(nelems, shape_1d, grad_1d, &loperand_q[%d * N_QP * %d * VECTOR_SIZE], %s, %d);"
            % (dim, row, dim, out_streams, row)
        )


def _scaled_cpp_term(factor, expression):
    factor = sp.sympify(factor)
    if factor == 1:
        return expression
    if factor == -1:
        return "-(%s)" % expression
    return "(%s) * (%s)" % (_sfem_ccode(factor), expression)


def _sum_cpp_terms(terms):
    terms = tuple(term for term in terms if term)
    if not terms:
        return "scalar_t(0)"
    expression = terms[0]
    for term in terms[1:]:
        if term.startswith("-("):
            expression += " - " + term[2:-1]
        else:
            expression += " + " + term
    return expression


def _constant_reference_gradient_expr(reference_gradients, shape, component):
    return sp.sympify(reference_gradients[int(shape)][int(component)])


def _constant_p1_field_gradient_expr(reference_gradients, dim, field_value, component):
    terms = []
    for shape in range(dim + 1):
        factor = _constant_reference_gradient_expr(reference_gradients, shape, component)
        if factor == 0:
            continue
        terms.append(_scaled_cpp_term(factor, field_value(shape)))
    return _sum_cpp_terms(terms)


def _append_constant_p1_sfem_soa_weak_form_lines(
    lines,
    form,
    dim,
    reference_gradients,
    use_stream_arrays,
    source_builder,
):
    work_item = _work_item_index(source_builder)
    weak_form = form.weak_form
    uses_current = _form_uses_current(form, default=True)
    uses_direction = _form_uses_direction(form, default=form.has_direction)

    def field_value(field, row, shape):
        stream_prefix = "" if use_stream_arrays else "weak_"
        return "%s%s_streams[%d * %d + %d][%s]" % (
            stream_prefix,
            field,
            shape,
            dim,
            row,
            work_item,
        )

    def geometry_value(name, component):
        return _work_item_name(source_builder, name, component)

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "grad_u",
        scalar_temporaries=True,
    )

    lines.append("        for (int q = 0; q < N_QP; ++q) {")
    lines.append("            const scalar_t qw = q_weight[q];")
    lines.extend(_work_item_loop_lines(source_builder, "            "))
    lines.append("            const ptrdiff_t geometry_offset = q * geometry_stride + %s;" % work_item)
    for component in range(dim * dim):
        lines.append(
            "            const scalar_t %s = jacobian_adjugate%d[geometry_offset];"
            % (geometry_value("jacobian_adjugate", component), component)
        )
    lines.append(
        "            const scalar_t %s = jacobian_determinant0[geometry_offset];"
        % geometry_value("jacobian_determinant", 0)
    )
    for row in range(dim):
        for col in range(dim):
            idx = row * dim + col
            if uses_current:
                lines.append(
                    "            const scalar_t grad_u_ref%d = %s;"
                    % (
                        idx,
                        _constant_p1_field_gradient_expr(
                            reference_gradients,
                            dim,
                            lambda shape, row=row: field_value("u", row, shape),
                            col,
                        ),
                    )
                )
            if uses_direction:
                lines.append(
                    "            const scalar_t grad_h_ref%d = %s;"
                    % (
                        idx,
                        _constant_p1_field_gradient_expr(
                            reference_gradients,
                            dim,
                            lambda shape, row=row: field_value("h", row, shape),
                            col,
                        ),
                    )
                )
    lines.append(
        "            const scalar_t inv_jacobian_determinant = scalar_t(1) / %s;"
        % geometry_value("jacobian_determinant", 0)
    )
    for row in range(dim):
        for col in range(dim):
            if uses_current:
                terms = [
                    "grad_u_ref%d * %s"
                    % (
                        row * dim + k,
                        geometry_value("jacobian_adjugate", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "            const scalar_t grad_u%d = (%s) * inv_jacobian_determinant;"
                    % (row * dim + col, " + ".join(terms))
                )
            if uses_direction:
                terms = [
                    "grad_h_ref%d * %s"
                    % (
                        row * dim + k,
                        geometry_value("jacobian_adjugate", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "            const scalar_t trial_grad%d = (%s) * inv_jacobian_determinant;"
                    % (row * dim + col, " + ".join(terms))
                )

    if form.name == "objective":
        _append_cse_array_assignments(
            lines,
            [weak_form.energy_density.xreplace(deformation_gradient_substitutions)],
            ["value[%s] %s" % (work_item, "+=" if form.output_mode == "accumulate" else "=")],
            "weak_obj_tmp",
            scale="qw * %s" % geometry_value("jacobian_determinant", 0),
        )
        lines.extend(["            }", "        }"])
        return

    material = _weak_form_material_expression(
        weak_form,
        form.name,
        deformation_gradient_substitutions,
        tuple(sp.symbols("trial_grad%d" % i) for i in range(dim * dim)),
    )
    _append_transformed_loperand_lines(
        lines,
        material,
        dim,
        "weak_mat_tmp",
        geometry_value,
        scalar_temporaries=True,
    )
    output_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
    op = "+=" if form.output_mode == "accumulate" else "="
    for shape in range(dim + 1):
        for row in range(dim):
            terms = []
            for col in range(dim):
                factor = _constant_reference_gradient_expr(reference_gradients, shape, col)
                if factor == 0:
                    continue
                terms.append(_scaled_cpp_term(factor, "loperand%d" % (row * dim + col)))
            if terms:
                lines.append(
                    "            %s[%d * %d + %d][%s] %s %s;"
                    % (
                        output_streams,
                        shape,
                        dim,
                        row,
                        work_item,
                        op,
                        _sum_cpp_terms(terms),
                    )
                )
    lines.extend(["            }", "        }"])


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
    constant_p1_gradient_expansion=False,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
    weak_form = form.weak_form
    uses_current = _form_uses_current(form, default=True)
    uses_direction = _form_uses_direction(form, default=form.has_direction)
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
    reference_gradients = constant_p1_simplex_reference_gradients(quadrature_rule)
    if constant_p1_gradient_expansion and reference_gradients is not None:
        _append_constant_p1_sfem_soa_weak_form_lines(
            lines,
            form,
            dim,
            reference_gradients,
            use_stream_arrays,
            source_builder,
        )
        return

    def reference_gradient(component, shape="shape"):
        if use_tensor_product_reference:
            return _tensor_product_dynamic_reference_gradient_expr(dim, component)
        return "%s[q * N_SHAPE + %s]" % (
            _sfem_reference_gradient_vector_name(component),
            shape,
        )

    def field_value(field, row, shape="shape"):
        stream_prefix = "" if use_stream_arrays else "weak_"
        return "%s%s_streams[%s * %d + %d][%s]" % (
            stream_prefix,
            field,
            shape,
            dim,
            row,
            work_item,
        )

    def geometry_value(name, component):
        return _work_item_name(source_builder, name, component)

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "grad_u",
        scalar_temporaries=True,
    )

    lines.append("        for (int q = 0; q < N_QP; ++q) {")
    lines.append("            const scalar_t qw = q_weight[q];")
    for row in range(dim):
        for col in range(dim):
            idx = row * dim + col
            if uses_current:
                lines.append("            scalar_t grad_u_ref%d_values[VECTOR_SIZE];" % idx)
            if uses_direction:
                lines.append("            scalar_t grad_h_ref%d_values[VECTOR_SIZE];" % idx)
    if form.name != "objective":
        for component in range(dim * dim):
            lines.append("            scalar_t loperand%d_values[VECTOR_SIZE];" % component)
    for row in range(dim):
        for col in range(dim):
            idx = row * dim + col
            lines.extend(_work_item_loop_lines(source_builder, "            "))
            if uses_current:
                lines.append("                grad_u_ref%d_values[%s] = scalar_t(0);" % (idx, work_item))
            if uses_direction:
                lines.append("                grad_h_ref%d_values[%s] = scalar_t(0);" % (idx, work_item))
            lines.append("            }")
    lines.append("            for (int shape = 0; shape < N_SHAPE; ++shape) {")
    for row in range(dim):
        for col in range(dim):
            idx = row * dim + col
            lines.extend(_work_item_loop_lines(source_builder, "                "))
            if uses_current:
                lines.append(
                    "                    grad_u_ref%d_values[%s] += %s * %s;"
                    % (idx, work_item, field_value("u", row), reference_gradient(col))
                )
            if uses_direction:
                lines.append(
                    "                    grad_h_ref%d_values[%s] += %s * %s;"
                    % (idx, work_item, field_value("h", row), reference_gradient(col))
                )
            lines.append("                }")
    lines.append("            }")
    lines.extend(_work_item_loop_lines(source_builder, "            "))
    lines.append("            const ptrdiff_t geometry_offset = q * geometry_stride + %s;" % work_item)
    for component in range(dim * dim):
        lines.append(
            "            const scalar_t %s = jacobian_adjugate%d[geometry_offset];"
            % (geometry_value("jacobian_adjugate", component), component)
        )
    lines.append(
        "            const scalar_t %s = jacobian_determinant0[geometry_offset];"
        % geometry_value("jacobian_determinant", 0)
    )
    for row in range(dim):
        for col in range(dim):
            idx = row * dim + col
            if uses_current:
                lines.append("            const scalar_t grad_u_ref%d = grad_u_ref%d_values[%s];" % (idx, idx, work_item))
            if uses_direction:
                lines.append("            const scalar_t grad_h_ref%d = grad_h_ref%d_values[%s];" % (idx, idx, work_item))
    lines.append(
        "        const scalar_t inv_jacobian_determinant = scalar_t(1) / %s;"
        % geometry_value("jacobian_determinant", 0)
    )
    for row in range(dim):
        for col in range(dim):
            if uses_current:
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
            if uses_direction:
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

    if form.name == "objective":
        _append_cse_array_assignments(
            lines,
            [weak_form.energy_density.xreplace(deformation_gradient_substitutions)],
            ["value[%s] %s" % (work_item, "+=" if form.output_mode == "accumulate" else "=")],
            "weak_obj_tmp",
            scale="qw * %s" % geometry_value("jacobian_determinant", 0),
        )
        lines.extend(["            }", "        }"])
        return

    material = _weak_form_material_expression(
        weak_form,
        form.name,
        deformation_gradient_substitutions,
        tuple(sp.symbols("trial_grad%d" % i) for i in range(dim * dim)),
    )

    _append_transformed_loperand_lines(
        lines,
        material,
        dim,
        "weak_mat_tmp",
        geometry_value,
        scalar_temporaries=True,
    )
    for component in range(dim * dim):
        lines.append("            loperand%d_values[%s] = loperand%d;" % (component, work_item, component))
    lines.append("            }")
    lines.append("            for (int shape = 0; shape < N_SHAPE; ++shape) {")
    for row in range(dim):
        terms = [
            "loperand%d_values[%s] * %s" % (row * dim + col, work_item, reference_gradient(col))
            for col in range(dim)
        ]
        op = "+=" if form.output_mode == "accumulate" else "="
        output_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
        lines.extend(_work_item_loop_lines(source_builder, "                "))
        lines.append(
            "                    %s[shape * %d + %d][%s] %s %s;"
            % (output_streams, dim, row, work_item, op, " + ".join(terms))
        )
        lines.append("                }")
    lines.extend(["            }", "        }"])


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


def _weak_form_deformation_gradient_substitutions_from_symbols(weak_form, gradient):
    substitutions = {}
    for row in range(weak_form.dim):
        for col in range(weak_form.dim):
            idx = row * weak_form.dim + col
            value = gradient[idx]
            if row == col:
                value = sp.Integer(1) + value
            substitutions[weak_form.deformation_gradient[idx]] = value
    return substitutions


def _weak_form_expressions_are_identical(left, right):
    for a, b in zip(tuple(left), tuple(right)):
        diff = sp.expand(a - b)
        if diff != 0 and sp.simplify(diff) != 0:
            return False
    return True


def _weak_form_material_expression(
    weak_form,
    form_name,
    deformation_gradient_substitutions,
    trial_gradient=None,
):
    if form_name != "apply":
        return weak_form.first_piola().xreplace(deformation_gradient_substitutions)

    if trial_gradient is None:
        raise ValueError("apply weak form material requires a trial gradient")

    linearized = weak_form.linearized_first_piola(trial_gradient).xreplace(
        deformation_gradient_substitutions
    )
    deformation_symbols = set()
    for value in deformation_gradient_substitutions.values():
        deformation_symbols.update(value.free_symbols)
    if any(expr.free_symbols.intersection(deformation_symbols) for expr in tuple(linearized)):
        return linearized

    first_piola_at_trial = weak_form.first_piola().xreplace(
        _weak_form_deformation_gradient_substitutions_from_symbols(
            weak_form,
            trial_gradient,
        )
    )
    if _weak_form_expressions_are_identical(first_piola_at_trial, linearized):
        return first_piola_at_trial
    return linearized


def _append_cse_array_assignments(lines, expressions, targets, temporary_prefix, scale=None):
    temps, reduced = sp.cse(
        tuple(expressions),
        symbols=sp.numbered_symbols("%s" % temporary_prefix),
    )
    temps = _prune_dead_cse_intermediates(temps, reduced)
    for symbol, expression in temps:
        lines.append("        const scalar_t %s = %s;" % (symbol, _sfem_ccode(expression)))
    for target, expression in zip(targets, reduced):
        if scale is not None:
            lines.append("        %s %s * (%s);" % (target, scale, _sfem_ccode(expression)))
        else:
            lines.append("        %s %s;" % (target, _sfem_ccode(expression)))


def _form_dependencies(form):
    return getattr(form, "dependencies", None)


def _form_uses_current(form, default):
    dependencies = _form_dependencies(form)
    if dependencies is None:
        return bool(default)
    return bool(getattr(dependencies, "current", False))


def _form_uses_direction(form, default):
    dependencies = _form_dependencies(form)
    if dependencies is None:
        return bool(default)
    return bool(getattr(dependencies, "direction", False))


def _form_material_parameter_names(form):
    dependencies = _form_dependencies(form)
    if dependencies is None:
        return ()
    return tuple(str(parameter) for parameter in getattr(dependencies, "parameters", ()))


def _form_material_parameter_declarations(form, scalar_type="scalar_t"):
    return tuple(
        "const %s %s" % (scalar_type, parameter)
        for parameter in _form_material_parameter_names(form)
    )


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
        and (
            sfem_tensor_product_hex_uses_cartesian_ordering(quadrature_rule.element_type)
            or sfem_tensor_product_quad_uses_cartesian_ordering(quadrature_rule.element_type)
        )
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
    cartesian_quad = dim == 2 and n_shape_1d == 2 and sfem_tensor_product_quad_uses_cartesian_ordering(quadrature_rule.element_type)
    if n_shape_1d == 2 and dim == 2 and not cartesian_quad:
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
    if (
        n_shape_1d == 2
        and dim == 2
        and not sfem_tensor_product_quad_uses_cartesian_ordering(quadrature_rule.element_type)
    ):
        return (
            "%sconst int sx = ((shape + 1) >> 1) & 1;" % indent,
            "%sconst int sy = shape >> 1;" % indent,
        )
    if n_shape_1d == 2 and dim == 2:
        return (
            "%sconst int sx = shape & 1;" % indent,
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


def _tensor_product_shape_coordinate_arrays(quadrature_rule, indent):
    coords = _tensor_product_node_coords(quadrature_rule)
    axis_names = ("x", "y", "z")[: quadrature_rule.dim]
    lines = []
    for axis, name in enumerate(axis_names):
        lines.append(
            "%sstatic constexpr int SHAPE_%s[N_SHAPE] = {%s};"
            % (
                indent,
                name.upper(),
                ", ".join(str(coord[axis]) for coord in coords),
            )
        )
    return lines


def _tensor_product_shape_coordinate_lines(dim, shape_expr, prefix, indent):
    if dim == 2:
        return (
            "%sconst int %s_sx = %s %% N_SHAPE_1D;" % (indent, prefix, shape_expr),
            "%sconst int %s_sy = %s / N_SHAPE_1D;" % (indent, prefix, shape_expr),
        )
    if dim == 3:
        return (
            "%sconst int %s_sx = %s %% N_SHAPE_1D;" % (indent, prefix, shape_expr),
            "%sconst int %s_sy = (%s / N_SHAPE_1D) %% N_SHAPE_1D;"
            % (indent, prefix, shape_expr),
            "%sconst int %s_sz = %s / (N_SHAPE_1D * N_SHAPE_1D);"
            % (indent, prefix, shape_expr),
        )
    raise ValueError("tensor-product shape coordinates require dim 2 or 3")


def _tensor_product_reference_gradient_expr_from_coords(
    dim,
    derivative_axis,
    coord_prefix,
    shape_name="shape_1d",
    grad_name="grad_1d",
):
    factors = []
    for axis in range(dim):
        qp_name = ("qx", "qy", "qz")[axis]
        node_axis_name = "%s_%s" % (coord_prefix, ("sx", "sy", "sz")[axis])
        table_name = grad_name if axis == derivative_axis else shape_name
        factors.append("%s[%s * N_SHAPE_1D + %s]" % (table_name, qp_name, node_axis_name))
    return " * ".join(factors)


def _sfem_soa_reference_gradient_expr_for_shape(
    dim,
    component,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_inputs,
    shape_expr,
    coord_prefix=None,
    reference_prefix="",
):
    if use_tensor_product_reference:
        if coord_prefix is None:
            raise ValueError("tensor-product reference gradients require coordinate variables")
        return _tensor_product_reference_gradient_expr_from_coords(
            dim,
            component,
            coord_prefix,
            "%sshape_1d" % reference_prefix,
            "%sgrad_1d" % reference_prefix,
        )
    if use_reference_gradient_vectors:
        return "%s%s[q * N_SHAPE + %s]" % (
            reference_prefix,
            _sfem_reference_gradient_vector_name(component),
            shape_expr,
        )
    if len(reference_inputs) == 1 and reference_inputs[0].name == "grad_ref":
        return "%s%s[(q * N_SHAPE + %s) * %d + %d]" % (
            reference_prefix,
            reference_inputs[0].name,
            shape_expr,
            dim,
            component,
        )
    raise ValueError("hessian matrix generation requires reference gradients")


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
    coordinate_streams="block_coordinate_data",
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
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
    for row in range(dim):
        for col in range(dim):
            lines.append("            scalar_t J%d%d_values[VECTOR_SIZE];" % (row, col))
    for row in range(dim):
        for col in range(dim):
            lines.extend(_work_item_loop_lines(source_builder, "            "))
            lines.extend(
                [
                    "                J%d%d_values[%s] = scalar_t(0);" % (row, col, work_item),
                    "            }",
                ]
            )
    lines.append("            for (int shape = 0; shape < N_SHAPE; ++shape) {")
    if use_tensor_product_reference:
        lines.extend(_tensor_product_shape_index_lines(quadrature_rule, "                "))
    for col in range(dim):
        lines.append(
            "                const scalar_t g%d = %s;"
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
            lines.extend(
                [
                    *_work_item_loop_lines(source_builder, "                "),
                    "                    J%d%d_values[%s] += %s[shape * %d + %d][%s] * g%d;"
                    % (row, col, work_item, coordinate_streams, dim, row, work_item, col),
                    "                }",
                ]
            )
    lines.append("            }")
    lines.extend(_work_item_loop_lines(source_builder, "            "))
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "                const scalar_t J%d%d = J%d%d_values[%s];"
                % (row, col, row, col, work_item)
            )
    output_index = "q * VECTOR_SIZE + %s" % work_item if q_major else work_item
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


def _append_sfem_soa_output_lines(lines, form, dim, n_nodes, work_item):
    output_count = len(form.expression_graph.evaluation_plan.outputs)
    if output_count == 1:
        op = "+=" if form.output_mode == "accumulate" else "="
        lines.append("        value[%s] %s element_vector[0];" % (work_item, op))
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
            lines.append("        %s[%s] %s element_vector[%d];" % (stream, work_item, op, idx))


def _sfem_soa_operator_source(
    forms,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    local_name,
    hessian_name,
    geometry_name,
    diagnostics_name,
    array_inputs,
    quadrature_rule,
    affine_quadrature_rule,
    basis_family=None,
    geometry_family=None,
    use_shared_weak_local=False,
    matrix_format_plan=None,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    lines = [
        *source_builder.operator_preamble_lines(
            local_name,
            geometry_name,
            diagnostics_name,
            extra_headers=(() if hessian_name is None else (hessian_name,)),
        ),
        "",
        "#include <cstdint>",
        "#include <cstdlib>",
    ]
    if getattr(source_builder, "operator_extension", "cpp") == "cpp":
        lines.append('#include "packed_thread_scratch.hpp"')
    lines.extend(
        [
            "",
            "#ifndef SFEM_SUCCESS",
            "#define SFEM_SUCCESS 0",
            "#endif",
            "",
            "#ifndef SFEM_FAILURE",
            "#define SFEM_FAILURE 1",
            "#endif",
            "",
            "#ifndef MIN",
            "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
            "#endif",
            "",
        ]
    )
    lines = [line for line in lines if line != ""]
    lines.append("")
    lines.extend(
        _affine_geometry_stream_helper_lines(
            source_builder,
        )
    )
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
                    matrix_format_plan=matrix_format_plan,
                    source_builder=source_builder,
                )
            )
            fast_aos_unit_lines = _tet4_linear_elasticity_aos_unit_mesh_operator_function(
                form,
                prefix,
                dim,
                n_nodes,
                affine_rule,
                source_builder=source_builder,
            )
            if fast_aos_unit_lines:
                lines.append("")
                lines.extend(fast_aos_unit_lines)
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
                    matrix_format_plan=matrix_format_plan,
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
            if form.name == "apply":
                lines.append("")
                lines.extend(
                    _sfem_soa_hessian_matrix_assembly_function(
                        form,
                        prefix,
                        dim,
                        n_nodes,
                        n_qp,
                        local_prefix,
                        array_inputs,
                        quadrature_rule,
                        basis_family,
                        geometry_family,
                        use_shared_weak_local,
                        matrix_format_plan,
                        source_builder=source_builder,
                    )
                )
        lines.append("")

    return "\n".join(lines)


def _is_tet4_linear_elasticity_aos_unit_candidate(form, prefix, dim, n_nodes, quadrature_rule):
    if form.name not in ("gradient", "apply"):
        return False
    if form.weak_form is None:
        return False
    if dim != 3 or n_nodes != 4:
        return False
    if not str(prefix).startswith("linear_elasticity_"):
        return False
    if quadrature_rule is None:
        return False
    element_type = str(getattr(quadrature_rule, "element_type", "")).lower()
    return element_type == "tet4"


def _tet4_linear_elasticity_aos_unit_mesh_operator_function(
    form,
    prefix,
    dim,
    n_nodes,
    quadrature_rule,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    if not _is_tet4_linear_elasticity_aos_unit_candidate(
        form, prefix, dim, n_nodes, quadrature_rule
    ):
        return []

    function_name = "%s_aos_unit" % _sfem_soa_mesh_public_function_name(
        prefix,
        form.name,
        quadrature_rule,
        "affine",
    )
    implementation_name = "%s_impl" % function_name
    input_prefix = "u" if form.name == "gradient" else "h"
    stride_name = "%s_stride" % input_prefix
    input_components = tuple("%s%s" % (input_prefix, _component_name(d)) for d in range(3))

    impl_params = (
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate_aos",
        "const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0",
        "const scalar_t mu",
        "const scalar_t lmbda",
        "const ptrdiff_t %s" % stride_name,
        "const scalar_t *const SFEM_RESTRICT %s" % input_components[0],
        "const scalar_t *const SFEM_RESTRICT %s" % input_components[1],
        "const scalar_t *const SFEM_RESTRICT %s" % input_components[2],
        "const ptrdiff_t out_stride",
        "scalar_t *const SFEM_RESTRICT outx",
        "scalar_t *const SFEM_RESTRICT outy",
        "scalar_t *const SFEM_RESTRICT outz",
    )
    wrapper_params = tuple(
        param.replace("jacobian_t", "geom_t") for param in impl_params
    )

    vx, vy, vz = input_components
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, typename jacobian_t>",
        "static SFEM_INLINE int %s(" % implementation_name,
    ]
    for idx, param in enumerate(impl_params):
        comma = "," if idx + 1 < len(impl_params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    (void)nnodes;",
            "",
            "#pragma omp parallel for schedule(static)",
            "    for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "        const idx_t ev0 = elements[0][element];",
            "        const idx_t ev1 = elements[1][element];",
            "        const idx_t ev2 = elements[2][element];",
            "        const idx_t ev3 = elements[3][element];",
            "",
            "        const scalar_t ux0 = %s[ev0 * %s];" % (vx, stride_name),
            "        const scalar_t ux1 = %s[ev1 * %s];" % (vx, stride_name),
            "        const scalar_t ux2 = %s[ev2 * %s];" % (vx, stride_name),
            "        const scalar_t ux3 = %s[ev3 * %s];" % (vx, stride_name),
            "        const scalar_t uy0 = %s[ev0 * %s];" % (vy, stride_name),
            "        const scalar_t uy1 = %s[ev1 * %s];" % (vy, stride_name),
            "        const scalar_t uy2 = %s[ev2 * %s];" % (vy, stride_name),
            "        const scalar_t uy3 = %s[ev3 * %s];" % (vy, stride_name),
            "        const scalar_t uz0 = %s[ev0 * %s];" % (vz, stride_name),
            "        const scalar_t uz1 = %s[ev1 * %s];" % (vz, stride_name),
            "        const scalar_t uz2 = %s[ev2 * %s];" % (vz, stride_name),
            "        const scalar_t uz3 = %s[ev3 * %s];" % (vz, stride_name),
            "",
            "        const jacobian_t *const SFEM_RESTRICT adjugate = g_jacobian_adjugate_aos + element * 9;",
        ]
    )
    for component in range(9):
        lines.append(
            "        const scalar_t a%d = scalar_t(adjugate[%d]);"
            % (component, component)
        )
    lines.extend(
        [
            "        const scalar_t inv_det = scalar_t(1) / scalar_t(g_jacobian_determinant0[element]);",
            "",
            "        const scalar_t x1 = ux0 - ux1;",
            "        const scalar_t x2 = ux0 - ux2;",
            "        const scalar_t x3 = ux0 - ux3;",
            "        const scalar_t x4 = uy0 - uy1;",
            "        const scalar_t x5 = uy0 - uy2;",
            "        const scalar_t x6 = uy0 - uy3;",
            "        const scalar_t x7 = uz0 - uz1;",
            "        const scalar_t x8 = uz0 - uz2;",
            "        const scalar_t x9 = uz0 - uz3;",
            "",
            "        scalar_t p0 = inv_det * (-a0 * x1 - a3 * x2 - a6 * x3);",
            "        scalar_t p1 = inv_det * (-a1 * x1 - a4 * x2 - a7 * x3);",
            "        scalar_t p2 = inv_det * (-a2 * x1 - a5 * x2 - a8 * x3);",
            "        scalar_t p3 = inv_det * (-a0 * x4 - a3 * x5 - a6 * x6);",
            "        scalar_t p4 = inv_det * (-a1 * x4 - a4 * x5 - a7 * x6);",
            "        scalar_t p5 = inv_det * (-a2 * x4 - a5 * x5 - a8 * x6);",
            "        scalar_t p6 = inv_det * (-a0 * x7 - a3 * x8 - a6 * x9);",
            "        scalar_t p7 = inv_det * (-a1 * x7 - a4 * x8 - a7 * x9);",
            "        scalar_t p8 = inv_det * (-a2 * x7 - a5 * x8 - a8 * x9);",
            "",
            "        const scalar_t m0 = (scalar_t(1) / scalar_t(6)) * mu;",
            "        const scalar_t m1 = m0 * (p1 + p3);",
            "        const scalar_t m2 = m0 * (p2 + p6);",
            "        const scalar_t m3 = scalar_t(2) * mu;",
            "        const scalar_t m4 = lmbda * (p0 + p4 + p8);",
            "        const scalar_t m5 = (scalar_t(1) / scalar_t(6)) * p0 * m3 + (scalar_t(1) / scalar_t(6)) * m4;",
            "        const scalar_t m6 = m0 * (p5 + p7);",
            "        const scalar_t m7 = (scalar_t(1) / scalar_t(6)) * p4 * m3 + (scalar_t(1) / scalar_t(6)) * m4;",
            "        const scalar_t m8 = (scalar_t(1) / scalar_t(6)) * p8 * m3 + (scalar_t(1) / scalar_t(6)) * m4;",
            "",
            "        const scalar_t q0 = a0 * m5 + a1 * m1 + a2 * m2;",
            "        const scalar_t q1 = a3 * m5 + a4 * m1 + a5 * m2;",
            "        const scalar_t q2 = a6 * m5 + a7 * m1 + a8 * m2;",
            "        const scalar_t q3 = a0 * m1 + a1 * m7 + a2 * m6;",
            "        const scalar_t q4 = a3 * m1 + a4 * m7 + a5 * m6;",
            "        const scalar_t q5 = a6 * m1 + a7 * m7 + a8 * m6;",
            "        const scalar_t q6 = a0 * m2 + a1 * m6 + a2 * m8;",
            "        const scalar_t q7 = a3 * m2 + a4 * m6 + a5 * m8;",
            "        const scalar_t q8 = a6 * m2 + a7 * m6 + a8 * m8;",
            "",
        ]
    )
    scatter_values = (
        ("outx", "ev0", "-q0 - q1 - q2"),
        ("outx", "ev1", "q0"),
        ("outx", "ev2", "q1"),
        ("outx", "ev3", "q2"),
        ("outy", "ev0", "-q3 - q4 - q5"),
        ("outy", "ev1", "q3"),
        ("outy", "ev2", "q4"),
        ("outy", "ev3", "q5"),
        ("outz", "ev0", "-q6 - q7 - q8"),
        ("outz", "ev1", "q6"),
        ("outz", "ev2", "q7"),
        ("outz", "ev3", "q8"),
    )
    for out, ev, value in scatter_values:
        lines.extend(
            [
                "        #pragma omp atomic update",
                "        %s[%s * out_stride] += %s;" % (out, ev, value),
            ]
        )
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
                "    return sfem::codegen::%s<%s, geom_t>(%s);"
                % (implementation_name, scalar_type, ", ".join(wrapper_args)),
                "}",
                "",
            ]
        )
    return lines


def _sfem_soa_packed_objective_steps_public_wrappers(
    function_name,
    dim,
    n_nodes,
    n_qp,
    prefix,
    local_prefix,
    block_name,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_tensor_product_geometry,
    use_reference_gradient_vectors,
    omit_reference_basis_inputs,
    stream_shape_order,
    identity_stream_shape_order,
    vector_size,
    geometry_mode,
    material_parameter_names,
    source_builder,
):
    if geometry_mode not in ("affine", "isoparametric"):
        return []
    if getattr(source_builder, "operator_extension", "cpp") != "cpp":
        return []

    public_base = function_name.replace("_objective_steps_", "_objective_steps_packed_")
    is_affine = geometry_mode == "affine"
    reference_prefix = "%s_" % geometry_mode
    tensor_shape_name = "%sshape_1d" % reference_prefix
    tensor_grad_name = "%sgrad_1d" % reference_prefix
    tensor_weight_name = "%sq_weight_1d" % reference_prefix
    scalar_weight_name = "%sq_weight" % reference_prefix
    lines = ["namespace sfem {", "namespace codegen {", ""]

    for scalar_type in ("double", "float"):
        suffix = "" if scalar_type == "double" else "_float"
        public_name = "%s%s" % (public_base, suffix)
        lines.extend(
            [
                'extern "C" int %s(' % public_name,
                "        const ptrdiff_t n_packs,",
                "        const ptrdiff_t n_elements_per_pack,",
                "        const ptrdiff_t nelements,",
                "        const ptrdiff_t nnodes,",
                "        const ptrdiff_t max_nodes_per_pack,",
                "        uint16_t **const SFEM_RESTRICT elements,",
                "        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,",
                "        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,",
                "        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,",
                "        const idx_t *const SFEM_RESTRICT ghost_idx,",
            ]
        )
        if is_affine:
            for stream in _soa_array_stream_names(_adjugate_input(dim)):
                lines.append("        const geom_t *const SFEM_RESTRICT g_%s," % stream)
            lines.append("        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,")
        else:
            lines.append("        const geom_t *const *const SFEM_RESTRICT points,")
        lines.extend(
            "        const %s %s," % (scalar_type, parameter)
            for parameter in material_parameter_names
        )
        lines.append("        const ptrdiff_t u_stride,")
        for d in range(dim):
            lines.append(
                "        const %s *const SFEM_RESTRICT u%s,"
                % (scalar_type, _component_name(d))
            )
        lines.append("        const ptrdiff_t h_stride,")
        for d in range(dim):
            lines.append(
                "        const %s *const SFEM_RESTRICT h%s,"
                % (scalar_type, _component_name(d))
            )
        lines.extend(
            [
                "        const int nsteps,",
                "        const %s *const SFEM_RESTRICT steps," % scalar_type,
                "        %s *const SFEM_RESTRICT value" % scalar_type,
                ") {",
                "    using scalar_t = %s;" % scalar_type,
                "    static constexpr int DIM = %d;" % dim,
                "    static constexpr int N_QP = %d;" % n_qp,
                "    static constexpr int N_SHAPE = %d;" % n_nodes,
                "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
                "    (void)nnodes;",
                "    (void)n_shared_nodes;",
                "",
            ]
        )
        if not is_affine:
            for d in range(dim):
                lines.append(
                    "    const geom_t *const SFEM_RESTRICT %s = points[%d];"
                    % (_component_name(d), d)
                )
            lines.extend(
                _ordered_element_pointer_array_lines(
                    "uint16_t",
                    "coordinate_elements",
                    "elements",
                    stream_shape_order,
                    "    ",
                )
            )
        lines.extend(
            _sfem_soa_mesh_reference_alias_lines(
                prefix,
                quadrature_rule,
                reference_inputs,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                geometry_mode,
                emit_reference_basis=True,
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
                "#pragma omp parallel",
                "    {",
            ]
        )
        if not is_affine:
            lines.append(
                "        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);"
            )
        lines.extend(
            [
                "        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);",
                "        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);",
                "",
                "#pragma omp for schedule(static)",
                "        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "            const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
                "            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                "            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];",
            ]
        )
        if not is_affine:
            lines.append(
                "            const geom_t *const coordinate_components[DIM] = {%s};"
                % ", ".join(_component_name(d) for d in range(dim))
            )
        lines.extend(
            [
                "            const scalar_t *const u_components[DIM] = {%s};"
                % ", ".join("u%s" % _component_name(d) for d in range(dim)),
                "            const scalar_t *const h_components[DIM] = {%s};"
                % ", ".join("h%s" % _component_name(d) for d in range(dim)),
                "            for (int d = 0; d < DIM; ++d) {",
            ]
        )
        if not is_affine:
            lines.append(
                "                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;"
            )
        lines.extend(
            [
                "                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;",
                "                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;",
            ]
        )
        if not is_affine:
            lines.append(
                "                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];"
            )
        lines.extend(
            [
                "                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];",
                "                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];",
                "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                "                    const idx_t node = owned_nodes_ptr[pack] + k;",
            ]
        )
        if not is_affine:
            lines.append("                    pack_coordinate[k] = scalar_t(coordinate_component[node]);")
        lines.extend(
            [
                "                    pack_u_base_component[k] = u_component[node * u_stride];",
                "                    pack_h_component[k] = h_component[node * h_stride];",
                "                }",
                "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                "                    const idx_t node = ghosts[k];",
            ]
        )
        if not is_affine:
            lines.append(
                "                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);"
            )
        lines.extend(
            [
                "                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];",
                "                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];",
                "                }",
                "            }",
                "",
                "            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {",
                "                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);",
                "                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];",
                "                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];",
                "                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];",
                "                scalar_t block_value[VECTOR_SIZE];",
            ]
        )
        if not is_affine:
            lines.append("                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
            for stream in _soa_array_stream_names(_adjugate_input(dim)):
                lines.append("                scalar_t block_%s[N_QP * VECTOR_SIZE];" % stream)
            lines.extend(
                [
                    "                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];",
                    "                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {%s};"
                    % ", ".join("block_jacobian_adjugate%d" % i for i in range(dim * dim)),
                ]
            )
        lines.extend(
            [
                "",
                "                const scalar_t *block_u_streams[N_SHAPE * DIM] = {%s};"
                % ", ".join(
                    "block_u_data[%d]" % stream
                    for stream in streams_in_shape_order(
                        tuple(range(dim * n_nodes)),
                        dim,
                        stream_shape_order,
                    )
                ),
                "",
                "                for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];",
                *([] if is_affine or identity_stream_shape_order else ["                    const uint16_t *const SFEM_RESTRICT coordinate_shape = coordinate_elements[shape];"]),
                "                    for (int d = 0; d < DIM; ++d) {",
                "#pragma omp simd",
                "                        for (int lane = 0; lane < nelems; ++lane) {",
                "                            const uint16_t packed_node = element_shape[evbegin + lane];",
                *([] if is_affine or identity_stream_shape_order else ["                            const uint16_t coordinate_packed_node = coordinate_shape[evbegin + lane];"]),
            ]
        )
        if not is_affine:
            lines.append(
                "                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + %s];"
                % ("packed_node" if identity_stream_shape_order else "coordinate_packed_node")
            )
        lines.extend(
            [
                "                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];",
                "                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];",
                "                        }",
                "                    }",
                "                }",
                "",
            ]
        )
        if is_affine:
            lines.extend(
                _sfem_soa_affine_geometry_stream_lines(
                    source_builder,
                    (_adjugate_input(dim), sfem_soa_reference_input("jacobian_determinant", 1, 1, 1)),
                    "                ",
                    geometry_scalar_type="geom_t",
                )
            )
        elif use_tensor_product_geometry:
            lines.extend(
                tensor_product_gradient_isoparametric_geometry_lines(
                    dim=dim,
                    n_shape=n_nodes,
                    n_qp=quadrature_rule.n_qp,
                    local_prefix=local_prefix,
                    coordinate_streams="block_coordinate_data",
                    contiguous_coordinate_streams=True,
                    adjugate_target=lambda component, index: "block_jacobian_adjugate%d[%s]" % (component, index),
                    determinant_target=lambda index: "block_jacobian_determinant0[%s]" % index,
                    adjugate_streams=tuple("block_jacobian_adjugate%d" % component for component in range(dim * dim)),
                    determinant_stream="block_jacobian_determinant0",
                    shape_name=tensor_shape_name,
                    grad_name=tensor_grad_name,
                )
            )
        else:
            lines.extend(["", "                for (int q = 0; q < N_QP; ++q) {"])
            geometry_lines = _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_inputs,
                q_major=True,
                reference_prefix=reference_prefix,
                source_builder=source_builder,
                coordinate_streams="block_coordinate_data",
            )
            lines.extend("    %s" % line if line else line for line in geometry_lines)
            lines.append("                }")
        call_args = [
            "nelems",
            "0" if is_affine else "VECTOR_SIZE",
            *("block_jacobian_adjugate%d" % i for i in range(dim * dim)),
            "block_jacobian_determinant0",
        ]
        if omit_reference_basis_inputs:
            pass
        elif use_tensor_product_reference:
            call_args.extend((tensor_shape_name, tensor_grad_name))
        elif use_reference_gradient_vectors:
            call_args.extend(
                "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
                for component in range(dim)
            )
        else:
            call_args.extend(
                "%s%s" % (reference_prefix, array_input.name)
                for array_input in reference_inputs
            )
        call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
        call_args.extend(material_parameter_names)
        call_args.extend(("block_u_streams", "block_value"))
        lines.extend(
            [
                "",
                "                for (int step = 0; step < nsteps; ++step) {",
                "                    const scalar_t alpha = steps[step];",
                "                    for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "                        for (int d = 0; d < DIM; ++d) {",
                "#pragma omp simd",
                "                            for (int lane = 0; lane < nelems; ++lane) {",
                "                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];",
                "                            }",
                "                        }",
                "                    }",
                "#pragma omp simd",
                "                    for (int lane = 0; lane < nelems; ++lane) {",
                "                        block_value[lane] = scalar_t(0);",
                "                    }",
                "",
                "                    %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
                % (block_name, ", ".join(call_args)),
                "",
                "#pragma omp simd",
                "                    for (int lane = 0; lane < nelems; ++lane) {",
                "                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];",
                "                    }",
                "                }",
                "            }",
                "        }",
                "    }",
                "    return SFEM_SUCCESS;",
                "}",
                "",
            ]
        )

    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])
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
    matrix_format_plan=None,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
    effective_vector_size = source_builder.effective_vector_size(vector_size)
    if geometry_mode not in ("affine", "isoparametric"):
        raise ValueError("mesh geometry_mode must be 'affine' or 'isoparametric'")
    if quadrature_rule is None:
        raise ValueError("mesh SoA wrappers require an element quadrature rule")
    material_parameter_names = _form_material_parameter_names(form)

    function_name = _sfem_soa_mesh_public_function_name(
        prefix,
        form.name,
        quadrature_rule,
        geometry_mode,
    )
    implementation_name = "%s_impl" % function_name
    block_name = "%s_%s_block" % (local_prefix, form.name)
    specialized_prefix = _constant_p1_specialized_local_prefix(
        local_prefix,
        quadrature_rule,
    )
    if specialized_prefix is not None and form.weak_form is not None:
        block_name = "%s_%s_block" % (specialized_prefix, form.name)
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
    omit_reference_basis_inputs = (
        specialized_prefix is not None
        and form.weak_form is not None
        and not use_tensor_product_reference
    )
    use_stream_arrays = use_shared_weak_local and form.weak_form is not None
    compact_coordinate_buffers = geometry_mode == "isoparametric"
    uses_current = _form_uses_current(form, default=True)
    uses_direction = _form_uses_direction(form, default=form.has_direction)
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )
    identity_stream_shape_order = tuple(stream_shape_order) == tuple(range(n_nodes))
    coordinate_streams_name = "block_coordinate_data"

    base_params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
    ]
    if geometry_mode == "affine":
        base_params.extend(
            "const jacobian_t *const SFEM_RESTRICT g_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        base_params.append("const geometry_t *const *const SFEM_RESTRICT points")

    material_params = _form_material_parameter_declarations(form)
    field_params = []
    if uses_current:
        field_params.append("const ptrdiff_t u_stride")
        field_params.extend(
            "const scalar_t *const SFEM_RESTRICT u%s" % _component_name(d)
            for d in range(dim)
        )
    if uses_direction:
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
        param.replace("geometry_t", "geom_t").replace("jacobian_t", "geom_t")
        for param in impl_params
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
        if compact_coordinate_buffers:
            lines.extend(
                _ordered_element_pointer_array_lines(
                    "idx_t",
                    "coordinate_elements",
                    "elements",
                    stream_shape_order,
                    "    ",
                )
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
            emit_reference_basis=(
                not omit_reference_basis_inputs or geometry_mode == "isoparametric"
            ),
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
        if uses_current:
            lines.append("        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if uses_direction:
            lines.append("        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];")
        if form.name != "objective":
            lines.append("        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];")
        else:
            lines.append("        scalar_t block_value[VECTOR_SIZE];")
        if compact_coordinate_buffers:
            lines.append("        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
    elif compact_coordinate_buffers:
        lines.append("        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
    elif geometry_mode == "isoparametric":
        for stream in _coordinate_stream_names(dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
    if geometry_mode == "isoparametric":
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                extent = "N_QP * VECTOR_SIZE" if form.weak_form is not None else "VECTOR_SIZE"
                lines.append("        scalar_t block_%s[%s];" % (stream, extent))
    if not compact_stream_buffers:
        if uses_current:
            for stream in _field_stream_names("u", dim, n_nodes):
                lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
        if uses_direction:
            for stream in _field_stream_names("h", dim, n_nodes):
                lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
        for stream in _output_stream_names(form, dim, n_nodes):
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)

    lines.extend(
        [
            "",
            "        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {",
            "            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];",
            *_work_item_loop_lines(source_builder, "            "),
            "                ev[element_node * VECTOR_SIZE + %s] = element_shape[evbegin + %s];" % (work_item, work_item),
            "            }",
            "        }",
        ]
    )

    if geometry_mode == "isoparametric":
        if compact_coordinate_buffers:
            lines.append("        const geometry_t *const coordinate_components[DIM] = {%s};" % ", ".join(_component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "",
                    "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    *([] if identity_stream_shape_order else ["            const idx_t *const SFEM_RESTRICT coordinate_element_shape = coordinate_elements[shape];"]),
                    "            for (int d = 0; d < DIM; ++d) {",
                    *_work_item_loop_lines(source_builder, "                "),
                    "                    block_coordinate_data[shape * DIM + d][%s] = coordinate_components[d][%s];"
                    % (
                        work_item,
                        "ev[shape * VECTOR_SIZE + %s]" % work_item
                        if identity_stream_shape_order
                        else "coordinate_element_shape[evbegin + %s]" % work_item,
                    ),
                    "                }",
                    "            }",
                    "        }",
                ]
            )
        else:
            lines.extend([""])
            lines.extend(_work_item_loop_lines(source_builder, "        "))
            for shape in range(n_nodes):
                for d in range(dim):
                    stream = "%s%d" % (_component_name(d), shape)
                    lines.append(
                        "            block_%s[%s] = %s[ev[%d * VECTOR_SIZE + %s]];"
                        % (stream, work_item, _component_name(d), shape, work_item)
                    )
            lines.append("        }")

    if compact_stream_buffers:
        if uses_current:
            lines.append("        const scalar_t *const u_components[DIM] = {%s};" % ", ".join("u%s" % _component_name(d) for d in range(dim)))
        if uses_direction:
            lines.append("        const scalar_t *const h_components[DIM] = {%s};" % ", ".join("h%s" % _component_name(d) for d in range(dim)))
        lines.extend(
            [
                "",
            ]
        )
        lines.extend(
            [
                "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "            for (int d = 0; d < DIM; ++d) {",
                *_work_item_loop_lines(source_builder, "                "),
                "                    const idx_t node = ev[shape * VECTOR_SIZE + %s];" % work_item,
            ]
        )
        if uses_current:
            lines.append("                    block_u_data[shape * DIM + d][%s] = u_components[d][node * u_stride];" % work_item)
        if uses_direction:
            lines.append("                    block_h_data[shape * DIM + d][%s] = h_components[d][node * h_stride];" % work_item)
        lines.extend(["                }", "            }", "        }"])
        if form.name == "objective":
            lines.extend(_work_item_loop_lines(source_builder, "        "))
            lines.extend(["            block_value[%s] = scalar_t(0);" % work_item, "        }"])
        else:
            lines.extend(
                [
                    "        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    *_work_item_loop_lines(source_builder, "            "),
                    "                block_out_data[stream][%s] = scalar_t(0);" % work_item,
                    "            }",
                    "        }",
                ]
            )
    else:
        lines.extend([""])
        lines.extend(_work_item_loop_lines(source_builder, "        "))
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                if uses_current:
                    lines.append(
                        "            block_u%s%d[%s] = u%s[ev[%d * VECTOR_SIZE + %s] * u_stride];"
                        % (component, shape, work_item, component, shape, work_item)
                    )
                if uses_direction:
                    lines.append(
                        "            block_h%s%d[%s] = h%s[ev[%d * VECTOR_SIZE + %s] * h_stride];"
                        % (component, shape, work_item, component, shape, work_item)
                    )
        for stream in _output_stream_names(form, dim, n_nodes):
            lines.append("            block_%s[%s] = scalar_t(0);" % (stream, work_item))
        lines.append("        }")

    if use_stream_arrays:
        lines.append("")
        if uses_current and compact_stream_buffers:
            lines.extend(
                _ordered_stream_pointer_array_lines(
                    "const scalar_t *",
                    "block_u_streams",
                    "block_u_data",
                    dim,
                    stream_shape_order,
                    "        ",
                )
            )
        elif uses_current:
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
        if uses_direction:
            if compact_stream_buffers:
                lines.extend(
                    _ordered_stream_pointer_array_lines(
                        "const scalar_t *",
                        "block_h_streams",
                        "block_h_data",
                        dim,
                        stream_shape_order,
                        "        ",
                    )
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
                    _ordered_stream_pointer_array_lines(
                        "scalar_t *",
                        "block_out_streams",
                        "block_out_data",
                        dim,
                        stream_shape_order,
                        "        ",
                    )
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
                coordinate_streams="block_coordinate_data",
                contiguous_coordinate_streams=True,
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
                coordinate_streams=coordinate_streams_name,
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
                coordinate_streams=coordinate_streams_name,
            )
        )
    elif geometry_mode == "affine":
        lines.extend(
            _sfem_soa_affine_geometry_stream_lines(
                source_builder,
                element_inputs,
                "        ",
            )
        )

    call_args = ["nelems"]
    if form.weak_form is not None:
        call_args.append("0" if geometry_mode == "affine" else "VECTOR_SIZE")
    else:
        call_args.append("q")
    if geometry_mode == "affine":
        call_args.extend(
            "block_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        call_args.extend(
            "block_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    if omit_reference_basis_inputs:
        pass
    elif use_tensor_product_reference:
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
        call_args.extend(material_parameter_names)
    elif use_tensor_product_reference:
        call_args.append("tensor_q_weight")
        call_args.extend(material_parameter_names)
    else:
        call_args.append("%s[q]" % scalar_weight_name)
        call_args.extend(material_parameter_names)
    if use_stream_arrays:
        if uses_current:
            call_args.append("block_u_streams")
        if uses_direction:
            call_args.append("block_h_streams")
        if form.name == "objective":
            call_args.append("block_value")
        else:
            call_args.append("block_out_streams")
    else:
        if uses_current:
            call_args.extend("block_%s" % stream for stream in _field_stream_names("u", dim, n_nodes))
        if uses_direction:
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
        lines.extend(_work_item_loop_lines(source_builder, "        "))
        lines.append("            value[evbegin + %s] += block_value[%s];" % (work_item, work_item))
        lines.append("        }")
    else:
        if compact_stream_buffers:
            lines.append("        scalar_t *const out_components[DIM] = {%s};" % ", ".join("out%s" % _component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "",
                    "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    "            for (int d = 0; d < DIM; ++d) {",
                    *_scatter_add_lines(
                        source_builder,
                        "out_components[d]",
                        "ev[shape * VECTOR_SIZE + %s] * out_stride",
                        "block_out_data[shape * DIM + d][%s]",
                        "                ",
                    ),
                    "            }",
                    "        }",
                ]
            )
        else:
            for shape in range(n_nodes):
                for d in range(dim):
                    component = _component_name(d)
                    lines.extend(
                        list(
                            _scatter_add_lines(
                                source_builder,
                                "out%s" % component,
                                "ev[%d * VECTOR_SIZE + %%s] * out_stride" % shape,
                                "block_out%s%d[%%s]" % (component, shape),
                                "        ",
                            )
                        )
                        + [""]
                    )

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
                ", geom_t",
                wrapper_args,
            ),
            "}",
                "",
            ]
        )
    if form.name in ("gradient", "apply"):
        lines.extend(
            _sfem_soa_packed_apply_public_wrappers(
                function_name=function_name,
                form_name=form.name,
                dim=dim,
                n_nodes=n_nodes,
                n_qp=n_qp,
                prefix=prefix,
                local_prefix=local_prefix,
                block_name=block_name,
                quadrature_rule=quadrature_rule,
                reference_inputs=reference_inputs,
                use_tensor_product_reference=use_tensor_product_reference,
                use_tensor_product_geometry=use_tensor_product_geometry,
                use_reference_gradient_vectors=use_reference_gradient_vectors,
                omit_reference_basis_inputs=omit_reference_basis_inputs,
                stream_shape_order=stream_shape_order,
                identity_stream_shape_order=identity_stream_shape_order,
                vector_size=effective_vector_size,
                geometry_mode=geometry_mode,
                uses_current=uses_current,
                uses_direction=uses_direction,
                material_parameter_names=material_parameter_names,
                source_builder=source_builder,
            )
        )
    return lines


def _sfem_soa_packed_apply_public_wrappers(
    function_name,
    form_name,
    dim,
    n_nodes,
    n_qp,
    prefix,
    local_prefix,
    block_name,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_tensor_product_geometry,
    use_reference_gradient_vectors,
    omit_reference_basis_inputs,
    stream_shape_order,
    identity_stream_shape_order,
    vector_size,
    geometry_mode,
    uses_current,
    uses_direction,
    material_parameter_names,
    source_builder,
):
    if geometry_mode not in ("affine", "isoparametric"):
        return []
    if getattr(source_builder, "operator_extension", "cpp") != "cpp":
        return []
    if form_name not in ("gradient", "apply"):
        return []
    if form_name == "apply" and not uses_direction:
        return []

    is_affine = geometry_mode == "affine"
    reference_prefix = "%s_" % geometry_mode
    tensor_shape_name = "%sshape_1d" % reference_prefix
    tensor_grad_name = "%sgrad_1d" % reference_prefix
    tensor_weight_name = "%sq_weight_1d" % reference_prefix
    scalar_weight_name = "%sq_weight" % reference_prefix
    coordinate_streams_name = "block_coordinate_data"
    lines = ["namespace sfem {", "namespace codegen {", ""]

    for pass_mode in ("one_pass", "two_pass"):
        two_pass = pass_mode == "two_pass"
        packed_token = "_%s_packed_two_pass_" % form_name if two_pass else "_%s_packed_" % form_name
        public_base = function_name.replace("_%s_" % form_name, packed_token)
        for scalar_type in ("double", "float"):
            suffix = "" if scalar_type == "double" else "_float"
            public_name = "%s%s" % (public_base, suffix)
            lines.extend(
                [
                    'extern "C" int %s(' % public_name,
                    "        const ptrdiff_t n_packs,",
                    "        const ptrdiff_t n_elements_per_pack,",
                    "        const ptrdiff_t nelements,",
                    "        const ptrdiff_t nnodes,",
                    "        const ptrdiff_t max_nodes_per_pack,",
                    "        uint16_t **const SFEM_RESTRICT elements,",
                    "        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,",
                    "        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,",
                    "        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,",
                    "        const idx_t *const SFEM_RESTRICT ghost_idx,",
                ]
            )
            if two_pass:
                lines.extend(
                    [
                        "        const ptrdiff_t n_ghost_entries,",
                        "        const ptrdiff_t n_ghost_reduce_rows,",
                        "        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,",
                        "        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,",
                        "        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,",
                        "        %s *const SFEM_RESTRICT ghost_buf," % scalar_type,
                    ]
                )
            if is_affine:
                for stream in _soa_array_stream_names(_adjugate_input(dim)):
                    lines.append("        const geom_t *const SFEM_RESTRICT g_%s," % stream)
                lines.append("        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,")
            else:
                lines.append("        const geom_t *const *const SFEM_RESTRICT points,")
            lines.extend(
                "        const %s %s," % (scalar_type, parameter)
                for parameter in material_parameter_names
            )
            if uses_current:
                lines.append("        const ptrdiff_t u_stride,")
                for d in range(dim):
                    lines.append(
                        "        const %s *const SFEM_RESTRICT u%s,"
                        % (scalar_type, _component_name(d))
                    )
            if uses_direction:
                lines.append("        const ptrdiff_t h_stride,")
                for d in range(dim):
                    lines.append(
                        "        const %s *const SFEM_RESTRICT h%s,"
                        % (scalar_type, _component_name(d))
                    )
            lines.append("        const ptrdiff_t out_stride,")
            for d in range(dim):
                comma = "," if d + 1 < dim else ""
                lines.append(
                    "        %s *const SFEM_RESTRICT out%s%s"
                    % (scalar_type, _component_name(d), comma)
                )
            lines.extend(
                [
                    ") {",
                    "    using scalar_t = %s;" % scalar_type,
                    "    static constexpr int DIM = %d;" % dim,
                    "    static constexpr int N_QP = %d;" % n_qp,
                    "    static constexpr int N_SHAPE = %d;" % n_nodes,
                    "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
                    "    (void)nnodes;",
                    "",
                ]
            )
            if not is_affine:
                for d in range(dim):
                    lines.append(
                        "    const geom_t *const SFEM_RESTRICT %s = points[%d];"
                        % (_component_name(d), d)
                    )
                lines.extend(
                    _ordered_element_pointer_array_lines(
                        "uint16_t",
                        "coordinate_elements",
                        "elements",
                        stream_shape_order,
                        "    ",
                    )
                )
            lines.extend(
                _sfem_soa_mesh_reference_alias_lines(
                    prefix,
                    quadrature_rule,
                    reference_inputs,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    geometry_mode,
                    emit_reference_basis=True,
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
                    "#pragma omp parallel",
                    "    {",
                ]
            )
            if not is_affine:
                lines.append(
                    "        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);"
                )
            if uses_current:
                lines.append(
                    "        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);"
                )
            if uses_direction:
                lines.append(
                    "        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);"
                )
            lines.extend(
                [
                    "        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);",
                    "",
                    "#pragma omp for schedule(static)",
                    "        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                    "            const ptrdiff_t e_start = pack * n_elements_per_pack;",
                    "            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                    "            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
                ]
            )
            if two_pass:
                lines.extend(
                    [
                        "            (void)n_shared_nodes;",
                        "            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                        "            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;",
                        "            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];",
                        "            const ptrdiff_t ghost_off = ghost_ptr[pack];",
                    ]
                )
            else:
                lines.extend(
                    [
                        "            const ptrdiff_t n_shared = n_shared_nodes[pack];",
                        "            const ptrdiff_t n_not_shared = n_contiguous - n_shared;",
                        "            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                        "            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;",
                        "            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];",
                    ]
                )
            if not is_affine:
                lines.append(
                    "            const geom_t *const coordinate_components[DIM] = {%s};"
                    % ", ".join(_component_name(d) for d in range(dim))
                )
            if uses_current:
                lines.append(
                    "            const scalar_t *const u_components[DIM] = {%s};"
                    % ", ".join("u%s" % _component_name(d) for d in range(dim))
                )
            if uses_direction:
                lines.append(
                    "            const scalar_t *const h_components[DIM] = {%s};"
                    % ", ".join("h%s" % _component_name(d) for d in range(dim))
                )
            lines.extend(
                [
                    "            scalar_t *const out_components[DIM] = {%s};"
                    % ", ".join("out%s" % _component_name(d) for d in range(dim)),
                    "            for (int d = 0; d < DIM; ++d) {",
                    "                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;",
                ]
            )
            if not is_affine:
                lines.append(
                    "                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;"
                )
            if uses_current:
                lines.append(
                    "                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;"
                )
            if uses_direction:
                lines.append(
                    "                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;"
                )
            if not is_affine:
                lines.append(
                    "                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];"
                )
            if uses_current:
                lines.append(
                    "                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];"
                )
            if uses_direction:
                lines.append("                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];")
            lines.extend(
                [
                    "                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {",
                    "                    pack_component_out[k] = scalar_t(0);",
                    "                }",
                    "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                    "                    const idx_t node = owned_nodes_ptr[pack] + k;",
                ]
            )
            if not is_affine:
                lines.append("                    pack_coordinate[k] = scalar_t(coordinate_component[node]);")
            if uses_current:
                lines.append("                    pack_u_component[k] = u_component[node * u_stride];")
            if uses_direction:
                lines.append("                    pack_h_component[k] = h_component[node * h_stride];")
            lines.extend(
                [
                    "                }",
                    "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                    "                    const idx_t node = ghosts[k];",
                ]
            )
            if not is_affine:
                lines.append(
                    "                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);"
                )
            if uses_current:
                lines.append(
                    "                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];"
                )
            if uses_direction:
                lines.append(
                    "                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];"
                )
            lines.extend(
                [
                    "                }",
                    "            }",
                    "",
                    "            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {",
                    "                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);",
                ]
            )
            if uses_current:
                lines.append("                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];")
            if uses_direction:
                lines.append("                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];")
            lines.extend(
                [
                    "                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];",
                ]
            )
            if not is_affine:
                lines.append("                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
                for stream in _soa_array_stream_names(_adjugate_input(dim)):
                    lines.append("                scalar_t block_%s[N_QP * VECTOR_SIZE];" % stream)
                lines.extend(
                    [
                        "                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];",
                        "                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {%s};"
                        % ", ".join("block_jacobian_adjugate%d" % i for i in range(dim * dim)),
                    ]
                )
            if uses_current:
                lines.extend(
                    _ordered_stream_pointer_array_lines(
                            "const scalar_t *",
                            "block_u_streams",
                            "block_u_data",
                            dim,
                            stream_shape_order,
                            "                ",
                        )
                )
            if uses_direction:
                lines.extend(
                    _ordered_stream_pointer_array_lines(
                            "const scalar_t *",
                            "block_h_streams",
                            "block_h_data",
                            dim,
                            stream_shape_order,
                            "                ",
                        )
                )
            lines.extend(
                [
                    *_ordered_stream_pointer_array_lines(
                        "scalar_t *",
                        "block_out_streams",
                        "block_out_data",
                        dim,
                        stream_shape_order,
                        "                ",
                    ),
                    "",
                    "                for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    "                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];",
                    *([] if identity_stream_shape_order else ["                    const uint16_t *const SFEM_RESTRICT coordinate_shape = coordinate_elements[shape];"]),
                    "                    for (int d = 0; d < DIM; ++d) {",
                    "#pragma omp simd",
                    "                        for (int lane = 0; lane < nelems; ++lane) {",
                    "                            const uint16_t packed_node = element_shape[evbegin + lane];",
                    *([] if identity_stream_shape_order else ["                            const uint16_t coordinate_packed_node = coordinate_shape[evbegin + lane];"]),
                ]
            )
            if not is_affine:
                lines.append(
                    "                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + %s];"
                    % ("packed_node" if identity_stream_shape_order else "coordinate_packed_node")
                )
            if uses_current:
                lines.append(
                    "                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];"
                )
            if uses_direction:
                lines.append(
                    "                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];"
                )
            lines.extend(
                [
                    "                            block_out_data[shape * DIM + d][lane] = scalar_t(0);",
                    "                        }",
                    "                    }",
                    "                }",
                    "",
                ]
            )
            if is_affine:
                lines.extend(
                    _sfem_soa_affine_geometry_stream_lines(
                        source_builder,
                        (_adjugate_input(dim), sfem_soa_reference_input("jacobian_determinant", 1, 1, 1)),
                        "                ",
                        geometry_scalar_type="geom_t",
                    )
                )
            elif use_tensor_product_geometry:
                lines.extend(
                    tensor_product_gradient_isoparametric_geometry_lines(
                        dim=dim,
                        n_shape=n_nodes,
                        n_qp=quadrature_rule.n_qp,
                        local_prefix=local_prefix,
                        coordinate_streams="block_coordinate_data",
                        contiguous_coordinate_streams=True,
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
            else:
                lines.extend(["", "                for (int q = 0; q < N_QP; ++q) {"])
                geometry_lines = _sfem_soa_isoparametric_geometry_lines(
                    dim,
                    n_nodes,
                    quadrature_rule,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    reference_inputs,
                    q_major=True,
                    reference_prefix=reference_prefix,
                    source_builder=source_builder,
                    coordinate_streams=coordinate_streams_name,
                )
                lines.extend("    %s" % line if line else line for line in geometry_lines)
                lines.append("                }")

            call_args = [
                "nelems",
                "0" if is_affine else "VECTOR_SIZE",
                *("block_jacobian_adjugate%d" % i for i in range(dim * dim)),
                "block_jacobian_determinant0",
            ]
            if omit_reference_basis_inputs:
                pass
            elif use_tensor_product_reference:
                call_args.extend((tensor_shape_name, tensor_grad_name))
            elif use_reference_gradient_vectors:
                call_args.extend(
                    "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
                    for component in range(dim)
                )
            else:
                call_args.extend(
                    "%s%s" % (reference_prefix, array_input.name)
                    for array_input in reference_inputs
                )
            call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
            call_args.extend(material_parameter_names)
            if uses_current:
                call_args.append("block_u_streams")
            if uses_direction:
                call_args.append("block_h_streams")
            call_args.append("block_out_streams")
            lines.extend(
                [
                    "",
                    "                %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
                    % (block_name, ", ".join(call_args)),
                    "",
                    "                for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    "                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];",
                    "                    for (int d = 0; d < DIM; ++d) {",
                    "                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;",
                    "                        for (int lane = 0; lane < nelems; ++lane) {",
                    "                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];",
                    "                        }",
                    "                    }",
                    "                }",
                    "            }",
                    "",
                ]
            )
            if two_pass:
                lines.extend(
                    [
                        "            for (int d = 0; d < DIM; ++d) {",
                        "                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;",
                        "                scalar_t *const SFEM_RESTRICT global_out = out_components[d];",
                        "                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;",
                        "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                        "                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];",
                        "                    pack_component_out[k] = scalar_t(0);",
                        "                }",
                        "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                        "                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];",
                        "                    pack_component_out[n_contiguous + k] = scalar_t(0);",
                        "                }",
                        "            }",
                        "        }",
                    ]
                )
            else:
                lines.extend(
                    [
                        "            for (int d = 0; d < DIM; ++d) {",
                        "                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;",
                        "                scalar_t *const SFEM_RESTRICT global_out = out_components[d];",
                        "                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {",
                        "                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];",
                        "                    pack_component_out[k] = scalar_t(0);",
                        "                }",
                        "                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {",
                        "#pragma omp atomic update",
                        "                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];",
                        "                    pack_component_out[k] = scalar_t(0);",
                        "                }",
                        "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                        "#pragma omp atomic update",
                        "                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];",
                        "                    pack_component_out[n_contiguous + k] = scalar_t(0);",
                        "                }",
                        "            }",
                        "        }",
                    ]
                )
            if two_pass:
                lines.extend(
                    [
                        "    }",
                        "",
                        "    scalar_t *const out_components[DIM] = {%s};"
                        % ", ".join("out%s" % _component_name(d) for d in range(dim)),
                        "#pragma omp parallel for schedule(static)",
                        "    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {",
                        "        const idx_t dest = ghost_reduce_dest[row];",
                        "        const ptrdiff_t begin = ghost_reduce_ptr[row];",
                        "        const ptrdiff_t end = ghost_reduce_ptr[row + 1];",
                        "        for (int d = 0; d < DIM; ++d) {",
                        "            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;",
                        "            scalar_t sum = scalar_t(0);",
                        "            for (ptrdiff_t j = begin; j < end; ++j) {",
                        "                sum += ghost_component[ghost_reduce_idx[j]];",
                        "            }",
                        "            out_components[d][dest * out_stride] += sum;",
                        "        }",
                        "    }",
                        "    return SFEM_SUCCESS;",
                        "}",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "    }",
                        "    return SFEM_SUCCESS;",
                        "}",
                        "",
                    ]
                )

    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])
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
    work_item = _work_item_index(source_builder)
    material_parameter_names = _form_material_parameter_names(form)

    function_name = _sfem_soa_mesh_public_function_name(
        prefix,
        "objective_steps",
        quadrature_rule,
        geometry_mode,
    )
    implementation_name = "%s_impl" % function_name
    block_name = "%s_objective_block" % local_prefix
    specialized_prefix = _constant_p1_specialized_local_prefix(
        local_prefix,
        quadrature_rule,
    )
    if specialized_prefix is not None:
        block_name = "%s_objective_block" % specialized_prefix
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
    omit_reference_basis_inputs = (
        specialized_prefix is not None and not use_tensor_product_reference
    )
    use_stream_arrays = use_shared_weak_local
    compact_coordinate_buffers = geometry_mode == "isoparametric"
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )
    identity_stream_shape_order = tuple(stream_shape_order) == tuple(range(n_nodes))
    coordinate_streams_name = "block_coordinate_data"

    base_params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
    ]
    if geometry_mode == "affine":
        base_params.extend(
            "const jacobian_t *const SFEM_RESTRICT g_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        base_params.append("const geometry_t *const *const SFEM_RESTRICT points")

    material_params = _form_material_parameter_declarations(form)
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
        param.replace("geometry_t", "geom_t").replace("jacobian_t", "geom_t")
        for param in impl_params
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
        if compact_coordinate_buffers:
            lines.extend(
                _ordered_element_pointer_array_lines(
                    "idx_t",
                    "coordinate_elements",
                    "elements",
                    stream_shape_order,
                    "    ",
                )
            )
    lines.extend(
        _sfem_soa_mesh_reference_alias_lines(
            prefix,
            quadrature_rule,
            reference_inputs,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
            geometry_mode,
            emit_reference_basis=(
                not omit_reference_basis_inputs or geometry_mode == "isoparametric"
            ),
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
        if compact_coordinate_buffers:
            lines.append("        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
    elif compact_coordinate_buffers:
        lines.append("        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];")
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
            *_work_item_loop_lines(source_builder, "            "),
            "                ev[element_node * VECTOR_SIZE + %s] = element_shape[evbegin + %s];"
            % (work_item, work_item),
            "            }",
            "        }",
        ]
    )

    if geometry_mode == "isoparametric":
        if compact_coordinate_buffers:
            lines.append("        const geometry_t *const coordinate_components[DIM] = {%s};" % ", ".join(_component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "",
                    "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                    *([] if identity_stream_shape_order else ["            const idx_t *const SFEM_RESTRICT coordinate_element_shape = coordinate_elements[shape];"]),
                    "            for (int d = 0; d < DIM; ++d) {",
                    *_work_item_loop_lines(source_builder, "                "),
                    "                    block_coordinate_data[shape * DIM + d][%s] = coordinate_components[d][%s];"
                    % (
                        work_item,
                        "ev[shape * VECTOR_SIZE + %s]" % work_item
                        if identity_stream_shape_order
                        else "coordinate_element_shape[evbegin + %s]" % work_item,
                    ),
                    "                }",
                    "            }",
                    "        }",
                ]
            )
        else:
            lines.extend(["", *_work_item_loop_lines(source_builder, "        ")])
            for shape in range(n_nodes):
                for d in range(dim):
                    stream = "%s%d" % (_component_name(d), shape)
                    lines.append(
                        "            block_%s[%s] = %s[ev[%d * VECTOR_SIZE + %s]];"
                        % (stream, work_item, _component_name(d), shape, work_item)
                    )
            lines.append("        }")

    if compact_stream_buffers:
        lines.append("")
        lines.append("        const scalar_t *const u_components[DIM] = {%s};" % ", ".join("u%s" % _component_name(d) for d in range(dim)))
        lines.append("        const scalar_t *const h_components[DIM] = {%s};" % ", ".join("h%s" % _component_name(d) for d in range(dim)))
        lines.extend(
            [
                *_ordered_stream_pointer_array_lines(
                    "const scalar_t *",
                    "block_u_streams",
                    "block_u_data",
                    dim,
                    stream_shape_order,
                    "        ",
                ),
            ]
        )
        lines.extend(
            [
                "",
                "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "            for (int d = 0; d < DIM; ++d) {",
                *_work_item_loop_lines(source_builder, "                "),
                "                    const idx_t node = ev[shape * VECTOR_SIZE + %s];" % work_item,
                "                    block_u_base_data[shape * DIM + d][%s] = u_components[d][node * u_stride];" % work_item,
                "                    block_h_data[shape * DIM + d][%s] = h_components[d][node * h_stride];" % work_item,
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
        lines.extend(["", *_work_item_loop_lines(source_builder, "        ")])
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                lines.append(
                    "            block_u%s%d_base[%s] = u%s[ev[%d * VECTOR_SIZE + %s] * u_stride];"
                    % (component, shape, work_item, component, shape, work_item)
                )
                lines.append(
                    "            block_h%s%d[%s] = h%s[ev[%d * VECTOR_SIZE + %s] * h_stride];"
                    % (component, shape, work_item, component, shape, work_item)
                )
        lines.append("        }")

    if geometry_mode == "isoparametric" and use_tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_nodes,
                n_qp=quadrature_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinate_data",
                contiguous_coordinate_streams=True,
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
                coordinate_streams=coordinate_streams_name,
            )
        )
        lines.append("        }")
    elif geometry_mode == "affine":
        lines.extend(
            _sfem_soa_affine_geometry_stream_lines(
                source_builder,
                element_inputs,
                "        ",
            )
        )

    call_args = ["nelems"]
    call_args.append("0" if geometry_mode == "affine" else "VECTOR_SIZE")
    if geometry_mode == "affine":
        call_args.extend(
            "block_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        call_args.extend(
            "block_%s" % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    if omit_reference_basis_inputs:
        pass
    elif use_tensor_product_reference:
        call_args.extend((tensor_shape_name, tensor_grad_name))
    elif use_reference_gradient_vectors:
        call_args.extend(
            "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
            for component in range(dim)
        )
    else:
        call_args.extend("%s%s" % (reference_prefix, array_input.name) for array_input in reference_inputs)
    call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
    call_args.extend(material_parameter_names)
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
                *_work_item_loop_lines(source_builder, "                    "),
                "                        block_u_data[shape * DIM + d][%s] = block_u_base_data[shape * DIM + d][%s] + alpha * block_h_data[shape * DIM + d][%s];"
                % (work_item, work_item, work_item),
                "                    }",
                "                }",
                "            }",
            ]
        )
    else:
        lines.extend(_work_item_loop_lines(source_builder, "            "))
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                lines.append(
                    "                block_u%s%d[%s] = block_u%s%d_base[%s] + alpha * block_h%s%d[%s];"
                    % (
                        component,
                        shape,
                        work_item,
                        component,
                        shape,
                        work_item,
                        component,
                        shape,
                        work_item,
                    )
                )
        lines.append("            }")
    lines.extend(
        [
            *_work_item_loop_lines(source_builder, "            "),
            "                block_value[%s] = scalar_t(0);" % work_item,
            "            }",
            "",
            "            %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_name, ", ".join(call_args)),
            "",
            *_work_item_loop_lines(source_builder, "            "),
            "                value[(ptrdiff_t)step * nelements + evbegin + %s] = block_value[%s];"
            % (work_item, work_item),
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
                    ", geom_t",
                    wrapper_args,
                ),
                "}",
                "",
            ]
        )
    lines.extend(
        _sfem_soa_packed_objective_steps_public_wrappers(
            function_name=function_name,
            dim=dim,
            n_nodes=n_nodes,
            n_qp=n_qp,
            prefix=prefix,
            local_prefix=local_prefix,
            block_name=block_name,
            quadrature_rule=quadrature_rule,
            reference_inputs=reference_inputs,
            use_tensor_product_reference=use_tensor_product_reference,
            use_tensor_product_geometry=use_tensor_product_geometry,
            use_reference_gradient_vectors=use_reference_gradient_vectors,
            omit_reference_basis_inputs=omit_reference_basis_inputs,
            stream_shape_order=stream_shape_order,
            identity_stream_shape_order=identity_stream_shape_order,
            vector_size=vector_size,
            geometry_mode=geometry_mode,
            material_parameter_names=material_parameter_names,
            source_builder=source_builder,
        )
    )
    return lines


def _sfem_soa_direct_hessian_matrix_assembly_lines(
    form,
    dim,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_prefix,
    indent,
    emit_tensor_product_static_constants=True,
):
    if _form_uses_current(form, default=True):
        raise ValueError("direct hessian matrix assembly currently requires no current field")
    weak_form = form.weak_form
    material = _weak_form_material_expression(
        weak_form,
        form.name,
        _weak_form_deformation_gradient_substitutions(weak_form, "grad_u"),
        tuple(sp.symbols("trial_grad[%d]" % i) for i in range(dim * dim)),
    )
    lines = [
        "%sfor (int entry = 0; entry < NDOFS * NDOFS; ++entry) {" % indent,
        "%s    element_matrix[entry] = scalar_t(0);" % indent,
        "%s}" % indent,
    ]
    if use_tensor_product_reference and emit_tensor_product_static_constants:
        lines.extend(
            [
                "%sstatic constexpr int N_QP_1D = %d;"
                % (indent, quadrature_rule.tensor_product_n_qp_1d),
                "%sstatic constexpr int N_SHAPE_1D = %d;"
                % (indent, quadrature_rule.tensor_product_n_shape_1d),
            ]
        )
    lines.append("%sfor (int q = 0; q < N_QP; ++q) {" % indent)
    if use_tensor_product_reference:
        lines.extend(_tensor_product_q_index_lines(dim, indent + "    "))
        lines.append(
            "%s    const scalar_t qw = %s;"
            % (indent, _tensor_product_quadrature_weight_expr(dim, "%sq_weight_1d" % reference_prefix))
        )
    else:
        lines.append("%s    const scalar_t qw = %sq_weight[q];" % (indent, reference_prefix))
    lines.extend(
        [
            "%s    const int lane = 0;" % indent,
            "%s    const ptrdiff_t geometry_offset = q * VECTOR_SIZE + lane;" % indent,
        ]
    )
    for component in range(dim * dim):
        lines.append(
            "%s    const scalar_t jacobian_adjugate_lane%d = block_jacobian_adjugate%d[geometry_offset];"
            % (indent, component, component)
        )
    lines.extend(
        [
            "%s    const scalar_t jacobian_determinant_lane0 = block_jacobian_determinant0[geometry_offset];"
            % indent,
            "%s    const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;"
            % indent,
            "%s    for (int trial_component = 0; trial_component < DIM; ++trial_component) {"
            % indent,
            "%s        for (int trial_shape = 0; trial_shape < N_SHAPE; ++trial_shape) {"
            % indent,
        ]
    )
    if use_tensor_product_reference:
        lines.extend(
            _tensor_product_shape_coordinate_lines(
                dim,
                "trial_shape",
                "trial",
                "%s            " % indent,
            )
        )
    for ref_component in range(dim):
        lines.append(
            "%s            const scalar_t trial_grad_ref%d = %s;"
            % (
                indent,
                ref_component,
                _sfem_soa_reference_gradient_expr_for_shape(
                    dim,
                    ref_component,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    reference_inputs,
                    "trial_shape",
                    coord_prefix="trial",
                    reference_prefix=reference_prefix,
                ),
            )
        )
    lines.extend(
        [
            "%s            scalar_t trial_grad[DIM * DIM];" % indent,
            "%s            for (int i = 0; i < DIM * DIM; ++i) {" % indent,
            "%s                trial_grad[i] = scalar_t(0);" % indent,
            "%s            }" % indent,
        ]
    )
    for phys_component in range(dim):
        terms = [
            "trial_grad_ref%d * jacobian_adjugate_lane%d"
            % (ref_component, ref_component * dim + phys_component)
            for ref_component in range(dim)
        ]
        lines.append(
            "%s            trial_grad[trial_component * DIM + %d] = (%s) * inv_jacobian_determinant;"
            % (indent, phys_component, " + ".join(terms))
        )
    lines.append("%s            scalar_t material[DIM * DIM];" % indent)
    local_material_lines = []
    _append_cse_array_assignments(
        local_material_lines,
        tuple(material),
        ["material[%d] =" % i for i in range(dim * dim)],
        "weak_hess_tmp",
    )
    lines.extend("%s            %s" % (indent, line.strip()) for line in local_material_lines)
    lines.extend(
        [
            "%s            for (int test_component = 0; test_component < DIM; ++test_component) {"
            % indent,
            "%s                for (int test_shape = 0; test_shape < N_SHAPE; ++test_shape) {"
            % indent,
        ]
    )
    if use_tensor_product_reference:
        lines.extend(
            _tensor_product_shape_coordinate_lines(
                dim,
                "test_shape",
                "test",
                "%s                    " % indent,
            )
        )
    for ref_component in range(dim):
        lines.append(
            "%s                    const scalar_t test_grad_ref%d = %s;"
            % (
                indent,
                ref_component,
                _sfem_soa_reference_gradient_expr_for_shape(
                    dim,
                    ref_component,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    reference_inputs,
                    "test_shape",
                    coord_prefix="test",
                    reference_prefix=reference_prefix,
                ),
            )
        )
    lines.append("%s                    scalar_t entry = scalar_t(0);" % indent)
    for ref_component in range(dim):
        terms = [
            "material[test_component * DIM + %d] * jacobian_adjugate_lane%d"
            % (k, ref_component * dim + k)
            for k in range(dim)
        ]
        lines.append(
            "%s                    entry += test_grad_ref%d * qw * (%s);"
            % (indent, ref_component, " + ".join(terms))
        )
    lines.extend(
        [
            "%s                    const int row = test_component * N_SHAPE + test_shape;"
            % indent,
            "%s                    const int col = trial_component * N_SHAPE + trial_shape;"
            % indent,
            "%s                    element_matrix[row * NDOFS + col] += entry;" % indent,
            "%s                }" % indent,
            "%s            }" % indent,
            "%s        }" % indent,
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    return lines


def _sfem_soa_direct_hessian_element_matrix_call_lines(
    function_name,
    dim,
    material_parameter_names,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_inputs,
    tensor_shape_name,
    tensor_grad_name,
    tensor_weight_name,
    scalar_weight_name,
    reference_prefix,
    indent,
):
    args = [
        *("block_jacobian_adjugate%d" % i for i in range(dim * dim)),
        "block_jacobian_determinant0",
    ]
    if use_tensor_product_reference:
        args.extend((tensor_shape_name, tensor_grad_name, tensor_weight_name))
    elif use_reference_gradient_vectors:
        args.extend(
            "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
            for component in range(dim)
        )
        args.append(scalar_weight_name)
    else:
        args.extend(
            "%s%s" % (reference_prefix, array_input.name)
            for array_input in reference_inputs
        )
        args.append(scalar_weight_name)
    args.extend(material_parameter_names)
    args.append("element_matrix")
    return (
        "%s%s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
        % (indent, function_name, ", ".join(args)),
    )


def _sfem_soa_hessian_matrix_assembly_function(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    local_prefix,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    geometry_family=None,
    use_shared_weak_local=False,
    matrix_format_plan=None,
    source_builder=None,
):
    formats = _matrix_formats_from_plan(matrix_format_plan)
    if not formats:
        return []
    if form.name != "apply" or form.weak_form is None:
        return []
    if quadrature_rule is None:
        raise ValueError("hessian matrix assembly requires an element quadrature rule")
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    packed_crs_passes = _packed_crs_passes(matrix_format_plan)
    material_parameter_names = _form_material_parameter_names(form)
    uses_current = _form_uses_current(form, default=True)

    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_tensor_product_geometry = str(geometry_family) == "tensor_product"
    use_reference_gradient_vectors = (
        not use_tensor_product_reference
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    omit_reference_basis_inputs = (
        _constant_p1_specialized_local_prefix(local_prefix, quadrature_rule) is not None
        and not use_tensor_product_reference
    )
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )
    identity_stream_shape_order = tuple(stream_shape_order) == tuple(range(n_nodes))
    coordinate_streams_name = "block_coordinate_data"
    block_name = "%s_apply_block" % local_prefix
    specialized_prefix = _constant_p1_specialized_local_prefix(
        local_prefix,
        quadrature_rule,
    )
    if specialized_prefix is not None and not use_tensor_product_reference:
        block_name = "%s_apply_block" % specialized_prefix
    direct_hessian_assembly = _sfem_soa_direct_hessian_matrix_assembly_available(
        form,
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    direct_hessian_function_name = _sfem_soa_direct_hessian_function_name(
        local_prefix,
        use_tensor_product_reference,
    )

    function_base = _sfem_soa_hessian_matrix_public_function_base(
        prefix,
        quadrature_rule,
        "isoparametric",
    )
    implementation_name = "%s_assemble_impl" % function_base
    reference_prefix = "isoparametric_"
    tensor_shape_name = "%sshape_1d" % reference_prefix
    tensor_grad_name = "%sgrad_1d" % reference_prefix
    tensor_weight_name = "%sq_weight_1d" % reference_prefix
    scalar_weight_name = "%sq_weight" % reference_prefix

    lines = []
    lines.extend(_sfem_soa_hessian_scatter_lines(function_base, dim, n_nodes, formats))
    if packed_crs_passes:
        lines.extend(
            _sfem_soa_hessian_packed_crs_helper_lines(
                function_base,
                dim,
                n_nodes,
            )
        )
    lines.extend(
        [
            "template <typename scalar_t, typename geometry_t, int FORMAT>",
            "static int %s(" % implementation_name,
            "        const ptrdiff_t nelements,",
            "        const ptrdiff_t nnodes,",
            "        idx_t **const SFEM_RESTRICT elements,",
            "        const geometry_t *const *const SFEM_RESTRICT points,",
        ]
    )
    lines.extend(
        "        const scalar_t %s," % parameter
        for parameter in material_parameter_names
    )
    if uses_current:
        lines.append("        const ptrdiff_t u_stride,")
        for d in range(dim):
            comma = "," if d + 1 < dim else ","
            lines.append(
                "        const scalar_t *const SFEM_RESTRICT u%s%s"
                % (_component_name(d), comma)
            )
    lines.extend(
        [
            "        const count_t *const SFEM_RESTRICT rowptr,",
            "        const idx_t *const SFEM_RESTRICT colidx,",
            "        scalar_t *const SFEM_RESTRICT values,",
            "        const int *const SFEM_RESTRICT diag_offsets,",
            "        const ptrdiff_t ndiag,",
            "        const ptrdiff_t coo_nnz,",
            "        const idx_t *const SFEM_RESTRICT coo_rows,",
            "        const idx_t *const SFEM_RESTRICT coo_cols,",
            "        idx_t *const SFEM_RESTRICT coo_triplet_rows,",
            "        idx_t *const SFEM_RESTRICT coo_triplet_cols) {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % n_qp,
            "    static constexpr int N_SHAPE = %d;" % n_nodes,
            "    static constexpr int VECTOR_SIZE = 1;",
            "    static constexpr int NDOFS = DIM * N_SHAPE;",
            "    (void)nnodes;",
        ]
    )
    if uses_current:
        lines.append(
            "    const scalar_t *const u_components[DIM] = {%s};"
            % ", ".join("u%s" % _component_name(d) for d in range(dim))
        )
    for d in range(dim):
        lines.append(
            "    const geometry_t *const SFEM_RESTRICT %s = points[%d];"
            % (_component_name(d), d)
        )
    lines.extend(
        _ordered_element_pointer_array_lines(
            "idx_t",
            "coordinate_elements",
            "elements",
            stream_shape_order,
            "    ",
        )
    )
    lines.extend(
        _sfem_soa_mesh_reference_alias_lines(
            prefix,
            quadrature_rule,
            reference_inputs,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
            "isoparametric",
            emit_reference_basis=True,
        )
    )
    lines.extend(
        [
            "",
            "    int invalid_matrix_graph = 0;",
            "#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)",
            "    for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "        idx_t ev[N_SHAPE];",
            "        scalar_t element_matrix[NDOFS * NDOFS];",
            "        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];",
            "        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];",
            "        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];",
            "        static constexpr int nelems = VECTOR_SIZE;",
        ]
    )
    if uses_current:
        lines.append("        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];")
    for stream in _soa_array_stream_names(_adjugate_input(dim)):
        lines.append("        scalar_t block_%s[N_QP * VECTOR_SIZE];" % stream)
    lines.append("        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];")
    lines.append(
            "        scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {%s};"
            % ", ".join("block_jacobian_adjugate%d" % i for i in range(dim * dim))
        )
    if uses_current:
        lines.extend(
            _ordered_stream_pointer_array_lines(
                "const scalar_t *",
                "block_u_streams",
                "block_u_data",
                dim,
                stream_shape_order,
                "        ",
            )
        )
    lines.extend(
        [
            *_ordered_stream_pointer_array_lines(
                "const scalar_t *",
                "block_h_streams",
                "block_h_data",
                dim,
                stream_shape_order,
                "        ",
            ),
            *_ordered_stream_pointer_array_lines(
                "scalar_t *",
                "block_out_streams",
                "block_out_data",
                dim,
                stream_shape_order,
                "        ",
            ),
            "",
            "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
            "            const idx_t node = elements[shape][element];",
            *([] if identity_stream_shape_order else ["            const idx_t coordinate_node = coordinate_elements[shape][element];"]),
            "            ev[shape] = node;",
            "            for (int d = 0; d < DIM; ++d) {",
            "                block_coordinate_data[shape * DIM + d][0] = scalar_t(points[d][%s]);" % ("node" if identity_stream_shape_order else "coordinate_node"),
        ]
    )
    if uses_current:
        lines.append("                block_u_data[shape * DIM + d][0] = u_components[d][node * u_stride];")
    lines.extend(
        [
            "            }",
            "        }",
            "",
        ]
    )
    if use_tensor_product_geometry:
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_nodes,
                n_qp=quadrature_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinate_data",
                contiguous_coordinate_streams=True,
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
    else:
        lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
        lines.extend(
            _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_inputs,
                q_major=True,
                reference_prefix=reference_prefix,
                source_builder=source_builder,
                coordinate_streams=coordinate_streams_name,
            )
        )
        lines.append("        }")

    call_args = [
        "1",
        "1",
        *("block_jacobian_adjugate%d" % i for i in range(dim * dim)),
        "block_jacobian_determinant0",
    ]
    if omit_reference_basis_inputs:
        pass
    elif use_tensor_product_reference:
        call_args.extend((tensor_shape_name, tensor_grad_name))
    elif use_reference_gradient_vectors:
        call_args.extend(
            "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
            for component in range(dim)
        )
    else:
        call_args.extend(
            "%s%s" % (reference_prefix, array_input.name)
            for array_input in reference_inputs
        )
    call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
    call_args.extend(material_parameter_names)
    if uses_current:
        call_args.append("block_u_streams")
    call_args.extend(("block_h_streams", "block_out_streams"))

    lines.append("")
    if direct_hessian_assembly:
        lines.extend(
            _sfem_soa_direct_hessian_element_matrix_call_lines(
                direct_hessian_function_name,
                dim,
                material_parameter_names,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_inputs,
                tensor_shape_name,
                tensor_grad_name,
                tensor_weight_name,
                scalar_weight_name,
                reference_prefix,
                "        ",
            )
        )
    else:
        lines.extend(
            [
                "        for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {",
                "            element_matrix[entry] = scalar_t(0);",
                "        }",
                "",
                "        for (int trial_component = 0; trial_component < DIM; ++trial_component) {",
                "            for (int trial_shape = 0; trial_shape < N_SHAPE; ++trial_shape) {",
                "                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                "                    block_h_data[stream][0] = scalar_t(0);",
                "                    block_out_data[stream][0] = scalar_t(0);",
                "                }",
                "                block_h_data[trial_shape * DIM + trial_component][0] = scalar_t(1);",
                "                %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
                % (block_name, ", ".join(call_args)),
                "                const int col = trial_component * N_SHAPE + trial_shape;",
                "                for (int test_component = 0; test_component < DIM; ++test_component) {",
                "                    for (int test_shape = 0; test_shape < N_SHAPE; ++test_shape) {",
                "                        const int row = test_component * N_SHAPE + test_shape;",
                "                        element_matrix[row * NDOFS + col] = block_out_data[test_shape * DIM + test_component][0];",
                "                    }",
                "                }",
                "            }",
                "        }",
            ]
        )
    lines.append("")
    lines.extend(_sfem_soa_hessian_scatter_dispatch_lines(function_base, formats, "        "))
    lines.extend(
        [
            "    }",
            "",
            "    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;",
            "}",
            "",
        ]
    )
    if packed_crs_passes:
        packed_fill_impl = "%s_packed_fill_impl" % function_base
        packed_discover_impl = "%s_packed_discover_impl" % function_base
        packed_common_params = [
            "const ptrdiff_t n_packs",
            "const ptrdiff_t n_elements_per_pack",
            "const ptrdiff_t nelements",
            "const ptrdiff_t nnodes",
            "const ptrdiff_t max_nodes_per_pack",
            "uint16_t **const SFEM_RESTRICT elements",
            "const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr",
            "const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes",
            "const ptrdiff_t *const SFEM_RESTRICT ghost_ptr",
            "const idx_t *const SFEM_RESTRICT ghost_idx",
        ]
        packed_state_params = [
            "const geometry_t *const *const SFEM_RESTRICT points",
        ]
        packed_state_params.extend(
            "const scalar_t %s" % parameter
            for parameter in material_parameter_names
        )
        if uses_current:
            packed_state_params.append("const ptrdiff_t u_stride")
            packed_state_params.extend(
                "const scalar_t *const SFEM_RESTRICT u%s" % _component_name(d)
                for d in range(dim)
            )
        packed_fill_params = tuple(
            packed_common_params
            + packed_state_params
            + [
                "const count_t *const SFEM_RESTRICT packed_element_entries",
                "scalar_t *const SFEM_RESTRICT values",
            ]
        )
        packed_discover_params = tuple(
            packed_common_params
            + [
                "const count_t *const SFEM_RESTRICT rowptr",
                "const idx_t *const SFEM_RESTRICT colidx",
                "count_t *const SFEM_RESTRICT packed_element_entries",
            ]
        )
        lines.extend(
            [
                "template <typename scalar_t, typename geometry_t>",
                "static int %s(" % packed_discover_impl,
            ]
        )
        for idx, param in enumerate(packed_discover_params):
            comma = "," if idx + 1 < len(packed_discover_params) else ""
            lines.append("        %s%s" % (param, comma))
        lines.extend(
            [
                ") {",
                "    static constexpr int DIM = %d;" % dim,
                "    static constexpr int N_SHAPE = %d;" % n_nodes,
                "    (void)nnodes;",
                "    (void)max_nodes_per_pack;",
                "    (void)n_shared_nodes;",
                "    int invalid_matrix_graph = 0;",
                "#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)",
                "    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "        const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "        const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "        for (ptrdiff_t element = e_start; element < e_end; ++element) {",
                "            idx_t ev[N_SHAPE];",
                "            for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "                ev[shape] = %s_packed_global_node(elements[shape][element], pack, owned_nodes_ptr, ghost_ptr, ghost_idx);" % function_base,
                "            }",
                "            count_t *const entries = &packed_element_entries[element * (DIM * N_SHAPE) * (DIM * N_SHAPE)];",
                "            invalid_matrix_graph |= (%s_discover_packed_crs_entries<scalar_t>(ev, rowptr, colidx, entries) != SFEM_SUCCESS);" % function_base,
                "        }",
                "    }",
                "    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;",
                "}",
                "",
                "template <typename scalar_t, typename geometry_t>",
                "static int %s(" % packed_fill_impl,
            ]
        )
        for idx, param in enumerate(packed_fill_params):
            comma = "," if idx + 1 < len(packed_fill_params) else ""
            lines.append("        %s%s" % (param, comma))
        lines.extend(
            [
                ") {",
                "    static constexpr int DIM = %d;" % dim,
                "    static constexpr int N_QP = %d;" % n_qp,
                "    static constexpr int N_SHAPE = %d;" % n_nodes,
                "    static constexpr int VECTOR_SIZE = 1;",
                "    static constexpr int NDOFS = DIM * N_SHAPE;",
                "    (void)nnodes;",
                "    (void)n_shared_nodes;",
            ]
        )
        for d in range(dim):
            lines.append(
                "    const geometry_t *const SFEM_RESTRICT %s = points[%d];"
                % (_component_name(d), d)
            )
        lines.extend(
            _ordered_element_pointer_array_lines(
                "uint16_t",
                "coordinate_elements",
                "elements",
                stream_shape_order,
                "    ",
            )
        )
        lines.extend(
            _sfem_soa_mesh_reference_alias_lines(
                prefix,
                quadrature_rule,
                reference_inputs,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                "isoparametric",
                emit_reference_basis=True,
            )
        )
        lines.extend(
            [
                "",
                "#pragma omp parallel",
                "    {",
                "        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);",
            ]
        )
        if uses_current:
            lines.append(
                "        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);"
            )
        lines.extend(
            [
                "",
                "#pragma omp for schedule(static)",
                "        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "            const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
                "            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                "            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];",
                "            const geometry_t *const coordinate_components[DIM] = {%s};"
                % ", ".join(_component_name(d) for d in range(dim)),
            ]
        )
        if uses_current:
            lines.append(
                "            const scalar_t *const u_components[DIM] = {%s};"
                % ", ".join("u%s" % _component_name(d) for d in range(dim))
            )
        lines.extend(
            [
                "            for (int d = 0; d < DIM; ++d) {",
                "                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;",
                "                const geometry_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];",
            ]
        )
        if uses_current:
            lines.extend(
                [
                    "                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;",
                    "                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];",
                ]
            )
        lines.extend(
            [
                "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                "                    const idx_t node = owned_nodes_ptr[pack] + k;",
                "                    pack_coordinate[k] = scalar_t(coordinate_component[node]);",
            ]
        )
        if uses_current:
            lines.append("                    pack_u_component[k] = u_component[node * u_stride];")
        lines.extend(
            [
                "                }",
                "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                "                    const idx_t node = ghosts[k];",
                "                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);",
            ]
        )
        if uses_current:
            lines.append("                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];")
        lines.extend(
            [
                "                }",
                "            }",
                "",
                "            for (ptrdiff_t element = e_start; element < e_end; ++element) {",
                "                scalar_t element_matrix[NDOFS * NDOFS];",
                "                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];",
                "                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];",
                "                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];",
                "                static constexpr int nelems = VECTOR_SIZE;",
            ]
        )
        if uses_current:
            lines.append("                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];")
        for stream in _soa_array_stream_names(_adjugate_input(dim)):
            lines.append("                scalar_t block_%s[N_QP * VECTOR_SIZE];" % stream)
        lines.append("                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];")
        lines.append(
            "                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {%s};"
            % ", ".join("block_jacobian_adjugate%d" % i for i in range(dim * dim))
        )
        if uses_current:
            lines.extend(
                _ordered_stream_pointer_array_lines(
                    "const scalar_t *",
                    "block_u_streams",
                    "block_u_data",
                    dim,
                    stream_shape_order,
                    "                ",
                )
            )
        lines.extend(
            [
                *_ordered_stream_pointer_array_lines(
                    "const scalar_t *",
                    "block_h_streams",
                    "block_h_data",
                    dim,
                    stream_shape_order,
                    "                ",
                ),
                *_ordered_stream_pointer_array_lines(
                    "scalar_t *",
                    "block_out_streams",
                    "block_out_data",
                    dim,
                    stream_shape_order,
                    "                ",
                ),
                "",
                "                for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "                    const uint16_t packed_node = elements[shape][element];",
                *([] if identity_stream_shape_order else ["                    const uint16_t coordinate_packed_node = coordinate_elements[shape][element];"]),
                "                    for (int d = 0; d < DIM; ++d) {",
                "                        block_coordinate_data[shape * DIM + d][0] = pack_coordinates[d * max_nodes_per_pack + %s];" % ("packed_node" if identity_stream_shape_order else "coordinate_packed_node"),
            ]
        )
        if uses_current:
            lines.append("                        block_u_data[shape * DIM + d][0] = pack_u[d * max_nodes_per_pack + packed_node];")
        lines.extend(
            [
                "                    }",
                "                }",
                "",
            ]
        )
        if use_tensor_product_geometry:
            lines.extend(
                tensor_product_gradient_isoparametric_geometry_lines(
                    dim=dim,
                    n_shape=n_nodes,
                    n_qp=quadrature_rule.n_qp,
                    local_prefix=local_prefix,
                    coordinate_streams="block_coordinate_data",
                    contiguous_coordinate_streams=True,
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
        else:
            lines.extend(["", "            for (int q = 0; q < N_QP; ++q) {"])
            geometry_lines = _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_inputs,
                q_major=True,
                reference_prefix=reference_prefix,
                source_builder=source_builder,
                coordinate_streams=coordinate_streams_name,
            )
            lines.extend("    %s" % line if line else line for line in geometry_lines)
            lines.append("            }")
        packed_call_args = [
            "1",
            "1",
            *("block_jacobian_adjugate%d" % i for i in range(dim * dim)),
            "block_jacobian_determinant0",
        ]
        if omit_reference_basis_inputs:
            pass
        elif use_tensor_product_reference:
            packed_call_args.extend((tensor_shape_name, tensor_grad_name))
        elif use_reference_gradient_vectors:
            packed_call_args.extend(
                "%s%s" % (reference_prefix, _sfem_reference_gradient_vector_name(component))
                for component in range(dim)
            )
        else:
            packed_call_args.extend(
                "%s%s" % (reference_prefix, array_input.name)
                for array_input in reference_inputs
            )
        packed_call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
        packed_call_args.extend(material_parameter_names)
        if uses_current:
            packed_call_args.append("block_u_streams")
        packed_call_args.extend(("block_h_streams", "block_out_streams"))
        lines.append("")
        if direct_hessian_assembly:
            lines.extend(
                _sfem_soa_direct_hessian_element_matrix_call_lines(
                    direct_hessian_function_name,
                    dim,
                    material_parameter_names,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    reference_inputs,
                    tensor_shape_name,
                    tensor_grad_name,
                    tensor_weight_name,
                    scalar_weight_name,
                    reference_prefix,
                    "            ",
                )
            )
        else:
            lines.extend(
                [
                    "            for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {",
                    "                element_matrix[entry] = scalar_t(0);",
                    "            }",
                    "",
                    "            for (int trial_component = 0; trial_component < DIM; ++trial_component) {",
                    "                for (int trial_shape = 0; trial_shape < N_SHAPE; ++trial_shape) {",
                    "                    for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {",
                    "                        block_h_data[stream][0] = scalar_t(0);",
                    "                        block_out_data[stream][0] = scalar_t(0);",
                    "                    }",
                    "                    block_h_data[trial_shape * DIM + trial_component][0] = scalar_t(1);",
                    "                    %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
                    % (block_name, ", ".join(packed_call_args)),
                    "                    const int col = trial_component * N_SHAPE + trial_shape;",
                    "                    for (int test_component = 0; test_component < DIM; ++test_component) {",
                    "                        for (int test_shape = 0; test_shape < N_SHAPE; ++test_shape) {",
                    "                            const int row = test_component * N_SHAPE + test_shape;",
                    "                            element_matrix[row * NDOFS + col] = block_out_data[test_shape * DIM + test_component][0];",
                    "                        }",
                    "                    }",
                    "                }",
                    "            }",
                ]
            )
        lines.extend(
            [
                "",
                "            const count_t *const entries = &packed_element_entries[element * NDOFS * NDOFS];",
                "            %s_scatter_packed_crs_entries(element_matrix, entries, values);" % function_base,
                "            }",
                "        }",
                "    }",
                "    return SFEM_SUCCESS;",
                "}",
                "",
            ]
        )
    lines.extend(
        [
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    lines.extend(
        _sfem_soa_hessian_matrix_public_wrappers(
            function_base,
            implementation_name,
            dim,
            formats,
            material_parameter_names,
            uses_current,
            packed_crs_passes,
        )
    )
    return lines


def _matrix_formats_from_plan(matrix_format_plan):
    if matrix_format_plan is None or getattr(matrix_format_plan, "is_empty", True):
        return ()
    formats = []
    for matrix_format in matrix_format_plan.formats:
        value = getattr(matrix_format, "value", str(matrix_format)).lower()
        if value not in formats:
            formats.append(value)
    return tuple(formats)


def _packed_crs_passes(matrix_format_plan):
    if matrix_format_plan is None or getattr(matrix_format_plan, "is_empty", True):
        return ()
    passes = []
    for variant in matrix_format_plan.variants:
        matrix_format = getattr(variant.matrix_format, "value", str(variant.matrix_format)).lower()
        mesh_layout = getattr(variant.mesh_layout, "value", str(variant.mesh_layout)).lower()
        if matrix_format != "crs" or mesh_layout != "packed":
            continue
        packed_pass = getattr(variant.packed_pass, "value", str(variant.packed_pass)).lower()
        if packed_pass and packed_pass != "none" and packed_pass not in passes:
            passes.append(packed_pass)
    return tuple(passes)


def _sfem_soa_hessian_scatter_dispatch_lines(function_base, formats, indent):
    cases = (
        (
            "bsr",
            1,
            "invalid_matrix_graph |= (%s_scatter_bsr(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);"
            % function_base,
        ),
        (
            "crs",
            0,
            "invalid_matrix_graph |= (%s_scatter_crs(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);"
            % function_base,
        ),
        (
            "dia",
            2,
            "invalid_matrix_graph |= (%s_scatter_dia(ev, element_matrix, nnodes, diag_offsets, ndiag, values) != SFEM_SUCCESS);"
            % function_base,
        ),
        (
            "coo",
            3,
            "invalid_matrix_graph |= (%s_scatter_coo(ev, element_matrix, coo_nnz, coo_rows, coo_cols, values) != SFEM_SUCCESS);"
            % function_base,
        ),
        (
            "coo",
            5,
            "%s_scatter_coo_triplets(ev, element_matrix, element, coo_triplet_rows, coo_triplet_cols, values);"
            % function_base,
        ),
        (
            "patch",
            4,
            "invalid_matrix_graph |= (%s_scatter_patch(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);"
            % function_base,
        ),
        (
            "block_diag_sym",
            6,
            "%s_scatter_block_diag_sym(ev, element_matrix, values);" % function_base,
        ),
    )

    lines = []
    first = True
    for matrix_format, format_id, statement in cases:
        if matrix_format not in formats:
            continue
        keyword = "if" if first else "} else if"
        lines.append("%s%s constexpr (FORMAT == %d) {" % (indent, keyword, format_id))
        lines.append("%s    %s" % (indent, statement))
        first = False
    if first:
        lines.append("%sinvalid_matrix_graph |= 1;" % indent)
        return lines
    lines.extend(
        [
            "%s} else {" % indent,
            "%s    invalid_matrix_graph |= 1;" % indent,
            "%s}" % indent,
        ]
    )
    return lines


def _sfem_soa_hessian_matrix_public_function_base(prefix, quadrature_rule, geometry_mode):
    element = quadrature_rule.element_type.lower()
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_hessian_%s_mesh_soa" % (
            prefix,
            geometry_mode,
        )
    return "%s_%s_hessian_%s_mesh_soa" % (
        prefix,
        element,
        geometry_mode,
    )


def _ordered_stream_names(name, dim, stream_shape_order):
    return tuple(
        "%s[%d]" % (name, shape * dim + d)
        for shape in stream_shape_order
        for d in range(dim)
    )


def _ordered_element_pointer_array_lines(pointer_type, array_name, source_name, stream_shape_order, indent):
    if tuple(stream_shape_order) == tuple(range(len(stream_shape_order))):
        return []
    return [
        "%sconst %s *const SFEM_RESTRICT %s[N_SHAPE] = {%s};"
        % (
            indent,
            pointer_type,
            array_name,
            ", ".join("%s[%d]" % (source_name, shape) for shape in stream_shape_order),
        )
    ]


def _ordered_stream_pointer_array_lines(pointer_type, array_name, storage_name, dim, stream_shape_order, indent):
    lines = [
        "%s%s%s[N_SHAPE * DIM];" % (indent, pointer_type, array_name),
    ]
    if tuple(stream_shape_order) == tuple(range(len(stream_shape_order))):
        lines.extend(
            [
                "%sfor (int stream = 0; stream < N_SHAPE * DIM; ++stream) {" % indent,
                "%s    %s[stream] = %s[stream];" % (indent, array_name, storage_name),
                "%s}" % indent,
            ]
        )
        return lines

    return [
        "%s%sconst %s[N_SHAPE * DIM] = {%s};"
        % (
            indent,
            pointer_type,
            array_name,
            ", ".join(
                "%s[%d]" % (storage_name, shape * dim + d)
                for shape in stream_shape_order
                for d in range(dim)
            ),
        )
    ]


def _adjugate_input(dim):
    return sfem_soa_reference_input("jacobian_adjugate", 1, 1, dim * dim)


def _sfem_soa_hessian_scatter_lines(function_base, dim, n_nodes, formats):
    lines = ["namespace sfem {", "namespace codegen {", ""]
    if "bsr" in formats or "crs" in formats or "patch" in formats:
        find_cols_lines = [
            "static SFEM_INLINE void %s_find_cols(" % function_base,
            "        const idx_t *const SFEM_RESTRICT targets,",
            "        const idx_t *const SFEM_RESTRICT row,",
            "        const int lenrow,",
            "        idx_t *const SFEM_RESTRICT ks) {",
        ]
        if n_nodes <= 10:
            find_cols_lines.append("#pragma unroll(%d)" % n_nodes)
        find_cols_lines.extend(
            [
                "    for (int d = 0; d < %d; ++d) {" % n_nodes,
                "        ks[d] = 0;",
                "    }",
                "    for (int k = 0; k < lenrow; ++k) {",
            ]
        )
        if n_nodes <= 10:
            find_cols_lines.append("#pragma unroll(%d)" % n_nodes)
        find_cols_lines.extend(
            [
                "        for (int d = 0; d < %d; ++d) {" % n_nodes,
                "            ks[d] += row[k] < targets[d];",
                "        }",
                "    }",
                "}",
                "",
            ]
        )
        lines.extend(find_cols_lines)
    if "bsr" in formats:
        lines.extend(_sfem_soa_hessian_scatter_bsr_lines(function_base, dim, n_nodes))
    if "crs" in formats:
        lines.extend(_sfem_soa_hessian_scatter_crs_lines(function_base, dim, n_nodes))
    if "dia" in formats:
        lines.extend(_sfem_soa_hessian_scatter_dia_lines(function_base, dim, n_nodes))
    if "coo" in formats:
        lines.extend(_sfem_soa_hessian_scatter_coo_lines(function_base, dim, n_nodes))
        lines.extend(_sfem_soa_hessian_scatter_coo_triplet_lines(function_base, dim, n_nodes))
    if "patch" in formats:
        lines.extend(_sfem_soa_hessian_scatter_patch_lines(function_base, dim, n_nodes))
    if "block_diag_sym" in formats:
        lines.extend(_sfem_soa_hessian_scatter_block_diag_sym_lines(function_base, dim, n_nodes))
    return lines


def _sfem_soa_hessian_scatter_bsr_lines(function_base, dim, n_nodes):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_bsr(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const count_t *const SFEM_RESTRICT rowptr,",
        "        const idx_t *const SFEM_RESTRICT colidx,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    count_t entries[N_SHAPE * N_SHAPE];",
        "    idx_t ks[N_SHAPE];",
        "    bool valid_block_graph = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        const idx_t dof_i = ev[i];",
        "        const count_t row_begin = rowptr[dof_i];",
        "        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);",
        "        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];",
        "        %s_find_cols(ev, cols, lenrow, ks);" % function_base,
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {",
        "                if (valid_block_graph) {",
        "                    std::fprintf(stderr, \"%s_scatter_bsr missing block graph entry (%%ld, %%ld)\\n\", (long)ev[i], (long)ev[j]);"
        % function_base,
        "                }",
        "                entries[i * N_SHAPE + j] = row_begin;",
        "                valid_block_graph = false;",
        "            } else {",
        "                entries[i * N_SHAPE + j] = row_begin + ks[j];",
        "            }",
        "        }",
        "    }",
        "    if (!valid_block_graph) return SFEM_FAILURE;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];",
        "            for (int bi = 0; bi < DIM; ++bi) {",
        "                const int row = bi * N_SHAPE + i;",
        "                for (int bj = 0; bj < DIM; ++bj) {",
        "                    const int col = bj * N_SHAPE + j;",
        "#pragma omp atomic update",
        "                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _sfem_soa_hessian_scatter_crs_lines(function_base, dim, n_nodes):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_crs(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const count_t *const SFEM_RESTRICT rowptr,",
        "        const idx_t *const SFEM_RESTRICT colidx,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    count_t row_begin[N_SHAPE];",
        "    int lenrow[N_SHAPE];",
        "    int local_col[N_SHAPE * N_SHAPE];",
        "    idx_t ks[N_SHAPE];",
        "    bool valid_graph = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        row_begin[i] = rowptr[ev[i]];",
        "        lenrow[i] = (int)(rowptr[ev[i] + 1] - row_begin[i]);",
        "        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin[i]];",
        "        %s_find_cols(ev, cols, lenrow[i], ks);" % function_base,
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            if (ks[j] < 0 || ks[j] >= lenrow[i] || cols[ks[j]] != ev[j]) {",
        "                if (valid_graph) {",
        "                    std::fprintf(stderr, \"%s_scatter_crs missing graph entry (%%ld, %%ld)\\n\", (long)ev[i], (long)ev[j]);"
        % function_base,
        "                }",
        "                local_col[i * N_SHAPE + j] = 0;",
        "                valid_graph = false;",
        "            } else {",
        "                local_col[i * N_SHAPE + j] = (int)ks[j];",
        "            }",
        "        }",
        "    }",
        "    if (!valid_graph) return SFEM_FAILURE;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        const count_t rb = row_begin[i];",
        "        const int lr = lenrow[i];",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            const int lc = local_col[i * N_SHAPE + j];",
        "            for (int bi = 0; bi < DIM; ++bi) {",
        "                const int row = bi * N_SHAPE + i;",
        "                scalar_t *const row_values = &values[rb * DIM * DIM + bi * lr * DIM];",
        "                for (int bj = 0; bj < DIM; ++bj) {",
        "                    const int col = bj * N_SHAPE + j;",
        "#pragma omp atomic update",
        "                    row_values[lc * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _sfem_soa_hessian_packed_crs_helper_lines(function_base, dim, n_nodes):
    return [
        "static SFEM_INLINE idx_t %s_packed_global_node(" % function_base,
        "        const uint16_t packed_node,",
        "        const ptrdiff_t pack,",
        "        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,",
        "        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,",
        "        const idx_t *const SFEM_RESTRICT ghost_idx) {",
        "    const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
        "    return packed_node < n_contiguous ? idx_t(owned_nodes_ptr[pack] + packed_node) : ghost_idx[ghost_ptr[pack] + packed_node - n_contiguous];",
        "}",
        "",
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_discover_packed_crs_entries(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const count_t *const SFEM_RESTRICT rowptr,",
        "        const idx_t *const SFEM_RESTRICT colidx,",
        "        count_t *const SFEM_RESTRICT entries) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    static constexpr int NDOFS = DIM * N_SHAPE;",
        "    idx_t ks[N_SHAPE];",
        "    bool valid_graph = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        const count_t row_begin = rowptr[ev[i]];",
        "        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);",
        "        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];",
        "        %s_find_cols(ev, cols, lenrow, ks);" % function_base,
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {",
        "                if (valid_graph) {",
        "                    std::fprintf(stderr, \"%s_discover_packed_crs_entries missing graph entry (%%ld, %%ld)\\n\", (long)ev[i], (long)ev[j]);"
        % function_base,
        "                }",
        "                for (int bi = 0; bi < DIM; ++bi) {",
        "                    for (int bj = 0; bj < DIM; ++bj) {",
        "                        const int row = bi * N_SHAPE + i;",
        "                        const int col = bj * N_SHAPE + j;",
        "                        entries[row * NDOFS + col] = row_begin;",
        "                    }",
        "                }",
        "                valid_graph = false;",
        "            } else {",
        "                const int local_col = (int)ks[j];",
        "                for (int bi = 0; bi < DIM; ++bi) {",
        "                    const count_t row_value_offset = row_begin * DIM * DIM + bi * lenrow * DIM;",
        "                    for (int bj = 0; bj < DIM; ++bj) {",
        "                        const int row = bi * N_SHAPE + i;",
        "                        const int col = bj * N_SHAPE + j;",
        "                        entries[row * NDOFS + col] = row_value_offset + local_col * DIM + bj;",
        "                    }",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return valid_graph ? SFEM_SUCCESS : SFEM_FAILURE;",
        "}",
        "",
        "template <typename scalar_t>",
        "static SFEM_INLINE void %s_scatter_packed_crs_entries(" % function_base,
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const count_t *const SFEM_RESTRICT entries,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    static constexpr int NDOFS = DIM * N_SHAPE;",
        "    for (int row = 0; row < NDOFS; ++row) {",
        "        for (int col = 0; col < NDOFS; ++col) {",
        "#pragma omp atomic update",
        "            values[entries[row * NDOFS + col]] += element_matrix[row * NDOFS + col];",
        "        }",
        "    }",
        "}",
        "",
    ]


def _sfem_soa_hessian_scatter_dia_lines(function_base, dim, n_nodes):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_dia(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const ptrdiff_t nnodes,",
        "        const int *const SFEM_RESTRICT diag_offsets,",
        "        const ptrdiff_t ndiag,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    ptrdiff_t diagonals[N_SHAPE * N_SHAPE];",
        "    bool valid_diagonal_offsets = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            const int offset = (int)(ev[j] - ev[i]);",
        "            ptrdiff_t diagonal = 0;",
        "            while (diagonal < ndiag && diag_offsets[diagonal] != offset) ++diagonal;",
        "            if (diagonal == ndiag) {",
        "                if (valid_diagonal_offsets) {",
        "                    std::fprintf(stderr, \"%s_scatter_dia missing diagonal offset %%d\\n\", offset);"
        % function_base,
        "                }",
        "                diagonals[i * N_SHAPE + j] = 0;",
        "                valid_diagonal_offsets = false;",
        "            } else {",
        "                diagonals[i * N_SHAPE + j] = diagonal;",
        "            }",
        "        }",
        "    }",
        "    if (!valid_diagonal_offsets) return SFEM_FAILURE;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            const ptrdiff_t diagonal = diagonals[i * N_SHAPE + j];",
        "            scalar_t *const block = &values[(diagonal * nnodes + ev[i]) * DIM * DIM];",
        "            for (int bi = 0; bi < DIM; ++bi) {",
        "                const int row = bi * N_SHAPE + i;",
        "                for (int bj = 0; bj < DIM; ++bj) {",
        "                    const int col = bj * N_SHAPE + j;",
        "#pragma omp atomic update",
        "                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _sfem_soa_hessian_scatter_coo_lines(function_base, dim, n_nodes):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_coo(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const ptrdiff_t nnz,",
        "        const idx_t *const SFEM_RESTRICT rows,",
        "        const idx_t *const SFEM_RESTRICT cols,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    ptrdiff_t entries[N_SHAPE * N_SHAPE];",
        "    bool valid_coo_entries = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            ptrdiff_t lo = 0;",
        "            ptrdiff_t hi = nnz;",
        "            while (lo < hi) {",
        "                const ptrdiff_t mid = lo + (hi - lo) / 2;",
        "                if (rows[mid] < ev[i] || (rows[mid] == ev[i] && cols[mid] < ev[j])) lo = mid + 1;",
        "                else hi = mid;",
        "            }",
        "            if (lo == nnz || rows[lo] != ev[i] || cols[lo] != ev[j]) {",
        "                if (valid_coo_entries) {",
        "                    std::fprintf(stderr, \"%s_scatter_coo missing graph entry (%%ld, %%ld)\\n\", (long)ev[i], (long)ev[j]);"
        % function_base,
        "                }",
        "                entries[i * N_SHAPE + j] = 0;",
        "                valid_coo_entries = false;",
        "            } else {",
        "                entries[i * N_SHAPE + j] = lo;",
        "            }",
        "        }",
        "    }",
        "    if (!valid_coo_entries) return SFEM_FAILURE;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];",
        "            for (int bi = 0; bi < DIM; ++bi) {",
        "                const int row = bi * N_SHAPE + i;",
        "                for (int bj = 0; bj < DIM; ++bj) {",
        "                    const int col = bj * N_SHAPE + j;",
        "#pragma omp atomic update",
        "                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _sfem_soa_hessian_scatter_coo_triplet_lines(function_base, dim, n_nodes):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE void %s_scatter_coo_triplets(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const ptrdiff_t element,",
        "        idx_t *const SFEM_RESTRICT rows,",
        "        idx_t *const SFEM_RESTRICT cols,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    static constexpr int NDOFS = DIM * N_SHAPE;",
        "    const ptrdiff_t element_offset = element * NDOFS * NDOFS;",
        "    for (int bi = 0; bi < DIM; ++bi) {",
        "        for (int i = 0; i < N_SHAPE; ++i) {",
        "            const int row = bi * N_SHAPE + i;",
        "            const idx_t global_row = ev[i] * DIM + bi;",
        "            for (int bj = 0; bj < DIM; ++bj) {",
        "                for (int j = 0; j < N_SHAPE; ++j) {",
        "                    const int col = bj * N_SHAPE + j;",
        "                    const ptrdiff_t entry = element_offset + row * NDOFS + col;",
        "                    rows[entry] = global_row;",
        "                    cols[entry] = ev[j] * DIM + bj;",
        "                    values[entry] = element_matrix[row * NDOFS + col];",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
        "",
    ]


def _sfem_soa_hessian_scatter_patch_lines(function_base, dim, n_nodes):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_patch(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const count_t *const SFEM_RESTRICT rowptr,",
        "        const idx_t *const SFEM_RESTRICT colidx,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    count_t entries[N_SHAPE * N_SHAPE];",
        "    idx_t ks[N_SHAPE];",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        const count_t row_begin = rowptr[ev[i]];",
        "        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);",
        "        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];",
        "        %s_find_cols(ev, cols, lenrow, ks);" % function_base,
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            entries[i * N_SHAPE + j] = row_begin + ks[j];",
        "        }",
        "    }",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];",
        "            for (int bi = 0; bi < DIM; ++bi) {",
        "                const int row = bi * N_SHAPE + i;",
        "                for (int bj = 0; bj < DIM; ++bj) {",
        "                    const int col = bj * N_SHAPE + j;",
        "#pragma omp atomic update",
        "                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _sfem_soa_hessian_scatter_block_diag_sym_lines(function_base, dim, n_nodes):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE void %s_scatter_block_diag_sym(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int DIM = %d;" % dim,
        "    static constexpr int N_SHAPE = %d;" % n_nodes,
        "    static constexpr int NDOFS = DIM * N_SHAPE;",
        "    static constexpr int SYM_DIM = (DIM * (DIM + 1)) / 2;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        scalar_t *const block = &values[(ptrdiff_t)ev[i] * SYM_DIM];",
        "        int sym = 0;",
        "        for (int bi = 0; bi < DIM; ++bi) {",
        "            const int row = bi * N_SHAPE + i;",
        "            for (int bj = bi; bj < DIM; ++bj) {",
        "                const int col = bj * N_SHAPE + i;",
        "#pragma omp atomic update",
        "                block[sym++] += element_matrix[row * NDOFS + col];",
        "            }",
        "        }",
        "    }",
        "}",
        "",
    ]


def _sfem_soa_hessian_matrix_public_wrappers(
    function_base,
    implementation_name,
    dim,
    formats,
    material_parameter_names,
    uses_current,
    packed_crs_passes=(),
):
    format_tags = {"crs": 0, "bsr": 1, "dia": 2, "coo": 3, "patch": 4, "coo_triplet": 5, "block_diag_sym": 6}
    lines = []
    for matrix_format in formats:
        emitted_formats = ("coo", "coo_triplet") if matrix_format == "coo" else (matrix_format,)
        for emitted_format in emitted_formats:
            public_name = function_base.replace(
                "_hessian_",
                "_hessian_%s_" % emitted_format,
            )
            lines.extend(
                _sfem_soa_hessian_matrix_public_wrapper(
                    public_name,
                    implementation_name,
                    dim,
                    emitted_format,
                    format_tags[emitted_format],
                    material_parameter_names,
                    uses_current,
                )
            )
    if "crs" in formats:
        if "one_pass" in packed_crs_passes:
            lines.extend(
                _sfem_soa_hessian_packed_crs_public_wrapper(
                    function_base.replace(
                        "_hessian_",
                        "_hessian_crs_packed_one_pass_",
                    ),
                    function_base,
                    dim,
                    material_parameter_names,
                    uses_current,
                    two_pass=False,
                )
            )
        if "two_pass" in packed_crs_passes:
            lines.extend(
                _sfem_soa_hessian_packed_crs_public_wrapper(
                    function_base.replace(
                        "_hessian_",
                        "_hessian_crs_packed_two_pass_",
                    ),
                    function_base,
                    dim,
                    material_parameter_names,
                    uses_current,
                    two_pass=True,
                )
            )
    return lines


def _sfem_soa_hessian_packed_crs_public_wrapper(
    public_name,
    function_base,
    dim,
    material_parameter_names,
    uses_current,
    two_pass,
):
    lines = []
    for scalar_type in ("double", "float"):
        suffix = "" if scalar_type == "double" else "_float"
        concrete_name = "%s%s" % (public_name, suffix)
        params = _hessian_packed_crs_public_params(
            dim,
            scalar_type,
            material_parameter_names,
            uses_current,
            two_pass,
        )
        lines.append('extern "C" int %s(' % concrete_name)
        for idx, param in enumerate(params):
            comma = "," if idx + 1 < len(params) else ""
            lines.append("        %s%s" % (param, comma))
        common_args = [
            "n_packs",
            "n_elements_per_pack",
            "nelements",
            "nnodes",
            "max_nodes_per_pack",
            "elements",
            "owned_nodes_ptr",
            "n_shared_nodes",
            "ghost_ptr",
            "ghost_idx",
        ]
        fill_args = common_args + [
            "points",
            *material_parameter_names,
        ]
        if uses_current:
            fill_args.append("u_stride")
            fill_args.extend("u%s" % _component_name(d) for d in range(dim))
        fill_args.extend(("packed_element_entries", "values"))
        lines.append(") {")
        if two_pass:
            lines.append(
                "    const int graph_status = sfem::codegen::%s_packed_discover_impl<%s, geom_t>(%s);"
                % (
                    function_base,
                    scalar_type,
                    ", ".join(common_args + ["rowptr", "colidx", "packed_element_entries"]),
                )
            )
            lines.append("    if (graph_status != SFEM_SUCCESS) return graph_status;")
        lines.append(
            "    return sfem::codegen::%s_packed_fill_impl<%s, geom_t>(%s);"
            % (function_base, scalar_type, ", ".join(fill_args))
        )
        lines.extend(["}", ""])
    return lines


def _sfem_soa_hessian_matrix_public_wrapper(
    public_name,
    implementation_name,
    dim,
    matrix_format,
    format_tag,
    material_parameter_names,
    uses_current,
):
    lines = []
    for scalar_type in ("double", "float"):
        suffix = "" if scalar_type == "double" else "_float"
        concrete_name = "%s%s" % (public_name, suffix)
        params = _hessian_matrix_public_params(
            matrix_format,
            dim,
            scalar_type,
            material_parameter_names,
            uses_current,
        )
        lines.append('extern "C" int %s(' % concrete_name)
        for idx, param in enumerate(params):
            comma = "," if idx + 1 < len(params) else ""
            lines.append("        %s%s" % (param, comma))
        lines.append(") {")
        lines.append(
            "    return sfem::codegen::%s<%s, geom_t, %d>(%s);"
            % (
                implementation_name,
                scalar_type,
                format_tag,
                ", ".join(
                    _hessian_matrix_impl_args(
                        matrix_format,
                        dim,
                        material_parameter_names,
                        uses_current,
                    )
                ),
            )
        )
        lines.extend(["}", ""])
    return lines


def _hessian_matrix_public_params(
    matrix_format,
    dim,
    scalar_type,
    material_parameter_names,
    uses_current,
):
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend(
        "const %s %s" % (scalar_type, parameter)
        for parameter in material_parameter_names
    )
    if uses_current:
        params.append("const ptrdiff_t u_stride")
        params.extend(
            "const %s *const SFEM_RESTRICT u%s" % (scalar_type, _component_name(d))
            for d in range(dim)
        )
    if matrix_format in ("crs", "bsr"):
        params.extend(
            [
                "const count_t *const SFEM_RESTRICT rowptr",
                "const idx_t *const SFEM_RESTRICT colidx",
                "%s *const SFEM_RESTRICT values" % scalar_type,
            ]
        )
    elif matrix_format == "dia":
        params.extend(
            [
                "const int *const SFEM_RESTRICT diag_offsets",
                "const ptrdiff_t ndiag",
                "%s *const SFEM_RESTRICT values" % scalar_type,
            ]
        )
    elif matrix_format == "coo":
        params.extend(
            [
                "const ptrdiff_t nnz",
                "const idx_t *const SFEM_RESTRICT rows",
                "const idx_t *const SFEM_RESTRICT cols",
                "%s *const SFEM_RESTRICT values" % scalar_type,
            ]
        )
    elif matrix_format == "coo_triplet":
        params.extend(
            [
                "idx_t *const SFEM_RESTRICT rows",
                "idx_t *const SFEM_RESTRICT cols",
                "%s *const SFEM_RESTRICT values" % scalar_type,
            ]
        )
    elif matrix_format == "patch":
        params.extend(
            [
                "const count_t *const SFEM_RESTRICT rowptr",
                "const idx_t *const SFEM_RESTRICT colidx",
                "%s *const SFEM_RESTRICT values" % scalar_type,
            ]
        )
    elif matrix_format == "block_diag_sym":
        params.append("%s *const SFEM_RESTRICT values" % scalar_type)
    else:
        raise ValueError("unsupported matrix format '%s'" % matrix_format)
    return params


def _hessian_packed_crs_public_params(
    dim,
    scalar_type,
    material_parameter_names,
    uses_current,
    two_pass,
):
    params = [
        "const ptrdiff_t n_packs",
        "const ptrdiff_t n_elements_per_pack",
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "const ptrdiff_t max_nodes_per_pack",
        "uint16_t **const SFEM_RESTRICT elements",
        "const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr",
        "const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes",
        "const ptrdiff_t *const SFEM_RESTRICT ghost_ptr",
        "const idx_t *const SFEM_RESTRICT ghost_idx",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend(
        "const %s %s" % (scalar_type, parameter)
        for parameter in material_parameter_names
    )
    if uses_current:
        params.append("const ptrdiff_t u_stride")
        params.extend(
            "const %s *const SFEM_RESTRICT u%s" % (scalar_type, _component_name(d))
            for d in range(dim)
        )
    if two_pass:
        params.extend(
            [
                "const count_t *const SFEM_RESTRICT rowptr",
                "const idx_t *const SFEM_RESTRICT colidx",
                "count_t *const SFEM_RESTRICT packed_element_entries",
            ]
        )
    else:
        params.append("const count_t *const SFEM_RESTRICT packed_element_entries")
    params.append("%s *const SFEM_RESTRICT values" % scalar_type)
    return params


def _hessian_matrix_impl_args(
    matrix_format,
    dim,
    material_parameter_names,
    uses_current,
):
    args = [
        "nelements",
        "nnodes",
        "elements",
        "points",
        *material_parameter_names,
    ]
    if uses_current:
        args.append("u_stride")
        args.extend("u%s" % _component_name(d) for d in range(dim))
    if matrix_format in ("crs", "bsr"):
        args.extend(
            [
                "rowptr",
                "colidx",
                "values",
                "nullptr",
                "0",
                "0",
                "nullptr",
                "nullptr",
                "nullptr",
                "nullptr",
            ]
        )
    elif matrix_format == "dia":
        args.extend(
            [
                "nullptr",
                "nullptr",
                "values",
                "diag_offsets",
                "ndiag",
                "0",
                "nullptr",
                "nullptr",
                "nullptr",
                "nullptr",
            ]
        )
    elif matrix_format == "coo":
        args.extend(
            [
                "nullptr",
                "nullptr",
                "values",
                "nullptr",
                "0",
                "nnz",
                "rows",
                "cols",
                "nullptr",
                "nullptr",
            ]
        )
    elif matrix_format == "coo_triplet":
        args.extend(
            [
                "nullptr",
                "nullptr",
                "values",
                "nullptr",
                "0",
                "0",
                "nullptr",
                "nullptr",
                "rows",
                "cols",
            ]
        )
    elif matrix_format == "patch":
        args.extend(
            [
                "rowptr",
                "colidx",
                "values",
                "nullptr",
                "0",
                "0",
                "nullptr",
                "nullptr",
                "nullptr",
                "nullptr",
            ]
        )
    elif matrix_format == "block_diag_sym":
        args.extend(
            [
                "nullptr",
                "nullptr",
                "values",
                "nullptr",
                "0",
                "0",
                "nullptr",
                "nullptr",
                "nullptr",
                "nullptr",
            ]
        )
    else:
        raise ValueError("unsupported matrix format '%s'" % matrix_format)
    return tuple(args)


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


def _sfem_soa_diagnostics_header(
    work_item="lane",
    header_guard_suffix="HPP",
    inline_qualifier="SFEM_INLINE",
    define_sfem_inline=True,
):
    struct_name = _sfem_soa_diagnostics_struct_name()
    guard = "SFEM_CODEGEN_KERNEL_DIAGNOSTICS_%s" % header_guard_suffix
    per_qp = "per_qp_%s" % work_item
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "",
    ]
    if define_sfem_inline:
        lines.extend(
            [
                "#ifndef SFEM_INLINE",
                "#define SFEM_INLINE inline",
                "#endif",
                "",
            ]
        )
    lines.extend([
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
        "    long add_instructions_%s;" % per_qp,
        "    long mul_instructions_%s;" % per_qp,
        "    long div_instructions_%s;" % per_qp,
        "    long sqrt_instructions_%s;" % per_qp,
        "    long pow_instructions_%s;" % per_qp,
        "    long exp_instructions_%s;" % per_qp,
        "    long log_instructions_%s;" % per_qp,
        "    long trig_instructions_%s;" % per_qp,
        "    long load_instructions_%s;" % per_qp,
        "    long store_instructions_%s;" % per_qp,
        "    long flops_%s;" % per_qp,
        "    long affine_mesh_flops_per_element;",
        "    long isoparametric_mesh_flops_per_element;",
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
        "static %s double %s_total_flops(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements) {",
        "    const double n = nelements > 0 ? (double)nelements : 0.0;",
        "    return n * ((double)d->n_qp * (double)d->flops_%s + (double)d->isoparametric_mesh_flops_per_element);" % per_qp,
        "}",
        "",
        "static %s double %s_total_flops_affine_mesh(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements) {",
        "    const double n = nelements > 0 ? (double)nelements : 0.0;",
        "    return n * ((double)d->n_qp * (double)d->flops_%s + (double)d->affine_mesh_flops_per_element);" % per_qp,
        "}",
        "",
        "static %s double %s_total_flops_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements) {",
        "    const double n = nelements > 0 ? (double)nelements : 0.0;",
        "    return n * ((double)d->n_qp * (double)d->flops_%s + (double)d->isoparametric_mesh_flops_per_element);" % per_qp,
        "}",
        "",
        "static %s size_t %s_total_bytes(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    (void)accumulator_bytes;",
        "    const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "    const size_t geometry_bytes = n * (size_t)d->n_qp * (size_t)d->geometry_streams * scalar_bytes;",
        "    const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "    const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "    const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "    return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static %s size_t %s_total_bytes_affine_mesh(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    (void)accumulator_bytes;",
        "    const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "    const size_t geometry_bytes = n * (size_t)(d->dim * d->dim + 1) * scalar_bytes;",
        "    const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "    const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "    const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "    return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static %s size_t %s_total_bytes_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    (void)accumulator_bytes;",
        "    const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "    const size_t geometry_bytes = n * (size_t)d->dim * (size_t)d->n_shape * scalar_bytes;",
        "    const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "    const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "    const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "    return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static %s double %s_arithmetic_intensity(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const size_t bytes = %s_total_bytes(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "    return bytes ? %s_total_flops(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "static %s double %s_arithmetic_intensity_affine_mesh(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const size_t bytes = %s_total_bytes_affine_mesh(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "    return bytes ? %s_total_flops_affine_mesh(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "static %s double %s_arithmetic_intensity_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const size_t bytes = %s_total_bytes_isoparametric_mesh(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "    return bytes ? %s_total_flops_isoparametric_mesh(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "static %s void %s_print_rate_with_ai(" % (inline_qualifier, struct_name),
        "        const char *const name,",
        "        const %s *const d," % struct_name,
        "        const double elapsed,",
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t ndofs,",
        "        const double ai,",
        "        const double total_flops) {",
        "    const double element_rate = elapsed > 0.0 ? 1e-6 * (double)nelements / elapsed : 0.0;",
        "    const double dof_rate = elapsed > 0.0 ? 1e-6 * (double)ndofs / elapsed : 0.0;",
        "    const double gflops = elapsed > 0.0",
        "            ? 1e-9 * total_flops / elapsed",
        "            : 0.0;",
        '    printf("%-72s %12.6e %16.3f %13.3f %10.3f %13.3f\\n",',
        "           name ? name : d->kernel_name,",
        "           elapsed, element_rate, dof_rate, ai, gflops);",
        "}",
        "",
        "static %s void %s_print_rate(" % (inline_qualifier, struct_name),
        "        const char *const name,",
        "        const %s *const d," % struct_name,
        "        const double elapsed,",
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t ndofs,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const double ai = %s_arithmetic_intensity(" % struct_name,
        "            d, nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "    const double total_flops = %s_total_flops(d, nelements);" % struct_name,
        "    %s_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);" % struct_name,
        "}",
        "",
        "static %s void %s_print_rate_affine_mesh(" % (inline_qualifier, struct_name),
        "        const char *const name,",
        "        const %s *const d," % struct_name,
        "        const double elapsed,",
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t ndofs,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const double ai = %s_arithmetic_intensity_affine_mesh(" % struct_name,
        "            d, nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "    const double total_flops = %s_total_flops_affine_mesh(d, nelements);" % struct_name,
        "    %s_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);" % struct_name,
        "}",
        "",
        "static %s void %s_print_rate_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "        const char *const name,",
        "        const %s *const d," % struct_name,
        "        const double elapsed,",
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t ndofs,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const double ai = %s_arithmetic_intensity_isoparametric_mesh(" % struct_name,
        "            d, nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "    const double total_flops = %s_total_flops_isoparametric_mesh(d, nelements);" % struct_name,
        "    %s_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);" % struct_name,
        "}",
        "",
        "} // namespace codegen",
        "} // namespace sfem",
        "",
        "#endif",
    ])
    return lines


def _sfem_soa_diagnostic_print_wrapper_lines(
    function_name,
    variable_name,
    scalar_type,
    print_rate_helper="KernelDiagnostics_print_rate",
):
    suffix = "" if scalar_type == "double" else "_float"
    public_name = "%s%s_print_rate" % (function_name, suffix)
    return [
        'extern "C" void %s(' % public_name,
        "        const double elapsed,",
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t ndofs) {",
        "    sfem::codegen::%s(" % print_rate_helper,
        '            "%s%s",' % (function_name, suffix),
        "            &sfem::codegen::%s," % variable_name,
        "            elapsed, nelements, ndofs,",
        "            sizeof(%s), sizeof(%s), sizeof(%s));"
        % (scalar_type, scalar_type, scalar_type),
        "}",
    ]


def _tensor_product_field_gradient_flops(dim, n_qp_1d, n_shape_1d):
    dim = int(dim)
    q = int(n_qp_1d)
    s = int(n_shape_1d)
    if dim == 2:
        return 4 * q * s * s + 6 * q * q * s
    if dim == 3:
        return 4 * q * s * s * s + 6 * q * q * s * s + 6 * q * q * q * s
    return 0


def _tensor_product_test_gradient_flops(dim, n_qp_1d, n_shape_1d):
    dim = int(dim)
    q = int(n_qp_1d)
    s = int(n_shape_1d)
    if dim == 2:
        return 6 * q * q * s + 5 * q * s * s
    if dim == 3:
        return 6 * q * q * q * s + 6 * q * q * s * s + 5 * q * s * s * s
    return 0


def _adjugate_and_determinant_flops_per_qp(dim):
    dim = int(dim)
    if dim == 2:
        return 11
    if dim == 3:
        return 41
    return 0


def _physical_gradient_transform_flops_per_qp(dim):
    dim = int(dim)
    if dim <= 0:
        return 0
    return 1 + dim * dim * (2 * dim)


def _weak_operand_transform_flops_per_qp(dim):
    dim = int(dim)
    if dim <= 0:
        return 0
    return dim * dim * (2 * dim)


def _objective_weight_flops_per_qp():
    return 3


def _tensor_product_mesh_extra_flops_per_element(form, dim, n_qp, quadrature_rule, basis_family):
    if quadrature_rule is None or str(basis_family) != "tensor_product":
        return 0, 0
    n_qp_1d = int(quadrature_rule.tensor_product_n_qp_1d)
    n_shape_1d = int(quadrature_rule.tensor_product_n_shape_1d)
    field_gradient = _tensor_product_field_gradient_flops(dim, n_qp_1d, n_shape_1d)
    test_gradient = _tensor_product_test_gradient_flops(dim, n_qp_1d, n_shape_1d)
    grad_transform = int(n_qp) * _physical_gradient_transform_flops_per_qp(dim)

    form_name = str(getattr(form, "name", ""))
    if form_name == "objective":
        local_extra = dim * field_gradient + grad_transform + int(n_qp) * _objective_weight_flops_per_qp()
    elif form_name in ("gradient", "apply"):
        local_extra = (
            dim * field_gradient
            + grad_transform
            + int(n_qp) * _weak_operand_transform_flops_per_qp(dim)
            + dim * test_gradient
        )
    else:
        local_extra = 0

    geometry_extra = (
        dim * field_gradient
        + int(n_qp) * _adjugate_and_determinant_flops_per_qp(dim)
    )
    return local_extra, local_extra + geometry_extra


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
    uses_current = _form_uses_current(form, default=True)
    uses_direction = _form_uses_direction(form, default=form.has_direction)
    if form.expression_graph is not None:
        cost = form.expression_graph.cost
    elif form.weak_form is not None:
        diagnostic_deformation_substitutions = _weak_form_deformation_gradient_substitutions(
            form.weak_form,
            "diag_grad",
            scalar_temporaries=True,
        )
        if form.name == "objective":
            diagnostic_expressions = (
                form.weak_form.energy_density.xreplace(
                    diagnostic_deformation_substitutions
                ),
            )
        else:
            diagnostic_expressions = tuple(
                _weak_form_material_expression(
                    form.weak_form,
                    form.name,
                    diagnostic_deformation_substitutions,
                    tuple(
                        sp.symbols("diag_trial_grad%d" % i)
                        for i in range(form.weak_form.dim * form.weak_form.dim)
                    ),
                )
            )
        diagnostic_graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, diagnostic_expressions)
            .build_graph(
                data_symbols=tuple(diagnostic_deformation_substitutions.values()),
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
    u_streams = dim * n_nodes if uses_current else 0
    h_streams = dim * n_nodes if uses_direction else 0
    element_type = quadrature_rule.element_type if quadrature_rule is not None else "GENERIC"
    quadrature_order = quadrature_rule.order if quadrature_rule is not None else 0
    affine_extra_flops, isoparametric_extra_flops = _tensor_product_mesh_extra_flops_per_element(
        form,
        dim,
        n_qp,
        quadrature_rule,
        basis_family,
    )
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
        "    %d," % affine_extra_flops,
        "    %d," % isoparametric_extra_flops,
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
                (
                    _sfem_soa_mesh_public_function_name(
                        prefix,
                        form.name,
                        quadrature_rule,
                        "affine",
                    ),
                    "KernelDiagnostics_print_rate_affine_mesh",
                ),
                (
                    _sfem_soa_mesh_public_function_name(
                        prefix,
                        form.name,
                        quadrature_rule,
                        "isoparametric",
                    ),
                    "KernelDiagnostics_print_rate_isoparametric_mesh",
                ),
            )
        )
    for function_name_entry in function_names:
        if isinstance(function_name_entry, tuple):
            function_name, print_rate_helper = function_name_entry
        else:
            function_name = function_name_entry
            print_rate_helper = "KernelDiagnostics_print_rate"
        for scalar_type in ("double", "float"):
            lines.append("")
            lines.extend(
                _sfem_soa_diagnostic_print_wrapper_lines(
                    function_name,
                    variable_name,
                    scalar_type,
                    print_rate_helper,
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
        compact = " ".join(param.replace(",", "").split())
        if (
            "*" not in compact
            and (
                compact.startswith("const scalar_t ")
                or compact.startswith("const real_t ")
            )
        ):
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


def _sfem_soa_element_api_header(
    forms,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    local_name,
    geometry_name,
    array_inputs,
    quadrature_rule,
    basis_family=None,
    use_shared_weak_local=False,
    source_builder=None,
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    if quadrature_rule is None or not _sfem_soa_has_adjugate_geometry_inputs(array_inputs, dim):
        guard = "%s_ELEMENT_API_%s" % (
            _cpp_macro_name(prefix),
            source_builder.header_guard_suffix(),
        )
        return "\n".join(["#ifndef %s" % guard, "#define %s" % guard, "", "#endif", ""])

    alias = _sfem_tensor_product_element_api_alias(prefix, dim, n_nodes)
    if alias is not None:
        return _sfem_soa_element_api_alias_header(
            forms,
            prefix,
            alias,
            dim,
            n_nodes,
            n_qp,
            vector_size,
            source_builder,
        )

    forms_by_name = {form.name: form for form in forms}
    guard = "%s_ELEMENT_API_%s" % (
        _cpp_macro_name(prefix),
        source_builder.header_guard_suffix(),
    )
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

    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "#include <stddef.h>",
        '#include "%s"' % local_name,
        '#include "%s"' % geometry_name,
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "",
        "#ifndef SFEM_FAILURE",
        "#define SFEM_FAILURE 1",
        "#endif",
        "",
        "#ifndef MIN",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
        "#endif",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(
        quadrature_reference_struct_lines(
            prefix,
            "isoparametric",
            sfem_mesh_reference_data(quadrature_rule),
        )
    )
    lines.append("")
    for operation in ("objective", "gradient"):
        form = forms_by_name.get(operation)
        if form is None or form.weak_form is None:
            continue
        public = "energy" if operation == "objective" else "gradient"
        lines.extend(
            _sfem_soa_element_api_operation_lines(
                form,
                public,
                prefix,
                dim,
                n_nodes,
                n_qp,
                vector_size,
                local_prefix,
                quadrature_rule,
                reference_inputs,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                use_shared_weak_local,
                source_builder,
            )
        )
        lines.append("")
    apply_form = forms_by_name.get("apply")
    if apply_form is not None and apply_form.weak_form is not None:
        lines.extend(
            _sfem_soa_element_api_hessian_lines(
                apply_form,
                prefix,
                dim,
                n_nodes,
                n_qp,
                vector_size,
                local_prefix,
                quadrature_rule,
                reference_inputs,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                use_shared_weak_local,
                source_builder,
            )
        )
        lines.append("")
    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _sfem_tensor_product_element_api_alias(prefix, dim, n_nodes):
    aliases = (
        ("quad4", "proteus_quad4", 2, 4),
        ("hex8", "proteus_hex8", 3, 8),
        ("hex27", "proteus_hex27", 3, 27),
    )
    for element_name, proteus_name, alias_dim, alias_n_nodes in aliases:
        suffix = "_%s" % element_name
        proteus_suffix = "_%s" % proteus_name
        if dim == alias_dim and n_nodes == alias_n_nodes and prefix.endswith(suffix) and not prefix.endswith(proteus_suffix):
            return {
                "element_name": element_name,
                "proteus_name": proteus_name,
                "target_prefix": "%s_%s" % (prefix[: -len(suffix)], proteus_name),
                "include": "../%s/%s_%s_element.hpp" % (
                    proteus_name,
                    prefix[: -len(suffix)],
                    proteus_name,
                ),
                "shape_order": tensor_product_cartesian_shape_order(dim, n_nodes),
            }
    return None


def _sfem_soa_element_api_alias_header(
    forms,
    prefix,
    alias,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    source_builder,
):
    guard = "%s_ELEMENT_API_%s" % (
        _cpp_macro_name(prefix),
        source_builder.header_guard_suffix(),
    )
    forms_by_name = {form.name: form for form in forms}
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        '#include "%s"' % alias["include"],
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    for operation in ("objective", "gradient"):
        form = forms_by_name.get(operation)
        if form is None or form.weak_form is None:
            continue
        public = "energy" if operation == "objective" else "gradient"
        output_param = "scalar_t *const SFEM_RESTRICT values" if public == "energy" else "scalar_t *const *const SFEM_RESTRICT out_streams"
        for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
            name = "%s_%s_element_%ssoa" % (prefix, public, ("%s_" % suffix) if suffix else "")
            target_name = "%s_%s_element_%ssoa" % (
                alias["target_prefix"],
                public,
                ("%s_" % suffix) if suffix else "",
            )
            params = _sfem_soa_element_api_common_params(form, dim, include_coords)
            params.append(output_param)
            lines.extend(
                _sfem_soa_element_api_alias_function_lines(
                    name,
                    target_name,
                    params,
                    alias["shape_order"],
                    dim,
                    n_nodes,
                    n_qp,
                    vector_size,
                )
            )
            lines.append("")
    apply_form = forms_by_name.get("apply")
    if apply_form is not None and apply_form.weak_form is not None:
        for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
            name = "%s_hessian_element_%ssoa" % (prefix, ("%s_" % suffix) if suffix else "")
            target_name = "%s_hessian_element_%ssoa" % (alias["target_prefix"], ("%s_" % suffix) if suffix else "")
            params = _sfem_soa_element_api_common_params(apply_form, dim, include_coords)
            params.append("scalar_t *const *const SFEM_RESTRICT matrix_streams")
            lines.extend(
                _sfem_soa_element_api_alias_function_lines(
                    name,
                    target_name,
                    params,
                    alias["shape_order"],
                    dim,
                    n_nodes,
                    n_qp,
                    vector_size,
                )
            )
            lines.append("")
    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _sfem_soa_element_api_alias_function_lines(
    name,
    target_name,
    params,
    shape_order,
    dim,
    n_nodes,
    n_qp,
    vector_size,
):
    lines = [
        "template <typename scalar_t, int VECTOR_SIZE = %d>" % vector_size,
        "static SFEM_INLINE int %s(" % name,
    ]
    for idx, param in enumerate(params):
        comma = "," if idx + 1 < len(params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_SHAPE = %d;" % n_nodes,
            "    static constexpr int N_QP = %d;" % n_qp,
            "    static constexpr int NDOFS = DIM * N_SHAPE;",
            "    static constexpr int SHAPE_ORDER[N_SHAPE] = {%s};" % ", ".join(str(i) for i in shape_order),
        ]
    )
    param_names = [_cpp_argument_name(param) for param in params]
    if "coords" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines("coords", "ordered_coords", "const scalar_t *", False))
    if "u_streams" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines("u_streams", "ordered_u_streams", "const scalar_t *", False))
    if "out_streams" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines("out_streams", "ordered_out_streams", "scalar_t *", False))
    if "matrix_streams" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines("matrix_streams", "ordered_matrix_streams", "scalar_t *", True))

    args = []
    for name_ in param_names:
        if name_ == "coords":
            args.append("ordered_coords")
        elif name_ == "u_streams":
            args.append("ordered_u_streams")
        elif name_ == "out_streams":
            args.append("ordered_out_streams")
        elif name_ == "matrix_streams":
            args.append("ordered_matrix_streams")
        else:
            args.append(name_)
    lines.append("    return %s<scalar_t, VECTOR_SIZE>(%s);" % (target_name, ", ".join(args)))
    lines.append("}")
    return lines


def _sfem_soa_element_api_alias_stream_lines(source_name, ordered_name, pointer_type, matrix):
    lines = ["    %s%s[NDOFS%s];" % (pointer_type, ordered_name, " * NDOFS" if matrix else "")]
    if matrix:
        lines.extend(
            [
                "    for (int row_shape = 0; row_shape < N_SHAPE; ++row_shape) {",
                "        const int source_row_shape = SHAPE_ORDER[row_shape];",
                "        for (int row_component = 0; row_component < DIM; ++row_component) {",
                "            const int row = row_shape * DIM + row_component;",
                "            const int source_row = source_row_shape * DIM + row_component;",
                "            for (int col_shape = 0; col_shape < N_SHAPE; ++col_shape) {",
                "                const int source_col_shape = SHAPE_ORDER[col_shape];",
                "                for (int col_component = 0; col_component < DIM; ++col_component) {",
                "                    const int col = col_shape * DIM + col_component;",
                "                    const int source_col = source_col_shape * DIM + col_component;",
                "                    %s[row * NDOFS + col] = %s[source_row * NDOFS + source_col];" % (ordered_name, source_name),
                "                }",
                "            }",
                "        }",
                "    }",
            ]
        )
    else:
        lines.extend(
            [
                "    for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "        const int source_shape = SHAPE_ORDER[shape];",
                "        for (int component = 0; component < DIM; ++component) {",
                "            %s[shape * DIM + component] = %s[source_shape * DIM + component];" % (ordered_name, source_name),
                "        }",
                "    }",
            ]
        )
    return lines


def _sfem_soa_element_api_common_params(form, dim, include_coords):
    params = ["const ptrdiff_t nelements"]
    if include_coords:
        params.append("const scalar_t *const *const SFEM_RESTRICT coords")
    else:
        params.append("const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate")
        params.append("const scalar_t *const SFEM_RESTRICT jacobian_determinant")
    params.extend(_form_material_parameter_declarations(form))
    if _form_uses_current(form, default=True):
        params.append("const scalar_t *const *const SFEM_RESTRICT u_streams")
    return params


def _sfem_soa_element_api_reference_args(prefix, quadrature_rule, use_tensor_product_reference, use_reference_gradient_vectors):
    if use_tensor_product_reference:
        return (
            quadrature_reference_accessor(prefix, "isoparametric", "shape_1d"),
            quadrature_reference_accessor(prefix, "isoparametric", "grad_1d"),
            quadrature_reference_accessor(prefix, "isoparametric", "q_weight_1d"),
        )
    if use_reference_gradient_vectors:
        return tuple(
            quadrature_reference_accessor(
                prefix,
                "isoparametric",
                _sfem_reference_gradient_vector_name(component),
            )
            for component in range(quadrature_rule.dim)
        ) + (quadrature_reference_accessor(prefix, "isoparametric", "q_weight"),)
    return (
        quadrature_reference_accessor(prefix, "isoparametric", "grad_ref"),
        quadrature_reference_accessor(prefix, "isoparametric", "q_weight"),
    )


def _sfem_soa_element_api_geometry_args(dim):
    return tuple("block_jacobian_adjugate%d" % component for component in range(dim * dim)) + (
        "block_jacobian_determinant0",
    )


def _sfem_soa_element_api_block_call(
    form,
    local_prefix,
    dim,
    n_qp,
    prefix,
    quadrature_rule,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    output_arg,
    use_shared_weak_local,
):
    block_name = "%s_%s_block" % (local_prefix, form.name)
    args = [
        "nelems",
        "VECTOR_SIZE",
        *_sfem_soa_element_api_geometry_args(dim),
        *_sfem_soa_element_api_reference_args(
            prefix,
            quadrature_rule,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
        ),
        *_form_material_parameter_names(form),
    ]
    if _form_uses_current(form, default=True):
        args.append("block_u_streams")
    if _form_uses_direction(form, default=form.has_direction):
        args.append("block_h_streams")
    args.append(output_arg)
    return "%s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);" % (
        block_name,
        ", ".join(args),
    )


def _sfem_soa_element_api_tile_setup_lines(form, dim, n_nodes, output_kind):
    lines = [
        "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
    ]
    if _form_uses_current(form, default=True):
        lines.append("        const scalar_t *block_u_streams[NDOFS];")
        lines.append("        for (int stream = 0; stream < NDOFS; ++stream) {")
        lines.append("            block_u_streams[stream] = u_streams[stream] + evbegin;")
        lines.append("        }")
    if output_kind == "value":
        lines.append("        scalar_t *const block_value = values + evbegin;")
        lines.append("        #pragma omp simd")
        lines.append("        for (int lane = 0; lane < nelems; ++lane) {")
        lines.append("            block_value[lane] = scalar_t(0);")
        lines.append("        }")
    elif output_kind == "vector":
        lines.append("        scalar_t *block_out_streams[NDOFS];")
        lines.append("        for (int stream = 0; stream < NDOFS; ++stream) {")
        lines.append("            block_out_streams[stream] = out_streams[stream] + evbegin;")
        lines.append("            #pragma omp simd")
        lines.append("            for (int lane = 0; lane < nelems; ++lane) {")
        lines.append("                block_out_streams[stream][lane] = scalar_t(0);")
        lines.append("            }")
        lines.append("        }")
    return lines


def _sfem_soa_element_api_geometry_tile_lines(dim):
    lines = []
    for component in range(dim * dim):
        lines.append("        scalar_t block_jacobian_adjugate%d[N_QP * VECTOR_SIZE];" % component)
    lines.append("        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];")
    lines.append("        for (int q = 0; q < N_QP; ++q) {")
    lines.append("            #pragma omp simd")
    lines.append("            for (int lane = 0; lane < nelems; ++lane) {")
    for component in range(dim * dim):
        lines.append(
            "                block_jacobian_adjugate%d[q * VECTOR_SIZE + lane] = jacobian_adjugate[%d][q * nelements + evbegin + lane];"
            % (component, component)
        )
    lines.append("                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];")
    lines.append("            }")
    lines.append("        }")
    return lines


def _sfem_soa_element_api_coords_tile_lines(
    prefix,
    dim,
    n_nodes,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    source_builder,
):
    lines = [
        "        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];",
        "        for (int stream = 0; stream < NDOFS; ++stream) {",
        "            #pragma omp simd",
        "            for (int lane = 0; lane < nelems; ++lane) {",
        "                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];",
        "            }",
        "        }",
    ]
    for component in range(dim * dim):
        lines.append("        scalar_t block_jacobian_adjugate%d[N_QP * VECTOR_SIZE];" % component)
    lines.append("        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];")
    if use_tensor_product_reference:
        lines.extend(
            [
                "        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];",
            ]
        )
        for d in range(dim):
            lines.append(
                "        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, %d>(nelems, %s, %s, block_coordinate_data, %d, coordinate_grad_ref + %d * N_QP * DIM * VECTOR_SIZE);"
                % (
                    dim,
                    quadrature_reference_accessor(prefix, "isoparametric", "shape_1d"),
                    quadrature_reference_accessor(prefix, "isoparametric", "grad_1d"),
                    d,
                    d,
                )
            )
        lines.append(
            "        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {%s};"
            % ", ".join("block_jacobian_adjugate%d" % component for component in range(dim * dim))
        )
        lines.append(
            "        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);"
        )
        return lines
    if use_reference_gradient_vectors:
        for component in range(dim):
            reference_name = _sfem_reference_gradient_vector_name(component)
            lines.append(
                "        const scalar_t *const %s = %s;"
                % (
                    reference_name,
                    quadrature_reference_accessor(prefix, "isoparametric", reference_name),
                )
            )
    else:
        lines.append(
            "        const scalar_t *const grad_ref = %s;"
            % quadrature_reference_accessor(prefix, "isoparametric", "grad_ref")
        )
    lines.append("        for (int q = 0; q < N_QP; ++q) {")
    lines.extend(
        _sfem_soa_isoparametric_geometry_lines(
            dim,
            n_nodes,
            quadrature_rule,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
            reference_inputs,
            q_major=True,
            source_builder=source_builder,
            coordinate_streams="block_coordinate_data",
        )
    )
    lines.append("        }")
    return lines


def _sfem_soa_element_api_operation_lines(
    form,
    public,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    use_shared_weak_local,
    source_builder,
):
    output_kind = "value" if public == "energy" else "vector"
    output_param = "scalar_t *const SFEM_RESTRICT values" if public == "energy" else "scalar_t *const *const SFEM_RESTRICT out_streams"
    lines = []
    for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
        name = "%s_%s_element_%ssoa" % (prefix, public, ("%s_" % suffix) if suffix else "")
        params = _sfem_soa_element_api_common_params(form, dim, include_coords)
        params.append(output_param)
        lines.extend(
            [
                "template <typename scalar_t, int VECTOR_SIZE = %d>" % vector_size,
                "static SFEM_INLINE int %s(" % name,
            ]
        )
        for idx, param in enumerate(params):
            comma = "," if idx + 1 < len(params) else ""
            lines.append("        %s%s" % (param, comma))
        lines.extend(
            [
                ") {",
                "    static constexpr int DIM = %d;" % dim,
                "    static constexpr int N_SHAPE = %d;" % n_nodes,
                "    static constexpr int N_QP = %d;" % n_qp,
                "    static constexpr int NDOFS = DIM * N_SHAPE;",
                "    if (nelements <= 0) return SFEM_SUCCESS;",
                "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            ]
        )
        lines.extend(_sfem_soa_element_api_tile_setup_lines(form, dim, n_nodes, output_kind))
        if include_coords:
            lines.extend(
                _sfem_soa_element_api_coords_tile_lines(
                    prefix,
                    dim,
                    n_nodes,
                    quadrature_rule,
                    reference_inputs,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    source_builder,
                )
            )
        else:
            lines.extend(_sfem_soa_element_api_geometry_tile_lines(dim))
        lines.append(
            "        %s"
            % _sfem_soa_element_api_block_call(
                form,
                local_prefix,
                dim,
                n_qp,
                prefix,
                quadrature_rule,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                "block_value" if public == "energy" else "block_out_streams",
                use_shared_weak_local,
            )
        )
        lines.extend(["    }", "    return SFEM_SUCCESS;", "}", ""])
    return lines


def _sfem_soa_element_api_hessian_lines(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    use_shared_weak_local,
    source_builder,
):
    lines = []
    for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
        name = "%s_hessian_element_%ssoa" % (prefix, ("%s_" % suffix) if suffix else "")
        params = _sfem_soa_element_api_common_params(form, dim, include_coords)
        params.append("scalar_t *const *const SFEM_RESTRICT matrix_streams")
        lines.extend(
            [
                "template <typename scalar_t, int VECTOR_SIZE = %d>" % vector_size,
                "static SFEM_INLINE int %s(" % name,
            ]
        )
        for idx, param in enumerate(params):
            comma = "," if idx + 1 < len(params) else ""
            lines.append("        %s%s" % (param, comma))
        lines.extend(
            [
                ") {",
                "    static constexpr int DIM = %d;" % dim,
                "    static constexpr int N_SHAPE = %d;" % n_nodes,
                "    static constexpr int N_QP = %d;" % n_qp,
                "    static constexpr int NDOFS = DIM * N_SHAPE;",
                "    if (nelements <= 0) return SFEM_SUCCESS;",
                "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
                "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
            ]
        )
        if _form_uses_current(form, default=True):
            lines.append("        const scalar_t *block_u_streams[NDOFS];")
            lines.append("        for (int stream = 0; stream < NDOFS; ++stream) block_u_streams[stream] = u_streams[stream] + evbegin;")
        if include_coords:
            lines.extend(
                _sfem_soa_element_api_coords_tile_lines(
                    prefix,
                    dim,
                    n_nodes,
                    quadrature_rule,
                    reference_inputs,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    source_builder,
                )
            )
        else:
            lines.extend(_sfem_soa_element_api_geometry_tile_lines(dim))
        lines.extend(
            [
                "        scalar_t block_h_data[NDOFS][VECTOR_SIZE];",
                "        scalar_t block_out_data[NDOFS][VECTOR_SIZE];",
                "        const scalar_t *block_h_streams[NDOFS];",
                "        scalar_t *block_out_streams[NDOFS];",
                "        for (int stream = 0; stream < NDOFS; ++stream) {",
                "            block_h_streams[stream] = block_h_data[stream];",
                "            block_out_streams[stream] = block_out_data[stream];",
                "        }",
                "        for (int col = 0; col < NDOFS; ++col) {",
                "            for (int stream = 0; stream < NDOFS; ++stream) {",
                "                #pragma omp simd",
                "                for (int lane = 0; lane < nelems; ++lane) {",
                "                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);",
                "                    block_out_data[stream][lane] = scalar_t(0);",
                "                }",
                "            }",
                "            %s" % _sfem_soa_element_api_block_call(
                    form,
                    local_prefix,
                    dim,
                    n_qp,
                    prefix,
                    quadrature_rule,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    "block_out_streams",
                    use_shared_weak_local,
                ),
                "            for (int row = 0; row < NDOFS; ++row) {",
                "                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;",
                "                #pragma omp simd",
                "                for (int lane = 0; lane < nelems; ++lane) {",
                "                    matrix_stream[lane] = block_out_data[row][lane];",
                "                }",
                "            }",
                "        }",
                "    }",
                "    return SFEM_SUCCESS;",
                "}",
                "",
            ]
        )
    return lines


def _sfem_soa_mesh_reference_alias_lines(
    prefix,
    quadrature_rule,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    geometry_mode,
    emit_reference_basis=True,
):
    lines = []
    reference_prefix = "%s_" % geometry_mode
    if not emit_reference_basis:
        lines.append(
            "    const scalar_t *const %sq_weight = %s;"
            % (
                reference_prefix,
                quadrature_reference_accessor(prefix, geometry_mode, "q_weight"),
            )
        )
        return lines
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
    element = quadrature_rule.element_type.lower()
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_%s_soa" % (prefix, form_name)
    return "%s_%s_%s_soa" % (
        prefix,
        element,
        form_name,
    )


def _sfem_soa_isoparametric_public_function_name(prefix, form_name, quadrature_rule):
    if quadrature_rule is None:
        return "%s_%s_isoparametric_soa" % (prefix, form_name)
    element = quadrature_rule.element_type.lower()
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_%s_isoparametric_soa" % (prefix, form_name)
    return "%s_%s_%s_isoparametric_soa" % (
        prefix,
        element,
        form_name,
    )


def _sfem_soa_mesh_public_function_name(prefix, form_name, quadrature_rule, geometry_mode):
    if quadrature_rule is None:
        return "%s_%s_%s_mesh_soa" % (prefix, form_name, geometry_mode)
    element = quadrature_rule.element_type.lower()
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_%s_%s_mesh_soa" % (prefix, form_name, geometry_mode)
    return "%s_%s_%s_%s_mesh_soa" % (
        prefix,
        element,
        form_name,
        geometry_mode,
    )


def _sfem_soa_prefix_has_element_suffix(prefix, element):
    return str(prefix).lower().endswith("_%s" % str(element).lower())


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
