import sympy as sp

from codegen.framework.emitters.kernel_prologue import (
    discard_unused,
    kernel_constant,
    resolve_dead_parameters,
    resolve_kernel_constants,
)
from codegen.framework.plans.conventions import (
    PREFIXES,
    abi_geometry_name,
    abi_local_level,
    abi_mesh_fragment,
    restrict_prelude,
)

#: The staged-buffer and per-thread-scratch prefixes, from the one
#: table that owns them.  Spelling either here again is what made the
#: declaration and the use disagree six times over.
_BLOCK_FMT = PREFIXES["block"] + "%s"
_PACK_FMT = PREFIXES["pack"] + "%s"

from codegen.framework.plans.affine_element_kernel import (
    dof_symbols,
    metric_symbols,
    p1_simplex_metric_apply_plan,
)
from codegen.framework.plans.geometry_variants import geometry_variant_plan
from codegen.framework.emitters.kernel_diagnostics_record import (
    STRUCT_NAME,
    DiagnosticsRecord,
    diagnostics_accessor_lines,
    diagnostics_record_lines,
)
from codegen.framework.emitters.runtime_typed_abi import (
    cast_arguments,
    runtime_typed_entry_point_lines,
)
from codegen.framework.plans.flops import element_flops_plan
from codegen.framework.plans.form_transformations import (
    metric_value_scale,
)
from codegen.framework.plans.form_emission import (
    FormAccumulation,
    element_api_field_roles,
    form_material_parameter_names,
    form_reads_current,
    form_reads_direction,
    publishes_objective_steps,
    form_accumulation,
    mesh_output_shape,
    mesh_output_streams,
    output_assignment,
    output_is_accumulated,
    FormContraction,
    form_contraction,
    form_order,
    form_n_field_components,
    objective_kernel_variants,
    writes_per_shape,
)
from codegen.framework.plans.evaluation_strategy import (
    quadrature_scope_lines,
)
from codegen.framework.plans.affine_element_kernel import (
    expanded_simplex_metric_plan,
    expanded_simplex_metric_value_plan,
)

from codegen.framework.ir.kernel_ast import (
    AssignmentNode,
    BufferDeclNode,
    CallNode,
    FunctionDefNode,
    IfNode,
    LoopHeaderNode,
    LoopKind,
    LoopNode,
    RawLinesNode,
    ReturnNode,
    ScatterNode,
    add_assign_increment,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
)
from codegen.framework.emitters.ast_printer import (
    CLikeKernelASTPrinter,
    PrinterLayout,
    lane_loop_header_lines,
    render_kernel_ast_lines,
)
from codegen.framework.targets import current_target
from codegen.framework.plans.diagnostics import energy_reference_data_traffic
from codegen.framework.plans.kernel_signature import (
    PACKED_MESH_CORE_ARGUMENTS,
    packed_mesh_ghost_reduce_arguments,
)
from codegen.framework.plans.layout import is_tensor_product_family
from codegen.framework.plans.matrix_formats import (
    pattern_scattered_formats,
    BSRAssemblyPlan,
    BlockDiagSymAssemblyPlan,
    packed_crs_passes,
    published_matrix_formats,
)
from codegen.framework.symbolic.core import (
    ExpressionRole,
    KernelExpressions,
)
from codegen.framework.plans.scheduling import (
    ExpressionCost,
    _prune_dead_cse_intermediates,
)
from codegen.framework.emitters.artifacts import (
    GeneratedKernelFile,
)
from codegen.framework.emitters.cprinter import (
    c_group,
    c_product,
    c_sum,
    kernel_status_macro_lines,
    _component_name,
    parameter_list_lines,
    _cpp_argument_name,
    _cpp_macro_name,
    _sfem_ccode,
    _sfem_math_header_source,
)
from codegen.framework.fem.reference import (
    SfemElementQuadratureRule,
    SfemSoAElementSpecialization,
    sfem_element_quadrature_rule,
    sfem_mesh_reference_data,
    sfem_soa_element_specialization,
    sfem_soa_reference_input,
    sfem_tensor_product_hex_uses_cartesian_ordering,
    sfem_tensor_product_quad_uses_cartesian_ordering,
)
from codegen.framework.emitters.tensor_product_geometry import (
    isoparametric_adjugate_call_lines,
    isoparametric_adjugate_stream_array_lines,
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_coordinate_gradient_lines,
    tensor_product_current_q_isoparametric_geometry_lines,
    tensor_product_gradient_isoparametric_geometry_lines,
)
from codegen.framework.emitters.quadrature_codegen import (
    quadrature_reference_accessor,
    reference_header_files,
    reference_include_lines,
    tensor_product_q_index_lines,
    tensor_product_quadrature_weight_expr,
)
from codegen.framework.plans.diagnostics import validate_diagnostics_plan_names
from codegen.framework.plans.reference_data import validate_reference_data_plan
from codegen.framework.plans.form_transformations import (
    constant_p1_simplex_reference_gradients,
)
from codegen.framework.plans.scheduling import build_expression_graph


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
            *restrict_prelude(),
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
            "  T *data{nullptr};",
            "  size_t capacity{0};",
            "",
            "  ~ThreadScratchBuffer() { std::free(data); }",
            "",
            "  T *ensure(const size_t size) {",
            "    if (capacity < size) {",
            "      std::free(data);",
            "      data = static_cast<T *>(std::calloc(size, sizeof(T)));",
            "      capacity = data ? size : 0;",
            "    }",
            "    return data;",
            "  }",
            "};",
            "",
            "template <typename T>",
            "SFEM_INLINE T *thread_scratch(const int slot, const size_t size) {",
            "  static thread_local ThreadScratchBuffer<T> buffers[4];",
            "  return buffers[slot].ensure(size);",
            "}",
            "",
            "template <typename T>",
            "SFEM_INLINE void prealloc_thread_scratch(const int slot, const size_t size) {",
            "#pragma omp parallel",
            "  {",
            "    (void)thread_scratch<T>(slot, size);",
            "  }",
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


def _reference_gradient_offset_lines(
    name, n_field_components, dim, indent, pointer="const s_t *const RSTR"
):
    """Name each quadrature point's slice of a reference gradient, above the loop.

    The slice belongs to the quadrature point, not to the lane, so computing
    `(3 * (NQ + q) + 1) * VS + lane` once per lane rebuilds a base address the
    whole loop shares.  Naming the base outside leaves `gu_ref1[lane]` inside,
    which is the same load without the index arithmetic and without the
    `s_t gu_ref[9]` the kernel used to fill in order to avoid writing the index
    out nine times.
    """
    return tuple(
        "%s%s %s%d = &%s_q[%s * VS];"
        % (
            indent,
            pointer,
            name,
            row * dim + col,
            name,
            c_group(
                c_sum(
                    c_product(c_group(c_sum(c_product(row, "NQ"), "q")), dim),
                    col,
                )
            ),
        )
        for row in range(n_field_components)
        for col in range(dim)
    )


def _lane_loop_header_lines(source_builder, indent):
    """A packed kernel's lane loop: its pragma and its `for`, from the IR.

    Seven sites spliced `*source_builder.simd_lines()` and then wrote the `for`
    out by hand.  The loop is `LoopHeaderNode` over the same `LoopNode` that
    `_work_item_loop_lines` below renders, so the two stop being separate
    spellings of one loop.

    The pragma lands at column zero and the `for` at `indent`, which is what
    these sites have always emitted -- `simd_lines()` returns the bare pragma
    and the splice never indented it.  That asymmetry is visible in the shipped
    tree, where 210 of 2935 `#pragma omp simd` lines sit at column zero.  It is
    preserved here rather than quietly corrected, because correcting it moves
    shipped bytes and that is a decision to take deliberately, not a side effect
    of moving a loop into the IR.
    """
    pragma = tuple(source_builder.simd_lines())
    return lane_loop_header_lines(
        pragma[0] if pragma else None, indent
    )


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
                            LoopHeaderNode(
                                LoopNode(
                                    LoopKind.SIMD,
                                    lane_iterator,
                                    iteration_range(0, expr_ref("ne", "tile_extent")),
                                    pre_increment(lane_iterator),
                                    vectorized=bool(pragma),
                                )
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
                LoopHeaderNode(
                    LoopNode(
                        LoopKind.SIMD,
                        lane_iterator,
                        iteration_range(0, expr_ref("ne", "tile_extent")),
                        pre_increment(lane_iterator),
                        vectorized=bool(simd_lines),
                    )
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
    geometry_scalar_type="g_t",
):
    lines = []
    for array_input in element_inputs:
        for stream in _soa_array_stream_names(array_input):
            lines.extend(
                [
                    "%ss_t b%s_data[VS];" % (indent, stream),
                    "%sconst s_t *const b%s = ageom_stream<s_t, %s, VS>("
                    % (indent, stream, geometry_scalar_type),
                    "%s    ne, %s + evb, b%s_data, std::is_same<%s, s_t>());"
                    % (indent, abi_geometry_name(stream), stream, geometry_scalar_type),
                ]
            )
    return lines


def _affine_geometry_stream_helper_lines(source_builder):
    inline_qualifier = source_builder.inline_qualifier()
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename s_t, typename g_t, int VS>",
        "%s const s_t *ageom_stream(" % inline_qualifier,
        "    const int,",
        "    const g_t *const RSTR source,",
        "    s_t *const RSTR,",
        "    std::true_type) {",
        "  return source;",
        "}",
        "",
        "template <typename s_t, typename g_t, int VS>",
        "%s const s_t *ageom_stream(" % inline_qualifier,
        "    const int ne,",
        "    const g_t *const RSTR source,",
        "    s_t *const RSTR converted,",
        "    std::false_type) {",
    ]
    if _emits_vector_lane_loop(source_builder):
        lines.extend("  %s" % line for line in source_builder.simd_lines())
        lines.extend(
            [
                "  for (int lane = 0; lane < ne; ++lane) {",
                "    converted[lane] = s_t(source[lane]);",
                "  }",
            ]
        )
    else:
        index = _work_item_index(source_builder)
        lines.extend(
            [
                discard_unused("ne", indent="  "),
                "  converted[%s] = s_t(source[%s]);" % (index, index),
            ]
        )
    lines.extend(
        [
            "  return converted;",
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
            "%s  %s" % (indent, line)
            for line in render_kernel_ast_lines(
                "scatter_loop_header",
                (
                    LoopHeaderNode(
                        LoopNode(
                            LoopKind.SCATTER,
                            iterator("scatter", "int"),
                            iteration_range(0, expr_ref("ne", "tile_extent")),
                            pre_increment(iterator("scatter", "int")),
                        )
                    ),
                ),
            )
        ),
    ]
    if atomic_pragma is not None:
        lines.extend(
            "%s    %s" % (indent, line)
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
                "%s    " % indent,
            )
        )
    lines.extend(
        [
            "%s  }" % indent,
            "%s}" % indent,
        ]
    )
    return tuple(lines)


def _work_item_name(source_builder, name, component):
    if hasattr(source_builder, "work_item_name"):
        return source_builder.work_item_name(name, component)
    return "%s_%s%d" % (name, _work_item_index(source_builder), component)


def _host_function_qualifier(source_builder):
    """The bound target's spelling for a function the device never calls."""
    target = getattr(source_builder, "target", None)
    if target is not None and hasattr(target, "host_function_qualifier"):
        return target.host_function_qualifier()
    return _inline_qualifier(source_builder)


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
    if quadrature_rule is None:
        # Established once, here, so that nothing below has to ask.  Every
        # production path already supplies one -- the element entry point
        # passes the specialization's rule -- and roughly twenty helpers
        # nonetheless carried an `is None` arm, three of which raised this same
        # complaint from deep inside emission.  A precondition belongs at the
        # boundary the caller crosses, not at each place that would trip over
        # its absence.
        raise ValueError(
            "energy code generation requires an element quadrature rule: pass "
            "quadrature_rule, or element_type for one to be derived from"
        )
    dim = quadrature_rule.dim
    n_nodes = quadrature_rule.n_shape
    n_qp = quadrature_rule.n_qp
    if affine_quadrature_rule is None:
        affine_quadrature_rule = quadrature_rule
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
    _validate_sfem_soa_quadrature_rule(quadrature_rule, dim, n_nodes, n_qp, array_inputs)
    validate_reference_data_plan(
        reference_data_plan,
        prefix,
        affine_quadrature_rule,
        quadrature_rule,
        basis_family,
    )
    validate_diagnostics_plan_names(
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
    # The header holds one thing, the direct element matrix, and only a matrix
    # assembly calls it.  A material with no matrix format publishes no header.
    emits_hessian_header = bool(
        published_matrix_formats(matrix_format_plan)
    ) and _sfem_soa_emits_hessian_header(
        forms,
        array_inputs,
    )
    files = list(shared_primitive_files(source_builder, basis_family))
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
    return tuple(files) + _sfem_soa_reference_header_files(
        (affine_quadrature_rule, quadrature_rule)
    )


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


def _metric_array_inputs(array_inputs, dim):
    """The same reference data, with the metric in place of the adjugate.

    The reference gradients are unchanged -- the basis is the basis -- and only
    the geometry differs: one symmetric metric where there were `dim * dim`
    adjugate components and a determinant.
    """
    from codegen.framework.fem.reference import sfem_soa_array_input
    from codegen.framework.plans.form_transformations import (
        symmetric_metric_component_count,
    )

    # The metric kernel reads no reference data at all: the basis gradients of a
    # constant-P1 simplex are folded into the plan, and the quadrature weight is
    # already carried by `FFF`.  Keeping `grad_ref` and `q_weight` here would
    # put two unused parameters on every one of these kernels.
    return (
        sfem_soa_array_input("geom_metric", symmetric_metric_component_count(dim)),
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
        "typedef ptrdiff_t count_t;",
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
            # And, when the contraction factors through the metric, a third
            # block for the affine variant alone.  It cannot replace the one
            # above: the isoparametric variant builds its geometry from
            # coordinates and has an adjugate rather than a cached metric, so
            # the two modes genuinely need different kernels.  That is the
            # shape the residual path already has.
            metric = geometry_variant_plan(form.weak_form, specialized_rule).cached_metric
            if metric is not None:
                metric_array_inputs = _metric_array_inputs(array_inputs, dim)
                lines.extend(
                    _sfem_soa_block_function(
                        form,
                        prefix,
                        dim,
                        n_nodes,
                        metric_array_inputs,
                        specialized_rule,
                        basis_family,
                        use_shared_weak_local,
                        source_builder,
                        function_name="%s_metric_%s_block"
                        % (specialized_prefix, form.name),
                        constant_p1_gradient_expansion=True,
                    )
                )
                lines.append("")

    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(resolve_dead_parameters(resolve_kernel_constants(lines)))


def _sfem_soa_emits_hessian_header(
    forms,
    array_inputs,
):
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    return any(
        _sfem_soa_direct_hessian_matrix_assembly_available(
            form,
            reference_inputs,
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
        "typedef ptrdiff_t count_t;",
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
            reference_inputs,
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
    return "\n".join(resolve_dead_parameters(resolve_kernel_constants(lines)))


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
    n_field_components = form_n_field_components(form, dim)
    params = [
        *(
            "const s_t *const RSTR badj%d" % component
            for component in range(dim * dim)
        ),
        "const s_t *const RSTR bdet0",
    ]
    if use_tensor_product_reference:
        params.extend(
            (
                "const s_t *const RSTR shape_1d",
                "const s_t *const RSTR grad_1d",
                "const s_t *const RSTR q_weight_1d",
            )
        )
    elif use_reference_gradient_vectors:
        params.extend(_sfem_reference_gradient_vector_params(dim))
        params.append("const s_t *const RSTR q_weight")
    else:
        params.extend(
            "const %s *const RSTR %s"
            % (array_input.scalar_type, _sfem_soa_reference_param_name(array_input))
            for array_input in reference_inputs
        )
        params.append("const s_t *const RSTR q_weight")
    params.extend(_form_material_parameter_declarations(form))
    # The current state, where the tangent depends on it.  Same shape the apply
    # takes it in, so the caller hands over the streams it already gathered.
    # The component count is settled by the form, and the signature is written
    # before the body declares `NC`, so it goes in as a number.
    params.extend(
        "const s_t b%s_data[NS * %d][VS]" % (stream_prefix, n_field_components)
        for _role, stream_prefix in element_api_field_roles(form)
        if stream_prefix == "u"
    )
    params.append("s_t *const RSTR element_matrix")

    lines = [
        "template <typename s_t, int NQ, int NS, int VS>",
        "static %s void %s(" % (_inline_qualifier(source_builder), name),
    ]
    lines.extend(parameter_list_lines(params))
    lines.extend(
        [
            ") {",
            "  static_assert(NQ > 0, \"NQ must be positive\");",
            "  static_assert(NS > 0, \"NS must be positive\");",
            "  static_assert(VS > 0, \"VS must be positive\");",
            kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
            kernel_constant("NDOFS", "NC * NS", indent="  "),
        ]
    )
    if use_tensor_product_reference:
        lines.extend(
            [
                kernel_constant("NQ1", "integer_root(NQ, %d)" % quadrature_rule.dim, indent="  "),
                kernel_constant("NS1", "integer_root(NS, %d)" % quadrature_rule.dim, indent="  "),
                "  static_assert(ipow(NQ1, %d) == NQ, \"NQ must be tensor-product compatible\");"
                % quadrature_rule.dim,
                "  static_assert(ipow(NS1, %d) == NS, \"NS must be tensor-product compatible\");"
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
            "  ",
            emit_tensor_product_static_constants=False,
        )
    )
    lines.append("}")
    # The signature is a tree; the body is not yet.  Splitting the
    # finished list at its opening brace is deliberately the least
    # invasive way to get this kernel onto the IR spine: the body
    # keeps building exactly as before and is carried verbatim.
    opening = lines.index(") {")
    return _print_energy_kernel(
        FunctionDefNode(
            name,
            params=tuple(params),
            body=(
                RawLinesNode(
                    tuple(lines[opening + 1 : -1]),
                    reason="_sfem_soa_direct_hessian_element_matrix_function body: not yet IR",
                ),
            ),
            qualifier="static %s" % _inline_qualifier(source_builder),
            template_params=(
                "typename s_t",
                "int NQ",
                "int NS",
                "int VS",
            ),
        )
    )


class _BlockKernelInputs:
    """What both element block kernels need before they diverge.

    Computed once by the dispatcher and handed to whichever kernel the form
    turns out to be, so the two do not each re-derive it.
    """

    __slots__ = (
        "name",
        "work_item",
        "element_inputs",
        "reference_inputs",
        "use_tensor_product_reference",
        "use_reference_gradient_vectors",
        "stream_shape_order",
        "uses_current",
        "uses_direction",
        "source_builder",
    )

    def __init__(self, **fields):
        for field in self.__slots__:
            setattr(self, field, fields[field])


def _sfem_soa_block_signature_lines(name, params, source_builder):
    """The template header, parameter list and asserts both kernels open with."""
    lines = [
        "template <typename s_t, int NQ, int NS, int VS>",
        "static %s void %s(" % (_inline_qualifier(source_builder), name),
    ]
    lines.extend(parameter_list_lines(params))
    lines.extend(
        [
            ") {",
            '  static_assert(NQ > 0, \"NQ must be positive\");',
            '  static_assert(VS > 0, \"VS must be positive\");',
        ]
    )
    return lines


def _sfem_soa_element_stream_params(shared):
    """One parameter per stream of every element input."""
    return [
        "const %s *const RSTR %s" % (array_input.scalar_type, stream)
        for array_input in shared.element_inputs
        for stream in _soa_array_stream_names(array_input)
    ]


def _sfem_soa_reference_basis_params(dim, shared, omit_reference_basis_inputs):
    """The reference basis a block kernel is handed, in its four shapes."""
    if omit_reference_basis_inputs:
        return []
    if shared.use_tensor_product_reference:
        return [
            "const s_t *const RSTR shape_1d",
            "const s_t *const RSTR grad_1d",
        ]
    if shared.use_reference_gradient_vectors:
        return list(_sfem_reference_gradient_vector_params(dim))
    return [
        "const %s *const RSTR %s"
        % (array_input.scalar_type, _sfem_soa_reference_param_name(array_input))
        for array_input in shared.reference_inputs
    ]


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
    """Emit one element block kernel, of whichever kind this form is.

    Two kernels have always lived behind this name and the form decides which.
    A form carrying a lowered weak form becomes a quadrature kernel: it takes a
    geometry stride and an array of quadrature weights, and its body comes from
    the weak-form emitters.  A form without one becomes the older per-point
    kernel: it takes a quadrature index and a single scalar weight, gathers its
    inputs into local arrays, and evaluates an expression graph.  They share a
    signature shape and little else -- they do not even return the same thing,
    one a list of lines and the other a printed IR function.

    ``form.weak_form is not None`` used to be asked thirteen times through a
    single three-hundred-line body, which is what made the two hard to see.  It
    is asked once, here.  Each kernel then knows which one it is, and the
    conditions that followed collapse: the per-point kernel cannot have stream
    arrays and cannot omit its reference basis, because both are weak-form-only,
    and it no longer has to say so.
    """
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    shared = _BlockKernelInputs(
        name=function_name or "%s_%s_block" % (prefix, form.name),
        work_item=_work_item_index(source_builder),
        element_inputs=_sfem_soa_element_inputs(array_inputs),
        reference_inputs=reference_inputs,
        use_tensor_product_reference=use_tensor_product_reference,
        use_reference_gradient_vectors=(
            not use_tensor_product_reference
            and len(reference_inputs) == 1
            and reference_inputs[0].name == "grad_ref"
        ),
        stream_shape_order=(
            _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
            if use_tensor_product_reference
            else tuple(range(n_nodes))
        ),
        uses_current=form_reads_current(form, default=True),
        uses_direction=form_reads_direction(form, default=form.has_direction),
        source_builder=source_builder,
    )
    # Selected, not branched on.  Both take the same arguments now, so which
    # one runs is a lookup on a property of the form rather than a conditional
    # in emission.
    return _BLOCK_FUNCTION_BY_CONTRACTION[form_contraction(form)](
        form,
        prefix,
        dim,
        n_nodes,
        array_inputs,
        quadrature_rule,
        shared,
        use_shared_weak_local=use_shared_weak_local,
        constant_p1_gradient_expansion=constant_p1_gradient_expansion,
    )


def _sfem_soa_weak_form_block_function(
    form,
    prefix,
    dim,
    n_nodes,
    array_inputs,
    quadrature_rule,
    shared,
    use_shared_weak_local=False,
    constant_p1_gradient_expansion=False,
):
    """The quadrature kernel: a form with a lowered weak form."""
    n_field_components = form_n_field_components(form, dim)
    source_builder = shared.source_builder
    use_stream_arrays = use_shared_weak_local
    omit_reference_basis_inputs = (
        constant_p1_gradient_expansion and not shared.use_tensor_product_reference
    )

    params = ["const int ne", "const ptrdiff_t geometry_stride"]
    params.extend(_sfem_soa_element_stream_params(shared))
    params.extend(
        _sfem_soa_reference_basis_params(dim, shared, omit_reference_basis_inputs)
    )
    if shared.use_tensor_product_reference:
        params.append("const s_t *const RSTR q_weight_1d")
    else:
        params.append("const s_t *const RSTR q_weight")
    params.extend(_form_material_parameter_declarations(form))
    if use_stream_arrays:
        if shared.uses_current:
            params.append(
                "const s_t *const RSTR u_streams[NS * %d]" % n_field_components
            )
        if shared.uses_direction:
            params.append(
                "const s_t *const RSTR h_streams[NS * %d]" % n_field_components
            )
        params.append(
            _BLOCK_OUTPUT_PARAMETER[form_accumulation(form)](n_field_components)
        )
    else:
        if shared.uses_current:
            params.extend(
                "const s_t *const RSTR %s" % name
                for name in _field_stream_names("u", n_field_components, n_nodes)
            )
        if shared.uses_direction:
            params.extend(
                "const s_t *const RSTR %s" % name
                for name in _field_stream_names("h", n_field_components, n_nodes)
            )
        params.extend(
            "s_t *const RSTR %s" % name
            for name in _output_stream_names(form, n_field_components, n_nodes)
        )

    lines = _sfem_soa_block_signature_lines(shared.name, params, source_builder)
    if not use_stream_arrays:
        lines.append(
            '  static_assert(NS == %d, \"NS does not match generated expression\");'
            % n_nodes
        )
    if shared.use_tensor_product_reference:
        if use_stream_arrays:
            lines.extend(
                [
                    kernel_constant("NQ1", "integer_root(NQ, %d)" % quadrature_rule.dim, indent="  "),
                    kernel_constant("NS1", "integer_root(NS, %d)" % quadrature_rule.dim, indent="  "),
                    '  static_assert(ipow(NQ1, %d) == NQ, \"NQ must be tensor-product compatible\");'
                    % quadrature_rule.dim,
                    '  static_assert(ipow(NS1, %d) == NS, \"NS must be tensor-product compatible\");'
                    % quadrature_rule.dim,
                ]
            )
        else:
            lines.extend(
                [
                    kernel_constant("NQ1", "%d" % quadrature_rule.tensor_product_n_qp_1d, indent="  "),
                    kernel_constant("NS1", "%d" % quadrature_rule.tensor_product_n_shape_1d, indent="  "),
                ]
            )
    if not use_stream_arrays and shared.uses_current:
        lines.append(
            "  const s_t *const weak_u_streams[NS * %d] = {%s};"
            % (
                n_field_components,
                ", ".join(
                    streams_in_shape_order(
                        _field_stream_names("u", n_field_components, n_nodes),
                        n_field_components,
                        shared.stream_shape_order,
                    )
                ),
            )
        )
    if not use_stream_arrays and shared.uses_direction:
        lines.append(
            "  const s_t *const weak_h_streams[NS * %d] = {%s};"
            % (
                n_field_components,
                ", ".join(
                    streams_in_shape_order(
                        _field_stream_names("h", n_field_components, n_nodes),
                        n_field_components,
                        shared.stream_shape_order,
                    )
                ),
            )
        )
    if not use_stream_arrays:
        # Whether the block takes stream arrays is the caller's axis; what there
        # is to declare is the plan's, and a scalar accumulator declares nothing.
        lines.extend(
            _WEAK_OUTPUT_STREAM_ARRAY[form_accumulation(form)](
                form, dim, n_field_components, n_nodes, shared.stream_shape_order
            )
        )
    if shared.use_tensor_product_reference:
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

    _append_sfem_soa_weak_form_lines(
        lines,
        form,
        prefix,
        dim,
        n_nodes,
        shared.reference_inputs,
        shared.use_tensor_product_reference,
        quadrature_rule,
        use_stream_arrays,
        source_builder,
        constant_p1_gradient_expansion=constant_p1_gradient_expansion,
        # The geometry this block was given says which kernel it is: the metric
        # block is emitted with metric inputs, the general one with the
        # adjugate, and the body follows.
        metric=geometry_variant_plan(
            form.weak_form,
            quadrature_rule,
            specialized=any(
                getattr(entry, "name", "") == "geom_metric"
                for entry in shared.element_inputs
            ),
        ).cached_metric,
    )
    lines.append("}")
    return lines


def _sfem_soa_pointwise_block_function(
    form,
    prefix,
    dim,
    n_nodes,
    array_inputs,
    quadrature_rule,
    shared,
    use_shared_weak_local=False,
    constant_p1_gradient_expansion=True,
):
    """Emit the block kernel for a form whose contraction is already done.

    Takes the same arguments as its deferred-flux counterpart and ignores the
    ones it does not need, so the two can be selected from a table instead of
    branched on.  That interchangeability is what "no distinction below the
    form layer" means concretely for this emitter.
    """
    n_field_components = form_n_field_components(form, dim)
    """The per-point kernel: a form with no lowered weak form.

  Takes one quadrature index and one weight, gathers every input into a local
  array, and evaluates the expression graph.  Stream arrays and an omitted
  reference basis are weak-form-only, so neither appears here.
  """
    source_builder = shared.source_builder
    work_item = shared.work_item

    params = ["const int ne", "const int q"]
    params.extend(_sfem_soa_element_stream_params(shared))
    params.extend(
        _sfem_soa_reference_basis_params(
            dim, shared, omit_reference_basis_inputs=False
        )
    )
    params.append("const s_t qw")
    params.extend(_form_material_parameter_declarations(form))
    if shared.uses_current:
        params.extend(
            "const s_t *const RSTR %s" % name
            for name in _field_stream_names("u", n_field_components, n_nodes)
        )
    if shared.uses_direction:
        params.extend(
            "const s_t *const RSTR %s" % name
            for name in _field_stream_names("h", n_field_components, n_nodes)
        )
    params.extend(
        "s_t *const RSTR %s" % name
        for name in _output_stream_names(form, n_field_components, n_nodes)
    )

    lines = _sfem_soa_block_signature_lines(shared.name, params, source_builder)
    lines.append(
        '  static_assert(NS == %d, \"NS does not match generated expression\");'
        % n_nodes
    )
    if shared.use_tensor_product_reference:
        lines.extend(
            [
                kernel_constant("NQ1", "%d" % quadrature_rule.tensor_product_n_qp_1d, indent="  "),
                kernel_constant("NS1", "%d" % quadrature_rule.tensor_product_n_shape_1d, indent="  "),
            ]
        )
        lines.extend(tensor_product_q_index_lines(quadrature_rule.dim, "  "))

    lines.extend(_work_item_loop_lines(source_builder, "  "))
    lines.append("    s_t u[NS * %d];" % dim)
    for array_input in array_inputs:
        if array_input.is_reference_qp_shape:
            array_decl = "%s %s[NS * %d];" % (
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
        lines.append("    %s" % array_decl)
    if form.has_direction:
        lines.append("    s_t du[NS * %d];" % dim)

    for array_input in shared.element_inputs:
        for i, stream in enumerate(_soa_array_stream_names(array_input)):
            lines.append("    %s[%d] = %s[%s];" % (array_input.name, i, stream, work_item))

    if shared.use_tensor_product_reference:
        _append_tensor_product_reference_gradient_lines(
            lines,
            shared.reference_inputs[0].name,
            quadrature_rule,
        )
    elif shared.use_reference_gradient_vectors:
        array_input = shared.reference_inputs[0]
        for shape in range(array_input.n_shape):
            for component in range(array_input.components):
                local_idx = shape * array_input.components + component
                lines.append(
                    "    %s[%d] = %s[q * NS + %d];"
                    % (
                        array_input.name,
                        local_idx,
                        _sfem_reference_gradient_vector_name(component),
                        shape,
                    )
                )
    else:
        for array_input in shared.reference_inputs:
            source = _sfem_soa_reference_param_name(array_input)
            for shape in range(array_input.n_shape):
                for component in range(array_input.components):
                    local_idx = shape * array_input.components + component
                    lines.append(
                        "    %s[%d] = %s[(q * NS + %d) * %d + %d];"
                        % (
                            array_input.name,
                            local_idx,
                            source,
                            shape,
                            array_input.components,
                            component,
                        )
                    )

    for node in range(n_nodes):
        for d in range(dim):
            idx = node * dim + d
            component = _component_name(d)
            lines.append("    u[%d] = u%s%d[%s];" % (idx, component, node, work_item))
            if form.has_direction:
                lines.append("    du[%d] = h%s%d[%s];" % (idx, component, node, work_item))

    output_count = len(form.expression_graph.evaluation_plan.outputs)
    lines.append("    s_t element_vector[%d];" % max(1, output_count))
    _append_sfem_soa_statement_lines(lines, form.expression_graph, "element_vector")
    _append_sfem_soa_output_lines(lines, form, dim, n_nodes, work_item)
    lines.extend(["  }", "}"])
    # The signature is a tree; the body is not yet.  Splitting the
    # finished list at its opening brace is deliberately the least
    # invasive way to get this kernel onto the IR spine: the body
    # keeps building exactly as before and is carried verbatim.
    opening = lines.index(") {")
    return _print_energy_kernel(
        FunctionDefNode(
            shared.name,
            params=tuple(params),
            body=(
                RawLinesNode(
                    tuple(lines[opening + 1 : -1]),
                    reason="_sfem_soa_block_function body: not yet IR",
                ),
            ),
            qualifier="static %s" % _inline_qualifier(source_builder),
            template_params=(
                "typename s_t",
                "int NQ",
                "int NS",
                "int VS",
            ),
        )
    )

#: Which block emitter serves which contraction.  A table because the choice
#: is a property of the form and not a decision emission makes; the two
#: functions take the same arguments so either can be selected.
_BLOCK_FUNCTION_BY_CONTRACTION = {
    FormContraction.DEFERRED_FLUX: _sfem_soa_weak_form_block_function,
    FormContraction.POINTWISE: _sfem_soa_pointwise_block_function,
}



def _sfem_soa_direct_hessian_matrix_assembly_available(
    form,
    reference_inputs,
):
    if form.name != "apply" or form.weak_form is None:
        return False
    if len(reference_inputs) != 1 or reference_inputs[0].name != "grad_ref":
        return False
    # A state-dependent form used to be excluded here, which sent every
    # hyperelastic material down a fallback that recovered the element matrix one
    # column at a time by applying the operator to unit basis vectors: 24 applies
    # per HEX8 element, each recomputing the geometry, the deformation gradient
    # and the material tangent at every quadrature point and keeping one column
    # of the result.  The tangent of a hyperelastic operator at a state is a
    # perfectly ordinary expression -- it just needs the current deformation
    # gradient, which the kernel now takes the state to compute -- so the
    # fallback is gone and this is the only way an element matrix is formed.
    return True


def _constant_p1_specialized_local(local_prefix, quadrature_rule):
    specialized_prefix = _constant_p1_specialized_local_prefix(
        local_prefix,
        quadrature_rule,
    )
    if specialized_prefix is not None:
        return specialized_prefix, quadrature_rule

    if not str(local_prefix).endswith("_simplex"):
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
    if constant_p1_simplex_reference_gradients(quadrature_rule) is None:
        return None
    element_type = str(getattr(quadrature_rule, "element_type", "")).lower()
    if element_type not in ("tri3", "tet4"):
        return None
    return "%s_%s" % (local_prefix, element_type)


#: The two tails a tensor-product weak-form body can have.  Both take the same
#: arguments because they are two implementations of one step -- what this body
#: does with its integrand -- and which applies is
#: `plans.form_emission.form_accumulation`.  They were an early return in the
#: middle of the emitter, which is the same choice made where it cannot be
#: named.
def _tensor_weak_scalar_tail(
    lines, form, weak_form, substitutions, dim, n_field_components,
    work_item, geometry_value, out_streams, closing,
):
    """A 0-form: weight the density and add it in.  Nothing to contract."""
    _append_weak_objective_accumulation(
        lines, form, weak_form, substitutions, work_item, geometry_value, closing
    )


def _tensor_weak_per_shape_tail(
    lines, form, weak_form, substitutions, dim, n_field_components,
    work_item, geometry_value, out_streams, closing,
):
    """A 1- or 2-form: form the loperand, then sum-factorise it onto the tests."""
    material = _weak_form_material_expression(
        weak_form,
        form.name,
        substitutions,
        tuple(
            sp.symbols("trial_grad[%d]" % i)
            for i in range(weak_form.n_field_components * dim)
        ),
    )
    lines.append("      s_t loperand[%d];" % (n_field_components * dim))
    _append_transformed_loperand_lines(
        lines,
        material,
        dim,
        "weak_mat_tmp",
        geometry_value,
    )
    for row in range(n_field_components):
        for col in range(dim):
            lines.append(
                "      loperand%d[%s] = loperand[%d];"
                % (row * dim + col, work_item, row * dim + col)
            )
    lines.extend(["    }", "  }"])
    for row in range(n_field_components):
        lines.append(
            "  tensor_test<s_t, NQ, NS, VS, %d, %d>(ne, shape_1d, grad_1d, &loperand_q[%s], %s, %d);"
            % (dim, n_field_components, c_product(row, "NQ", dim, "VS"), out_streams, row)
        )


_TENSOR_WEAK_TAIL = {
    FormAccumulation.SCALAR: _tensor_weak_scalar_tail,
    FormAccumulation.PER_SHAPE: _tensor_weak_per_shape_tail,
}


def _append_sfem_soa_tensor_weak_form_lines(
    lines,
    form,
    prefix,
    dim,
    use_stream_arrays,
    source_builder=None,
):
    n_field_components = form_n_field_components(form, dim)
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
    weak_form = form.weak_form
    uses_current = form_reads_current(form, default=True)
    uses_direction = form_reads_direction(form, default=form.has_direction)
    u_streams = "u_streams" if use_stream_arrays else "weak_u_streams"
    h_streams = "h_streams" if use_stream_arrays else "weak_h_streams"
    out_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
    block_extent = "NQ * %d * VS" % (dim * dim)

    if uses_current:
        lines.append("  s_t gu_ref_q[%s];" % block_extent)
    if uses_direction:
        lines.append("  s_t grad_h_ref_q[%s];" % block_extent)
    lines.extend(
        "  s_t %s[%s];" % (name, block_extent)
        for name in _POINT_BUFFERS[form_accumulation(form)]
    )

    for row in range(n_field_components):
        output_offset = c_product(row, "NQ", dim, "VS")
        if uses_current:
            lines.append(
                "  tensor_gradient<s_t, NQ, NS, VS, %d, %d>(ne, shape_1d, grad_1d, %s, %d, &gu_ref_q[%s]);"
                % (dim, n_field_components, u_streams, row, output_offset)
            )
        if uses_direction:
            lines.append(
                "  tensor_gradient<s_t, NQ, NS, VS, %d, %d>(ne, shape_1d, grad_1d, %s, %d, &grad_h_ref_q[%s]);"
                % (dim, n_field_components, h_streams, row, output_offset)
            )

    lines.append("  for (int q = 0; q < NQ; ++q) {")
    lines.extend(tensor_product_q_index_lines(dim, "    "))
    lines.append(
        "    const s_t qw = %s;"
        % tensor_product_quadrature_weight_expr(dim)
    )
    if uses_current:
        lines.extend(
            _reference_gradient_offset_lines("gu_ref", n_field_components, dim, "    ")
        )
    if uses_direction:
        lines.extend(
            _reference_gradient_offset_lines(
                "grad_h_ref", n_field_components, dim, "    "
            )
        )
    for _written in [candidate for candidate in (form,) if writes_per_shape(candidate)]:
        lines.extend(
            _reference_gradient_offset_lines(
                "loperand", n_field_components, dim, "    ", pointer="s_t *const RSTR"
            )
        )
    # The geometry stream is indexed by quadrature point and lane, and only the
    # lane part varies inside the loop, so the point's base is named once here.
    for component in range(dim * dim):
        lines.append(
            "    const s_t *const RSTR adj_q%d = adj%d + q * geometry_stride;"
            % (component, component)
        )
    lines.append("    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;")
    lines.extend(_work_item_loop_lines(source_builder, "    "))
    for component in range(dim * dim):
        lines.append(
            "      const s_t %s = adj_q%d[%s];"
            % (
                _work_item_name(source_builder, "adj", component),
                component,
                work_item,
            )
        )
    lines.append(
        "      const s_t %s = det_q0[%s];"
        % (_work_item_name(source_builder, "det", 0), work_item)
    )
    def geometry_value(name, component):
        return _work_item_name(source_builder, name, component)

    if uses_current:
        lines.append(
            "      s_t gu[%d];" % (weak_form.n_field_components * dim)
        )
    if uses_direction:
        lines.append(
            "      s_t trial_grad[%d];" % (weak_form.n_field_components * dim)
        )

    lines.append(
        "      const s_t idet = s_t(1) / %s;"
        % geometry_value("det", 0)
    )
    for row in range(weak_form.n_field_components):
        for col in range(dim):
            if uses_current:
                terms = [
                    "gu_ref%d[%s] * %s"
                    % (
                        row * dim + k,
                        work_item,
                        geometry_value("adj", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "      gu[%d] = (%s) * idet;"
                    % (row * dim + col, " + ".join(terms))
                )
            if uses_direction:
                terms = [
                    "grad_h_ref%d[%s] * %s"
                    % (
                        row * dim + k,
                        work_item,
                        geometry_value("adj", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "      trial_grad[%d] = (%s) * idet;"
                    % (row * dim + col, " + ".join(terms))
                )

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "gu",
    )
    _TENSOR_WEAK_TAIL[form_accumulation(form)](
        lines,
        form,
        weak_form,
        deformation_gradient_substitutions,
        dim,
        n_field_components,
        work_item,
        geometry_value,
        out_streams,
        ["    }", "  }"],
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
        return "s_t(0)"
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


def _metric_plan_bindings(form, dim, source_builder, use_stream_arrays):
    """The plan's abstract `fff` and `u` bound to what this ABI calls them."""
    n_field_components = form_n_field_components(form, dim)
    work_item = _work_item_index(source_builder)
    uses_direction = form_reads_direction(form, default=form.has_direction)
    field = "h" if uses_direction else "u"
    stream_prefix = "" if use_stream_arrays else "weak_"
    bindings = {}
    for index, symbol in enumerate(metric_symbols(dim)):
        bindings[symbol] = sp.Symbol(
            _work_item_name(source_builder, "geom_metric", index)
        )
    for shape, symbol in enumerate(dof_symbols(dim)):
        bindings[symbol] = sp.Symbol(
            "%s%s_streams[%s][%s]"
            % (stream_prefix, field, c_product(shape, n_field_components), work_item)
        )
    return bindings, n_field_components, work_item


def _metric_scatter_lines(dim, metric, plan, bindings, n_field_components,
                          work_item, use_stream_arrays):
    """A 1- or 2-form: one contribution per shape, scattered."""
    output_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
    return [
        "      %s[%s][%s] += %s;"
        % (
            output_streams,
            c_product(shape, n_field_components),
            work_item,
            _sfem_ccode((metric.scale * expression).xreplace(bindings)),
        )
        for shape, expression in enumerate(plan.outputs)
    ]


def _metric_value_lines(dim, metric, plan, bindings, n_field_components,
                        work_item, use_stream_arrays):
    """A 0-form: half the gradient contracted with its own flux, per element.

    `plan.outputs` after the first are the flux entries -- the first is minus
    their sum -- so the energy is half the gradient dotted with them, reusing
    the same temporaries.
    """
    dofs = dof_symbols(dim)
    gradient = [dofs[d + 1] - dofs[0] for d in range(dim)]
    energy = sp.Rational(1, 2) * metric.scale * sum(
        gradient[d] * plan.outputs[d + 1] for d in range(dim)
    )
    return [
        "      value[%s] += %s;"
        % (work_item, _sfem_ccode(energy.xreplace(bindings)))
    ]


#: Which body a metric-carried form emits, by whether it scatters per shape.
#: A table rather than a test, so the choice reads as the form algebra's.
_METRIC_BODY_BY_WRITES_PER_SHAPE = {
    True: _metric_scatter_lines,
    False: _metric_value_lines,
}


def _append_constant_p1_metric_weak_form_lines(
    lines,
    form,
    dim,
    metric,
    use_stream_arrays,
    source_builder,
):
    """The P1 simplex contraction carried through a cached metric.

    `grad(v) . flux` factors as `B^T (scale * FFF) B`, and the element
    contribution is the plan in `plans.affine_element_kernel` -- the same one
    the residual path spells, so both formulations emit the same arithmetic for
    the same operator rather than each deriving it.

    The saving is the geometry: six symmetric components where the adjugate
    form reads nine and a determinant.
    """
    plan = p1_simplex_metric_apply_plan(dim)
    bindings, n_field_components, work_item = _metric_plan_bindings(
        form, dim, source_builder, use_stream_arrays
    )
    lines.extend(_work_item_loop_lines(source_builder, "    "))
    # A constant-P1 simplex has one quadrature point, so the offset is just the
    # work item: `q` is zero and the stride term vanishes.
    lines.append(
        "      const ptrdiff_t goff = %s;" % work_item
    )
    for index in range(metric.metric_components):
        lines.append(
            "      const s_t %s = geom_metric%d[goff];"
            % (_work_item_name(source_builder, "geom_metric", index), index)
        )
    for symbol, expression in plan.temporaries:
        lines.append(
            "      const s_t %s = %s;"
            % (symbol, _sfem_ccode(expression.xreplace(bindings)))
        )
    lines.extend(
        _METRIC_BODY_BY_WRITES_PER_SHAPE[writes_per_shape(form)](
            dim, metric, plan, bindings, n_field_components,
            work_item, use_stream_arrays,
        )
    )
    lines.append("    }")


#: The two tails A constant-P1 specialized body can have.  Both take the same arguments
#: because they are two implementations of one step -- what this body does
#: with its integrand -- and which applies is
#: `plans.form_emission.form_accumulation`.
def _constant_p1_weak_scalar_tail(
    lines, form, weak_form, substitutions, dim, n_field_components,
    work_item, geometry_value, reference_gradients, use_stream_arrays, closing,
):
    """A 0-form: weight the density and add it in."""
    _append_weak_objective_accumulation(
        lines, form, weak_form, substitutions, work_item, geometry_value, closing
    )


def _constant_p1_weak_per_shape_tail(
    lines, form, weak_form, substitutions, dim, n_field_components,
    work_item, geometry_value, reference_gradients, use_stream_arrays, closing,
):
    """A 1- or 2-form: form the loperand and contract it onto the tests."""
    material = _weak_form_material_expression(
        weak_form,
        form.name,
        substitutions,
        tuple(
            sp.symbols("trial_grad%d" % i)
            for i in range(weak_form.n_field_components * dim)
        ),
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
    op = output_assignment(form)
    for shape in range(dim + 1):
        for row in range(n_field_components):
            terms = []
            for col in range(dim):
                factor = _constant_reference_gradient_expr(reference_gradients, shape, col)
                if factor == 0:
                    continue
                terms.append(_scaled_cpp_term(factor, "loperand%d" % (row * dim + col)))
            if terms:
                lines.append(
                    "      %s[%s][%s] %s %s;"
                    % (
                        output_streams,
                        c_sum(c_product(shape, n_field_components), row),
                        work_item,
                        op,
                        _sum_cpp_terms(terms),
                    )
                )
    lines.extend(["      }", "    }"])


_CONSTANT_P1_WEAK_TAIL = {
    FormAccumulation.SCALAR: _constant_p1_weak_scalar_tail,
    FormAccumulation.PER_SHAPE: _constant_p1_weak_per_shape_tail,
}


def _append_constant_p1_sfem_soa_weak_form_lines(
    lines,
    form,
    dim,
    reference_gradients,
    use_stream_arrays,
    source_builder,
):
    n_field_components = form_n_field_components(form, dim)
    work_item = _work_item_index(source_builder)
    weak_form = form.weak_form
    uses_current = form_reads_current(form, default=True)
    uses_direction = form_reads_direction(form, default=form.has_direction)

    def field_value(field, row, shape):
        stream_prefix = "" if use_stream_arrays else "weak_"
        return "%s%s_streams[%s][%s]" % (
            stream_prefix,
            field,
            c_sum(c_product(shape, n_field_components), row),
            work_item,
        )

    def geometry_value(name, component):
        return _work_item_name(source_builder, name, component)

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "gu",
        scalar_temporaries=True,
    )

    # Only reached for a constant-P1 simplex: the caller gates on
    # `constant_p1_simplex_reference_gradients`, so NQ is one and this
    # loop has a single trip.  No strategy test is wanted here -- a
    # higher-order simplex never arrives, so a branch would describe a
    # case that cannot happen.
    lines.append("    { const int q = 0;  // constant-P1 simplex")
    lines.append("      const s_t qw = q_weight[q];")
    lines.extend(_work_item_loop_lines(source_builder, "      "))
    lines.append("      const ptrdiff_t goff = q * geometry_stride + %s;" % work_item)
    for component in range(dim * dim):
        lines.append(
            "      const s_t %s = adj%d[goff];"
            % (geometry_value("adj", component), component)
        )
    lines.append(
        "      const s_t %s = det0[goff];"
        % geometry_value("det", 0)
    )
    for row in range(n_field_components):
        for col in range(dim):
            idx = row * dim + col
            if uses_current:
                lines.append(
                    "      const s_t gu_ref%d = %s;"
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
                    "      const s_t grad_h_ref%d = %s;"
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
        "      const s_t idet = s_t(1) / %s;"
        % geometry_value("det", 0)
    )
    for row in range(weak_form.n_field_components):
        for col in range(dim):
            if uses_current:
                terms = [
                    "gu_ref%d * %s"
                    % (
                        row * dim + k,
                        geometry_value("adj", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "      const s_t gu%d = (%s) * idet;"
                    % (row * dim + col, " + ".join(terms))
                )
            if uses_direction:
                terms = [
                    "grad_h_ref%d * %s"
                    % (
                        row * dim + k,
                        geometry_value("adj", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "      const s_t trial_grad%d = (%s) * idet;"
                    % (row * dim + col, " + ".join(terms))
                )

    _CONSTANT_P1_WEAK_TAIL[form_accumulation(form)](
        lines,
        form,
        weak_form,
        deformation_gradient_substitutions,
        dim,
        n_field_components,
        work_item,
        geometry_value,
        reference_gradients,
        use_stream_arrays,
        ["      }", "    }"],
    )


#: The two tails A simplex weak-form body can have.  Both take the same arguments
#: because they are two implementations of one step -- what this body does
#: with its integrand -- and which applies is
#: `plans.form_emission.form_accumulation`.
def _simplex_weak_scalar_tail(
    lines, form, weak_form, substitutions, dim, n_field_components,
    work_item, geometry_value, reference_gradient, source_builder, use_stream_arrays, closing,
):
    """A 0-form: weight the density and add it in."""
    _append_weak_objective_accumulation(
        lines, form, weak_form, substitutions, work_item, geometry_value, closing
    )


def _simplex_weak_per_shape_tail(
    lines, form, weak_form, substitutions, dim, n_field_components,
    work_item, geometry_value, reference_gradient, source_builder, use_stream_arrays, closing,
):
    """A 1- or 2-form: form the loperand and contract it onto the tests."""
    material = _weak_form_material_expression(
        weak_form,
        form.name,
        substitutions,
        tuple(
            sp.symbols("trial_grad%d" % i)
            for i in range(weak_form.n_field_components * dim)
        ),
    )

    _append_transformed_loperand_lines(
        lines,
        material,
        dim,
        "weak_mat_tmp",
        geometry_value,
        scalar_temporaries=True,
    )
    for component in range(n_field_components * dim):
        lines.append("      loperand%d_values[%s] = loperand%d;" % (component, work_item, component))
    lines.append("      }")
    lines.append("      for (int shape = 0; shape < NS; ++shape) {")
    for row in range(n_field_components):
        terms = [
            "loperand%d_values[%s] * %s" % (row * dim + col, work_item, reference_gradient(col))
            for col in range(dim)
        ]
        op = output_assignment(form)
        output_streams = "out_streams" if use_stream_arrays else "weak_out_streams"
        lines.extend(_work_item_loop_lines(source_builder, "        "))
        lines.append(
            "          %s[%s][%s] %s %s;"
            % (
                output_streams,
                c_sum(c_product("shape", n_field_components), row),
                work_item,
                op,
                " + ".join(terms),
            )
        )
        lines.append("        }")
    lines.extend(["      }", "    }"])


_SIMPLEX_WEAK_TAIL = {
    FormAccumulation.SCALAR: _simplex_weak_scalar_tail,
    FormAccumulation.PER_SHAPE: _simplex_weak_per_shape_tail,
}


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
    metric=None,
):
    # A metric-carrying kernel reads no reference data, so the checks below --
    # which say a weak form takes exactly one `grad_ref` -- describe the general
    # kernel rather than this one, and it is answered before them.
    if constant_p1_gradient_expansion and metric is not None:
        _append_constant_p1_metric_weak_form_lines(
            lines, form, dim, metric, use_stream_arrays, source_builder,
        )
        return

    n_field_components = form_n_field_components(form, dim)
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
    weak_form = form.weak_form
    uses_current = form_reads_current(form, default=True)
    uses_direction = form_reads_direction(form, default=form.has_direction)
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
        return "%s[q * NS + %s]" % (
            _sfem_reference_gradient_vector_name(component),
            shape,
        )

    def field_value(field, row, shape="shape"):
        stream_prefix = "" if use_stream_arrays else "weak_"
        return "%s%s_streams[%s][%s]" % (
            stream_prefix,
            field,
            c_sum(c_product(shape, n_field_components), row),
            work_item,
        )

    def geometry_value(name, component):
        return _work_item_name(source_builder, name, component)

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "gu",
        scalar_temporaries=True,
    )

    lines.append("    for (int q = 0; q < NQ; ++q) {")
    lines.append("      const s_t qw = q_weight[q];")
    for row in range(n_field_components):
        for col in range(dim):
            idx = row * dim + col
            if uses_current:
                lines.append("      s_t gu_ref%d_values[VS];" % idx)
            if uses_direction:
                lines.append("      s_t grad_h_ref%d_values[VS];" % idx)
    lines.extend(
        "      s_t loperand%d_values[VS];" % component
        for component in _LOPERAND_COMPONENTS[form_accumulation(form)](
            n_field_components, dim
        )
    )
    zeroed = []
    for row in range(n_field_components):
        for col in range(dim):
            idx = row * dim + col
            if uses_current:
                zeroed.append("gu_ref%d_values" % idx)
            if uses_direction:
                zeroed.append("grad_h_ref%d_values" % idx)
    lines.extend(
        _zero_lane_block_lines(source_builder, "      ", zeroed, work_item)
    )
    lines.append("      for (int shape = 0; shape < NS; ++shape) {")
    for row in range(n_field_components):
        for col in range(dim):
            idx = row * dim + col
            lines.extend(_work_item_loop_lines(source_builder, "        "))
            if uses_current:
                lines.append(
                    "          gu_ref%d_values[%s] += %s * %s;"
                    % (idx, work_item, field_value("u", row), reference_gradient(col))
                )
            if uses_direction:
                lines.append(
                    "          grad_h_ref%d_values[%s] += %s * %s;"
                    % (idx, work_item, field_value("h", row), reference_gradient(col))
                )
            lines.append("        }")
    lines.append("      }")
    lines.extend(_work_item_loop_lines(source_builder, "      "))
    lines.append("      const ptrdiff_t goff = q * geometry_stride + %s;" % work_item)
    for component in range(dim * dim):
        lines.append(
            "      const s_t %s = adj%d[goff];"
            % (geometry_value("adj", component), component)
        )
    lines.append(
        "      const s_t %s = det0[goff];"
        % geometry_value("det", 0)
    )
    for row in range(n_field_components):
        for col in range(dim):
            idx = row * dim + col
            if uses_current:
                lines.append("      const s_t gu_ref%d = gu_ref%d_values[%s];" % (idx, idx, work_item))
            if uses_direction:
                lines.append("      const s_t grad_h_ref%d = grad_h_ref%d_values[%s];" % (idx, idx, work_item))
    lines.append(
        "    const s_t idet = s_t(1) / %s;"
        % geometry_value("det", 0)
    )
    for row in range(weak_form.n_field_components):
        for col in range(dim):
            if uses_current:
                terms = [
                    "gu_ref%d * %s"
                    % (
                        row * dim + k,
                        geometry_value("adj", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "    const s_t gu%d = (%s) * idet;"
                    % (row * dim + col, " + ".join(terms))
                )
            if uses_direction:
                terms = [
                    "grad_h_ref%d * %s"
                    % (
                        row * dim + k,
                        geometry_value("adj", k * dim + col),
                    )
                    for k in range(dim)
                ]
                lines.append(
                    "    const s_t trial_grad%d = (%s) * idet;"
                    % (row * dim + col, " + ".join(terms))
                )

    _SIMPLEX_WEAK_TAIL[form_accumulation(form)](
        lines,
        form,
        weak_form,
        deformation_gradient_substitutions,
        dim,
        n_field_components,
        work_item,
        geometry_value,
        reference_gradient,
        source_builder,
        use_stream_arrays,
        ["      }", "    }"],
    )


def _append_transformed_loperand_lines(
    lines,
    material,
    dim,
    temporary_prefix,
    geometry_value,
    scalar_temporaries=False,
):
    material_exprs = tuple(material)
    # The flux has one row per field component and one column per direction, so
    # its own length says how many rows there are.  This counted `dim * dim`,
    # which is right only when the field has as many components as the domain
    # has dimensions -- and emitted `material3..8` for a scalar field, which
    # nothing defined.
    n_material = len(material_exprs)
    n_field_components = n_material // dim
    if scalar_temporaries:
        material_names = ["const s_t material%d =" % i for i in range(n_material)]
    else:
        material_names = ["material[%d] =" % i for i in range(n_material)]
        lines.append("    s_t material[%d];" % n_material)
    _append_cse_array_assignments(lines, material_exprs, material_names, temporary_prefix)
    for row in range(n_field_components):
        for col in range(dim):
            terms = [
                "%s * %s"
                % (
                    "material%d" % (row * dim + k)
                    if scalar_temporaries
                    else "material[%d]" % (row * dim + k),
                    geometry_value("adj", col * dim + k),
                )
                for k in range(dim)
            ]
            if scalar_temporaries:
                lines.append(
                    "    const s_t loperand%d = qw * (%s);"
                    % (row * dim + col, " + ".join(terms))
                )
            else:
                lines.append(
                    "    loperand[%d] = qw * (%s);"
                    % (row * dim + col, " + ".join(terms))
                )


def _weak_form_deformation_gradient_substitutions(
    weak_form,
    gradient_name,
    scalar_temporaries=False,
):
    substitutions = {}
    # Loop-invariant: whether the identity belongs in the variable is a
    # property of the weak form, not of the entry being substituted.
    adds_identity = weak_form.is_deformation_gradient
    for row in range(weak_form.n_field_components):
        for col in range(weak_form.dim):
            idx = row * weak_form.dim + col
            if scalar_temporaries:
                value = sp.Symbol("%s%d" % (gradient_name, idx))
            else:
                value = sp.Symbol("%s[%d]" % (gradient_name, idx))
            if row == col and adds_identity:
                value = sp.Integer(1) + value
            substitutions[weak_form.deformation_gradient[idx]] = value
    return substitutions


def _weak_form_deformation_gradient_substitutions_from_symbols(weak_form, gradient):
    substitutions = {}
    adds_identity = weak_form.is_deformation_gradient
    for row in range(weak_form.n_field_components):
        for col in range(weak_form.dim):
            idx = row * weak_form.dim + col
            value = gradient[idx]
            if row == col and adds_identity:
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


def _append_weak_objective_accumulation(
    lines, form, weak_form, substitutions, work_item, geometry_value, closing
):
    """Accumulate the energy density into an objective's scalar output.

    An objective is the one form whose output is a single number rather than a
    stream per shape function, so every weak-form emitter reaches this point
    and stops: there is no test contraction to follow.  Three of them carried
    the block character for character, differing only in how deep the loops
    they close are, which ``closing`` now says.

    ``output_mode`` decides accumulate-into versus assign-over, and that is the
    form's property rather than the emitter's -- it is read here, not decided.
    ``geometry_value`` is passed in because each emitter defines its own: the
    determinant is spelled differently depending on how that kernel reaches its
    geometry, which is the caller's fact and not this block's.
    """
    _append_cse_array_assignments(
        lines,
        [weak_form.energy_density.xreplace(substitutions)],
        [
            "value[%s] %s"
            % (work_item, output_assignment(form))
        ],
        "weak_obj_tmp",
        scale="qw * %s" % geometry_value("det", 0),
    )
    lines.extend(closing)

def _append_cse_array_assignments(lines, expressions, targets, temporary_prefix, scale=None):
    temps, reduced = sp.cse(
        tuple(expressions),
        symbols=sp.numbered_symbols("%s" % temporary_prefix),
    )
    temps = _prune_dead_cse_intermediates(temps, reduced)
    for symbol, expression in temps:
        lines.append("    const s_t %s = %s;" % (symbol, _sfem_ccode(expression)))
    for target, expression in zip(targets, reduced):
        if scale is not None:
            lines.append("    %s %s * (%s);" % (target, scale, _sfem_ccode(expression)))
        else:
            lines.append("    %s %s;" % (target, _sfem_ccode(expression)))


def _form_material_parameter_declarations(form, scalar_type="s_t"):
    return tuple(
        "const %s %s" % (scalar_type, parameter)
        for parameter in form_material_parameter_names(form)
    )


def _append_sfem_soa_statement_lines(lines, expression_graph, output_name):
    output_index = 0
    for statement in expression_graph.evaluation_plan.statements:
        expression = _sfem_ccode(statement.expression)
        if statement.kind == "intermediate":
            lines.append("    const s_t %s = %s;" % (statement.target, expression))
        else:
            lines.append("    %s[%d] = %s;" % (output_name, output_index, expression))
            output_index += 1


def _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes):
    if (
        (
            sfem_tensor_product_hex_uses_cartesian_ordering(quadrature_rule.element_type)
            or sfem_tensor_product_quad_uses_cartesian_ordering(quadrature_rule.element_type)
        )
    ):
        return tuple(range(n_nodes))
    return tensor_product_cartesian_shape_order(dim, n_nodes)


def _use_tensor_product_reference(quadrature_rule, reference_inputs, basis_family=None):
    return (
        is_tensor_product_family(basis_family)
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
            "%sconst int sx = shape %% NS1;" % indent,
            "%sconst int sy = shape / NS1;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int sx = shape %% NS1;" % indent,
            "%sconst int sy = (shape / NS1) %% NS1;" % indent,
            "%sconst int sz = shape / (NS1 * NS1);" % indent,
        )
    raise ValueError("tensor-product shape indices require dim 2 or 3")


def _tensor_product_generic_shape_index_lines(dim, indent):
    if dim == 2:
        return (
            "%sconst int sx = shape %% NS1;" % indent,
            "%sconst int sy = shape / NS1;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int sx = shape %% NS1;" % indent,
            "%sconst int sy = (shape / NS1) %% NS1;" % indent,
            "%sconst int sz = shape / (NS1 * NS1);" % indent,
        )
    raise ValueError("tensor-product shape indices require dim 2 or 3")


def _sfem_reference_gradient_vector_name(component):
    return "grad_ref_%s" % _component_name(component)


def _sfem_reference_gradient_vector_params(dim):
    return tuple(
        "const s_t *const RSTR %s"
        % _sfem_reference_gradient_vector_name(component)
        for component in range(dim)
    )


def _tensor_product_1d_factor(axis, node_axis, derivative_axis):
    qp_name = ("qx", "qy", "qz")[axis]
    table_name = "grad_1d" if axis == derivative_axis else "shape_1d"
    return "%s[%s * NS1 + %d]" % (table_name, qp_name, node_axis)


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
        factors.append("%s[%s * NS1 + %s]" % (table_name, qp_name, node_axis_name))
    return " * ".join(factors)


def _tensor_product_shape_coordinate_arrays(quadrature_rule, indent):
    coords = _tensor_product_node_coords(quadrature_rule)
    axis_names = ("x", "y", "z")[: quadrature_rule.dim]
    lines = []
    for axis, name in enumerate(axis_names):
        lines.append(
            "%sstatic constexpr int SHAPE_%s[NS] = {%s};"
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
            "%sconst int %s_sx = %s %% NS1;" % (indent, prefix, shape_expr),
            "%sconst int %s_sy = %s / NS1;" % (indent, prefix, shape_expr),
        )
    if dim == 3:
        return (
            "%sconst int %s_sx = %s %% NS1;" % (indent, prefix, shape_expr),
            "%sconst int %s_sy = (%s / NS1) %% NS1;"
            % (indent, prefix, shape_expr),
            "%sconst int %s_sz = %s / (NS1 * NS1);"
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
        factors.append("%s[%s * NS1 + %s]" % (table_name, qp_name, node_axis_name))
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
        return "%s%s[q * NS + %s]" % (
            reference_prefix,
            _sfem_reference_gradient_vector_name(component),
            shape_expr,
        )
    if len(reference_inputs) == 1 and reference_inputs[0].name == "grad_ref":
        return "%s%s[(q * NS + %s) * %d + %d]" % (
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
                "    %s[%d] = %s;"
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
    coordinate_streams="bcoordinate_data",
):
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
    stream_array_name = "badj_streams"
    lines = isoparametric_adjugate_stream_array_lines(
        dim_name="ND",
        dim=dim,
        indent="      ",
        stream_array_name=stream_array_name,
        adjugate_streams=tuple(
            "badj%d" % component
            for component in range(dim * dim)
        ),
    )
    for row in range(dim):
        for col in range(dim):
            lines.append("      s_t J%d%d_values[VS];" % (row, col))
    lines.extend(
        _zero_lane_block_lines(
            source_builder,
            "      ",
            [
                "J%d%d_values" % (row, col)
                for row in range(dim)
                for col in range(dim)
            ],
            work_item,
        )
    )
    lines.append("      for (int shape = 0; shape < NS; ++shape) {")
    if use_tensor_product_reference:
        lines.extend(_tensor_product_shape_index_lines(quadrature_rule, "        "))
    for col in range(dim):
        lines.append(
            "        const s_t g%d = %s;"
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
    # One lane loop for the whole Jacobian, not one per component: the loop
    # bound and the pragma are the same for all of them, and a three-
    # dimensional element opened nine `#pragma omp simd` regions in a row for
    # nine accumulations that belong together.  Same stores, one region.
    lines.extend(_work_item_loop_lines(source_builder, "        "))
    lines.extend(
        "          J%d%d_values[%s] += %s[%s][%s] * g%d;"
        % (
            row,
            col,
            work_item,
            coordinate_streams,
            c_sum(c_product("shape", dim), row),
            work_item,
            col,
        )
        for row in range(dim)
        for col in range(dim)
    )
    lines.append("        }")
    lines.append("      }")
    lines.extend(_work_item_loop_lines(source_builder, "      "))
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "        const s_t J%d%d = J%d%d_values[%s];"
                % (row, col, row, col, work_item)
            )
    output_index = "q * VS + %s" % work_item if q_major else work_item
    lines.extend(
        isoparametric_adjugate_call_lines(
            dim=dim,
            indent="        ",
            index=output_index,
            stream_array_name=stream_array_name,
            determinant_stream="bdet0",
        )
    )
    lines.append("      }")
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
        return "%s%s[q * NS + shape]" % (
            reference_prefix,
            _sfem_reference_gradient_vector_name(component),
        )
    if len(reference_inputs) == 1 and reference_inputs[0].name == "grad_ref":
        return "%s%s[(q * NS + shape) * %d + %d]" % (
            reference_prefix,
            reference_inputs[0].name,
            dim,
            component,
        )
    raise ValueError("isoparametric geometry generation requires one grad_ref reference input")


def _append_sfem_soa_output_lines(lines, form, dim, n_nodes, work_item):
    output_count = len(form.expression_graph.evaluation_plan.outputs)
    if output_count == 1:
        op = output_assignment(form)
        lines.append("    value[%s] %s element_vector[0];" % (work_item, op))
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
            op = output_assignment(form)
            lines.append("    %s[%s] %s element_vector[%d];" % (stream, work_item, op, idx))


def _sfem_soa_reference_header_paths(rules):
    """The shared reference headers a source forwards into, once each.

    Paths rather than `#include` lines: `operator_preamble_lines` already takes
    `extra_headers` and spells the directive, so this goes through the hook that
    exists instead of adding a second one.
    """
    seen = []
    for rule in rules:
        for line in reference_include_lines(rule, sfem_mesh_reference_data(rule)):
            path = line.split('"')[1]
            if path not in seen:
                seen.append(path)
    return tuple(seen)


def _sfem_soa_reference_header_files(rules):
    seen = {}
    for rule in rules:
        for entry in reference_header_files(rule, sfem_mesh_reference_data(rule)):
            seen[entry.path] = entry
    return tuple(seen[path] for path in sorted(seen))


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
            extra_headers=(
                (() if hessian_name is None else (hessian_name,))
                + _sfem_soa_reference_header_paths(
                    (affine_quadrature_rule, quadrature_rule)
                )
            ),
        ),
        "",
        "#include <cstdint>",
        "#include <cstdlib>",
    ]
    if getattr(source_builder, "operator_extension", "cpp") == "cpp":
        lines.append('#include "packed_thread_scratch.hpp"')
    lines = [line for line in lines if line != ""]
    lines.append("")
    lines.extend(
        _affine_geometry_stream_helper_lines(
            source_builder,
        )
    )
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
                affine_quadrature_rule,
                local_prefix,
            )
        )
        lines.append("")
        objective_variants = objective_kernel_variants(
            form, source_builder.emit_objective_steps
        )
        if _sfem_soa_has_adjugate_geometry_inputs(array_inputs, dim):
            affine_rule = affine_quadrature_rule
            variants = geometry_variant_plan(
                form.weak_form,
                quadrature_rule,
                # The form-role test lives here rather than as a branch around
                # the emission: only the 2-form assembles, and the plan is where
                # that belongs.
                assembles_matrix=(
                    form.name == "apply"
                    and matrix_format_plan is not None
                    and not matrix_format_plan.is_empty
                ),
            )
            # Iterated, not tested.  A constant-P1 simplex yields no
            # isoparametric mode -- its affine kernel computes the same numbers
            # from a Jacobian that does not vary over the cell -- and a 2D
            # element yields no affine one.  Emission walks what it is given.
            for _affine_mode in variants.affine_modes:
                if "plain" in objective_variants:
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
                if "steps" in objective_variants:
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
            for _isoparametric_mode in variants.isoparametric_modes:
                if "plain" in objective_variants:
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
                if "steps" in objective_variants:
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
            # Assembly has its own axis: a form that publishes no
            # isoparametric matrix-free kernel still assembles a matrix if it
            # was asked to.  This used to sit inside the isoparametric loop,
            # which is why the P1 rule had to keep that mode alive.
            for _assembly_mode in variants.assembly_modes:
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

    return "\n".join(resolve_dead_parameters(resolve_kernel_constants(lines)))


def _is_tet4_linear_elasticity_aos_unit_candidate(form, prefix, dim, n_nodes, quadrature_rule):
    if form.name not in ("gradient", "apply"):
        return False
    if form.weak_form is None:
        return False
    if dim != 3 or n_nodes != 4:
        return False
    if not str(prefix).startswith("linear_elasticity_"):
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
        "idx_t **const RSTR elements",
        "const g_t *const RSTR g_adj_aos",
        "const g_t *const RSTR g_det0",
        "const s_t mu",
        "const s_t lmbda",
        "const ptrdiff_t %s" % stride_name,
        "const s_t *const RSTR %s" % input_components[0],
        "const s_t *const RSTR %s" % input_components[1],
        "const s_t *const RSTR %s" % input_components[2],
        "const ptrdiff_t out_stride",
        "s_t *const RSTR outx",
        "s_t *const RSTR outy",
        "s_t *const RSTR outz",
    )
    wrapper_params = tuple(
        param.replace("g_t", "geom_t") for param in impl_params
    )

    vx, vy, vz = input_components
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename s_t, typename g_t>",
        "static SFEM_INLINE int %s(" % implementation_name,
    ]
    lines.extend(parameter_list_lines(impl_params))
    lines.extend(
        [
            ") {",
            discard_unused("nnodes", indent="  "),
            "",
            *source_builder.parallel_for_lines(),
            "  for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "    const idx_t ev0 = elements[0][element];",
            "    const idx_t ev1 = elements[1][element];",
            "    const idx_t ev2 = elements[2][element];",
            "    const idx_t ev3 = elements[3][element];",
            "",
            "    const s_t ux0 = %s[ev0 * %s];" % (vx, stride_name),
            "    const s_t ux1 = %s[ev1 * %s];" % (vx, stride_name),
            "    const s_t ux2 = %s[ev2 * %s];" % (vx, stride_name),
            "    const s_t ux3 = %s[ev3 * %s];" % (vx, stride_name),
            "    const s_t uy0 = %s[ev0 * %s];" % (vy, stride_name),
            "    const s_t uy1 = %s[ev1 * %s];" % (vy, stride_name),
            "    const s_t uy2 = %s[ev2 * %s];" % (vy, stride_name),
            "    const s_t uy3 = %s[ev3 * %s];" % (vy, stride_name),
            "    const s_t uz0 = %s[ev0 * %s];" % (vz, stride_name),
            "    const s_t uz1 = %s[ev1 * %s];" % (vz, stride_name),
            "    const s_t uz2 = %s[ev2 * %s];" % (vz, stride_name),
            "    const s_t uz3 = %s[ev3 * %s];" % (vz, stride_name),
            "",
            "    const g_t *const RSTR adjugate = g_adj_aos + element * 9;",
        ]
    )
    for component in range(9):
        lines.append(
            "    const s_t a%d = s_t(adjugate[%d]);"
            % (component, component)
        )
    lines.extend(
        [
            "    const s_t inv_det = s_t(1) / s_t(g_det0[element]);",
            "",
            "    const s_t x1 = ux0 - ux1;",
            "    const s_t x2 = ux0 - ux2;",
            "    const s_t x3 = ux0 - ux3;",
            "    const s_t x4 = uy0 - uy1;",
            "    const s_t x5 = uy0 - uy2;",
            "    const s_t x6 = uy0 - uy3;",
            "    const s_t x7 = uz0 - uz1;",
            "    const s_t x8 = uz0 - uz2;",
            "    const s_t x9 = uz0 - uz3;",
            "",
            "    s_t p0 = inv_det * (-a0 * x1 - a3 * x2 - a6 * x3);",
            "    s_t p1 = inv_det * (-a1 * x1 - a4 * x2 - a7 * x3);",
            "    s_t p2 = inv_det * (-a2 * x1 - a5 * x2 - a8 * x3);",
            "    s_t p3 = inv_det * (-a0 * x4 - a3 * x5 - a6 * x6);",
            "    s_t p4 = inv_det * (-a1 * x4 - a4 * x5 - a7 * x6);",
            "    s_t p5 = inv_det * (-a2 * x4 - a5 * x5 - a8 * x6);",
            "    s_t p6 = inv_det * (-a0 * x7 - a3 * x8 - a6 * x9);",
            "    s_t p7 = inv_det * (-a1 * x7 - a4 * x8 - a7 * x9);",
            "    s_t p8 = inv_det * (-a2 * x7 - a5 * x8 - a8 * x9);",
            "",
            "    const s_t m0 = (s_t(1) / s_t(6)) * mu;",
            "    const s_t m1 = m0 * (p1 + p3);",
            "    const s_t m2 = m0 * (p2 + p6);",
            "    const s_t m3 = s_t(2) * mu;",
            "    const s_t m4 = lmbda * (p0 + p4 + p8);",
            "    const s_t m5 = (s_t(1) / s_t(6)) * p0 * m3 + (s_t(1) / s_t(6)) * m4;",
            "    const s_t m6 = m0 * (p5 + p7);",
            "    const s_t m7 = (s_t(1) / s_t(6)) * p4 * m3 + (s_t(1) / s_t(6)) * m4;",
            "    const s_t m8 = (s_t(1) / s_t(6)) * p8 * m3 + (s_t(1) / s_t(6)) * m4;",
            "",
            "    const s_t q0 = a0 * m5 + a1 * m1 + a2 * m2;",
            "    const s_t q1 = a3 * m5 + a4 * m1 + a5 * m2;",
            "    const s_t q2 = a6 * m5 + a7 * m1 + a8 * m2;",
            "    const s_t q3 = a0 * m1 + a1 * m7 + a2 * m6;",
            "    const s_t q4 = a3 * m1 + a4 * m7 + a5 * m6;",
            "    const s_t q5 = a6 * m1 + a7 * m7 + a8 * m6;",
            "    const s_t q6 = a0 * m2 + a1 * m6 + a2 * m8;",
            "    const s_t q7 = a3 * m2 + a4 * m6 + a5 * m8;",
            "    const s_t q8 = a6 * m2 + a7 * m6 + a8 * m8;",
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
                "    #pragma omp atomic update",
                "    %s[%s * out_stride] += %s;" % (out, ev, value),
            ]
        )
    lines.extend(
        [
            "  }",
            "",
            "  return SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )

    wrapper_args = tuple(_cpp_argument_name(param) for param in wrapper_params)
    lines.extend(
        runtime_typed_entry_point_lines(
            function_name,
            wrapper_params,
            lambda scalar_type, _positional: [
                "  return sfem::codegen::%s<%s, geom_t>(%s);"
                % (
                    implementation_name,
                    scalar_type,
                    ", ".join(
                        cast_arguments(wrapper_params, wrapper_args, scalar_type)
                    ),
                ),
            ],
            parameter_lines=parameter_list_lines,
        )
    )
    return lines


def _expanded_simplex_metric_packed_value_body(plan):
    """The pack's stepped 0-form, in closed form.

    The packed counterpart of `_expanded_simplex_metric_value_body`, reading
    the pack-local base state and direction the pack gather already produced.
    There is nothing to scatter: the 0-form writes one value per element per
    step, so the pack's output scratch and its reduction do not apply here.
    """
    from codegen.framework.plans.form_transformations import (
        symmetric_metric_component_count,
    )

    scale = _sfem_ccode(plan.scale)
    lines = ["      for (ptrdiff_t element = e_start; element < e_end; ++element) {"]
    lines.extend(
        "        const uint16_t ev%d = elements[%d][element];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "        const s_t x%d = pk_u_base[ev%d];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "        const s_t h%d = pk_h[ev%d];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    for component in range(symmetric_metric_component_count(plan.dim)):
        value = "s_t(g_met%d[element])" % component
        if scale != "1":
            value = "%s * %s" % (scale, value)
        lines.append("        const s_t fff%d = %s;" % (component, value))
    lines.append("        for (int step = 0; step < nsteps; ++step) {")
    lines.append("          const s_t alpha = steps[step];")
    lines.extend(
        "          const s_t u%d = x%d + alpha * h%d;" % (shape, shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "          const s_t %s = %s;" % (symbol, _sfem_ccode(expression))
        for symbol, expression in plan.kernel.temporaries
    )
    (energy,) = plan.kernel.outputs
    lines.extend(
        [
            "          value[(ptrdiff_t)step * nelements + element] = %s;"
            % _sfem_ccode(energy),
            "        }",
            "      }",
            "",
        ]
    )
    return lines


#: As for the other two, the choice is a table lookup on a plan the planning
#: layer produced, not a condition emission evaluates.
_PACKED_VALUE_BODY_BY_EXPANDED = {
    True: _expanded_simplex_metric_packed_value_body,
    False: lambda plan: None,
}


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
    n_field_components=None,
    metric=None,
    expanded_value_plan=None,
):
    n_field_components = dim if n_field_components is None else n_field_components
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

    # One body, two entry points -- see `_packed_precision_forwarders`.
    for scalar_type in ("s_t",):
        suffix = ""
        public_name = "%s_impl" % public_base
        signature_start = len(lines)
        lines.extend(
            [
                'extern "C" int %s(' % public_name,
                *["    %s," % argument.declaration
                  for argument in PACKED_MESH_CORE_ARGUMENTS],
            ]
        )
        if is_affine:
            for array_input in _packed_affine_geometry_inputs(dim, metric):
                for stream in _soa_array_stream_names(array_input):
                    lines.append(
                        "    const geom_t *const RSTR %s," % abi_geometry_name(stream)
                    )
        else:
            lines.append("    const geom_t *const *const RSTR points,")
        lines.extend(
            "    const %s %s," % (scalar_type, parameter)
            for parameter in material_parameter_names
        )
        lines.append("    const ptrdiff_t u_stride,")
        for d in range(n_field_components):
            lines.append(
                "    const %s *const RSTR u%s,"
                % (scalar_type, _component_name(d))
            )
        lines.append("    const ptrdiff_t h_stride,")
        for d in range(n_field_components):
            lines.append(
                "    const %s *const RSTR h%s,"
                % (scalar_type, _component_name(d))
            )
        lines.extend(
            [
                "    const int nsteps,",
                "    const %s *const RSTR steps," % scalar_type,
                "    %s *const RSTR value" % scalar_type,
                ") {",
                "  using s_t = %s;" % scalar_type,
                kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
                kernel_constant("NQ", n_qp, indent="  "),
                kernel_constant("NS", n_nodes, indent="  "),
                kernel_constant("VS", vector_size, indent="  "),
                discard_unused("nnodes", indent="  "),
                discard_unused("n_shared_nodes", indent="  "),
                "",
            ]
        )
        if not is_affine:
            for d in range(dim):
                lines.append(
                    "  const geom_t *const RSTR %s = points[%d];"
                    % (_component_name(d), d)
                )
            lines.extend(
                _ordered_element_pointer_array_lines(
                    "uint16_t",
                    "coordinate_elements",
                    "elements",
                    stream_shape_order,
                    "  ",
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
                    kernel_constant("NQ1", "%d" % quadrature_rule.tensor_product_n_qp_1d, indent="  "),
                    kernel_constant("NS1", "%d" % quadrature_rule.tensor_product_n_shape_1d, indent="  "),
                ]
            )
        lines.extend(
            [
                "",
                "#pragma omp parallel",
                "  {",
            ]
        )
        if not is_affine:
            lines.append(
                "    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);"
            )
        lines.extend(
            [
                "    s_t *const RSTR pk_u_base = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);",
                "    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);",
                "",
                "#pragma omp for schedule(static)",
                "    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "      const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
                "      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                "      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];",
            ]
        )
        if not is_affine:
            lines.append(
                "      const geom_t *const coordinate_components[ND] = {%s};"
                % ", ".join(_component_name(d) for d in range(dim))
            )
        # Two loops, because the counts differ: the mesh has ND
        # coordinates per node and the field has NC values.  One
        # loop served both for as long as every energy material was a
        # displacement, where the two are equal.
        if not is_affine:
            lines.extend(
                [
                    "      for (int d = 0; d < ND; ++d) {",
                    "        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;",
                    "        const geom_t *const RSTR coordinate_component = coordinate_components[d];",
                    "        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                    "          const idx_t node = owned_nodes_ptr[pack] + k;",
                    "          pk_coordinate[k] = s_t(coordinate_component[node]);",
                    "        }",
                    "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                    "          const idx_t node = ghosts[k];",
                    "          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);",
                    "        }",
                    "      }",
                ]
            )
        lines.extend(
            [
                "      const s_t *const u_components[NC] = {%s};"
                % ", ".join("u%s" % _component_name(d) for d in range(n_field_components)),
                "      const s_t *const h_components[NC] = {%s};"
                % ", ".join("h%s" % _component_name(d) for d in range(n_field_components)),
                "      for (int d = 0; d < NC; ++d) {",
                "        s_t *const RSTR pk_u_base_component = pk_u_base + d * max_nodes_per_pack;",
                "        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;",
                "        const s_t *const RSTR u_component = u_components[d];",
                "        const s_t *const RSTR h_component = h_components[d];",
                "        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                "          const idx_t node = owned_nodes_ptr[pack] + k;",
                "          pk_u_base_component[k] = u_component[node * u_stride];",
                "          pk_h_component[k] = h_component[node * h_stride];",
                "        }",
                "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                "          const idx_t node = ghosts[k];",
                "          pk_u_base_component[n_contiguous + k] = u_component[node * u_stride];",
                "          pk_h_component[n_contiguous + k] = h_component[node * h_stride];",
                "        }",
                "      }",
            ]
        )
        lines.append("")
        element_loop_start = len(lines)
        lines.extend(
            [
                "      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {",
                "        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);",
                "        s_t bu_data[NS * NC][VS];",
                "        s_t bu_base_data[NS * NC][VS];",
                "        s_t bh_data[NS * NC][VS];",
                "        s_t bvalue[VS];",
            ]
        )
        if not is_affine:
            lines.append("        s_t bcoordinate_data[NS * ND][VS];")
            for stream in _soa_array_stream_names(_adjugate_input(dim)):
                lines.append("        s_t b%s[NQ * VS];" % stream)
            lines.extend(
                [
                    "        s_t bdet0[NQ * VS];",
                    "        s_t *badj_streams[ND * ND] = {%s};"
                    % ", ".join("badj%d" % i for i in range(dim * dim)),
                ]
            )
        lines.extend(
            [
                "",
                "        const s_t *bu_streams[NS * NC] = {%s};"
                % ", ".join(
                    "bu_data[%d]" % stream
                    for stream in streams_in_shape_order(
                        tuple(range(n_field_components * n_nodes)),
                        n_field_components,
                        stream_shape_order,
                    )
                ),
                "",
                "        for (int shape = 0; shape < NS; ++shape) {",
                "          const uint16_t *const RSTR element_shape = elements[shape];",
                *([] if is_affine or identity_stream_shape_order else ["          const uint16_t *const RSTR coordinate_shape = coordinate_elements[shape];"]),
            ]
        )
                # The coordinates carry one component per spatial direction and
                # the field one per field component.  Those are the same number
                # for a displacement and different for everything else, which is
                # why gathering both in one loop bounded by ND went
                # unnoticed: a scalar field read `pack_h[2 * max_nodes_per_pack
                # + node]` out of a buffer holding a single component.  Found by
                # AddressSanitizer the first time the gate could reach a packed
                # kernel at all.
        if not is_affine:
            coordinate_node = (
                "packed_node" if identity_stream_shape_order else "coordinate_packed_node"
            )
            lines.extend(
                [
                    "          for (int d = 0; d < ND; ++d) {",
                    *_lane_loop_header_lines(source_builder, "            "),
                    "              const uint16_t %s = %s[evb + lane];"
                    % (
                        coordinate_node,
                        "element_shape" if identity_stream_shape_order else "coordinate_shape",
                    ),
                    "              bcoordinate_data[shape * ND + d][lane] = pk_coordinates[d * max_nodes_per_pack + %s];"
                    % coordinate_node,
                    "            }",
                    "          }",
                ]
            )
        lines.extend(
            [
                "          for (int d = 0; d < NC; ++d) {",
                *_lane_loop_header_lines(source_builder, "            "),
                "              const uint16_t packed_node = element_shape[evb + lane];",
                "              bu_base_data[shape * NC + d][lane] = pk_u_base[d * max_nodes_per_pack + packed_node];",
                "              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];",
                "            }",
                "          }",
                "        }",
                "",
            ]
        )
        if is_affine:
            lines.extend(
                _sfem_soa_affine_geometry_stream_lines(
                    source_builder,
                    _packed_affine_geometry_inputs(dim, metric),
                    "        ",
                    geometry_scalar_type="geom_t",
                )
            )
        elif use_tensor_product_geometry:
            lines.extend(
                tensor_product_gradient_isoparametric_geometry_lines(
                    dim_name="ND",
                    dim=dim,
                    n_shape=n_nodes,
                    n_qp=quadrature_rule.n_qp,
                    local_prefix=local_prefix,
                    coordinate_streams="bcoordinate_data",
                    contiguous_coordinate_streams=True,
                    adjugate_target=lambda component, index: "badj%d[%s]" % (component, index),
                    determinant_target=lambda index: "bdet0[%s]" % index,
                    adjugate_streams=tuple("badj%d" % component for component in range(dim * dim)),
                    determinant_stream="bdet0",
                    shape_name=tensor_shape_name,
                    grad_name=tensor_grad_name,
                )
            )
        else:
            lines.extend(["", *quadrature_scope_lines(quadrature_rule.element_type, "        ")])
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
                coordinate_streams="bcoordinate_data",
            )
            lines.extend("  %s" % line if line else line for line in geometry_lines)
            lines.append("        }")
        call_args = ["ne", "0" if is_affine else "VS"]
        if is_affine:
            call_args.extend(
                _BLOCK_FMT % stream
                for array_input in _packed_affine_geometry_inputs(dim, metric)
                for stream in _soa_array_stream_names(array_input)
            )
        else:
            call_args.extend(
                ["badj%d" % i for i in range(dim * dim)]
                + ["bdet0"]
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
            call_args.extend(
                "%s%s" % (reference_prefix, array_input.name)
                for array_input in reference_inputs
            )
        call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
        call_args.extend(material_parameter_names)
        call_args.extend(("bu_streams", "bvalue"))
        lines.extend(
            [
                "",
                "        for (int step = 0; step < nsteps; ++step) {",
                "          const s_t alpha = steps[step];",
                "          for (int shape = 0; shape < NS; ++shape) {",
                "            for (int d = 0; d < NC; ++d) {",
                *_lane_loop_header_lines(source_builder, "              "),
                "                bu_data[shape * NC + d][lane] = bu_base_data[shape * NC + d][lane] + alpha * bh_data[shape * NC + d][lane];",
                "              }",
                "            }",
                "          }",
                *_lane_loop_header_lines(source_builder, "          "),
                "            bvalue[lane] = s_t(0);",
                "          }",
                "",
                "          %s<s_t, NQ, NS, VS>(%s);"
                % (block_name, ", ".join(call_args)),
                "",
                *_lane_loop_header_lines(source_builder, "          "),
                "            value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];",
                "          }",
                "        }",
                "      }",
            ]
        )
        expanded_body = _PACKED_VALUE_BODY_BY_EXPANDED[expanded_value_plan is not None](
            expanded_value_plan
        )
        if expanded_body is not None:
            # The element's own strategy reaching the packed 0-form, the last
            # of the three matrix-free kernels to get it.
            lines[element_loop_start:] = expanded_body
        lines.extend(
            [
                "    }",
                "  }",
                "  return SFEM_SUCCESS;",
                "}",
                "",
            ]
        )

        brace = lines.index(") {", signature_start)
        signature = lines[signature_start + 1 : brace]
        lines[signature_start] = "static SFEM_INLINE int %s(" % public_name
        lines.insert(signature_start, "template <typename s_t>")
        lines.remove("  using s_t = s_t;")
        lines.extend(
            _packed_precision_forwarders(public_base, signature, public_name)
        )

    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])
    return lines


def _mesh_operator_parameters(
    form,
    dim,
    geometry_mode,
    element_inputs,
    uses_current,
    uses_direction,
):
    """The signature a mesh operator declares, and its wrapper spelling.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.  That
    function is six hundred lines with three axes woven through it --
    geometry mode, contraction, and whether the form writes per shape --
    and nothing in it can be selected from a table while they are
    interleaved.  Taking the parameter list out first is the smallest
    piece that stands on its own: it reads six values and produces two,
    and only `impl_params` and `wrapper_params` were used downstream.

    The `geometry_mode` branch inside is the one worth noticing.  The
    residual emitter asks the same question through
    `plans.geometry_quantities.mesh_geometry_parameters`, but derives the
    names from `dependencies` where this derives them from
    `element_inputs`.  Same concept, two local structures -- which is what
    the two emitters have to stop having before they can share the plan.
    """
    n_field_components = form_n_field_components(form, dim)
    base_params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const RSTR elements",
    ]
    if geometry_mode == "affine":
        base_params.extend(
            "const g_t *const RSTR %s" % abi_geometry_name(stream)
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        base_params.append("const g_t *const *const RSTR points")

    material_params = _form_material_parameter_declarations(form)
    field_params = []
    if uses_current:
        field_params.append("const ptrdiff_t u_stride")
        field_params.extend(
            "const s_t *const RSTR u%s" % _component_name(d)
            for d in range(n_field_components)
        )
    if uses_direction:
        field_params.append("const ptrdiff_t h_stride")
        field_params.extend(
            "const s_t *const RSTR h%s" % _component_name(d)
            for d in range(n_field_components)
        )
    # The output's shape is `plans.form_emission.mesh_output`; only how C
    # declares a pointer is this emitter's business.  A scalar output carries no
    # element stride, which is the plan's `stride` being empty.
    output = mesh_output_shape(form)
    output_params = tuple(
        "const ptrdiff_t %s" % name for name in (output.stride,) if name
    ) + tuple(
        "s_t *const RSTR %s" % name
        for name in _OUTPUT_ABI_BUFFERS[output.per_shape](
            n_field_components, _component_name
        )
    )

    impl_params = (
        tuple(base_params)
        + tuple(material_params)
        + tuple(field_params)
        + tuple(output_params)
    )
    wrapper_params = tuple(
        param.replace("g_t", "geom_t").replace("g_t", "geom_t")
        for param in impl_params
    )
    return impl_params, wrapper_params


def _append_mesh_operator_stream_arrays(
    lines,
    form,
    dim,
    n_nodes,
    use_stream_arrays,
    compact_stream_buffers,
    stream_shape_order,
    uses_current,
    uses_direction,
):
    """The stream pointer arrays a mesh operator hands its block kernel.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.  It reads
    nine values and writes only to `lines`, which is what makes it safe to
    move: nothing it computes is read further down.
    """
    n_field_components = form_n_field_components(form, dim)
    if use_stream_arrays:
        lines.append("")
        if uses_current and compact_stream_buffers:
            lines.extend(
                _ordered_stream_pointer_array_lines(
                    "const s_t *",
                    "bu_streams",
                    "bu_data",
                    dim,
                    stream_shape_order,
                    "    ",
                )
            )
        elif uses_current:
            lines.append(
                "    const s_t *const bu_streams[NS * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        _BLOCK_FMT % stream
                        for stream in streams_in_shape_order(
                            _field_stream_names("u", n_field_components, n_nodes),
                            n_field_components,
                            stream_shape_order,
                        )
                    ),
                )
            )
        if uses_direction:
            if compact_stream_buffers:
                lines.extend(
                    _ordered_stream_pointer_array_lines(
                        "const s_t *",
                        "bh_streams",
                        "bh_data",
                        dim,
                        stream_shape_order,
                        "    ",
                    )
                )
            else:
                lines.append(
                    "    const s_t *const bh_streams[NS * %d] = {%s};"
                    % (
                        dim,
                        ", ".join(
                            _BLOCK_FMT % stream
                            for stream in streams_in_shape_order(
                                _field_stream_names("h", n_field_components, n_nodes),
                                n_field_components,
                                stream_shape_order,
                            )
                        ),
                    )
                )
        lines.extend(
            _BLOCK_OUTPUT_STREAM_ARRAY[form_accumulation(form)](
                form, dim, n_field_components, n_nodes,
                compact_stream_buffers, stream_shape_order,
            )
        )


def _per_shape_weak_output_stream_array(form, dim, n_field_components, n_nodes,
                                        stream_shape_order):
    """The stream array a per-shape weak block writes through."""
    return [
        "  s_t *const weak_out_streams[NS * %d] = {%s};"
        % (
            dim,
            ", ".join(
                streams_in_shape_order(
                    _output_stream_names(form, n_field_components, n_nodes),
                    n_field_components,
                    stream_shape_order,
                )
            ),
        )
    ]


_WEAK_OUTPUT_STREAM_ARRAY = {
    FormAccumulation.SCALAR: (
        lambda form, dim, n_field_components, n_nodes, order: []
    ),
    FormAccumulation.PER_SHAPE: _per_shape_weak_output_stream_array,
}


def _per_shape_block_output_stream_array(form, dim, n_field_components, n_nodes,
                                         compact_stream_buffers, stream_shape_order):
    """The pointer array the block writes its per-shape output through."""
    if compact_stream_buffers:
        return _ordered_stream_pointer_array_lines(
            "s_t *", "bout_streams", "bout_data", dim, stream_shape_order, "    "
        )
    return [
        "    s_t *const bout_streams[NS * %d] = {%s};"
        % (
            dim,
            ", ".join(
                _BLOCK_FMT % stream
                for stream in streams_in_shape_order(
                    _output_stream_names(form, n_field_components, n_nodes),
                    n_field_components,
                    stream_shape_order,
                )
            ),
        )
    ]


#: The output pointer arrays a block declares.  A scalar accumulator has none --
#: an empty sequence, which emits nothing without anyone deciding not to.
_BLOCK_OUTPUT_STREAM_ARRAY = {
    FormAccumulation.SCALAR: (
        lambda form, dim, n_field_components, n_nodes, compact, order: []
    ),
    FormAccumulation.PER_SHAPE: _per_shape_block_output_stream_array,
}


#: How a mesh kernel's block reaches the mesh: the two are different operations
#: rather than one at two sizes.  A scalar accumulator adds its lane into the
#: element's slot; a per-shape output scatters through the connectivity, which
#: is an indirect write and needs the atomics the target supplies.  Which
#: applies is `plans.form_emission.form_accumulation`.
def _scalar_output_scatter(lines, form, dim, n_nodes, n_field_components,
                           compact_stream_buffers, work_item, source_builder):
    """One accumulator per element, written where the element is."""
    lines.extend(_work_item_loop_lines(source_builder, "    "))
    lines.append("      value[evb + %s] += bvalue[%s];" % (work_item, work_item))
    lines.append("    }")


def _per_shape_output_scatter(lines, form, dim, n_nodes, n_field_components,
                              compact_stream_buffers, work_item, source_builder):
    """One contribution per shape function, scattered through the connectivity."""
    if compact_stream_buffers:
        lines.append("    s_t *const out_components[NC] = {%s};" % ", ".join("out%s" % _component_name(d) for d in range(n_field_components)))
        lines.extend(
            [
                "",
                "    for (int shape = 0; shape < NS; ++shape) {",
                "      const idx_t *const RSTR ev_shape = &ev[shape * VS];",
                "      for (int d = 0; d < NC; ++d) {",
                *_scatter_add_lines(
                    source_builder,
                    "out_components[d]",
                    "ev_shape[%s] * out_stride",
                    "bout_data[shape * NC + d][%s]",
                    "        ",
                ),
                "      }",
                "    }",
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
                            "ev[%d * VS + %%s] * out_stride" % shape,
                            "bout%s%d[%%s]" % (component, shape),
                            "    ",
                        )
                    )
                    + [""]
                )


_OUTPUT_SCATTER = {
    FormAccumulation.SCALAR: _scalar_output_scatter,
    FormAccumulation.PER_SHAPE: _per_shape_output_scatter,
}


def _append_mesh_operator_scalar_output(
    lines,
    form,
    dim,
    n_nodes,
    compact_stream_buffers,
    work_item,
    source_builder,
):
    """How a 0-form returns its single accumulated value.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.  The guard
    stays inside because it is what the block is: everything here exists
    only for a form that accumulates a scalar rather than scattering to
    shape functions.  Seven inputs, and nothing it binds is read further
    down.
    """
    n_field_components = form_n_field_components(form, dim)
    _OUTPUT_SCATTER[form_accumulation(form)](
        lines, form, dim, n_nodes, n_field_components,
        compact_stream_buffers, work_item, source_builder,
    )


def _append_mesh_operator_compact_buffers(
    compact_coordinate_buffers,
    compact_stream_buffers,
    dim,
    form,
    geometry_mode,
    lines,
    n_nodes,
    uses_current,
    uses_direction
):
    """The compacted per-element buffers a mesh operator stages into.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.  Its loop
    variables shadow names the function reuses further down, which is why
    the automated escape check flagged them; they do not escape, and the
    byte-identity gate is what settles that rather than the reading.
    """
    if compact_stream_buffers:
        if uses_current:
            lines.append("    s_t bu_data[NS * NC][VS];")
        if uses_direction:
            lines.append("    s_t bh_data[NS * NC][VS];")
        output = mesh_output_shape(form)
        lines.append(
            "    s_t %s%s[VS];"
            % (output.block, "".join("[%s]" % extent for extent in output.extents))
        )
        if compact_coordinate_buffers:
            lines.append("    s_t bcoordinate_data[NS * ND][VS];")
    elif compact_coordinate_buffers:
        lines.append("    s_t bcoordinate_data[NS * ND][VS];")
    elif geometry_mode == "isoparametric":
        for stream in _coordinate_stream_names(dim, n_nodes):
            lines.append("    s_t b%s[VS];" % stream)


def _append_mesh_operator_isoparametric_flux(
    coordinate_streams_name,
    dim,
    element_inputs,
    form,
    geometry_mode,
    lines,
    local_prefix,
    n_nodes,
    quadrature_rule,
    reference_inputs,
    reference_prefix,
    source_builder,
    tensor_grad_name,
    tensor_shape_name,
    use_reference_gradient_vectors,
    use_tensor_product_geometry,
):
    """The isoparametric geometry and deferred-flux staging.

    The largest remaining block, and the one where the two axes this
    function weaves together actually meet: it runs only for an
    isoparametric element whose form defers its flux.  Sixteen inputs is a
    lot, and it is the honest measure of how coupled this was.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.
    """
    if (
        geometry_mode == "isoparametric"
        and form.weak_form is not None
        and use_tensor_product_geometry
    ):
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim_name="ND",
                dim=dim,
                n_shape=n_nodes,
                n_qp=quadrature_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="bcoordinate_data",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "badj%d[%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "bdet0[%s]" % index
                ),
                adjugate_streams=tuple(
                    "badj%d" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="bdet0",
                shape_name=tensor_shape_name,
                grad_name=tensor_grad_name,
            )
        )
    elif geometry_mode == "isoparametric" and form.weak_form is not None:
        lines.extend(["", *quadrature_scope_lines(quadrature_rule.element_type, "    ")])
        if use_tensor_product_geometry:
            lines.extend(tensor_product_q_index_lines(dim, "      "))
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
        lines.append("    }")
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
                "    ",
            )
        )


def _append_mesh_operator_compact_stream_buffers(
    geometry_mode,
    lines,
    omit_reference_basis_inputs,
    prefix,
    quadrature_rule,
    reference_inputs,
    use_reference_gradient_vectors,
    use_tensor_product_reference,
):
    """The compacted stream buffers a mesh operator stages into.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.
    """
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


def _append_mesh_operator_stream_buffer_views(
    compact_stream_buffers,
    dim,
    form,
    lines,
    n_nodes,
    source_builder,
    uses_current,
    uses_direction,
    work_item,
):
    """The per-element views a compacted stream layout is read through.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.
    """
    n_field_components = form_n_field_components(form, dim)
    if compact_stream_buffers:
        if uses_current:
            lines.append("    const s_t *const u_components[NC] = {%s};" % ", ".join("u%s" % _component_name(d) for d in range(n_field_components)))
        if uses_direction:
            lines.append("    const s_t *const h_components[NC] = {%s};" % ", ".join("h%s" % _component_name(d) for d in range(n_field_components)))
        lines.extend(
            [
                "",
            ]
        )
        lines.extend(
            [
                "    for (int shape = 0; shape < NS; ++shape) {",
                "      const idx_t *const RSTR ev_shape = &ev[shape * VS];",
                "      for (int d = 0; d < NC; ++d) {",
                *_work_item_loop_lines(source_builder, "        "),
                "          const idx_t node = ev_shape[%s];" % work_item,
            ]
        )
        if uses_current:
            lines.append("          bu_data[shape * NC + d][%s] = u_components[d][node * u_stride];" % work_item)
        if uses_direction:
            lines.append("          bh_data[shape * NC + d][%s] = h_components[d][node * h_stride];" % work_item)
        lines.extend(["        }", "      }", "    }"])
        # One loop per extent the plan states, innermost the lane loop.  A
        # scalar output has no extents and gets the lane loop alone, which is
        # the same text the two branches here used to spell separately.
        lines.extend(
            _zero_fill_lines(mesh_output_shape(form), source_builder, work_item)
        )
    else:
        lines.extend([""])
        lines.extend(_work_item_loop_lines(source_builder, "    "))
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                if uses_current:
                    lines.append(
                        "      bu%s%d[%s] = u%s[ev[%d * VS + %s] * u_stride];"
                        % (component, shape, work_item, component, shape, work_item)
                    )
                if uses_direction:
                    lines.append(
                        "      bh%s%d[%s] = h%s[ev[%d * VS + %s] * h_stride];"
                        % (component, shape, work_item, component, shape, work_item)
                    )
        for stream in _output_stream_names(form, n_field_components, n_nodes):
            lines.append("      b%s[%s] = s_t(0);" % (stream, work_item))
        lines.append("    }")


def _zero_lane_block_lines(source_builder, indent, targets, work_item):
    """One lane loop that zeroes every target, instead of one loop per target.

    The lane loop is the vectorised inner loop, so opening a fresh one per
    component emitted a `#pragma omp simd` region per component for a set of
    stores that belong in one.  A three-dimensional Jacobian produced nine of
    them in a row, a two-dimensional one four, and the reference-gradient
    accumulators three; across the tree there were 2173 such single-statement
    loops.  The stores are the same, the loop and the pragma are paid once.
    """
    if not targets:
        return []
    lines = list(_work_item_loop_lines(source_builder, indent))
    lines.extend(
        "%s  %s[%s] = s_t(0);" % (indent, target, work_item) for target in targets
    )
    lines.append("%s}" % indent)
    return lines


def _append_mesh_operator_isoparametric_jacobian(
    compact_coordinate_buffers,
    dim,
    geometry_mode,
    identity_stream_shape_order,
    lines,
    n_nodes,
    source_builder,
    work_item,
):
    """The Jacobian an isoparametric element builds from its coordinates.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.
    """
    if geometry_mode == "isoparametric":
        if compact_coordinate_buffers:
            lines.append("    const g_t *const coordinate_components[ND] = {%s};" % ", ".join(_component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "",
                    "    for (int shape = 0; shape < NS; ++shape) {",
                    *(["      const idx_t *const RSTR ev_shape = &ev[shape * VS];"] if identity_stream_shape_order else ["      const idx_t *const RSTR coordinate_element_shape = coordinate_elements[shape];"]),
                    "      for (int d = 0; d < ND; ++d) {",
                    *_work_item_loop_lines(source_builder, "        "),
                    "          bcoordinate_data[shape * ND + d][%s] = coordinate_components[d][%s];"
                    % (
                        work_item,
                        "ev_shape[%s]" % work_item
                        if identity_stream_shape_order
                        else "coordinate_element_shape[evb + %s]" % work_item,
                    ),
                    "        }",
                    "      }",
                    "    }",
                ]
            )
        else:
            lines.extend([""])
            lines.extend(_work_item_loop_lines(source_builder, "    "))
            for shape in range(n_nodes):
                for d in range(dim):
                    stream = "%s%d" % (_component_name(d), shape)
                    lines.append(
                        "      b%s[%s] = %s[ev[%d * VS + %s]];"
                        % (stream, work_item, _component_name(d), shape, work_item)
                    )
            lines.append("    }")


def _append_mesh_operator_packed_entry_points(
    block_name,
    dim,
    effective_vector_size,
    form,
    function_name,
    geometry_mode,
    identity_stream_shape_order,
    lines,
    local_prefix,
    material_parameter_names,
    n_nodes,
    n_qp,
    omit_reference_basis_inputs,
    prefix,
    quadrature_rule,
    reference_inputs,
    source_builder,
    stream_shape_order,
    use_reference_gradient_vectors,
    use_tensor_product_geometry,
    use_tensor_product_reference,
    uses_current,
    uses_direction,
    n_field_components=None,
    metric=None,
    metric_block_name=None,
    expanded_plan=None,
):
    """The packed entry points a 1- or 2-form additionally publishes.

    Lifted out of `_sfem_soa_mesh_operator_function` unchanged.
    """
    n_field_components = dim if n_field_components is None else n_field_components
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
                # The metric block when the contraction factors through it; its
                # reference gradients are folded into the plan, so it takes no
                # reference basis data either.
                block_name=block_name if metric is None else metric_block_name,
                quadrature_rule=quadrature_rule,
                reference_inputs=reference_inputs,
                use_tensor_product_reference=use_tensor_product_reference,
                use_tensor_product_geometry=use_tensor_product_geometry,
                use_reference_gradient_vectors=use_reference_gradient_vectors,
                omit_reference_basis_inputs=(
                    True if metric is not None else omit_reference_basis_inputs
                ),
                stream_shape_order=stream_shape_order,
                identity_stream_shape_order=identity_stream_shape_order,
                vector_size=effective_vector_size,
                geometry_mode=geometry_mode,
                uses_current=uses_current,
                uses_direction=uses_direction,
                material_parameter_names=material_parameter_names,
                source_builder=source_builder,
                n_field_components=n_field_components,
                metric=metric,
                expanded_plan=expanded_plan,
            )
        )


def _expanded_simplex_metric_body(plan, source_builder):
    """The direct element loop a lowest-order simplex calls for, or None.

    The rule in ``plans.evaluation_strategy`` says a lowest-order simplex is
    evaluated in closed form: no quadrature loop, no reference tables, and --
    the part that was missing here -- no blocking.  The general body stages
    every element through ``VS``-wide stack arrays: it gathers the
    connectivity, gathers the field, zeroes the output block, builds two arrays
    of pointers into those blocks, copies six geometry streams, calls through
    the pointer arrays, and scatters back.  On a HEX8 that staging is amortised
    over 8 nodes, 3 components and 8 quadrature points.  On a TET4 carrying one
    scalar per node it is the whole cost: measured on the same operator, the
    same element and the same mesh, the blocked kernel ran at 21 MDOF/s and
    this one runs at 44.

    So this is not a fast path bolted on beside the general one.  It is the
    strategy the element already asked for, applied to the mesh loop as well as
    to the element body -- and its arithmetic is printed from
    ``p1_simplex_metric_apply_plan``, the same plan the residual emitter
    prints, so the two formulations emit one kernel rather than two maintained
    copies of it.

    Reached only when ``plans.affine_element_kernel`` produced a plan; the
    table below routes the other case, so this only spells the result.
    """
    from codegen.framework.plans.form_transformations import (
        symmetric_metric_component_count,
    )

    dim = plan.dim
    scale = _sfem_ccode(plan.scale)
    scatter = _target_scatter_add_lines(source_builder)
    lines = [discard_unused("nnodes", indent="  "), ""]
    lines.extend(_target_parallel_element_loop_lines(source_builder))
    lines.append(
        "  for (ptrdiff_t element = 0; element < nelements; ++element) {"
    )
    lines.extend(
        "    const idx_t ev%d = elements[%d][element];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "    const s_t u%d = %sx[ev%d * %s_stride];"
        % (shape, plan.input_prefix, shape, plan.input_prefix)
        for shape in range(plan.n_shape)
    )
    for component in range(symmetric_metric_component_count(dim)):
        value = "s_t(g_met%d[element])" % component
        if scale != "1":
            value = "%s * %s" % (scale, value)
        lines.append("    const s_t fff%d = %s;" % (component, value))
    lines.extend(
        "    const s_t %s = %s;" % (symbol, _sfem_ccode(expression))
        for symbol, expression in plan.kernel.temporaries
    )
    for shape, expression in enumerate(plan.kernel.outputs):
        lines.append(
            "    const s_t e%d = %s;" % (shape, _sfem_ccode(expression))
        )
        lines.extend(
            scatter("outx[ev%d * out_stride]" % shape, "e%d" % shape, "    ")
        )
    lines.extend(["  }", "", "  return SFEM_SUCCESS;"])
    return lines


#: Whether the element's strategy produced a closed-form mesh loop decides
#: which body is spelled.  A table rather than a conditional, for the reason
#: `_METRIC_BODY_BY_WRITES_PER_SHAPE` above is one: the choice belongs to the
#: plan, and emission looks the answer up rather than making it again.
_MESH_OPERATOR_BODY_BY_EXPANDED = {
    True: lambda plan, source_builder: _expanded_simplex_metric_body(
        plan, source_builder
    ),
    False: lambda plan, source_builder: None,
}


def _target_parallel_element_loop_lines(source_builder):
    """The pragma opening a parallel element loop, from the bound target."""
    target = getattr(source_builder, "target", None)
    if target is None or not hasattr(target, "parallel_element_loop_lines"):
        return []
    return ["  %s" % line for line in target.parallel_element_loop_lines("static")]


def _target_scatter_add_lines(source_builder):
    """A scatter-add spelled by the bound target rather than by a literal."""
    target = getattr(source_builder, "target", None)
    if target is None or not hasattr(target, "scatter_add_lines"):
        return lambda lhs, rhs, indent: ["%s%s += %s;" % (indent, lhs, rhs)]
    return lambda lhs, rhs, indent: list(target.scatter_add_lines(lhs, rhs, indent))



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
    n_field_components = form_n_field_components(form, dim)
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    work_item = _work_item_index(source_builder)
    effective_vector_size = source_builder.effective_vector_size(vector_size)
    if geometry_mode not in ("affine", "isoparametric"):
        raise ValueError("mesh geometry_mode must be 'affine' or 'isoparametric'")
    material_parameter_names = form_material_parameter_names(form)

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
    # The affine variant of a metric-carrying form reads the cached metric.
    # The isoparametric one cannot: it builds its geometry from coordinates and
    # holds an adjugate, so it keeps the general kernel.  This is the one place
    # the two modes take different arguments for the same form.
    metric = geometry_variant_plan(
        form.weak_form,
        quadrature_rule,
        specialized=geometry_mode == "affine" and specialized_prefix is not None,
    ).cached_metric
    # The packed entry points build their own adjugate arguments and have not
    # been converted, so they keep the general block.  Only the kernel whose
    # arguments are built below moves to the metric one.
    general_block_name = block_name
    if metric is not None:
        block_name = "%s_metric_%s_block" % (specialized_prefix, form.name)
        array_inputs = _metric_array_inputs(array_inputs, dim)
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_tensor_product_geometry = (
        geometry_mode == "isoparametric"
        and is_tensor_product_family(geometry_family)
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
    uses_current = form_reads_current(form, default=True)
    uses_direction = form_reads_direction(form, default=form.has_direction)
    stream_shape_order = (
        _tensor_product_stream_shape_order(quadrature_rule, dim, n_nodes)
        if use_tensor_product_reference
        else tuple(range(n_nodes))
    )
    identity_stream_shape_order = tuple(stream_shape_order) == tuple(range(n_nodes))
    coordinate_streams_name = "bcoordinate_data"

    impl_params, wrapper_params = _mesh_operator_parameters(
        form,
        dim,
        geometry_mode,
        element_inputs,
        uses_current,
        uses_direction,
    )

    expanded_plan = expanded_simplex_metric_plan(
        metric,
        dim,
        n_nodes,
        n_qp,
        n_field_components,
        writes_per_shape(form),
        uses_current,
        uses_direction,
    )
    expanded_body = _MESH_OPERATOR_BODY_BY_EXPANDED[expanded_plan is not None](
        expanded_plan, source_builder
    )

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        source_builder.mesh_template_line(geometry_mode, ("int VS",)),
        source_builder.mesh_function_line(implementation_name),
    ]
    lines.extend(parameter_list_lines(impl_params))
    implementation_start = len(lines)
    lines.extend(
        [
            ") {",
            kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
            kernel_constant("NQ", n_qp, indent="  "),
            kernel_constant("NS", n_nodes, indent="  "),
            discard_unused("nnodes", indent="  "),
        ]
    )
    if geometry_mode == "isoparametric":
        for d in range(dim):
            lines.append(
                "  const g_t *const RSTR %s = points[%d];"
                % (_component_name(d), d)
            )
        if compact_coordinate_buffers:
            lines.extend(
                _ordered_element_pointer_array_lines(
                    "idx_t",
                    "coordinate_elements",
                    "elements",
                    stream_shape_order,
                    "  ",
                )
            )
    reference_prefix = "%s_" % geometry_mode
    tensor_shape_name = "%sshape_1d" % reference_prefix
    tensor_grad_name = "%sgrad_1d" % reference_prefix
    tensor_weight_name = "%sq_weight_1d" % reference_prefix
    scalar_weight_name = "%sq_weight" % reference_prefix
    _append_mesh_operator_compact_stream_buffers(
        geometry_mode,
        lines,
        omit_reference_basis_inputs,
        prefix,
        quadrature_rule,
        reference_inputs,
        use_reference_gradient_vectors,
        use_tensor_product_reference,
    )
    if use_tensor_product_reference:
        lines.extend(
            [
                kernel_constant("NQ1", "%d" % quadrature_rule.tensor_product_n_qp_1d, indent="  "),
                kernel_constant("NS1", "%d" % quadrature_rule.tensor_product_n_shape_1d, indent="  "),
            ]
        )
    lines.append("")
    lines.extend(source_builder.parallel_for_lines())
    lines.extend(source_builder.mesh_loop_lines())
    lines.append("    idx_t ev[VS * NS];")

    compact_stream_buffers = use_stream_arrays
    _append_mesh_operator_compact_buffers(
        compact_coordinate_buffers,
        compact_stream_buffers,
        dim,
        form,
        geometry_mode,
        lines,
        n_nodes,
        uses_current,
        uses_direction,
    )
    if geometry_mode == "isoparametric":
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                extent = _BLOCK_BUFFER_EXTENT[form_contraction(form)]
                lines.append("    s_t b%s[%s];" % (stream, extent))
    if not compact_stream_buffers:
        if uses_current:
            for stream in _field_stream_names("u", n_field_components, n_nodes):
                lines.append("    s_t b%s[VS];" % stream)
        if uses_direction:
            for stream in _field_stream_names("h", n_field_components, n_nodes):
                lines.append("    s_t b%s[VS];" % stream)
        for stream in _output_stream_names(form, n_field_components, n_nodes):
            lines.append("    s_t b%s[VS];" % stream)

    lines.extend(
        [
            "",
            "    for (int element_node = 0; element_node < NS; ++element_node) {",
            "      const idx_t *const RSTR element_shape = elements[element_node] + evb;",
            "      idx_t *const RSTR ev_node = &ev[element_node * VS];",
            *_work_item_loop_lines(source_builder, "      "),
            "        ev_node[%s] = element_shape[%s];" % (work_item, work_item),
            "      }",
            "    }",
        ]
    )

    _append_mesh_operator_isoparametric_jacobian(
        compact_coordinate_buffers,
        dim,
        geometry_mode,
        identity_stream_shape_order,
        lines,
        n_nodes,
        source_builder,
        work_item,
    )

    _append_mesh_operator_stream_buffer_views(
        compact_stream_buffers,
        dim,
        form,
        lines,
        n_nodes,
        source_builder,
        uses_current,
        uses_direction,
        work_item,
    )

    _append_mesh_operator_stream_arrays(
        lines,
        form,
        dim,
        n_nodes,
        use_stream_arrays,
        compact_stream_buffers,
        stream_shape_order,
        uses_current,
        uses_direction,
    )

    # A pointwise form is contracted at each quadrature point, so its block call
    # sits inside a loop this opens, indents and closes below.  A deferred-flux
    # form is handed the whole element and opens nothing.
    lines.extend(
        _MESH_QUADRATURE_SCOPE[form_contraction(form)](
            quadrature_rule, dim, use_tensor_product_reference, tensor_weight_name
        )
    )

    _append_mesh_operator_isoparametric_flux(
        coordinate_streams_name,
        dim,
        element_inputs,
        form,
        geometry_mode,
        lines,
        local_prefix,
        n_nodes,
        quadrature_rule,
        reference_inputs,
        reference_prefix,
        source_builder,
        tensor_grad_name,
        tensor_shape_name,
        use_reference_gradient_vectors,
        use_tensor_product_geometry,
    )

    call_args = ["ne"]
    call_args.append(
        _MESH_GEOMETRY_CALL_ARGUMENT[form_contraction(form)](geometry_mode)
    )
    if geometry_mode == "affine":
        call_args.extend(
            _BLOCK_FMT % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        call_args.extend(
            _BLOCK_FMT % stream
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
    call_args.append(
        _MESH_WEIGHT_CALL_ARGUMENT[form_contraction(form)](
            use_tensor_product_reference, tensor_weight_name, scalar_weight_name
        )
    )
    call_args.extend(material_parameter_names)
    if use_stream_arrays:
        if uses_current:
            call_args.append("bu_streams")
        if uses_direction:
            call_args.append("bh_streams")
        call_args.append(
            _OUTPUT_CALL_ARGUMENT[mesh_output_shape(form).per_shape]
        )
    else:
        if uses_current:
            call_args.extend(_BLOCK_FMT % stream for stream in _field_stream_names("u", n_field_components, n_nodes))
        if uses_direction:
            call_args.extend(_BLOCK_FMT % stream for stream in _field_stream_names("h", n_field_components, n_nodes))
        call_args.extend(_BLOCK_FMT % stream for stream in _output_stream_names(form, n_field_components, n_nodes))
    call_indent = _MESH_BLOCK_CALL_INDENT[form_contraction(form)]
    lines.extend(
        [
            "",
            "%s%s<s_t, NQ, NS, VS>(%s);"
            % (call_indent, block_name, ", ".join(call_args)),
        ]
    )
    lines.extend(_MESH_QUADRATURE_SCOPE_CLOSE[form_contraction(form)])
    lines.append("")

    _append_mesh_operator_scalar_output(
        lines,
        form,
        dim,
        n_nodes,
        compact_stream_buffers,
        work_item,
        source_builder,
    )

    lines.extend(
        [
            "  }",
            "",
            *source_builder.success_return_lines(),
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )

    if expanded_body is not None:
        # The element asked for the expanded strategy, and this is where it
        # reaches the mesh loop as well as the element body.  The general body
        # above is built and then dropped rather than skipped: it is two
        # hundred lines of straight-line emission that would have to be
        # extracted whole to be made conditional, and doing that as part of
        # this change would put an untested restructuring underneath a measured
        # one.  The cost is generator time, not emitted code.
        del lines[implementation_start:]
        lines.extend(
            [
                ") {",
                *expanded_body,
                "}",
                "",
                "} // namespace codegen",
                "} // namespace sfem",
                "",
            ]
        )


    wrapper_args = tuple(_cpp_argument_name(param) for param in wrapper_params)
    lines.extend(
        runtime_typed_entry_point_lines(
            function_name,
            wrapper_params,
            lambda scalar_type, _positional: source_builder.wrapper_call_lines(
                implementation_name,
                scalar_type,
                ", geom_t, %d" % effective_vector_size,
                cast_arguments(wrapper_params, wrapper_args, scalar_type),
            ),
            parameter_lines=parameter_list_lines,
        )
    )
    # Iterated, not tested: the plan says which packed layouts this element
    # publishes, and an element that publishes none simply does not enter the
    # loop.  A branch here would be emission choosing what to emit.
    for _packed_layout in geometry_variant_plan(
        form.weak_form, quadrature_rule
    ).packed_mesh_layouts:
        _append_mesh_operator_packed_entry_points(
            general_block_name,
            dim,
            effective_vector_size,
            form,
            function_name,
            geometry_mode,
            identity_stream_shape_order,
            lines,
            local_prefix,
            material_parameter_names,
            n_nodes,
            n_qp,
            omit_reference_basis_inputs,
            prefix,
            quadrature_rule,
            reference_inputs,
            source_builder,
            stream_shape_order,
            use_reference_gradient_vectors,
            use_tensor_product_geometry,
            use_tensor_product_reference,
            uses_current,
            uses_direction,
            n_field_components=n_field_components,
            metric=metric,
            metric_block_name=block_name,
            expanded_plan=expanded_plan,
        )
    return lines




def _expanded_simplex_metric_packed_body(plan, uses_current, uses_direction):
    """The pack's element loop, evaluated in closed form.

    The pack gather and the pack scatter are the packed traversal's reason for
    existing and are untouched; what this replaces is the block staging between
    them.  The general body walks the pack in ``VS`` slices, declaring
    two ``[NS * NC][VS]`` arrays and two arrays of
    pointers into them per slice, gathering, zeroing, calling through the
    pointers and scattering back.  On a TET4 carrying one scalar per node that
    staging is the whole cost, exactly as it was on the unpacked path.

    Writing straight into ``pk_out`` needs no atomic: the pack's scratch is
    thread-private, which is what the packed traversal buys.  The reduction to
    the global vector happens once per pack afterwards, and that code is shared.

    Returns ``None`` when the shape does not call for it; the table below routes
    the other case.
    """
    from codegen.framework.plans.form_transformations import (
        symmetric_metric_component_count,
    )

    scale = _sfem_ccode(plan.scale)
    prefix = plan.input_prefix
    if (prefix == "u") != bool(uses_current) or (prefix == "h") != bool(uses_direction):
        return None
    lines = [
        "      for (ptrdiff_t element = e_start; element < e_end; ++element) {",
    ]
    lines.extend(
        "        const uint16_t ev%d = elements[%d][element];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "        const s_t u%d = pk_%s[ev%d];" % (shape, prefix, shape)
        for shape in range(plan.n_shape)
    )
    for component in range(symmetric_metric_component_count(plan.dim)):
        value = "s_t(g_met%d[element])" % component
        if scale != "1":
            value = "%s * %s" % (scale, value)
        lines.append("        const s_t fff%d = %s;" % (component, value))
    lines.extend(
        "        const s_t %s = %s;" % (symbol, _sfem_ccode(expression))
        for symbol, expression in plan.kernel.temporaries
    )
    for shape, expression in enumerate(plan.kernel.outputs):
        lines.append(
            "        const s_t e%d = %s;" % (shape, _sfem_ccode(expression))
        )
        lines.append("        pk_out[ev%d] += e%d;" % (shape, shape))
    lines.extend(["      }", ""])
    return lines


#: Whether the element's strategy produced a closed-form pack loop decides
#: which body is spelled, the same way `_MESH_OPERATOR_BODY_BY_EXPANDED` does
#: for the unpacked traversal.
_PACKED_BODY_BY_EXPANDED = {
    True: lambda plan, uses_current, uses_direction: (
        _expanded_simplex_metric_packed_body(plan, uses_current, uses_direction)
    ),
    False: lambda plan, uses_current, uses_direction: None,
}

def _packed_affine_geometry_inputs(dim, metric):
    """The geometry a packed affine kernel reads, in ABI order.

    The same question `_sfem_soa_mesh_operator_function` asks of the unpacked
    kernel, asked here so that the two answer alike: a form whose contraction
    factors through the cached metric reads the symmetric metric, and
    everything else reads the adjugate and the determinant.  The packed entry
    points were left on the adjugate when the unpacked ones moved, so a packed
    laplace ran the general contraction where the unpacked one ran the compact
    form -- 18 MDOF/s against 44 on the same element.
    """
    if metric is not None:
        return _metric_array_inputs((), dim)
    return (
        _adjugate_input(dim),
        sfem_soa_reference_input("det", 1, 1, 1),
    )



def _packed_precision_forwarders(public_base, signature, impl_name):
    """Two `extern "C"` entry points over one template.

    The packed emission path wrote the whole kernel out once per precision, with
    `using s_t = double;` or `using s_t = float;` at the top and the identical
    text below it.  Every other path in this emitter already emits one
    `template <typename s_t>` and forwards to it; these did not, so the same
    arithmetic existed twice and could drift.

    `signature` is the parameter list as emitted for the template -- it already
    speaks `s_t` -- so each entry point is that list with the concrete type
    substituted, and a call passing the parameters straight through.
    """
    import re as _re

    names = []
    for line in signature:
        match = _re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*,?\s*$", line)
        if not match:
            raise ValueError("could not read a parameter name from %r" % line)
        names.append(match.group(1))

    params = [line.strip().rstrip(",") for line in signature]
    return runtime_typed_entry_point_lines(
        public_base,
        params,
        lambda scalar_type, arguments: [
            "  return %s<%s>(%s);"
            % (impl_name, scalar_type, ", ".join(arguments)),
        ],
    )


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
    n_field_components=None,
    metric=None,
    expanded_plan=None,
):
    n_field_components = dim if n_field_components is None else n_field_components
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
    coordinate_streams_name = "bcoordinate_data"
    lines = ["namespace sfem {", "namespace codegen {", ""]

    for pass_mode in ("one_pass", "two_pass"):
        two_pass = pass_mode == "two_pass"
        packed_token = "_%s_packed_two_pass_" % form_name if two_pass else "_%s_packed_" % form_name
        public_base = function_name.replace("_%s_" % form_name, packed_token)
        # One body, two entry points.  The body below already speaks `s_t`; it
        # was emitted once per precision only because the alias was bound to a
        # concrete type at the top instead of being a template parameter.
        for scalar_type in ("s_t",):
            suffix = ""
            public_name = "%s_impl" % public_base
            signature_start = len(lines)
            lines.extend(
                [
                    'extern "C" int %s(' % public_name,
                    *["    %s," % argument.declaration
                      for argument in PACKED_MESH_CORE_ARGUMENTS],
                ]
            )
            if two_pass:
                lines.extend(
                    [
                        "    %s," % argument.declaration
                        for argument in packed_mesh_ghost_reduce_arguments(scalar_type)
                    ]
                )
            if is_affine:
                for array_input in _packed_affine_geometry_inputs(dim, metric):
                    for stream in _soa_array_stream_names(array_input):
                        lines.append(
                            "    const geom_t *const RSTR %s," % abi_geometry_name(stream)
                        )
            else:
                lines.append("    const geom_t *const *const RSTR points,")
            lines.extend(
                "    const %s %s," % (scalar_type, parameter)
                for parameter in material_parameter_names
            )
            if uses_current:
                lines.append("    const ptrdiff_t u_stride,")
                for d in range(n_field_components):
                    lines.append(
                        "    const %s *const RSTR u%s,"
                        % (scalar_type, _component_name(d))
                    )
            if uses_direction:
                lines.append("    const ptrdiff_t h_stride,")
                for d in range(n_field_components):
                    lines.append(
                        "    const %s *const RSTR h%s,"
                        % (scalar_type, _component_name(d))
                    )
            lines.append("    const ptrdiff_t out_stride,")
            for d in range(n_field_components):
                comma = "," if d + 1 < n_field_components else ""
                lines.append(
                    "    %s *const RSTR out%s%s"
                    % (scalar_type, _component_name(d), comma)
                )
            lines.extend(
                [
                    ") {",
                    "  using s_t = %s;" % scalar_type,
                    kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
                    kernel_constant("NQ", n_qp, indent="  "),
                    kernel_constant("NS", n_nodes, indent="  "),
                    kernel_constant("VS", vector_size, indent="  "),
                    discard_unused("nnodes", indent="  "),
                    "",
                ]
            )
            if not is_affine:
                for d in range(dim):
                    lines.append(
                        "  const geom_t *const RSTR %s = points[%d];"
                        % (_component_name(d), d)
                    )
                lines.extend(
                    _ordered_element_pointer_array_lines(
                        "uint16_t",
                        "coordinate_elements",
                        "elements",
                        stream_shape_order,
                        "  ",
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
                        kernel_constant("NQ1", "%d" % quadrature_rule.tensor_product_n_qp_1d, indent="  "),
                        kernel_constant("NS1", "%d" % quadrature_rule.tensor_product_n_shape_1d, indent="  "),
                    ]
                )
            lines.extend(
                [
                    "",
                    "#pragma omp parallel",
                    "  {",
                ]
            )
            if not is_affine:
                lines.append(
                    "    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);"
                )
            if uses_current:
                lines.append(
                    "    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);"
                )
            if uses_direction:
                lines.append(
                    "    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);"
                )
            lines.extend(
                [
                    "    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);",
                    "",
                    "#pragma omp for schedule(static)",
                    "    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                    "      const ptrdiff_t e_start = pack * n_elements_per_pack;",
                    "      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                    "      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
                ]
            )
            if two_pass:
                lines.extend(
                    [
                        discard_unused("n_shared_nodes", indent="      "),
                        "      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                        "      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;",
                        "      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];",
                        "      const ptrdiff_t ghost_off = ghost_ptr[pack];",
                    ]
                )
            else:
                lines.extend(
                    [
                        "      const ptrdiff_t n_shared = n_shared_nodes[pack];",
                        "      const ptrdiff_t n_not_shared = n_contiguous - n_shared;",
                        "      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                        "      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;",
                        "      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];",
                    ]
                )
            if not is_affine:
                lines.append(
                    "      const geom_t *const coordinate_components[ND] = {%s};"
                    % ", ".join(_component_name(d) for d in range(dim))
                )
            if uses_current:
                lines.append(
                    "      const s_t *const u_components[NC] = {%s};"
                    % ", ".join("u%s" % _component_name(d) for d in range(n_field_components))
                )
            if uses_direction:
                lines.append(
                    "      const s_t *const h_components[NC] = {%s};"
                    % ", ".join("h%s" % _component_name(d) for d in range(n_field_components))
                )
            lines.extend(
                [
                    "      s_t *const out_components[NC] = {%s};"
                    % ", ".join("out%s" % _component_name(d) for d in range(n_field_components)),
                    "      for (int d = 0; d < NC; ++d) {",
                    "        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;",
                ]
            )
            if not is_affine:
                lines.append(
                    "        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;"
                )
            if uses_current:
                lines.append(
                    "        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;"
                )
            if uses_direction:
                lines.append(
                    "        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;"
                )
            if not is_affine:
                lines.append(
                    "        const geom_t *const RSTR coordinate_component = coordinate_components[d];"
                )
            if uses_current:
                lines.append(
                    "        const s_t *const RSTR u_component = u_components[d];"
                )
            if uses_direction:
                lines.append("        const s_t *const RSTR h_component = h_components[d];")
            lines.extend(
                [
                    "        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {",
                    "          pk_component_out[k] = s_t(0);",
                    "        }",
                    "        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                    "          const idx_t node = owned_nodes_ptr[pack] + k;",
                ]
            )
            if not is_affine:
                lines.append("          pk_coordinate[k] = s_t(coordinate_component[node]);")
            if uses_current:
                lines.append("          pk_u_component[k] = u_component[node * u_stride];")
            if uses_direction:
                lines.append("          pk_h_component[k] = h_component[node * h_stride];")
            lines.extend(
                [
                    "        }",
                    "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                    "          const idx_t node = ghosts[k];",
                ]
            )
            if not is_affine:
                lines.append(
                    "          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);"
                )
            if uses_current:
                lines.append(
                    "          pk_u_component[n_contiguous + k] = u_component[node * u_stride];"
                )
            if uses_direction:
                lines.append(
                    "          pk_h_component[n_contiguous + k] = h_component[node * h_stride];"
                )
            lines.extend(
                [
                    "        }",
                    "      }",
                    "",
                ]
            )
            element_loop_start = len(lines)
            lines.extend(
                [
                    "      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {",
                    "        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);",
                ]
            )
            if uses_current:
                lines.append("        s_t bu_data[NS * NC][VS];")
            if uses_direction:
                lines.append("        s_t bh_data[NS * NC][VS];")
            lines.extend(
                [
                    "        s_t bout_data[NS * NC][VS];",
                ]
            )
            if not is_affine:
                lines.append("        s_t bcoordinate_data[NS * ND][VS];")
                for stream in _soa_array_stream_names(_adjugate_input(dim)):
                    lines.append("        s_t b%s[NQ * VS];" % stream)
                lines.extend(
                    [
                        "        s_t bdet0[NQ * VS];",
                        "        s_t *badj_streams[ND * ND] = {%s};"
                        % ", ".join("badj%d" % i for i in range(dim * dim)),
                    ]
                )
            if uses_current:
                lines.extend(
                    _ordered_stream_pointer_array_lines(
                            "const s_t *",
                            "bu_streams",
                            "bu_data",
                            dim,
                            stream_shape_order,
                            "        ",
                        )
                )
            if uses_direction:
                lines.extend(
                    _ordered_stream_pointer_array_lines(
                            "const s_t *",
                            "bh_streams",
                            "bh_data",
                            dim,
                            stream_shape_order,
                            "        ",
                        )
                )
            lines.extend(
                [
                    *_ordered_stream_pointer_array_lines(
                        "s_t *",
                        "bout_streams",
                        "bout_data",
                        dim,
                        stream_shape_order,
                        "        ",
                    ),
                    "",
                    "        for (int shape = 0; shape < NS; ++shape) {",
                    "          const uint16_t *const RSTR element_shape = elements[shape];",
                    *([] if identity_stream_shape_order else ["          const uint16_t *const RSTR coordinate_shape = coordinate_elements[shape];"]),
                ]
            )
            # Two loops, for the reason spelled at the other packed gather: the
            # coordinates are indexed by spatial direction and the field by
            # field component, and only a displacement makes those the same.
            if not is_affine:
                coordinate_node = (
                    "packed_node"
                    if identity_stream_shape_order
                    else "coordinate_packed_node"
                )
                lines.extend(
                    [
                        "          for (int d = 0; d < ND; ++d) {",
                        *_lane_loop_header_lines(source_builder, "            "),
                        "              const uint16_t %s = %s[evb + lane];"
                        % (
                            coordinate_node,
                            "element_shape"
                            if identity_stream_shape_order
                            else "coordinate_shape",
                        ),
                        "              bcoordinate_data[shape * ND + d][lane] = pk_coordinates[d * max_nodes_per_pack + %s];"
                        % coordinate_node,
                        "            }",
                        "          }",
                    ]
                )
            lines.extend(
                [
                    "          for (int d = 0; d < NC; ++d) {",
                    *_lane_loop_header_lines(source_builder, "            "),
                    "              const uint16_t packed_node = element_shape[evb + lane];",
                ]
            )
            if uses_current:
                lines.append(
                    "              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];"
                )
            if uses_direction:
                lines.append(
                    "              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];"
                )
            lines.extend(
                [
                    "              bout_data[shape * NC + d][lane] = s_t(0);",
                    "            }",
                    "          }",
                    "        }",
                    "",
                ]
            )
            if is_affine:
                lines.extend(
                    _sfem_soa_affine_geometry_stream_lines(
                        source_builder,
                        _packed_affine_geometry_inputs(dim, metric),
                        "        ",
                        geometry_scalar_type="geom_t",
                    )
                )
            elif use_tensor_product_geometry:
                lines.extend(
                    tensor_product_gradient_isoparametric_geometry_lines(
                        dim_name="ND",
                        dim=dim,
                        n_shape=n_nodes,
                        n_qp=quadrature_rule.n_qp,
                        local_prefix=local_prefix,
                        coordinate_streams="bcoordinate_data",
                        contiguous_coordinate_streams=True,
                        adjugate_target=lambda component, index: (
                            "badj%d[%s]" % (component, index)
                        ),
                        determinant_target=lambda index: (
                            "bdet0[%s]" % index
                        ),
                        adjugate_streams=tuple(
                            "badj%d" % component
                            for component in range(dim * dim)
                        ),
                        determinant_stream="bdet0",
                        shape_name=tensor_shape_name,
                        grad_name=tensor_grad_name,
                    )
                )
            else:
                lines.extend(["", *quadrature_scope_lines(quadrature_rule.element_type, "        ")])
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
                lines.extend("  %s" % line if line else line for line in geometry_lines)
                lines.append("        }")

            call_args = ["ne", "0" if is_affine else "VS"]
            if is_affine:
                call_args.extend(
                    _BLOCK_FMT % stream
                    for array_input in _packed_affine_geometry_inputs(dim, metric)
                    for stream in _soa_array_stream_names(array_input)
                )
            else:
                call_args.extend(
                    ["badj%d" % i for i in range(dim * dim)]
                    + ["bdet0"]
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
                call_args.extend(
                    "%s%s" % (reference_prefix, array_input.name)
                    for array_input in reference_inputs
                )
            call_args.append(tensor_weight_name if use_tensor_product_reference else scalar_weight_name)
            call_args.extend(material_parameter_names)
            if uses_current:
                call_args.append("bu_streams")
            if uses_direction:
                call_args.append("bh_streams")
            call_args.append("bout_streams")
            lines.extend(
                [
                    "",
                    "        %s<s_t, NQ, NS, VS>(%s);"
                    % (block_name, ", ".join(call_args)),
                    "",
                    "        for (int shape = 0; shape < NS; ++shape) {",
                    "          const uint16_t *const RSTR element_shape = elements[shape];",
                    "          for (int d = 0; d < NC; ++d) {",
                    "            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;",
                    "            for (int lane = 0; lane < ne; ++lane) {",
                    "              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];",
                    "            }",
                    "          }",
                    "        }",
                    "      }",
                    "",
                ]
            )
            expanded_lines = _PACKED_BODY_BY_EXPANDED[expanded_plan is not None](
                expanded_plan, uses_current, uses_direction
            )
            if expanded_lines is not None:
                # The element's own strategy, reaching the pack loop.  Built and
                # dropped rather than skipped, for the reason the unpacked
                # traversal states: the general body is straight-line emission
                # that would have to be extracted whole to be made conditional.
                lines[element_loop_start:] = expanded_lines
            if two_pass:
                lines.extend(
                    [
                        "      for (int d = 0; d < NC; ++d) {",
                        "        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;",
                        "        s_t *const RSTR global_out = out_components[d];",
                        "        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;",
                        "        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                        "          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];",
                        "          pk_component_out[k] = s_t(0);",
                        "        }",
                        "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                        "          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];",
                        "          pk_component_out[n_contiguous + k] = s_t(0);",
                        "        }",
                        "      }",
                        "    }",
                    ]
                )
            else:
                lines.extend(
                    [
                        "      for (int d = 0; d < NC; ++d) {",
                        "        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;",
                        "        s_t *const RSTR global_out = out_components[d];",
                        "        for (ptrdiff_t k = 0; k < n_not_shared; ++k) {",
                        "          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];",
                        "          pk_component_out[k] = s_t(0);",
                        "        }",
                        "        for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {",
                        *source_builder.atomic_update_lines(),
                        "          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];",
                        "          pk_component_out[k] = s_t(0);",
                        "        }",
                        "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                        *source_builder.atomic_update_lines(),
                        "          global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];",
                        "          pk_component_out[n_contiguous + k] = s_t(0);",
                        "        }",
                        "      }",
                        "    }",
                    ]
                )
            if two_pass:
                lines.extend(
                    [
                        "  }",
                        "",
                        "  s_t *const out_components[NC] = {%s};"
                        % ", ".join("out%s" % _component_name(d) for d in range(n_field_components)),
                        *source_builder.parallel_for_lines(),
                        "  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {",
                        "    const idx_t dest = ghost_reduce_dest[row];",
                        "    const ptrdiff_t begin = ghost_reduce_ptr[row];",
                        "    const ptrdiff_t end = ghost_reduce_ptr[row + 1];",
                        "    for (int d = 0; d < NC; ++d) {",
                        "      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;",
                        "      s_t sum = s_t(0);",
                        "      for (ptrdiff_t j = begin; j < end; ++j) {",
                        "        sum += ghost_component[ghost_reduce_idx[j]];",
                        "      }",
                        "      out_components[d][dest * out_stride] += sum;",
                        "    }",
                        "  }",
                        "  return SFEM_SUCCESS;",
                        "}",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "  }",
                        "  return SFEM_SUCCESS;",
                        "}",
                        "",
                    ]
                )

            brace = lines.index(") {", signature_start)
            signature = lines[signature_start + 1 : brace]
            lines[signature_start] = "static SFEM_INLINE int %s(" % public_name
            lines.insert(signature_start, "template <typename s_t>")
            # The alias the body used to open with is the template parameter now.
            lines.remove("  using s_t = s_t;")
            lines.extend(
                _packed_precision_forwarders(public_base, signature, public_name)
            )

    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])
    return lines


def _expanded_simplex_metric_value_body(plan, source_builder):
    """The stepped 0-form of a lowest-order simplex, in closed form.

    The third member of the matrix-free triple, and the one that was left on
    the old path: while the gradient and the action moved to the cached metric
    and the closed-form loop, `objective_steps` still read nine adjugate
    components and a determinant and staged every element through
    ``VS``-wide block arrays.  Measured on TET4 at --refine 24 that is
    16.4 MDOF/s against 41.4 for the two kernels beside it.

    The element energy is `g^T (scale * FFF) g / 2` with `g` the reference
    gradient, which `p1_simplex_metric_value_plan` derives.  The geometry and
    the base state are read once per element and the step loop is inside, so
    `nsteps` evaluations share one gather -- the arrangement the blocked body
    could not express, because its staging was per element block and the step
    loop had to wrap it.

    Reached only when `plans.affine_element_kernel` produced a plan; the table
    routes the other case.
    """
    from codegen.framework.plans.form_transformations import (
        symmetric_metric_component_count,
    )

    dim = plan.dim
    scale = _sfem_ccode(plan.scale)
    lines = [
        discard_unused("nnodes", indent="  "),
        "",
    ]
    lines.extend(_target_parallel_element_loop_lines(source_builder))
    lines.append(
        "  for (ptrdiff_t element = 0; element < nelements; ++element) {"
    )
    lines.extend(
        "    const idx_t ev%d = elements[%d][element];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "    const s_t x%d = ux[ev%d * u_stride];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "    const s_t h%d = hx[ev%d * h_stride];" % (shape, shape)
        for shape in range(plan.n_shape)
    )
    for component in range(symmetric_metric_component_count(dim)):
        value = "s_t(g_met%d[element])" % component
        if scale != "1":
            value = "%s * %s" % (scale, value)
        lines.append("    const s_t fff%d = %s;" % (component, value))
    lines.append("    for (int step = 0; step < nsteps; ++step) {")
    lines.append("      const s_t alpha = steps[step];")
    lines.extend(
        "      const s_t u%d = x%d + alpha * h%d;" % (shape, shape, shape)
        for shape in range(plan.n_shape)
    )
    lines.extend(
        "      const s_t %s = %s;" % (symbol, _sfem_ccode(expression))
        for symbol, expression in plan.kernel.temporaries
    )
    (energy,) = plan.kernel.outputs
    lines.extend(
        [
            "      value[(ptrdiff_t)step * nelements + element] = %s;"
            % _sfem_ccode(energy),
            "    }",
            "  }",
            "",
            "  return SFEM_SUCCESS;",
        ]
    )
    return lines


#: Which element inputs the 0-form reads follows from the same answer: the
#: closed-form kernel takes the symmetric metric and no reference data, the
#: general one takes the adjugate, the determinant and the basis it integrates
#: against.  A table, for the reason the bodies below are one.
_OBJECTIVE_STEPS_INPUTS_BY_EXPANDED = {
    True: lambda array_inputs, dim: _metric_array_inputs(array_inputs, dim),
    False: lambda array_inputs, dim: array_inputs,
}


#: Whether the element's strategy produced a closed-form 0-form decides which
#: body is spelled, as it does for the 1-form and the action.
_OBJECTIVE_STEPS_BODY_BY_EXPANDED = {
    True: lambda plan, source_builder: _expanded_simplex_metric_value_body(
        plan, source_builder
    ),
    False: lambda plan, source_builder: None,
}


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
    """The stepped objective, where the form has one.

    Whether it does is `plans.form_emission.publishes_objective_steps`: a
    0-form carrying a weak form, because there is nothing to step in a
    gradient or a Hessian action and the stepping happens on the density
    before it is contracted.  This used to be a guard returning an empty
    list five hundred lines above the end of the function it guarded.
    """
    return _OBJECTIVE_STEPS_BY_APPLICABILITY[publishes_objective_steps(form)](
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
        geometry_mode,
        source_builder,
    )


def _objective_steps_lines(
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
    geometry_mode,
    source_builder,
):
    """The kernel itself, reached only when the plan layer says it exists."""
    n_field_components = form_n_field_components(form, dim)
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    if geometry_mode not in ("affine", "isoparametric"):
        raise ValueError("mesh geometry_mode must be 'affine' or 'isoparametric'")
    work_item = _work_item_index(source_builder)
    material_parameter_names = form_material_parameter_names(form)

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
    # The third member of the triple takes the cached metric on the same terms
    # the other two do, and additionally only when the energy really is the
    # quadratic invariant the metric can express.
    metric = geometry_variant_plan(
        form.weak_form,
        quadrature_rule,
        specialized=geometry_mode == "affine" and specialized_prefix is not None,
    ).cached_metric
    value_plan = expanded_simplex_metric_value_plan(
        metric,
        dim,
        n_nodes,
        n_qp,
        n_field_components,
        writes_per_shape(form),
        metric_value_scale(form.weak_form) if metric is not None else None,
    )
    array_inputs = _OBJECTIVE_STEPS_INPUTS_BY_EXPANDED[value_plan is not None](
        array_inputs, dim
    )
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_tensor_product_geometry = (
        geometry_mode == "isoparametric"
        and is_tensor_product_family(geometry_family)
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
    coordinate_streams_name = "bcoordinate_data"

    base_params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const RSTR elements",
    ]
    if geometry_mode == "affine":
        base_params.extend(
            "const g_t *const RSTR %s" % abi_geometry_name(stream)
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        base_params.append("const g_t *const *const RSTR points")

    material_params = _form_material_parameter_declarations(form)
    field_params = ["const ptrdiff_t u_stride"]
    field_params.extend(
        "const s_t *const RSTR u%s" % _component_name(d)
        for d in range(n_field_components)
    )
    field_params.append("const ptrdiff_t h_stride")
    field_params.extend(
        "const s_t *const RSTR h%s" % _component_name(d)
        for d in range(n_field_components)
    )
    step_params = (
        "const int nsteps",
        "const s_t *const RSTR steps",
    )
    output_params = ("s_t *const RSTR value",)

    impl_params = (
        tuple(base_params)
        + tuple(material_params)
        + tuple(field_params)
        + tuple(step_params)
        + tuple(output_params)
    )
    wrapper_params = tuple(
        param.replace("g_t", "geom_t").replace("g_t", "geom_t")
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
        source_builder.mesh_template_line(geometry_mode, ("int VS",)),
        source_builder.mesh_function_line(implementation_name),
    ]
    lines.extend(parameter_list_lines(impl_params))
    implementation_start = len(lines)
    lines.extend(
        [
            ") {",
            kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
            kernel_constant("NQ", n_qp, indent="  "),
            kernel_constant("NS", n_nodes, indent="  "),
            discard_unused("nnodes", indent="  "),
        ]
    )
    if geometry_mode == "isoparametric":
        for d in range(dim):
            lines.append(
                "  const g_t *const RSTR %s = points[%d];"
                % (_component_name(d), d)
            )
        if compact_coordinate_buffers:
            lines.extend(
                _ordered_element_pointer_array_lines(
                    "idx_t",
                    "coordinate_elements",
                    "elements",
                    stream_shape_order,
                    "  ",
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
                kernel_constant("NQ1", "%d" % quadrature_rule.tensor_product_n_qp_1d, indent="  "),
                kernel_constant("NS1", "%d" % quadrature_rule.tensor_product_n_shape_1d, indent="  "),
            ]
        )

    lines.extend(
        [
            "",
            *source_builder.parallel_for_lines(),
            *source_builder.mesh_loop_lines(),
            "    idx_t ev[VS * NS];",
        ]
    )

    compact_stream_buffers = use_stream_arrays
    if compact_stream_buffers:
        lines.append("    s_t bu_data[NS * NC][VS];")
        lines.append("    s_t bu_base_data[NS * NC][VS];")
        lines.append("    s_t bh_data[NS * NC][VS];")
        lines.append("    s_t bvalue[VS];")
        if compact_coordinate_buffers:
            lines.append("    s_t bcoordinate_data[NS * ND][VS];")
    elif compact_coordinate_buffers:
        lines.append("    s_t bcoordinate_data[NS * ND][VS];")
    elif geometry_mode == "isoparametric":
        for stream in _coordinate_stream_names(dim, n_nodes):
            lines.append("    s_t b%s[VS];" % stream)
    if geometry_mode == "isoparametric":
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                extent = "NQ * VS"
                lines.append("    s_t b%s[%s];" % (stream, extent))
    if not compact_stream_buffers:
        for stream in _field_stream_names("u", n_field_components, n_nodes):
            lines.append("    s_t b%s[VS];" % stream)
            lines.append("    s_t b%s_base[VS];" % stream)
        for stream in _field_stream_names("h", n_field_components, n_nodes):
            lines.append("    s_t b%s[VS];" % stream)
        lines.append("    s_t bvalue[VS];")

    lines.extend(
        [
            "",
            "    for (int element_node = 0; element_node < NS; ++element_node) {",
            "      const idx_t *const RSTR element_shape = elements[element_node] + evb;",
            "      idx_t *const RSTR ev_node = &ev[element_node * VS];",
            *_work_item_loop_lines(source_builder, "      "),
            "        ev_node[%s] = element_shape[%s];"
            % (work_item, work_item),
            "      }",
            "    }",
        ]
    )

    if geometry_mode == "isoparametric":
        if compact_coordinate_buffers:
            lines.append("    const g_t *const coordinate_components[ND] = {%s};" % ", ".join(_component_name(d) for d in range(dim)))
            lines.extend(
                [
                    "",
                    "    for (int shape = 0; shape < NS; ++shape) {",
                    *(["      const idx_t *const RSTR ev_shape = &ev[shape * VS];"] if identity_stream_shape_order else ["      const idx_t *const RSTR coordinate_element_shape = coordinate_elements[shape];"]),
                    "      for (int d = 0; d < ND; ++d) {",
                    *_work_item_loop_lines(source_builder, "        "),
                    "          bcoordinate_data[shape * ND + d][%s] = coordinate_components[d][%s];"
                    % (
                        work_item,
                        "ev_shape[%s]" % work_item
                        if identity_stream_shape_order
                        else "coordinate_element_shape[evb + %s]" % work_item,
                    ),
                    "        }",
                    "      }",
                    "    }",
                ]
            )
        else:
            lines.extend(["", *_work_item_loop_lines(source_builder, "    ")])
            for shape in range(n_nodes):
                for d in range(dim):
                    stream = "%s%d" % (_component_name(d), shape)
                    lines.append(
                        "      b%s[%s] = %s[ev[%d * VS + %s]];"
                        % (stream, work_item, _component_name(d), shape, work_item)
                    )
            lines.append("    }")

    if compact_stream_buffers:
        lines.append("")
        lines.append("    const s_t *const u_components[NC] = {%s};" % ", ".join("u%s" % _component_name(d) for d in range(n_field_components)))
        lines.append("    const s_t *const h_components[NC] = {%s};" % ", ".join("h%s" % _component_name(d) for d in range(n_field_components)))
        lines.extend(
            [
                *_ordered_stream_pointer_array_lines(
                    "const s_t *",
                    "bu_streams",
                    "bu_data",
                    dim,
                    stream_shape_order,
                    "    ",
                ),
            ]
        )
        lines.extend(
            [
                "",
                "    for (int shape = 0; shape < NS; ++shape) {",
                "      const idx_t *const RSTR ev_shape = &ev[shape * VS];",
                "      for (int d = 0; d < NC; ++d) {",
                *_work_item_loop_lines(source_builder, "        "),
                "          const idx_t node = ev_shape[%s];" % work_item,
                "          bu_base_data[shape * NC + d][%s] = u_components[d][node * u_stride];" % work_item,
                "          bh_data[shape * NC + d][%s] = h_components[d][node * h_stride];" % work_item,
                "        }",
                "      }",
                "    }",
            ]
        )
    else:
        lines.append(
            "    const s_t *const bu_streams[NS * %d] = {%s};"
            % (
                dim,
                ", ".join(
                    _BLOCK_FMT % stream
                    for stream in streams_in_shape_order(
                        _field_stream_names("u", n_field_components, n_nodes),
                        n_field_components,
                        stream_shape_order,
                    )
                ),
            )
        )
        lines.extend(["", *_work_item_loop_lines(source_builder, "    ")])
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                lines.append(
                    "      bu%s%d_base[%s] = u%s[ev[%d * VS + %s] * u_stride];"
                    % (component, shape, work_item, component, shape, work_item)
                )
                lines.append(
                    "      bh%s%d[%s] = h%s[ev[%d * VS + %s] * h_stride];"
                    % (component, shape, work_item, component, shape, work_item)
                )
        lines.append("    }")

    if geometry_mode == "isoparametric" and use_tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim_name="ND",
                dim=dim,
                n_shape=n_nodes,
                n_qp=quadrature_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="bcoordinate_data",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "badj%d[%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "bdet0[%s]" % index
                ),
                adjugate_streams=tuple(
                    "badj%d" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="bdet0",
                shape_name=tensor_shape_name,
                grad_name=tensor_grad_name,
            )
        )
    elif geometry_mode == "isoparametric":
        lines.extend(["", *quadrature_scope_lines(quadrature_rule.element_type, "    ")])
        if use_tensor_product_geometry:
            lines.extend(tensor_product_q_index_lines(dim, "      "))
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
        lines.append("    }")
    elif geometry_mode == "affine":
        lines.extend(
            _sfem_soa_affine_geometry_stream_lines(
                source_builder,
                element_inputs,
                "    ",
            )
        )

    call_args = ["ne"]
    call_args.append("0" if geometry_mode == "affine" else "VS")
    if geometry_mode == "affine":
        call_args.extend(
            _BLOCK_FMT % stream
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    else:
        call_args.extend(
            _BLOCK_FMT % stream
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
        call_args.append("bu_streams")
    else:
        call_args.extend(_BLOCK_FMT % stream for stream in _field_stream_names("u", n_field_components, n_nodes))
    call_args.append("bvalue")

    lines.extend(
        [
            "",
            "    for (int step = 0; step < nsteps; ++step) {",
            "      const s_t alpha = steps[step];",
        ]
    )
    if compact_stream_buffers:
        lines.extend(
            [
                "      for (int shape = 0; shape < NS; ++shape) {",
                "        for (int d = 0; d < NC; ++d) {",
                *_work_item_loop_lines(source_builder, "          "),
                "            bu_data[shape * NC + d][%s] = bu_base_data[shape * NC + d][%s] + alpha * bh_data[shape * NC + d][%s];"
                % (work_item, work_item, work_item),
                "          }",
                "        }",
                "      }",
            ]
        )
    else:
        lines.extend(_work_item_loop_lines(source_builder, "      "))
        for shape in range(n_nodes):
            for d in range(dim):
                component = _component_name(d)
                lines.append(
                    "        bu%s%d[%s] = bu%s%d_base[%s] + alpha * bh%s%d[%s];"
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
        lines.append("      }")
    lines.extend(
        [
            *_work_item_loop_lines(source_builder, "      "),
            "        bvalue[%s] = s_t(0);" % work_item,
            "      }",
            "",
            "      %s<s_t, NQ, NS, VS>(%s);"
            % (block_name, ", ".join(call_args)),
            "",
            *_work_item_loop_lines(source_builder, "      "),
            "        value[(ptrdiff_t)step * nelements + evb + %s] = bvalue[%s];"
            % (work_item, work_item),
            "      }",
            "    }",
            "  }",
            "",
            *source_builder.success_return_lines(),
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )

    expanded_body = _OBJECTIVE_STEPS_BODY_BY_EXPANDED[value_plan is not None](
        value_plan, source_builder
    )
    if expanded_body is not None:
        # The element's own strategy reaching the 0-form.  Built and dropped
        # rather than skipped, for the reason the other two bodies state.
        del lines[implementation_start:]
        lines.extend(
            [
                ") {",
                *expanded_body,
                "}",
                "",
                "} // namespace codegen",
                "} // namespace sfem",
                "",
            ]
        )

    wrapper_args = tuple(_cpp_argument_name(param) for param in wrapper_params)
    lines.extend(
        runtime_typed_entry_point_lines(
            function_name,
            wrapper_params,
            lambda scalar_type, _positional: source_builder.wrapper_call_lines(
                implementation_name,
                scalar_type,
                ", geom_t, %d" % vector_size,
                cast_arguments(wrapper_params, wrapper_args, scalar_type),
            ),
            parameter_lines=parameter_list_lines,
        )
    )
    # The objective_steps packed wrappers are the other half of the packed
    # family, emitted from here rather than from the mesh operator, and they
    # follow the same plan.
    for _packed_layout in geometry_variant_plan(
        form.weak_form, quadrature_rule
    ).packed_mesh_layouts:
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
                n_field_components=n_field_components,
                metric=metric,
                expanded_value_plan=value_plan,
            )
        )
    return lines


#: Whether the form publishes this kernel decides whether there is one.  A
#: table, as `_KERNEL_BY_APPLICABILITY` is in `inexact_apply_codegen.py`:
#: emission looks the answer up rather than deciding it again.
_OBJECTIVE_STEPS_BY_APPLICABILITY = {
    True: _objective_steps_lines,
    False: lambda *arguments: [],
}


def shared_primitive_files(source_builder, basis_family):
    """The headers that belong to the target rather than to any material.

    Kernel maths, the geometry kernels, the diagnostics record, the packed
    thread scratch and the sum-factorization micro-kernels.  Not one of them
    reads a form, an element or a material: they are what the target spells,
    and a material generation writes them only because it is the thing that
    happens to run.

    Which is why they are a function and not a list inside the material path.
    The `.cuh` twins of these five are in the shipped tree without anything
    regenerating them -- `regenerate_all.sh` does not build for CUDA -- and they
    had drifted several refactors behind their `.hpp` counterparts, silently,
    because `codegen_snapshot` exempts what a plain regeneration cannot produce.
    A generator that can write them without a material is what closes that, and
    it must be this code writing them rather than a second copy of it.
    """
    header_guard_suffix = source_builder.header_guard_suffix()
    files = [
        GeneratedKernelFile(
            source_builder.header_name("kernel_math"),
            _sfem_math_header_source(
                header_guard_suffix,
                _inline_qualifier(source_builder),
                _defines_sfem_inline(source_builder),
            ),
        ),
        GeneratedKernelFile(
            source_builder.header_name("geometry_kernels"),
            source_builder.geometry_header_source(),
        ),
        GeneratedKernelFile(
            source_builder.header_name("kernel_diagnostics"),
            "\n".join(
                _sfem_soa_diagnostics_header(
                    _diagnostic_work_item(source_builder),
                    header_guard_suffix,
                    _inline_qualifier(source_builder),
                    _defines_sfem_inline(source_builder),
                    host_qualifier=_host_function_qualifier(source_builder),
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
                source_builder.header_name("tensor_product_kernels"),
                source_builder.tensor_product_header_source(),
            )
        )
    return tuple(files)


def _sfem_soa_direct_hessian_push_forward_lines(weak_form, dim, indent):
    """Reference gradient to physical, through the adjugate over the determinant."""
    lines = []
    for row in range(weak_form.n_field_components):
        for col in range(dim):
            terms = [
                "gu_ref%d * adj_lane%d" % (row * dim + k, k * dim + col)
                for k in range(dim)
            ]
            lines.append(
                "%sconst s_t gu%d = (%s) * idet;"
                % (indent, row * dim + col, " + ".join(terms))
            )
    return lines


def _sfem_soa_direct_hessian_state_gradient_lines(dim, reference_prefix, indent):
    """The state's reference gradient at every quadrature point, sum-factorized.

    Contracting the state with each shape function's reference gradient inside
    the quadrature loop costs `NQ * NS` multiply-adds per component, which is
    `p^6` work for a degree-`p` element.  `tensor_gradient_contiguous_scalar` is
    the same contraction taken one direction at a time, at `p^4`, and it is the
    routine the matrix-free apply on this element already uses.  It answers for
    every quadrature point at once, so the call is hoisted above the loop that
    reads it.

    The `_scalar` spelling is the one this kernel wants.  Assembly has a single
    element in hand -- it scatters into a sparse row and has nowhere to put a
    block of them -- so the lane-blocked micro-kernels would carry a lane loop
    that runs once and stage buffers with `VS - 1` slots nobody writes.
    """
    return [
        "%ss_t state_gradient_ref[%s];" % (indent, c_product("NC", "NQ", "ND")),
        "%sfor (int component = 0; component < NC; ++component) {" % indent,
        "%s  tensor_gradient_contiguous_scalar<s_t, NQ, NS, VS, %d, NC>(" % (indent, dim),
        "%s      %sshape_1d, %sgrad_1d, bu_data, component,"
        % (indent, reference_prefix, reference_prefix),
        "%s      state_gradient_ref + %s);"
        % (indent, c_product("component", "NQ", "ND")),
        "%s}" % indent,
    ]


def _sfem_soa_direct_hessian_current_gradient_lines(
    weak_form,
    dim,
    n_field_components,
    reference_inputs,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_prefix,
    indent,
):
    """Rebuild the current deformation gradient from the state streams.

    A state-dependent material's tangent is an ordinary expression once `gu` is
    known, so the element matrix is computed the same way the apply computes it:
    contract the state with the reference gradients, push forward with the
    adjugate, and hand the result to the same material expression.
    """
    lines = []
    if use_tensor_product_reference:
        # Already computed for every quadrature point, above the loop.
        for row in range(n_field_components):
            for col in range(dim):
                lines.append(
                    "%sconst s_t gu_ref%d = state_gradient_ref[%s];"
                    % (
                        indent,
                        row * dim + col,
                        c_sum(c_product(row, "NQ", "ND"), c_product("q", "ND"), col),
                    )
                )
        return lines + _sfem_soa_direct_hessian_push_forward_lines(
            weak_form, dim, indent
        )
    for idx in range(n_field_components * dim):
        lines.append("%ss_t gu_ref%d = s_t(0);" % (indent, idx))
    lines.append("%sfor (int shape = 0; shape < NS; ++shape) {" % indent)
    if use_tensor_product_reference:
        lines.extend(
            _tensor_product_shape_coordinate_lines(
                dim,
                "shape",
                "state",
                "%s  " % indent,
            )
        )
    for ref_component in range(dim):
        lines.append(
            "%s  const s_t state_grad_ref%d = %s;"
            % (
                indent,
                ref_component,
                _sfem_soa_reference_gradient_expr_for_shape(
                    dim,
                    ref_component,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    reference_inputs,
                    "shape",
                    coord_prefix="state",
                    reference_prefix=reference_prefix,
                ),
            )
        )
    for row in range(n_field_components):
        lines.append(
            "%s  const s_t state_u%d = bu_data[%s][lane];"
            % (indent, row, c_sum(c_product("shape", "NC"), row))
        )
        for col in range(dim):
            lines.append(
                "%s  gu_ref%d += state_u%d * state_grad_ref%d;"
                % (indent, row * dim + col, row, col)
            )
    lines.append("%s}" % indent)
    return lines + _sfem_soa_direct_hessian_push_forward_lines(weak_form, dim, indent)


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
    n_field_components = form_n_field_components(form, dim)
    uses_current = form_reads_current(form, default=True)
    weak_form = form.weak_form
    material = _weak_form_material_expression(
        weak_form,
        form.name,
        _weak_form_deformation_gradient_substitutions(
            weak_form,
            "gu",
            scalar_temporaries=True,
        ),
        tuple(
            sp.symbols("trial_grad[%d]" % i)
            for i in range(weak_form.n_field_components * dim)
        ),
    )
    lines = [
        "%sfor (int entry = 0; entry < NDOFS * NDOFS; ++entry) {" % indent,
        "%s  element_matrix[entry] = s_t(0);" % indent,
        "%s}" % indent,
    ]
    if use_tensor_product_reference and emit_tensor_product_static_constants:
        lines.extend(
            [
                kernel_constant("NQ1", quadrature_rule.tensor_product_n_qp_1d, indent=indent),
                kernel_constant("NS1", quadrature_rule.tensor_product_n_shape_1d, indent=indent),
            ]
        )
    if uses_current and use_tensor_product_reference:
        lines.extend(
            _sfem_soa_direct_hessian_state_gradient_lines(dim, reference_prefix, indent)
        )
    lines.append("%sfor (int q = 0; q < NQ; ++q) {" % indent)
    if use_tensor_product_reference:
        lines.extend(tensor_product_q_index_lines(dim, indent + "  "))
        lines.append(
            "%s  const s_t qw = %s;"
            % (indent, tensor_product_quadrature_weight_expr(dim, "%sq_weight_1d" % reference_prefix))
        )
    else:
        lines.append("%s  const s_t qw = %sq_weight[q];" % (indent, reference_prefix))
    lines.extend(
        [
            "%s  const int lane = 0;" % indent,
            "%s  const ptrdiff_t goff = q * VS + lane;" % indent,
        ]
    )
    for component in range(dim * dim):
        lines.append(
            "%s  const s_t adj_lane%d = badj%d[goff];"
            % (indent, component, component)
        )
    lines.extend(
        [
            "%s  const s_t det_lane0 = bdet0[goff];"
            % indent,
            "%s  const s_t idet = s_t(1) / det_lane0;"
            % indent,
        ]
    )
    if uses_current:
        lines.extend(
            _sfem_soa_direct_hessian_current_gradient_lines(
                weak_form,
                dim,
                n_field_components,
                reference_inputs,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_prefix,
                "%s  " % indent,
            )
        )
    lines.extend(
        [
            "%s  for (int trial_component = 0; trial_component < NC; ++trial_component) {"
            % indent,
            "%s    for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {"
            % indent,
        ]
    )
    if use_tensor_product_reference:
        lines.extend(
            _tensor_product_shape_coordinate_lines(
                dim,
                "trial_shape",
                "trial",
                "%s      " % indent,
            )
        )
    for ref_component in range(dim):
        lines.append(
            "%s      const s_t trial_grad_ref%d = %s;"
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
    # A field gradient has one row per field component and one column per
    # spatial direction, so it is `NC * ND` entries
    # strided by ND.  Both used to be spelled NC,
    # which is right only for a displacement: a scalar field in two dimensions
    # then declared `trial_grad[1]` and wrote index 1, which the compiler will
    # tell you about under -Warray-bounds and which no material reached until
    # laplace was written as an energy.
    lines.extend(
        [
            "%s      s_t trial_grad[NC * ND];" % indent,
            "%s      for (int i = 0; i < NC * ND; ++i) {" % indent,
            "%s        trial_grad[i] = s_t(0);" % indent,
            "%s      }" % indent,
        ]
    )
    for phys_component in range(dim):
        terms = [
            "trial_grad_ref%d * adj_lane%d"
            % (ref_component, ref_component * dim + phys_component)
            for ref_component in range(dim)
        ]
        lines.append(
            "%s      trial_grad[trial_component * ND + %d] = (%s) * idet;"
            % (indent, phys_component, " + ".join(terms))
        )
    lines.append("%s      s_t material[NC * ND];" % indent)
    local_material_lines = []
    _append_cse_array_assignments(
        local_material_lines,
        tuple(material),
        ["material[%d] =" % i for i in range(dim * dim)],
        "weak_hess_tmp",
    )
    lines.extend("%s      %s" % (indent, line.strip()) for line in local_material_lines)
    lines.extend(
        [
            "%s      for (int test_component = 0; test_component < NC; ++test_component) {"
            % indent,
            "%s        for (int test_shape = 0; test_shape < NS; ++test_shape) {"
            % indent,
        ]
    )
    if use_tensor_product_reference:
        lines.extend(
            _tensor_product_shape_coordinate_lines(
                dim,
                "test_shape",
                "test",
                "%s          " % indent,
            )
        )
    for ref_component in range(dim):
        lines.append(
            "%s          const s_t test_grad_ref%d = %s;"
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
    lines.append("%s          s_t entry = s_t(0);" % indent)
    for ref_component in range(dim):
        terms = [
            "material[%s] * adj_lane%d"
            % (c_sum(c_product("test_component", "ND"), k), ref_component * dim + k)
            for k in range(dim)
        ]
        lines.append(
            "%s          entry += test_grad_ref%d * qw * (%s);"
            % (indent, ref_component, " + ".join(terms))
        )
    lines.extend(
        [
            "%s          const int row = test_component * NS + test_shape;"
            % indent,
            "%s          const int col = trial_component * NS + trial_shape;"
            % indent,
            "%s          element_matrix[row * NDOFS + col] += entry;" % indent,
            "%s        }" % indent,
            "%s      }" % indent,
            "%s    }" % indent,
            "%s  }" % indent,
            "%s}" % indent,
        ]
    )
    return lines


def _sfem_soa_direct_hessian_element_matrix_call_lines(
    function_name,
    dim,
    material_parameter_names,
    uses_current,
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
        *("badj%d" % i for i in range(dim * dim)),
        "bdet0",
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
    if uses_current:
        args.append("bu_data")
    args.append("element_matrix")
    return (
        "%s%s<s_t, NQ, NS, VS>(%s);"
        % (indent, function_name, ", ".join(args)),
    )


def _sfem_soa_hessian_packed_crs_passes(
    coordinate_streams_name,
    dim,
    direct_hessian_function_name,
    function_base,
    identity_stream_shape_order,
    lines,
    local_prefix,
    material_parameter_names,
    n_nodes,
    n_qp,
    omit_reference_basis_inputs,
    packed_crs_passes,
    prefix,
    quadrature_rule,
    reference_inputs,
    reference_prefix,
    scalar_weight_name,
    source_builder,
    stream_shape_order,
    tensor_grad_name,
    tensor_shape_name,
    tensor_weight_name,
    use_reference_gradient_vectors,
    use_tensor_product_geometry,
    use_tensor_product_reference,
    uses_current,
    n_field_components=None,
    assembly=None,
):
    """The multi-pass packed CRS assembly: discover the pattern, then fill it.

    Lifted out of `_sfem_soa_hessian_matrix_assembly_function` unchanged --
    except that `row_pointer` and `column_index` were free variables of the
    enclosing function and did not come across, so the discover pass raised
    `NameError` the moment anything reached it.  Nothing did while the only
    materials with a packed CRS variant were vector-valued and took another
    branch first.  They are parameters now, defaulted the way every other
    scatter in this file defaults them.
    """
    assembly = BSRAssemblyPlan() if assembly is None else assembly
    row_pointer = assembly.row_pointer
    column_index = assembly.column_index
    n_field_components = dim if n_field_components is None else n_field_components
    if packed_crs_passes:
        packed_fill_impl = "%s_packed_fill_impl" % function_base
        packed_discover_impl = "%s_packed_discover_impl" % function_base
        packed_common_params = [
            argument.declaration for argument in PACKED_MESH_CORE_ARGUMENTS
        ]
        packed_state_params = [
            "const g_t *const *const RSTR points",
        ]
        packed_state_params.extend(
            "const s_t %s" % parameter
            for parameter in material_parameter_names
        )
        if uses_current:
            packed_state_params.append("const ptrdiff_t u_stride")
            packed_state_params.extend(
                "const s_t *const RSTR u%s" % _component_name(d)
                for d in range(n_field_components)
            )
        packed_fill_params = tuple(
            packed_common_params
            + packed_state_params
            + [
                "const count_t *const RSTR packed_element_entries",
                "s_t *const RSTR values",
            ]
        )
        packed_discover_params = tuple(
            packed_common_params
            + [
                "const count_t *const RSTR %s" % row_pointer,
                "const idx_t *const RSTR %s" % column_index,
                "count_t *const RSTR packed_element_entries",
            ]
        )
        lines.extend(
            [
                "template <typename s_t, typename g_t>",
                "static int %s(" % packed_discover_impl,
            ]
        )
        lines.extend(parameter_list_lines(packed_discover_params))
        lines.extend(
            [
                ") {",
                kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
                kernel_constant("NS", n_nodes, indent="  "),
                discard_unused("nnodes", indent="  "),
                discard_unused("max_nodes_per_pack", indent="  "),
                discard_unused("n_shared_nodes", indent="  "),
                *source_builder.parallel_for_lines(),
                "  for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "    const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "    const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "    for (ptrdiff_t element = e_start; element < e_end; ++element) {",
                "      idx_t ev[NS];",
                "      for (int shape = 0; shape < NS; ++shape) {",
                "        ev[shape] = %s_packed_global_node(elements[shape][element], pack, owned_nodes_ptr, ghost_ptr, ghost_idx);" % function_base,
                "      }",
                "      count_t *const entries = &packed_element_entries[element * (NC * NS) * (NC * NS)];",
                "      %s_discover_packed_crs_entries<s_t>(ev, rowptr, colidx, entries);" % function_base,
                "    }",
                "  }",
                "  return SFEM_SUCCESS;",
                "}",
                "",
                "template <typename s_t, typename g_t>",
                "static int %s(" % packed_fill_impl,
            ]
        )
        lines.extend(parameter_list_lines(packed_fill_params))
        lines.extend(
            [
                ") {",
                kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
                kernel_constant("NQ", n_qp, indent="  "),
                kernel_constant("NS", n_nodes, indent="  "),
                kernel_constant("VS", "1", indent="  "),
                kernel_constant("NDOFS", "NC * NS", indent="  "),
                discard_unused("nnodes", indent="  "),
                discard_unused("n_shared_nodes", indent="  "),
            ]
        )
        for d in range(dim):
            lines.append(
                "  const g_t *const RSTR %s = points[%d];"
                % (_component_name(d), d)
            )
        lines.extend(
            _ordered_element_pointer_array_lines(
                "uint16_t",
                "coordinate_elements",
                "elements",
                stream_shape_order,
                "  ",
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
                "  {",
                "    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);",
            ]
        )
        if uses_current:
            lines.append(
                "    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);"
            )
        lines.extend(
            [
                "",
                "#pragma omp for schedule(static)",
                "    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "      const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
                "      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                "      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];",
                "      const g_t *const coordinate_components[ND] = {%s};"
                % ", ".join(_component_name(d) for d in range(dim)),
            ]
        )
        if uses_current:
            lines.append(
                "      const s_t *const u_components[NC] = {%s};"
                % ", ".join("u%s" % _component_name(d) for d in range(n_field_components))
            )
        lines.extend(
            [
                "      for (int d = 0; d < ND; ++d) {",
                "        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;",
                "        const g_t *const RSTR coordinate_component = coordinate_components[d];",
            ]
        )
        if uses_current:
            lines.extend(
                [
                    "        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;",
                    "        const s_t *const RSTR u_component = u_components[d];",
                ]
            )
        lines.extend(
            [
                "        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                "          const idx_t node = owned_nodes_ptr[pack] + k;",
                "          pk_coordinate[k] = s_t(coordinate_component[node]);",
            ]
        )
        if uses_current:
            lines.append("          pk_u_component[k] = u_component[node * u_stride];")
        lines.extend(
            [
                "        }",
                "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                "          const idx_t node = ghosts[k];",
                "          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);",
            ]
        )
        if uses_current:
            lines.append("          pk_u_component[n_contiguous + k] = u_component[node * u_stride];")
        lines.extend(
            [
                "        }",
                "      }",
                "",
                "      for (ptrdiff_t element = e_start; element < e_end; ++element) {",
                "        s_t element_matrix[NDOFS * NDOFS];",
                "        s_t bcoordinate_data[NS * ND][VS];",
                kernel_constant("ne", "VS", indent="        "),
            ]
        )
        if uses_current:
            lines.append("        s_t bu_data[NS * NC][VS];")
        for stream in _soa_array_stream_names(_adjugate_input(dim)):
            lines.append("        s_t b%s[NQ * VS];" % stream)
        lines.append("        s_t bdet0[NQ * VS];")
        lines.append(
            "        s_t *badj_streams[ND * ND] = {%s};"
            % ", ".join("badj%d" % i for i in range(dim * dim))
        )
        lines.extend(
            [
                "",
                "        for (int shape = 0; shape < NS; ++shape) {",
                "          const uint16_t packed_node = elements[shape][element];",
                *([] if identity_stream_shape_order else ["          const uint16_t coordinate_packed_node = coordinate_elements[shape][element];"]),
                "          for (int d = 0; d < ND; ++d) {",
                "            bcoordinate_data[shape * ND + d][0] = pk_coordinates[d * max_nodes_per_pack + %s];" % ("packed_node" if identity_stream_shape_order else "coordinate_packed_node"),
            ]
        )
        if uses_current:
            lines.append("            bu_data[shape * NC + d][0] = pk_u[d * max_nodes_per_pack + packed_node];")
        lines.extend(
            [
                "          }",
                "        }",
                "",
            ]
        )
        if use_tensor_product_geometry:
            lines.extend(
                tensor_product_gradient_isoparametric_geometry_lines(
                    dim_name="ND",
                    dim=dim,
                    n_shape=n_nodes,
                    n_qp=quadrature_rule.n_qp,
                    local_prefix=local_prefix,
                    coordinate_streams="bcoordinate_data",
                    contiguous_coordinate_streams=True,
                    adjugate_target=lambda component, index: (
                        "badj%d[%s]" % (component, index)
                    ),
                    determinant_target=lambda index: (
                        "bdet0[%s]" % index
                    ),
                    adjugate_streams=tuple(
                        "badj%d" % component
                        for component in range(dim * dim)
                    ),
                    determinant_stream="bdet0",
                    shape_name=tensor_shape_name,
                    grad_name=tensor_grad_name,
                )
            )
        else:
            lines.extend(["", "      for (int q = 0; q < NQ; ++q) {"])
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
            lines.extend("  %s" % line if line else line for line in geometry_lines)
            lines.append("      }")
        lines.append("")
        lines.extend(
            _sfem_soa_direct_hessian_element_matrix_call_lines(
                direct_hessian_function_name,
                dim,
                material_parameter_names,
                uses_current,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_inputs,
                tensor_shape_name,
                tensor_grad_name,
                tensor_weight_name,
                scalar_weight_name,
                reference_prefix,
                "      ",
            )
        )
        lines.extend(
            [
                "",
                "      const count_t *const entries = &packed_element_entries[element * NDOFS * NDOFS];",
                "      %s_scatter_packed_crs_entries(element_matrix, entries, values);" % function_base,
                "      }",
                "    }",
                "  }",
                "  return SFEM_SUCCESS;",
                "}",
                "",
            ]
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
    n_field_components = form_n_field_components(form, dim)
    formats = published_matrix_formats(matrix_format_plan)
    if not formats:
        return []
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    # The assembly is the direct element matrix and nothing else -- there is no
    # fallback that recovers the matrix by applying the operator to unit basis
    # vectors -- so a form that cannot reach the direct kernel publishes no
    # matrix at all.  This is the predicate the header consults, so the header
    # and the assembly cannot disagree about which kernels exist.
    if not _sfem_soa_direct_hessian_matrix_assembly_available(
        form,
        reference_inputs,
    ):
        return []
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    crs_passes = packed_crs_passes(matrix_format_plan)
    material_parameter_names = form_material_parameter_names(form)
    uses_current = form_reads_current(form, default=True)

    use_tensor_product_reference = _use_tensor_product_reference(
        quadrature_rule,
        reference_inputs,
        basis_family,
    )
    use_tensor_product_geometry = is_tensor_product_family(geometry_family)
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
    coordinate_streams_name = "bcoordinate_data"
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
    lines.extend(_sfem_soa_hessian_scatter_lines(function_base, dim, n_nodes, formats, n_field_components=n_field_components))
    if crs_passes:
        lines.extend(
            _sfem_soa_hessian_packed_crs_helper_lines(
                function_base,
                dim,
                n_nodes,
                n_field_components=n_field_components,
            )
        )
    lines.extend(
        [
            "template <typename s_t, typename g_t, int FORMAT>",
            "static int %s(" % implementation_name,
            "    const ptrdiff_t nelements,",
            "    const ptrdiff_t nnodes,",
            "    idx_t **const RSTR elements,",
            "    const g_t *const *const RSTR points,",
        ]
    )
    lines.extend(
        "    const s_t %s," % parameter
        for parameter in material_parameter_names
    )
    if uses_current:
        lines.append("    const ptrdiff_t u_stride,")
        for d in range(dim):
            comma = "," if d + 1 < dim else ","
            lines.append(
                "    const s_t *const RSTR u%s%s"
                % (_component_name(d), comma)
            )
    lines.extend(
        [
            "    const count_t *const RSTR rowptr,",
            "    const idx_t *const RSTR colidx,",
            "    s_t *const RSTR values,",
            "    const int *const RSTR diag_offsets,",
            "    const ptrdiff_t ndiag,",
            "    const ptrdiff_t coo_nnz,",
            "    const idx_t *const RSTR coo_rows,",
            "    const idx_t *const RSTR coo_cols,",
            "    idx_t *const RSTR coo_triplet_rows,",
            "    idx_t *const RSTR coo_triplet_cols) {",
            kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
            kernel_constant("NQ", n_qp, indent="  "),
            kernel_constant("NS", n_nodes, indent="  "),
            kernel_constant("VS", "1", indent="  "),
            kernel_constant("NDOFS", "NC * NS", indent="  "),
            discard_unused("nnodes", indent="  "),
            # One signature carries every sparse format's arrays so that the
            # dispatch can call it whatever the format is, but a given kernel
            # emits one format's scatter.  The rest are named by nobody, which
            # `-Wextra -Werror` rejects; the resolver checks each against the
            # body it ends up with and only unnames the ones truly unread.
            *(
                discard_unused(parameter, indent="  ")
                for parameter in (
                    "rowptr",
                    "colidx",
                    "diag_offsets",
                    "ndiag",
                    "coo_nnz",
                    "coo_rows",
                    "coo_cols",
                    "coo_triplet_rows",
                    "coo_triplet_cols",
                )
            ),
        ]
    )
    if uses_current:
        lines.append(
            "  const s_t *const u_components[NC] = {%s};"
            % ", ".join("u%s" % _component_name(d) for d in range(n_field_components))
        )
    for d in range(dim):
        lines.append(
            "  const g_t *const RSTR %s = points[%d];"
            % (_component_name(d), d)
        )
    lines.extend(
        _ordered_element_pointer_array_lines(
            "idx_t",
            "coordinate_elements",
            "elements",
            stream_shape_order,
            "  ",
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
            *_sfem_soa_matrix_format_assertion_lines(formats, "  "),
            *source_builder.parallel_for_lines(),
            "  for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "    idx_t ev[NS];",
            "    s_t element_matrix[NDOFS * NDOFS];",
            "    s_t bcoordinate_data[NS * ND][VS];",
            kernel_constant("ne", "VS", indent="    "),
        ]
    )
    if uses_current:
        lines.append("    s_t bu_data[NS * NC][VS];")
    for stream in _soa_array_stream_names(_adjugate_input(dim)):
        lines.append("    s_t b%s[NQ * VS];" % stream)
    lines.append("    s_t bdet0[NQ * VS];")
    lines.append(
            "    s_t *badj_streams[ND * ND] = {%s};"
            % ", ".join("badj%d" % i for i in range(dim * dim))
        )
    lines.extend(
        [
            "",
            "    for (int shape = 0; shape < NS; ++shape) {",
            "      const idx_t node = elements[shape][element];",
            *([] if identity_stream_shape_order else ["      const idx_t coordinate_node = coordinate_elements[shape][element];"]),
            "      ev[shape] = node;",
            "      for (int d = 0; d < ND; ++d) {",
            "        bcoordinate_data[shape * ND + d][0] = s_t(points[d][%s]);" % ("node" if identity_stream_shape_order else "coordinate_node"),
        ]
    )
    if uses_current:
        lines.append("        bu_data[shape * NC + d][0] = u_components[d][node * u_stride];")
    lines.extend(
        [
            "      }",
            "    }",
            "",
        ]
    )
    if use_tensor_product_geometry:
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim_name="ND",
                dim=dim,
                n_shape=n_nodes,
                n_qp=quadrature_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="bcoordinate_data",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "badj%d[%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "bdet0[%s]" % index
                ),
                adjugate_streams=tuple(
                    "badj%d" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="bdet0",
                shape_name=tensor_shape_name,
                grad_name=tensor_grad_name,
            )
        )
    else:
        lines.extend(["", "    for (int q = 0; q < NQ; ++q) {"])
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
        lines.append("    }")

    lines.append("")
    lines.extend(
        _sfem_soa_direct_hessian_element_matrix_call_lines(
            direct_hessian_function_name,
            dim,
            material_parameter_names,
            uses_current,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
            reference_inputs,
            tensor_shape_name,
            tensor_grad_name,
            tensor_weight_name,
            scalar_weight_name,
            reference_prefix,
            "    ",
        )
    )
    lines.append("")
    lines.extend(_sfem_soa_hessian_scatter_dispatch_lines(function_base, formats, "    "))
    lines.extend(
        [
            "  }",
            "",
            "  return SFEM_SUCCESS;",
            "}",
            "",
        ]
    )
    _sfem_soa_hessian_packed_crs_passes(
        coordinate_streams_name,
        dim,
        direct_hessian_function_name,
        function_base,
        identity_stream_shape_order,
        lines,
        local_prefix,
        material_parameter_names,
        n_nodes,
        n_qp,
        omit_reference_basis_inputs,
        crs_passes,
        prefix,
        quadrature_rule,
        reference_inputs,
        reference_prefix,
        scalar_weight_name,
        source_builder,
        stream_shape_order,
        tensor_grad_name,
        tensor_shape_name,
        tensor_weight_name,
        use_reference_gradient_vectors,
        use_tensor_product_geometry,
        use_tensor_product_reference,
        uses_current,
        n_field_components=n_field_components,
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
            crs_passes,
        )
    )
    return lines


def _sfem_soa_hessian_scatter_dispatch_lines(function_base, formats, indent):
    """Call the scatter for the format this kernel was instantiated with.

    None of the scatters can fail any more.  They used to: each re-established
    per element that the sparsity pattern contained the entries it was about to
    write, which is a property of the mesh and the pattern together and cannot
    change between elements or between calls.  That check belongs where the
    graph is built -- the operator's setup -- and the kernels now assume it.

    Nothing survives to check at run time.  A kernel instantiated for a format
    it has no scatter for is a programming error, and ``FORMAT`` is a template
    parameter, so `_sfem_soa_matrix_format_assertion_lines` states it as a
    `static_assert` at the top of the kernel instead of as a flag reduced over
    the element loop.
    """
    lines = []
    first = True
    for matrix_format, format_id in _MATRIX_FORMAT_IDS:
        if matrix_format not in formats:
            continue
        keyword = "if" if first else "} else if"
        lines.append("%s%s constexpr (FORMAT == %d) {" % (indent, keyword, format_id))
        lines.append(
            "%s  %s"
            % (indent, _MATRIX_FORMAT_SCATTERS[matrix_format] % function_base)
        )
        first = False
    if not first:
        lines.append("%s}" % indent)
    return lines


#: The matrix formats a scatter can be instantiated for, and the `FORMAT` value
#: that selects each.  Shared by the dispatch and the assertion below so the two
#: cannot disagree about which formats exist.
_MATRIX_FORMAT_IDS = (
    ("bsr", 1),
    ("crs", 0),
    ("block_diag_sym", 6),
)

_MATRIX_FORMAT_SCATTERS = {
    "bsr": "%s_scatter_bsr(ev, element_matrix, rowptr, colidx, values);",
    "crs": "%s_scatter_crs(ev, element_matrix, rowptr, colidx, values);",
    "block_diag_sym": "%s_scatter_block_diag_sym(ev, element_matrix, values);",
}


def _sfem_soa_matrix_format_assertion_lines(formats, indent):
    """Refuse, at compile time, a kernel instantiated for a format it cannot scatter.

    `FORMAT` is a template parameter, so which formats a kernel can scatter is a
    compile-time property and deserves a compile-time diagnostic.  It used to be
    a runtime flag -- `int unsupported_matrix_format` reduced over the element
    loop with `reduction(|:unsupported_matrix_format)` -- which made every thread
    carry and combine a copy of a value that a correctly instantiated kernel can
    never set, inside the hot loop, to catch a programming error.  A kernel is
    not the place to check what the compiler already knows.
    """
    ids = [format_id for name, format_id in _MATRIX_FORMAT_IDS if name in formats]
    if not ids:
        return []
    condition = " || ".join("FORMAT == %d" % format_id for format_id in sorted(ids))
    return [
        "%sstatic_assert(%s," % (indent, condition),
        '%s              "this kernel has no scatter for the requested matrix format");'
        % indent,
    ]


def _sfem_soa_hessian_matrix_public_function_base(prefix, quadrature_rule, geometry_mode):
    element = quadrature_rule.element_type.lower()
    # The geometry arrives as a value here, so the fragment is looked up rather
    # than spelled -- otherwise this site keeps emitting the long form after the
    # table moves, and the two halves of a name disagree.
    fragment = abi_mesh_fragment(geometry_mode)
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_hessian_%s" % (prefix, fragment)
    return "%s_%s_hessian_%s" % (
        prefix,
        element,
        fragment,
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
        "%sconst %s *const RSTR %s[NS] = {%s};"
        % (
            indent,
            pointer_type,
            array_name,
            ", ".join("%s[%d]" % (source_name, shape) for shape in stream_shape_order),
        )
    ]


def _ordered_stream_pointer_array_lines(pointer_type, array_name, storage_name, dim, stream_shape_order, indent):
    lines = [
        "%s%s%s[NS * NC];" % (indent, pointer_type, array_name),
    ]
    if tuple(stream_shape_order) == tuple(range(len(stream_shape_order))):
        lines.extend(
            [
                "%sfor (int stream = 0; stream < NS * NC; ++stream) {" % indent,
                "%s  %s[stream] = %s[stream];" % (indent, array_name, storage_name),
                "%s}" % indent,
            ]
        )
        return lines

    return [
        "%s%sconst %s[NS * NC] = {%s};"
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
    return sfem_soa_reference_input("adj", 1, 1, dim * dim)


def _sfem_soa_hessian_scatter_lines(function_base, dim, n_nodes, formats, n_field_components=None):
    n_field_components = dim if n_field_components is None else n_field_components
    lines = ["namespace sfem {", "namespace codegen {", ""]
    if pattern_scattered_formats(formats):
        find_cols_lines = [
            "static SFEM_INLINE void %s_find_cols(" % function_base,
            "    const idx_t *const RSTR targets,",
            "    const idx_t *const RSTR row,",
            "    const int lenrow,",
            "    idx_t *const RSTR ks) {",
        ]
        if n_nodes <= 10:
            find_cols_lines.append("#pragma unroll(%d)" % n_nodes)
        find_cols_lines.extend(
            [
                "  for (int d = 0; d < %d; ++d) {" % n_nodes,
                "    ks[d] = 0;",
                "  }",
                "  for (int k = 0; k < lenrow; ++k) {",
            ]
        )
        if n_nodes <= 10:
            find_cols_lines.append("#pragma unroll(%d)" % n_nodes)
        find_cols_lines.extend(
            [
                "    for (int d = 0; d < %d; ++d) {" % n_nodes,
                "      ks[d] += row[k] < targets[d];",
                "    }",
                "  }",
                "}",
                "",
            ]
        )
        lines.extend(find_cols_lines)
    for matrix_format in [f for f in _SCATTER_LINES_BY_FORMAT if f in formats]:
        lines.extend(
            _SCATTER_LINES_BY_FORMAT[matrix_format](
                function_base, dim, n_nodes, n_field_components=n_field_components
            )
        )
    return lines


SCATTER_LAYOUT = PrinterLayout(
    close_signature_on_last_param=True, atomic_pragma_at_column_zero=True
)


def _print_energy_kernel(node):
    """Render one energy local kernel.

    Default layout: these signatures close on their own line, unlike the
    scatter helpers.  The body rides along as RawLinesNode, so no pragma is
    needed -- it is already spelled in the text it carries.
    """
    return list(CLikeKernelASTPrinter().print_node(node))


def _print_scatter_function(node):
    """Render one scatter helper, in the layout the energy path already uses.

    The signature closes on its last parameter and the atomic pragma sits at
    column zero -- both different from the residual path, both facts about
    existing generated code rather than choices being made here.
    """
    target = current_target()
    printer = CLikeKernelASTPrinter(
        atomic_update_pragma=target.atomic_update_pragma() or "",
        layout=SCATTER_LAYOUT,
    )
    return list(printer.print_node(node)) + [""]


def _counting_loop(name, begin, end, body):
    """``for (int <name> = <begin>; <name> < <end>; ++<name>)``."""
    index = iterator(name, "int")
    return LoopNode(
        LoopKind.SHAPE,
        index,
        iteration_range(begin, end),
        pre_increment(index),
        body=tuple(body),
    )


def _assembly_reduction_is_atomic(reduction_policy, format_name):
    """How an assembly plan's reduction policy is spelled for this target.

    Every format's plan currently says ``atomic_add`` and every scatter emits
    an atomic update, so this reads as a formality -- but it is the point of
    connecting the plan: a policy the emitter cannot spell stops generation
    instead of silently becoming a plain add.
    """
    if str(reduction_policy) != "atomic_add":
        raise ValueError(
            "unsupported %s reduction policy '%s'; the emitter can spell "
            "atomic_add only" % (format_name, reduction_policy)
        )
    return True


def _sfem_soa_hessian_scatter_bsr_lines(function_base, dim, n_nodes, assembly=None, n_field_components=None):
    """Scatter one element block into a BSR matrix.

    The stream names and the reduction come from ``BSRAssemblyPlan``, which is
    where they are defined; this function spells them.  Its defaults are the
    names the generated kernels have always used, so the emitted text is
    unchanged.  That is the same arrangement
    ``_scalar_crs_matrix_scatter_lines`` has, and it is what takes
    BSRAssemblyPlan off the unread list -- with the same limit: the rename
    reaches this scatter's parameters and not the callers that pass the
    buffers in, which still name them as literals.

    The format vector problems use, so the one worth having in the IR.

    It no longer validates the matrix graph.  It used to: the locate loop tested
    each of the NS x NS candidate entries with a three-condition
    branch, kept a ``valid_block_graph`` flag, and reported the first failure
    through ``std::fprintf`` from inside whatever parallel region the caller had
    opened.  Whether the sparsity pattern contains this mesh's entries is a
    property of the mesh and the pattern together -- invariant across every
    element and across the whole call -- so re-deciding it per element cost
    O(elements x NS^2) to learn something that was already true or already
    false before the loop began.  It is the caller that builds the graph, and
    the caller that can establish it once.

    What is left is the locate loop itself, which is not validation: ``ks[j]``
    is the search result the scatter needs either way.  The check was fused into
    the search rather than sitting beside it, so removing it keeps the search
    and drops the test, the flag and the I/O.

    The function can no longer fail, so it returns void and its caller does not
    fold a status -- which is the arrangement ``_scatter_block_diag_sym`` has
    always had.
    """
    # The block is the field's components, not the spatial dimension.
    n_field_components = dim if n_field_components is None else n_field_components
    assembly = BSRAssemblyPlan() if assembly is None else assembly
    row_pointer = assembly.row_pointer
    column_index = assembly.column_index
    value_stream = assembly.value_stream
    atomic = _assembly_reduction_is_atomic(assembly.reduction_policy, "BSR")

    locate = _counting_loop(
        "j",
        0,
        expr_ref("NS"),
        [
            AssignmentNode(
                expr_ref("entries[i * NS + j]"),
                expr_ref("row_begin + ks[j]"),
            )
        ],
    )
    accumulate = [
        BufferDeclNode("const int", "col", (), expr_ref("bj * NS + j")),
        ScatterNode(
            expr_ref("block[bi * NC + bj]"),
            expr_ref("element_matrix[row * (NC * NS) + col]"),
            "+=",
            atomic=atomic,
        ),
    ]
    body = [
        BufferDeclNode("static constexpr int", "NC", (), expr_ref(str(n_field_components))),
        BufferDeclNode("static constexpr int", "NS", (), expr_ref(str(n_nodes))),
        BufferDeclNode("count_t", "entries", ("NS * NS",)),
        BufferDeclNode("idx_t", "ks", ("NS",)),
        _counting_loop(
            "i",
            0,
            expr_ref("NS"),
            [
                BufferDeclNode("const idx_t", "dof_i", (), expr_ref("ev[i]")),
                BufferDeclNode(
                    "const count_t", "row_begin", (), expr_ref("%s[dof_i]" % row_pointer)
                ),
                BufferDeclNode(
                    "const int",
                    "lenrow",
                    (),
                    expr_ref("(int)(%s[dof_i + 1] - row_begin)" % row_pointer),
                ),
                BufferDeclNode(
                    "const idx_t *const RSTR",
                    "cols",
                    (),
                    expr_ref("&%s[row_begin]" % column_index),
                ),
                CallNode(
                    "%s_find_cols" % function_base, ("ev", "cols", "lenrow", "ks")
                ),
                locate,
            ],
        ),
        _counting_loop(
            "i",
            0,
            expr_ref("NS"),
            [
                _counting_loop(
                    "j",
                    0,
                    expr_ref("NS"),
                    [
                        BufferDeclNode(
                            "s_t *const",
                            "block",
                            (),
                            expr_ref("&%s[entries[i * NS + j] * NC * NC]" % value_stream),
                        ),
                        _counting_loop(
                            "bi",
                            0,
                            expr_ref("NC"),
                            [
                                BufferDeclNode(
                                    "const int", "row", (), expr_ref("bi * NS + i")
                                ),
                                _counting_loop("bj", 0, expr_ref("NC"), accumulate),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ]
    return _print_scatter_function(
        FunctionDefNode(
            "%s_scatter_bsr" % function_base,
            params=(
                "const idx_t *const RSTR ev",
                "const s_t *const RSTR element_matrix",
                "const count_t *const RSTR rowptr",
                "const idx_t *const RSTR colidx",
                "s_t *const RSTR values",
            ),
            body=tuple(body),
            qualifier="static SFEM_INLINE",
            template_params=("typename s_t",),
        )
    )


def _sfem_soa_hessian_scatter_crs_lines(function_base, dim, n_nodes, n_field_components=None):
    n_field_components = dim if n_field_components is None else n_field_components
    return [
        "template <typename s_t>",
        "static SFEM_INLINE void %s_scatter_crs(" % function_base,
        "    const idx_t *const RSTR ev,",
        "    const s_t *const RSTR element_matrix,",
        "    const count_t *const RSTR rowptr,",
        "    const idx_t *const RSTR colidx,",
        "    s_t *const RSTR values) {",
        kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
        kernel_constant("NS", n_nodes, indent="  "),
        "  count_t row_begin[NS];",
        "  int lenrow[NS];",
        "  int local_col[NS * NS];",
        "  idx_t ks[NS];",
        "  for (int i = 0; i < NS; ++i) {",
        "    row_begin[i] = rowptr[ev[i]];",
        "    lenrow[i] = (int)(rowptr[ev[i] + 1] - row_begin[i]);",
        "    const idx_t *const RSTR cols = &colidx[row_begin[i]];",
        "    %s_find_cols(ev, cols, lenrow[i], ks);" % function_base,
        "    for (int j = 0; j < NS; ++j) {",
        "      local_col[i * NS + j] = (int)ks[j];",
        "    }",
        "  }",
        "  for (int i = 0; i < NS; ++i) {",
        "    const count_t rb = row_begin[i];",
        "    const int lr = lenrow[i];",
        "    for (int j = 0; j < NS; ++j) {",
        "      const int lc = local_col[i * NS + j];",
        "      for (int bi = 0; bi < NC; ++bi) {",
        "        const int row = bi * NS + i;",
        "        s_t *const row_values = &values[rb * NC * NC + bi * lr * NC];",
        "        for (int bj = 0; bj < NC; ++bj) {",
        "          const int col = bj * NS + j;",
        "#pragma omp atomic update",
        "          row_values[lc * NC + bj] += element_matrix[row * (NC * NS) + col];",
        "        }",
        "      }",
        "    }",
        "  }",
        "}",
        "",
    ]


def _sfem_soa_hessian_packed_crs_helper_lines(function_base, dim, n_nodes, n_field_components=None):
    n_field_components = dim if n_field_components is None else n_field_components
    return [
        "static SFEM_INLINE idx_t %s_packed_global_node(" % function_base,
        "    const uint16_t packed_node,",
        "    const ptrdiff_t pack,",
        "    const ptrdiff_t *const RSTR owned_nodes_ptr,",
        "    const ptrdiff_t *const RSTR ghost_ptr,",
        "    const idx_t *const RSTR ghost_idx) {",
        "  const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
        "  return packed_node < n_contiguous ? idx_t(owned_nodes_ptr[pack] + packed_node) : ghost_idx[ghost_ptr[pack] + packed_node - n_contiguous];",
        "}",
        "",
        "template <typename s_t>",
        "static SFEM_INLINE void %s_discover_packed_crs_entries(" % function_base,
        "    const idx_t *const RSTR ev,",
        "    const count_t *const RSTR rowptr,",
        "    const idx_t *const RSTR colidx,",
        "    count_t *const RSTR entries) {",
        kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
        kernel_constant("NS", n_nodes, indent="  "),
        kernel_constant("NDOFS", "NC * NS", indent="  "),
        "  idx_t ks[NS];",
        "  for (int i = 0; i < NS; ++i) {",
        "    const count_t row_begin = rowptr[ev[i]];",
        "    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);",
        "    const idx_t *const RSTR cols = &colidx[row_begin];",
        "    %s_find_cols(ev, cols, lenrow, ks);" % function_base,
        "    for (int j = 0; j < NS; ++j) {",
        "      const int local_col = (int)ks[j];",
        "      for (int bi = 0; bi < NC; ++bi) {",
        "        const count_t row_value_offset = row_begin * NC * NC + bi * lenrow * NC;",
        "        for (int bj = 0; bj < NC; ++bj) {",
        "          const int row = bi * NS + i;",
        "          const int col = bj * NS + j;",
        "          entries[row * NDOFS + col] = row_value_offset + local_col * NC + bj;",
        "        }",
        "      }",
        "    }",
        "  }",
        "}",
        "",
        "template <typename s_t>",
        "static SFEM_INLINE void %s_scatter_packed_crs_entries(" % function_base,
        "    const s_t *const RSTR element_matrix,",
        "    const count_t *const RSTR entries,",
        "    s_t *const RSTR values) {",
        kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
        kernel_constant("NS", n_nodes, indent="  "),
        kernel_constant("NDOFS", "NC * NS", indent="  "),
        "  for (int row = 0; row < NDOFS; ++row) {",
        "    for (int col = 0; col < NDOFS; ++col) {",
        "#pragma omp atomic update",
        "      values[entries[row * NDOFS + col]] += element_matrix[row * NDOFS + col];",
        "    }",
        "  }",
        "}",
        "",
    ]


def _sfem_soa_hessian_scatter_block_diag_sym_lines(
    function_base, dim, n_nodes, assembly=None, n_field_components=None
):
    """Accumulate only the symmetric block diagonal.

    The simplest of the six: one value stream and a reduction, both from
    ``BlockDiagSymAssemblyPlan``, and no index structure of its own because the
    node index is the block index.
    """
    # The block is the field's components, not the spatial dimension.
    n_field_components = dim if n_field_components is None else n_field_components
    assembly = BlockDiagSymAssemblyPlan() if assembly is None else assembly
    value_stream = assembly.value_stream
    _assembly_reduction_is_atomic(assembly.reduction_policy, "block-diagonal-symmetric")
    body = [
        BufferDeclNode("static constexpr int", "NC", (), expr_ref(str(n_field_components))),
        BufferDeclNode("static constexpr int", "NS", (), expr_ref(str(n_nodes))),
        BufferDeclNode("static constexpr int", "NDOFS", (), expr_ref("NC * NS")),
        BufferDeclNode(
            "static constexpr int", "SYM_DIM", (), expr_ref("(NC * (NC + 1)) / 2")
        ),
        _counting_loop(
            "i",
            0,
            expr_ref("NS"),
            [
                BufferDeclNode(
                    "s_t *const",
                    "block",
                    (),
                    expr_ref("&%s[(ptrdiff_t)ev[i] * SYM_DIM]" % value_stream),
                ),
                BufferDeclNode("int", "sym", (), expr_ref("0")),
                _counting_loop(
                    "bi",
                    0,
                    expr_ref("NC"),
                    [
                        BufferDeclNode(
                            "const int", "row", (), expr_ref("bi * NS + i")
                        ),
                        _counting_loop(
                            "bj",
                            expr_ref("bi"),
                            expr_ref("NC"),
                            [
                                BufferDeclNode(
                                    "const int", "col", (), expr_ref("bj * NS + i")
                                ),
                                ScatterNode(
                                    expr_ref("block[sym++]"),
                                    expr_ref("element_matrix[row * NDOFS + col]"),
                                    "+=",
                                    atomic=True,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
    return _print_scatter_function(
        FunctionDefNode(
            "%s_scatter_block_diag_sym" % function_base,
            params=(
                "const idx_t *const RSTR ev",
                "const s_t *const RSTR element_matrix",
                "s_t *const RSTR %s" % value_stream,
            ),
            body=tuple(body),
            qualifier="static SFEM_INLINE",
            template_params=("typename s_t",),
        )
    )



#: What each matrix format's scatter is emitted from.  Iterated in this order
#: rather than asked for one `if "<fmt>" in formats` per format, so a format the
#: plan publishes reaches this site by being in the sequence instead of by
#: someone remembering to add a branch beside the others.
_SCATTER_LINES_BY_FORMAT = {
    "bsr": _sfem_soa_hessian_scatter_bsr_lines,
    "crs": _sfem_soa_hessian_scatter_crs_lines,
    "block_diag_sym": _sfem_soa_hessian_scatter_block_diag_sym_lines,
}


def _sfem_soa_hessian_matrix_public_wrappers(
    function_base,
    implementation_name,
    dim,
    formats,
    material_parameter_names,
    uses_current,
    packed_crs_passes=(),
):
    # The ids the `FORMAT` template parameter is instantiated with.  They are
    # sparse because DIA, COO and patch were removed; the remaining three keep
    # the numbers they had, so nothing already compiled changes meaning.
    format_tags = {"crs": 0, "bsr": 1, "block_diag_sym": 6}
    lines = []
    for matrix_format in formats:
        emitted_formats = (matrix_format,)
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
    n_field_components = dim
    params = _hessian_packed_crs_public_params(
        dim,
        "s_t",
        material_parameter_names,
        uses_current,
        two_pass,
    )
    if True:
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
            fill_args.extend("u%s" % _component_name(d) for d in range(n_field_components))
        fill_args.extend(("packed_element_entries", "values"))

    def body(scalar_type, arguments):
        discover_args = cast_arguments(
            params,
            common_args + ["rowptr", "colidx", "packed_element_entries"],
            scalar_type,
        )
        spelled = []
        if two_pass:
            spelled.append(
                "  const int graph_status = sfem::codegen::%s_packed_discover_impl<%s, geom_t>(%s);"
                % (function_base, scalar_type, ", ".join(discover_args))
            )
            spelled.append("  if (graph_status != SFEM_SUCCESS) return graph_status;")
        spelled.append(
            "  return sfem::codegen::%s_packed_fill_impl<%s, geom_t>(%s);"
            % (
                function_base,
                scalar_type,
                ", ".join(cast_arguments(params, fill_args, scalar_type)),
            )
        )
        return spelled

    return runtime_typed_entry_point_lines(
        public_name,
        params,
        body,
        parameter_lines=parameter_list_lines,
    )


def _sfem_soa_hessian_matrix_public_wrapper(
    public_name,
    implementation_name,
    dim,
    matrix_format,
    format_tag,
    material_parameter_names,
    uses_current,
):
    params = _hessian_matrix_public_params(
        matrix_format,
        dim,
        "s_t",
        material_parameter_names,
        uses_current,
    )
    impl_args = _hessian_matrix_impl_args(
        matrix_format,
        dim,
        material_parameter_names,
        uses_current,
    )
    return runtime_typed_entry_point_lines(
        public_name,
        params,
        lambda scalar_type, _positional: [
            "  return sfem::codegen::%s<%s, geom_t, %d>(%s);"
            % (
                implementation_name,
                scalar_type,
                format_tag,
                ", ".join(cast_arguments(params, impl_args, scalar_type)),
            ),
        ],
        parameter_lines=parameter_list_lines,
    )


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
        "idx_t **const RSTR elements",
        "const geom_t *const *const RSTR points",
    ]
    params.extend(
        "const %s %s" % (scalar_type, parameter)
        for parameter in material_parameter_names
    )
    if uses_current:
        params.append("const ptrdiff_t u_stride")
        params.extend(
            "const %s *const RSTR u%s" % (scalar_type, _component_name(d))
            for d in range(dim)
        )
    if matrix_format in ("crs", "bsr"):
        params.extend(
            [
                "const count_t *const RSTR rowptr",
                "const idx_t *const RSTR colidx",
                "%s *const RSTR values" % scalar_type,
            ]
        )
    elif matrix_format == "dia":
        params.extend(
            [
                "const int *const RSTR diag_offsets",
                "const ptrdiff_t ndiag",
                "%s *const RSTR values" % scalar_type,
            ]
        )
    elif matrix_format == "coo":
        params.extend(
            [
                "const ptrdiff_t nnz",
                "const idx_t *const RSTR rows",
                "const idx_t *const RSTR cols",
                "%s *const RSTR values" % scalar_type,
            ]
        )
    elif matrix_format == "coo_triplet":
        params.extend(
            [
                "idx_t *const RSTR rows",
                "idx_t *const RSTR cols",
                "%s *const RSTR values" % scalar_type,
            ]
        )
    elif matrix_format == "patch":
        params.extend(
            [
                "const count_t *const RSTR rowptr",
                "const idx_t *const RSTR colidx",
                "%s *const RSTR values" % scalar_type,
            ]
        )
    elif matrix_format == "block_diag_sym":
        params.append("%s *const RSTR values" % scalar_type)
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
        argument.declaration for argument in PACKED_MESH_CORE_ARGUMENTS
    ] + [
        "const geom_t *const *const RSTR points",
    ]
    params.extend(
        "const %s %s" % (scalar_type, parameter)
        for parameter in material_parameter_names
    )
    if uses_current:
        params.append("const ptrdiff_t u_stride")
        params.extend(
            "const %s *const RSTR u%s" % (scalar_type, _component_name(d))
            for d in range(dim)
        )
    if two_pass:
        params.extend(
            [
                "const count_t *const RSTR rowptr",
                "const idx_t *const RSTR colidx",
                "count_t *const RSTR packed_element_entries",
            ]
        )
    else:
        params.append("const count_t *const RSTR packed_element_entries")
    params.append("%s *const RSTR values" % scalar_type)
    return params


def _hessian_matrix_impl_args(
    matrix_format,
    dim,
    material_parameter_names,
    uses_current,
):
    n_field_components = dim
    args = [
        "nelements",
        "nnodes",
        "elements",
        "points",
        *material_parameter_names,
    ]
    if uses_current:
        args.append("u_stride")
        args.extend("u%s" % _component_name(d) for d in range(n_field_components))
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
        ("adj", dim * dim) in names_and_sizes
        and ("det", 1) in names_and_sizes
    )


def _sfem_soa_diagnostics_header(
    work_item="lane",
    header_guard_suffix="HPP",
    inline_qualifier="SFEM_INLINE",
    define_sfem_inline=True,
    host_qualifier=None,
):
    # `unsupported_dispatch` reports a dispatch, and dispatch happens on the
    # host; every caller is an `extern "C"` launcher.  It is the one thing in
    # this header that must not be compiled for a device.
    host_qualifier = inline_qualifier if host_qualifier is None else host_qualifier
    struct_name = _sfem_soa_diagnostics_struct_name()
    guard = "SFEM_CODEGEN_KERNEL_DIAGNOSTICS_%s" % header_guard_suffix
    per_qp = "per_qp_%s" % work_item
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "#include <stddef.h>",
        "#include <cstdio>",
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
    lines.extend(kernel_status_macro_lines())
    lines.extend([
        "namespace sfem {",
        "namespace codegen {",
        "",
        "//! Reports a dispatch that has no kernel for this combination.",
        "//!",
        "//! One function rather than the five-line `std::fprintf` every",
        "//! dispatch entry point used to carry: there were 248 copies of it,",
        "//! differing only in the name they print.",
        "static %s int unsupported_dispatch(" % host_qualifier,
        "    const char *const name,",
        "    const int element_type,",
        "    const int real_type) {",
        '  std::fprintf(stderr,',
        '      "%s does not support element type %d with real type %d\\n",',
        "      name, element_type, real_type);",
        "  return SFEM_FAILURE;",
        "}",
        "",
        "struct %s {" % struct_name,
        "  const char *kernel_name;",
        "  const char *element_type;",
        "  int dim;",
        "  int n_qp;",
        "  int n_shape;",
        "  int vector_size;",
        "  int quadrature_order;",
        "  long add_instructions_%s;" % per_qp,
        "  long mul_instructions_%s;" % per_qp,
        "  long div_instructions_%s;" % per_qp,
        "  long sqrt_instructions_%s;" % per_qp,
        "  long pow_instructions_%s;" % per_qp,
        "  long exp_instructions_%s;" % per_qp,
        "  long log_instructions_%s;" % per_qp,
        "  long trig_instructions_%s;" % per_qp,
        "  long load_instructions_%s;" % per_qp,
        "  long store_instructions_%s;" % per_qp,
        "  long flops_%s;" % per_qp,
        "  long affine_mesh_flops_per_element;",
        "  long isoparametric_mesh_flops_per_element;",
        "  long temporaries;",
        "  long estimated_registers;",
        "  int geometry_streams;",
        "  int reference_scalars;",
        "  int quadrature_weight_scalars;",
        "  int material_scalars;",
        "  int u_streams;",
        "  int h_streams;",
        "  int output_streams;",
        "  int output_reads_per_element;",
        "  int output_writes_per_element;",
        "  double add_cpi;",
        "  double mul_cpi;",
        "  double div_cpi;",
        "  double sqrt_cpi;",
        "  double pow_cpi;",
        "  double exp_cpi;",
        "  double log_cpi;",
        "  double trig_cpi;",
        "  double load_cpi;",
        "  double store_cpi;",
        "};",
        "",
        "static %s double %s_total_flops(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements) {",
        "  const double n = nelements > 0 ? (double)nelements : 0.0;",
        "  return n * ((double)d->n_qp * (double)d->flops_%s + (double)d->isoparametric_mesh_flops_per_element);" % per_qp,
        "}",
        "",
        "static %s double %s_total_flops_affine_mesh(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements) {",
        "  const double n = nelements > 0 ? (double)nelements : 0.0;",
        "  return n * ((double)d->n_qp * (double)d->flops_%s + (double)d->affine_mesh_flops_per_element);" % per_qp,
        "}",
        "",
        "static %s double %s_total_flops_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements) {",
        "  const double n = nelements > 0 ? (double)nelements : 0.0;",
        "  return n * ((double)d->n_qp * (double)d->flops_%s + (double)d->isoparametric_mesh_flops_per_element);" % per_qp,
        "}",
        "",
        "static %s size_t %s_total_bytes(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t) {",
        "  const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "  const size_t geometry_bytes = n * (size_t)d->n_qp * (size_t)d->geometry_streams * scalar_bytes;",
        "  const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "  const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "  const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "  return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static %s size_t %s_total_bytes_affine_mesh(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t) {",
        "  const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "  const size_t geometry_bytes = n * (size_t)(d->dim * d->dim + 1) * scalar_bytes;",
        "  const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "  const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "  const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "  return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static %s size_t %s_total_bytes_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t) {",
        "  const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "  const size_t geometry_bytes = n * (size_t)d->dim * (size_t)d->n_shape * scalar_bytes;",
        "  const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "  const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "  const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "  return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static %s double %s_arithmetic_intensity(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t accumulator_bytes) {",
        "  const size_t bytes = %s_total_bytes(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "  return bytes ? %s_total_flops(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "static %s double %s_arithmetic_intensity_affine_mesh(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t accumulator_bytes) {",
        "  const size_t bytes = %s_total_bytes_affine_mesh(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "  return bytes ? %s_total_flops_affine_mesh(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "static %s double %s_arithmetic_intensity_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "    const %s *const d," % struct_name,
        "    const ptrdiff_t nelements,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t accumulator_bytes) {",
        "  const size_t bytes = %s_total_bytes_isoparametric_mesh(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "  return bytes ? %s_total_flops_isoparametric_mesh(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "static %s void %s_print_rate_with_ai(" % (inline_qualifier, struct_name),
        "    const char *const name,",
        "    const %s *const d," % struct_name,
        "    const double elapsed,",
        "    const ptrdiff_t nelements,",
        "    const ptrdiff_t ndofs,",
        "    const double ai,",
        "    const double total_flops) {",
        "  const double element_rate = elapsed > 0.0 ? 1e-6 * (double)nelements / elapsed : 0.0;",
        "  const double dof_rate = elapsed > 0.0 ? 1e-6 * (double)ndofs / elapsed : 0.0;",
        "  const double gflops = elapsed > 0.0",
        "      ? 1e-9 * total_flops / elapsed",
        "      : 0.0;",
        '  printf("%-72s %12.6e %16.3f %13.3f %10.3f %13.3f\\n",',
        "           name ? name : d->kernel_name,",
        "           elapsed, element_rate, dof_rate, ai, gflops);",
        "}",
        "",
        "static %s void %s_print_rate(" % (inline_qualifier, struct_name),
        "    const char *const name,",
        "    const %s *const d," % struct_name,
        "    const double elapsed,",
        "    const ptrdiff_t nelements,",
        "    const ptrdiff_t ndofs,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t accumulator_bytes) {",
        "  const double ai = %s_arithmetic_intensity(" % struct_name,
        "      d, nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "  const double total_flops = %s_total_flops(d, nelements);" % struct_name,
        "  %s_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);" % struct_name,
        "}",
        "",
        "static %s void %s_print_rate_affine_mesh(" % (inline_qualifier, struct_name),
        "    const char *const name,",
        "    const %s *const d," % struct_name,
        "    const double elapsed,",
        "    const ptrdiff_t nelements,",
        "    const ptrdiff_t ndofs,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t accumulator_bytes) {",
        "  const double ai = %s_arithmetic_intensity_affine_mesh(" % struct_name,
        "      d, nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "  const double total_flops = %s_total_flops_affine_mesh(d, nelements);" % struct_name,
        "  %s_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);" % struct_name,
        "}",
        "",
        "static %s void %s_print_rate_isoparametric_mesh(" % (inline_qualifier, struct_name),
        "    const char *const name,",
        "    const %s *const d," % struct_name,
        "    const double elapsed,",
        "    const ptrdiff_t nelements,",
        "    const ptrdiff_t ndofs,",
        "    const size_t scalar_bytes,",
        "    const size_t real_bytes,",
        "    const size_t accumulator_bytes) {",
        "  const double ai = %s_arithmetic_intensity_isoparametric_mesh(" % struct_name,
        "      d, nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "  const double total_flops = %s_total_flops_isoparametric_mesh(d, nelements);" % struct_name,
        "  %s_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);" % struct_name,
        "}",
        "",
        "} // namespace codegen",
        "} // namespace sfem",
        "",
        "#endif",
    ])
    return lines


def _element_flops_plan(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    n_field_components,
    quadrature_rule,
    affine_quadrature_rule,
    local_prefix,
    material_flops_per_qp,
):
    """The element cost this kernel's loops imply, from `plans.flops`.

    The emitter's job here is to say which kernel is being described -- which
    element, which form, which rule, and whether the affine variant is the
    closed-form simplex body -- and then to print the answer.  The arithmetic
    that composes those facts into a number belongs to the plan layer, which
    is where it now is.
    """
    affine_rule = affine_quadrature_rule or quadrature_rule
    specialized_prefix = _constant_p1_specialized_local_prefix(
        local_prefix, affine_rule
    )
    metric = geometry_variant_plan(
        form.weak_form,
        affine_rule,
        specialized=specialized_prefix is not None,
    ).cached_metric
    expanded_plan = expanded_simplex_metric_plan(
        metric,
        dim,
        n_nodes,
        affine_rule.n_qp if affine_rule is not None else n_qp,
        n_field_components,
        writes_per_shape(form),
        form_reads_current(form, default=True),
        form_reads_direction(form, default=form.has_direction),
    )
    return element_flops_plan(
        getattr(form, "name", ""),
        quadrature_rule.element_type,
        dim,
        n_qp,
        n_nodes,
        n_field_components,
        quadrature_rule,
        material_flops_per_qp,
        expanded_plan=expanded_plan,
        scale_is_unit=expanded_plan is not None and expanded_plan.scale == 1,
    )


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
    affine_quadrature_rule,
    local_prefix,
):
    n_field_components = form_n_field_components(form, dim)
    public_name = _sfem_soa_public_function_name(prefix, form.name, quadrature_rule)
    struct_name = _sfem_soa_diagnostics_struct_name()
    variable_name = "%s_diagnostics_data" % public_name
    uses_current = form_reads_current(form, default=True)
    uses_direction = form_reads_direction(form, default=form.has_direction)
    if form.expression_graph is not None:
        cost = form.expression_graph.cost
    elif form.weak_form is not None:
        diagnostic_deformation_substitutions = _weak_form_deformation_gradient_substitutions(
            form.weak_form,
            "diag_grad",
            scalar_temporaries=True,
        )
        diagnostic_expressions = _DIAGNOSTIC_EXPRESSIONS[form_accumulation(form)](
            form, diagnostic_deformation_substitutions
        )
        diagnostic_graph = (
            build_expression_graph(
                KernelExpressions()
                .add(ExpressionRole.OPERATOR_EVALUATION, diagnostic_expressions),
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
    reference_scalars, quadrature_weight_scalars = energy_reference_data_traffic(
        quadrature_rule, reference_inputs, n_qp
    )
    output_streams = len(_output_stream_names(form, n_field_components, n_nodes))
    output_reads = output_streams if output_is_accumulated(form) else 0
    output_writes = output_streams
    u_streams = dim * n_nodes if uses_current else 0
    h_streams = dim * n_nodes if uses_direction else 0
    element_type = quadrature_rule.element_type
    quadrature_order = quadrature_rule.order
    flops = _element_flops_plan(
        form,
        prefix,
        dim,
        n_nodes,
        n_qp,
        n_field_components,
        quadrature_rule,
        affine_quadrature_rule,
        local_prefix,
        cost.flops,
    )
    affine_extra_flops = flops.affine_mesh_flops_per_element
    isoparametric_extra_flops = flops.isoparametric_mesh_flops_per_element
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(
        diagnostics_record_lines(
            DiagnosticsRecord(
                public_name=public_name,
                element_type=element_type,
                dim=dim,
                n_qp=n_qp,
                n_shape=n_nodes,
                vector_size=vector_size,
                quadrature_order=quadrature_order,
                cost=cost,
                affine_mesh_flops_per_element=affine_extra_flops,
                isoparametric_mesh_flops_per_element=isoparametric_extra_flops,
                geometry_streams=geometry_streams,
                reference_scalars=reference_scalars,
                quadrature_weight_scalars=quadrature_weight_scalars,
                material_scalars=2,
                u_streams=u_streams,
                h_streams=h_streams,
                output_streams=output_streams,
                output_reads_per_element=output_reads,
                output_writes_per_element=output_writes,
            )
        )
    )
    lines.extend(
        [
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    lines.extend(diagnostics_accessor_lines(public_name))
    return lines


def _sfem_soa_diagnostics_struct_name():
    return STRUCT_NAME


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
    if is_tensor_product_family(basis_family):
        offset = 1 + _sfem_soa_element_stream_count_from_params(wrapper_params)
        arguments.insert(
            offset,
            quadrature_reference_accessor(quadrature_rule, "shape_1d", "real_t"),
        )
        arguments.insert(
            offset + 1,
            quadrature_reference_accessor(quadrature_rule, "grad_1d", "real_t"),
        )
        arguments.insert(
            offset + 2,
            quadrature_reference_accessor(quadrature_rule, "q_weight_1d", "real_t"),
        )
        return tuple(arguments)
    if use_reference_gradient_vectors:
        offset = 1 + _sfem_soa_element_stream_count_from_params(wrapper_params)
        for component in range(quadrature_rule.dim):
            arguments.insert(
                offset + component,
                quadrature_reference_accessor(quadrature_rule, _sfem_reference_gradient_vector_name(component),
                    "real_t",
                ),
            )
        arguments.insert(
            offset + quadrature_rule.dim,
            quadrature_reference_accessor(quadrature_rule, "q_weight", "real_t"),
        )
        return tuple(arguments)
    for array_input in reference_inputs:
        arguments.insert(
            1 + _sfem_soa_element_stream_count_from_params(wrapper_params),
            quadrature_reference_accessor(quadrature_rule, array_input.name, "real_t"),
        )
    arguments.insert(
        1 + _sfem_soa_element_stream_count_from_params(wrapper_params) + len(reference_inputs),
        quadrature_reference_accessor(quadrature_rule, "q_weight", "real_t"),
    )
    return tuple(arguments)


def _sfem_soa_element_stream_count_from_params(params):
    count = 0
    for param in params[1:]:
        compact = " ".join(param.replace(",", "").split())
        if (
            "*" not in compact
            and (
                compact.startswith("const s_t ")
                or compact.startswith("const real_t ")
            )
        ):
            break
        count += 1
    return count


def _sfem_soa_public_wrapper_params(params):
    return tuple(param.replace("s_t", "real_t") for param in params)


def _sfem_soa_concrete_scalar_params(params, scalar_type):
    return tuple(
        param.replace("s_t", scalar_type)
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
    n_field_components=None,
):
    n_field_components = dim if n_field_components is None else n_field_components
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    if not _sfem_soa_has_adjugate_geometry_inputs(array_inputs, dim):
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
            n_field_components=n_field_components,
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
        *reference_include_lines(
            quadrature_rule, sfem_mesh_reference_data(quadrature_rule)
        ),
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
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
    return "\n".join(resolve_dead_parameters(resolve_kernel_constants(lines)))


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
    n_field_components=None,
):
    n_field_components = dim if n_field_components is None else n_field_components
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
        output_param = "s_t *const RSTR values" if public == "energy" else "s_t *const *const RSTR out_streams"
        for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
            name = "%s_%s_%s" % (prefix, public, abi_local_level(("%s_" % suffix) if suffix else ""))
            target_name = "%s_%s_%s" % (
                alias["target_prefix"],
                public,
                abi_local_level(("%s_" % suffix) if suffix else ""),
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
                    n_field_components=n_field_components,
                    source_builder=source_builder,
                )
            )
            lines.append("")
    apply_form = forms_by_name.get("apply")
    if apply_form is not None and apply_form.weak_form is not None:
        for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
            name = "%s_hessian_%s" % (prefix, abi_local_level(("%s_" % suffix) if suffix else ""))
            target_name = "%s_hessian_%s" % (alias["target_prefix"], abi_local_level(("%s_" % suffix) if suffix else ""))
            params = _sfem_soa_element_api_common_params(apply_form, dim, include_coords)
            params.append("s_t *const *const RSTR matrix_streams")
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
                    n_field_components=n_field_components,
                    source_builder=source_builder,
                )
            )
            lines.append("")
    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(resolve_dead_parameters(resolve_kernel_constants(lines)))


def _sfem_soa_element_api_alias_function_lines(
    name,
    target_name,
    params,
    shape_order,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    n_field_components=None,
    source_builder=None,
):
    n_field_components = dim if n_field_components is None else n_field_components
    if source_builder is None:
        source_builder = _default_openmp_energy_source_builder()
    lines = [
        "template <typename s_t, int VS>",
        "static %s int %s(" % (_inline_qualifier(source_builder), name),
    ]
    lines.extend(parameter_list_lines(params))
    lines.extend(
        [
            ") {",
            kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
            kernel_constant("NS", n_nodes, indent="  "),
            kernel_constant("NQ", n_qp, indent="  "),
            kernel_constant("NDOFS", "NC * NS", indent="  "),
        ]
    )
    param_names = [_cpp_argument_name(param) for param in params]
    if "matrix_streams" in param_names:
        # The matrix variant still walks a table: flat would be NDOFS * NDOFS
        # entries -- 576 for a three-component HEX8 -- which is not leaner than
        # the loop that fills it, only longer.
        lines.append(
            "  static constexpr int SHAPE_ORDER[NS] = {%s};"
            % ", ".join(str(i) for i in shape_order)
        )
    if "coords" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines(
            "coords", "ordered_coords", "const s_t *", False, shape_order, n_field_components
        ))
    if "u_streams" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines(
            "u_streams", "ordered_u_streams", "const s_t *", False, shape_order, n_field_components
        ))
    if "out_streams" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines(
            "out_streams", "ordered_out_streams", "s_t *", False, shape_order, n_field_components
        ))
    if "matrix_streams" in param_names:
        lines.extend(_sfem_soa_element_api_alias_stream_lines("matrix_streams", "ordered_matrix_streams", "s_t *", True))

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
    lines.append("  return %s<s_t, VS>(%s);" % (target_name, ", ".join(args)))
    lines.append("}")
    return lines


def _sfem_soa_element_api_alias_stream_lines(
    source_name,
    ordered_name,
    pointer_type,
    matrix,
    shape_order=None,
    n_field_components=None,
):
    if not matrix:
        # One flat initializer, the shape the C ABI wrappers already use for
        # `proteus_elements`.  A table plus a loop said the same thing with a
        # runtime indirection in between, and this is a pointer shuffle at the
        # element API boundary, not work a kernel should be doing.
        return [
            "  %sconst %s[NDOFS] = {%s};"
            % (
                pointer_type,
                ordered_name,
                ", ".join(
                    "%s[%d]" % (source_name, source * n_field_components + component)
                    for source in shape_order
                    for component in range(n_field_components)
                ),
            )
        ]
    lines = ["  %s%s[NDOFS%s];" % (pointer_type, ordered_name, " * NDOFS" if matrix else "")]
    if matrix:
        lines.extend(
            [
                "  for (int row_shape = 0; row_shape < NS; ++row_shape) {",
                "    const int source_row_shape = SHAPE_ORDER[row_shape];",
                "    for (int row_component = 0; row_component < NC; ++row_component) {",
                "      const int row = row_shape * NC + row_component;",
                "      const int source_row = source_row_shape * NC + row_component;",
                "      for (int col_shape = 0; col_shape < NS; ++col_shape) {",
                "        const int source_col_shape = SHAPE_ORDER[col_shape];",
                "        for (int col_component = 0; col_component < NC; ++col_component) {",
                "          const int col = col_shape * NC + col_component;",
                "          const int source_col = source_col_shape * NC + col_component;",
                "          %s[row * NDOFS + col] = %s[source_row * NDOFS + source_col];" % (ordered_name, source_name),
                "        }",
                "      }",
                "    }",
                "  }",
            ]
        )
    return lines


def _sfem_soa_element_api_common_params(form, dim, include_coords):
    params = ["const ptrdiff_t nelements"]
    if include_coords:
        params.append("const s_t *const *const RSTR coords")
    else:
        params.append("const s_t *const *const RSTR adj")
        params.append("const s_t *const RSTR det")
    params.extend(_form_material_parameter_declarations(form))
    params.extend(
        "const s_t *const *const RSTR %s_streams" % stream_prefix
        for _role, stream_prefix in element_api_field_roles(form)
        if stream_prefix == "u"
    )
    return params


def _sfem_soa_element_api_reference_args(prefix, quadrature_rule, use_tensor_product_reference, use_reference_gradient_vectors):
    if use_tensor_product_reference:
        return (
            quadrature_reference_accessor(quadrature_rule, "shape_1d"),
            quadrature_reference_accessor(quadrature_rule, "grad_1d"),
            quadrature_reference_accessor(quadrature_rule, "q_weight_1d"),
        )
    if use_reference_gradient_vectors:
        return tuple(
            quadrature_reference_accessor(quadrature_rule, _sfem_reference_gradient_vector_name(component),
            )
            for component in range(quadrature_rule.dim)
        ) + (quadrature_reference_accessor(quadrature_rule, "q_weight"),)
    return (
        quadrature_reference_accessor(quadrature_rule, "grad_ref"),
        quadrature_reference_accessor(quadrature_rule, "q_weight"),
    )


def _sfem_soa_element_api_geometry_args(dim):
    return tuple("badj%d" % component for component in range(dim * dim)) + (
        "bdet0",
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
        "ne",
        "VS",
        *_sfem_soa_element_api_geometry_args(dim),
        *_sfem_soa_element_api_reference_args(
            prefix,
            quadrature_rule,
            use_tensor_product_reference,
            use_reference_gradient_vectors,
        ),
        *form_material_parameter_names(form),
    ]
    args.extend(
        "b%s_streams" % stream_prefix
        for _role, stream_prefix in element_api_field_roles(form)
    )
    args.append(output_arg)
    return "%s<s_t, NQ, NS, VS>(%s);" % (
        block_name,
        ", ".join(args),
    )


def _element_api_lane_loop(source_builder, indent):
    """The element API's work-item scope, as its target lowers it.

    Six sites here wrote `#pragma omp simd` as a literal beside a hand-written
    lane `for`, which is why the CUDA and HIP backends rejected their own
    output.  Two contract violations came from that -- an OpenMP pragma in a
    CUDA file, and vector-lane lowering in one -- and both are the same mistake:
    asserting the CPU shape instead of asking the target for its own.

    `_work_item_loop_lines` already answers this.  A target whose policy says
    `emits_lane_loop` gets the pragma and the loop; one that maps a lane to a
    thread gets a bare scope, because on a GPU the lane is the thread and there
    is nothing to iterate.  The caller closes the brace either way.
    """
    return list(_work_item_loop_lines(source_builder, indent))


def _sfem_soa_element_api_tile_setup_lines(form, dim, n_nodes, output_kind, source_builder):
    item = _work_item_index(source_builder)
    lines = [
        "    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);",
    ]
    for _role, stream_prefix in element_api_field_roles(form):
        if stream_prefix != "u":
            continue
        lines.append("    const s_t *b%s_streams[NDOFS];" % stream_prefix)
        lines.append("    for (int stream = 0; stream < NDOFS; ++stream) {")
        lines.append(
            "      b%s_streams[stream] = %s_streams[stream] + evb;"
            % (stream_prefix, stream_prefix)
        )
        lines.append("    }")
    if output_kind == "value":
        lines.append("    s_t *const bvalue = values + evb;")
        lines.extend(_element_api_lane_loop(source_builder, "    "))
        lines.append("      bvalue[%s] = s_t(0);" % item)
        lines.append("    }")
    elif output_kind == "vector":
        lines.append("    s_t *bout_streams[NDOFS];")
        lines.append("    for (int stream = 0; stream < NDOFS; ++stream) {")
        lines.append("      bout_streams[stream] = out_streams[stream] + evb;")
        lines.extend(_element_api_lane_loop(source_builder, "      "))
        lines.append("        bout_streams[stream][%s] = s_t(0);" % item)
        lines.append("      }")
        lines.append("    }")
    return lines


def _sfem_soa_element_api_geometry_tile_lines(dim, quadrature_rule, source_builder):
    item = _work_item_index(source_builder)
    lines = []
    for component in range(dim * dim):
        lines.append("    s_t badj%d[NQ * VS];" % component)
    lines.append("    s_t bdet0[NQ * VS];")
    # The element API tiles are per element, so the scope the
    # element calls for can be printed here without the shared
    # local header disagreeing with itself about it.
    lines.extend(
        quadrature_scope_lines(quadrature_rule.element_type, "    ")
    )
    # Both sides of this copy are indexed by the quadrature point and the lane,
    # and the point is fixed for the whole loop, so name both slices out here.
    for component in range(dim * dim):
        lines.append(
            "      s_t *const RSTR badj%d_q = &badj%d[q * VS];" % (component, component)
        )
        lines.append(
            "      const s_t *const RSTR adj%d_q = adj[%d] + q * nelements + evb;"
            % (component, component)
        )
    lines.append("      s_t *const RSTR bdet0_q = &bdet0[q * VS];")
    lines.append("      const s_t *const RSTR det_q = det + q * nelements + evb;")
    lines.extend(_element_api_lane_loop(source_builder, "      "))
    for component in range(dim * dim):
        lines.append(
            "        badj%d_q[%s] = adj%d_q[%s];" % (component, item, component, item)
        )
    lines.append("        bdet0_q[%s] = det_q[%s];" % (item, item))
    lines.append("      }")
    lines.append("    }")
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
        "    s_t bcoordinate_data[NDOFS][VS];",
        "    for (int stream = 0; stream < NDOFS; ++stream) {",
        *_element_api_lane_loop(source_builder, "      "),
        "        bcoordinate_data[stream][%s] = coords[stream][evb + %s];"
        % (_work_item_index(source_builder), _work_item_index(source_builder)),
        "      }",
        "    }",
    ]
    for component in range(dim * dim):
        lines.append("    s_t badj%d[NQ * VS];" % component)
    lines.append("    s_t bdet0[NQ * VS];")
    if use_tensor_product_reference:
        lines.extend(
            [
                "    s_t coordinate_grad_ref[ND * NQ * ND * VS];",
            ]
        )
        for d in range(dim):
            lines.append(
                "    tensor_gradient_contiguous<s_t, NQ, NS, VS, %d>(ne, %s, %s, bcoordinate_data, %d, coordinate_grad_ref + %s);"
                % (
                    dim,
                    quadrature_reference_accessor(quadrature_rule, "shape_1d"),
                    quadrature_reference_accessor(quadrature_rule, "grad_1d"),
                    d,
                    c_product(d, "NQ", "ND", "VS"),
                )
            )
        lines.append(
            "    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {%s};"
            % ", ".join("badj%d" % component for component in range(dim * dim))
        )
        lines.append(
            "    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);"
        )
        return lines
    if use_reference_gradient_vectors:
        for component in range(dim):
            reference_name = _sfem_reference_gradient_vector_name(component)
            lines.append(
                "    const s_t *const %s = %s;"
                % (
                    reference_name,
                    quadrature_reference_accessor(quadrature_rule, reference_name),
                )
            )
    else:
        lines.append(
            "    const s_t *const grad_ref = %s;"
            % quadrature_reference_accessor(quadrature_rule, "grad_ref")
        )
    # The element API tiles are per element, so the scope the
    # element calls for can be printed here without the shared
    # local header disagreeing with itself about it.
    lines.extend(
        quadrature_scope_lines(quadrature_rule.element_type, "    ")
    )
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
            coordinate_streams="bcoordinate_data",
        )
    )
    lines.append("    }")
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
    n_field_components = form_n_field_components(form, dim)
    output_kind = "value" if public == "energy" else "vector"
    output_param = "s_t *const RSTR values" if public == "energy" else "s_t *const *const RSTR out_streams"
    lines = []
    for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
        name = "%s_%s_%s" % (prefix, public, abi_local_level(("%s_" % suffix) if suffix else ""))
        params = _sfem_soa_element_api_common_params(form, dim, include_coords)
        params.append(output_param)
        lines.extend(
            [
                "template <typename s_t, int VS>",
                "static %s int %s(" % (_inline_qualifier(source_builder), name),
            ]
        )
        lines.extend(parameter_list_lines(params))
        lines.extend(
            [
                ") {",
                kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
                kernel_constant("NS", n_nodes, indent="  "),
                kernel_constant("NQ", n_qp, indent="  "),
                kernel_constant("NDOFS", "NC * NS", indent="  "),
                "  if (nelements <= 0) return SFEM_SUCCESS;",
                "  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {",
            ]
        )
        lines.extend(_sfem_soa_element_api_tile_setup_lines(form, dim, n_nodes, output_kind, source_builder))
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
            lines.extend(_sfem_soa_element_api_geometry_tile_lines(dim, quadrature_rule, source_builder))
        lines.append(
            "    %s"
            % _sfem_soa_element_api_block_call(
                form,
                local_prefix,
                dim,
                n_qp,
                prefix,
                quadrature_rule,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                "bvalue" if public == "energy" else "bout_streams",
                use_shared_weak_local,
            )
        )
        lines.extend(["  }", "  return SFEM_SUCCESS;", "}", ""])
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
    n_field_components = form_n_field_components(form, dim)
    lines = []
    for suffix, include_coords in (("geometry", False), ("coords", True), ("", True)):
        name = "%s_hessian_%s" % (prefix, abi_local_level(("%s_" % suffix) if suffix else ""))
        params = _sfem_soa_element_api_common_params(form, dim, include_coords)
        params.append("s_t *const *const RSTR matrix_streams")
        lines.extend(
            [
                "template <typename s_t, int VS>",
                "static %s int %s(" % (_inline_qualifier(source_builder), name),
            ]
        )
        lines.extend(parameter_list_lines(params))
        lines.extend(
            [
                ") {",
                kernel_constant("NC", n_field_components, indent="  "),
                kernel_constant("ND", dim, indent="  "),
                kernel_constant("NS", n_nodes, indent="  "),
                kernel_constant("NQ", n_qp, indent="  "),
                kernel_constant("NDOFS", "NC * NS", indent="  "),
                "  if (nelements <= 0) return SFEM_SUCCESS;",
                "  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {",
                "    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);",
            ]
        )
        for _role, stream_prefix in element_api_field_roles(form):
            if stream_prefix != "u":
                continue
            lines.append("    const s_t *b%s_streams[NDOFS];" % stream_prefix)
            lines.append(
                "    for (int stream = 0; stream < NDOFS; ++stream) "
                "b%s_streams[stream] = %s_streams[stream] + evb;"
                % (stream_prefix, stream_prefix)
            )
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
            lines.extend(_sfem_soa_element_api_geometry_tile_lines(dim, quadrature_rule, source_builder))
        lines.extend(
            [
                "    s_t bh_data[NDOFS][VS];",
                "    s_t bout_data[NDOFS][VS];",
                "    const s_t *bh_streams[NDOFS];",
                "    s_t *bout_streams[NDOFS];",
                "    for (int stream = 0; stream < NDOFS; ++stream) {",
                "      bh_streams[stream] = bh_data[stream];",
                "      bout_streams[stream] = bout_data[stream];",
                "    }",
                "    for (int col = 0; col < NDOFS; ++col) {",
                "      for (int stream = 0; stream < NDOFS; ++stream) {",
                *_element_api_lane_loop(source_builder, "        "),
                "          bh_data[stream][%s] = stream == col ? s_t(1) : s_t(0);"
                % _work_item_index(source_builder),
                "          bout_data[stream][%s] = s_t(0);" % _work_item_index(source_builder),
                "        }",
                "      }",
                "      %s" % _sfem_soa_element_api_block_call(
                    form,
                    local_prefix,
                    dim,
                    n_qp,
                    prefix,
                    quadrature_rule,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    "bout_streams",
                    use_shared_weak_local,
                ),
                "      for (int row = 0; row < NDOFS; ++row) {",
                "        s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;",
                *_element_api_lane_loop(source_builder, "        "),
                "          matrix_stream[%s] = bout_data[row][%s];"
                % (_work_item_index(source_builder), _work_item_index(source_builder)),
                "        }",
                "      }",
                "    }",
                "  }",
                "  return SFEM_SUCCESS;",
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
            "  const s_t *const %sq_weight = %s;"
            % (
                reference_prefix,
                quadrature_reference_accessor(quadrature_rule, "q_weight"),
            )
        )
        return lines
    if use_tensor_product_reference:
        for name in ("shape_1d", "grad_1d", "q_weight_1d"):
            lines.append(
                "  const s_t *const %s = %s;"
                % (
                    "%s%s" % (reference_prefix, name),
                    quadrature_reference_accessor(quadrature_rule, name),
                )
            )
        return lines
    if use_reference_gradient_vectors:
        for component in range(quadrature_rule.dim):
            reference_name = _sfem_reference_gradient_vector_name(component)
            lines.append(
                "  const s_t *const %s%s = %s;"
                % (
                    reference_prefix,
                    reference_name,
                    quadrature_reference_accessor(quadrature_rule, reference_name),
                )
            )
        lines.append(
            "  const s_t *const %sq_weight = %s;"
            % (
                reference_prefix,
                quadrature_reference_accessor(quadrature_rule, "q_weight"),
            )
        )
        return lines
    for array_input in reference_inputs:
        if array_input.name != "grad_ref":
            raise ValueError("mesh reference aliases require grad_ref")
        lines.append(
            "  const s_t *const %s%s = %s;"
            % (
                reference_prefix,
                array_input.name,
                quadrature_reference_accessor(quadrature_rule, array_input.name),
            )
        )
    lines.append(
        "  const s_t *const %sq_weight = %s;"
        % (
            reference_prefix,
            quadrature_reference_accessor(quadrature_rule, "q_weight"),
        )
    )
    return lines


def _sfem_soa_public_function_name(prefix, form_name, quadrature_rule):
    element = quadrature_rule.element_type.lower()
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_%s_soa" % (prefix, form_name)
    return "%s_%s_%s_soa" % (
        prefix,
        element,
        form_name,
    )


def _sfem_soa_isoparametric_public_function_name(prefix, form_name, quadrature_rule):
    element = quadrature_rule.element_type.lower()
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_%s_i_soa" % (prefix, form_name)
    return "%s_%s_%s_i_soa" % (
        prefix,
        element,
        form_name,
    )


def _sfem_soa_mesh_public_function_name(prefix, form_name, quadrature_rule, geometry_mode):
    element = quadrature_rule.element_type.lower()
    fragment = abi_mesh_fragment(geometry_mode)
    if _sfem_soa_prefix_has_element_suffix(prefix, element):
        return "%s_%s_%s" % (prefix, form_name, fragment)
    return "%s_%s_%s_%s" % (
        prefix,
        element,
        form_name,
        fragment,
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


#: The output parameter a block kernel declares, per body shape.  A scalar body
#: is handed one accumulator; a per-shape body is handed the stream array.
_BLOCK_OUTPUT_PARAMETER = {
    FormAccumulation.SCALAR: lambda n_field_components: "s_t *const RSTR value",
    FormAccumulation.PER_SHAPE: (
        lambda n_field_components: "s_t *const RSTR out_streams[NS * %d]"
        % n_field_components
    ),
}

#: The per-point buffers a body needs beyond its gradients.  A scalar body forms
#: no loperand, so the sequence is empty and nothing is declared.
_POINT_BUFFERS = {
    FormAccumulation.SCALAR: (),
    FormAccumulation.PER_SHAPE: ("loperand_q",),
}

#: The same, per component, for the bodies that keep the loperand in lane-major
#: scalars rather than one block.
_LOPERAND_COMPONENTS = {
    FormAccumulation.SCALAR: lambda n_field_components, dim: (),
    FormAccumulation.PER_SHAPE: (
        lambda n_field_components, dim: range(n_field_components * dim)
    ),
}


def _scalar_diagnostic_expressions(form, substitutions):
    """A 0-form's cost is its energy density."""
    return (form.weak_form.energy_density.xreplace(substitutions),)


def _per_shape_diagnostic_expressions(form, substitutions):
    """A 1- or 2-form's cost is the material expression it contracts."""
    return tuple(
        _weak_form_material_expression(
            form.weak_form,
            form.name,
            substitutions,
            tuple(
                sp.symbols("diag_trial_grad%d" % i)
                for i in range(form.weak_form.n_field_components * form.weak_form.dim)
            ),
        )
    )


#: What the diagnostics cost model counts, per body shape.  It has to be the
#: expressions the body actually evaluates, which is the same fact the body
#: emitters read -- a record that counted the wrong half would be wrong in a
#: way nothing compiles against.
_DIAGNOSTIC_EXPRESSIONS = {
    FormAccumulation.SCALAR: _scalar_diagnostic_expressions,
    FormAccumulation.PER_SHAPE: _per_shape_diagnostic_expressions,
}


def _zero_fill_lines(output, source_builder, work_item, indent="    "):
    """Clear the block's output staging, whatever shape the plan gives it.

    `MeshOutputShape.extents` is the buffer's extents inside the lane: empty for
    a scalar accumulator, one entry for a per-shape output.  Emission opens a
    loop per extent and the lane loop inside them, so the two shapes are the
    same text at two depths rather than two texts.
    """
    lines = []
    subscripts = ""
    for depth, extent in enumerate(output.extents):
        index = "stream" if depth == 0 else "stream%d" % depth
        lines.append(
            "%sfor (int %s = 0; %s < %s; ++%s) {"
            % (indent + "  " * depth, index, index, extent, index)
        )
        subscripts += "[%s]" % index
    inner = indent + "  " * len(output.extents)
    lines.extend(_work_item_loop_lines(source_builder, inner))
    lines.append(
        "%s  %s%s[%s] = s_t(0);" % (inner, output.block, subscripts, work_item)
    )
    lines.append("%s}" % inner)
    lines.extend(
        "%s}" % (indent + "  " * depth)
        for depth in reversed(range(len(output.extents)))
    )
    return lines


#: Whether the mesh kernel's block call sits inside a quadrature loop, and
#: everything that follows from it.  `plans.form_emission.form_contraction`
#: decides: a pointwise form is contracted at each point and its block is called
#: inside a loop this file opens, indents and closes; a deferred-flux form is
#: handed the whole element and opens nothing.
#:
#: Six sites in `_sfem_soa_mesh_operator_function` asked `form.weak_form is
#: None` separately -- the buffer extent, the loop, the geometry argument, the
#: weight, the indent and the closing brace.  `form_contraction`'s own docstring
#: named this function as where eight of its twenty-three copies lived.
_BLOCK_BUFFER_EXTENT = {
    FormContraction.DEFERRED_FLUX: "NQ * VS",
    FormContraction.POINTWISE: "VS",
}

_MESH_BLOCK_CALL_INDENT = {
    FormContraction.DEFERRED_FLUX: "    ",
    FormContraction.POINTWISE: "      ",
}

_MESH_QUADRATURE_SCOPE_CLOSE = {
    FormContraction.DEFERRED_FLUX: (),
    FormContraction.POINTWISE: ("    }",),
}


def _pointwise_mesh_quadrature_scope(quadrature_rule, dim, use_tensor_product_reference,
                                     tensor_weight_name):
    lines = ["", *quadrature_scope_lines(quadrature_rule.element_type, "    ")]
    if use_tensor_product_reference:
        lines.extend(tensor_product_q_index_lines(dim, "      "))
        lines.append(
            "      const s_t tensor_q_weight = %s;"
            % tensor_product_quadrature_weight_expr(dim, tensor_weight_name)
        )
    return lines


_MESH_QUADRATURE_SCOPE = {
    FormContraction.DEFERRED_FLUX: (
        lambda rule, dim, tensor, weight_name: []
    ),
    FormContraction.POINTWISE: _pointwise_mesh_quadrature_scope,
}

_MESH_GEOMETRY_CALL_ARGUMENT = {
    FormContraction.DEFERRED_FLUX: (
        lambda geometry_mode: "0" if geometry_mode == "affine" else "VS"
    ),
    FormContraction.POINTWISE: lambda geometry_mode: "q",
}

_MESH_WEIGHT_CALL_ARGUMENT = {
    FormContraction.DEFERRED_FLUX: (
        lambda tensor, tensor_name, scalar_name: tensor_name if tensor else scalar_name
    ),
    FormContraction.POINTWISE: (
        lambda tensor, tensor_name, scalar_name: "tensor_q_weight"
        if tensor
        else "%s[q]" % scalar_name
    ),
}


#: How the two output shapes cross the C boundary.  A scalar is one pointer; a
#: per-shape output is one pointer per field component beside its stride.  The
#: shape is `plans.form_emission.mesh_output`; this is only the spelling.
_OUTPUT_ABI_BUFFERS = {
    False: lambda n_field_components, component_name: ("value",),
    True: lambda n_field_components, component_name: tuple(
        "out%s" % component_name(d) for d in range(n_field_components)
    ),
}

#: What the block kernel is handed.  The scalar accumulator is one buffer; the
#: per-shape output is the pointer array built above it.
_OUTPUT_CALL_ARGUMENT = {False: "bvalue", True: "bout_streams"}


def _output_stream_names(form, n_field_components, n_nodes):
    """The streams this kernel writes, named by `plans.form_emission`."""
    return mesh_output_streams(form, n_field_components, n_nodes, _component_name)
