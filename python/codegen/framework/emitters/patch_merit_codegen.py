"""The node-centric merit kernel: one node, two vector loops.

An element-centric traversal computes a flux once per element and contracts it
against every test function, then scatters `n_shape` node contributions.  That
is the right shape for an assembly and the wrong one for a merit, because a
squared nodal residual may not be taken until the node's sum is finished, and a
scatter leaves it unfinished until the whole mesh has been walked.

This kernel inverts the traversal.  It takes one node, sums over the elements
incident on it, and squares the result while it is still in registers -- so
there is no accumulator array, no scatter, and nothing whose size grows with the
number of sampled step lengths.

What it costs is that an element is visited once per node it carries, so the
constitutive law runs about `n_shape` times where the element-centric pass runs
it once.  Which of the two wins is measured rather than argued; see the merit
document's cost model.

## Why the body is two loops and not one nest

Both axes -- the elements incident on the node, and the sampled step lengths --
are vectorised, and neither contains the other.  A lane loop with another loop
inside it is no longer innermost and does not vectorise at all, which is the
measured fact `ObjectiveStepLoop` records for the stepped objective.  So the
first loop puts the incident elements in its lanes and precomputes everything
the step length does not change, and the second puts the step lengths in its
lanes and reads what the first stored.

The step count carries the lanes because it is the axis the caller controls: a
line search chooses how many trial steps to sample, so it is rounded to the
vector width and the lanes are always full and identical for every node.  The
number of elements incident on a node is a property of the mesh and cannot be
chosen.
"""

from codegen.framework.plans.residual_structure import (
    patch_merit_staged_quantities,
    patch_merit_staged_roles,
)
from codegen.framework.fem.patch_orientation import (
    patch_orientation_permutations,
    supports_patch_orientation,
)


#: What loop 1 computes per incident element and quadrature point, for loop 2
#: to read.  Each entry is a buffer name and how many scalars it carries.
#:
#: `state_grad` and `direction_grad` are *physical* gradients.  Mapping them
#: separately here, rather than combining in reference space and mapping once in
#: loop 2, is what the split is for: the map is linear, so `J^-T (G + a H)` and
#: `J^-T G + a J^-T H` are the same number, and the first form would drag the
#: adjugate and the determinant into the step loop while the second leaves loop
#: 2 with no geometry at all.  Loop 2 then reads four buffers and touches
#: nothing else.
#:
#: `test_grad` is the physical gradient of the *fixed* basis function -- the one
#: the orientation permutation put at local slot 0 -- which is why it belongs
#: here at all rather than inside the step loop.
#: How each staged role is spelled.  `gathers` are the kernel arguments loop 1
#: reads for it, `buffers` the prefix its loop-1 buffers take, and `suffix` what
#: loop 2 calls the symbol it defines -- the same suffix the element-centric
#: bodies use, so a material's expressions need no translation.
#:
#: `current` carries two gathers because the trial step combines it with the
#: direction.  `previous` carries one: a history is fixed within the step.
_ROLE_SPELLING = {
    "current": {
        "gathers": (("state", "x"), ("direction", "h")),
        "buffers": ("pm_state", "pm_direction"),
        "suffix": "",
    },
    "previous": {
        "gathers": (("previous", "p"),),
        "buffers": ("pm_previous",),
        "suffix": "_old",
    },
}


#: The two buffers a staged quantity needs, and how wide each is.  A quantity
#: that a form does not read contributes no entry, which is the plan's answer
#: rather than a test here: see `patch_merit_staged_quantities`.
#: How wide one staged quantity is, per field component.
_QUANTITY_WIDTH = {
    "value": lambda dim, nc: nc,
    "gradient": lambda dim, nc: nc * dim,
}

#: What loop 1 names the buffer holding a quantity.
_QUANTITY_BUFFER_SUFFIX = {"value": "_value", "gradient": "_grad"}


def patch_loop_one_buffers(dim, n_field_components, dependencies):
    buffers = [("pm_test_grad", dim), ("pm_weight", 1)]
    for role in patch_merit_staged_roles(dependencies):
        for prefix in _ROLE_SPELLING[role]["buffers"]:
            for quantity in patch_merit_staged_quantities(dependencies):
                buffers.append(
                    (
                        prefix + _QUANTITY_BUFFER_SUFFIX[quantity],
                        _QUANTITY_WIDTH[quantity](dim, n_field_components),
                    )
                )
    return tuple(buffers)


def patch_merit_kernel_is_available(element_type):
    """Whether a node-centric merit kernel can be emitted for this element.

    Gated on the orientation, because the whole arrangement rests on the node
    being presentable at local slot 0 by a permutation that does not reverse the
    element.  Only the affine simplices have one today.
    """
    return supports_patch_orientation(element_type)


def patch_orientation_table_lines(element_type, name="pm_orientation"):
    """The permutation table, emitted as a literal.

    It is a property of the element rather than of the mesh, so it is a constant
    in the generated source and not an argument: `n_shape * n_shape` small
    integers that the compiler can see through.
    """
    table = patch_orientation_permutations(element_type)
    n_shape = len(table)
    rows = ", ".join(
        "{%s}" % ", ".join(str(index) for index in permutation)
        for permutation in table
    )
    return [
        "static constexpr int %s[%d][%d] = {%s};" % (name, n_shape, n_shape, rows),
    ]


def _buffer_index(offset, count, lane="lane"):
    """`(q * count + offset) * VS + <element lane>`.

    The lane axis of every loop-1 buffer is the *element*, in both loops.  That
    is worth being explicit about, because loop 2's own lane variable is the
    trial step: reading these buffers with the step index would take one
    element's value for another's, which is why the lane is a parameter here
    rather than the word `lane` written into the format string.
    """
    return "(q * %d + %s) * VS + %s" % (count, offset, lane)


def _stage_value_lines(indent, name, dim, n_fields):
    target = "pm_%s_value" % name
    return [
        "%s    // interpolated %s value" % (indent, name),
        "%s    for (int c = 0; c < NC; ++c) {" % indent,
        "%s      s_t acc = s_t(0);" % indent,
        "%s      for (int j = 0; j < NS; ++j) {" % indent,
        "%s        acc += %s[j * NC + c] * shape[q * NS + j];" % (indent, name),
        "%s      }" % indent,
        "%s      %s[%s] = acc;" % (indent, target, _buffer_index("c", n_fields)),
        "%s    }" % indent,
    ]


def _stage_gradient_lines(indent, name, dim, n_fields):
    target = "pm_%s_grad" % name
    return [
        "%s    // physical gradient of the %s: summed over shape functions," % (indent, name),
        "%s    // then mapped.  Mapped here and not in loop 2 because the map" % indent,
        "%s    // is linear and does not depend on the step length." % indent,
        "%s    for (int c = 0; c < NC; ++c) {" % indent,
        "%s      for (int d = 0; d < ND; ++d) {" % indent,
        "%s        s_t mapped = s_t(0);" % indent,
        "%s        for (int k = 0; k < ND; ++k) {" % indent,
        "%s          s_t acc = s_t(0);" % indent,
        "%s          for (int j = 0; j < NS; ++j) {" % indent,
        "%s            acc += %s[j * NC + c] * grad_ref[k][q * NS + j];" % (indent, name),
        "%s          }" % indent,
        "%s          mapped += acc * adjugate[k * ND + d][element];" % indent,
        "%s        }" % indent,
        "%s        %s[%s] = mapped / det;"
        % (indent, target, _buffer_index("c * ND + d", n_fields * dim)),
        "%s      }" % indent,
        "%s    }" % indent,
    ]


#: How loop 1 fills each staged quantity's pair of buffers.  Walked in the
#: plan's order, so the buffers exist in the order they are declared.
_STAGE_QUANTITY = {
    "value": _stage_value_lines,
    "gradient": _stage_gradient_lines,
}


def patch_loop_one_lines(system, rule, dependencies, indent="  "):
    """Everything the trial step does not change, per incident element.

    Lanes are the elements incident on the node.  The element is read through
    its orientation permutation, so slot 0 is the node being visited and the
    basis function contracted at the end of loop 2 is always `phi_0`.
    """
    dim = system.dim
    n_fields = len(system.fields)
    lines = [
        "%s// loop 1 -- lanes are the elements incident on this node." % indent,
        "%sfor (int q = 0; q < NQ; ++q) {" % indent,
        "%s  #pragma omp simd" % indent,
        "%s  for (int lane = 0; lane < ne; ++lane) {" % indent,
        "%s    const idx_t element = pm_incident[lane];" % indent,
        "%s    const int *const RSTR perm = pm_orientation[pm_local_node[lane]];" % indent,
    ]
    # The gather reads geometry and state through the same permutation, so the
    # two cannot disagree about which vertex is which.
    gathers = [
        pair
        for role in patch_merit_staged_roles(dependencies)
        for pair in _ROLE_SPELLING[role]["gathers"]
    ]
    lines.extend(
        "%s    s_t %s[%d];" % (indent, name, n_fields * rule.n_shape)
        for name, _ in gathers
    )
    lines.extend(
        [
            "%s    for (int j = 0; j < NS; ++j) {" % indent,
            "%s      const idx_t node = elements[perm[j]][element];" % indent,
            "%s      for (int c = 0; c < NC; ++c) {" % indent,
        ]
    )
    lines.extend(
        "%s        %s[j * NC + c] = %s[node * NC + c];" % (indent, name, source)
        for name, source in gathers
    )
    lines.extend(["%s      }" % indent, "%s    }" % indent])
    lines.extend(
        [
            "%s    // The geometry is indexed by the element, not by a block" % indent,
            "%s    // offset: a node's incident elements are scattered through" % indent,
            "%s    // the mesh rather than contiguous.  It carries no quadrature" % indent,
            "%s    // index either, because the orientation gate has already" % indent,
            "%s    // restricted this kernel to affine simplices, whose Jacobian" % indent,
            "%s    // is constant over the cell." % indent,
            "%s    const s_t det = determinant[element];" % indent,
        ]
    )
    for quantity in patch_merit_staged_quantities(dependencies):
        for name, _ in gathers:
            lines.extend(
                _STAGE_QUANTITY[quantity](indent, name, dim, n_fields)
            )
    lines.extend(
        [
            "%s    // the fixed basis function's physical gradient, and the" % indent,
            "%s    // integration weight.  Both are what the orientation buys:" % indent,
            "%s    // `phi_0` is the same function in every lane and at every" % indent,
            "%s    // step, so this leaves the step loop entirely." % indent,
            "%s    for (int d = 0; d < ND; ++d) {" % indent,
            "%s      s_t mapped = s_t(0);" % indent,
            "%s      for (int k = 0; k < ND; ++k) {" % indent,
            "%s        mapped += grad_ref[k][q * NS + 0] * adjugate[k * ND + d][element];"
            % indent,
            "%s      }" % indent,
            "%s      pm_test_grad[%s] = mapped / det;" % (indent, _buffer_index("d", dim)),
            "%s    }" % indent,
            "%s    pm_weight[q * VS + lane] = q_weight[q] * det;" % indent,
            "%s  }" % indent,
            "%s}" % indent,
        ]
    )
    return lines


def _combine_value_lines(indent, field, field_index, dim, n_fields, components, role):
    spelling = _ROLE_SPELLING[role]
    source = _buffer_index("%d" % field_index, n_fields, lane="lane_e")
    reads = " + alpha * ".join(
        "%s_value[%s]" % (prefix, source) for prefix in spelling["buffers"]
    )
    return [
        "%s      const s_t %s%s = %s;"
        % (indent, field.name, spelling["suffix"], reads)
    ]


def _combine_gradient_lines(indent, field, field_index, dim, n_fields, components, role):
    spelling = _ROLE_SPELLING[role]
    lines = []
    for d in range(dim):
        source = _buffer_index(
            "%d" % (field_index * dim + d), components, lane="lane_e"
        )
        reads = " + alpha * ".join(
            "%s_grad[%s]" % (prefix, source) for prefix in spelling["buffers"]
        )
        lines.append(
            "%s      const s_t %s%s_grad_%d = %s;"
            % (indent, field.name, spelling["suffix"], d, reads)
        )
    return lines


#: `grad(x + alpha h) = grad x + alpha grad h`, and the same for the value.
#: One entry per staged quantity, walked in the plan's order so loop 2 combines
#: exactly what loop 1 staged.
_STAGED_COMBINATION = {
    "value": _combine_value_lines,
    "gradient": _combine_gradient_lines,
}


def patch_loop_two_lines(system, coefficients, dependencies, material_lines,
                         indent="  "):
    """What the step length changes, and nothing else.

    Lanes are the sampled step lengths.  The elements are walked serially here
    because they carried the lanes in loop 1 and the two loops must not nest --
    a lane loop with another loop inside it is no longer innermost and does not
    vectorise.

    The body reads the four buffers loop 1 filled and touches no geometry, no
    connectivity and no mesh array: the combination is an FMA per component, the
    constitutive law follows, and the contraction is against the one basis
    function the orientation fixed.  `rho` stays in registers -- it is the
    node's residual at each step, and it is squared as soon as the element loop
    closes.
    """
    dim = system.dim
    n_fields = len(system.fields)
    components = n_fields * dim
    lines = [
        "%s// loop 2 -- lanes are the sampled step lengths." % indent,
        "%sfor (int lane_e = 0; lane_e < ne; ++lane_e) {" % indent,
        "%s  for (int q = 0; q < NQ; ++q) {" % indent,
        "%s    #pragma omp simd" % indent,
        "%s    for (int lane = 0; lane < nsteps; ++lane) {" % indent,
        "%s      const s_t alpha = steps[lane];" % indent,
    ]
    # `grad(x + alpha h) = grad x + alpha grad h`, one fused multiply-add per
    # component.  This is the whole of the affine identity at the point the
    # state enters the arithmetic.
    # `current` joins its direction under alpha; `previous` has no direction to
    # join, so the same table produces a plain read for it.  The join is the
    # role's buffer list, not a test on which role this is.
    for role in patch_merit_staged_roles(dependencies):
        for quantity in patch_merit_staged_quantities(dependencies):
            for field_index, field in enumerate(system.fields):
                lines.extend(
                    _STAGED_COMBINATION[quantity](
                        indent, field, field_index, dim, n_fields, components, role
                    )
                )
    lines.extend("%s      %s" % (indent, line) for line in material_lines)
    lines.append(
        "%s      const s_t weight = pm_weight[q * VS + lane_e];" % indent
    )
    for row in range(n_fields):
        terms = " + ".join(
            "grad_coeff%d_%d * pm_test_grad[%s]"
            % (row, d, _buffer_index("%d" % d, dim, lane="lane_e"))
            for d in range(dim)
        )
        lines.append(
            "%s      rho[%d * VS + lane] += weight * (%s);" % (indent, row, terms)
        )
    lines.extend(
        [
            "%s    }" % indent,
            "%s  }" % indent,
            "%s}" % indent,
        ]
    )
    return lines


def patch_reduction_lines(system, indent="  "):
    """Square the node's residual and add it to the thread's per-step totals.

    Legal exactly here: the element loop has closed, so by the node-centric
    identity the node's value is complete.  Nothing is written to memory -- the
    residual was never in memory -- and the output is the `m` scalars this
    whole kernel exists to produce.
    """
    n_fields = len(system.fields)
    return [
        "%s// The node is finished, so it may be squared." % indent,
        "%s#pragma omp simd" % indent,
        "%sfor (int lane = 0; lane < nsteps; ++lane) {" % indent,
        "%s  s_t squared = s_t(0);" % indent,
    ] + [
        "%s  squared += rho[%d * VS + lane] * rho[%d * VS + lane];" % (indent, c, c)
        for c in range(n_fields)
    ] + [
        "%s  merit[lane] += s_t(0.5) * squared;" % indent,
        "%s}" % indent,
    ]


def patch_merit_kernel_parameters(system, rule, dependencies, parameters=()):
    """The kernel's parameter list, in ABI order.

    Returned separately from the body so the entry point over it can be built
    from the same sequence rather than a second copy of the list -- the two
    disagreeing is a call that compiles and passes the wrong pointer.
    """
    dim = system.dim
    params = [
        "const ptrdiff_t n_owned_nodes",
        "const count_t *const RSTR n2e_ptr",
        "const element_idx_t *const RSTR n2e_idx",
        "const uint8_t *const RSTR n2e_local",
        "idx_t **const RSTR elements",
        "const s_t *const RSTR determinant",
        "const s_t *const RSTR adjugate[%d]" % (dim * dim),
        "const s_t *const RSTR shape",
        "const s_t *const RSTR grad_ref[%d]" % dim,
        "const s_t *const RSTR q_weight",
    ]
    params.extend("const s_t %s" % parameter for parameter in parameters)
    params.extend(["const int nsteps", "const s_t *const RSTR steps"])
    params.extend(
        "const s_t *const RSTR %s" % source
        for role in patch_merit_staged_roles(dependencies)
        for _, source in _ROLE_SPELLING[role]["gathers"]
    )
    params.extend(["const s_t *const RSTR accumulator", "s_t *const RSTR merit"])
    return tuple(params)


def patch_merit_argument_names(system, rule, dependencies, parameters=()):
    """The same sequence as names, for the entry point to forward."""
    names = []
    for param in patch_merit_kernel_parameters(system, rule, dependencies, parameters):
        name = param.split("[")[0].split()[-1].lstrip("*")
        names.append(name)
    return tuple(names)


def patch_merit_kernel_lines(system, rule, coefficients, dependencies,
                             element_type, function_name, material_lines,
                             parameters=()):
    """The whole kernel: threads over nodes, two vector loops inside each.

    The output is `nsteps` scalars and nothing else.  There is no residual
    vector, no node accumulator and no scatter: a node's residual lives in
    `rho` for the length of its element loop and is squared when that loop
    closes, which is legal exactly because the loop ran over every element
    incident on the node.

    `accumulator` carries what the operators that do not move with the state
    contribute -- tractions, body force, constraints -- gathered per node rather
    than assembled per step.  It seeds `rho`, so the square is taken over the
    whole residual and not just this operator's share.
    """
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    buffers = patch_loop_one_buffers(dim, n_fields, dependencies)
    declared = patch_merit_kernel_parameters(system, rule, dependencies, parameters)
    lines = [
        "template <typename s_t, int NQ, int NS, int VS>",
        "static int %s(" % function_name,
    ]
    lines.extend(
        "    %s%s" % (param, "," if index + 1 < len(declared) else "")
        for index, param in enumerate(declared)
    )
    lines.extend(
        [
            ") {",
            "  static constexpr int ND = %d;" % dim,
            "  static constexpr int NC = %d;" % n_fields,
            "",
            "#pragma omp parallel",
            "  {",
            "    // Per thread, and never larger than the vector width: the",
            "    // caller rounds the step count to it, which is the whole",
            "    // reason the steps carry the lanes.",
            "    s_t merit_local[VS];",
            "    for (int lane = 0; lane < VS; ++lane) merit_local[lane] = s_t(0);",
            "    s_t rho[NC * VS];",
        ]
    )
    lines.extend(
        "    s_t %s[NQ * %d * VS];" % (name, count) for name, count in buffers
    )
    lines.extend(
        [
            "    element_idx_t pm_incident[VS];",
            "    uint8_t pm_local_node[VS];",
            "",
            "#pragma omp for schedule(static)",
            "    for (ptrdiff_t node = 0; node < n_owned_nodes; ++node) {",
            "      // Seed with everything that does not move with the state, so",
            "      // the square below is over the whole residual.",
            "      for (int c = 0; c < NC; ++c) {",
            "        for (int lane = 0; lane < VS; ++lane) {",
            "          rho[c * VS + lane] = accumulator[node * NC + c];",
            "        }",
            "      }",
            "      const count_t begin = n2e_ptr[node];",
            "      const count_t end = n2e_ptr[node + 1];",
            "      for (count_t block = begin; block < end; block += VS) {",
            "        const int ne = (int)MIN((count_t)VS, end - block);",
            "        for (int lane = 0; lane < ne; ++lane) {",
            "          pm_incident[lane] = n2e_idx[block + lane];",
            "          pm_local_node[lane] = n2e_local[block + lane];",
            "        }",
        ]
    )
    lines.extend(patch_loop_one_lines(system, rule, dependencies, indent="        "))
    lines.extend(
        patch_loop_two_lines(
            system, coefficients, dependencies, material_lines, indent="        "
        )
    )
    lines.extend(["      }", ""])
    lines.extend(
        "      %s" % line for line in patch_reduction_lines(system, indent="")
    )
    lines.extend(
        [
            "    }",
            "",
            "    // One reduction per thread, not one per node.",
            "    for (int lane = 0; lane < nsteps; ++lane) {",
            "#pragma omp atomic update",
            "      merit[lane] += merit_local[lane];",
            "    }",
            "  }",
            "  return SFEM_SUCCESS;",
            "}",
        ]
    )
    return lines
