"""The node-centric merit kernel's two loops, held to what makes them worth it.

The arrangement only pays if loop 1 really does take the step-independent work
out of loop 2, and if the two loops vectorise on different axes without nesting.
Those are structural properties of the emitted text, so they are checked on it.
"""
import pytest

from codegen.framework.emitters import residual_codegen as rc
from codegen.framework.emitters.patch_merit_codegen import (
    patch_loop_one_buffers,
    patch_loop_one_lines,
    patch_loop_two_lines,
    patch_merit_kernel_is_available,
    patch_reduction_lines,
)
from codegen.framework.fem.reference import sfem_element_quadrature_rule
from codegen.framework.forms.residual import (
    CoupledResidualSystem,
    coupled_residual_weak_coefficients,
)
from codegen.framework.plans.dependencies import (
    residual_codegen_dependencies,
    stepped_residual_dependencies,
)

DIM = 3


def _fixture():
    """A residual whose flux is nonlinear in both the value and the gradient, so
    neither can be hoisted out of the step loop by accident."""
    system = CoupledResidualSystem(DIM)
    u = system.add_field("u", previous=False)
    system.add_residual(
        u,
        sum(
            (1 + u.value ** 2) * u.gradient[d] * u.test_gradient[d]
            for d in range(DIM)
        ),
    )
    coefficients = coupled_residual_weak_coefficients(system, False)
    dependencies = stepped_residual_dependencies(
        residual_codegen_dependencies(
            system, coefficients, system.residual_dependencies()
        )
    )
    return system, coefficients, dependencies


def _loop_two():
    system, coefficients, dependencies = _fixture()
    material = rc._print_statement_nodes(
        rc._coefficient_evaluation_nodes(system, coefficients, dependencies), ""
    )
    return "\n".join(
        patch_loop_two_lines(system, coefficients, dependencies, material)
    )


def _loop_one():
    system, _, dependencies = _fixture()
    return "\n".join(
        patch_loop_one_lines(system, sfem_element_quadrature_rule("TET4"), dependencies)
    )


def test_loop_two_touches_no_geometry():
    """The map happens in loop 1, so the step loop reads no adjugate and no
    determinant.  If either appears here the split has failed and the geometry
    is being carried through the step count."""
    text = _loop_two()
    assert "adjugate" not in text
    assert "determinant" not in text
    assert "elements[" not in text


def test_loop_two_reads_loop_one_by_the_element_lane():
    """Loop 2's own lane is the trial step, so every buffer loop 1 filled must
    be indexed by the element lane instead.  Indexing them by `lane` would take
    one element's value for another's -- a wrong answer that still runs."""
    _, _, dependencies = _fixture()
    for line in _loop_two().splitlines():
        for buffer_name, _ in patch_loop_one_buffers(DIM, 1, dependencies):
            if buffer_name in line:
                assert "+ lane_e]" in line, line


def test_the_step_lane_carries_alpha_and_the_accumulator():
    text = _loop_two()
    assert "const s_t alpha = steps[lane];" in text
    assert "rho[0 * VS + lane] +=" in text


def test_the_loops_do_not_nest():
    """Two vector loops, never one inside the other: the hardware has no such
    arrangement, and a lane loop containing a loop stops being innermost."""
    for text in (_loop_one(), _loop_two()):
        depth = 0
        for line in text.splitlines():
            if "#pragma omp simd" in line:
                depth += 1
        assert depth == 1, text


def test_loop_one_reads_the_element_through_its_orientation():
    """Without this the node sits at a different local slot per lane and the
    contraction in loop 2 is against a different basis function each time."""
    text = _loop_one()
    assert "pm_orientation[pm_local_node[lane]]" in text
    assert "elements[perm[j]][element]" in text


def test_the_square_is_taken_per_step_after_the_elements_close():
    text = "\n".join(patch_reduction_lines(_fixture()[0]))
    assert "for (int lane = 0; lane < nsteps; ++lane)" in text
    assert "merit[lane] += s_t(0.5) * squared;" in text


def test_the_kernel_is_gated_on_having_an_orientation():
    assert patch_merit_kernel_is_available("TET4")
    assert not patch_merit_kernel_is_available("HEX8")


def _kernel():
    system, coefficients, dependencies = _fixture()
    material = rc._print_statement_nodes(
        rc._coefficient_evaluation_nodes(system, coefficients, dependencies), ""
    )
    from codegen.framework.emitters.patch_merit_codegen import patch_merit_kernel_lines

    return "\n".join(
        patch_merit_kernel_lines(
            system,
            sfem_element_quadrature_rule("TET4"),
            coefficients,
            dependencies,
            "TET4",
            "demo_merit_patch",
            material,
        )
    )


def test_the_kernel_outputs_only_the_step_scalars():
    """No residual vector, no node accumulator, no scatter.  The only writable
    output is `merit`, which is `nsteps` long."""
    text = _kernel()
    assert "s_t *const RSTR merit" in text
    writable = [
        line
        for line in text.splitlines()
        if line.strip().startswith("s_t *const RSTR")
        or line.strip().startswith("void *const RSTR")
    ]
    assert writable == ["    s_t *const RSTR merit"], writable


def test_the_node_residual_is_seeded_from_the_accumulator():
    """Otherwise the square is over this operator's share rather than the whole
    residual, and the merit is not zero at the solution of the system."""
    text = _kernel()
    assert "rho[c * VS + lane] = accumulator[node * NC + c];" in text


def test_the_element_loop_closes_before_the_square():
    """The square is only legal once every incident element has contributed."""
    text = _kernel()
    square = text.index("The node is finished")
    block_loop = text.index("for (count_t block = begin;")
    close = text.index("      }\n\n", block_loop)
    assert block_loop < close < square


def test_threads_reduce_once_each():
    text = _kernel()
    assert "merit_local[VS]" in text
    assert text.count("#pragma omp atomic update") == 1


def test_the_incident_buffer_holds_element_indices():
    assert "element_idx_t pm_incident[VS];" in _kernel()


def test_a_gradient_only_form_stages_no_value():
    """The staged sequence is the plan's answer, and both loops walk it.

    A flux that never reads the field value must produce no value buffer and no
    value combination -- otherwise loop 1 fills something loop 2 ignores, which
    is dead traffic proportional to the incident count.
    """
    from codegen.framework.emitters.patch_merit_codegen import patch_loop_one_buffers

    system = CoupledResidualSystem(DIM)
    u = system.add_field("u", previous=False)
    system.add_residual(
        u, sum(u.gradient[d] * u.test_gradient[d] for d in range(DIM))
    )
    coefficients = coupled_residual_weak_coefficients(system, False)
    dependencies = stepped_residual_dependencies(
        residual_codegen_dependencies(
            system, coefficients, system.residual_dependencies()
        )
    )
    material = rc._print_statement_nodes(
        rc._coefficient_evaluation_nodes(system, coefficients, dependencies), ""
    )
    text = "\n".join(
        patch_loop_two_lines(system, coefficients, dependencies, material)
    )
    names = [name for name, _ in patch_loop_one_buffers(DIM, 1, dependencies)]
    assert "pm_state_value" not in names
    assert "pm_state_value" not in text
