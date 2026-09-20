"""The material's whole residual is one 1-form, whatever its units are written in.

A residual merit has to square a node's *complete* value, so the residual it
squares cannot arrive in two traversals.  `total_residual_weak_coefficients`
sums every unit's 1-form into one, and these tests hold it to the property that
makes that legitimate: the combined form is the sum of the units evaluated
apart, exactly, with each part contributing something.
"""
import random

import sympy as sp

from codegen.framework.forms.equations import (
    FormOrder,
    total_residual_weak_coefficients,
)
from codegen.framework.materials.mooney_rivlin_kelvin_voigt import (
    _build_system,
    _mooney_rivlin_energy,
)

DIM = 3
PARAMETERS = {
    sp.Symbol("mu"): sp.Rational(3),
    sp.Symbol("lmbda"): sp.Rational(7, 2),
    sp.Symbol("eta_s"): sp.Rational(1, 20),
    sp.Symbol("eta_b"): sp.Rational(1, 100),
    sp.Symbol("u_dt_shift"): sp.Rational(2),
}


def _grad(i, j):
    return sp.Symbol("u%d_grad_%d" % (i, j))


def _test_grad(i, j):
    return sp.Symbol("u%d_test_grad_%d" % (i, j))


def _test_value(i):
    return sp.Symbol("u%d_test" % i)


def _random_point(seed):
    random.seed(seed)
    point = {}
    for i in range(DIM):
        point[_test_value(i)] = sp.Rational(random.randint(-30, 30), 100)
        for j in range(DIM):
            point[_grad(i, j)] = sp.Rational(random.randint(-40, 40), 1000)
            point[_test_grad(i, j)] = sp.Rational(random.randint(-30, 30), 100)
            point[sp.Symbol("u%d_old_grad_%d" % (i, j))] = sp.Rational(
                random.randint(-40, 40), 1000
            )
    return point


def _evaluate(expression, point):
    return sp.nsimplify(sp.sympify(expression).xreplace(point).xreplace(PARAMETERS))


def _combined_weak_form(system):
    coefficients = total_residual_weak_coefficients(system)
    return sum(
        (
            sp.sympify(entry.value) * _test_value(row)
            + sum(
                sp.sympify(entry.gradient[j]) * _test_grad(row, j) for j in range(DIM)
            )
            for row, entry in enumerate(coefficients)
        ),
        sp.S.Zero,
    )


def _viscous_weak_form(system):
    equation = next(e for e in system.equations if e.name == "viscous")
    collection = system.form_collection(equation, orders=(FormOrder.ONE,))
    return sum(
        (sp.sympify(row) for row in collection.residual_expressions), sp.S.Zero
    )


def _elastic_weak_form():
    """The elastic unit from its energy, by differentiation -- an independent
    route to the same 1-form, so the test does not check the combiner against
    the machinery it is built from."""
    deformation = sp.Matrix(
        DIM, DIM, lambda i, j: (1 if i == j else 0) + _grad(i, j)
    )

    class _Variable:
        pass

    variable = _Variable()
    variable.value = deformation
    energy = _mooney_rivlin_energy(variable, DIM)
    return sum(
        (
            sp.diff(energy, _grad(i, j)) * _test_grad(i, j)
            for i in range(DIM)
            for j in range(DIM)
        ),
        sp.S.Zero,
    )


def test_the_combined_form_is_the_sum_of_the_units():
    system = _build_system(DIM)
    point = _random_point(11)

    combined = _evaluate(_combined_weak_form(system), point)
    elastic = _evaluate(_elastic_weak_form(), point)
    viscous = _evaluate(_viscous_weak_form(system), point)

    assert sp.simplify(combined - (elastic + viscous)) == 0
    # Otherwise the agreement above is agreement about zero.
    assert elastic != 0
    assert viscous != 0


def test_every_unit_reaches_the_combined_form():
    """Dropping either unit must change the answer.

    The negative control for the test above: a combiner that silently skipped
    the energy, or the residual, would still satisfy an equality written
    against whichever half it kept.
    """
    system = _build_system(DIM)
    point = _random_point(23)

    combined = _evaluate(_combined_weak_form(system), point)
    elastic = _evaluate(_elastic_weak_form(), point)
    viscous = _evaluate(_viscous_weak_form(system), point)

    assert sp.simplify(combined - elastic) != 0
    assert sp.simplify(combined - viscous) != 0


def test_one_row_per_lowered_field_component():
    system = _build_system(DIM)
    coefficients = total_residual_weak_coefficients(system)
    assert tuple(entry.row_field for entry in coefficients) == ("u0", "u1", "u2")
    for entry in coefficients:
        assert len(entry.gradient) == DIM


def test_the_combined_collection_carries_the_combined_residual():
    """The sum, lowered the way a hand-written residual is lowered.

    `total_residual_collection` exists so the rest of the pipeline sees an
    ordinary residual unit.  This checks the two halves of that: the collection
    reports the combined residual, and it reports it through the same
    `residual_expressions` / `coefficients` surface a residual material fills.
    """
    from codegen.framework.forms.equations import total_residual_collection

    system = _build_system(DIM)
    collection = total_residual_collection(system)
    point = _random_point(37)

    assert len(collection.residual_expressions) == DIM
    lowered = sum(
        (sp.sympify(row) for row in collection.residual_expressions), sp.S.Zero
    )

    elastic = _evaluate(_elastic_weak_form(), point)
    viscous = _evaluate(_viscous_weak_form(system), point)
    assert sp.simplify(_evaluate(lowered, point) - (elastic + viscous)) == 0

    # The material constants have to arrive as parameters; anything left
    # unclassified would be taken for field data and asked for as a stream.
    names = set(map(str, collection.parameters))
    assert {"mu", "lmbda", "eta_s", "eta_b", "u_dt_shift"} <= names


def test_the_combined_collection_reads_the_previous_state():
    """Kelvin-Voigt has a rate, so the combined unit must keep its history.

    A combiner that built fields without a previous state would drop the
    viscous term's `u_old` silently -- the expressions would still evaluate,
    just against symbols nothing supplies.
    """
    from codegen.framework.forms.equations import total_residual_collection

    system = _build_system(DIM)
    collection = total_residual_collection(system)
    dependencies = collection.dependencies
    assert dependencies.previous
    assert dependencies.current


# Every material has a residual and therefore a residual merit, so the combined
# form is built for materials with one unit as well as for materials with
# several.  The two shapes that unlocks are the ones below: a material written
# as one energy, whose residual exists only as that energy's 1-form, and a
# material written as one residual, which already is its own total and must
# combine to itself rather than to something near it.


def _weak_form(coefficients):
    """The 1-form the coefficients stand for, named as the lowering names it.

    `_combined_weak_form` spells its test symbols `u0_test`, which is right for
    a vector field and wrong for a scalar one -- laplace's row is `u`, not
    `u0`.  This reads the row name the coefficients carry instead.
    """
    return sum(
        (
            sp.sympify(entry.value) * sp.Symbol("%s_test" % entry.row_field)
            + sum(
                sp.sympify(entry.gradient[j])
                * sp.Symbol("%s_test_grad_%d" % (entry.row_field, j))
                for j in range(DIM)
            )
            for entry in coefficients
        ),
        sp.S.Zero,
    )


def test_an_energy_only_material_combines_to_its_energy_gradient():
    """Laplace has no residual unit at all; its residual is the 1-form.

    Checked against the energy differentiated by hand rather than against the
    form collection the combiner reads, so this is not the machinery agreeing
    with itself.
    """
    from codegen.framework.materials.laplace import _build_system

    coefficients = total_residual_weak_coefficients(_build_system(DIM))
    assert tuple(entry.row_field for entry in coefficients) == ("u",)

    kappa = sp.Symbol("kappa")
    expected = sum(
        (
            kappa
            * sp.Symbol("u_grad_%d" % j)
            * sp.Symbol("u_test_grad_%d" % j)
            for j in range(DIM)
        ),
        sp.S.Zero,
    )
    assert sp.simplify(_weak_form(coefficients) - expected) == 0
    # A gradient energy contributes nothing against the test value, and the
    # combiner leaving that row zero is the correct answer rather than a
    # dropped term -- the guard in `total_residual_weak_coefficients` is what
    # makes an energy that *would* contribute one fail instead of land here.
    assert all(sp.sympify(entry.value) == 0 for entry in coefficients)


def test_a_single_residual_unit_combines_to_itself():
    """A material already written as one residual is its own total.

    Exactly, not approximately: the combined form is the route the merit
    kernel lowers through, so any drift between it and the unit's own 1-form
    is a merit that disagrees with the gradient it is supposed to measure.
    """
    from codegen.framework.materials.body_force import _build_system

    system = _build_system(DIM)
    equation = system.equations[0]
    own = sum(
        (
            sp.sympify(row)
            for row in system.form_collection(
                equation, orders=(FormOrder.ONE,)
            ).residual_expressions
        ),
        sp.S.Zero,
    )

    assert sp.simplify(_weak_form(total_residual_weak_coefficients(system)) - own) == 0
    # Otherwise the agreement above is agreement about zero.
    assert own != 0
