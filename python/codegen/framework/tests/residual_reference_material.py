"""The Dirichlet form written as a residual, for tests that need one.

Several tests pin behaviour that only a residual-formulated scalar operator on
a lowest-order simplex reaches: the gradient-metric kernel body, and the
residual emitter's IR path through it.  `laplace` used to be that operator and
is written as an energy now, and no other material in the tree has the shape --
`stokes` and `navier_stokes` are mixed, `two_phase_flow` carries two fields,
`neumann` contracts against the test value alone.

Rather than let those tests quietly stop proving anything, or add a material to
the shipped set for their benefit, the form lives here.  It is the Dirichlet
residual `kappa * grad(u) . grad(v)` -- exactly what `laplace` was -- built the
same way a material builds one, so the code under test sees no difference.
"""

from sfem import gen


kappa = gen.material_parameter("kappa")
element = gen.FiniteElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u")
        v = gen.TestFunction(V, name="u_test")
        system.add_residual(
            "", kappa * gen.inner(gen.grad(u), gen.grad(v)), fields=(u,)
        )
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "laplace_residual_reference",
    systems,
    elements=("TRI3", "TET4"),
    op_name="GeneratedLaplaceResidualReference",
    parameter_defaults=(("kappa", 1.0),),
)
