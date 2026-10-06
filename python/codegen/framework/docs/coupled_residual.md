# Coupled Residual Example

Coupled residual kernels are generated through the same public `sfem.gen` path
as single-field materials: build symbolic fields and weak forms, wrap them in a
`CodeGenerator`, then call `gen.generate(...)` or `gen.run(...)`.

```python
import sympy as sp

from sfem import gen

Q = gen.FunctionSpace(gen.FiniteElement("Lagrange", degree=1))
W = gen.MixedFunctionSpace(Q, Q)

system = gen.EquationSystemBuilder(2)
with gen.geometric_dimension_context(2):
    u = gen.Function(W[0], "u")
    v = gen.Function(W[1], "v")
    q_u = gen.TestFunction(W[0], name="u_test")
    q_v = gen.TestFunction(W[1], name="v_test")

    dt, k_u, k_v, coupling = sp.symbols("dt k_u k_v coupling")
    form = (
        (u - gen.old(u)) * q_u / dt
        + k_u * gen.inner(gen.grad(u), gen.grad(q_u))
        + coupling * (u - v) * q_u
        + (v - gen.old(v)) * q_v / dt
        + k_v * gen.inner(gen.grad(v), gen.grad(q_v))
        + coupling * (v - u) * q_v
    )
    system.add_residual("", form, fields=(u, v))

material = gen.CodeGenerator(
    "coupled_diffusion",
    gen.EquationSystems([system.build()]),
    elements=("TRI3", "QUAD4"),
    parameter_defaults=(
        ("dt", 1.0),
        ("k_u", 1.0),
        ("k_v", 1.0),
        ("coupling", 0.0),
    ),
)

result = gen.generate(material, "/tmp/coupled_diffusion", elements=("TRI3",))
```

The generated files are returned in `result.sources`. Material-specific scripts
under `python/codegen/framework` use the same `gen.run(...)` or
`gen.generate(...)` entry points.
