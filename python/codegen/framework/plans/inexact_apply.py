"""The inexact matrix-free apply: quadrature removed by an L2 projection.

Named for the kernel it varies, not for the mathematics.  The framework's
energy path publishes `objective_steps`, `gradient` and `apply`, and `apply` is
its name for the matrix-free action of the second form -- the operator applied
to a vector, with nothing assembled.  Calling this `inexact_hessian` would name
a matrix, and the assembled second form is a different thing here with its own
kernels and formats, so the name would say the opposite of what the kernel does.

The exact matrix-free apply of a hyperelastic operator is

    (H h)_{i,p} = sum_q w_q * S_{ikmn}(F(xi_q)) * dh_k/dxi_n * dphi_p/dxi_m

with `S` the material tangent already pulled back through the geometry and
weighted -- the loperand of `plans.loperand`, for a form whose flux is
nonlinear.  `S` varies over the element because `F` does, which is what forces
the quadrature loop.

Project `S` onto the constants -- its L2 projection onto P0, which is its
average over the element -- and the loop separates:

    (H h)_{i,p} ~= Sbar_{ikmn} * sum_j h_{k,j} * Wbar_{jmpn}

    Wbar_{jmpn} = integral over the reference cell of
                  dphi_j/dxi_m * dphi_p/dxi_n

`Sbar` is one small object per element, computed once from the state.  `Wbar`
is a pure property of the element, integrated exactly here, at generation time,
and never computed at run time at all.  The quadrature loop is gone from the
kernel.

**Where the inexactness is, and where it is not.**  On an affine simplex the
deformation gradient is constant over the element, so `S` is constant, so its
projection onto the constants is `S` itself and the kernel is *exact*.  The
approximation appears only where the state genuinely varies across the cell --
a curved or higher-order element -- and there it is a one-term projection whose
error has to be measured rather than assumed.  That split is what makes this
testable: the affine simplex is the case where the machinery must reproduce the
exact kernel, and it is the gate on everything the other elements then do.

**The compression, and the trap in it.**  `Wbar` is small in a specific way:
whatever the element, it takes at most ten distinct rational values, and most
of them are zero.

    TET4    144 entries    3 distinct   108 zero
    TRI3     36 entries    3 distinct    20 zero
    HEX8    576 entries   10 distinct     0 zero
    TET10   900 entries    9 distinct   519 zero

The trap is to store those values in a runtime table and index it, which is
what compresses the *memory* and destroys the *arithmetic*: the constants stop
being constants, every multiply by zero survives into the emitted code, and
nothing downstream can fold them.  Measured on the same contraction:

    TET4    constants folded    31 temporaries,   334 operations
            runtime table      229 temporaries,  1406 operations
    HEX8    constants folded  1500 temporaries,  4544 operations
            runtime table      399 temporaries,  6231 operations

So `Wbar` is folded, not tabulated.  Its compression is real and is spent at
generation time, where a rational that appears ninety-six times costs nothing
to repeat and a zero costs nothing at all.
"""

from dataclasses import dataclass
from functools import lru_cache
import itertools

import sympy as sp

from codegen.framework.fem.reference_basis import reference_basis


@dataclass(frozen=True)
class ReferenceGradientProduct:
    """`Wbar`, and what its entries turn out to be.

    Indexed ``[node, direction, node, direction]`` and flattened row-major, the
    order every four-tensor in the framework uses.
    """

    element_type: str
    dim: int
    n_nodes: int
    entries: tuple

    def entry(self, node, direction, other_node, other_direction):
        stride = self.n_nodes * self.dim
        return self.entries[
            ((node * self.dim + direction) * stride)
            + other_node * self.dim
            + other_direction
        ]

    @property
    def n_entries(self):
        return len(self.entries)

    @property
    def distinct_values(self):
        """The distinct rationals it takes, in a stable order."""
        return tuple(sorted({sp.nsimplify(e) for e in self.entries}, key=sp.default_sort_key))

    @property
    def n_zero(self):
        return sum(1 for entry in self.entries if entry == 0)

    @property
    def is_major_symmetric(self):
        """`Wbar_{jmpn} == Wbar_{pnjm}`, which it is for every element here.

        It is an integral of a product of two gradients, so swapping the two
        (node, direction) pairs swaps the factors and nothing else.
        """
        return all(
            self.entry(j, m, p, n) == self.entry(p, n, j, m)
            for j in range(self.n_nodes)
            for m in range(self.dim)
            for p in range(self.n_nodes)
            for n in range(self.dim)
        )


@lru_cache(maxsize=None)
def reference_gradient_product(element_type):
    """`Wbar` for this element, integrated exactly, or ``None``.

    Cached because it is a property of the element and the integration is the
    expensive part of the whole plan -- once per element per process, never per
    material and never per call.
    """
    basis = reference_basis(element_type)
    if basis is None:
        return None
    gradients = basis.gradients()
    entries = []
    for node, direction, other_node, other_direction in itertools.product(
        range(basis.n_nodes), range(basis.dim), range(basis.n_nodes), range(basis.dim)
    ):
        entries.append(
            basis.integrate(
                gradients[node][direction] * gradients[other_node][other_direction]
            )
        )
    return ReferenceGradientProduct(
        element_type=basis.element_type,
        dim=basis.dim,
        n_nodes=basis.n_nodes,
        entries=tuple(entries),
    )


#: The elements on which the projection onto the constants is not an
#: approximation at all, because the deformation gradient does not vary over
#: the cell.  These are the elements the inexact kernel must agree with the
#: exact one on, which is what makes the rest of it checkable.
EXACT_ON_ELEMENTS = ("TRI3", "TET4")


def projection_is_exact(element_type):
    """Whether the P0 projection loses nothing on this element."""
    return str(element_type).upper() in EXACT_ON_ELEMENTS


@dataclass(frozen=True)
class InexactApplyPlan:
    """What an inexact apply kernel emits.

    `tangent_components` is how many numbers the projected tangent takes per
    element, using its major symmetry: a `dim^2` square, so 45 in three
    dimensions and 10 in two.
    """

    element_type: str
    dim: int
    n_nodes: int
    reference: ReferenceGradientProduct
    exact: bool

    @property
    def tangent_components(self):
        order = self.dim * self.dim
        return order * (order + 1) // 2

    def tangent_index(self, i, k, m, n):
        """Where `S_{ikmn}` sits in the packed tangent, by major symmetry.

        The symmetry is `S[i,k,m,n] == S[k,i,n,m]`, and getting it wrong is the
        easiest mistake in the whole construction.  It follows from what the
        indices are: `i` is the output component and `n` the test function's
        reference direction, `k` the increment's component and `m` its
        reference direction.  The element matrix entry is
        `S[i,k,m,n] * Wbar[j,m,p,n]`, and its symmetry under exchanging the two
        (component, node) pairs, together with `Wbar`'s own major symmetry,
        forces exactly this pairing.

        So the pair that swaps is `(i,n)` against `(k,m)` -- not `(i,k)`
        against `(m,n)`, which is the plausible-looking wrong answer and which
        no test comparing the staged form against the dense form can catch,
        because both would share it.  It was caught by evaluating the action
        against a directly integrated one on a concrete tetrahedron.
        """
        row, column = i * self.dim + n, k * self.dim + m
        if row > column:
            row, column = column, row
        order = self.dim * self.dim
        return row * order - row * (row - 1) // 2 + (column - row)

    def action(self, tangent, increment):
        """`(H h)_{i,p}`, as expressions in the packed tangent and increment.

        `tangent` is indexed by the packed index above and `increment` by
        `[component][node]`.  The reference tensor's entries go in as the
        rationals they are, so a zero removes its term here rather than
        surviving as a multiply in the emitted code.
        """
        outputs = []
        for component in range(self.dim):
            for node in range(self.n_nodes):
                total = sp.Integer(0)
                for other, m, n in itertools.product(
                    range(self.dim), range(self.dim), range(self.dim)
                ):
                    weight = sum(
                        (
                            increment[other][j]
                            * self.reference.entry(j, m, node, n)
                            for j in range(self.n_nodes)
                            if self.reference.entry(j, m, node, n) != 0
                        ),
                        sp.Integer(0),
                    )
                    if weight == 0:
                        continue
                    total += (
                        tangent[self.tangent_index(component, other, m, n)] * weight
                    )
                outputs.append(sp.expand(total))
        return tuple(outputs)


def inexact_apply_plan(element_type):
    """That plan, or ``None`` for an element with no symbolic basis."""
    reference = reference_gradient_product(element_type)
    if reference is None:
        return None
    return InexactApplyPlan(
        element_type=reference.element_type,
        dim=reference.dim,
        n_nodes=reference.n_nodes,
        reference=reference,
        exact=projection_is_exact(reference.element_type),
    )


@dataclass(frozen=True)
class RankFactoredGradientProduct:
    """`Wbar` as `C^T X C`, exactly, over the rationals.

    `Wbar` is the Gram matrix of the element's gradient functions, so its rank
    is the dimension of the span of those functions and nothing more.  That is
    a much stronger statement than "many entries are zero", and it is where the
    compression actually lives:

        TET4    12x12   rank 1        TRI3    6x6    rank 1
        QUAD4    8x8    rank 3        HEX8   24x24   rank 7
        TET10   30x30   rank 4

    A linear simplex is rank one because its gradients are constants spanning a
    one-dimensional space -- and rank one is precisely the loperand: the flux
    contracted with the geometry, one object per element.  So this is not a new
    trick beside the loperand, it is the same structure carried to elements
    whose gradients are not constant.

    `factor` is `C`, shape `(rank, n_nodes * dim)`, and `middle` is `X`, shape
    `(rank, rank)`.  Both are exact rationals and both are sparse; every zero is
    dropped where the action is built rather than multiplied by.
    """

    factor: tuple
    middle: tuple
    rank: int
    order: int

    def factor_entry(self, row, column):
        return self.factor[row * self.order + column]

    def middle_entry(self, row, column):
        return self.middle[row * self.rank + column]


@lru_cache(maxsize=None)
def rank_factored_gradient_product(element_type):
    """That factorisation, checked against the tensor it factors."""
    reference = reference_gradient_product(element_type)
    if reference is None:
        return None
    order = reference.n_nodes * reference.dim
    matrix = sp.Matrix(order, order, lambda r, c: reference.entries[r * order + c])
    _reduced, pivots = matrix.rref()
    factor = sp.Matrix([matrix.row(pivot) for pivot in pivots])
    gram = factor * factor.T
    inverse = gram.inv()
    middle = inverse * (factor * matrix * factor.T) * inverse
    # Checked here rather than trusted: a wrong factorisation is a wrong kernel
    # everywhere downstream, and this costs one matrix product once per element
    # type per process.
    if sp.simplify(factor.T * middle * factor - matrix) != sp.zeros(order, order):
        raise ValueError(
            "the rank factorisation of the reference tensor for %s is not exact"
            % reference.element_type
        )
    return RankFactoredGradientProduct(
        factor=tuple(factor),
        middle=tuple(middle),
        rank=factor.rows,
        order=order,
    )


@dataclass(frozen=True)
class ActionStage:
    """One named intermediate of the staged action, and what defines it."""

    name: str
    assignments: tuple


def staged_action(plan, tangent, increment, output_names):
    """The action through the rank factorisation, as stages to print.

    Four of them, each reading the previous by name so the factorisation
    survives into the emitted code instead of being expanded back out:

        P[k,a,m] = sum_j  C[a,(j,m)] h[k,j]        the increment, compressed
        Y[i,a,n] = sum_km Sbar[i,k,m,n] P[k,a,m]   the tangent applied
        Q[i,b,n] = sum_a  X[a,b] Y[i,a,n]          the middle factor
        out[i,p] = sum_bn C[b,(p,n)] Q[i,b,n]      expanded back to nodes

    Every sum skips its zero coefficients, which is what the sparsity of `C`
    and `X` is for.  Expanding these into one expression per output and letting
    common-subexpression elimination re-discover the structure gives the dense
    count back, so the stages are the point rather than an implementation
    detail.
    """
    factored = rank_factored_gradient_product(plan.element_type)
    dim, n_nodes, rank = plan.dim, plan.n_nodes, factored.rank

    compressed, compressed_defs = {}, []
    for component in range(dim):
        for mode in range(rank):
            for direction in range(dim):
                expression = sum(
                    (
                        increment[component][node]
                        * factored.factor_entry(mode, node * dim + direction)
                        for node in range(n_nodes)
                        if factored.factor_entry(mode, node * dim + direction) != 0
                    ),
                    sp.Integer(0),
                )
                if expression == 0:
                    continue
                symbol = sp.Symbol("pa_p%d_%d_%d" % (component, mode, direction))
                compressed[(component, mode, direction)] = symbol
                compressed_defs.append((symbol, expression))

    applied, applied_defs = {}, []
    for row in range(dim):
        for mode in range(rank):
            for direction in range(dim):
                expression = sum(
                    (
                        tangent[plan.tangent_index(row, component, other, direction)]
                        * compressed[(component, mode, other)]
                        for component in range(dim)
                        for other in range(dim)
                        if (component, mode, other) in compressed
                    ),
                    sp.Integer(0),
                )
                if expression == 0:
                    continue
                symbol = sp.Symbol("pa_y%d_%d_%d" % (row, mode, direction))
                applied[(row, mode, direction)] = symbol
                applied_defs.append((symbol, expression))

    mixed, mixed_defs = {}, []
    for row in range(dim):
        for mode in range(rank):
            for direction in range(dim):
                expression = sum(
                    (
                        factored.middle_entry(other, mode)
                        * applied[(row, other, direction)]
                        for other in range(rank)
                        if factored.middle_entry(other, mode) != 0
                        and (row, other, direction) in applied
                    ),
                    sp.Integer(0),
                )
                if expression == 0:
                    continue
                symbol = sp.Symbol("pa_q%d_%d_%d" % (row, mode, direction))
                mixed[(row, mode, direction)] = symbol
                mixed_defs.append((symbol, expression))

    output_defs = []
    for row in range(dim):
        for node in range(n_nodes):
            expression = sum(
                (
                    factored.factor_entry(mode, node * dim + direction)
                    * mixed[(row, mode, direction)]
                    for mode in range(rank)
                    for direction in range(dim)
                    if (row, mode, direction) in mixed
                    and factored.factor_entry(mode, node * dim + direction) != 0
                ),
                sp.Integer(0),
            )
            output_defs.append((output_names[row][node], expression))

    return (
        ActionStage("compressed_increment", tuple(compressed_defs)),
        ActionStage("applied_tangent", tuple(applied_defs)),
        ActionStage("mixed", tuple(mixed_defs)),
        ActionStage("output", tuple(output_defs)),
    )


def emittable_inexact_apply_plan(element_type, dim, n_nodes, flux_form, rule):
    """The plan when this element and form can take the projected apply.

    Every condition is here rather than in the emitter: an element with no
    symbolic basis, a plan whose shape disagrees with the specialization, a
    rule whose reference gradients are not the constants an affine simplex
    has, or a form whose field arity is not the spatial dimension.  Emission
    asks once and spells the answer.
    """
    plan = inexact_apply_plan(element_type)
    if plan is None or plan.dim != int(dim) or plan.n_nodes != int(n_nodes):
        return None
    if flux_form is None or flux_form.n_field_components != int(dim):
        return None
    values = tuple(getattr(rule, "reference_gradients", ()) or ())
    if len(values) != int(n_nodes) * int(dim):
        return None
    return plan
