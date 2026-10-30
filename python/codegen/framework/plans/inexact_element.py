"""What an element contributes to the stored tangent, whatever the material.

The inexact apply stores the material tangent pulled back to the reference cell
and weighted, at the points of the rule the exact operator integrates with:

    (H h)_{i,p} = sum_q  S^q_{ikmn}  dh_k/dxi_m(xi_q)  dphi_p/dxi_n(xi_q)

`S^q` carries the weight, the geometry and the material, so the apply reads it
and the increment and nothing else.  Everything else in that sum belongs to the
element and its rule: which reference derivatives the store contracts with,
how the stored numbers are grouped, which of them the rule cannot see.  That is
this plan.  The material's half -- `S^q` at one point -- is
`plans/point_tangent.py`, and the two meet only in the library, which is why
neither knows the other.

Three facts about the rule decide the store, and each is a property of the
reference basis and the points alone:

* **What is stored.**  The point tangents themselves (`points`), or, on TET10,
  their P2 moments `M^{ab} = sum_q lambda_a lambda_b S^q` (`moments`).  The
  moments are exact because the reference derivatives of a quadratic basis are
  linear, `dphi/dxi = sum_a lambda_a G_a`, on any cell: the tangent is pulled
  back per point, so the identity holds on a curved TET10 as well as on an
  affine one.  Ten tensors replace eleven.
* **The gradient vectors** each stored tensor contracts with: the derivatives
  at the points, or at the vertices for the moments.
* **The fold.**  On a tensor-product element a component with `m = n = d`
  multiplies a product of two d-derivatives, a polynomial of degree `2(p-1)` in
  `xi_d` along every line of points in direction `d`.  When the line has more
  points than that degree needs, one point's value moves exactly to the others,
  weighted by their Lagrange polynomials at it.  The dropped point is the one
  whose weights are smallest.

The numbers are computed in floating point in the order the spike computed
them, so that the generated tables reproduce the measured stores bit for bit;
the rationals they approximate are checked where they are exact.
"""

from dataclasses import dataclass
from fractions import Fraction
import itertools

from codegen.framework.fem.reference import (
    sfem_cell_rule_points,
    sfem_is_tensor_product_hex_element,
)
from codegen.framework.fem.reference_basis import reference_basis
from codegen.framework.plans.layout import gather_shape_order


#: The two ways a store can hold what the rule sees.
POINTS = "points"
MOMENTS = "moments"

#: Elements whose reference derivatives are linear in the reference
#: coordinates, so that the moments against the barycentric products are exact.
_MOMENT_ELEMENTS = ("TET10",)


def symmetric_components(dim):
    """Independent numbers of a tangent with the major symmetry: 45 in 3D."""
    order = dim * dim
    return order * (order + 1) // 2


def full_components(dim):
    """Numbers of a tangent without it: 81 in 3D."""
    return (dim * dim) ** 2


def packed_slot(row, column, order, symmetric):
    """Where entry (row, column) of the `order x order` tangent is stored.

    Rows are `r = i * dim + n` (output component, test direction) and columns
    `c = k * dim + m` (increment component, increment direction); the major
    symmetry is the symmetry of this matrix, so a symmetric tangent keeps the
    upper triangle row by row and an unsymmetric one keeps the whole square.
    The same packing as `plans/point_tangent.PointTangentPlan.tangent_index`.
    """
    if not symmetric:
        return row * order + column
    if row > column:
        row, column = column, row
    return row * order - row * (row - 1) // 2 + (column - row)


def slot_direction(dim, symmetric):
    """The direction `d` of each stored slot whose two directions agree, else -1."""
    order = dim * dim
    n_slots = symmetric_components(dim) if symmetric else full_components(dim)
    directions = [-1] * n_slots
    for row in range(order):
        for column in range(order):
            if symmetric and column < row:
                continue
            if row % dim == column % dim:
                directions[packed_slot(row, column, order, symmetric)] = row % dim
    return tuple(directions)


@dataclass(frozen=True)
class TensorProductLine:
    """A tensor-product element's one-dimensional data.

    `values[q][s]` and `derivatives[q][s]` are the 1D shape functions and their
    derivatives at the 1D points `points[q]`; nodes and points are numbered
    `s = sx + n1 (sy + n1 sz)` and `q = qx + q1 (qy + q1 qz)`.
    """

    points: tuple
    values: tuple
    derivatives: tuple

    @property
    def n_points(self):
        return len(self.points)

    @property
    def n_shape(self):
        return len(self.values[0])

    @property
    def degree(self):
        return self.n_shape - 1


@dataclass(frozen=True)
class Fold:
    """The folded point store of a tensor-product element.

    The point whose index in direction `d` is `drop` keeps no `m = n = d`
    components (bit `d` of `mask[q]`); its values move to the kept points of
    its line, `T_q += coefficient[q][d] * T_{source[q][d]}`.
    """

    drop: int
    mask: tuple
    source: tuple
    coefficient: tuple


@dataclass(frozen=True)
class InexactElementPlan:
    """The element's half of the stored tangent, for one rule.

    `gradients[r][m][j]` is `dphi_j/dxi_m` at gradient point `r`, nodes in the
    order the gather stages them (`shape_order`).  The store holds `n_stored`
    tensors; tensor `t` contracts the increment's gradient at `pairs[t][0]` into
    the output at `pairs[t][1]` and, when the two differ, the other way round.
    `reference_weight[t]` is what the element mean contributes to tensor `t`
    (the mean-plus-deviation encoding subtracts it); `multiplicity[t]` counts
    how often the tensor occurs in the symmetric sum.
    """

    element_type: str
    rule_key: str
    dim: int
    n_nodes: int
    kind: str
    shape_order: tuple
    gradients: tuple
    pairs: tuple
    reference_weight: tuple
    multiplicity: tuple
    volume: float
    setup_weights: tuple
    setup_gradients: tuple
    moment_factor: tuple
    line: TensorProductLine
    fold: Fold

    @property
    def n_gradients(self):
        return len(self.gradients)

    @property
    def n_stored(self):
        return len(self.pairs)

    @property
    def n_setup_points(self):
        return len(self.setup_weights)

    @property
    def is_moments(self):
        return self.kind == MOMENTS

    @property
    def foldable(self):
        return self.fold is not None

    @property
    def struct_name(self):
        return "inexact_%s" % self.rule_key

    @property
    def reordered(self):
        """Whether the stages read the mesh's nodes in another order (HEX8's Cartesian one)."""
        return tuple(self.shape_order) != tuple(range(self.n_nodes))

    @property
    def stage_form(self):
        """How the stages are written: from the derivative table (simplices), by hand-eliminated
        linear sweeps (HEX8's 2x2x2), or by general one-dimensional sweeps."""
        if self.line is None:
            return "tabulated"
        if self.line.n_shape == 2 and self.line.n_points == 2:
            return "linear_sweeps"
        return "sweeps"

    @property
    def setup_form(self):
        """Whether the set-up's points are the stored ones (its gradients are `stage_in`'s) or
        the rule's own, which the moments integrate over."""
        return "explicit" if self.setup_gradients is not None else "stages"

    @property
    def line_coefficients(self):
        """The fold's weight of each one-dimensional index of a line: the Lagrange polynomial of a
        kept point at the dropped one, zero at the dropped index itself."""
        drop = self.fold.drop
        keep = [k for k in range(self.line.n_points) if k != drop]
        return tuple(
            0.0 if index == drop else _lagrange_weight(
                [self.line.points[k] for k in keep], self.line.points[drop], keep.index(index)
            )
            for index in range(self.line.n_points)
        )

    def fold_layout(self, symmetric):
        """The folded store's tables for a tangent of this symmetry.

        Off the tensor-product elements no point drops anything, so the kernels
        have one form whatever the element.
        """
        n_slots = symmetric_components(self.dim) if symmetric else full_components(self.dim)
        directions = slot_direction(self.dim, symmetric)
        if self.fold is None:
            mask = (0,) * self.n_stored
            source = tuple((-1,) * self.dim for _ in range(self.n_stored))
            coefficient = tuple((0.0,) * self.dim for _ in range(self.n_stored))
        else:
            mask, source, coefficient = self.fold.mask, self.fold.source, self.fold.coefficient
        keep = tuple(
            tuple(
                s
                for s in range(n_slots)
                if not (directions[s] >= 0 and (bits >> directions[s]) & 1)
            )
            for bits in range(1 << self.dim)
        )
        offsets = [0]
        for t in range(self.n_stored):
            offsets.append(offsets[-1] + len(keep[mask[t]]))
        weights = self.reference_weight
        folded_weight = tuple(
            tuple(
                weights[t]
                if directions[s] < 0 or (mask[t] >> directions[s]) & 1
                else weights[t]
                + coefficient[t][directions[s]] * weights[source[t][directions[s]]]
                for s in range(n_slots)
            )
            for t in range(self.n_stored)
        )
        return FoldLayout(
            n_slots=n_slots,
            directions=directions,
            mask=tuple(mask),
            source=tuple(source),
            coefficient=tuple(coefficient),
            keep=keep,
            offsets=tuple(offsets),
            folded_weight=folded_weight,
        )


@dataclass(frozen=True)
class FoldLayout:
    n_slots: int
    directions: tuple
    mask: tuple
    source: tuple
    coefficient: tuple
    keep: tuple
    offsets: tuple
    folded_weight: tuple

    @property
    def n_numbers(self):
        """Stored numbers per element."""
        return self.offsets[-1]


def _rule_key(element_type, rule):
    """`<element>_q<points>`, e.g. `tet10_q11`, `hex8_q8`, `proteus_hex27_q64`.

    Keyed by the element and not only by its basis: HEX8 and PROTEUS_HEX8 share
    a basis and a rule, and differ in the order their nodes are staged in.
    """
    return "%s_q%d" % (str(element_type).lower(), int(rule.n_qp))


def _flat_gradients(rule, n_nodes, dim):
    """`rule.reference_gradients` as `[q][m][j]`."""
    values = tuple(rule.reference_gradients)
    stride = n_nodes * dim
    return tuple(
        tuple(
            tuple(float(values[q * stride + j * dim + m]) for j in range(n_nodes))
            for m in range(dim)
        )
        for q in range(len(values) // stride)
    )


def _simplex_points_plan(element_type, rule, dim, n_nodes, key):
    """TET4: one point, constant derivatives; the points are the rule's own."""
    gradients = _flat_gradients(rule, n_nodes, dim)
    weights = tuple(float(w) for w in rule.weights)
    return InexactElementPlan(
        element_type=element_type,
        rule_key=key,
        dim=dim,
        n_nodes=n_nodes,
        kind=POINTS,
        shape_order=tuple(range(n_nodes)),
        gradients=gradients,
        pairs=tuple((q, q) for q in range(len(weights))),
        reference_weight=weights,
        multiplicity=(1,) * len(weights),
        volume=sum(weights),
        setup_weights=weights,
        setup_gradients=None,
        moment_factor=None,
        line=None,
        fold=None,
    )


def _barycentric(point):
    """`(1 - x - y - z, x, y, z)`, subtracted left to right as the spike rounds it."""
    first = 1
    for coordinate in point:
        first = first - coordinate
    return (first,) + tuple(point)


def _moments_plan(element_type, rule, dim, n_nodes, key):
    """TET10: the P2 moments of the tangent against the barycentric products.

    The derivatives the moments contract with are the basis derivatives at the
    vertices, exact rationals from the symbolic basis.
    """
    basis = reference_basis(element_type)
    vertices = (tuple(Fraction(0) for _ in range(dim)),) + tuple(
        tuple(Fraction(int(i == axis)) for axis in range(dim)) for i in range(dim)
    )
    coordinates = basis.coordinates
    gradients = []
    for vertex in vertices:
        at = dict(zip(coordinates, vertex))
        gradients.append(
            tuple(
                tuple(
                    Fraction(str(basis.gradients()[j][m].subs(at)))
                    for j in range(n_nodes)
                )
                for m in range(dim)
            )
        )
    points = sfem_cell_rule_points(rule)
    weights = tuple(float(w) for w in rule.weights)
    n_vertices = dim + 1
    pairs = tuple((a, b) for a in range(n_vertices) for b in range(a, n_vertices))
    factor = tuple(
        tuple(
            _barycentric(point)[a] * _barycentric(point)[b] for (a, b) in pairs
        )
        for point in points
    )
    reference_weight = tuple(
        sum(weights[q] * factor[q][t] for q in range(len(points)))
        for t in range(len(pairs))
    )
    return InexactElementPlan(
        element_type=element_type,
        rule_key=key,
        dim=dim,
        n_nodes=n_nodes,
        kind=MOMENTS,
        shape_order=tuple(range(n_nodes)),
        gradients=tuple(
            tuple(tuple(float(v) for v in row) for row in vertex) for vertex in gradients
        ),
        pairs=pairs,
        reference_weight=reference_weight,
        multiplicity=tuple(1 if a == b else 2 for (a, b) in pairs),
        volume=sum(weights),
        setup_weights=weights,
        setup_gradients=_flat_gradients(rule, n_nodes, dim),
        moment_factor=factor,
        line=None,
        fold=None,
    )


def _tensor_product_line(rule, dim):
    n1 = int(rule.tensor_product_n_shape_1d)
    q1 = int(rule.tensor_product_n_qp_1d)
    values = tuple(float(v) for v in rule.tensor_product_shape_values_1d)
    derivatives = tuple(float(v) for v in rule.tensor_product_shape_gradients_1d)
    points = tuple(float(p[0]) for p in sfem_cell_rule_points(rule)[:q1])
    return TensorProductLine(
        points=points,
        values=tuple(values[q * n1 : (q + 1) * n1] for q in range(q1)),
        derivatives=tuple(derivatives[q * n1 : (q + 1) * n1] for q in range(q1)),
    )


def _lagrange_weight(points, target, k):
    """The Lagrange polynomial of `points[k]` over `points`, at `target`."""
    value = 1.0
    for i, x in enumerate(points):
        if i != k:
            value *= (target - x) / (points[k] - x)
    return value


def _fold(line, dim):
    """The fold, when the line has a point the `m = n = d` products do not need.

    The products are of degree `2 (p - 1)` in `xi_d`, so `2 p - 1` points
    determine them; a line with more can drop one.  The dropped index is the one
    with the smallest largest weight, ties to the lowest index.
    """
    q1 = line.n_points
    if q1 - 1 < 2 * line.degree - 1 or q1 < 2:
        return None
    best = None
    for drop in range(q1):
        keep = [k for k in range(q1) if k != drop]
        coefficient = [
            _lagrange_weight([line.points[k] for k in keep], line.points[drop], i)
            for i in range(len(keep))
        ]
        largest = max(abs(c) for c in coefficient)
        if best is None or largest < best[2] - 1e-12:
            best = (drop, keep, largest, coefficient)
    drop, keep, _largest, coefficient = best
    n_points = q1**dim

    def flat(index):
        flat_index = 0
        for axis in reversed(range(dim)):
            flat_index = flat_index * q1 + index[axis]
        return flat_index

    masks, sources, coefficients = [], [], []
    for q in range(n_points):
        index = [(q // q1**axis) % q1 for axis in range(dim)]
        masks.append(sum(1 << d for d in range(dim) if index[d] == drop))
        source, weight = [], []
        for d in range(dim):
            if index[d] == drop:
                source.append(-1)
                weight.append(0.0)
            else:
                moved = list(index)
                moved[d] = drop
                source.append(flat(moved))
                weight.append(coefficient[keep.index(index[d])])
        sources.append(tuple(source))
        coefficients.append(tuple(weight))
    return Fold(
        drop=drop,
        mask=tuple(masks),
        source=tuple(sources),
        coefficient=tuple(coefficients),
    )


def _tensor_product_plan(element_type, rule, dim, n_nodes, key):
    """A lexicographic tensor-product element: the point values, and the fold."""
    line = _tensor_product_line(rule, dim)
    q1, n1 = line.n_points, line.n_shape
    n_points = q1**dim
    gradients = []
    weights = []
    weights_1d = tuple(float(w) for w in rule.tensor_product_weights_1d)
    for q in range(n_points):
        qi = [(q // q1**axis) % q1 for axis in range(dim)]
        rows = []
        for m in range(dim):
            row = []
            for s in range(n_nodes):
                si = [(s // n1**axis) % n1 for axis in range(dim)]
                value = 1.0
                for axis in range(dim):
                    table = line.derivatives if axis == m else line.values
                    value *= table[qi[axis]][si[axis]]
                row.append(value)
            rows.append(tuple(row))
        gradients.append(tuple(rows))
        weight = 1.0
        for axis in range(dim):
            weight *= weights_1d[qi[axis]]
        weights.append(weight)
    weights = tuple(weights)
    return InexactElementPlan(
        element_type=element_type,
        rule_key=key,
        dim=dim,
        n_nodes=n_nodes,
        kind=POINTS,
        shape_order=tuple(gather_shape_order(element_type, dim, n_nodes, True)),
        gradients=tuple(gradients),
        pairs=tuple((q, q) for q in range(n_points)),
        reference_weight=weights,
        multiplicity=(1,) * n_points,
        volume=sum(weights),
        setup_weights=weights,
        setup_gradients=None,
        moment_factor=None,
        line=line,
        fold=_fold(line, dim),
    )


def inexact_plans(element_type, rule, flux_forms):
    """`(element plan, point-tangent plan)` for fluxes on an element, or ``None``.

    ``None`` where either half has no plan, or where they disagree on the
    dimension.
    """
    from codegen.framework.plans.point_tangent import point_tangent_plan

    element_plan = inexact_element_plan(element_type, rule)
    tangent_plan = point_tangent_plan(flux_forms)
    if element_plan is None or tangent_plan is None or tangent_plan.dim != element_plan.dim:
        return None
    return element_plan, tangent_plan


def inexact_element_plan(element_type, rule):
    """The plan for this element at this rule, or ``None`` where there is none.

    Covered: the linear tetrahedron, the quadratic one by its moments, and the
    lexicographic hexahedra of any order -- HEX8 numbers its nodes otherwise and
    is staged through `plans/layout`'s Cartesian order.
    """
    element_type = str(element_type).upper()
    dim = int(rule.dim)
    n_nodes = int(rule.n_shape)
    key = _rule_key(element_type, rule)
    if dim != 3:
        return None
    if sfem_is_tensor_product_hex_element(element_type) and getattr(
        rule, "is_tensor_product", False
    ):
        return _tensor_product_plan(element_type, rule, dim, n_nodes, key)
    if element_type in _MOMENT_ELEMENTS:
        return _moments_plan(element_type, rule, dim, n_nodes, key)
    if element_type == "TET4":
        return _simplex_points_plan(element_type, rule, dim, n_nodes, key)
    return None
