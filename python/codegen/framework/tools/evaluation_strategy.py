"""Does each element get the evaluation strategy its family calls for?

Three defaults govern how a form is evaluated, and all three are properties of
the element rather than of the material or of how the material was written:

  tensor-product        sum factorization, always.  There is no decision here.
  simplex, lowest order the CSE expanded for a compact physical gradient or for
                        the basis functions directly, and no quadrature-point
                        information generated at all -- the basis gradients are
                        constant over the element, so a quadrature loop and its
                        per-point data are pure waste.
  simplex, higher order rules of its own, distinct from both.

Both front ends reach sum factorization on tensor-product elements, through
different template families: the energy path defers a flux and contracts it with
TensorProductWeakOps (`tensor_test`), the residual path evaluates and integrates
with TensorProductResidualOps (`tensor_evaluate` / `tensor_integrate`).  So rule
one is largely met already, by two separate implementations of it.

Rule two is met in the volume kernels for the loop, where it is stated: every
lowest-order simplex opens `{ const int q = 0; }` instead of a one-trip loop, and
the `qp loops` count below is zero.
`tests/test_evaluation_strategy_conformance.py` holds it there.

Rule two also says *no quadrature-point information generated at all*, and the
loop count does not see that half.  A lowest-order simplex still asks the mesh
for `ref_tet4_q1::grad_ref_x()`, `ref_tet4_q1::shape()` and
`quad_tet_q1::q_weight()` and hands them to its kernel, which reads
`q_weight[q]` and `shape[q * NS + test]` at run time -- all of them constants of
a one-point rule on a constant-basis element.  The `ref data` column counts those
references, per element, in the element's own operator, and a lowest-order
simplex should carry none.  The loop count missed it because a shared local
header is templated on `NQ` and serves TET4 and TET10 from one file, so the loop
sits one directory above the element it belongs to and is attributable to
neither; what the element's own operator names is attributable.

Rule two says nothing about a facet.  What makes a lowest-order simplex
loop-free is that its volume integrand is built from basis gradients that are
constant over the cell, so one point is exact and the rule carries exactly one;
a facet load vector integrates the shape function itself, so its rule carries
two points on an edge and three on a triangle and its loop is not a one-trip
loop at all.  Surface integrals are therefore counted out of both rules rather
than reported as departures -- see `survey`.

A facet has a rule of its own, though, and the `facet grad` column states it.
The *measure* of a flat facet is the same at every point of its rule, because it
is built from reference gradients that do not vary over the facet -- so a table
of them is one row repeated once per point, and the loop that reads it is a loop
over constants.  On a device that table is a function-local
`static const s_t data[18]`: three identical copies of the six numbers a triangle
has.  This is the half of ISSUES.md item 11 the loop count cannot see, for the
same reason it could not see rule two's data half.

Told apart by the repetition rather than by the element's name, because it is a
property of the facet's geometry and not of the cell it bounds: a TRISHELL6 on a
TET10 is curved, its rows genuinely differ, and it is not a departure.  The first
version of this column looked instead for a `grad()` accessor beside a
point-free measure, which reported zero both before and after the fix it was
meant to describe -- a measurement that cannot fail is not one.

This reports the gap against the three rules, so closing it is measurable
rather than asserted.  It reads a generated tree:

    python -m codegen.framework.tools.codegen_snapshot capture /tmp/gen
    python -m codegen.framework.tools.evaluation_strategy /tmp/gen

See ARCHITECTURE.html OP 20 and OP 18.
"""

import argparse
import glob
import os
import re
import sys

from codegen.framework.fem.element_family import element_family as _element_family

#: A facet's reference-gradient table, with the node count its struct declares.
_FACET_GRADIENT_TABLE = re.compile(
    r"static constexpr int NS = (?P<n_shape>\d+);"
    r".*?const s_t \*grad\(\)\s*\{"
    r"\s*static const s_t data\[\d+\] = \{(?P<values>[^}]*)\}",
    re.S,
)
_FACET_GRADIENT_VALUE = re.compile(r"s_t\(([^)]*)\)")


def flat_facet_tables(source, n_shape, values):
    """Whether this table is one row repeated once per quadrature point.

    A flat facet -- a straight edge, a planar triangle -- has the same reference
    gradients at every point of its rule, so the table is `n_qp` identical
    copies of the row the facet actually has, and the loop that reads it is a
    loop over constants.  Told apart by the repetition rather than by the
    element's name, because it is a property of the facet's geometry: a
    TRISHELL6 on a curved TET10 facet has genuinely differing rows and is not
    this.

    The period is the node count times the reference dimension, and the facet's
    reference dimension is not written down, so both are tried -- an edge is one
    and a surface is two.
    """
    for reference_dim in (1, 2):
        period = n_shape * reference_dim
        if period and len(values) > period and len(values) % period == 0:
            if values == values[:period] * (len(values) // period):
                return True
    return False

def element_family(element):
    """The element's family, from the taxonomy the generator uses.

    This was three literal tuples here.  A measurement and a generator that
    disagree about what TET4 is would report conformance the generator does not
    have, so both now read `fem.element_family`.
    """
    return _element_family(element).value


def survey(generated):
    """Per material and element: sum factorization reached, quadrature loops emitted."""
    rows = []
    for material_dir in sorted(glob.glob(os.path.join(generated, "*"))):
        if not os.path.isdir(material_dir):
            continue
        material = os.path.basename(material_dir)
        # Sum factorization is reached through the material's tensor-product
        # local header, so it is a per-material fact, not a per-element one.
        # Two sum-factorised template families, not one.  The energy front end
        # contracts a deferred flux through TensorProductWeakOps (`tensor_test`);
        # the residual front end evaluates and integrates through
        # TensorProductResidualOps (`tensor_evaluate` / `tensor_integrate`).
        # Both are sum factorization.  An earlier version of this tool looked
        # only for `tensor_test` and therefore reported every residual material
        # as unfactorised, which was wrong -- and wrong in the direction that
        # invents work, since it made rule 1 look unmet where it is met.
        markers = ("tensor_test<", "tensor_evaluate<", "tensor_evaluate_value<",
                   "tensor_integrate<", "tensor_integrate_value<")
        factorised = any(
            any(marker in open(path, errors="ignore").read() for marker in markers)
            for path in glob.glob(os.path.join(material_dir, "**", "*.hpp"), recursive=True)
            if "tensor_product_kernels" not in path
        )
        by_element = {}
        volume = {}
        reference_data = {}
        facet_gradients = {}
        for path in glob.glob(os.path.join(material_dir, "d*", "*", "*")):
            if os.path.isdir(path):
                continue
            element = os.path.basename(os.path.dirname(path))
            source = open(path, errors="ignore").read()
            # The rules govern volume kernels.  A material whose only kernels
            # are surface integrals -- the Neumann conditions emit nothing but
            # `*_boundary_operator.cpp` -- is not a tensor-product volume
            # element missing sum factorization; it is a different kind of
            # integral, and counting it as a departure was a category error
            # that put sixteen phantom entries in this report.
            #
            # The same category error applied to the quadrature-loop rule, and
            # it put four more here.  What makes a lowest-order simplex
            # loop-free is that its *volume* integrand is built from basis
            # gradients that are constant over the cell, so one point is exact
            # and `sfem_element_quadrature_rule("TET4")` carries exactly one.
            # A facet load vector is a different integrand: it contains the
            # shape function itself, and for `neumann_general` a spatially
            # varying traction besides, so `boundary_codegen` asks for degree
            # `2 * order` and a flat TRISHELL3 facet genuinely carries three
            # points.  Its loop runs three times and is not a departure --
            # collapsing it to `q = 0` would keep a third of the traction.
            # So loops are counted where the rules apply: in volume kernels.
            volume.setdefault(element, False)
            by_element.setdefault(element, 0)
            reference_data.setdefault(element, 0)
            facet_gradients.setdefault(element, 0)
            if "_boundary_operator" in os.path.basename(path):
                # A flat facet still carrying one reference-gradient row per
                # quadrature point -- see the module docstring.
                facet_gradients[element] += sum(
                    flat_facet_tables(
                        source,
                        int(match.group("n_shape")),
                        _FACET_GRADIENT_VALUE.findall(match.group("values")),
                    )
                    for match in _FACET_GRADIENT_TABLE.finditer(source)
                )
            if "_boundary_operator" not in os.path.basename(path):
                volume[element] = True
                by_element[element] += len(re.findall(r"for \(int q = 0", source))
                # The one-point rule's own tables, named by this element's
                # operator.  Only the `_q1` families: a P1 element carrying a
                # higher rule -- `ref_tet4_q11`, for a form whose integrand is
                # not constant over the cell -- is integrating, not spelling
                # constants, and is not what rule two is about.
                reference_data[element] += len(
                    re.findall(
                        r"sfem::codegen::(?:ref_(?:tet4|tri3)|quad_(?:tet|tri))_q1<",
                        source,
                    )
                )
        for element, quadrature_loops in sorted(by_element.items()):
            rows.append(
                {
                    "material": material,
                    "element": element,
                    "family": element_family(element),
                    "sum_factorised": factorised,
                    "quadrature_loops": quadrature_loops,
                    "reference_data": reference_data.get(element, 0),
                    "facet_gradients": facet_gradients.get(element, 0),
                    "has_volume_kernels": volume.get(element, False),
                }
            )
    return rows


def violations(rows):
    """Where the generated tree departs from the three defaults."""
    found = []
    for row in rows:
        if (
            row["family"] == "tensor-product"
            and row.get("has_volume_kernels", True)
            and not row["sum_factorised"]
        ):
            found.append(
                (
                    "tensor-product without sum factorization",
                    "%s/%s" % (row["material"], row["element"]),
                )
            )
        if row["family"] == "simplex-lowest" and row["quadrature_loops"]:
            found.append(
                (
                    "lowest-order simplex generating quadrature data",
                    "%s/%s (%d loops)"
                    % (row["material"], row["element"], row["quadrature_loops"]),
                )
            )
        if row["family"] == "simplex-lowest" and row.get("reference_data"):
            found.append(
                (
                    "lowest-order simplex reading reference tables",
                    "%s/%s (%d references)"
                    % (row["material"], row["element"], row["reference_data"]),
                )
            )
        if row.get("facet_gradients"):
            found.append(
                (
                    "flat facet carrying one gradient row per quadrature point",
                    "%s/%s (%d tables)"
                    % (row["material"], row["element"], row["facet_gradients"]),
                )
            )
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("generated", help="a captured generated tree")
    args = parser.parse_args(argv)

    root = args.generated
    if os.path.isdir(os.path.join(root, "generated")):
        root = os.path.join(root, "generated")
    rows = survey(root)
    if not rows:
        print("no generated materials under %s" % root)
        return 2

    print(
        "%-24s %-16s %-16s %10s %10s %10s %11s"
        % (
            "material", "element", "family", "sum-fact", "qp loops", "ref data",
            "facet grad",
        )
    )
    for row in rows:
        if row["family"] == "mixed":
            continue
        print(
            "%-24s %-16s %-16s %10s %10d %10d %11d"
            % (
                row["material"],
                row["element"],
                row["family"],
                "yes" if row["sum_factorised"] else "NO",
                row["quadrature_loops"],
                row["reference_data"],
                row.get("facet_gradients", 0),
            )
        )

    found = violations(rows)
    counts = {}
    for kind, _ in found:
        counts[kind] = counts.get(kind, 0) + 1
    print("\ndepartures from the element's default strategy:")
    for kind, count in sorted(counts.items()):
        print("    %-46s %4d" % (kind, count))
    print("    %-46s %4d" % ("total", len(found)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
