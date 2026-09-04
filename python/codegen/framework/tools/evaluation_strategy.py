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

Rule two is not.  Lowest-order simplices still generate quadrature loops and
per-point data across every material, and the compact expanded form exists only
as hand-written C transcribed into the emitter for the Laplacian.

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
        for path in glob.glob(os.path.join(material_dir, "d*", "*", "*")):
            if os.path.isdir(path):
                continue
            element = os.path.basename(os.path.dirname(path))
            source = open(path, errors="ignore").read()
            by_element.setdefault(element, 0)
            by_element[element] += len(re.findall(r"for \(int q = 0", source))
            # The rules govern volume kernels.  A material whose only kernels
            # are surface integrals -- the Neumann conditions emit nothing but
            # `*_boundary_operator.cpp` -- is not a tensor-product volume
            # element missing sum factorization; it is a different kind of
            # integral, and counting it as a departure was a category error
            # that put sixteen phantom entries in this report.
            volume.setdefault(element, False)
            if "_boundary_operator" not in os.path.basename(path):
                volume[element] = True
        for element, quadrature_loops in sorted(by_element.items()):
            rows.append(
                {
                    "material": material,
                    "element": element,
                    "family": element_family(element),
                    "sum_factorised": factorised,
                    "quadrature_loops": quadrature_loops,
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

    print("%-24s %-16s %-16s %10s %10s" % ("material", "element", "family", "sum-fact", "qp loops"))
    for row in rows:
        if row["family"] == "mixed":
            continue
        print(
            "%-24s %-16s %-16s %10s %10d"
            % (
                row["material"],
                row["element"],
                row["family"],
                "yes" if row["sum_factorised"] else "NO",
                row["quadrature_loops"],
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
