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

The framework does not work this way yet.  It decides evaluation by
formulation: the energy front end lowers a strong form and contracts it later
through the sum-factorised `tensor_test`, while the residual front end lowers a
weak form that is already contracted and has nothing left to factorise.  So the
strategy a material gets follows from which `add_*` call was used, and Laplace
-- written as a residual, and the most performance-critical operator here --
gets sum factorization on no element at all.

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

TENSOR_PRODUCT = ("hex8", "hex27", "hex64", "hex125", "quad4")
SIMPLEX_LOWEST = ("tet4", "tri3")
SIMPLEX_HIGHER = ("tet10", "tri6")


def element_family(element):
    base = element[len("proteus_"):] if element.startswith("proteus_") else element
    if base in TENSOR_PRODUCT:
        return "tensor-product"
    if base in SIMPLEX_LOWEST:
        return "simplex-lowest"
    if base in SIMPLEX_HIGHER:
        return "simplex-higher"
    return "mixed"


def survey(generated):
    """Per material and element: sum factorization reached, quadrature loops emitted."""
    rows = []
    for material_dir in sorted(glob.glob(os.path.join(generated, "*"))):
        if not os.path.isdir(material_dir):
            continue
        material = os.path.basename(material_dir)
        # Sum factorization is reached through the material's tensor-product
        # local header, so it is a per-material fact, not a per-element one.
        factorised = any(
            "tensor_test<" in open(path, errors="ignore").read()
            for path in glob.glob(os.path.join(material_dir, "**", "*.hpp"), recursive=True)
            if "tensor_product_kernels" not in path
        )
        by_element = {}
        for path in glob.glob(os.path.join(material_dir, "d*", "*", "*")):
            if os.path.isdir(path):
                continue
            element = os.path.basename(os.path.dirname(path))
            source = open(path, errors="ignore").read()
            by_element.setdefault(element, 0)
            by_element[element] += len(re.findall(r"for \(int q = 0", source))
        for element, quadrature_loops in sorted(by_element.items()):
            rows.append(
                {
                    "material": material,
                    "element": element,
                    "family": element_family(element),
                    "sum_factorised": factorised,
                    "quadrature_loops": quadrature_loops,
                }
            )
    return rows


def violations(rows):
    """Where the generated tree departs from the three defaults."""
    found = []
    for row in rows:
        if row["family"] == "tensor-product" and not row["sum_factorised"]:
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
