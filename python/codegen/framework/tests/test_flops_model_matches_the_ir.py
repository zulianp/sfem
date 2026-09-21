"""The FLOPs model must describe the kernel the generator printed.

Every consumer of a timing from these kernels divides by

    nelements * (n_qp * flops_per_qp_lane + mesh_flops_per_element)

so the model rescales every GFLOP/s figure the tree can produce.  This checks
it against the arithmetic actually emitted rather than against itself.

The order below is the order the check was built, and it matters.  A simple
user-defined material comes first: two forms on a lowest-order simplex, whose
element bodies are straight-line, so the operation count from the text is
exact and can be confirmed by hand.  Only once the model is right there is it
worth asking anything of the full generated suite -- and what the suite is
asked is the structural property the simple case exposed, that a kernel which
builds geometry and interpolates a field cannot cost nothing beyond its
material evaluation.
"""

import os
import tempfile
import unittest

from sfem import gen

from codegen.framework.plans.flops import (
    ADJUGATE_AND_DETERMINANT_FLOPS_PER_QP,
    closed_form_element_flops,
    element_flops_plan,
    expression_flops,
)
from codegen.framework.plans.affine_element_kernel import expanded_simplex_metric_plan
from codegen.framework.tools.flops_audit import (
    audit,
    count_flops,
    diagnostics_records,
    function_body,
    model_flops_per_element,
)


def _dirichlet_material(name, dimensions):
    """A user-defined material, written the way a user writes one.

    The Dirichlet energy: the smallest form that still exercises the whole
    path -- a gradient, a material parameter, and both a 0-form and the two
    matrix-free kernels derived from it.
    """
    kappa = gen.material_parameter("kappa")
    space = gen.FunctionSpace(gen.FiniteElement("Lagrange", degree=1))
    systems = gen.EquationSystems()
    for dim in dimensions:
        builder = gen.EquationSystemBuilder(dim)
        with gen.geometric_dimension_context(dim):
            u = gen.Function(space, "u")
            gradient = gen.variable(gen.grad(u).T, name="G")
            builder.add_energy(
                "",
                kappa / 2 * gen.inner(gradient, gradient),
                fields=(u,),
                variables=(gradient,),
            )
        systems.add(builder.build())
    return gen.CodeGenerator(
        name,
        systems,
        elements=("TRI3", "TET4"),
        op_name="GeneratedFlopsProbe",
        parameter_defaults=(("kappa", 1.0),),
        matrix_formats=("crs",),
    )


class FlopsModelTest(unittest.TestCase):
    def test_the_operation_counter_reads_c_the_way_the_model_counts_sympy(self):
        """The witness itself, on a body whose cost is obvious by inspection."""
        counted = count_flops(
            """
            const s_t a = b * c + d;
            const s_t e = -a;
            const s_t f = a / b;
            const s_t g = sqrt(f);
            out[i * NS + j] += g;
            """
        )
        # One multiply and one add on the first line; the negation is a sign,
        # not a multiply; the division is weighted 8 and the square root 12;
        # `i * NS + j` is an address, not arithmetic; the `+=` is one add.
        self.assertEqual(counted["mul"], 1)
        self.assertEqual(counted["add"], 2)
        self.assertEqual(counted["div"], 1)
        self.assertEqual(counted["weighted"], 1 + 2 + 8 + 12)

    def test_the_declared_geometry_cost_matches_the_template_it_counts(self):
        """The adjugate and determinant are hand-written, so the count is pinned.

        These are the only numbers in the model that are declared rather than
        derived, because the templates are hand-written C rather than
        expressions the generator builds.  Pinning them here is what keeps a
        declared number from drifting away from the code it describes -- which
        is how the two-dimensional entry came to read 11 for a body containing
        3.
        """
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))),
            "..",
            "frontend",
            "ops",
            "generated",
        )
        with open(os.path.join(root, "geometry_kernels.hpp"), encoding="utf-8") as stream:
            source = stream.read()
        for dim, declared in sorted(ADJUGATE_AND_DETERMINANT_FLOPS_PER_QP.items()):
            body = function_body(
                source,
                "static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant_%d("
                % dim,
            )
            self.assertIsNotNone(body, "no %dD adjugate template to count" % dim)
            self.assertEqual(
                count_flops(body)["weighted"],
                declared,
                "the declared %dD adjugate cost no longer matches the template" % dim,
            )

    def test_the_closed_form_simplex_count_is_the_arithmetic_it_emits(self):
        """The IR count, by hand, before any generation runs.

        The TET4 body is: six metric components scaled by kappa, three nodal
        differences, three flux components of three multiplies and two adds
        each, one negated sum of the three, and four accumulations into the
        element vector.  Fifteen multiplies and fifteen adds.
        """
        class _Metric(object):
            scale = 1

        for dim, expected in ((2, 15), (3, 30)):
            plan = expanded_simplex_metric_plan(
                _Metric(), dim, dim + 1, 1, 1, True, True, False
            )
            self.assertIsNotNone(plan)
            self.assertEqual(
                closed_form_element_flops(plan, scale_is_unit=False), expected
            )

    def test_a_user_defined_material_reports_the_flops_it_emits(self):
        """The model against the emitted text, on a material written here.

        Straight-line bodies only: with no loop there is no trip count to get
        wrong, so the comparison is exact rather than an estimate.  This is the
        case the model used to fail -- it reported 3 for a TET4 body of 30 and
        2 for a TRI3 body of 15, because the mesh term returned zero for every
        element that was not tensor-product.
        """
        material = _dirichlet_material("flops_probe", (2, 3))
        with tempfile.TemporaryDirectory() as out_dir:
            gen.generate(material, out_dir, elements=("TRI3", "TET4"))
            rows = audit(out_dir)
        self.assertTrue(rows, "the probe emitted no straight-line kernel to audit")
        by_element = {}
        for row in rows:
            self.assertEqual(
                row["model"],
                row["counted"],
                "%s: the model claims %d operations and the body contains %d"
                % (row["kernel"], row["model"], row["counted"]),
            )
            by_element[row["element_type"]] = row["counted"]
        # And the counts themselves, so a change that keeps the model and the
        # body agreeing while both drift is still visible.
        self.assertEqual(by_element, {"TRI3": 15, "TET4": 30})

    def test_no_kernel_in_the_tree_claims_its_geometry_is_free(self):
        """The structural property, across the whole generated suite.

        A kernel interpolates a field onto its quadrature points and contracts
        the result back onto the test functions; an isoparametric one also
        builds its Jacobian there.  None of that is free, so an isoparametric
        mesh term of zero means the model is not describing the kernel.  Before
        the model moved into `plans/flops.py` this held for every element that
        was not tensor-product -- 40 of the 44 kernels in the tree.
        """
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))),
            "..",
            "frontend",
            "ops",
            "generated",
        )
        free = []
        seen = 0
        for directory, _subdirectories, files in os.walk(root):
            for name in files:
                if not name.endswith("_operator.cpp"):
                    continue
                path = os.path.join(directory, name)
                with open(path, encoding="utf-8") as stream:
                    records = diagnostics_records(stream.read())
                for kernel, record in sorted(records.items()):
                    seen += 1
                    if model_flops_per_element(record, "isoparametric") <= record[
                        "n_qp"
                    ] * record["flops_per_qp_lane"]:
                        free.append(kernel)
        self.assertGreater(seen, 40, "no diagnostics records were inspected")
        self.assertEqual(free, [], "these kernels claim their geometry costs nothing")


if __name__ == "__main__":
    unittest.main()
