"""The BSR assembly on the projected tangent is opt-in, and what it publishes when opted into.

A material that asks for it (`CodeGenerator.tangent_projection`) gives its element-matrix
kernel and its assembly a `PROJECT` template flag that projects each point's tangent D to
positive semidefinite (`project_abs_symmetric`), publishes the assembly with the flag on
as a second entry point, `<unit>_<element>_hessian_bsr_psd_<geometry>`, and gives its Op
the switch to it (`Op::set_tangent_projection`).  The exact assembly instantiates the flag
off, so it is the kernel it was.  These tests pin both halves: that asking nothing changes
nothing, and that asking produces exactly that.

The numerical check is the library's `sfem_TangentProjectionTest` and the torsion case.
"""

import dataclasses
import importlib
import os
import re
import sys
import unittest


MATERIALS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials"
)


def _material(name):
    if MATERIALS not in sys.path:
        sys.path.insert(0, MATERIALS)
    return importlib.import_module(name).material


def _generate(name, element, opt_in, target="openmp"):
    import tempfile

    from sfem import gen

    material = dataclasses.replace(_material(name), tangent_projection=opt_in)
    with tempfile.TemporaryDirectory() as out_dir:
        result = gen.generate(material, out_dir, elements=(element,), clean=True, target=target)
        files = {}
        for path in result.sources:
            relative = os.path.relpath(path, out_dir)
            relative = relative[3:] if relative.startswith("../") else relative
            with open(path, encoding="utf-8") as handle:
                files[relative] = handle.read()
        return files


def _entry_point(source, name):
    """The definition of `extern "C" int <name>(`, to its closing brace at column zero."""
    start = source.index('extern "C" int %s(' % name)
    return source[start : source.index("\n}\n", start) + 3]


class TangentProjectionGenerationTest(unittest.TestCase):
    _cache = {}

    def files(self, element, opt_in):
        key = (element, opt_in)
        if key not in self._cache:
            self._cache[key] = _generate("neohookean_ogden", element, opt_in)
        return self._cache[key]

    def test_it_is_off_by_default(self):
        from sfem import gen

        defaults = {f.name: f.default for f in dataclasses.fields(gen.CodeGenerator)}
        self.assertFalse(defaults["tangent_projection"])
        self.assertFalse(_material("neohookean_ogden").tangent_projection)

    def test_it_needs_the_bsr_format(self):
        from codegen.framework.plans.matrix_formats import matrix_format_plan_from_request

        with self.assertRaises(ValueError):
            matrix_format_plan_from_request(("crs",), tangent_projection=True)

    def test_nothing_is_emitted_unless_asked(self):
        for name, source in self.files("TET10", False).items():
            for absent in ("PROJECT", "_psd_", "project_abs_symmetric", "TangentProjection"):
                with self.subTest(file=name, absent=absent):
                    self.assertNotIn(absent, source)

    def test_the_element_matrix_projects_each_point_tangent(self):
        source = self.files("TET10", True)["d3/neohookean_ogden_d3_simplex_hessian.hpp"]
        self.assertIn('#include "sfem_TangentProjection.hpp"', source)
        self.assertIn("template <typename s_t, int NQ, int NS, bool PROJECT = false>", source)
        # After D is complete, its last mirrored entry included.
        self.assertRegex(
            source,
            r"D\[79\] = D\[71\];\n\s*if constexpr \(PROJECT\) \{\n\s*project_abs_symmetric<s_t, 9>\(D\);\n\s*\}",
        )

    def test_the_projected_assembly_is_a_second_entry_point(self):
        source = self.files("TET10", True)["d3/tet10/neohookean_ogden_tet10_operator.cpp"]
        exact = _entry_point(source, "neohookean_ogden_tet10_hessian_bsr_i_msoa")
        projected = _entry_point(source, "neohookean_ogden_tet10_hessian_bsr_psd_i_msoa")
        self.assertNotIn("true>", exact)
        self.assertEqual(len(re.findall(r", geom_t, 1, true>\(", projected)), 3)
        self.assertEqual(
            projected.replace("_hessian_bsr_psd_", "_hessian_bsr_").replace(", 1, true>(", ", 1>("),
            exact,
        )
        self.assertIn("int FORMAT, bool PROJECT = false>", source)
        self.assertIn("direct_hessian_reference_element_matrix<s_t, NQ, NS, PROJECT>(", source)

    def test_the_exact_assembly_is_unchanged(self):
        """The exact entry point and the kernels it instantiates read the same with the
        flag off: only the template parameter list and the hook are new."""
        plain = self.files("TET10", False)["d3/tet10/neohookean_ogden_tet10_operator.cpp"]
        opted = self.files("TET10", True)["d3/tet10/neohookean_ogden_tet10_operator.cpp"]
        name = "neohookean_ogden_tet10_hessian_bsr_i_msoa"
        self.assertEqual(_entry_point(plain, name), _entry_point(opted, name))

    def test_the_closed_form_has_nothing_to_project(self):
        files = self.files("TET4", True)
        source = files["d3/tet4/neohookean_ogden_tet4_operator.cpp"]
        self.assertNotIn("_psd_", source)
        self.assertNotIn("tangent_projection_supported", files["op/sfem_GeneratedNeoHookeanOgden.hpp"])

    def test_the_tensor_product_point_tangent_is_projected(self):
        files = self.files("HEX8", True)
        hessian = files["d3/neohookean_ogden_d3_tensor_product_hessian.hpp"]
        self.assertIn("if constexpr (PROJECT) {", hessian)
        self.assertIn("project_abs_symmetric<s_t, 9>(D);", hessian)
        # HEX8 reaches its PROTEUS twin through the node permutation, projected or not.
        operator = files["d3/hex8/neohookean_ogden_hex8_operator.cpp"]
        self.assertIn("neohookean_ogden_proteus_hex8_hessian_bsr_psd_i_msoa(", operator)

    def test_the_op_switches_to_the_projected_assembly(self):
        files = self.files("TET10", True)
        header = files["op/sfem_GeneratedNeoHookeanOgden.hpp"]
        source = files["op/sfem_GeneratedNeoHookeanOgden.cpp"]
        self.assertIn("bool tangent_projection_supported() const override;", header)
        self.assertIn("void set_tangent_projection(const bool on) override;", header)
        self.assertIn("bool project_tangent{false};", source)
        self.assertIn("if (!(type == smesh::TET10)) return false;", source)
        body = source[source.index("::hessian_bsr_typed("):]
        body = body[: body.index("\n  }\n")]
        self.assertRegex(
            body,
            r"if \(impl_->project_tangent\) \{\n\s*return neohookean_ogden_hessian_bsr_psd_3d_i_msoa\(",
        )

    def test_the_device_projects_as_the_host_does(self):
        files = _generate("neohookean_ogden", "TET10", True, target="cuda")
        hessian = files["d3/cuda/neohookean_ogden_d3_simplex_hessian.cuh"]
        self.assertIn('#include "sfem_TangentProjection.hpp"', hessian)
        self.assertIn("project_abs_symmetric<s_t, 9>(D);", hessian)
        operator = files["d3/tet10/cuda/neohookean_ogden_tet10_operator.cu"]
        self.assertIn('extern "C" int cu_neohookean_ogden_tet10_hessian_bsr_psd_i_msoa(', operator)
        source = files["op/cuda/sfem_GeneratedNeoHookeanOgden_cuda.cpp"]
        self.assertIn("::set_tangent_projection(const bool on)", source)
        self.assertIn("return cu_neohookean_ogden_hessian_bsr_psd_3d_i_msoa(", source)

    def test_an_op_without_it_declares_nothing(self):
        files = self.files("TET10", False)
        self.assertNotIn("tangent_projection", files["op/sfem_GeneratedNeoHookeanOgden.hpp"])
        self.assertNotIn("project_tangent", files["op/sfem_GeneratedNeoHookeanOgden.cpp"])


if __name__ == "__main__":
    unittest.main()
