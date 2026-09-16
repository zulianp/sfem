"""The generated tree links, and the check that says so is not vacuous.

One defect shape kept coming back, six times, one compiler error apiece: a call
site that knows a symbol's *logical* name but not its *emitted* one.  The
emitted spelling belongs to the target -- `TargetPlatform.entry_point_name`
prefixes `cu_` on CUDA, and `entry_point_suffix_parameters` appends a stream
parameter -- while `plans/conventions` strips prefixes on the way in, because it
parses material, element and operation out of a name and `cu_` is none of those.
An emitter that splices the logical name therefore builds a call that no longer
names, or no longer fits, the function it meant.

Every instance was found by compiling.  That is the wrong gate for two reasons:
it needs a material that happens to publish both halves for the same element
before anything is visible at all, and one instance -- a metric dispatch handed
nine adjugate components where six metric components belong -- had the right
arity for the wrong parameters, compiled cleanly, and would have given a wrong
answer.

`driver._validate_generated_call_graph` runs over the whole emitted file set at
the end of every generation, so both shapes now fail at generation time on every
material and every target.  The tests here pin that it still *can* fail: a
checker that has quietly become a no-op passes every tree, including the broken
ones, and the only way to know is to hand it a broken one.
"""

import unittest

from codegen.framework.pipeline import driver


DEFINITION = '''
extern "C" int cu_material_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    void *const RSTR output,
    void *const stream
) {
  return 0;
}
'''

FORWARDER = '''
extern "C" int cu_material_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    void *const RSTR output,
    void *const stream
) {
  return %s(scalar_bytes, nelements, output%s);
}
'''


def _tree(callee, tail):
    return {"d3/hex8/material_hex8_operator.cu": DEFINITION + FORWARDER % (callee, tail)}


class GeneratedTreeLinksTest(unittest.TestCase):
    def test_a_correct_forwarder_passes(self):
        driver._validate_generated_call_graph(
            _tree("cu_material_hex8_residual_i_msoa", ", stream")
        )

    def test_the_logical_name_at_a_call_site_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            driver._validate_generated_call_graph(
                _tree("material_hex8_residual_i_msoa", ", stream")
            )
        message = str(caught.exception)
        self.assertIn("which nothing defines", message)
        self.assertIn("cu_material_hex8_residual_i_msoa", message)

    def test_a_dropped_suffix_argument_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            driver._validate_generated_call_graph(
                _tree("cu_material_hex8_residual_i_msoa", "")
            )
        message = str(caught.exception)
        self.assertIn("with 3 arguments", message)
        self.assertIn("it is emitted with 4", message)

    def test_an_extra_argument_is_rejected(self):
        # The instance that compiled and answered wrongly passed *more* than the
        # callee wanted in one of its two spellings, so the check is two-sided.
        with self.assertRaises(ValueError) as caught:
            driver._validate_generated_call_graph(
                _tree("cu_material_hex8_residual_i_msoa", ", stream, stream")
            )
        self.assertIn("with 5 arguments", str(caught.exception))

    def test_two_files_may_not_spell_one_symbol_with_different_arities(self):
        files = _tree("cu_material_hex8_residual_i_msoa", ", stream")
        files["op/material_c_abi.hpp"] = (
            'extern "C" int cu_material_hex8_residual_i_msoa('
            "const int scalar_bytes, const ptrdiff_t nelements, void *const output);"
        )
        with self.assertRaises(ValueError) as caught:
            driver._validate_generated_call_graph(files)
        self.assertIn("disagrees with an earlier file", str(caught.exception))

    def test_calls_out_of_the_generated_tree_are_not_this_check_s_business(self):
        # `atomicAdd`, `std::fprintf`, SFEM's own entry points: the tree calls
        # plenty it does not define, and only a name the tree defines under a
        # prefix is evidence of the defect.
        files = _tree("cu_material_hex8_residual_i_msoa", ", stream")
        files["d3/hex8/material_hex8_operator.cu"] += (
            "\nvoid use() { atomicAdd(nullptr, 1.0); "
            "cu_laplacian_apply(1, 2, 3, 4, 5, 6, 7); }\n"
        )
        driver._validate_generated_call_graph(files)

    def test_a_zero_argument_call_counts_as_zero(self):
        files = {
            "d3/hex8/material_hex8_operator.cu": (
                'extern "C" int cu_material_hex8_setup(void) { return 0; }\n'
                "int use() { return cu_material_hex8_setup(); }\n"
            )
        }
        driver._validate_generated_call_graph(files)


if __name__ == "__main__":
    unittest.main()
