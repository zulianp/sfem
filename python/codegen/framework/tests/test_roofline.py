"""The roofline model must count what the kernel actually moves.

Both of its coordinates are read out of emitted C++, so both can drift silently:
a change in how a kernel is spelled would rescale every intensity the reports
quote without failing anything.  These tests pin the counts against a kernel
whose traffic can be worked out by hand from the element -- HEX8 linear
elasticity's stored apply, 8 nodes, 3 components, 45 tangent components -- and
against the properties the model must have whatever the kernel.
"""

import os
import unittest

from codegen.framework.tools import roofline


HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                         "frontend", "ops", "generated"))
HEX8 = os.path.join(GENERATED, "linear_elasticity", "d3", "hex8",
                    "linear_elasticity_hex8_inexact_apply_inline.hpp")

STORED = "linear_elasticity_hex8_inexact_apply_stored_a_msoa"
COMPRESSED = "linear_elasticity_hex8_inexact_apply_compressed_a_msoa"
TANGENT = "linear_elasticity_hex8_inexact_apply_tangent_a_msoa"


def analyse(kernel, nodes_per_element=1.0, overrides=None):
    source = open(HEX8).read()
    siblings = [
        open(os.path.join(os.path.dirname(HEX8), name)).read()
        for name in sorted(os.listdir(os.path.dirname(HEX8)))
        if name.endswith(".cpp")
    ]
    return roofline.analyse_kernel(
        source, "static SFEM_INLINE int %s_impl(" % kernel,
        roofline.resolve_widths(), nodes_per_element, siblings, overrides,
    )


class RooflineTrafficTest(unittest.TestCase):
    def test_the_stored_apply_moves_what_the_element_says_it_moves(self):
        """8 nodes, 3 components, 45 tangent components, counted by hand.

        Connectivity  8 * 4                       =  32
        Increment     8 * 3 * 8                   = 192
        Tangent      45 * 4                       = 180
        Output        8 * 3 * 8, read and written = 384
        """
        record = analyse(STORED)
        self.assertEqual(record["streamed_bytes_per_element"], 32 + 192 + 180 + 384)

    def test_the_scatter_is_charged_for_the_read_it_does(self):
        """`out[node] += ...` touches the line twice.

        Charging it once would halve the cost of every scatter in the tree, and
        the scatter is the larger half of this kernel's traffic.
        """
        record = analyse(STORED)
        outputs = [name for name in record["arrays"] if name.startswith("out")]
        self.assertEqual(len(outputs), 3)
        for name in outputs:
            self.assertEqual(record["arrays"][name]["updates"], 8)
            self.assertEqual(record["arrays"][name]["reads"], 0)

    def test_the_staging_buffers_are_not_traffic(self):
        """`s_t bhx_0[VS]` is stack scratch, not memory the roofline is drawn against.

        The blocked kernels stage every gathered value into one of these; counting
        them would charge the kernel twice for each value, once at the load and
        once at the reload, and would make the blocking look like a regression in
        intensity when it is the reason the arithmetic vectorises at all.
        """
        record = analyse(STORED)
        staged = [name for name in record["arrays"]
                  if name.startswith("b") and not name.startswith("btangent")]
        self.assertEqual(staged, [])

    def test_the_store_width_comes_from_the_instantiation_not_the_name(self):
        """Both kernels call it `tangent_t`; one is f32 and the other f16.

        Reading the width off the template parameter's name would report the two
        as moving the same bytes, which is precisely the quantity the store's
        precision work is about.
        """
        stored = analyse(STORED)
        compressed = analyse(COMPRESSED)
        self.assertEqual(stored["instantiation"].get("tangent_t"), "metric_tensor_t")
        self.assertEqual(compressed["instantiation"].get("tangent_t"), "compressed_t")
        # 45 components, four bytes against two, plus the compressed kernel's
        # one scaling factor per element.
        self.assertEqual(
            stored["streamed_bytes_per_element"]
            - compressed["streamed_bytes_per_element"],
            45 * 4 - 45 * 2 - 4,
        )

    def test_an_explicit_binding_outranks_the_instantiation(self):
        """So a configuration the ABI does not publish can still be modelled."""
        shipped = analyse(STORED)
        doubled = analyse(STORED, overrides={"tangent_t": 8})
        self.assertEqual(
            doubled["streamed_bytes_per_element"]
            - shipped["streamed_bytes_per_element"],
            45 * 4,
        )

    def test_compulsory_traffic_shares_a_node_between_the_elements_touching_it(self):
        """A node is fetched once, not once per element that reaches it.

        Scaling per *reference* rather than per array was the first version of
        this and made the compulsory model larger than the streamed one, which
        is not a bound at all.
        """
        record = analyse(STORED, nodes_per_element=1.0769)
        self.assertLess(
            record["compulsory_bytes_per_element"],
            record["streamed_bytes_per_element"],
        )
        # The tangent is element-indexed, so it is in both models unchanged.
        self.assertGreater(record["compulsory_bytes_per_element"], 45 * 4)

    def test_a_kernel_reading_no_state_still_models(self):
        """Linear elasticity's tangent takes no connectivity and no state."""
        record = analyse(TANGENT)
        self.assertGreater(record["flops_per_element"], 0)
        self.assertEqual(
            record["streamed_bytes_per_element"],
            record["compulsory_bytes_per_element"],
        )


class RooflineMachineTest(unittest.TestCase):
    def test_narrower_scalars_raise_the_ceiling_and_the_ridge(self):
        for machine in roofline.MACHINES.values():
            self.assertEqual(
                machine.peak_flops(4), 2 * machine.peak_flops(8),
                "%s: a half-width scalar doubles the lane count" % machine.name,
            )
            self.assertGreater(machine.ridge_intensity(4), machine.ridge_intensity(8))

    def test_the_no_simd_ceiling_is_below_the_vector_one(self):
        for machine in roofline.MACHINES.values():
            self.assertLess(machine.scalar_flops(8), machine.peak_flops(8))

    def test_every_machine_says_where_its_numbers_came_from(self):
        """These are quoted peaks, and a report that hides that is misleading."""
        for machine in roofline.MACHINES.values():
            self.assertTrue(machine.source.strip())


class RooflineMeasurementTest(unittest.TestCase):
    def test_a_dof_rate_becomes_a_flop_rate_through_the_kernel_s_own_model(self):
        record = analyse(STORED)
        point = {"nelements": 64000, "ndof": 206763, "mdofs": 100.0}
        expected = 100.0e6 * record["flops_per_element"] * 64000 / 206763.0 / 1e9
        self.assertAlmostEqual(roofline.measured_gflops(point, record), expected, 6)


if __name__ == "__main__":
    unittest.main()
