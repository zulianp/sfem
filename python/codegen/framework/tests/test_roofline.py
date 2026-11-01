"""The roofline model's machines: their ceilings and where their numbers came from.

The traffic model was pinned against the projected (P0) inexact apply, the one
flat kernel body in the tree it could read -- HEX8 linear elasticity's stored
apply, counted by hand.  That kernel is retired: the stored-tangent apply is a
library traversal now (`src/frontend/ops/inexact/`), and the exact kernels call
sub-kernels the analyser does not follow, so the traffic tests have no subject
in the tree until the analyser reads one.
"""

import unittest

from codegen.framework.tools import roofline


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


if __name__ == "__main__":
    unittest.main()
