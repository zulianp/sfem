import sys
import unittest
from pathlib import Path

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

from common.geometry import (  # noqa: E402
    annular_sector_mesh,
    annulus_mesh,
    box_mesh,
    cylindrical_sector_mesh,
    rectangle_mesh,
    spherical_shell_mesh,
)
from common.mesh import Mesh  # noqa: E402
from common.sets import (  # noqa: E402
    boundary_sides,
    select_boundary_axis,
    surface_geometry,
    validate_sideset_orientation,
)


class GeometryTests(unittest.TestCase):
    def test_rectangle_and_box_counts(self):
        quad = rectangle_mesh(2.0, 1.0, 3, 2, "QUAD4")
        tri = rectangle_mesh(2.0, 1.0, 3, 2, "TRI3")
        hexahedron = box_mesh(2.0, 1.0, 3.0, 3, 2, 2, "HEX8")
        tetrahedron = box_mesh(2.0, 1.0, 3.0, 3, 2, 2, "TET4")

        self.assertEqual((12, 2), quad.points.shape)
        self.assertEqual((6, 4), quad.elements.shape)
        self.assertEqual((12, 3), tri.elements.shape)
        self.assertEqual((36, 3), hexahedron.points.shape)
        self.assertEqual((12, 8), hexahedron.elements.shape)
        self.assertEqual((72, 4), tetrahedron.elements.shape)

    def test_curved_domain_generators_are_deterministic_and_outward(self):
        factories = (
            lambda: annulus_mesh(1.0, 2.0, 2, 12),
            lambda: annulus_mesh(1.0, 2.0, 2, 12, "TRI3"),
            lambda: annular_sector_mesh(1.0, 2.0, 2, 4),
            lambda: annular_sector_mesh(1.0, 2.0, 2, 4, element_type="TRI3"),
            lambda: cylindrical_sector_mesh(1.0, 2.0, 3.0, 2, 4, 2),
            lambda: cylindrical_sector_mesh(1.0, 2.0, 3.0, 2, 4, 2, element_type="TET4"),
            lambda: spherical_shell_mesh(1.0, 2.0, 2, 2),
        )
        for factory in factories:
            with self.subTest(factory=factory):
                first = factory()
                second = factory()
                np.testing.assert_array_equal(first.points, second.points)
                np.testing.assert_array_equal(first.elements, second.elements)
                sides = boundary_sides(first)
                diagnostics = validate_sideset_orientation(first, sides)
                self.assertEqual(sides.size, diagnostics["side_count"])
                self.assertGreater(diagnostics["minimum_orientation_cosine"], 0.0)

    def test_spherical_shell_nodes_lie_on_layer_radii(self):
        mesh = spherical_shell_mesh(2.0, 3.0, 2, 2)
        radii = np.linalg.norm(mesh.points, axis=1)
        np.testing.assert_allclose(np.unique(np.round(radii, 12)), (2.0, 2.5, 3.0))


class SidesetTests(unittest.TestCase):
    def test_axis_selection_has_expected_measure_and_normal(self):
        mesh = box_mesh(2.0, 1.0, 3.0, 2, 1, 3)
        right = select_boundary_axis(mesh, axis=0, value=2.0)
        geometry = surface_geometry(mesh, right)

        self.assertEqual(3, right.size)
        self.assertAlmostEqual(3.0, np.sum(geometry.measures))
        np.testing.assert_allclose(geometry.normals, np.asarray(((1.0, 0.0, 0.0),) * 3))

    def test_orientation_check_rejects_clockwise_element(self):
        mesh = rectangle_mesh(1.0, 1.0, 1, 1)
        inverted = Mesh(mesh.points, mesh.elements[:, (0, 3, 2, 1)], "QUAD4")
        with self.assertRaisesRegex(ValueError, "inward or ambiguous"):
            validate_sideset_orientation(inverted, boundary_sides(inverted))


if __name__ == "__main__":
    unittest.main()
