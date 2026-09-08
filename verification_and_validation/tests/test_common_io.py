import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

from common.fields import (  # noqa: E402
    read_component_field,
    read_nodal_field,
    write_boundary_values,
    write_initial_field,
)
from common.geometry import rectangle_mesh  # noqa: E402
from common.mesh import read_mesh, write_mesh  # noqa: E402
from common.raw import dtype_from_path, read_raw, typed_raw_name, write_raw  # noqa: E402
from common.sets import (  # noqa: E402
    boundary_sides,
    nodeset_from_sideset,
    read_nodeset,
    read_sideset,
    write_nodeset,
    write_sideset,
)


class RawIOTests(unittest.TestCase):
    def test_typed_raw_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / typed_raw_name("values", np.float64)
            expected = np.asarray((1.25, -2.5, 4.0))
            write_raw(path, expected, np.float64, require_finite=True)

            self.assertEqual(np.dtype(np.float64), dtype_from_path(path))
            np.testing.assert_array_equal(expected, read_raw(path, require_finite=True))

    def test_untyped_raw_requires_explicit_dtype(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "values.raw"
            write_raw(path, np.asarray((1, 2), dtype=np.int32), np.int32)
            with self.assertRaisesRegex(ValueError, "cannot infer"):
                read_raw(path)
            np.testing.assert_array_equal((1, 2), read_raw(path, dtype=np.int32))

    def test_lossy_integer_conversion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "indices.int16.raw"
            with self.assertRaisesRegex(ValueError, "represented exactly"):
                write_raw(path, np.asarray((0, 40000), dtype=np.int64), np.int16)


class MeshIOTests(unittest.TestCase):
    def test_mesh_metadata_round_trip(self):
        mesh = rectangle_mesh(2.0, 1.0, 3, 2, element_type="TRI3")
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "mesh"
            metadata_path = write_mesh(folder, mesh)
            loaded = read_mesh(folder)
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual("TRI3", metadata["element_type"])
            self.assertEqual(mesh.n_points, metadata["n_points"])
            self.assertTrue((folder / "x.float32.raw").is_file())
            self.assertTrue((folder / "i0.int32.raw").is_file())
            np.testing.assert_allclose(mesh.points, loaded.points, atol=1.0e-7)
            np.testing.assert_array_equal(mesh.elements, loaded.elements)

    def test_minimal_legacy_metadata_uses_default_dtypes(self):
        mesh = rectangle_mesh(1.0, 1.0, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            for component, name in enumerate(("x", "y")):
                write_raw(folder / f"{name}.raw", mesh.points[:, component], np.float32)
            for local_node in range(4):
                write_raw(folder / f"i{local_node}.raw", mesh.elements[:, local_node], np.int32)
            (folder / "meta.yaml").write_text("element_type: QUAD4\n", encoding="utf-8")

            loaded = read_mesh(folder)
            np.testing.assert_allclose(mesh.points, loaded.points)
            np.testing.assert_array_equal(mesh.elements, loaded.elements)

    def test_sideset_and_nodeset_round_trip(self):
        mesh = rectangle_mesh(1.0, 1.0, 2, 2)
        sideset = boundary_sides(mesh)
        nodes = nodeset_from_sideset(mesh, sideset)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_sideset(root / "sides", mesh, sideset)
            write_nodeset(root / "nodes.int32.raw", nodes)

            loaded_sides = read_sideset(root / "sides")
            loaded_nodes = read_nodeset(root / "nodes.int32.raw")
            np.testing.assert_array_equal(sideset.parent, loaded_sides.parent)
            np.testing.assert_array_equal(sideset.local_side, loaded_sides.local_side)
            np.testing.assert_array_equal(nodes, loaded_nodes)


class FieldIOTests(unittest.TestCase):
    def test_boundary_and_initial_fields_are_generated_from_coordinates(self):
        mesh = rectangle_mesh(2.0, 1.0, 2, 1)
        nodes = np.asarray((0, 1, 2), dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            values_path = root / "boundary.float64.raw"
            expected_boundary = write_boundary_values(
                values_path, mesh, nodes, lambda points: 2.0 * points[:, 0] - points[:, 1]
            )
            initial_paths = write_initial_field(
                root,
                "initial_displacement",
                mesh,
                lambda points: np.column_stack((points[:, 0], -points[:, 1])),
            )

            np.testing.assert_allclose(expected_boundary, read_nodal_field(values_path, node_count=3))
            loaded_initial = read_component_field(initial_paths, node_count=mesh.n_points)
            np.testing.assert_allclose(loaded_initial[:, 0], mesh.points[:, 0])
            np.testing.assert_allclose(loaded_initial[:, 1], -mesh.points[:, 1])


if __name__ == "__main__":
    unittest.main()
