#!/usr/bin/env python3
"""Tests for the paper-specific Stokes verification helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from drivers.verification.run_stokes_convergence import Level, level_errors
from drivers.verification.stokes_mms import CASES, case_by_name


class StokesVerificationTest(unittest.TestCase):
    def _write_exact_level(self, root: Path, case_name: str, coords) -> Level:
        case = case_by_name(case_name)
        mesh = root / case_name / "mesh"
        solution = root / case_name / "solution"
        mesh.mkdir(parents=True)
        solution.mkdir(parents=True)

        stored_coords = tuple(np.asarray(values, dtype=np.float32) for values in coords)
        for name, values in zip(("x", "y", "z")[: case.dim], stored_coords):
            values.tofile(mesh / ("%s.raw" % name))

        real_coords = tuple(values.astype(np.float64) for values in stored_coords)
        for d, values in enumerate(case.velocity(*real_coords)):
            values.astype(np.float64).tofile(solution / ("u%d.raw" % d))
        case.pressure(*real_coords).astype(np.float64).tofile(solution / "p.raw")
        return Level(case_name, 1.0, mesh, solution)

    def _write_exact_typed_level(self, root: Path, case_name: str, coords) -> Level:
        case = case_by_name(case_name)
        mesh = root / ("%s_typed" % case_name) / "mesh"
        solution = root / ("%s_typed" % case_name) / "solution"
        mesh.mkdir(parents=True)
        solution.mkdir(parents=True)

        stored_coords = tuple(np.asarray(values, dtype=np.float32) for values in coords)
        for name, values in zip(("x", "y", "z")[: case.dim], stored_coords):
            values.tofile(mesh / ("%s.float32" % name))

        real_coords = tuple(values.astype(np.float64) for values in stored_coords)
        for d, values in enumerate(case.velocity(*real_coords)):
            values.astype(np.float64).tofile(solution / ("u%d.float64" % d))
        case.pressure(*real_coords).astype(np.float64).tofile(solution / "p.float64")
        return Level(case_name, 1.0, mesh, solution)

    def test_paper_cases_are_available(self):
        self.assertIn("bercovier_engelman_2d", CASES)
        self.assertIn("taylor_green_3d", CASES)
        self.assertEqual(case_by_name("bercovier_engelman_2d").paper_section, "3.1")
        self.assertEqual(case_by_name("taylor_green_3d").paper_section, "3.2")

    def test_taylor_green_paper_force_is_x_component_only_at_unit_viscosity(self):
        case = case_by_name("taylor_green_3d")
        x = np.array([0.125, 0.25, 0.375])
        y = np.array([0.2, 0.3, 0.4])
        z = np.array([0.15, 0.35, 0.55])
        fx, fy, fz = case.forcing(1.0, x, y, z)
        expected_fx = (
            36.0
            * np.pi
            * np.pi
            * np.cos(2.0 * np.pi * x)
            * np.sin(2.0 * np.pi * y)
            * np.sin(2.0 * np.pi * z)
        )
        np.testing.assert_allclose(fx, expected_fx)
        np.testing.assert_allclose(fy, 0.0, atol=1e-14)
        np.testing.assert_allclose(fz, 0.0, atol=1e-14)

    def test_exact_2d_and_3d_fields_have_zero_collector_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            levels = (
                self._write_exact_level(
                    root,
                    "bercovier_engelman_2d",
                    (
                        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                    ),
                ),
                self._write_exact_level(
                    root,
                    "taylor_green_3d",
                    (
                        np.array([0.0, 0.25, 0.5, 0.75]),
                        np.array([0.125, 0.375, 0.625, 0.875]),
                        np.array([0.2, 0.4, 0.6, 0.8]),
                    ),
                ),
            )
            for level in levels:
                errors = level_errors(level.name, level, pressure_mean_free=False)
                self.assertEqual(errors["velocity_l2_abs"], 0.0)
                self.assertEqual(errors["pressure_l2_abs"], 0.0)

    def test_typed_sfem_output_fields_have_zero_collector_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            level = self._write_exact_typed_level(
                Path(tmp),
                "taylor_green_3d",
                (
                    np.array([0.0, 0.25, 0.5, 0.75]),
                    np.array([0.125, 0.375, 0.625, 0.875]),
                    np.array([0.2, 0.4, 0.6, 0.8]),
                ),
            )
            errors = level_errors(level.name, level, pressure_mean_free=False)
            self.assertEqual(errors["velocity_l2_abs"], 0.0)
            self.assertEqual(errors["pressure_l2_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
