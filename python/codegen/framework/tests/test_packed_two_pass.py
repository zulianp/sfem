"""Emitter coverage for packed matrix-free two-pass apply/gradient."""

from __future__ import annotations

import unittest
from pathlib import Path


class PackedTwoPassEmissionTests(unittest.TestCase):
    def test_energy_codegen_contains_two_pass_scatter_modes(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "emitters"
            / "energy_codegen.py"
        )
        text = path.read_text()
        self.assertIn('for pass_mode in ("one_pass", "two_pass")', text)
        self.assertIn("_packed_two_pass_", text)
        self.assertIn("ghost_reduce_ptr", text)
        self.assertIn("n_ghost_reduce_rows", text)
        self.assertIn("ghost_buf", text)

    def test_residual_codegen_accepts_two_pass_flag(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "emitters"
            / "residual_codegen.py"
        )
        text = path.read_text()
        self.assertIn("two_pass=False", text)
        self.assertIn("jacobian_action_packed_two_pass_isoparametric_mesh_soa", text)


if __name__ == "__main__":
    unittest.main()
