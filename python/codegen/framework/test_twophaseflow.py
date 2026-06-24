import math
import unittest

import sympy as sp

from .constitutive import TwoPhaseFlowConstitutiveModel
from .symbolic import ExpressionRole
from .twophaseflow import TwoPhaseFlowImplicitEulerModel


def parameter_values(model):
    p = model.parameters
    return {
        p.porosity: 0.2,
        p.residual_water_saturation: 0.1,
        p.reference_capillary_pressure: 1.0e6,
        p.retention_exponent: 2.0,
        p.reference_water_density: 1000.0,
        p.water_compressibility: 4.0e-10,
        p.reference_water_pressure: 15.0e6,
        p.co2_molar_mass: 44.01e-3,
        p.co2_compressibility_factor: 0.9,
        p.gas_constant: 8.314462618,
        p.temperature: 320.0,
        p.water_viscosity: 1.0e-3,
        p.co2_viscosity: 6.0e-5,
        p.water_permeability_exponent: 2.0,
        p.co2_permeability_exponent_1: 2.0,
        p.co2_permeability_exponent_2: 2.0,
    }


class TwoPhaseFlowConstitutiveModelTest(unittest.TestCase):
    def setUp(self):
        self.model = TwoPhaseFlowConstitutiveModel.symbolic()
        self.pw, self.pc = sp.symbols("p_w p_c")
        self.values = parameter_values(self.model)
        self.values.update({self.pw: 15.0e6, self.pc: 15.5e6})

    def test_saturations_are_complementary_and_admissible(self):
        state = self.model.state(self.pw, self.pc)
        self.assertEqual(sp.simplify(state.water_saturation + state.co2_saturation), 1)
        for suction in (0.0, 1.0e3, 5.0e5, 5.0e6, 5.0e7):
            values = dict(self.values)
            values[self.pc] = values[self.pw] + suction
            sw = float(state.water_saturation.evalf(subs=values))
            sc = float(state.co2_saturation.evalf(subs=values))
            self.assertGreaterEqual(sw, 0.0)
            self.assertLessEqual(sw, 1.0)
            self.assertGreaterEqual(sc, 0.0)
            self.assertLessEqual(sc, 1.0)

    def test_pressure_derivatives_match_centered_finite_differences(self):
        state = self.model.state(self.pw, self.pc)
        derivatives = self.model.pressure_derivatives(self.pw, self.pc)
        step = 10.0
        for name, expression in state.as_dict().items():
            scale = max(1.0, abs(float(expression.evalf(subs=self.values))))
            for pressure, derivative in zip(
                (self.pw, self.pc),
                derivatives[name],
            ):
                plus = dict(self.values)
                minus = dict(self.values)
                plus[pressure] += step
                minus[pressure] -= step
                finite_difference = (
                    float(expression.evalf(subs=plus))
                    - float(expression.evalf(subs=minus))
                ) / (2.0 * step)
                symbolic = float(derivative.evalf(subs=self.values))
                self.assertTrue(math.isfinite(symbolic), name)
                self.assertLessEqual(
                    abs(symbolic - finite_difference),
                    2.0e-7 * scale / step + 1.0e-12,
                    "%s derivative with respect to %s" % (name, pressure),
                )

    def test_parameter_and_state_validation(self):
        validated = self.model.parameters.validate(self.values)
        self.assertEqual(validated["porosity"], 0.2)
        self.model.validate_state(15.0e6, 15.5e6)

        invalid = dict(self.values)
        invalid[self.model.parameters.water_viscosity] = 0.0
        with self.assertRaisesRegex(ValueError, "water_viscosity"):
            self.model.parameters.validate(invalid)
        invalid = dict(self.values)
        invalid[self.model.parameters.retention_exponent] = 0.5
        with self.assertRaisesRegex(ValueError, "retention_exponent"):
            self.model.parameters.validate(invalid)
        with self.assertRaisesRegex(ValueError, "capillary pressure"):
            self.model.validate_state(15.5e6, 15.0e6)

    def test_builds_codegen_expression_graph_with_named_outputs(self):
        graph = self.model.build_expression_graph(
            self.pw,
            self.pc,
            include_derivatives=True,
        )
        names = {output.name for output in graph.outputs}
        self.assertIn("water_saturation", names)
        self.assertIn("water_mobility", names)
        self.assertIn("d_water_saturation_d_pw", names)
        self.assertIn("d_co2_mobility_d_pc", names)
        self.assertGreater(graph.cost.flops, 0)
        self.assertGreater(graph.cost.temporaries, 0)


class TwoPhaseFlowImplicitEulerModelTest(unittest.TestCase):
    def setUp(self):
        self.model = TwoPhaseFlowImplicitEulerModel.symbolic(2)
        self.pw, self.pc, self.pw_old, self.pc_old, self.dt = sp.symbols(
            "p_w p_c p_w_old p_c_old dt"
        )
        self.grad_pw = sp.Matrix(sp.symbols("grad_pw_0:2"))
        self.grad_pc = sp.Matrix(sp.symbols("grad_pc_0:2"))
        self.grad_vw = sp.Matrix(sp.symbols("grad_vw_0:2"))
        self.grad_vc = sp.Matrix(sp.symbols("grad_vc_0:2"))
        self.hw, self.hc = sp.symbols("h_w h_c")
        self.grad_hw = sp.Matrix(sp.symbols("grad_hw_0:2"))
        self.grad_hc = sp.Matrix(sp.symbols("grad_hc_0:2"))
        self.vw, self.vc = sp.symbols("v_w v_c")
        self.permeability = sp.Matrix(2, 2, sp.symbols("K_0:4"))
        self.values = parameter_values(self.model.constitutive)
        self.values.update(
            {
                self.pw: 15.0e6,
                self.pc: 15.5e6,
                self.pw_old: 14.9e6,
                self.pc_old: 15.3e6,
                self.dt: 10.0,
                self.vw: 0.7,
                self.vc: -0.2,
                self.grad_pw[0]: 2.0,
                self.grad_pw[1]: -1.0,
                self.grad_pc[0]: -0.5,
                self.grad_pc[1]: 1.5,
                self.grad_vw[0]: 0.25,
                self.grad_vw[1]: -0.75,
                self.grad_vc[0]: -0.4,
                self.grad_vc[1]: 0.6,
                self.hw: 2.0e4,
                self.hc: -1.5e4,
                self.grad_hw[0]: 0.3,
                self.grad_hw[1]: -0.2,
                self.grad_hc[0]: -0.1,
                self.grad_hc[1]: 0.4,
                self.permeability[0, 0]: 2.0e-13,
                self.permeability[0, 1]: 0.3e-13,
                self.permeability[1, 0]: 0.3e-13,
                self.permeability[1, 1]: 1.5e-13,
            }
        )

    def linearization(self):
        return self.model.linearized_weak_residual(
            self.pw,
            self.pc,
            self.pw_old,
            self.pc_old,
            self.grad_pw,
            self.grad_pc,
            self.hw,
            self.hc,
            self.grad_hw,
            self.grad_hc,
            self.vw,
            self.vc,
            self.grad_vw,
            self.grad_vc,
            self.permeability,
            self.dt,
        )

    def state(self):
        return self.model.weak_residual(
            self.pw,
            self.pc,
            self.pw_old,
            self.pc_old,
            self.grad_pw,
            self.grad_pc,
            self.vw,
            self.vc,
            self.grad_vw,
            self.grad_vc,
            self.permeability,
            self.dt,
        )

    def test_constant_unchanged_pressures_have_zero_volume_residual(self):
        zero = sp.zeros(2, 1)
        state = self.model.weak_residual(
            self.pw,
            self.pc,
            self.pw,
            self.pc,
            zero,
            zero,
            self.vw,
            self.vc,
            self.grad_vw,
            self.grad_vc,
            self.permeability,
            self.dt,
        )
        self.assertEqual(sp.simplify(state.water_residual), 0)
        self.assertEqual(sp.simplify(state.co2_residual), 0)

    def test_accumulation_flux_and_residual_match_python_reference(self):
        state = self.state()
        actual = {
            "water_accumulation": float(state.water_accumulation.evalf(subs=self.values)),
            "co2_accumulation": float(state.co2_accumulation.evalf(subs=self.values)),
            "water_flux": tuple(
                float(value.evalf(subs=self.values)) for value in state.water_mass_flux
            ),
            "co2_flux": tuple(
                float(value.evalf(subs=self.values)) for value in state.co2_mass_flux
            ),
            "water_residual": float(state.water_residual.evalf(subs=self.values)),
            "co2_residual": float(state.co2_residual.evalf(subs=self.values)),
        }
        expected = direct_implicit_euler_reference(self.values, self.model)
        for name in ("water_accumulation", "co2_accumulation", "water_residual", "co2_residual"):
            scale = max(1.0, abs(expected[name]))
            self.assertLessEqual(abs(actual[name] - expected[name]), 1.0e-12 * scale)
        for name in ("water_flux", "co2_flux"):
            for actual_value, expected_value in zip(actual[name], expected[name]):
                scale = max(1.0, abs(expected_value))
                self.assertLessEqual(abs(actual_value - expected_value), 1.0e-12 * scale)

    def test_builds_one_coupled_residual_expression_graph(self):
        graph = self.model.build_expression_graph(
            self.pw,
            self.pc,
            self.pw_old,
            self.pc_old,
            self.grad_pw,
            self.grad_pc,
            self.vw,
            self.vc,
            self.grad_vw,
            self.grad_vc,
            self.permeability,
            self.dt,
        )
        self.assertEqual(tuple(output.role for output in graph.outputs), (ExpressionRole.RESIDUAL,) * 2)
        self.assertEqual(
            tuple(output.name for output in graph.outputs),
            ("water_residual", "co2_residual"),
        )
        self.assertGreater(graph.cost.flops, 0)
        self.assertGreater(graph.cost.temporaries, 0)

    def test_rejects_invalid_dimensions_and_time_step(self):
        with self.assertRaisesRegex(ValueError, "water_pressure_gradient"):
            self.model.weak_residual(
                self.pw,
                self.pc,
                self.pw_old,
                self.pc_old,
                sp.zeros(3, 1),
                self.grad_pc,
                self.vw,
                self.vc,
                self.grad_vw,
                self.grad_vc,
                self.permeability,
                self.dt,
            )
        with self.assertRaisesRegex(ValueError, "dt"):
            self.model.weak_residual(
                self.pw,
                self.pc,
                self.pw_old,
                self.pc_old,
                self.grad_pw,
                self.grad_pc,
                self.vw,
                self.vc,
                self.grad_vw,
                self.grad_vc,
                self.permeability,
                0,
            )

    def test_jacobian_blocks_match_centered_finite_differences(self):
        state = self.state()
        linearization = self.linearization()
        epsilon = 1.0e-5
        block_cases = (
            (
                "ww",
                state.water_residual,
                (self.pw,) + tuple(self.grad_pw),
                (self.hw,) + tuple(self.grad_hw),
            ),
            (
                "wc",
                state.water_residual,
                (self.pc,) + tuple(self.grad_pc),
                (self.hc,) + tuple(self.grad_hc),
            ),
            (
                "cw",
                state.co2_residual,
                (self.pw,) + tuple(self.grad_pw),
                (self.hw,) + tuple(self.grad_hw),
            ),
            (
                "cc",
                state.co2_residual,
                (self.pc,) + tuple(self.grad_pc),
                (self.hc,) + tuple(self.grad_hc),
            ),
        )
        for name, residual, variables, directions in block_cases:
            plus = dict(self.values)
            minus = dict(self.values)
            for variable, direction in zip(variables, directions):
                increment = epsilon * self.values[direction]
                plus[variable] += increment
                minus[variable] -= increment
            finite_difference = (
                float(residual.evalf(subs=plus))
                - float(residual.evalf(subs=minus))
            ) / (2.0 * epsilon)
            symbolic = float(getattr(linearization, name).evalf(subs=self.values))
            scale = max(1.0, abs(finite_difference))
            self.assertLessEqual(
                abs(symbolic - finite_difference),
                2.0e-7 * scale,
                name,
            )

    def test_combined_action_cross_blocks_and_merit(self):
        state = self.state()
        linearization = self.linearization()
        ww = float(linearization.ww.evalf(subs=self.values))
        wc = float(linearization.wc.evalf(subs=self.values))
        cw = float(linearization.cw.evalf(subs=self.values))
        cc = float(linearization.cc.evalf(subs=self.values))
        self.assertNotEqual(wc, 0.0)
        self.assertNotEqual(cw, 0.0)
        self.assertAlmostEqual(
            float(linearization.water_action.evalf(subs=self.values)),
            ww + wc,
        )
        self.assertAlmostEqual(
            float(linearization.co2_action.evalf(subs=self.values)),
            cw + cc,
        )
        residual_values = [
            float(value.evalf(subs=self.values)) for value in state.residual
        ]
        expected_merit = 0.5 * sum(value * value for value in residual_values)
        self.assertAlmostEqual(
            float(linearization.merit.evalf(subs=self.values)),
            expected_merit,
        )
        self.assertGreaterEqual(expected_merit, 0.0)

    def test_builds_linearized_graph_with_cost_diagnostics(self):
        graph = self.model.build_linearized_expression_graph(
            self.pw,
            self.pc,
            self.pw_old,
            self.pc_old,
            self.grad_pw,
            self.grad_pc,
            self.hw,
            self.hc,
            self.grad_hw,
            self.grad_hc,
            self.vw,
            self.vc,
            self.grad_vw,
            self.grad_vc,
            self.permeability,
            self.dt,
        )
        names = tuple(output.name for output in graph.outputs)
        self.assertEqual(
            names,
            (
                "jacobian_ww",
                "jacobian_wc",
                "jacobian_cw",
                "jacobian_cc",
                "water_jacobian_action",
                "co2_jacobian_action",
                "residual_norm_merit",
            ),
        )
        self.assertEqual(
            tuple(output.role for output in graph.outputs[:6]),
            (ExpressionRole.JACOBIAN_ACTION,) * 6,
        )
        self.assertEqual(graph.outputs[6].role, ExpressionRole.MERIT)
        self.assertGreater(graph.cost.flops, 0)
        self.assertGreater(graph.cost.temporaries, 0)
        self.assertGreater(graph.cost.exps, 0)


def direct_implicit_euler_reference(values, model):
    p = model.constitutive.parameters
    pw = values[sp.Symbol("p_w")]
    pc = values[sp.Symbol("p_c")]
    pw_old = values[sp.Symbol("p_w_old")]
    pc_old = values[sp.Symbol("p_c_old")]

    def constitutive(water_pressure, co2_pressure):
        suction = co2_pressure - water_pressure
        s_res = values[p.residual_water_saturation]
        exponent = values[p.retention_exponent]
        sw = s_res + (1.0 - s_res) * (
            1.0
            + (suction / values[p.reference_capillary_pressure]) ** exponent
        ) ** (1.0 / exponent - 1.0)
        sc = 1.0 - sw
        se = (sw - s_res) / (1.0 - s_res)
        rho_w = values[p.reference_water_density] * math.exp(
            values[p.water_compressibility]
            * (water_pressure - values[p.reference_water_pressure])
        )
        rho_c = (
            values[p.co2_molar_mass]
            * co2_pressure
            / (
                values[p.co2_compressibility_factor]
                * values[p.gas_constant]
                * values[p.temperature]
            )
        )
        krw = math.sqrt(sw) * (
            1.0
            - (
                1.0
                - sw ** (1.0 / values[p.water_permeability_exponent])
            )
            ** values[p.water_permeability_exponent]
        ) ** 2
        krc = (1.0 - se) ** values[p.co2_permeability_exponent_1] * (
            1.0 - se ** values[p.co2_permeability_exponent_2]
        )
        return sw, sc, rho_w, rho_c, krw, krc

    sw, sc, rho_w, rho_c, krw, krc = constitutive(pw, pc)
    sw_old, sc_old, rho_w_old, rho_c_old, _, _ = constitutive(pw_old, pc_old)
    porosity = values[p.porosity]
    dt = values[sp.Symbol("dt")]
    accumulation_w = porosity * (sw * rho_w - sw_old * rho_w_old) / dt
    accumulation_c = porosity * (sc * rho_c - sc_old * rho_c_old) / dt
    K = (
        (values[sp.Symbol("K_0")], values[sp.Symbol("K_1")]),
        (values[sp.Symbol("K_2")], values[sp.Symbol("K_3")]),
    )
    grad_pw = (values[sp.Symbol("grad_pw_0")], values[sp.Symbol("grad_pw_1")])
    grad_pc = (values[sp.Symbol("grad_pc_0")], values[sp.Symbol("grad_pc_1")])

    def matvec(matrix, vector):
        return tuple(sum(row[j] * vector[j] for j in range(2)) for row in matrix)

    water_flux = tuple(
        -rho_w * krw / values[p.water_viscosity] * value
        for value in matvec(K, grad_pw)
    )
    co2_flux = tuple(
        -rho_c * krc / values[p.co2_viscosity] * value
        for value in matvec(K, grad_pc)
    )
    grad_vw = (values[sp.Symbol("grad_vw_0")], values[sp.Symbol("grad_vw_1")])
    grad_vc = (values[sp.Symbol("grad_vc_0")], values[sp.Symbol("grad_vc_1")])
    return {
        "water_accumulation": accumulation_w,
        "co2_accumulation": accumulation_c,
        "water_flux": water_flux,
        "co2_flux": co2_flux,
        "water_residual": accumulation_w * values[sp.Symbol("v_w")]
        - sum(water_flux[i] * grad_vw[i] for i in range(2)),
        "co2_residual": accumulation_c * values[sp.Symbol("v_c")]
        - sum(co2_flux[i] * grad_vc[i] for i in range(2)),
    }


if __name__ == "__main__":
    unittest.main()
