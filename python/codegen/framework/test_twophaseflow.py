import math
import unittest

import sympy as sp

from .constitutive import TwoPhaseFlowConstitutiveModel


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


if __name__ == "__main__":
    unittest.main()
