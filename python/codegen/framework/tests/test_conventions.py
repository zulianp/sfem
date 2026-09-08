"""The naming tables are the generator's vocabulary, and they must stay sound.

`plans/conventions.py` is the editable form of `CONVENTIONS.md`.  Three things
have to hold for it to be worth having, and none of them is obvious by reading:

  the tables are injective   two concepts may not share a spelling, or the
                             abbreviation table quietly merges them
  the reservation is real    a material may not declare a name the generator
                             claims, and the refusal happens at generation
  no material collides       the materials that ship today pass the check

The second and third are what turn `CONVENTIONS.md`'s reserved-namespace section
from a note into a rule.  The first is the check that would have caught folding
`field_grad_ref` onto `grad_ref` -- two distinct objects, one spelling, and a
kernel that does not compile.
"""

import os
import unittest

from codegen.framework.plans import conventions


class ConventionTablesTest(unittest.TestCase):
    def test_the_tables_are_injective(self):
        self.assertTrue(conventions.check_tables())

    def test_every_reserved_name_comes_from_a_table(self):
        """`reserved()` is derived, never typed twice."""
        derived = set(conventions.TYPES.values())
        derived |= set(conventions.CONSTANTS.values())
        derived |= set(conventions.LITERALS.values())
        derived |= set(conventions.INDICES.values())
        self.assertEqual(conventions.reserved(), frozenset(derived))

    def test_no_reserved_name_is_a_bare_single_capital(self):
        """That namespace belongs to the material author.

        `two_phase_flow` publishes `T`, `R` and `Z`; a hyperelastic material
        would reach for `S` for the second Piola-Kirchhoff stress and `G` for a
        shear modulus.  The generator's own names must leave those alone, which
        is why the scalar type is `s_t` rather than `S`.
        """
        capitals = sorted(
            name
            for name in conventions.reserved()
            if len(name) == 1 and name.isupper()
        )
        self.assertEqual(capitals, [])

    def test_g_is_not_claimed_as_a_memory_space(self):
        """`g_` already means geometry at the frozen C ABI."""
        self.assertNotEqual(conventions.SPACES.get("global"), "g_")
        self.assertIn("g_", conventions.RESERVED_PREFIXES)


class ReservationIsEnforcedTest(unittest.TestCase):
    def test_a_clean_material_is_accepted(self):
        self.assertTrue(
            conventions.check_material("demo", ["mu", "lmbda", "u", "p", "T"])
        )

    def test_a_reserved_name_is_refused(self):
        for name in ("s_t", "g_t", "NC", "ND", "VS", "lane", "q"):
            with self.subTest(name=name):
                with self.assertRaises(conventions.NameCollision):
                    conventions.check_material("demo", [name])

    def test_a_reserved_prefix_is_refused(self):
        for name in ("g_metric", "block_u", "pack_u", "sh_tile"):
            with self.subTest(name=name):
                with self.assertRaises(conventions.NameCollision):
                    conventions.check_material("demo", [name])

    def test_the_refusal_names_the_material_and_the_name(self):
        """A collision is the material author's to fix, so both must be in the message."""
        with self.assertRaises(conventions.NameCollision) as caught:
            conventions.check_material("two_phase_flow", ["NC"])
        message = str(caught.exception)
        self.assertIn("two_phase_flow", message)
        self.assertIn("NC", message)
        self.assertIn("CONVENTIONS.md", message)


class ShippedMaterialsTest(unittest.TestCase):
    """The materials that generate today must pass their own rule."""

    def test_no_shipped_material_collides(self):
        from codegen.framework.pipeline import driver

        materials = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials"
        )
        names = sorted(
            f[:-3]
            for f in os.listdir(materials)
            if f.endswith(".py") and not f.startswith("_")
        )
        self.assertGreater(len(names), 4, "expected the material package to be found")
        checked = 0
        for name in names:
            module = __import__(
                "codegen.framework.materials.%s" % name, fromlist=["*"]
            )
            for value in vars(module).values():
                if isinstance(value, driver.CodeGenerator):
                    # Construction runs the check; reaching here is the assertion.
                    checked += 1
        self.assertGreater(checked, 0, "no CodeGenerator found in the materials")


class ComposedNamesTest(unittest.TestCase):
    """A composed name has two owners, and `compose` is the only correct spelling."""

    def test_compose_uses_the_editable_prefix(self):
        self.assertEqual(
            conventions.compose("block", "adj0"),
            conventions.PREFIXES["block"] + "adj0",
        )

    def test_an_unknown_prefix_is_refused(self):
        with self.assertRaises(KeyError):
            conventions.compose("scratch", "u")


if __name__ == "__main__":
    unittest.main()
