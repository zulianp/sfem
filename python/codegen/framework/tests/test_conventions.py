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
        derived |= set(conventions.QUALIFIERS.values())
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
        for name in ("g_metric", "pk_u", "sh_tile", "gl_tile"):
            with self.subTest(name=name):
                with self.assertRaises(conventions.NameCollision):
                    conventions.check_material("demo", [name])

    def test_a_one_character_prefix_is_not_reserved(self):
        """`b` prefixes staged buffers, but reserving it would refuse `beta`.

        A single letter is far more of the material author's namespace than the
        generator can claim.  The buffers are protected by their full names.
        """
        self.assertTrue(conventions.check_material("demo", ["beta", "b0", "bulk"]))
        for prefix in conventions.RESERVED_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertGreater(len(prefix), 1)

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


class RestrictQualifierTest(unittest.TestCase):
    """The short qualifier is 4.3% of the tree, so it needs exactly one owner."""

    def test_the_prelude_defines_the_short_form_from_the_long_one(self):
        lines = conventions.restrict_prelude()
        short = conventions.QUALIFIERS["restrict"]
        source = conventions.QUALIFIERS["restrict_source"]
        self.assertIn("#define %s %s" % (short, source), lines)
        # Deferring to SFEM's macro rather than redefining `__restrict__` is the
        # point: base/sfem_base.hpp picks `__restrict__` or `__restrict` by
        # compiler, and the generator must not second-guess it.
        self.assertIn("#ifndef %s" % source, lines)

    def test_a_disabled_qualifier_still_defines_the_short_form(self):
        lines = conventions.restrict_prelude("")
        self.assertIn("#define %s" % conventions.QUALIFIERS["restrict_source"], lines)
        self.assertIn(
            "#define %s %s"
            % (conventions.QUALIFIERS["restrict"], conventions.QUALIFIERS["restrict_source"]),
            lines,
        )

    def test_the_long_form_appears_only_in_the_prelude(self):
        """Emitted signatures use the short form; the long one is the alias target.

        A signature that still spells `SFEM_RESTRICT` is a site that did not go
        through `restrict_prelude`, and it is invisible until something parses
        the declaration -- `tools/reproducibility.py` matches parameter types by
        exact text and turns an unrecognised one into a silent `skipped`.
        """
        import os

        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
            "frontend", "ops", "generated",
        )
        if not os.path.isdir(root):
            self.skipTest("generated tree not present")
        source = conventions.QUALIFIERS["restrict_source"]
        offenders = []
        for base, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith((".cpp", ".hpp")):
                    continue
                path = os.path.join(base, name)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    for number, line in enumerate(handle, 1):
                        if source not in line:
                            continue
                        if line.lstrip().startswith("#"):
                            continue  # the prelude
                        offenders.append("%s:%d" % (path, number))
        self.assertEqual(offenders[:10], [], "%d signatures still spell the long form" % len(offenders))


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
