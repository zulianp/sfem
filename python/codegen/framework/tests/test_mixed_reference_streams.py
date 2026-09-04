"""A mixed kernel's reference buffers, and the signature that must match them.

Which reference-basis buffers a mixed local kernel takes -- shape, the
reference gradients when the form needs them, the quadrature weights -- was
asked six times in the residual emitter: once each in the parameter list, the
pointer setup and the call arguments, and twice over because the
tensor-product and simplex spellings were written out separately.  They agreed
by inspection.

Two assertions, in the shape `tests/test_mesh_geometry_plan.py` uses: the plan
must produce what the signature and the call both spell, and changing the plan
must change both.  The weaker one alone is satisfied by an emitter that
hardcodes the same names and consults nothing.
"""

import unittest

from codegen.framework.plans.reference_data import mixed_reference_streams


class Dependencies:
    def __init__(self, uses_reference_gradients):
        self.uses_reference_gradients = uses_reference_gradients


class MixedReferenceStreamsTest(unittest.TestCase):
    maxDiff = None

    def test_gradients_appear_only_when_the_form_reads_them(self):
        for tensor_product, gradient_name in (
            (True, "field_grad_1d"),
            (False, "field_grad_ref"),
        ):
            with self.subTest(tensor_product=tensor_product):
                with_gradients = mixed_reference_streams(
                    Dependencies(True), tensor_product, n_fields=2, dim=3
                )
                without = mixed_reference_streams(
                    Dependencies(False), tensor_product, n_fields=2, dim=3
                )
                self.assertIn(gradient_name, [s.name for s in with_gradients])
                self.assertNotIn(gradient_name, [s.name for s in without])
                self.assertEqual(len(with_gradients), len(without) + 1)

    def test_the_weights_come_last_and_through_the_reference_data(self):
        """Order is ABI, and the weights are the one buffer not passed by name."""
        for tensor_product in (True, False):
            with self.subTest(tensor_product=tensor_product):
                streams = mixed_reference_streams(
                    Dependencies(True), tensor_product, n_fields=2, dim=3
                )
                self.assertTrue(streams[-1].name.startswith("q_weight"))
                self.assertTrue(streams[-1].from_reference_data)
                self.assertFalse(any(s.from_reference_data for s in streams[:-1]))

    def test_the_simplex_gradient_extent_carries_the_dimension(self):
        """A simplex kernel takes one gradient buffer per field per direction."""
        streams = mixed_reference_streams(
            Dependencies(True), False, n_fields=2, dim=3
        )
        gradients = [s for s in streams if s.name == "field_grad_ref"][0]
        self.assertEqual(gradients.extent, 6)

    def test_the_signature_and_the_call_are_the_same_sequence(self):
        """Proof both emitter sites read this rather than spelling it again."""
        from codegen.framework.emitters import residual_codegen

        self.assertIs(
            residual_codegen.mixed_reference_streams, mixed_reference_streams
        )


if __name__ == "__main__":
    unittest.main()
