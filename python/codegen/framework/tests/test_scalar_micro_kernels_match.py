"""The scalar micro-kernels compute what the lane-blocked ones compute.

`tensor_product_kernels` renders one body twice: `TensorProductWeakOps` and
`TensorProductResidualOps` for a work item that strides over a block of
elements, and `...Scalar` twins for one that holds a single element.  Rendering
twice is what keeps the second from drifting into a transcription of the first,
but "rendered from the same source" is a claim about the generator, not about
the numbers.

This is the claim about the numbers, and it is the gate that makes it safe to
point a device kernel at the scalar family: with one element in hand the two
must agree exactly, not nearly.  Bitwise, because they are the same arithmetic
in the same order -- anything else would mean the renderings had diverged in a
way a tolerance would hide.

Four operations, because the two families divide the work differently and a
check of one proves nothing about the other: `gradient` and `test` from the
weak-form family, `evaluate` and `integrate` from the residual one.  Three
element shapes, because the sum factorization is written per dimension and the
2D and 3D bodies are separate code.

The driver is `data/scalar_micro_kernel_equivalence.cpp` rather than a string
here: it is C++ that wants to stay readable, and a test that prints its own
subject is easier to run by hand when it fails.
"""

import os
import shutil
import subprocess
import tempfile
import unittest


def _generated_tree():
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "frontend", "ops", "generated",
        )
    )


def _driver():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "scalar_micro_kernel_equivalence.cpp",
    )


class ScalarMicroKernelsMatch(unittest.TestCase):
    def setUp(self):
        if shutil.which("c++") is None:
            self.skipTest("no c++ compiler")

    def test_the_scalar_renderings_agree_with_the_lane_blocked_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "equivalence")
            build = subprocess.run(
                [
                    "c++", "-std=c++17",
                    "-I", _generated_tree(),
                    _driver(),
                    "-o", binary,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(
                run.returncode,
                0,
                "a scalar micro-kernel disagrees with its lane-blocked twin:\n%s"
                % run.stdout,
            )
            # The driver reports one line per element shape; a silent pass
            # would mean it had stopped checking anything.
            self.assertGreaterEqual(len(run.stdout.strip().splitlines()), 3, run.stdout)
            self.assertNotIn("DIFFER", run.stdout)


if __name__ == "__main__":
    unittest.main()
