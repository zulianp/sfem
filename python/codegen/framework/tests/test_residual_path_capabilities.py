"""What each residual emitter path can generate, and how little they share.

PIPELINE.md says that after form lowering there is one shared code-generation
process.  For the residual path there are two, and they are nearly disjoint:
``generate_coupled_residual_sfem_files`` reaches 6,716 lines of code that
``generate_mixed_residual_sfem_files`` does not, the mixed path reaches 2,155
the coupled one does not, and the two share 419 -- 4.5% of the total.

The consequence is a capability gap rather than a stylistic one.  Several
kernels exist only on the coupled path, so a mixed-order (Taylor-Hood)
formulation cannot have them at all:

    AoS dispatch    coupled only
    packed apply    coupled only    <- the matrix-free apply that matters most
    CRS assembly    coupled only
    DIA assembly    coupled only
    COO assembly    both
    BSR assembly    neither

That is why poro-hyperelasticity and Stokes emit no AoS and no packed apply
variants.  It reads like a rule in the variant matrix, and it is not one: it is
the absence of an implementation.

This module pins the matrix so the gap is tracked rather than rediscovered.
Closing a cell -- teaching the mixed path packed apply, say -- is a real
capability gain and should require deliberately editing this table.
"""

import ast
import os
import unittest


EMITTER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "emitters",
    "residual_codegen.py",
)

COUPLED_ENTRY = "generate_coupled_residual_sfem_files"
MIXED_ENTRY = "generate_mixed_residual_sfem_files"

#: capability -> substring identifying the functions that implement it.
CAPABILITIES = {
    "aos_dispatch": "aos_dispatch",
    "packed_apply": "packed_jacobian_action",
    "packed_affine_apply": "packed_affine_jacobian",
    "crs_assembly": "crs_matrix_assembly",
}

#: The matrix as it stands.  Each False is a kernel a Taylor-Hood formulation
#: cannot get.  Flipping one to True is a capability gain, not a refactor.
EXPECTED = {
    "aos_dispatch": (True, False),
    "packed_apply": (True, False),
    "packed_affine_apply": (True, False),
    "crs_assembly": (True, False),
}


def _functions():
    with open(EMITTER, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=EMITTER)
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _reachable(functions, root):
    """Every module-level function reachable from ``root`` by direct call."""
    seen, frontier = set(), {root}
    while frontier:
        nxt = set()
        for name in frontier:
            node = functions.get(name)
            if node is None:
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in functions
                    and call.func.id not in seen
                ):
                    nxt.add(call.func.id)
        seen |= frontier
        frontier = nxt - seen
    return seen


def capability_matrix():
    functions = _functions()
    coupled = _reachable(functions, COUPLED_ENTRY)
    mixed = _reachable(functions, MIXED_ENTRY)
    return {
        capability: (
            any(needle in name for name in coupled),
            any(needle in name for name in mixed),
        )
        for capability, needle in CAPABILITIES.items()
    }


class ResidualPathCapabilitiesTest(unittest.TestCase):
    maxDiff = None

    def test_capability_matrix_is_as_recorded(self):
        self.assertEqual(
            capability_matrix(),
            EXPECTED,
            "the coupled/mixed capability matrix changed; if a cell was closed "
            "that is a capability gain worth stating explicitly, and if one was "
            "lost it is a regression",
        )

    def test_the_two_paths_remain_nearly_disjoint(self):
        """A rising number here means the paths are converging, which is the goal.

        Read it as an upper bound rather than a measurement.  Reachability is
        static, so a function the shared code calls on only one branch --
        ``_local_function``, which ``_local_header`` reaches only when the
        fields share a shape count -- counts as shared even though the
        mixed-order path never executes it.  The capability matrix above is the
        honest statement of what each path can actually produce.
        """
        functions = _functions()
        coupled = _reachable(functions, COUPLED_ENTRY)
        mixed = _reachable(functions, MIXED_ENTRY)
        shared = coupled & mixed
        self.assertGreaterEqual(
            len(shared),
            48,
            "the residual paths share fewer functions than before; unifying "
            "them should only ever increase this",
        )

    def test_mixed_order_formulations_cannot_get_packed_apply(self):
        """The headline consequence, stated so it cannot be forgotten."""
        matrix = capability_matrix()
        coupled_has, mixed_has = matrix["packed_apply"]
        self.assertTrue(coupled_has)
        self.assertFalse(
            mixed_has,
            "the mixed path gained packed apply -- update EXPECTED and the "
            "apply-variant rule, which currently predicts no packed variants "
            "for mixed-order formulations",
        )


if __name__ == "__main__":
    unittest.main()
