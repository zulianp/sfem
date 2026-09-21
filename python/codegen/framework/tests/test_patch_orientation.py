"""The permutation that fronts a node must not turn the element inside out.

A patch kernel presents each element with the node it is visiting at local slot
0, so the basis function paired with that node is the same in every SIMD lane.
The relabelling has to be orientation preserving: an odd permutation reverses
the element, `det J` changes sign, and the element's contribution is subtracted
from the node's residual instead of added to it -- a wrong merit, silently.

These tests hold the table to that, both combinatorially and geometrically.
"""
import itertools

import pytest

from codegen.framework.fem.patch_orientation import (
    _permutation_sign,
    patch_orientation_permutations,
    supports_patch_orientation,
)

#: A tetrahedron and a triangle with positive volume, in reference order.
REFERENCE_CELLS = {
    "TET4": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "TRI3": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
}


def _determinant(vertices):
    """`det J` of the affine map from the reference cell, by its edge vectors."""
    origin = vertices[0]
    rows = [
        [vertices[i + 1][d] - origin[d] for d in range(len(origin))]
        for i in range(len(origin))
    ]
    if len(rows) == 2:
        return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
    return (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )


@pytest.mark.parametrize("element_type", sorted(REFERENCE_CELLS))
def test_each_node_reaches_slot_zero(element_type):
    table = patch_orientation_permutations(element_type)
    n = len(REFERENCE_CELLS[element_type])
    assert len(table) == n
    for node, permutation in enumerate(table):
        assert sorted(permutation) == list(range(n)), "not a permutation"
        assert permutation[0] == node, "node does not reach slot 0"


@pytest.mark.parametrize("element_type", sorted(REFERENCE_CELLS))
def test_the_permutation_preserves_the_jacobian_sign(element_type):
    """The property the kernel depends on, checked on real coordinates."""
    vertices = REFERENCE_CELLS[element_type]
    reference = _determinant(vertices)
    assert reference > 0

    for permutation in patch_orientation_permutations(element_type):
        permuted = tuple(vertices[i] for i in permutation)
        assert _permutation_sign(permutation) == 1
        assert _determinant(permuted) == pytest.approx(reference)


@pytest.mark.parametrize("element_type", sorted(REFERENCE_CELLS))
def test_an_odd_permutation_would_flip_it(element_type):
    """The negative control.

    Without this the test above would pass for a table that happened to contain
    only the identity, and would say nothing about why evenness is required.
    """
    vertices = REFERENCE_CELLS[element_type]
    reference = _determinant(vertices)

    odd = [
        p
        for p in itertools.permutations(range(len(vertices)))
        if _permutation_sign(p) == -1
    ]
    assert odd, "no odd permutations to test against"
    for permutation in odd:
        permuted = tuple(vertices[i] for i in permutation)
        assert _determinant(permuted) == pytest.approx(-reference)


def test_elements_without_an_orientation_say_so():
    """A tensor-product element's orientation-preserving relabellings are the
    cube's rotations, not a vertex permutation, and a curved simplex has to drag
    its edge nodes along.  Neither is built, and both must refuse rather than
    return something that quietly reverses an element."""
    assert supports_patch_orientation("TET4")
    assert not supports_patch_orientation("HEX8")
    assert not supports_patch_orientation("TET10")
    with pytest.raises(ValueError):
        patch_orientation_permutations("HEX8")
