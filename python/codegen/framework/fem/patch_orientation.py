"""How an element is presented when a patch kernel visits one of its nodes.

A patch kernel works one node at a time and puts the elements incident on that
node in its SIMD lanes.  The node occupies a different local index in each of
them, so without intervention the basis function paired with it differs from
lane to lane and nothing about it can be hoisted out of the vectorised loop.

Presenting each element with its local numbering permuted so the node under
consideration is always local node ``0`` fixes that: the test basis function is
then the same function in every lane and at every sampled step, so its reference
gradient is a literal the generator can emit once, and its physical gradient is
computed in the element loop and merely read in the step loop.

The permutation is not free.  Relabelling an element's vertices by an odd
permutation reverses its orientation, which flips the sign of ``det J`` and
subtracts that element's contribution instead of adding it.  So the permutation
carrying a chosen vertex to position ``0`` has to be even, and this module's job
is to produce one that is.
"""

def _permutation_sign(permutation):
    """``+1`` for an even permutation, ``-1`` for an odd one.

    Counted as inversions rather than decomposed into cycles: the two agree in
    parity and the inversion count needs no bookkeeping to be obviously right.
    """
    sign = 1
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] > permutation[j]:
                sign = -sign
    return sign


#: The even permutations that carry each vertex of a simplex to position 0.
#:
#: Read ``PERMUTATIONS[type][v][i]`` as: local slot ``i`` of the presented
#: element holds what was local ``PERMUTATIONS[type][v][i]``, so slot 0 holds
#: vertex ``v``.
#:
#: For a triangle these are the three cyclic rotations, which are exactly the
#: even permutations of three things.  For a tetrahedron they are the identity
#: and the three double transpositions -- the Klein four-group -- which is the
#: unique choice that both moves each vertex to the front and stays even.
_SIMPLEX_PERMUTATIONS = {
    "TRI3": (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
    ),
    "TET4": (
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
    ),
}


def supports_patch_orientation(element_type):
    """Whether this element can be presented with a chosen node at local 0.

    True for the affine simplices, whose vertices are interchangeable by an even
    permutation.  A higher-order simplex carries edge and face nodes that the
    vertex permutation has to drag along, and a tensor-product element's
    orientation-preserving relabellings are the cube's rotation group rather
    than a vertex permutation; neither is built, and a caller is told so rather
    than handed a permutation that quietly reverses an element.
    """
    return str(element_type) in _SIMPLEX_PERMUTATIONS


def patch_orientation_permutations(element_type):
    """One even permutation per local node, each bringing that node to slot 0.

    The returned table is indexed by the node's local index in the element as
    the mesh stores it, and every row is a permutation of all the element's
    local slots -- the gather reads the element through it, so the geometry and
    the state are permuted together and stay consistent.
    """
    name = str(element_type)
    try:
        table = _SIMPLEX_PERMUTATIONS[name]
    except KeyError:
        raise ValueError(
            "patch orientation is not defined for element type '%s'; "
            "only the affine simplices have one" % name
        )
    return table
