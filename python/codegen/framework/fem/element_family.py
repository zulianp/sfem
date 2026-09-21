"""Which family an element belongs to, for the rules that are keyed on it.

Three defaults govern how a form is evaluated, and each is a property of the
element rather than of the material or of how the material was written:

    tensor-product          sum factorization, always
    simplex, lowest order   the CSE expanded for a compact physical gradient or
                            for the basis functions directly, with no
                            quadrature-point data generated at all
    simplex, higher order   rules of its own

``fem.basis.BasisFamily`` already splits elements two ways, simplex against
tensor-product, which is the split the basis data needs.  The evaluation rules
need three, because a lowest-order simplex and a higher-order one are evaluated
differently while sharing a basis family.  This is that finer taxonomy, in the
layer that already owns element facts.

It was previously written down twice: as three literal tuples in
``tools/evaluation_strategy.py``, and implicitly in the conditions
``plans/form_transformations._is_constant_p1_simplex_rule`` tests.  A
measurement and a generator that disagree about what TET4 is would report
conformance the generator does not have.
"""

from enum import Enum

from codegen.framework.fem.reference import (
    sfem_is_proteus_hex_element,
    sfem_is_proteus_quad_element,
    sfem_is_tensor_product_hex_element,
)


class ElementFamily(Enum):
    """The families the evaluation rules are stated over."""

    TENSOR_PRODUCT = "tensor-product"
    SIMPLEX_LOWEST = "simplex-lowest"
    SIMPLEX_HIGHER = "simplex-higher"
    MIXED = "mixed"


#: Lowest-order simplices: the basis gradients are constant over the element, so
#: there is nothing for a quadrature loop to vary over.
SIMPLEX_LOWEST_ELEMENTS = ("TRI3", "TET4")

#: Higher-order simplices.  Listed rather than derived so that adding one is a
#: deliberate act, since its rules are not the same as either neighbour's.
SIMPLEX_HIGHER_ELEMENTS = ("TRI6", "TET10")

#: Tensor-product elements that are not caught by the PROTEUS predicates.
TENSOR_PRODUCT_ELEMENTS = ("QUAD4", "HEX8", "HEX27", "HEX64", "HEX125")


def element_family(element_type):
    """The family of an element, by name.

    A mixed-order pair such as ``HEX27_HEX8`` is reported as ``MIXED``: its
    fields live on different spaces and the rules do not yet say what it gets.
    """
    name = str(element_type).upper()
    if "_" in name and not name.startswith("PROTEUS_"):
        return ElementFamily.MIXED
    if (
        name in TENSOR_PRODUCT_ELEMENTS
        or sfem_is_tensor_product_hex_element(name)
        or sfem_is_proteus_hex_element(name)
        or sfem_is_proteus_quad_element(name)
    ):
        return ElementFamily.TENSOR_PRODUCT
    if name in SIMPLEX_LOWEST_ELEMENTS:
        return ElementFamily.SIMPLEX_LOWEST
    if name in SIMPLEX_HIGHER_ELEMENTS:
        return ElementFamily.SIMPLEX_HIGHER
    return ElementFamily.MIXED


def is_lowest_order_simplex(element_type):
    """Whether this element evaluates without any quadrature-point data."""
    return element_family(element_type) is ElementFamily.SIMPLEX_LOWEST


def element_family_for_rule(rule):
    """The family of the element a quadrature rule was built for."""
    return element_family(getattr(rule, "element_type", ""))
