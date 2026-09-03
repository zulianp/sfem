"""Which per-quadrature-point geometry values a local kernel reads.

Three quantities can appear inside a kernel's lane loop: the offset into the
geometry streams for this point and lane, the Jacobian determinant, and the
Jacobian adjugate's ``dim * dim`` components.  Which of them a kernel needs
follows from the lowered form -- a form with no test-value coefficients and no
reference gradients reads neither -- and that is a planning decision in exactly
the way the field roles are.

It was being derived inside ``emitters/residual_codegen.py``, twice verbatim::

    uses_determinant = any(dependencies.value_coefficients) or dependencies.uses_adjugate
    uses_geometry_offset = uses_determinant or dependencies.uses_adjugate

and then asked again as ``if dependencies.uses_adjugate:`` at some twenty
further sites that declare the values, name them in a signature, or pass them
to a kernel.

Note what the second line above says.  ``uses_determinant`` already contains
``uses_adjugate``, so ``uses_determinant or uses_adjugate`` is
``uses_determinant``: the offset and the determinant are needed under exactly
the same condition, and the disjunction was a restatement rather than a
distinction.  Writing the derivation once is what makes that visible; it is
preserved as a single predicate here rather than as two that must agree.
"""

from dataclasses import dataclass


#: The order the quantities are declared in, which is the order the generated
#: lane bodies have always used: the offset first because the other two index
#: through it, then the determinant, then the adjugate components.
GEOMETRY_QUANTITY_ORDER = ("geometry_offset", "determinant", "adjugate")


@dataclass(frozen=True)
class GeometryQuantity:
    """One geometry value a kernel reads, and how many components it has."""

    name: str
    components: int = 1

    @property
    def is_indexed(self):
        """Whether the value is a component array rather than a scalar."""
        return self.components > 1


def uses_geometry_values(dependencies):
    """Whether the kernel reads any per-point geometry at all.

    True when the form contracts a test value -- which needs the determinant to
    weight it -- or needs the adjugate to map a reference gradient.  This is the
    single predicate that ``uses_determinant`` and ``uses_geometry_offset`` were
    both computing.
    """
    return bool(
        any(dependencies.value_coefficients) or dependencies.uses_adjugate
    )


def local_geometry_quantities(dependencies, dim):
    """The geometry values this kernel's lane body reads, in declaration order.

    ``dim`` fixes the adjugate's component count.  A kernel that reads no
    geometry gets an empty sequence, and emits no offset, no determinant and no
    adjugate -- which is the same thing the conditionals said, in one place.
    """
    if not uses_geometry_values(dependencies):
        return ()
    quantities = [
        GeometryQuantity("geometry_offset"),
        GeometryQuantity("determinant"),
    ]
    if dependencies.uses_adjugate:
        quantities.append(GeometryQuantity("adjugate", components=dim * dim))
    return tuple(quantities)


def geometry_quantity(dependencies, dim, name):
    """One quantity by name, or ``None`` if this kernel does not read it."""
    for quantity in local_geometry_quantities(dependencies, dim):
        if quantity.name == name:
            return quantity
    return None
