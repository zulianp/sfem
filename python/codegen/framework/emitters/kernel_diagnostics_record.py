"""One kernel's `KernelDiagnostics` record, printed in one place.

`kernel_diagnostics.hpp` declares the struct and the helpers that read it --
`KernelDiagnostics_arithmetic_intensity`, `_total_flops`, `_total_bytes` -- so
what a kernel publishes is one static initializer and one accessor.  The field
order is the struct's, and it is long: forty entries, ten of them machine cost
constants that are the same for every kernel in the tree.

It lives here rather than in `energy_codegen.py`, where it was written, because
the inexact-apply family publishes the same record and a second initializer
written against the same struct is a second place for the field order to drift.
The struct is what makes that drift silent: every field is an integer or a
double, so two fields exchanged still compiles and still runs, and reports a
number that is wrong in a way no test on the generator would see.

The caller supplies the measurements; nothing is decided here.
"""

import dataclasses


#: The struct these records initialize, from `kernel_diagnostics.hpp`.
STRUCT_NAME = "KernelDiagnostics"

#: Cycles per instruction for each operation class, in the struct's order.
#: One machine model for the whole tree: these describe the hardware the
#: arithmetic intensity is reported against, not the kernel, so a kernel does
#: not get to disagree with another about them.
_COST_MODEL = (1.0, 1.0, 8.0, 12.0, 16.0, 20.0, 20.0, 24.0, 1.0, 1.0)


@dataclasses.dataclass(frozen=True)
class DiagnosticsRecord:
    """Everything the struct holds that varies from kernel to kernel."""

    public_name: str
    element_type: str
    dim: int
    n_qp: int
    n_shape: int
    vector_size: int
    quadrature_order: int
    #: An `ExpressionCost`, whose per-operation counts the record carries.
    cost: object
    affine_mesh_flops_per_element: int
    isoparametric_mesh_flops_per_element: int
    geometry_streams: int
    reference_scalars: int
    quadrature_weight_scalars: int
    material_scalars: int
    u_streams: int
    h_streams: int
    output_streams: int
    output_reads_per_element: int
    output_writes_per_element: int


def diagnostics_record_lines(record):
    """The static initializer and the accessor, as a kernel publishes them."""
    variable_name = "%s_diagnostics_data" % record.public_name
    cost = record.cost
    values = [
        '"%s"' % record.public_name,
        '"%s"' % record.element_type,
        record.dim,
        record.n_qp,
        record.n_shape,
        record.vector_size,
        record.quadrature_order,
        cost.adds,
        cost.muls,
        cost.divs,
        cost.sqrts,
        cost.pows,
        cost.exps,
        cost.logs,
        cost.trigs,
        cost.loads,
        cost.stores,
        cost.flops,
        record.affine_mesh_flops_per_element,
        record.isoparametric_mesh_flops_per_element,
        cost.temporaries,
        cost.estimated_registers,
        record.geometry_streams,
        record.reference_scalars,
        record.quadrature_weight_scalars,
        record.material_scalars,
        record.u_streams,
        record.h_streams,
        record.output_streams,
        record.output_reads_per_element,
        record.output_writes_per_element,
    ]
    lines = ["static const %s %s = {" % (STRUCT_NAME, variable_name)]
    lines.extend("  %s," % value for value in values)
    lines.extend("  %s," % value for value in _COST_MODEL[:-1])
    lines.append("  %s" % _COST_MODEL[-1])
    lines.append("};")
    return lines


def diagnostics_accessor_lines(public_name):
    """The `extern "C"` accessor the diagnostics dispatch collects by name.

    The record is the only thing a caller cannot compute for itself: the
    intensity and print-rate helpers already take it and live in
    `kernel_diagnostics.hpp`, so a per-kernel wrapper around each published a
    name for a call the caller can spell.  There were 1136 print-rate wrappers
    and 264 intensity wrappers in the tree and nothing outside the generator
    referenced any of them.
    """
    return [
        'extern "C" const sfem::codegen::%s *%s_diagnostics(void) {'
        % (STRUCT_NAME, public_name),
        "  return &sfem::codegen::%s_diagnostics_data;" % public_name,
        "}",
    ]
