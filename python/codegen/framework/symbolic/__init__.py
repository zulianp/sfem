import codegen.framework.symbolic.core as _impl

# The package surface is the union of its modules, not just core.py.  Splitting
# core.py is ongoing; aggregating here means a relocation inside the layer does
# not change what `codegen.framework.symbolic` exposes.
# `weak_forms` used to be aggregated here too, and that was the whole of what
# made the L0/L1 boundary a convention: a caller writing
# `from codegen.framework.symbolic import sfem_soa_weak_form` could not tell the
# name lived a layer up.  It is L1 now, and reaching it from here would be this
# layer importing the one above it.
for _module in (_impl,):
    globals().update({
        _name: _value
        for _name, _value in vars(_module).items()
        if not _name.startswith("__")
    })
