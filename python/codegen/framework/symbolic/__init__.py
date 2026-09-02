import codegen.framework.symbolic.core as _impl
import codegen.framework.symbolic.weak_forms as _weak_forms

# The package surface is the union of its modules, not just core.py.  Splitting
# core.py is ongoing; aggregating here means a relocation inside the layer does
# not change what `codegen.framework.symbolic` exposes.
for _module in (_impl, _weak_forms):
    globals().update({
        _name: _value
        for _name, _value in vars(_module).items()
        if not _name.startswith("__")
    })
