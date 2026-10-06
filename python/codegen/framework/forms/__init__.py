"""L1: form lowering, where provenance dies.

A layer of its own rather than four modules inside `symbolic/`.  The contract
says a form may name the system it came from and a plan may not, and while these
sat beside the specification that boundary was a convention: nothing structural
stopped `symbolic/` from reaching into them, and `symbolic/__init__.py`
aggregated `weak_forms` into the package surface, so a caller could not tell
which side of the boundary a name came from.

`tests/test_layering.py` can see a package.  It could not see a convention.
"""
