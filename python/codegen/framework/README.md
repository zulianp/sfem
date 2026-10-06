# SFEM Codegen Framework Layout

This package is organized by lowering layer. Implementation code lives inside one of the layer
packages; `__init__.py` is the only file at the package root and is a thin façade that re-exports
moved symbols for existing callers.

Read `PRESCRIBED_ARCHITECTURE.md` first: it states what each layer owns and the rules that keep
the boundaries real. `ARCHITECTURE.html` records how far the code has got against them.

## The lowering stack

Imports point up this list and never down, enforced by `tests/test_layering.py`.

- `symbolic/`: symbolic fields, forms, equations, residual systems, and constitutive helpers.
  Names no other layer.
- `fem/`: finite-element reference data, basis plans, element families, and tensor-product helpers.
- `plans/`: what a kernel is — phases, data streams, blocks, geometry and variants, layouts,
  kernel signatures, matrix formats, evaluation strategy, cost. SymPy ends here.
- `ir/`: the SFEM kernel AST and optional adapters to external IR/codegen tools.
- `targets/`: OpenMP, CUDA/HIP, AVX512 and SVE/SME — pragmas, qualifiers, atomics, widths.
- `emitters/`: target-language source emitters and code printers. Prints plans; decides nothing.
- `backends/`: backend orchestration, which drives emission for one target.

## Beside the stack

- `pipeline/`: the driver. Calls each layer in sequence and is imported by none of them.
- `package/`: the C ABI, the generated `sfem::Op` wrapper and factory integration. Runs after
  emission and never changes a kernel body.
- `materials/`: material model definitions consumed by the framework.
- `generators/`: executable generation scripts. `generators/regenerate_all.sh` runs the standard
  generation set.
- `tools/`: gates and measurement — snapshot, reproducibility digests, apply benchmark.
- `tests/`: the suite, including the architectural ratchets.
- `mlir/`: MLIR/OpenMP/OpenCL lowering experiments. Self-contained and out of scope for the
  layering work; nothing outside it imports it.

## Documentation

- `PRESCRIBED_ARCHITECTURE.md` — what the layers must own, and the rules.
- `ARCHITECTURE.html` — what is actually built, with the open points.
- `INEXACT.md` — the projected (partial-assembly) apply.
- `docs/` — worked examples: matrix formats, coupled residual, Mooney-Rivlin.
- `retired/` — superseded design documents, kept for their reasoning. Not maintained, and not
  cited from code.
