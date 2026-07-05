# SFEM Codegen Framework Layout

This package is organized by lowering layer. New implementation code should live
inside one of these layer packages; the legacy files in this directory are
compatibility shims for existing imports.

- `symbolic/`: symbolic fields, forms, equations, residual systems, and
  constitutive helpers.
- `fem/`: finite-element reference data, basis plans, geometry plans, and
  tensor-product helpers.
- `plans/`: semantic emission plans, kernel signatures, diagnostics, reference
  data plans, and mesh/local phase planning.
- `ir/`: SFEM kernel AST and optional adapters to external IR/codegen tools.
- `emitters/`: target-language source emitters and code printers.
- `backends/`: backend boundaries such as OpenMP, CUDA, and optional OpenCL
  experiments.
- `generators/`: executable generation scripts. Use
  `generators/regenerate_all.sh` to run the standard generation set.
- `mlir/`: MLIR/OpenMP/OpenCL lowering experiments, runtime helpers, scripts,
  and C++ benchmark drivers.
- `materials/`: material model definitions consumed by the framework.

Compatibility shims at the package root should stay thin: they re-export moved
symbols for old callers, but implementation modules should import from the
layer packages directly.
