# SFEM MLIR Lowering

This package contains the experimental MLIR-based matrix-free lowering path.

- `model.py`: SFEM kernel model extraction from the existing generated framework.
- `common.py`: shared inspection and optimization strategy data.
- `tools.py`: wrappers around `mlir-opt`, `mlir-translate`, and `mlir-runner`.
- `openmp.py`: SCF/OpenMP/LLVM/EmitC lowering and OpenMP runner support.
- `opencl.py`: OpenCL lowering, SPIR-V inspection artifacts, and Apple OpenCL C source emission.
- `runtime.py`: Python reference/threaded EBE executor helpers used by tests.
- `cpp/`: benchmark C++ harnesses used by the MLIR benchmark script.
- `scripts/`: MLIR benchmark drivers.

The legacy `codegen.framework.mlir_ebe` module is a compatibility shim that
re-exports this package.
