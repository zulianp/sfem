# SFEM MLIR Lowering

This package contains the experimental MLIR-based matrix-free lowering path.

- `model.py`: SFEM kernel model extraction from the existing generated framework.
- `common.py`: shared inspection and optimization strategy data.
- `tools.py`: wrappers around `mlir-opt`, `mlir-translate`, and `mlir-runner`.
- `openmp.py`: SCF/OpenMP/LLVM/EmitC lowering and OpenMP runner support.
- `opencl.py`: OpenCL lowering, SPIR-V inspection artifacts, and Apple OpenCL C source emission.
- `runtime.py`: Python reference/threaded EBE executor helpers used by tests.
- `sum_factorization.py`: tensor-product SFEM IR extraction plus linalg, generic GPU dialect MLIR, Metal smoke-test source emission, and fused Laplace tensor-product apply emission in GPU dialect and Metal.
- `cpp/`: benchmark C++ harnesses used by the MLIR benchmark script.
- `scripts/`: MLIR benchmark drivers.

Use this package directly through `codegen.framework.mlir`.

Tensor-product Laplace inspection artifacts can be emitted from the existing SFEM form:

```python
from codegen.framework.materials.laplace import material
from codegen.framework.mlir import (
    TensorProductLaplaceFormBatchedGPULowering,
    TensorProductLaplaceFormEBEGPULowering,
    TensorProductLaplaceFormEBEMetalLowering,
    TensorProductLaplaceFormGPULowering,
    TensorProductLaplaceFormLinalgLowering,
    TensorProductSumFactorMLIRLowering,
    tensor_product_laplace_form_ir_from_user_input_stage,
    tensor_product_sum_factor_ir_from_user_input_stage,
)
from sfem import gen

stage = gen.UserInputStage.create(material, ("HEX27",), 8, None)
sum_factor = tensor_product_sum_factor_ir_from_user_input_stage(stage)
TensorProductSumFactorMLIRLowering(sum_factor).write_inspection_artifacts("artifacts/sum_factor")

form = tensor_product_laplace_form_ir_from_user_input_stage(stage)
TensorProductLaplaceFormLinalgLowering(form).write_inspection_artifacts("artifacts/laplace_form_linalg")
TensorProductLaplaceFormGPULowering(form).write_inspection_artifacts("artifacts/laplace_form")
TensorProductLaplaceFormBatchedGPULowering(form, max_elements=1024, max_nodes=4096).write_inspection_artifacts("artifacts/laplace_ebe")
TensorProductLaplaceFormEBEGPULowering(
    form,
    max_elements=1024,
    max_nodes=4096,
    max_node_degree=32,
).write_inspection_artifacts("artifacts/laplace_ebe_full")
TensorProductLaplaceFormEBEMetalLowering(
    form,
    max_elements=1024,
    max_nodes=4096,
    max_node_degree=32,
).write_inspection_artifacts("artifacts/laplace_ebe_metal")
```

The artifact driver emits `.sum_factor.ir.json` and `.laplace_form.ir.json` files for the SFEM IR handoff. The sum-factor writer emits `.linalg.mlir`, whole-chain `.linalg_pipeline.mlir`, raw `.vector.mlir`, padded `.matrix_unit.mlir`, raw-buffer `.matrix_unit_memref.mlir`, whole-chain `.matrix_unit_pipeline.mlir`, `.gpu.mlir`, `.metal`, and `.metal_smoke.mm` files. The sum-factor linalg pipeline invokes the structured tensor stage functions in SFEM IR schedule order and uses static `linalg.generic` bridge copies when adjacent stages use different 2D tensor views. The Laplace form linalg pipeline composes those derivative pipelines with quadrature weights and `kappa` into a full local residual apply. These structured tensor pipelines are the IREE Metal VMFB inputs. The padded matrix-unit artifact rounds each `vector.matrix_multiply` dimension up to the requested `vector_size` tile so high-order contractions such as HEX27 expose hardware-aligned matrix boundaries while preserving raw dimensions as attributes. The raw-buffer matrix-unit artifact adds `vector.transfer_read`, `vector.shape_cast`, and `vector.transfer_write` around the same padded multiply so raw sum-factor contraction buffers can be packed into matrix-unit tiles without changing the SFEM IR shape. The whole-chain matrix-unit pipeline invokes the raw-buffer stage functions in the SFEM IR schedule order with explicit scratch buffers and remains the matrix-unit inspection boundary. The generic GPU dialect artifacts include static `spirv.entry_point_abi` workgroup metadata and `spirv.interface_var_abi` buffer binding metadata for downstream GPU-to-SPIR-V lowering. The fused form writer emits generic GPU dialect `.gpu.mlir` plus the Metal source and ObjC++ smoke-test harness used by the Apple GPU runtime tests.
The batched EBE map writer emits `.ebe.gpu.mlir`, launching one GPU block per element and one thread per local test function while writing element-local residual scratch to avoid atomics.
The full EBE writer emits `.ebe.full.gpu.mlir`, adding an inverse-topology reduce kernel that accumulates element-local scratch into nodal output without atomics.
The Metal EBE writer emits `.ebe.metal` and `.ebe_metal_smoke.mm`, using the same map/reduce split for Apple GPU runtime smoke tests.

The full tensor-product Laplace artifact bundle can also be emitted from the repo venv:

```bash
PYTHONPATH=python venv/bin/python python/codegen/framework/mlir/scripts/tensor_product_laplace_artifacts.py \
  --output-dir /tmp/sfem_laplace_hex27_mlir \
  --element HEX27 \
  --max-elements 1024 \
  --max-nodes 4096 \
  --max-node-degree 32 \
  --verify-reference \
  --verify-performance-shape \
  --validate-mlir \
  --probe-iree-metal \
  --run-iree-metal-runtime \
  --probe-metal-toolchain
```

Add `--run-metal-smoke` on an Apple GPU host to compile and dispatch the generated Metal smoke tests. `--verify-reference` runs deterministic CPU checks from the SFEM IR, including a constant-field residual check and EBE map/reduce checksum. `--verify-performance-shape` checks that generated GPU/Metal kernel artifacts keep the expected linalg/vector structure and avoid hot-loop branches and atomics. Use `--require-metal-device` when the command should fail instead of recording a no-device skip.
Use `--require-iree-metal` when missing `iree-compile` or a failed IREE Metal VMFB build from a sum-factor or full-form `.linalg_pipeline.mlir` artifact should fail the command instead of being recorded in the manifest.
Use `--require-iree-metal-runtime` when the full-form `.linalg_pipeline.mlir` VMFB must dispatch through the IREE Metal runtime and match the SFEM reference residual instead of being recorded as a skip or failure.
Use `--require-metal-toolchain` when missing `xcrun metal` offline compilation support should fail the command instead of being recorded in the manifest.
