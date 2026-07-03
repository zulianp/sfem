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

The artifact driver emits `.sum_factor.ir.json` and `.laplace_form.ir.json` files for the SFEM IR handoff. The sum-factor writer emits `.linalg.mlir`, whole-chain `.linalg_pipeline.mlir`, raw `.vector.mlir`, padded `.matrix_unit.mlir`, raw-buffer `.matrix_unit_memref.mlir`, whole-chain `.matrix_unit_pipeline.mlir`, `.gpu.mlir`, direct `.spirv.opencl.mlir`, `.spirv.opencl.op.mlir`, `.spirv.opencl.spv`, `.spirv.opencl.dispatch.json`, `.metal`, and `.metal_smoke.mm` files for tensor-product elements including QUAD4, HEX8/HEX27, and higher-order PROTEUS_HEX cases. The sum-factor linalg pipeline invokes the structured tensor stage functions in SFEM IR schedule order and uses static `linalg.generic` bridge copies when adjacent stages use different 2D tensor views. The Laplace form linalg pipeline composes those derivative pipelines with quadrature weights and `kappa` into a full local residual apply. These structured tensor pipelines are the IREE Metal VMFB inputs. The padded matrix-unit artifact rounds each `vector.matrix_multiply` dimension up to the requested `vector_size` tile so high-order contractions such as HEX27 and PROTEUS_HEX64 expose hardware-aligned matrix boundaries while preserving raw dimensions as attributes. The raw-buffer matrix-unit artifact adds `vector.transfer_read`, `vector.shape_cast`, and `vector.transfer_write` around the same padded multiply so raw sum-factor contraction buffers can be packed into matrix-unit tiles without changing the SFEM IR shape. The whole-chain matrix-unit pipeline invokes the raw-buffer stage functions in the SFEM IR schedule order with explicit scratch buffers and remains the matrix-unit inspection boundary. The generic GPU dialect artifacts include static `spirv.entry_point_abi` workgroup metadata and `spirv.interface_var_abi` buffer binding metadata for downstream GPU-to-SPIR-V lowering. The direct SPIR-V/OpenCL artifact emits one branch-free kernel per sum-factor stage, uses CrossWorkgroup buffers, unrolls each tensor-product contraction in SPIR-V dialect, records per-stage dispatch sizes, and serializes the extracted `spirv.module` to a SPIR-V binary for generic GPU inspection. With `--validate-mlir`, the driver deserializes each generated `.spv` back to SPIR-V dialect and records the result in `spirv_binary_validation`. The sum-factor Metal smoke harness dispatches every generated tensor-product stage kernel on Apple GPU. The fused form writer emits generic GPU dialect `.gpu.mlir` plus the Metal source and ObjC++ smoke-test harness used by the Apple GPU runtime tests.
The driver also writes `pipeline_evidence.json`, a compact summary of the SFEM IR, generated GPU artifacts, IREE Metal VMFB/executable/backend-file probes, SPIR-V validation, Metal toolchain compilation, and runtime smoke status. This is intended as the first place to check the end-to-end tensor-product Laplace pipeline state for a generated artifact directory.
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

Add `--run-metal-smoke` on an Apple GPU host to compile and dispatch the generated Metal smoke tests. `--verify-reference` runs deterministic CPU checks from the SFEM IR, including a constant-field residual check and EBE map/reduce checksum. `--verify-performance-shape` checks that generated GPU/Metal kernel artifacts keep the expected linalg/vector structure, that the full-form linalg pipeline covers every derivative direction, that the sum-factor generic GPU artifact has one launch/kernel pair per stage with SPIR-V ABI metadata, that the local form and EBE generic GPU artifacts have the expected dispatch and descriptor ABI shape, and that hot-loop branches and atomics are avoided. Use `--require-metal-device` when the command should fail instead of recording a no-device skip.
Use `--require-iree-metal` when missing `iree-compile` or a failed IREE Metal VMFB build from a sum-factor or full-form `.linalg_pipeline.mlir` artifact should fail the command instead of being recorded in the manifest. The same probe also records `iree_metal_executable_sources` by compiling those linalg pipelines with `--compile-to=executable-sources`; these artifacts expose the successful IREE HAL boundary with `hal.executable`, `stream.cmd.dispatch`, and `flow.dispatch.tensor` operations before VMFB emission. It also records `iree_metal_executable_files` from `--iree-hal-dump-executable-files-to`, including configured executable MLIR, Metal target MLIR, nonempty SPIR-V binaries that deserialize back to SPIR-V dialect, and generated MSL source files with kernel/device/thread-position entry points.
Use `--probe-iree-metal-matrix-unit` to also record whether the sum-factor `.matrix_unit.mlir`, `.matrix_unit_memref.mlir`, and `.matrix_unit_pipeline.mlir` inspection boundaries currently compile to IREE Metal VMFBs. This probe is separate because the matrix-unit artifacts intentionally expose raw `vector.transfer_read`/`vector.matrix_multiply` structure that may require additional IREE lowering support before they can be required runtime gates; add `--require-iree-metal-matrix-unit` when those boundaries must pass. Failed IREE probe stdout/stderr streams are written next to the attempted VMFB input and the manifest keeps compact previews plus byte counts. Known IREE VM conversion failures around matrix-unit vector/memref ABI materialization are classified as `iree_vm_matrix_unit_abi_conversion`.
Use `--probe-iree-metal-gpu` to record whether the generated generic GPU dialect artifacts currently compile directly to IREE Metal VMFBs. This probe covers the sum-factor stage kernels, local form apply, EBE map, and EBE map/reduce GPU artifacts. Each record keeps the baseline command plus VM/SPIR-V index-width and input-demotion compile attempts so IREE compatibility failures are reproducible from the manifest. Known IREE VM conversion failures around `gpu.launch_func` and SCF/index materialization are classified as `iree_vm_generic_gpu_index_conversion`; add `--require-iree-metal-gpu` when those boundaries must pass.
Use `--require-iree-metal-runtime` when the sum-factor and full-form `.linalg_pipeline.mlir` VMFBs must dispatch through the IREE Metal runtime and match the SFEM reference values instead of being recorded as a skip or failure.
Use `--require-metal-toolchain` when missing `xcrun metal` offline compilation support should fail the command instead of being recorded in the manifest. The toolchain probe tags each `.metal` input as either SFEM-generated Metal or IREE-emitted MSL from `iree_metal_executable_files`, and writes `.air` outputs under matching relative paths so the manifest preserves which backend source path was compiled.
