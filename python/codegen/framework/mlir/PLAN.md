# Architectural Specification: MLIR Matrix Accelerator and Execution Pipeline

## 1. Matrix Unit Targeting Topography
To harness hardware accelerators like Apple Silicon AMX (Apple Matrix Coprocessor) / ARM SME and NVIDIA Tensor Cores, the compiler bypasses traditional thread-by-thread scalar math. Instead, your `pymlir` engine models computations as **cooperative tile-based matrix blocks**.

The pipeline relies on progressive lowerings to target distinct hardware blocks simultaneously:


          [ Python Frontend (pymlir Model) ]
                           │
                           ▼ (Emit high-level matrix tiles)
            [ linalg.matmul / vector.outerproduct ]
                           │
                           ▼ (Tiling & Vectorization Pass)
              [ vector.matrix_multiply / gpu.mma ]
                           │
          ┌────────────────┴────────────────┐
          ▼ (Target: Apple Silicon)         ▼ (Target: NVIDIA CUDA)
   [ Apple CPU / GPU Compute ]             [ NVIDIA Tensor Cores ]
   • Lowered via IREE Vector pipeline     • Lowered via gpu.mma -> nvvm
   • Triggers ARM SME / AMX Matrix Units  • Triggers Hardware Tensor Cores
   • Emits Outer-Product Accelerations    • Emits mma.sync PTX Assembly


---

## 2. Ingestion Design for `pymlir` (Agnostic Matrix Layout)
When using `pymlir` to generate your initial text representations, you should structure the math utilizing fixed-dimension matrix tiles. For matrix hardware to trigger efficiently, the target arrays must be tile-aligned (typically multiples of `8x8` or `16x16`).

### Blueprint: Pure Python MLIR Generation Strategy
Your `pymlir` script should emit standard MLIR text utilizing **`linalg` structures** or abstract **`vector` blocks**. This decouples your frontend script from hardware-specific intrinsic libraries.

```python
import pymlir

# Example snippet utilizing pyMLIR to structure an accelerated matrix layer
# Generates a platform-blind 16x16 matrix tile operation
mlir_string = """
module @accelerated_graph {
  func.func @matrix_compute(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>, %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // Standard linalg node that downstream passes translate to matrix units
    %0 = linalg.matmul ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>)
                       outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
"""
ctx = pymlir.parse(mlir_string)
```

---

## 3. Host/Device Compilation Lowering Matrix Units
When running `iree-compile` or your custom LLVM optimization pipeline on the output text, the compilation pass transforms the payload through specific matrix dialects:

### A. The Vectorization Phase (`vector.matrix_multiply`)
The abstract matrix operations are tiled down to match the register sizes of the execution hardware. The loops are unrolled into cooperative vector instructions:

```mlir
// Structural representation within the intermediate pipeline
%matrix_a = vector.transfer_read %mem_a[%idx_x, %idx_y] : memref<16x16xf32>, vector<16x16xf32>
%matrix_b = vector.transfer_read %mem_b[%idx_x, %idx_y] : memref<16x16xf32>, vector<16x16xf32>

// Unified MLIR abstraction representing hardware accelerator units
%result = vector.matrix_multiply %matrix_a, %matrix_b 
          {lhs_rows = 16, lhs_columns = 16, rhs_columns = 16} 
          : vector<16x16xf32>, vector<16x16xf32> -> vector<16x16xf32>
```

### B. Apple AMX / ARM SME Translation Path
When targeting **Apple Silicon**, IREE or LLVM takes the abstract `vector.matrix_multiply` block and applies an architectural mapping pass:
1. **Target Identification:** Recognizes M-series chip capabilities (e.g., M-series chips with native AMX or ARM SME support).
2. **Instruction Generation:** Lowers the tile logic directly to hardware **Outer Product instructions** (`fmopa` / `smopa`).
3. **Execution Routing:** Instead of loading vectors cell-by-cell into ordinary CPU/GPU cores, the hardware loads data into the specialized **matrix computing grid registers**, completing the multi-accumulation cycle in a single clock segment.

### C. NVIDIA Tensor Core Translation Path
When targeting **NVIDIA GPUs**, the compilation pass swaps the backend mapping rules:
1. **Dialect Shift:** Lowers the `vector` components into the **`gpu.mma` (Warp Matrix Multiply Accumulate)** dialect.
2. **Warp Synchronization:** Maps the execution blocks directly across a 32-thread hardware warp.
3. **Assembly Output:** Translates the generic warp logic into native **PTX assembly** instructions (`mma.sync.aligned.m16n8k16`), binding the execution to actual Tensor Core silicon units.

---

## 4. Compilation Instructions for Matrix Generation
To ensure that your `pymlir` code effectively switches from scalar processing paths to full matrix-coprocessor pipelines, your compiler pipeline invocation scripts must explicitly enable the specialized target features:

```bash
# 1. Compile for Apple Silicon using Vector extensions targeting AMX / Matrix Co-processors
# (Enables specialized LLVM vectorization and lowering pipelines)
iree-compile frontend_pymlir_output.mlir \
  --iree-hal-target-backends=metal-spirv \
  --iree-llvmcpu-target-cpu=apple-m4 \
  -o build/apple_amx_enabled.vmfb

# 2. Compile for NVIDIA architectures utilizing Tensor Core hardware math blocks
iree-compile frontend_pymlir_output.mlir \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-features=+ptx70,+sm_80 \
  -o build/nvidia_tensorcore_enabled.vmfb
```

---

## 5. Codex Implementation Rules for `pymlir` Generator
When writing the generation logic:
1. **Enforce Rigid Tile Dimensions:** Ensure your code generator forces matrix math operations into explicit sizes (like `16x16` or `8x8` tiles). Vector hardware matrix units cannot compile arbitrary or unaligned shapes natively.
2. **Leverage Structural Primitives:** Stick entirely to generating **`linalg.matmul`** or **`linalg.generic`** blocks inside your text printer. Do not try to write custom assembly macros in Python; let the target-tuned vector compilers handle register scheduling automatically.