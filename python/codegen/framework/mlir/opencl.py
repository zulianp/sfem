from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess

import sympy as sp

from .common import CodeInspectionArtifacts, MLIR_OPTIMIZATION_PLANS, mlir_optimization_strategy
from .model import linear_elasticity_mlir_model
from .tools import (
    _extract_single_top_level_operation,
    _find_mlir_opt,
    _serialize_spirv_module,
    llvm_mlir_availability,
)


class SPIRVAddressingModel(Enum):
    LOGICAL = "Logical"


class SPIRVMemoryModel(Enum):
    OPENCL = "OpenCL"


class SPIRVExecutionModel(Enum):
    KERNEL = "Kernel"


class SPIRVFunctionControl(Enum):
    NONE = "None"


class SPIRVStorageClass(Enum):
    CROSS_WORKGROUP = "CrossWorkgroup"
    INPUT = "Input"


class SPIRVBuiltIn(Enum):
    GLOBAL_INVOCATION_ID = "GlobalInvocationId"


class KernelScalar(Enum):
    F32 = "f32"
    I32 = "i32"


class OpenCLBufferSymbol(Enum):
    CONNECTIVITY = "connectivity"
    DIRECTION = "direction"
    SCRATCH = "scratch"
    NODE_DEGREE = "node_degree"
    NODE_TO_ELEMENT_MAP = "node_to_element_map"
    NODE_TO_LOCAL_IDX = "node_to_local_idx"
    OUTPUT = "output"


class OpenCLBuiltInSymbol(Enum):
    GLOBAL_INVOCATION_ID = "gid"


@dataclass(frozen=True)
class OpenCLDeviceBuffer:
    symbol: OpenCLBufferSymbol
    scalar: KernelScalar
    extent: int

    @property
    def mlir_symbol(self):
        return self.symbol.value


@dataclass(frozen=True)
class OpenCLWorkItemBuiltIn:
    symbol: OpenCLBuiltInSymbol
    built_in: SPIRVBuiltIn

    @property
    def mlir_symbol(self):
        return self.symbol.value


@dataclass(frozen=True)
class OpenCLCopyKernel:
    symbol: str
    source: OpenCLDeviceBuffer
    destination: OpenCLDeviceBuffer


class MatrixFreeOpenCLMLIRLowering:
    """OpenCL device lowering from the SFEM-generated kernel model.

    This target emits SPIR-V dialect directly for an OpenCL Kernel execution
    environment.  It is deliberately separate from the CPU/threaded lowering:
    OpenCL global buffers are CrossWorkgroup storage, work item ids come from
    GlobalInvocationId, and the schedule is split into map and reduce kernels so
    global atomics are not required.
    """

    def __init__(
        self,
        model=None,
        *,
        max_elements=1024,
        max_nodes=4096,
        max_node_degree=32,
        optimization_strategy=None,
    ):
        self.model = linear_elasticity_mlir_model() if model is None else model
        self.max_elements = int(max_elements)
        self.max_nodes = int(max_nodes)
        self.max_node_degree = int(max_node_degree)
        self.optimization_strategy = mlir_optimization_strategy(optimization_strategy)
        if self.max_elements <= 0 or self.max_nodes <= 0 or self.max_node_degree <= 0:
            raise ValueError("OpenCL MLIR bounds must be positive")

    @property
    def optimization_plan(self):
        return MLIR_OPTIMIZATION_PLANS[self.optimization_strategy]

    @property
    def map_kernel_name(self):
        return f"{self.model.mesh_kernel_name}_opencl_map"

    @property
    def reduce_kernel_name(self):
        return f"{self.model.mesh_kernel_name}_opencl_reduce"

    @property
    def scratch_size(self):
        return self.max_elements * self.model.scratch_components

    @property
    def node_field_size(self):
        return self.max_nodes * self.model.n_field_components

    @property
    def connectivity_size(self):
        return self.max_elements * self.model.n_shape

    @property
    def node_degree_size(self):
        return self.max_nodes

    @property
    def node_element_map_size(self):
        return self.max_nodes * self.max_node_degree

    @property
    def connectivity_buffer(self):
        return OpenCLDeviceBuffer(OpenCLBufferSymbol.CONNECTIVITY, KernelScalar.I32, self.connectivity_size)

    @property
    def direction_buffer(self):
        return OpenCLDeviceBuffer(OpenCLBufferSymbol.DIRECTION, KernelScalar.F32, self.node_field_size)

    @property
    def scratch_buffer(self):
        return OpenCLDeviceBuffer(OpenCLBufferSymbol.SCRATCH, KernelScalar.F32, self.scratch_size)

    @property
    def node_degree_buffer(self):
        return OpenCLDeviceBuffer(OpenCLBufferSymbol.NODE_DEGREE, KernelScalar.I32, self.node_degree_size)

    @property
    def node_to_element_buffer(self):
        return OpenCLDeviceBuffer(OpenCLBufferSymbol.NODE_TO_ELEMENT_MAP, KernelScalar.I32, self.node_element_map_size)

    @property
    def node_to_local_idx_buffer(self):
        return OpenCLDeviceBuffer(OpenCLBufferSymbol.NODE_TO_LOCAL_IDX, KernelScalar.I32, self.node_element_map_size)

    @property
    def output_buffer(self):
        return OpenCLDeviceBuffer(OpenCLBufferSymbol.OUTPUT, KernelScalar.F32, self.node_field_size)

    @property
    def device_buffers(self):
        return (
            self.connectivity_buffer,
            self.direction_buffer,
            self.scratch_buffer,
            self.node_degree_buffer,
            self.node_to_element_buffer,
            self.node_to_local_idx_buffer,
            self.output_buffer,
        )

    @property
    def work_item_builtins(self):
        return (OpenCLWorkItemBuiltIn(OpenCLBuiltInSymbol.GLOBAL_INVOCATION_ID, SPIRVBuiltIn.GLOBAL_INVOCATION_ID),)

    @property
    def map_kernel(self):
        return OpenCLCopyKernel(
            self.map_kernel_name,
            self.direction_buffer,
            self.scratch_buffer,
        )

    @property
    def reduce_kernel(self):
        return OpenCLCopyKernel(
            self.reduce_kernel_name,
            self.scratch_buffer,
            self.output_buffer,
        )

    @property
    def copy_kernels(self):
        return (self.map_kernel, self.reduce_kernel)

    def pymlir_availability(self):
        return llvm_mlir_availability()

    def render_spirv_opencl_module(self):
        return str(self.build_spirv_opencl_module())

    def optimize_spirv_opencl_module(self, mlir_opt=None, optimization_strategy=None):
        mlir_opt = mlir_opt or _find_mlir_opt()
        strategy = mlir_optimization_strategy(
            self.optimization_strategy if optimization_strategy is None else optimization_strategy
        )
        result = subprocess.run(
            [
                mlir_opt,
                "-",
                *MLIR_OPTIMIZATION_PLANS[strategy].pre_lowering_passes,
                "--verify-diagnostics",
            ],
            input=self.render_spirv_opencl_module(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    def render_spirv_module_op(self):
        module = self.build_spirv_opencl_module()
        operations = list(module.body.operations)
        if len(operations) != 1:
            raise ValueError("expected exactly one top-level spirv.module operation")
        return str(operations[0])

    def render_optimized_spirv_module_op(self, mlir_opt=None, optimization_strategy=None):
        return _extract_single_top_level_operation(
            self.optimize_spirv_opencl_module(
                mlir_opt=mlir_opt,
                optimization_strategy=optimization_strategy,
            ),
            "spirv.module",
        )

    def build_spirv_opencl_module(self):
        return _SPIRVOpenCLModuleBuilder(self).build()

    def lower_to_opencl_c_source(self):
        return _OpenCLCKernelSourceBuilder(self).build()

    def validate_with_mlir_opt(self, mlir_opt=None):
        mlir_opt = mlir_opt or _find_mlir_opt()
        module = self.render_spirv_opencl_module()
        result = subprocess.run(
            [mlir_opt, "-", "--verify-diagnostics"],
            input=module,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    def write_inspection_artifacts(
        self,
        output_dir,
        *,
        mlir_translate=None,
        include_spirv_binary=True,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / f"{self.model.mesh_kernel_name}_opencl"
        strategy_suffix = self.optimization_strategy.value

        module_path = prefix.with_suffix(".spirv.module.mlir")
        optimized_module_path = prefix.with_suffix(f".{strategy_suffix}.optimized.spirv.module.mlir")
        op_path = prefix.with_suffix(".spirv.mlir")
        optimized_op_path = prefix.with_suffix(f".{strategy_suffix}.optimized.spirv.mlir")
        binary_path = prefix.with_suffix(".spv")
        optimized_binary_path = prefix.with_suffix(f".{strategy_suffix}.optimized.spv")
        opencl_c_path = prefix.with_suffix(".cl")

        module_path.write_text(self.render_spirv_opencl_module())
        optimized_module_path.write_text(self.optimize_spirv_opencl_module())
        op_path.write_text(self.render_spirv_module_op())
        optimized_op_path.write_text(self.render_optimized_spirv_module_op())
        opencl_c_path.write_text(self.lower_to_opencl_c_source())

        files = [module_path, optimized_module_path, op_path, optimized_op_path, opencl_c_path]
        if include_spirv_binary:
            _serialize_spirv_module(
                op_path,
                binary_path,
                mlir_translate=mlir_translate,
            )
            _serialize_spirv_module(
                optimized_op_path,
                optimized_binary_path,
                mlir_translate=mlir_translate,
            )
            files.extend((binary_path, optimized_binary_path))

        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=tuple(str(path) for path in files),
        )


class _SPIRVOpenCLModuleBuilder:
    ADDRESSING_MODEL = SPIRVAddressingModel.LOGICAL
    MEMORY_MODEL = SPIRVMemoryModel.OPENCL
    EXECUTION_MODEL = SPIRVExecutionModel.KERNEL
    FUNCTION_CONTROL = SPIRVFunctionControl.NONE
    STORAGE_GLOBAL = SPIRVStorageClass.CROSS_WORKGROUP
    STORAGE_INPUT = SPIRVStorageClass.INPUT

    def __init__(self, lowering):
        self.lowering = lowering
        self.context = None
        self.ir = None
        self.spirv = None

    def build(self):
        self._load_bindings()
        with self.ir.Context() as context, self.ir.Location.unknown():
            self.context = context
            module = self.ir.Module.create()
            with self.ir.InsertionPoint(module.body):
                spirv_module = self._create_spirv_module()
                body = self.ir.Block.create_at_start(spirv_module.regions[0])
                with self.ir.InsertionPoint(body):
                    self._declare_globals()
                    for kernel in self.lowering.copy_kernels:
                        self._copy_kernel(kernel)
                    self._entry_point(
                        self.lowering.map_kernel.symbol,
                        (
                            self.lowering.connectivity_buffer,
                            self.lowering.map_kernel.source,
                            self.lowering.map_kernel.destination,
                            self.lowering.work_item_builtins[0],
                        ),
                    )
                    self._entry_point(
                        self.lowering.reduce_kernel.symbol,
                        (
                            self.lowering.reduce_kernel.source,
                            self.lowering.reduce_kernel.destination,
                            self.lowering.work_item_builtins[0],
                        ),
                    )
            return module

    def _load_bindings(self):
        from mlir import ir
        import mlir.dialects.spirv as spirv

        self.ir = ir
        self.spirv = spirv

    def _create_spirv_module(self):
        module = self.spirv.ModuleOp(
            self._spirv_attr("addressing_model", self.ADDRESSING_MODEL),
            self._spirv_attr("memory_model", self.MEMORY_MODEL),
            vce_triple=self.ir.Attribute.parse(
                "#spirv.vce<v1.0, [Kernel, Addresses], []>"
            ),
        )
        metadata = self._metadata()
        for key, value in metadata.items():
            module.attributes[key] = self.ir.StringAttr.get(value)
        return module

    def _metadata(self):
        model = self.lowering.model
        return {
            "sfem.codegen.model": model.material_name,
            "sfem.codegen.kernel_plan": model.kernel_name,
            "sfem.codegen.kernel_kind": model.kernel_kind,
            "sfem.codegen.element": model.element_type,
            "sfem.codegen.mesh_kernel": model.mesh_kernel_name,
            "sfem.codegen.local_apply": model.local_apply_name,
            "sfem.codegen.mesh_phases": ",".join(model.mesh_phases),
            "sfem.codegen.expression_plans": ",".join(model.expression_names),
            "sfem.opencl.execution_model": self.EXECUTION_MODEL.value,
            "sfem.opencl.memory_model": self.MEMORY_MODEL.value,
            "sfem.opencl.global_storage": self.STORAGE_GLOBAL.value,
            "sfem.opencl.map_global_size": "nelements * n_shape * dim",
            "sfem.opencl.reduce_global_size": "nnodes * dim",
        }

    def _declare_globals(self):
        for buffer in self.lowering.device_buffers:
            self._global(buffer)
        for built_in in self.lowering.work_item_builtins:
            self._gid_global(built_in)

    def _gid_global(self, built_in):
        self.spirv.GlobalVariableOp(
            self._ptr(
                self.ir.VectorType.get([3], self.ir.IntegerType.get_signless(64)),
                self.STORAGE_INPUT,
            ),
            built_in.mlir_symbol,
            built_in=self._spirv_attr("built_in", built_in.built_in),
        )

    def _entry_point(self, kernel_name, interface):
        self.spirv.EntryPointOp(
            self._spirv_attr("execution_model", self.EXECUTION_MODEL),
            kernel_name,
            tuple(entity.mlir_symbol for entity in interface),
        )

    def _global(self, buffer, type_=None, built_in=None):
        if type_ is None:
            type_ = self._global_array_ptr(self._scalar_type(buffer.scalar), buffer.extent)
        self.spirv.GlobalVariableOp(type_, buffer.mlir_symbol, built_in=built_in)

    def _copy_kernel(self, kernel):
        function = self.spirv.FuncOp(
            self.ir.TypeAttr.get(self.ir.FunctionType.get([], [])),
            kernel.symbol,
            self._spirv_attr("function_control", self.FUNCTION_CONTROL),
        )
        body = self.ir.Block.create_at_start(function.regions[0])
        with self.ir.InsertionPoint(body):
            index = self._global_invocation_index()
            input_ptr = self._element_pointer(kernel.source, index)
            output_ptr = self._element_pointer(kernel.destination, index)
            value = self.spirv.LoadOp(self._scalar_type(kernel.source.scalar), input_ptr).result
            self.spirv.StoreOp(output_ptr, value)
            self.spirv.ReturnOp()

    def _global_invocation_index(self):
        i32 = self.ir.IntegerType.get_signless(32)
        i64 = self.ir.IntegerType.get_signless(64)
        gid_vector = self.ir.VectorType.get([3], i64)
        gid_pointer = self._ptr(gid_vector, self.STORAGE_INPUT)
        gid_address = self.spirv.AddressOfOp(
            gid_pointer,
            self.lowering.work_item_builtins[0].mlir_symbol,
        ).result
        gid_value = self.spirv.LoadOp(gid_vector, gid_address).result
        gid_x = self.spirv.CompositeExtractOp(i64, gid_value, [0]).result
        return self.spirv.UConvertOp(i32, gid_x).result

    def _element_pointer(self, buffer, index):
        element_type = self._scalar_type(buffer.scalar)
        base_pointer_type = self._global_array_ptr(element_type, buffer.extent)
        element_pointer_type = self._ptr(element_type, self.STORAGE_GLOBAL)
        base = self.spirv.AddressOfOp(base_pointer_type, buffer.mlir_symbol).result
        return self.spirv.AccessChainOp(element_pointer_type, base, [index]).result

    def _scalar_type(self, scalar):
        if scalar is KernelScalar.F32:
            return self.ir.F32Type.get()
        if scalar is KernelScalar.I32:
            return self.ir.IntegerType.get_signless(32)
        raise ValueError(f"unsupported scalar type: {scalar}")

    def _global_array_ptr(self, element_type, size):
        return self._ptr(self._spirv_array(element_type, size), self.STORAGE_GLOBAL)

    def _spirv_array(self, element_type, size):
        return self.ir.Type.parse(
            f"!spirv.array<{int(size)}x{element_type}>"
        )

    def _ptr(self, pointee, storage_class):
        storage_value = storage_class.value if isinstance(storage_class, Enum) else storage_class
        return self.ir.Type.parse(
            f"!spirv.ptr<{pointee}, {storage_value}>"
        )

    def _spirv_attr(self, kind, value):
        attr_value = value.value if isinstance(value, Enum) else value
        return self.ir.Attribute.parse(f"#spirv.{kind}<{attr_value}>")


class _OpenCLCKernelSourceBuilder:
    def __init__(self, lowering):
        self.lowering = lowering

    def build(self):
        self._validate_affine_tet4_apply()
        model = self.lowering.model
        rg = ", ".join(self._float_literal(value) for value in model.reference_gradients)
        qw = self._float_literal(model.quadrature_weights[0])
        material = [self._c_expression(expr) for expr in model.apply_material_expressions]
        lines = [
            "// Generated by MatrixFreeOpenCLMLIRLowering from the SFEM kernel model.",
            "// Apple OpenCL 1.2 consumes this OpenCL C target because SPIR-V is not accepted by the runtime.",
            "__kernel void tet4_le_map(__global const int *connectivity,",
            "                          __global const float *direction,",
            "                          __global const float *adjugate,",
            "                          __global const float *determinant,",
            "                          const float lmbda,",
            "                          const float mu,",
            "                          __global float *scratch) {",
            "    const int elem = get_global_id(0);",
            f"    const float rg[{model.reference_gradient_size}] = {{{rg}}};",
            "",
            f"    float ref[{model.dim * model.dim}];",
            f"    for (int row = 0; row < {model.dim}; ++row) {{",
            f"        for (int col = 0; col < {model.dim}; ++col) {{",
            "            float acc = 0.0f;",
            f"            for (int shape = 0; shape < {model.n_shape}; ++shape) {{",
            f"                const int node = connectivity[{model.n_shape} * elem + shape];",
            f"                acc += direction[{model.dim} * node + row] * rg[{model.dim} * shape + col];",
            "            }",
            f"            ref[{model.dim} * row + col] = acc;",
            "        }",
            "    }",
            "",
            "    const float inv_det = 1.0f / determinant[elem];",
            f"    float grad[{model.dim * model.dim}];",
            f"    for (int row = 0; row < {model.dim}; ++row) {{",
            f"        for (int col = 0; col < {model.dim}; ++col) {{",
            "            float acc = 0.0f;",
            f"            for (int k = 0; k < {model.dim}; ++k) {{",
            f"                acc += ref[{model.dim} * row + k] * adjugate[{model.dim * model.dim} * elem + {model.dim} * k + col];",
            "            }",
            f"            grad[{model.dim} * row + col] = acc * inv_det;",
            "        }",
            "    }",
            "",
            f"    float stress[{model.dim * model.dim}];",
        ]
        for index, expression in enumerate(material):
            lines.append(f"    stress[{index}] = {expression};")
        lines.extend(
            [
                "",
                f"    float lop[{model.dim * model.dim}];",
                f"    for (int row = 0; row < {model.dim}; ++row) {{",
                f"        for (int col = 0; col < {model.dim}; ++col) {{",
                "            float acc = 0.0f;",
                f"            for (int k = 0; k < {model.dim}; ++k) {{",
                f"                acc += stress[{model.dim} * row + k] * adjugate[{model.dim * model.dim} * elem + {model.dim} * col + k];",
                "            }",
                f"            lop[{model.dim} * row + col] = {qw} * acc;",
                "        }",
                "    }",
                "",
                f"    for (int shape = 0; shape < {model.n_shape}; ++shape) {{",
                f"        for (int row = 0; row < {model.dim}; ++row) {{",
                "            float acc = 0.0f;",
                f"            for (int col = 0; col < {model.dim}; ++col) {{",
                f"                acc += lop[{model.dim} * row + col] * rg[{model.dim} * shape + col];",
                "            }",
                f"            scratch[{model.scratch_components} * elem + {model.dim} * shape + row] = acc;",
                "        }",
                "    }",
                "}",
                "",
                "__kernel void tet4_le_reduce(__global const float *scratch,",
                "                             __global const int *node_degree,",
                "                             __global const int *node_to_element_map,",
                "                             __global const int *node_to_local_idx,",
                "                             const int max_node_degree,",
                "                             __global float *output) {",
                "    const int node = get_global_id(0);",
                "    const int degree = node_degree[node];",
                f"    for (int component = 0; component < {model.dim}; ++component) {{",
                "        float acc = 0.0f;",
                "        for (int i = 0; i < degree; ++i) {",
                "            const int map_index = node * max_node_degree + i;",
                "            const int elem = node_to_element_map[map_index];",
                "            const int local_idx = node_to_local_idx[map_index];",
                f"            acc += scratch[{model.scratch_components} * elem + {model.dim} * local_idx + component];",
                "        }",
                f"        output[{model.dim} * node + component] = acc;",
                "    }",
                "}",
                "",
            ]
        )
        return "\n".join(lines)

    def _c_expression(self, expression):
        code = sp.ccode(sp.sympify(expression))
        for index in range(self.lowering.model.dim * self.lowering.model.dim):
            code = code.replace(f"trial_grad{index}", f"grad[{index}]")
        return code

    def _float_literal(self, value):
        text = f"{float(value):.20g}"
        if "e" not in text and "E" not in text and "." not in text:
            text = f"{text}.0"
        return f"{text}f"

    def _validate_affine_tet4_apply(self):
        model = self.lowering.model
        if model.element_type != "TET4" or model.dim != 3 or model.n_shape != 4 or model.n_qp != 1:
            raise ValueError("initial OpenCL C MLIR target supports affine TET4 only")
        if tuple(model.parameters) != ("lmbda", "mu"):
            raise ValueError("initial OpenCL C MLIR target expects linear-elasticity parameters lmbda, mu")
