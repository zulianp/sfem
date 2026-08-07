from dataclasses import dataclass
from enum import Enum


class TargetLanguage(Enum):
    CPP = "c++"
    CUDA = "cuda"
    HIP = "hip"


class ExecutionModel(Enum):
    VECTOR_LANES = "vector_lanes"
    VECTOR_LENGTH_AGNOSTIC = "vector_length_agnostic"
    SIMT_THREADS = "simt_threads"


class MatrixUnitKind(Enum):
    NONE = "none"
    ARM_SME = "arm_sme"
    CUDA_TENSOR_CORES = "cuda_tensor_cores"
    HIP_MATRIX_CORES = "hip_matrix_cores"


@dataclass(frozen=True)
class KernelVariantPolicy:
    mesh_layouts: tuple = ("standard",)
    thread_variants: tuple = ()
    warp_variants: tuple = ()
    packed_mesh_passes: tuple = ()

    @property
    def supports_packed_mesh(self):
        return "packed" in self.mesh_layouts


@dataclass(frozen=True)
class LoopLoweringPolicy:
    execution_model: ExecutionModel = ExecutionModel.VECTOR_LANES
    emits_lane_loop: bool = True
    maps_lane_to_thread: bool = False
    vectorize_lane_loop: bool = False
    parallel_element_loop: bool = False
    supports_shared_memory: bool = False
    lane_index: str = "lane"
    lane_index_type: str = "int"
    vector_size_symbol: str = "VECTOR_SIZE"
    thread_index: str = "threadIdx.x"
    block_index: str = "blockIdx.x"
    block_dim: str = "blockDim.x"
    grid_dim: str = "gridDim.x"
    vector_isa: str = ""
    vector_length_aware: bool = False
    preferred_vector_bits: int = 0
    matrix_unit: MatrixUnitKind = MatrixUnitKind.NONE


@dataclass(frozen=True)
class TargetPlatform:
    name: str
    language: TargetLanguage
    default_alignment: int = 64

    @property
    def generated_language(self):
        return self.language.value

    def includes(self):
        return ()

    def function_qualifier(self):
        return "static inline"

    def inline_qualifier(self):
        return "inline"

    def inline_definition_lines(self, inline_definition="inline"):
        return ()

    def restrict_qualifier(self):
        return ""

    def parallel_for_pragma(self, schedule=None):
        return None

    def vectorize_pragma(self):
        return None

    def atomic_update_pragma(self):
        return None

    def alignment_assumption(self, pointer, alignment=None):
        return str(pointer)

    def math_header(self):
        return "kernel_math.hpp"

    def math_helper_name(self, function, exponent=None):
        function = str(function)
        if function == "pow" and exponent is not None:
            return _pow_helper_name(exponent)
        return function

    def diagnostics_header(self):
        return "kernel_diagnostics.hpp"

    def diagnostic_print_function(self):
        return "sfem::codegen::KernelDiagnostics_print_rate"

    def kernel_launch_style(self):
        return "host_function"

    def wrapper_style(self):
        return "c_abi"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy()

    def work_item_index(self):
        policy = self.loop_lowering_policy()
        return policy.lane_index if policy.emits_lane_loop else "0"

    def work_item_name(self, name, component):
        return "%s_%s%d" % (str(name), self.work_item_index(), int(component))

    def diagnostic_work_item(self):
        return self.work_item_index()

    def work_item_loop_lines(self, indent):
        policy = self.loop_lowering_policy()
        if not policy.emits_lane_loop:
            return ("%s{" % indent,)
        lines = []
        pragma = self.vectorize_pragma() if policy.vectorize_lane_loop else None
        if pragma:
            lines.append("%s%s" % (indent, pragma))
        index = policy.lane_index
        lines.append(
            "%sfor (%s %s = 0; %s < nelems; ++%s) {"
            % (indent, policy.lane_index_type, index, index, index)
        )
        return tuple(lines)

    def parallel_element_loop_lines(self, schedule=None):
        pragma = self.parallel_for_pragma(schedule)
        return () if pragma is None else (pragma,)

    def scatter_add_lines(self, lhs, rhs, indent):
        pragma = self.atomic_update_pragma()
        lines = []
        if pragma:
            lines.append("%s%s" % (indent, pragma))
        lines.append("%s%s += %s;" % (indent, lhs, rhs))
        return tuple(lines)

    @property
    def supports_device_kernels(self):
        return False

    def vectorization_diagnostic_flags(self, compiler="c++"):
        compiler = str(compiler)
        if "clang" in compiler or compiler in ("c++", "cc"):
            return ("-Rpass=loop-vectorize", "-Werror=pass-failed")
        if "gcc" in compiler or "g++" in compiler:
            return ("-fopt-info-vec-optimized",)
        return ()

    def target_compile_flags(self):
        return ()

    def variant_policy(self):
        return KernelVariantPolicy()

    @property
    def supports_matrix_units(self):
        return self.loop_lowering_policy().matrix_unit is not MatrixUnitKind.NONE


@dataclass(frozen=True)
class OpenMPTarget(TargetPlatform):
    name: str = "openmp"
    language: TargetLanguage = TargetLanguage.CPP
    default_alignment: int = 64

    def includes(self):
        return (
            "#ifdef _OPENMP",
            "#include <omp.h>",
            "#endif",
        )

    def function_qualifier(self):
        return "static SFEM_INLINE"

    def inline_qualifier(self):
        return "SFEM_INLINE"

    def inline_definition_lines(self, inline_definition="inline"):
        return (
            "#ifndef SFEM_INLINE",
            "#define SFEM_INLINE %s" % str(inline_definition),
            "#endif",
        )

    def restrict_qualifier(self):
        return "SFEM_RESTRICT"

    def parallel_for_pragma(self, schedule=None):
        if schedule:
            return "#pragma omp parallel for schedule(%s)" % str(schedule)
        return "#pragma omp parallel for"

    def vectorize_pragma(self):
        return "#pragma omp simd"

    def atomic_update_pragma(self):
        return "#pragma omp atomic update"

    def alignment_assumption(self, pointer, alignment=None):
        alignment = self.default_alignment if alignment is None else int(alignment)
        return "__builtin_assume_aligned(%s, %d)" % (str(pointer), alignment)

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LANES,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
        )

    def work_item_name(self, name, component):
        return "%s_lane%d" % (str(name), int(component))


@dataclass(frozen=True)
class AVX512Target(OpenMPTarget):
    name: str = "avx512"
    default_alignment: int = 64

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LANES,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
            vector_isa="avx512",
            preferred_vector_bits=512,
        )

    def target_compile_flags(self):
        return ("-mavx512f",)

    def variant_policy(self):
        return KernelVariantPolicy(
            mesh_layouts=("standard", "packed"),
            packed_mesh_passes=("one_pass", "two_pass"),
        )


@dataclass(frozen=True)
class ARMSVETarget(OpenMPTarget):
    name: str = "arm_sve"
    default_alignment: int = 64

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LENGTH_AGNOSTIC,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
            lane_index_type="ptrdiff_t",
            vector_isa="arm_sve",
            vector_length_aware=True,
        )

    def target_compile_flags(self):
        return ("-march=armv8-a+sve",)

    def variant_policy(self):
        return KernelVariantPolicy(
            mesh_layouts=("standard", "packed"),
            packed_mesh_passes=("one_pass", "two_pass"),
        )


@dataclass(frozen=True)
class ARMSMETarget(ARMSVETarget):
    name: str = "arm_sme"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LENGTH_AGNOSTIC,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
            lane_index_type="ptrdiff_t",
            vector_isa="arm_sme",
            vector_length_aware=True,
            matrix_unit=MatrixUnitKind.ARM_SME,
        )

    def target_compile_flags(self):
        return ("-march=armv9-a+sme",)


@dataclass(frozen=True)
class CUDATarget(TargetPlatform):
    name: str = "cuda"
    language: TargetLanguage = TargetLanguage.CUDA
    default_alignment: int = 16

    def includes(self):
        return ("#include <cuda_runtime.h>",)

    def function_qualifier(self):
        return "__host__ __device__ __forceinline__"

    def inline_qualifier(self):
        return "__host__ __device__ __forceinline__"

    def restrict_qualifier(self):
        return "__restrict__"

    def parallel_for_pragma(self, schedule=None):
        return None

    def vectorize_pragma(self):
        return None

    def alignment_assumption(self, pointer, alignment=None):
        alignment = self.default_alignment if alignment is None else int(alignment)
        return "__builtin_assume_aligned(%s, %d)" % (str(pointer), alignment)

    def kernel_launch_style(self):
        return "cuda_grid_stride"

    def wrapper_style(self):
        return "cuda_launcher"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.SIMT_THREADS,
            emits_lane_loop=False,
            maps_lane_to_thread=True,
            vectorize_lane_loop=False,
            parallel_element_loop=False,
            supports_shared_memory=True,
            vector_isa="cuda",
            matrix_unit=MatrixUnitKind.CUDA_TENSOR_CORES,
        )

    def work_item_name(self, name, component):
        return "%s_value%d" % (str(name), int(component))

    def diagnostic_work_item(self):
        return "scalar"

    def scatter_add_lines(self, lhs, rhs, indent):
        return ("%satomicAdd(&(%s), %s);" % (indent, lhs, rhs),)

    @property
    def supports_device_kernels(self):
        return True

    def vectorization_diagnostic_flags(self, compiler="nvcc"):
        return ()

    def variant_policy(self):
        return KernelVariantPolicy(
            mesh_layouts=("standard", "packed"),
            thread_variants=("per_thread",),
            warp_variants=("per_warp",),
            packed_mesh_passes=("one_pass", "two_pass"),
        )


@dataclass(frozen=True)
class HIPTarget(CUDATarget):
    name: str = "hip"
    language: TargetLanguage = TargetLanguage.HIP

    def includes(self):
        return ("#include <hip/hip_runtime.h>",)

    def kernel_launch_style(self):
        return "hip_grid_stride"

    def wrapper_style(self):
        return "hip_launcher"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.SIMT_THREADS,
            emits_lane_loop=False,
            maps_lane_to_thread=True,
            vectorize_lane_loop=False,
            parallel_element_loop=False,
            supports_shared_memory=True,
            vector_isa="hip",
            matrix_unit=MatrixUnitKind.HIP_MATRIX_CORES,
        )


def _pow_helper_name(exponent):
    if isinstance(exponent, int):
        value = exponent
    else:
        try:
            value = int(exponent)
        except (TypeError, ValueError):
            return "pow"
        if value != exponent:
            return "pow"
    if value < 0:
        return "pow_m%d" % abs(value)
    return "pow_%d" % value
