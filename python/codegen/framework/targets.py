from dataclasses import dataclass
from enum import Enum


class TargetLanguage(Enum):
    CPP = "c++"
    CUDA = "cuda"


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

    @property
    def supports_device_kernels(self):
        return False


@dataclass(frozen=True)
class OpenMPTarget(TargetPlatform):
    name: str = "openmp"
    language: TargetLanguage = TargetLanguage.CPP
    default_alignment: int = 64

    def function_qualifier(self):
        return "static SFEM_INLINE"

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


@dataclass(frozen=True)
class CUDATarget(TargetPlatform):
    name: str = "cuda"
    language: TargetLanguage = TargetLanguage.CUDA
    default_alignment: int = 16

    def includes(self):
        return ("#include <cuda_runtime.h>",)

    def function_qualifier(self):
        return "__device__ __forceinline__"

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

    @property
    def supports_device_kernels(self):
        return True


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
