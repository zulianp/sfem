from dataclasses import dataclass
from enum import Enum


class TargetLanguage(Enum):
    CPP = "c++"
    CUDA = "cuda"


@dataclass(frozen=True)
class TargetPlatform:
    name: str
    language: TargetLanguage

    @property
    def generated_language(self):
        return self.language.value

    def includes(self):
        return ()

    def function_qualifier(self):
        return "static inline"

    def parallel_for_pragma(self):
        return None


@dataclass(frozen=True)
class OpenMPTarget(TargetPlatform):
    name: str = "openmp"
    language: TargetLanguage = TargetLanguage.CPP

    def parallel_for_pragma(self):
        return "#pragma omp parallel for"


@dataclass(frozen=True)
class CUDATarget(TargetPlatform):
    name: str = "cuda"
    language: TargetLanguage = TargetLanguage.CUDA

    def includes(self):
        return ("#include <cuda_runtime.h>",)

    def function_qualifier(self):
        return "__device__ __forceinline__"
