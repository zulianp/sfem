# SFEMBuildTypes.cmake

include(cmake/SFEMSanitizer.cmake)
include(CheckCCompilerFlag)

option(SFEM_CPU_ARCH "CPU architecture" OFF)
set(SFEM_MARCH_SWITCH "-march=")

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    check_c_compiler_flag("-mcpu=apple-m1" APPLE_M1)
    if(APPLE_M1)
        set(SFEM_MARCH_SWITCH "-mcpu=")
        set(SFEM_CPU_ARCH "apple-m1")
    endif()
endif()

# recognize if the CPU is ARM64 or x86_64
if (NOT SFEM_CPU_ARCH OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(SFEM_CPU_ARCH "native")
else()
    set(SFEM_CPU_ARCH "${SFEM_CPU_ARCH}")
endif()

if(NOT SFEM_CUDA_ARCH)
    set(SFEM_CUDA_ARCH 60 CACHE STRING "Choose the CUDA device capabilities." FORCE)
endif()

set(ARM64_VECTOR_BITS 128) ## Default value for ARM64 (at the moment)

## TODO:
## Verify if the aarm64 option -msve-vector-bits is supported also by the Apple Silicon M CPU
## and verify whath the best simd size for Apple Silicon M CPU V8 or V4?

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # set (VECTOR_OPT "-msve-vector-bits=${ARM64_VECTOR_BITS}")
    # set (VECTOR_OPT_v8 "-msve-vector-bits=${ARM64_VECTOR_BITS}")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set (VECTOR_OPT "-mavx2 -DSFEM_ENABLE_AVX2_SORT ")
    set (VECTOR_OPT_v8 "-mavx512f")
else()
    set (VECTOR_OPT "")
endif()

##  -DSFEM_ENABLE_AVX512_SORT ## the lib is not present in the code
## -DSFEM_ENABLE_AVX2_SORT ## the lib is not present in the code

message(STATUS "SFEM_CPU_ARCH: ${SFEM_CPU_ARCH}, CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}") 

set(CMAKE_CXX_FLAGS_AVX2
    "-Ofast ${SFEM_MARCH_SWITCH}${SFEM_CPU_ARCH} ${VECTOR_OPT} -fno-trapping-math   -DNDEBUG  -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx2" FORCE)

set(CMAKE_C_FLAGS_AVX2
    "-Ofast ${SFEM_MARCH_SWITCH}${SFEM_CPU_ARCH} ${VECTOR_OPT} -fno-trapping-math   -DNDEBUG  -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx2" FORCE)

set(CMAKE_CXX_FLAGS_AVX512
    "-Ofast ${SFEM_MARCH_SWITCH}${SFEM_CPU_ARCH} ${VECTOR_OPT_v8} -fno-trapping-math   -DNDEBUG  -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx512" FORCE)

set(CMAKE_C_FLAGS_AVX512
    "-Ofast ${SFEM_MARCH_SWITCH}${SFEM_CPU_ARCH} ${VECTOR_OPT_v8} -fno-trapping-math   -DNDEBUG  -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx512" FORCE)


message(STATUS "CMAKE_CXX_FLAGS_AVX2: ${CMAKE_CXX_FLAGS_AVX2}")
message(STATUS "CMAKE_C_FLAGS_AVX2: ${CMAKE_C_FLAGS_AVX2}")

set(CMAKE_CXX_FLAGS_PROF
    "-O1 -g -DNDEBUG "
    CACHE STRING "Flags for profiling configuration" FORCE
    )


# set(CMAKE_CUDA_FLAGS "--compiler-options \"-fPIC\" -arch=sm_90 -Xptxas=-O3,-v -use_fast_math")
set(CMAKE_CUDA_FLAGS " -O3 -use_fast_math -Xcompiler=-O3,-march=native,-mtune=native,-fPIC -arch=sm_${SFEM_CUDA_ARCH} -Xptxas=-O3,-v ")

message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
