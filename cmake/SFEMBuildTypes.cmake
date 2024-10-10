# SFEMBuildTypes.cmake

include(cmake/SFEMSanitizer.cmake)

option(SFEM_CPU_ARCH "CPU architecture" OFF)

# recognize if the CPU is ARM64 or x86_64

if (NOT SFEM_CPU_ARCH OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(SFEM_CPU_ARCH "native")
else()
    set(SFEM_CPU_ARCH "${SFEM_CPU_ARCH}")
endif()

option(CUDA_ARCH "Use CUDA architecture" OFF)

if(NOT CUDA_ARCH)
    set(CUDA_ARCH "60") ## default CUDA_ARCH
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set (VECTOR_OPT "---msve-vector-bits=128")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set (VECTOR_OPT "-mavx2")
    set (VECTOR_OPT_v8 "-mavx512f")
else()
    set (VECTOR_OPT "")
endif()

message(STATUS "SFEM_CPU_ARCH: ${SFEM_CPU_ARCH}, CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}") 

set(CMAKE_CXX_FLAGS_AVX2
    "-Ofast -march=${SFEM_CPU_ARCH} ${VECTOR_OPT} -fno-trapping-math -fassociative-math -DNDEBUG -DSFEM_ENABLE_AVX2_SORT -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx2" FORCE)

set(CMAKE_C_FLAGS_AVX2
    "-Ofast -march=${SFEM_CPU_ARCH} ${VECTOR_OPT} -fno-trapping-math -fassociative-math -DNDEBUG -DSFEM_ENABLE_AVX2_SORT -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx2" FORCE)

set(CMAKE_CXX_FLAGS_AVX512
    "-Ofast -march=${SFEM_CPU_ARCH} ${VECTOR_OPT_v8} -fno-trapping-math -fassociative-math -DNDEBUG -DSFEM_ENABLE_AVX512_SORT -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx512" FORCE)

set(CMAKE_C_FLAGS_AVX512
    "-Ofast -march=${SFEM_CPU_ARCH} ${VECTOR_OPT_v8} -fno-trapping-math -fassociative-math -DNDEBUG -DSFEM_ENABLE_AVX512_SORT -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx512" FORCE)


message(STATUS "CMAKE_CXX_FLAGS_AVX2: ${CMAKE_CXX_FLAGS_AVX2}")
message(STATUS "CMAKE_C_FLAGS_AVX2: ${CMAKE_C_FLAGS_AVX2}")

set(CMAKE_CXX_FLAGS_PROF
    "-O1 -g -DNDEBUG "
    CACHE STRING "Flags for profiling configuration" FORCE
    )


# set(CMAKE_CUDA_FLAGS "--compiler-options \"-fPIC\" -arch=sm_90 -Xptxas=-O3,-v -use_fast_math")
set(CMAKE_CUDA_FLAGS " -O3 -use_fast_math  -Xcompiler=-O3,-march=native,-mtune=native,-fPIC -arch=sm_${CUDA_ARCH} -Xptxas=-O3,-v ")

message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
