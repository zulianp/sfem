# SFEMBuildTypes.cmake

include(cmake/SFEMSanitizer.cmake)

set(CMAKE_CXX_FLAGS_AVX2
    "-Ofast -march=core-avx2 -fno-trapping-math -fassociative-math -DNDEBUG -DSFEM_ENABLE_AVX2_SORT -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx2" FORCE)

set(CMAKE_C_FLAGS_AVX2
    "-Ofast -march=core-avx2 -fno-trapping-math -fassociative-math -DNDEBUG -DSFEM_ENABLE_AVX2_SORT -DSFEM_ENABLE_EXPLICIT_VECTORIZATION -Iexternal"
    CACHE STRING "Flags for using fast operations and avx2" FORCE)


set(CMAKE_CXX_FLAGS_PROF
    "-O1 -g -DNDEBUG "
    CACHE STRING "Flags for profiling configuration" FORCE
    )


# set(CMAKE_CUDA_FLAGS "--compiler-options \"-fPIC\" -arch=sm_90 -Xptxas=-O3,-v -use_fast_math")
set(CMAKE_CUDA_FLAGS "--compiler-options \"-fPIC\" -arch=sm_60 -Xptxas=-O3,-v -use_fast_math")
