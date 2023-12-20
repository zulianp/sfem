# SFEMBuildTypes.cmake

include(cmake/SFEMSanitizer.cmake)

set(CMAKE_CXX_FLAGS_AVX2
    "-Ofast -DNDEBUG  -mavx2 "
    CACHE STRING "Flags for using fast operations and avx2" FORCE)
