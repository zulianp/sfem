cmake_minimum_required(VERSION 3.20)

project(cvfem_tet4_ns_upwind_bench LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(DEFINED SFEM_DIR)
    foreach(_dep ryml matrixio smesh SCCD ssdf)
        if(NOT DEFINED ${_dep}_DIR AND EXISTS "${SFEM_DIR}/${_dep}/${_dep}Config.cmake")
            set(${_dep}_DIR "${SFEM_DIR}/${_dep}")
        endif()
    endforeach()
endif()

find_package(ryml CONFIG REQUIRED)
find_package(matrixio CONFIG REQUIRED)
find_package(smesh CONFIG REQUIRED)
find_package(SCCD CONFIG REQUIRED)
find_package(ssdf CONFIG REQUIRED)
find_package(SFEM CONFIG REQUIRED)

add_executable(cvfem_tet4_ns_upwind_bench
    cvfem_tet4_ns_upwind_bench.cpp)

target_link_libraries(cvfem_tet4_ns_upwind_bench
    PRIVATE
        SFEM::sfem)

target_compile_options(cvfem_tet4_ns_upwind_bench
    PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-O3>
        $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-DNDEBUG>)

set_target_properties(cvfem_tet4_ns_upwind_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
