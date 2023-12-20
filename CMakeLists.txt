cmake_minimum_required(VERSION 3.5...3.19)
cmake_policy(SET CMP0048 NEW)

project(
    sfem
    VERSION 0.1.0
    LANGUAGES "C" "CXX"
    HOMEPAGE_URL "https://bitbucket.org/zulianp/sfem"
    DESCRIPTION
        "SFEM: Simple FEM"
)

# Std CMake imports
include(CMakePackageConfigHelpers)

# Our CMake imports
include(cmake/SFEMCompilerFeatures.cmake)
include(cmake/SFEMVersion.cmake)
include(cmake/SFEMDefaults.cmake)
include(cmake/SFEMOptions.cmake)
include(cmake/SFEMBuildTypes.cmake)
include(cmake/SFEMCMakeFunctions.cmake)
include(cmake/SFEMDependencies.cmake)

# ##############################################################################
# ##############################################################################
# Library
# ##############################################################################

set(SFEM_MODULES
    algebra base graphs matrix resampling solver
    operators 
        operators/beam2 
        operators/cvfem 
        operators/kernels 
        operators/navier_stokes
        operators/phase_field_for_fracture
        operators/tet10
        operators/tet4
        operators/tri3
        operators/tri6
        operators/trishell3
            )

foreach(MODULE ${SFEM_MODULES})
    # add_subdirectory("${MODULE}")

    # find_project_files(${CMAKE_SOURCE_DIR}/$MODULE "." SFEM_C_HEADERS
    #                              SFEM_C_SOURCES)

    scan_directories(${CMAKE_SOURCE_DIR}/${MODULE} "." SFEM_BUILD_INCLUDES
                     SFEM_HEADERS SFEM_SOURCES)


endforeach(MODULE)

message(STATUS "SFEM_BUILD_INCLUDES: ${SFEM_BUILD_INCLUDES}")
message(STATUS "SFEM_HEADERS: ${SFEM_HEADERS}") 
message(STATUS "SFEM_SOURCES: ${SFEM_SOURCES}")

add_library(sfem ${SFEM_SOURCES} ${SFEM_HEADERS})

# ##############################################################################
# include

foreach(include ${SFEM_BUILD_INCLUDES})
    target_include_directories(sfem BEFORE
                               PUBLIC $<BUILD_INTERFACE:${include}>)
endforeach()

target_include_directories(
    sfem BEFORE
    PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:include> $<INSTALL_INTERFACE:include/kernels>)

target_include_directories(sfem PUBLIC ${SFEM_DEP_INCLUDES})

# ##############################################################################
# link

target_link_libraries(sfem PUBLIC ${SFEM_DEP_LIBRARIES})

# ##############################################################################
# properties

set_compiler_features(sfem)
set_target_properties(
    sfem PROPERTIES VERSION ${SFEM_VERSION}
                             SOVERSION ${SFEM_VERSION_MAJOR})

# ##############################################################################
# ##############################################################################
# Executables
# add_subdirectory(examples)

if(SFEM_ENABLE_TESTING)
    include(tests/tests.cmake)
endif()

if(SFEM_ENABLE_BENCHMARK)
    include(benchmarks/benchmarks.cmake)
endif()

# ##############################################################################
# ##############################################################################
# ##############################################################################
# Installation
# ##############################################################################

install(
    TARGETS sfem
    EXPORT SFEMTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configuration/sfem_config.hpp.in
    ${CMAKE_BINARY_DIR}/sfem_config.hpp)

install(FILES ${CMAKE_BINARY_DIR}/sfem_config.hpp DESTINATION include)
install(FILES ${SFEM_HEADERS} DESTINATION include)

install(
    EXPORT SFEMTargets
    FILE SFEMTargets.cmake
    NAMESPACE SFEM::
    DESTINATION lib/cmake)

# ##############################################################################

# generate the config file that is includes the exports
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SFEMConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SFEMConfig.cmake"
    INSTALL_DESTINATION "lib/cmake")

# generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SFEMConfigVersion.cmake"
    VERSION "${SFEM_VERSION}"
    COMPATIBILITY AnyNewerVersion)

# install the configuration file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SFEMConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/SFEMConfigVersion.cmake
        DESTINATION lib/cmake/)

include(cmake/PostInstall.cmake)

# ##############################################################################
# ##############################################################################
# Extra targets
# ##############################################################################

# ##############################################################################
