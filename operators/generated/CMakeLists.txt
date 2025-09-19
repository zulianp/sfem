cmake_minimum_required(VERSION 3.10)

# FIXME
# Codegen configuration (used when SFEM_ENABLE_CODEGEN=ON)
set(SFEM_CODEGEN_PYTHON "${CMAKE_SOURCE_DIR}/venv/bin/python3" CACHE STRING "Python interpreter for codegen")
set(SFEM_CODEGEN_W "DEMO_NEOHOOKEAN_OGDEN" CACHE STRING "Strain energy function string (invariants) or DEMO_NEOHOOKEAN_OGDEN")
set(SFEM_CODEGEN_FE "TET4" CACHE STRING "Element for codegen (e.g., TET4, HEX8)")

# Collect all generated hyperelasticity sources (configure-time and subsequent regenerations)
if(SFEM_ENABLE_CODEGEN)
    # Configure-time codegen: generate specific outputs so target always exists
    string(TOLOWER ${SFEM_CODEGEN_FE} FE_TAG)
    set(GEN_OUT_GRAD "${CMAKE_CURRENT_LIST_DIR}/hyperelasticity_${FE_TAG}_gradient.c")
    set(GEN_OUT_APPLY "${CMAKE_CURRENT_LIST_DIR}/hyperelasticity_${FE_TAG}_apply.c")
    set(GEN_OUT_VALUE "${CMAKE_CURRENT_LIST_DIR}/hyperelasticity_${FE_TAG}_value.c")

    file(GLOB CODEGEN_TPLS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tpl/hyperelasticity/*.c")
    add_custom_command(
        OUTPUT ${GEN_OUT_GRAD} ${GEN_OUT_APPLY} ${GEN_OUT_VALUE}
        COMMAND ${SFEM_CODEGEN_PYTHON} ${CMAKE_SOURCE_DIR}/python/codegen/sr_hyperelasticity.py ${SFEM_CODEGEN_W} ${SFEM_CODEGEN_FE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${CMAKE_SOURCE_DIR}/python/codegen/sr_hyperelasticity.py
                ${CMAKE_SOURCE_DIR}/python/codegen/tpl_hyperelasticity.py
                ${CODEGEN_TPLS}
        COMMENT "Generating hyperelasticity sources (${SFEM_CODEGEN_FE}) from ${SFEM_CODEGEN_W}"
        VERBATIM)

    set(GEN_SRC ${GEN_OUT_GRAD} ${GEN_OUT_APPLY} ${GEN_OUT_VALUE})
else()
    file(GLOB GEN_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/hyperelasticity_*.c")
endif()

add_library(hyperelasticity SHARED EXCLUDE_FROM_ALL ${GEN_SRC})

target_include_directories(hyperelasticity PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    ${SFEM_BUILD_INCLUDES}
)

set_target_properties(hyperelasticity PROPERTIES OUTPUT_NAME "hyperelasticity")

# Optional Neo-Hookean plugin (auto-detected)
file(GLOB NEOHOOKEAN_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/neohookean_*.c")
if(NEOHOOKEAN_SRC)
    add_library(neohookean SHARED EXCLUDE_FROM_ALL ${NEOHOOKEAN_SRC})
    target_include_directories(neohookean PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        ${SFEM_BUILD_INCLUDES}
    )
    set_target_properties(neohookean PROPERTIES OUTPUT_NAME "neohookean")
endif()
