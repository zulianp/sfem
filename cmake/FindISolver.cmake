# FindISolver.cmake

if(NOT ISolver_DIR)

find_path(
    ISolver_DIR  isolver_function.h 
    HINTS ${CMAKE_SOURCE_DIR}/../isolver/interfaces/nlsolve/
    )

endif()

if(NOT ISolver_DIR)
	message(FATAL_ERROR "ISolver not found!")
endif()

scan_directories(${ISolver_DIR} "." SFEM_BUILD_INCLUDES
                 SFEM_HEADERS SFEM_SOURCES)

# set(SFEM_BUILD_INCLUDES
#     ${SFEM_BUILD_INCLUDES}
#     PARENT_SCOPE)

# set(SFEM_HEADERS
#     ${SFEM_HEADERS}
#     PARENT_SCOPE)

# set(SFEM_SOURCES
#     ${SFEM_SOURCES}
#     PARENT_SCOPE)