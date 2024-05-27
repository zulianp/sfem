# FindISolverFunction.cmake

if(NOT ISolverFunction_DIR OR NOT ISolverLSolve_DIR OR NOT ISolverLSolveFrontend_DIR OR NOT ISolverPlugin_DIR)

    find_path(
        ISolverFunction_DIR  isolver_function.h 
        HINTS ${CMAKE_SOURCE_DIR}/../isolver/interfaces/nlsolve/
        )

    find_path(
        ISolverLSolve_DIR  isolver_lsolve.h
        HINTS ${CMAKE_SOURCE_DIR}/../isolver/interfaces/lsolve/
        )

    find_path(
        ISolverLSolveFrontend_DIR  isolver_lsolve_frontend.cpp
        HINTS ${CMAKE_SOURCE_DIR}/../isolver/plugin/lsolve/
        )

    find_path(
        ISolverPlugin_DIR  isolver_PlugIn.hpp
        HINTS ${CMAKE_SOURCE_DIR}/../isolver/plugin
        )

endif()

if(NOT ISolverFunction_DIR OR NOT ISolverLSolve_DIR OR NOT ISolverLSolveFrontend_DIR OR NOT ISolverPlugin_DIR)
	message(FATAL_ERROR "Isolver files not found not found! ISolverFunction_DIR=${ISolverFunction_DIR} ISolverLSolve_DIR=${ISolverLSolve_DIR} ISolverLSolveFrontend_DIR=${ISolverLSolveFrontend_DIR} ISolverPlugin_DIR=${ISolverPlugin_DIR}")
endif()

scan_directories(${ISolverFunction_DIR} "." SFEM_BUILD_INCLUDES
   SFEM_HEADERS SFEM_SOURCES)


scan_directories(${ISolverLSolve_DIR} "." SFEM_BUILD_INCLUDES
   SFEM_HEADERS SFEM_SOURCES)


scan_directories(${ISolverPlugin_DIR} "." SFEM_BUILD_INCLUDES
   SFEM_HEADERS SFEM_SOURCES)

scan_directories(${ISolverLSolveFrontend_DIR} "." SFEM_BUILD_INCLUDES
   SFEM_HEADERS SFEM_SOURCES)

list(APPEND SFEM_DEP_LIBRARIES ${CMAKE_DL_LIBS})

# message(STATUS "SFEM_DEP_LIBRARIES = ${SFEM_DEP_LIBRARIES}")
