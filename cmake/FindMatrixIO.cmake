# FindMatrixIO.cmake

if(NOT MatrixIO_DIR)

find_path(
    MatrixIO_DIR  matrixio_array.h
    HINTS 
    ${MATRIXIO_DIR}
    $ENV{MATRIXIO_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../matrix.io
    )

endif()

if(NOT MatrixIO_DIR)
	message(FATAL_ERROR "MatrixIO not found!")
endif()

scan_directories(${MatrixIO_DIR} "." SFEM_BUILD_INCLUDES
                 SFEM_HEADERS SFEM_SOURCES)

# Compiled into the library from another project's checkout: SFEM_ENABLE_DEV_MODE's warnings are
# not for it (smesh silences its vendored sources the same way).
file(GLOB _matrixio_sources "${MatrixIO_DIR}/*.c" "${MatrixIO_DIR}/*.cpp")
set_source_files_properties(${_matrixio_sources} PROPERTIES COMPILE_OPTIONS "-w")

# set(SFEM_BUILD_INCLUDES
#     ${SFEM_BUILD_INCLUDES}
#     PARENT_SCOPE)

# set(SFEM_HEADERS
#     ${SFEM_HEADERS}
#     PARENT_SCOPE)

# set(SFEM_SOURCES
#     ${SFEM_SOURCES}
#     PARENT_SCOPE)