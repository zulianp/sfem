find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python_EXECUTABLE=${Python_EXECUTABLE}")

if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)

  if(NOT nanobind_DIR)
    message(WARNING "Unable to execute nanobind query")
  endif()

endif()

list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")

find_package(nanobind CONFIG REQUIRED)
