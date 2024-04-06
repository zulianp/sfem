find_package(Python 3.11.5 COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python_EXECUTABLE=${Python_EXECUTABLE}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(nanobind CONFIG REQUIRED)
