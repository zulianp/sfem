cmake_minimum_required(VERSION 3.5)
project(sfem_gen_makefile_config)

include(CMakeFindDependencyMacro)

find_package(SFEM 0 REQUIRED)

include(../GenerateMakeConfig.cmake)

generate_make_config(sfem SFEM::sfem)
