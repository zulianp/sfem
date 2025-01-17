set(SFEM_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(SFEM_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(SFEM_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(SFEM_VERSION_TWEAK ${PROJECT_VERSION_TWEAK})
set(SFEM_VERSION "${PROJECT_VERSION}")

find_package(Git QUIET)

if(Git_FOUND)

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --always HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE res
        OUTPUT_VARIABLE SFEM_GIT_VERSION
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    set_property(GLOBAL APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                                        "${CMAKE_CURRENT_SOURCE_DIR}/.git/index")

endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configuration/sfem_version.hpp.in
    ${CMAKE_BINARY_DIR}/sfem_version.hpp)

# configure_file(
# ${CMAKE_CURRENT_SOURCE_DIR}/sfem_configuration_details.hpp.in
# ${CMAKE_BINARY_DIR}/sfem_configuration_details.hpp)

install(FILES "${CMAKE_BINARY_DIR}/sfem_version.hpp" DESTINATION include)

# install(FILES "${CMAKE_BINARY_DIR}/sfem_configuration_details.hp"
# DESTINATION include)
