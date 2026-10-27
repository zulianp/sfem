# GenerateMakeConfig.cmake

function(list_to_makefile_def output_string pre_list post_list item_prefix
         list_of_strings)

    list(REMOVE_DUPLICATES list_of_strings)

    set(tempstr ${${output_string}} ${pre_list})

    foreach(list_el ${list_of_strings})
        string(REGEX MATCH "(.*)framework" is_frame_work "${list_el}")

        if(is_frame_work)
            get_filename_component(FRAMEWORK_NAME "${list_el}" NAME)
            string(REGEX REPLACE "/${FRAMEWORK_NAME}" "" FRAMEWORK_PATH
                                 "${list_el}")
            string(REGEX REPLACE ".framework" "" FRAMEWORK_NAME
                                 "${FRAMEWORK_NAME}")

            set(tempstr
                "${tempstr}-F${FRAMEWORK_PATH} -framework ${FRAMEWORK_NAME} ")
        else()
            string(REGEX MATCH "LINK_ONLY" is_gen_expr "${list_el}")
            string(REGEX MATCH ";" is_trash "${list_el}")

            if(is_gen_expr OR is_trash)
                # message("${list_el}")
            else()
                set(tempstr "${tempstr}${item_prefix}${list_el} ")
            endif()
        endif()

    endforeach(list_el)

    set(tempstr "${tempstr} ${post_list}")

    set(${output_string}
        ${tempstr}
        PARENT_SCOPE)

endfunction()

# Appends to the caller's INTERFACE_INCLUDE_DIRECTORIES, INTERFACE_COMPILE_FLAGS and
# INTERFACE_LINK_LIBRARIES what linking `item` takes: a target's file (with an rpath to its
# directory when it is a shared library), its include directories, compile options and definitions,
# then the same for everything it links, depth first. Generator expressions other than LINK_ONLY are
# build-time only and skipped. _make_visited holds the targets already resolved.
function(_sfem_make_resolve item)
    if(item MATCHES "^\\$<LINK_ONLY:(.*)>$")
        set(item "${CMAKE_MATCH_1}")
    endif()
    if(item MATCHES "\\$<")
        return()
    endif()

    if(NOT TARGET ${item})
        if(item MATCHES "^-" OR IS_ABSOLUTE "${item}")
            list(APPEND INTERFACE_LINK_LIBRARIES "${item}")
        elseif(item)
            list(APPEND INTERFACE_LINK_LIBRARIES "-l${item}")
        endif()
        set(INTERFACE_LINK_LIBRARIES "${INTERFACE_LINK_LIBRARIES}" PARENT_SCOPE)
        return()
    endif()

    get_target_property(_aliased ${item} ALIASED_TARGET)
    if(_aliased)
        set(item ${_aliased})
    endif()
    if(item IN_LIST _make_visited)
        return()
    endif()
    list(APPEND _make_visited ${item})

    get_target_property(_type ${item} TYPE)
    if(NOT _type STREQUAL "INTERFACE_LIBRARY")
        get_target_property(_location ${item} LOCATION)
        if(_location)
            list(APPEND INTERFACE_LINK_LIBRARIES "${_location}")
            if(_type STREQUAL "SHARED_LIBRARY")
                get_filename_component(_dir "${_location}" DIRECTORY)
                list(APPEND INTERFACE_LINK_LIBRARIES "-Wl,-rpath,${_dir}")
            endif()
        endif()
    endif()

    get_target_property(_includes ${item} INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_include ${_includes})
        if(_include AND NOT _include MATCHES "\\$<")
            list(APPEND INTERFACE_INCLUDE_DIRECTORIES "${_include}")
        endif()
    endforeach()

    get_target_property(_options ${item} INTERFACE_COMPILE_OPTIONS)
    foreach(_option ${_options})
        # Options for C++ sources (OpenMP's -fopenmp) apply: the makefile compiles C++.
        if(_option MATCHES "^\\$<\\$<COMPILE_LANGUAGE:CXX>:(.*)>$")
            set(_option "${CMAKE_MATCH_1}")
        endif()
        string(REGEX REPLACE "^SHELL:" "" _option "${_option}")
        if(_option AND NOT _option MATCHES "\\$<")
            list(APPEND INTERFACE_COMPILE_FLAGS "${_option}")
        endif()
    endforeach()

    get_target_property(_definitions ${item} INTERFACE_COMPILE_DEFINITIONS)
    foreach(_definition ${_definitions})
        if(_definition AND NOT _definition MATCHES "\\$<")
            list(APPEND INTERFACE_COMPILE_FLAGS "-D${_definition}")
        endif()
    endforeach()

    get_target_property(_links ${item} INTERFACE_LINK_LIBRARIES)
    foreach(_link ${_links})
        if(_link)
            _sfem_make_resolve("${_link}")
        endif()
    endforeach()

    set(_make_visited "${_make_visited}" PARENT_SCOPE)
    set(INTERFACE_INCLUDE_DIRECTORIES "${INTERFACE_INCLUDE_DIRECTORIES}" PARENT_SCOPE)
    set(INTERFACE_COMPILE_FLAGS "${INTERFACE_COMPILE_FLAGS}" PARENT_SCOPE)
    set(INTERFACE_LINK_LIBRARIES "${INTERFACE_LINK_LIBRARIES}" PARENT_SCOPE)
endfunction()

function(generate_make_config prefix target)
    string(TOUPPER ${prefix} var_prefix)
    string(TOLOWER ${prefix} file_prefix)

    if(NOT TARGET ${target})
        message(STATUS "target=${target} does not exists")
        return()
    endif()

    # message(STATUS "target=${target}")

    # ##########################################################################

    # The target and every target it links, resolved to files and flags a makefile can use.
    set(_make_visited "")
    set(INTERFACE_INCLUDE_DIRECTORIES "")
    set(INTERFACE_COMPILE_FLAGS "")
    set(INTERFACE_LINK_LIBRARIES "")
    _sfem_make_resolve(${target})

    # get_target_property(COMPILE_FLAGS ${target} "COMPILE_FLAGS")
    # message("COMPILE_FLAGS=${COMPILE_FLAGS}")

    # get_target_property(INTERFACE_COMPILE_OPTIONS ${target}
    #                     "INTERFACE_COMPILE_OPTIONS")
    # message("INTERFACE_COMPILE_OPTIONS=${INTERFACE_COMPILE_OPTIONS}")

    # ##########################################################################

    set(header_str
        "#Automatically generated by CMake with generate_make_config for target ${target}\n\n"
    )

    list_to_makefile_def(include_str "\n${var_prefix}_INCLUDES=" "\n" "-I"
                         "${INTERFACE_INCLUDE_DIRECTORIES}")

    list_to_makefile_def(flags_str "\n${var_prefix}_CXXFLAGS=" "\n" ""
                         "${INTERFACE_COMPILE_FLAGS}")

    list_to_makefile_def(link_str "\n${var_prefix}_LIBRARIES=" "\n"
                         "" "${INTERFACE_LINK_LIBRARIES}")

    set(config_content
        "${header_str}SFEM_C_COMPILER=${CMAKE_C_COMPILER}\n\nSFEM_CXX_COMPILER=${CMAKE_CXX_COMPILER}\n${include_str} ${flags_str} ${link_str}"
    )

    # ##########################################################################

    set(filename "${CMAKE_BINARY_DIR}/${file_prefix}-config.makefile")

    file(WRITE ${filename} "${config_content}")

    install(FILES "${CMAKE_BINARY_DIR}/${file_prefix}-config.makefile"
            DESTINATION config)

endfunction()
