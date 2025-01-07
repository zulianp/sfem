if(NOT CMAKE_BUILD_TYPE MATCHES "[Cc][Oo][Vv][Ee][Rr][Aa][Gg][Ee]")
    message(
        FATAL_ERROR
            "This file should be included only if CMAKE_BUILD_TYPE=coverage, current is ${CMAKE_BUILD_TYPE}"
    )
endif()

# Check prereqs
find_program(GCOV_EXE gcov REQUIRED)
find_program(LCOV_EXE lcov REQUIRED)
find_program(GENHTML_EXE genhtml REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_COVERAGE
        "-O0 -g --coverage -fprofile-instr-generate -fcoverage-mapping -ftest-coverage"
        CACHE STRING "Flags used by the C++ compiler during coverage builds."
              FORCE)
else()
    set(CMAKE_CXX_FLAGS_COVERAGE
        "-O0 -g --coverage -ftest-coverage"
        CACHE STRING "Flags used by the C++ compiler during coverage builds."
              FORCE)
endif()

set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
    "--coverage"
    CACHE STRING "Flags used for linking binaries during coverage builds."
          FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
    "--coverage"
    CACHE STRING
          "Flags used by the shared libraries linker during coverage builds."
          FORCE)

mark_as_advanced(
    CMAKE_CXX_FLAGS_COVERAGE CMAKE_C_FLAGS_COVERAGE
    CMAKE_EXE_LINKER_FLAGS_COVERAGE CMAKE_SHARED_LINKER_FLAGS_COVERAGE)

function(add_codecov _targetname _testrunner _outputname)
    set(coverage_info "${CMAKE_BINARY_DIR}/${_outputname}.info")
    set(coverage_cleaned "${coverage_info}.cleaned")

    add_custom_target(
        ${_targetname}
        COMMAND ${LCOV_EXE} --directory . --zerocounters
        # Serial run
        COMMAND ./${_testrunner}
        # Parallel run
        COMMAND ${MPIEXEC_EXECUTABLE} -np ${MPIEXEC_MAX_NUMPROCS}
                ./${_testrunner}
        COMMAND ${LCOV_EXE} --version
        COMMAND ${GCOV_EXE} --version
        COMMAND ${CMAKE_CXX_COMPILER} --version
        # Capturing lcov counters and generating report
        COMMAND ${LCOV_EXE} --directory . --base-directory . --capture
                --output-file ${coverage_info}
        COMMAND
            ${LCOV_EXE} --remove ${coverage_info} '/usr*' '*/test/*' '*/_deps/*'
            '*/ext/*' '/usr/include/*' '/usr/lib/*' '/Applications/*'
            '/examples/*' '*/include/*' -o ${coverage_info}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT
            "Resetting code coverage counters to zero.\nProcessing code coverage counters and generating report."
    )

    add_dependencies(${_targetname} ${_testrunner})

    add_custom_target(
        ${_targetname}_html
        COMMAND ${GENHTML_EXE} --prefix ${CMAKE_CURRENT_SOURCE_DIR} --ignore-errors
                source ${coverage_info} --output-directory ${_outputname}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_dependencies(${_targetname}_html ${_targetname})

    add_custom_target(
        ${_targetname}_upload
        COMMAND cp ../scripts/coverage/codecov.sh ./
        COMMAND chmod +x codecov.sh
        COMMAND ./codecov.sh
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_dependencies(${_targetname}_upload ${_targetname})

    add_custom_target(
        ${_targetname}_upload_only
        COMMAND cp ../scripts/coverage/codecov.sh ./
        COMMAND chmod +x codecov.sh
        COMMAND ./codecov.sh
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    message(
        STATUS
            "Created targets: ${_targetname}, ${_targetname}_html, and ${_targetname}_upload"
    )

endfunction()
