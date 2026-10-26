if(SFEM_ENABLE_PYTHON)
	nanobind_add_module(pysfem NOMINSIZE pysfem.cpp)
	target_link_libraries(pysfem PUBLIC sfem)
	# nanobind is compiled here from its own sources: SFEM_ENABLE_DEV_MODE's warnings are not for
	# it, and its headers are system headers in pysfem.cpp (as for SCCD).
	if(TARGET nanobind-static)
		target_compile_options(nanobind-static PRIVATE -w)
		set_target_properties(nanobind-static PROPERTIES SYSTEM ON)
	endif()
	# target_link_libraries(pysfem PUBLIC ${SFEM_DEP_LIBRARIES})

	# scan_directories(${CMAKE_CURRENT_SOURCE_DIR} "." 
	# 	SFEM_BUILD_INCLUDES
	# 	SFEM_HEADERS 
	# 	SFEM_SOURCES
	# 	)

	# set(SFEM_BUILD_INCLUDES
	# 	${SFEM_BUILD_INCLUDES}
	# 	PARENT_SCOPE)

	# set(SFEM_HEADERS
	# 	${SFEM_HEADERS}
	# 	PARENT_SCOPE)

	# set(SFEM_SOURCES
	# 	${SFEM_SOURCES}
	# 	PARENT_SCOPE)
endif()

