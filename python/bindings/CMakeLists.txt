if(SFEM_ENABLE_PYTHON)
	nanobind_add_module(pysfem NOMINSIZE pysfem.cpp)
	target_link_libraries(pysfem PUBLIC sfem)
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

