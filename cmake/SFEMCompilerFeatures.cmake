
macro (set_compiler_features target_)
if(WIN32)
    target_compile_features(${target_} PUBLIC cxx_std_17)
else()
    target_compile_features(${target_} PUBLIC cxx_std_14)
endif()
endmacro(set_compiler_features)
