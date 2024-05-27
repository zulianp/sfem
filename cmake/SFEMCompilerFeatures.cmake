
macro (set_compiler_features target_)
target_compile_features(${target_} PUBLIC cxx_std_14)
endmacro(set_compiler_features)
