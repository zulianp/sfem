
macro (set_compiler_features target_)
target_compile_features(${target_} PUBLIC cxx_std_11)
endmacro(set_compiler_features)