macro (set_compiler_features target_)
# C++17 everywhere.  The generated matrix-format scatter dispatches on a
# compile-time `FORMAT` template parameter with `if constexpr`, so the emitted
# kernels require it; declaring C++14 only worked because compilers default
# higher than the minimum asked for.
target_compile_features(${target_} PUBLIC cxx_std_17)
endmacro(set_compiler_features)
