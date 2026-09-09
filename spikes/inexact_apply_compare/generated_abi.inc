// How these benchmarks reach a generated kernel's C entry point.
//
// A generated entry point no longer publishes one symbol per precision.  It
// takes the width, in bytes, of the scalar its buffers hold, and instantiates
// the template for that scalar itself:
//
//     extern "C" int <material>_<element>_apply_a_msoa(
//         const int scalar_bytes, ..., const void *const RSTR hx, ...);
//
// `smesh::PrimitiveType` numbers its scalars by the same widths, so the
// dispatch layer forwards the enum unchanged; a caller that already knows its
// scalar, as these benchmarks do, passes `sizeof` of it.
//
// The buffers cross as `void *`.  These declarations keep the concrete pointer
// types instead, which is ABI-identical and lets the call sites stay readable;
// what matters is that the leading width is declared, because omitting it
// links -- C linkage does not mangle the parameters -- and then calls with
// every argument in the wrong register.
#ifndef SFEM_SPIKE_GENERATED_ABI_INC
#define SFEM_SPIKE_GENERATED_ABI_INC

//: The width these benchmarks pass: they measure the double-precision path.
static constexpr int SFEM_CODEGEN_F64 = (int)sizeof(double);

#endif
