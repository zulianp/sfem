#pragma once

// THE DEFAULT TYPE ALIASES THE KERNEL HEADERS EXPECT, FOR A TRANSLATION UNIT WITHOUT A FAMILY HEADER.
//
// src/kernels/ names scalar_t, idx_t, count_t and geom_t unqualified and relies on the including
// translation unit to supply them. On the host that is a family header -- cvfem_hex8_ns_core.hpp
// or cvfem_hex8_best_common.hpp -- which aliases them to the smesh types. A .cu cannot include
// either of those, and neither can a standalone test. Each declared its own, and they had
// drifted: cvfem_hex8_ns_cuda.cu
// declared `namespace smesh { using count_t = int32_t; }` from when the generated kernels spelled
// the type qualified, and after they stopped there was no bare count_t in any CUDA translation
// unit at all -- so the two BSR-slot parameters of the generated sympy kernels did not compile.
// Nothing caught it because CUDA is not built on the development machine.
//
// This is that one place -- in support/ rather than in kernels/, because the kernels must NOT
// fix their own types: a caller choosing a different precision is exactly what the
// unqualified spelling exists to allow. A translation unit that wants the defaults includes
// this; one that wants something else declares its own, as before.
// One header, so there is one place the contract is written, and static_assert so it cannot
// drift from smesh silently: the sizes and signedness are checked against <cstdint> here, and
// against the smesh types themselves wherever a host TU includes both.

#include <cstddef>
#include "kernels/cvfem_portability.hpp"
#include <cstdint>
#include <type_traits>

using scalar_t = double;
using idx_t    = int32_t;
using count_t  = int32_t;
using geom_t   = float;

// Measured against the installed smesh_types.hpp: idx_t and count_t are 4-byte signed, geom_t is
// 4 bytes. A mismatch here is not a compile error at the call site -- the kernels would silently
// index with the wrong width -- so it is asserted rather than trusted.
static_assert(sizeof(idx_t) == 4 && std::is_signed<idx_t>::value, "idx_t must be 4-byte signed");
static_assert(sizeof(count_t) == 4 && std::is_signed<count_t>::value, "count_t must be 4-byte signed");
static_assert(sizeof(geom_t) == 4, "geom_t must be 4 bytes");
static_assert(sizeof(scalar_t) == 8, "scalar_t must be double");

// The qualified spelling, for the device code that still uses it.
namespace smesh {
using count_t = ::count_t;
}

