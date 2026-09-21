#pragma once
// The generated `op/*_c_abi.hpp` declares the whole material ABI, including the
// inexact-apply and partial-assembly entry points, whose scalar aliases live in
// SFEM's own base header rather than in the generated tree.  This spike links
// none of those, but the header still has to parse, so the aliases are supplied
// here with the widths `tools/roofline.py` records.  The mesh-level aliases
// match the generated header's own fallback block exactly, because supplying
// this file at all suppresses it.
#include <cstddef>
#include <cstdint>

typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double real_t;
typedef double geom_t;

typedef uint16_t half_t;
typedef half_t compressed_t;
typedef float scaling_t;
typedef float metric_tensor_t;
