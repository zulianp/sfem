#ifndef CVFEM_SCATTER_PROBE_HPP
#define CVFEM_SCATTER_PROBE_HPP

#include <cstddef>

// Set before entering OpenMP; both timing arms execute the same compiled sweep.
static volatile bool g_scatter_accumulate = true;
static double *g_scatter_capture = nullptr;

template <typename Pack>
static inline bool cvfem_scatter_probe(const Pack &out, const ptrdiff_t begin, const int nlanes) {
    // Make every result observable even when its nodal addition is bypassed.
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "m"(out) : "memory");
#else
#error "The scatter study requires a GNU-compatible compiler escape"
#endif
    if (g_scatter_capture) {
        for (int lane = 0; lane < nlanes; ++lane) {
            double *dest = g_scatter_capture + (begin + lane) * 32;
            for (int a = 0; a < 8; ++a) {
                dest[a * 4 + 0] = out.rx[a][lane];
                dest[a * 4 + 1] = out.ry[a][lane];
                dest[a * 4 + 2] = out.rz[a][lane];
                dest[a * 4 + 3] = out.rc[a][lane];
            }
        }
    }
    return g_scatter_accumulate;
}

#endif
