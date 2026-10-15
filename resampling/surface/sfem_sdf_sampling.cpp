#include "sfem_sdf_sampling.hpp"

#include <math.h>
#include <stdlib.h>

#include <algorithm>

namespace {
    // The pole of the cubic B-spline's inverse filter, and the filter's gain.
    const double pole = -0.267949192431122706472553658494;  // sqrt(3) - 2
    const double gain = 6;

    // Samples each line is extended by on either side before it is filtered: the filter's
    // response decays as |pole|^k, below 1e-7 at k = 12, so what the extension's own ends do
    // does not reach the grid.
    const ptrdiff_t extension = 12;

    // The causal filter's first value, truncated where |pole|^k is below double precision.
    double initial_causal(const double* const c, const ptrdiff_t n) {
        const ptrdiff_t horizon = std::min(n, (ptrdiff_t)ceil(log(1e-16) / log(fabs(pole))));
        double          sum     = c[0];
        double          zk      = pole;
        for (ptrdiff_t k = 1; k < horizon; ++k) {
            sum += zk * c[k];
            zk *= pole;
        }
        return sum;
    }

    // Samples to coefficients along one line of n values, in place.
    void filter_line(double* const c, const ptrdiff_t n) {
        for (ptrdiff_t k = 0; k < n; ++k) {
            c[k] *= gain;
        }
        c[0] = initial_causal(c, n);
        for (ptrdiff_t k = 1; k < n; ++k) {
            c[k] += pole * c[k - 1];
        }
        c[n - 1] = (pole / (pole * pole - 1)) * (pole * c[n - 2] + c[n - 1]);
        for (ptrdiff_t k = n - 2; k >= 0; --k) {
            c[k] = pole * (c[k + 1] - c[k]);
        }
    }

    // The line's samples s[0 .. n - 1] in line[extension ..], extended on both sides by point
    // reflection through the end sample (s[-k] = 2 s[0] - s[k]), linearly beyond the line's
    // length: a linear field stays linear, so planes are reproduced exactly.
    void extend_line(double* const line, const ptrdiff_t n) {
        double* const s = line + extension;
        for (ptrdiff_t k = 1; k <= extension; ++k) {
            s[-k] = k < n ? 2 * s[0] - s[k] : s[0] - k * (s[1] - s[0]);
            s[n - 1 + k] = k < n ? 2 * s[n - 1] - s[n - 1 - k] : s[n - 1] + k * (s[n - 1] - s[n - 2]);
        }
    }

    // Filters every line along `axis` of `src` into `dst` (they may be the same array), writing
    // the coefficients of the n nodes and of the pad nodes -1 and n. The lines run over
    // [lo, hi) on the other two axes.
    void filter_axis(const int                axis,
                     const ptrdiff_t* const   n,
                     const ptrdiff_t* const   lo,
                     const ptrdiff_t* const   hi,
                     const ptrdiff_t* const   src_stride,
                     const geom_t* const      src,
                     const ptrdiff_t* const   dst_stride,
                     geom_t* const            dst) {
        const int       a1 = (axis + 1) % 3;
        const int       a2 = (axis + 2) % 3;
        const ptrdiff_t nl = n[axis];
        const ptrdiff_t ne = nl + 2 * extension;

#pragma omp parallel
        {
            double* const line = (double*)malloc((size_t)ne * sizeof(double));
#pragma omp for collapse(2)
            for (ptrdiff_t j = lo[a2]; j < hi[a2]; ++j) {
                for (ptrdiff_t i = lo[a1]; i < hi[a1]; ++i) {
                    const ptrdiff_t s0 = i * src_stride[a1] + j * src_stride[a2];
                    const ptrdiff_t d0 = i * dst_stride[a1] + j * dst_stride[a2];
                    for (ptrdiff_t k = 0; k < nl; ++k) {
                        line[extension + k] = (double)src[s0 + k * src_stride[axis]];
                    }
                    extend_line(line, nl);
                    filter_line(line, ne);
                    for (ptrdiff_t k = -1; k <= nl; ++k) {
                        dst[d0 + k * dst_stride[axis]] = (geom_t)line[extension + k];
                    }
                }
            }
            free(line);
        }
    }
}  // namespace

int sdf_bspline_coefficients(const ptrdiff_t* const SFEM_RESTRICT n,
                             const ptrdiff_t* const SFEM_RESTRICT sample_stride,
                             const geom_t* const SFEM_RESTRICT    samples,
                             const ptrdiff_t* const SFEM_RESTRICT coeff_stride,
                             geom_t* const SFEM_RESTRICT          coeffs) {
    for (int d = 0; d < 3; ++d) {
        if (n[d] < 2) {
            return SFEM_FAILURE;
        }
    }

    // Axis by axis; each pass runs over the pads the previous ones wrote, so the pad's edges
    // and corners are the tensor product's too.
    ptrdiff_t lo[3] = {0, 0, 0}, hi[3] = {n[0], n[1], n[2]};
    filter_axis(0, n, lo, hi, sample_stride, samples, coeff_stride, coeffs);
    lo[0] = -1;
    hi[0] = n[0] + 1;
    filter_axis(1, n, lo, hi, coeff_stride, coeffs, coeff_stride, coeffs);
    lo[1] = -1;
    hi[1] = n[1] + 1;
    filter_axis(2, n, lo, hi, coeff_stride, coeffs, coeff_stride, coeffs);

    return SFEM_SUCCESS;
}
