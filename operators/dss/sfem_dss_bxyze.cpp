#include "sfem_dss_bxyze.hpp"

#include <limits>

namespace {

bool valid_shape(const int level, const ptrdiff_t nelements, const int element_block_size) {
    return level >= 1 && nelements >= 0 && element_block_size > 0;
}

ptrdiff_t block_offset(const int       n,
                       const int       element_block_size,
                       const ptrdiff_t element_block,
                       const int       ix,
                       const int       iy,
                       const int       iz) {
    return (ptrdiff_t)element_block_size *
           ((ptrdiff_t)ix + (ptrdiff_t)n * ((ptrdiff_t)iy + (ptrdiff_t)n * ((ptrdiff_t)iz + (ptrdiff_t)n * element_block)));
}

template <typename Scalar>
int global_to_bxyze(const int                  level,
                    const ptrdiff_t            nelements,
                    const int                  element_block_size,
                    const idx_t *const *const  elements,
                    const Scalar *const        global,
                    Scalar *const              bxyze) {
    if (!valid_shape(level, nelements, element_block_size)) {
        return SFEM_FAILURE;
    }

    if (nelements == 0) {
        return SFEM_SUCCESS;
    }

    if (!elements || !global || !bxyze) {
        return SFEM_FAILURE;
    }

    const int       n       = level + 1;
    const int       n2      = n * n;
    const int       bsize   = element_block_size;
    const ptrdiff_t nblocks = (nelements + bsize - 1) / bsize;

    _Pragma("omp parallel for collapse(4)")
    for (ptrdiff_t eb = 0; eb < nblocks; ++eb) {
        for (int iz = 0; iz < n; ++iz) {
            for (int iy = 0; iy < n; ++iy) {
                for (int ix = 0; ix < n; ++ix) {
                    const int          lidx    = iz * n2 + iy * n + ix;
                    const idx_t *const row     = elements[lidx];
                    const ptrdiff_t    e_begin = eb * bsize;
                    const ptrdiff_t    rem     = nelements - e_begin;
                    const int          b_count = rem < bsize ? (int)rem : bsize;
                    Scalar *const      dst     = &bxyze[block_offset(n, bsize, eb, ix, iy, iz)];

                    for (int b = 0; b < b_count; ++b) {
                        const idx_t gid = row[e_begin + b];
                        dst[b]          = global[gid];
                    }

                    for (int b = b_count; b < bsize; ++b) {
                        dst[b] = static_cast<Scalar>(0);
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

}  // namespace

int sfem_dss_bxyze_create_layout(const int                      level,
                                const ptrdiff_t                nelements,
                                const int                      element_block_size,
                                const int                      execution_block_size,
                                sfem_dss_bxyze_layout_t *const layout) {
    if (!layout || level < 1 || level == std::numeric_limits<int>::max() || nelements < 0 ||
        element_block_size <= 0 || execution_block_size <= 0 || execution_block_size > element_block_size) {
        return SFEM_FAILURE;
    }

    const ptrdiff_t max_size = std::numeric_limits<ptrdiff_t>::max();
    const ptrdiff_t n        = (ptrdiff_t)level + 1;
    if (n > max_size / n || n * n > max_size / n) {
        return SFEM_FAILURE;
    }

    const ptrdiff_t n3 = n * n * n;
    if (n3 > max_size / element_block_size) {
        return SFEM_FAILURE;
    }

    const ptrdiff_t block_stride = (ptrdiff_t)element_block_size * n3;
    const ptrdiff_t nblocks = nelements / element_block_size + (nelements % element_block_size != 0);
    const int chunks = element_block_size / execution_block_size + (element_block_size % execution_block_size != 0);
    if (nblocks > max_size / block_stride || nblocks > max_size / chunks) {
        return SFEM_FAILURE;
    }

    layout->level                = level;
    layout->n                    = (int)n;
    layout->nelements            = nelements;
    layout->element_block_size   = element_block_size;
    layout->execution_block_size = execution_block_size;
    layout->block_stride         = block_stride;
    layout->execution_stride     = (ptrdiff_t)execution_block_size * n3;
    layout->nblocks              = nblocks;
    layout->chunks_per_block     = chunks;
    layout->ntasks               = nblocks * (ptrdiff_t)chunks;
    return SFEM_SUCCESS;
}

ptrdiff_t sfem_dss_bxyze_n_values(const int level, const ptrdiff_t nelements, const int element_block_size) {
    if (!valid_shape(level, nelements, element_block_size)) {
        return -1;
    }

    const ptrdiff_t n      = (ptrdiff_t)level + 1;
    const ptrdiff_t nxe    = n * n * n;
    const ptrdiff_t nblock = (nelements + element_block_size - 1) / element_block_size;

    return nblock * nxe * element_block_size;
}

ptrdiff_t sfem_dss_bxyze_index(const int       level,
                               const int       element_block_size,
                               const ptrdiff_t element,
                               const int       ix,
                               const int       iy,
                               const int       iz) {
    if (level < 1 || element_block_size <= 0 || element < 0 || ix < 0 || iy < 0 || iz < 0) {
        return -1;
    }

    const int n = level + 1;

    if (ix >= n || iy >= n || iz >= n) {
        return -1;
    }

    const ptrdiff_t eb = element / element_block_size;
    const int       b  = (int)(element % element_block_size);

    return (ptrdiff_t)b + block_offset(n, element_block_size, eb, ix, iy, iz);
}

int sfem_dss_global_to_bxyze(const int                  level,
                             const ptrdiff_t            nelements,
                             const int                  element_block_size,
                             const idx_t *const *const  elements,
                             const real_t *const        global,
                             real_t *const              bxyze) {
    return global_to_bxyze<real_t>(level, nelements, element_block_size, elements, global, bxyze);
}

int sfem_dss_global_to_bxyze_real32(const int                  level,
                                    const ptrdiff_t            nelements,
                                    const int                  element_block_size,
                                    const idx_t *const *const  elements,
                                    const float *const         global,
                                    float *const               bxyze) {
    return global_to_bxyze<float>(level, nelements, element_block_size, elements, global, bxyze);
}

int sfem_dss_global_to_bxyze_real64(const int                  level,
                                    const ptrdiff_t            nelements,
                                    const int                  element_block_size,
                                    const idx_t *const *const  elements,
                                    const double *const        global,
                                    double *const              bxyze) {
    return global_to_bxyze<double>(level, nelements, element_block_size, elements, global, bxyze);
}
