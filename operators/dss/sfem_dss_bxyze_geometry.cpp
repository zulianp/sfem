#include "sfem_dss_bxyze_geometry.hpp"

#include "hex8_inline_cpu.hpp"
#include "sshex8.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

ptrdiff_t sfem_dss_bxyze_affine_geometry_n_values(const sfem_dss_bxyze_layout_t *const layout) {
    if (!layout) {
        return -1;
    }

    sfem_dss_bxyze_layout_t expected;
    if (sfem_dss_bxyze_create_layout(layout->level,
                                    layout->nelements,
                                    layout->element_block_size,
                                    layout->execution_block_size,
                                    &expected) != SFEM_SUCCESS ||
        layout->n != expected.n || layout->block_stride != expected.block_stride ||
        layout->execution_stride != expected.execution_stride || layout->nblocks != expected.nblocks ||
        layout->chunks_per_block != expected.chunks_per_block || layout->ntasks != expected.ntasks) {
        return -1;
    }

    // Corner lookup uses the int-valued sshex8 local indices.
    if (layout->block_stride / layout->element_block_size > std::numeric_limits<int>::max()) {
        return -1;
    }

    return layout->nblocks * ((ptrdiff_t)6 * layout->element_block_size);
}

int sfem_dss_bxyze_affine_geometry(const sfem_dss_bxyze_layout_t *const layout,
                                  const idx_t *const *const           elements,
                                  const geom_t *const *const          points,
                                  scalar_t *const SFEM_RESTRICT       geometry) {
    if (sfem_dss_bxyze_affine_geometry_n_values(layout) < 0) {
        return SFEM_FAILURE;
    }
    if (layout->nelements == 0) {
        return SFEM_SUCCESS;
    }
    if (!elements || !points || !points[0] || !points[1] || !points[2] || !geometry) {
        return SFEM_FAILURE;
    }

    const int level = layout->level;
    const int corners[8] = {sshex8_lidx(level, 0, 0, 0),
                            sshex8_lidx(level, level, 0, 0),
                            sshex8_lidx(level, level, level, 0),
                            sshex8_lidx(level, 0, level, 0),
                            sshex8_lidx(level, 0, 0, level),
                            sshex8_lidx(level, level, 0, level),
                            sshex8_lidx(level, level, level, level),
                            sshex8_lidx(level, 0, level, level)};
    for (int v = 0; v < 8; ++v) {
        if (!elements[corners[v]]) {
            return SFEM_FAILURE;
        }
    }

    const int       B            = layout->element_block_size;
    const ptrdiff_t ne           = layout->nelements;
    const ptrdiff_t nblocks      = layout->nblocks;
    const ptrdiff_t block_stride = (ptrdiff_t)6 * B;
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];
    int invalid_geometry = 0;

#pragma omp parallel for schedule(static) reduction(| : invalid_geometry)
    for (ptrdiff_t eb = 0; eb < nblocks; ++eb) {
        const ptrdiff_t e_begin = eb * B;
        const int count = (int)std::min<ptrdiff_t>(B, ne - e_begin);
        scalar_t *const SFEM_RESTRICT dst = &geometry[eb * block_stride];

#pragma omp simd reduction(| : invalid_geometry)
        for (int b = 0; b < count; ++b) {
            const ptrdiff_t e = e_begin + b;
            scalar_t lx[8], ly[8], lz[8], fff[6];
            for (int v = 0; v < 8; ++v) {
                const idx_t gid = elements[corners[v]][e];
                lx[v] = (scalar_t)x[gid];
                ly[v] = (scalar_t)y[gid];
                lz[v] = (scalar_t)z[gid];
            }

            hex8_fff(lx, ly, lz, (scalar_t)0.5, (scalar_t)0.5, (scalar_t)0.5, fff);
            invalid_geometry |= !(fff[0] > 0 && fff[3] > 0 && fff[5] > 0);
            for (int c = 0; c < 6; ++c) {
                invalid_geometry |= !std::isfinite(fff[c]);
                dst[(ptrdiff_t)c * B + b] = fff[c];
            }
        }

        for (int c = 0; c < 6; ++c) {
#pragma omp simd
            for (int b = count; b < B; ++b) {
                dst[(ptrdiff_t)c * B + b] = 0;
            }
        }
    }

    return invalid_geometry ? SFEM_FAILURE : SFEM_SUCCESS;
}
