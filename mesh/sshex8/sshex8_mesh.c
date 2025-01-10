#include "sshex8_mesh.h"
#include "sshex8.h"

static SFEM_INLINE void hex8_eval_f(const scalar_t x, const scalar_t y, const scalar_t z, scalar_t *const f) {
    const scalar_t xm = (1 - x);
    const scalar_t ym = (1 - y);
    const scalar_t zm = (1 - z);

    f[0] = xm * ym * zm;  // (0, 0, 0)
    f[1] = x * ym * zm;   // (1, 0, 0)
    f[2] = x * y * zm;    // (1, 1, 0)
    f[3] = xm * y * zm;   // (0, 1, 0)
    f[4] = xm * ym * z;   // (0, 0, 1)
    f[5] = x * ym * z;    // (1, 0, 1)
    f[6] = x * y * z;     // (1, 1, 1)
    f[7] = xm * y * z;    // (0, 1, 1)
}

int sshex8_fill_points(const int       level,
                       const ptrdiff_t nelements,
                       idx_t **const   elements,
                       geom_t **const  macro_mesh_points,
                       geom_t **const  points) {
    const int proteus_to_std_hex8_corners[8] = {// Bottom
                                                sshex8_lidx(level, 0, 0, 0),
                                                sshex8_lidx(level, level, 0, 0),
                                                sshex8_lidx(level, level, level, 0),
                                                sshex8_lidx(level, 0, level, 0),

                                                // Top
                                                sshex8_lidx(level, 0, 0, level),
                                                sshex8_lidx(level, level, 0, level),
                                                sshex8_lidx(level, level, level, level),
                                                sshex8_lidx(level, 0, level, level)};

    // Nodes
    const scalar_t h = 1. / level;
    scalar_t       f[8];

#pragma omp parallel for collapse(4)
    for (int zi = 0; zi < level + 1; zi++) {
        for (int yi = 0; yi < level + 1; yi++) {
            for (int xi = 0; xi < level + 1; xi++) {
                hex8_eval_f(xi * h, yi * h, zi * h, f);
                int lidx = sshex8_lidx(level, xi, yi, zi);

                for (int d = 0; d < 3; d++) {

                    for (ptrdiff_t e = 0; e < nelements; e++) {
                        scalar_t acc = 0;

                        for (int lnode = 0; lnode < 8; lnode++) {
                            const int corner_idx = proteus_to_std_hex8_corners[lnode];
                            geom_t    p          = macro_mesh_points[d][elements[corner_idx][e]];
                            acc += p * f[lnode];
                        }

                        points[d][elements[lidx][e]] = acc;
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
