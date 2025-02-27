#ifndef SFEM_SSHEX_8_H
#define SFEM_SSHEX_8_H

#include <assert.h>
#include "sfem_base.h"

static SFEM_INLINE int sshex8_nxe(int level) {
    const int corners    = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes  = (level - 1) * (level - 1) * (level - 1);
    return corners + edge_nodes + face_nodes + vol_nodes;
}

static SFEM_INLINE int sshex8_txe(int level) { return level * level * level; }

static SFEM_INLINE int sshex8_lidx(const int L, const int x, const int y, const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < sshex8_nxe(L));
    assert(ret >= 0);
    return ret;
}

static void sshex8_apply_element_matrix(const int                           level,
                                        const scalar_t *const SFEM_RESTRICT element_matrix,
                                        const scalar_t *const SFEM_RESTRICT input,
                                        scalar_t *const SFEM_RESTRICT       output) {
    scalar_t element_u[8];
    scalar_t element_vector[8];
    for (int zi = 0; zi < level; zi++) {
        for (int yi = 0; yi < level; yi++) {
            for (int xi = 0; xi < level; xi++) {
                int lev[8] = {// Bottom
                              sshex8_lidx(level, xi, yi, zi),
                              sshex8_lidx(level, xi + 1, yi, zi),
                              sshex8_lidx(level, xi + 1, yi + 1, zi),
                              sshex8_lidx(level, xi, yi + 1, zi),
                              // Top
                              sshex8_lidx(level, xi, yi, zi + 1),
                              sshex8_lidx(level, xi + 1, yi, zi + 1),
                              sshex8_lidx(level, xi + 1, yi + 1, zi + 1),
                              sshex8_lidx(level, xi, yi + 1, zi + 1)};

                for (int d = 0; d < 8; d++) {
                    element_u[d] = input[lev[d]];
                }

                for (int i = 0; i < 8; i++) {
                    element_vector[i] = 0;
                }

                for (int i = 0; i < 8; i++) {
                    const scalar_t *const row = &element_matrix[i * 8];
                    const scalar_t        ui  = element_u[i];
                    for (int j = 0; j < 8; j++) {
                        element_vector[j] += ui * row[j];
                    }
                }

                for (int d = 0; d < 8; d++) {
                    output[lev[d]] += element_vector[d];
                }
            }
        }
    }
}

#endif  // SFEM_SSHEX_8_H
