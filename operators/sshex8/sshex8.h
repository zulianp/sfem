#ifndef SFEM_SSHEX_8_H
#define SFEM_SSHEX_8_H

#include "sfem_base.h"
#include <assert.h>

static SFEM_INLINE int sshex8_nxe(int level) {
    const int corners = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes = (level - 1) * (level - 1) * (level - 1);
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

#endif  // SFEM_SSHEX_8_H
