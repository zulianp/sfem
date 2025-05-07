#ifndef CU_SSQUAD4_INLINE_CUH
#define CU_SSQUAD4_INLINE_CUH

#include <cassert>

static __host__ __device__ int cu_ssquad4_lidx(const int L, const int x, const int y) {
    int Lp1 = L + 1;
    int ret = y * Lp1 + x;

    assert(ret < Lp1 * Lp1);
    assert(ret >= 0);
    return ret;
}

static __host__ __device__ int cu_ssquad4_txe(int level) { return level * level; }

static __host__ __device__ int cu_ssquad4_nxe(int level) {
    const int corners    = 4;
    const int edge_nodes = 4 * (level - 1);
    const int area_nodes = (level - 1) * (level - 1);
    return corners + edge_nodes + area_nodes;
}

#endif //CU_SSQUAD4_INLINE_CUH
