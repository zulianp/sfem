#pragma once

// The LAYOUT-INDEPENDENT device kernels: the nodal pressure gradient's two passes and the
// block utilities. None of them touches a mesh layout -- they walk nodes or blocks.
//
// The device kernels, in this layout's own cuda/ subfolder: CUDA sources belong beside the host
// sources they mirror, and DESIGN.md asks each mesh format for one. They were all in
// cuda/cvfem_hex8_ns_cuda.cu, a single 2646-line translation unit holding every layout's kernels
// and every layout's launchers together.
//
// What stayed behind is the launchers -- the `launch_*` host functions that pick a block size, an
// instantiation and a stream. That is the same split as the host side: the kernel computes, the
// launcher decides.
//
// Included from inside cvfem_hex8_ns_cuda.cu's anonymous namespace, so these keep the internal
// linkage they had. One translation unit includes them; a __global__ in an anonymous namespace
// reached from two would be two different kernels with one name.



__global__ void cvfem_hex8_nodal_p_grad_accumulate_kernel(
        const ptrdiff_t nelements,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ pgx, double *const __restrict__ pgy,
        double *const __restrict__ pgz, double *const __restrict__ w) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        const double vol = fabs(det[e]);
        if (vol < 1e-30) continue;
        int32_t ev[CVFEM_HEX8_N_NODES];
        double  pe[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            ev[a] = g;
            pe[a] = u[(ptrdiff_t)g * CVFEM_CUDA_NF + 3];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        double gx, gy, gz;
        cvfem_hex8_grad_scalar(adj_e, det[e], pe, gx, gy, gz);
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            atomicAdd(&pgx[ev[a]], vol * gx);
            atomicAdd(&pgy[ev[a]], vol * gy);
            atomicAdd(&pgz[ev[a]], vol * gz);
            atomicAdd(&w[ev[a]], vol);
        }
    }
}

__global__ void cvfem_hex8_nodal_p_grad_normalize_kernel(
        const ptrdiff_t nnodes, double *const __restrict__ pgx,
        double *const __restrict__ pgy, double *const __restrict__ pgz,
        const double *const __restrict__ w) {
    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < nnodes;
         i += (ptrdiff_t)blockDim.x * gridDim.x) {
        const double wi = w[i];
        if (wi > 0.0) { pgx[i] /= wi; pgy[i] /= wi; pgz[i] /= wi; }
    }
}

// Copy back only the blocks the nonlinear half will overwrite, from a COMPACT side
// buffer holding just those blocks. Reads are contiguous, writes are block-scattered.
__global__ void cvfem_hex8_restore_blocks_kernel(
        const ptrdiff_t n_blocks,
        const int32_t *const __restrict__ block_ids,
        const double  *const __restrict__ compact,
        double *const __restrict__ dst) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_blocks * 4;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t b = t >> 2, q = t & 3;
        reinterpret_cast<double4 *>(&dst[(ptrdiff_t)block_ids[b] * 16])[q] =
                reinterpret_cast<const double4 *>(&compact[b * 16])[q];
    }
}

__global__ void cvfem_hex8_zero_blocks_kernel(
        const ptrdiff_t n_blocks,
        const int32_t  *const __restrict__ block_ids,
        const uint16_t *const __restrict__ masks,
        double *const __restrict__ values) {
    // One 32-byte store per thread instead of four 8-byte ones. A block's 16 doubles are
    // 128 contiguous bytes starting at a 32-byte boundary, so four consecutive threads
    // cover a block exactly. Measured earlier: vectorising this shape is worth ~1.2x.
    // One thread per entry, but only the entries that will be rewritten. Zeroing whole
    // blocks would clear viscous values that nothing recomputes.
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_blocks * 16;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t b  = t >> 4;
        const int       k  = (int)(t & 15);
        const ptrdiff_t id = (ptrdiff_t)block_ids[b];
        if (masks[id] & (uint16_t)(1u << k)) values[id * 16 + k] = 0.0;
    }
}

// Restore the constant viscous diagonal, then the velocity-dependent part goes on top.
__global__ void cvfem_hex8_diag_restore_kernel(const ptrdiff_t n, const double *const __restrict__ src,
                                               double *const __restrict__ dst) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n;
         t += (ptrdiff_t)blockDim.x * gridDim.x)
        dst[t] = src[t];
}

// The preconditioner block per node: 3x3 velocity inverse plus a scalar pressure
// reciprocal, matching build_block_jacobi in the solver. A plain 4x4 inverse would be
// wrong here -- the block is singular, because the pressure-pressure entry is zero.
__global__ void cvfem_hex8_invert_diag_kernel(const ptrdiff_t nnodes,
                                              const unsigned char *const __restrict__ constrained,
                                              double *const __restrict__ diag) {
    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < nnodes;
         i += (ptrdiff_t)blockDim.x * gridDim.x) {
        double blk[16], inv[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) blk[k] = diag[i * 16 + k];
        cvfem_hex8_block_jacobi_block(blk, constrained ? &constrained[i * 4] : nullptr, inv);
#pragma unroll
        for (int k = 0; k < 16; ++k) diag[i * 16 + k] = inv[k];
    }
}
