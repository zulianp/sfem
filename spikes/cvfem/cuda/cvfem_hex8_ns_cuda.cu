// Packed CVFEM HEX8 Navier-Stokes residual on CUDA.
//
// One CUDA block per pack, following bench/cuda/bench_packed_laplacian.cu. The pack's
// nodes are staged into dynamic shared memory, elements are processed strided by
// threadIdx.x, contributions are accumulated in shared memory, and the result is flushed
// once per node. The format contract this relies on -- in particular that a pack's owned
// ids are ordered non-shared before shared -- is written up in PACKED_FORMAT.md.
//
// The element kernel called per thread is the *scalar* cvfem_hex8_ns_upwind_residual.
// The host _simd family is not used and not device-callable: its Hex8*Pack structs exist
// to feed 512-bit lanes, and on a GPU the lane dimension is threadIdx.x already.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

using scalar_t = double;
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "cvfem_hex8_ns_upwind_kernels.hpp"

namespace smesh { using count_t = int32_t; }
#include "cvfem_hex8_ns_upwind_sympy_kernels.hpp"
#include "cvfem_hex8_boundary_scs.hpp"

#include "cvfem_hex8_ns_cuda.hpp"

#define CVFEM_CUDA_CHECK(expr)                                                       \
    do {                                                                             \
        cudaError_t _e = (expr);                                                     \
        if (_e != cudaSuccess) {                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(_e));                                    \
            return 1;                                                                \
        }                                                                            \
    } while (0)

static constexpr int CVFEM_CUDA_NF = CVFEM_HEX8_N_FIELDS;  // 4

struct cvfem_cuda_ctx {
    ptrdiff_t nnodes{0}, nelements{0};
    ptrdiff_t n_packs{0}, n_elements_per_pack{0}, max_pack_nodes{0};
    ptrdiff_t n_ghost_entries{0}, n_ghost_reduce_rows{0};

    uint16_t  *elems{nullptr};              // [8 * nelements], v * nelements + e
    ptrdiff_t *owned_nodes_ptr{nullptr};    // [n_packs + 1]
    ptrdiff_t *n_shared{nullptr};           // [n_packs]
    ptrdiff_t *ghost_ptr{nullptr};          // [n_packs + 1]
    int32_t   *ghost_idx{nullptr};          // [n_ghost_entries]
    ptrdiff_t *ghost_reduce_ptr{nullptr};
    ptrdiff_t *ghost_reduce_idx{nullptr};
    int32_t   *ghost_reduce_dest{nullptr};

    double *adj{nullptr};                   // [9 * nelements], c * nelements + e
    double *det{nullptr};                   // [nelements]
    double *u{nullptr};                     // [4 * nnodes] interleaved
    double *r{nullptr};                     // [4 * nnodes] interleaved
    double *ghost_buf{nullptr};             // [4 * n_ghost_entries] interleaved
    double *v{nullptr};                     // [4 * nnodes] interleaved (J*v direction)
    size_t  jv_shmem_bytes{0};
    bool    jv_optin_done{false};

    // assembled BSR
    ptrdiff_t nnz{0};
    int32_t  *elements_global{nullptr};   // [8 * nelements], GLOBAL ids
    int32_t  *element_slots{nullptr};     // [64 * nelements]
    double   *values{nullptr};            // [16 * nnz]
    int32_t  *rowptr{nullptr};            // [nnodes + 1], block rows
    int32_t  *colidx{nullptr};            // [nnz]
    cusparseHandle_t      sp{nullptr};
    cusparseMatDescr_t    spdesc{nullptr};

    ptrdiff_t  n_boundary{0};
    int32_t   *boundary_elems{nullptr};
    double    *px{nullptr}, *py{nullptr}, *pz{nullptr};
    double    *pgx{nullptr}, *pgy{nullptr}, *pgz{nullptr}, *pgw{nullptr};
    double     Lx{0}, Ly{0}, Lz{0};

    int        n_colors{0};
    ptrdiff_t *pack_order{nullptr};
    ptrdiff_t *color_ptr{nullptr};
    std::vector<ptrdiff_t> h_color_ptr;   // host copy: the launch loop needs the bounds

    size_t shmem_bytes{0};
    bool   shmem_optin_done{false};
};

namespace {

template <typename T>
int device_dup(T **dst, const T *src, size_t n) {
    if (n == 0) { *dst = nullptr; return 0; }
    CVFEM_CUDA_CHECK(cudaMalloc(dst, n * sizeof(T)));
    CVFEM_CUDA_CHECK(cudaMemcpy(*dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
    return 0;
}

// ---------------------------------------------------------------- residual kernel

template <int FLUSH, bool WITH_RC>
__global__ void cvfem_hex8_residual_pack_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const double rho, const double mu,
        const uint16_t  *const __restrict__ elems,
        const ptrdiff_t *const __restrict__ owned_nodes_ptr,
        const ptrdiff_t *const __restrict__ n_shared,
        const ptrdiff_t *const __restrict__ ghost_ptr,
        const int32_t   *const __restrict__ ghost_idx,
        const double    *const __restrict__ adj,
        const double    *const __restrict__ det,
        const double    *const __restrict__ u,
        double *const __restrict__ r,
        double *const __restrict__ ghost_buf,
        const double    *const __restrict__ px,
        const double    *const __restrict__ py,
        const double    *const __restrict__ pz,
        const double    *const __restrict__ pgx,
        const double    *const __restrict__ pgy,
        const double    *const __restrict__ pgz,
        const double rc_scale) {
    extern __shared__ double smem[];

    const ptrdiff_t p            = blockIdx.x;
    const ptrdiff_t owned        = owned_nodes_ptr[p];
    const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned;
    const ptrdiff_t gbegin       = ghost_ptr[p];
    const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - gbegin;
    const ptrdiff_t total_nodes  = n_contiguous + n_ghost;

    double *const s_u   = smem;
    double *const s_out = smem + (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;
    // Rhie-Chow needs the coordinates and the nodal pressure gradient of every pack node.
    double *const s_rc  = smem + 2 * (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;

    // Stage the pack's fields, and zero the accumulator. Owned ids map to a contiguous
    // global window; ghosts resolve through ghost_idx (PACKED_FORMAT.md section 2).
    for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
        const ptrdiff_t g = (i < n_contiguous) ? (owned + i)
                                               : (ptrdiff_t)ghost_idx[gbegin + i - n_contiguous];
        const double *const src = &u[g * CVFEM_CUDA_NF];
        double *const       dst = &s_u[i * CVFEM_CUDA_NF];
        double *const       acc = &s_out[i * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) { dst[f] = src[f]; acc[f] = 0.0; }
        if (WITH_RC) {
            double *const q = &s_rc[i * 6];
            q[0] = px[g]; q[1] = py[g]; q[2] = pz[g];
            q[3] = pgx[g]; q[4] = pgy[g]; q[5] = pgz[g];
        }
    }
    __syncthreads();

    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        uint16_t ev[CVFEM_HEX8_N_NODES];
        double   ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double   uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const uint16_t l = elems[(ptrdiff_t)a * nelements + e];
            ev[a]                    = l;
            const double *const node = &s_u[(ptrdiff_t)l * CVFEM_CUDA_NF];
            ux[a] = node[0]; uy[a] = node[1]; uz[a] = node[2]; pe[a] = node[3];
        }

        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        double re[CVFEM_HEX8_N_DOF];
        if (WITH_RC) {
            // The kernel wants element-local arrays of 8, so gather from shared.
            double rx[8], ry[8], rz[8], rgx[8], rgy[8], rgz[8];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const double *const q = &s_rc[(ptrdiff_t)ev[a] * 6];
                rx[a] = q[0]; ry[a] = q[1]; rz[a] = q[2];
                rgx[a] = q[3]; rgy[a] = q[4]; rgz[a] = q[5];
            }
            Hex8RhieChowT<double> rc;
            rc.x = rx; rc.y = ry; rc.z = rz;
            rc.pgx = rgx; rc.pgy = rgy; rc.pgz = rgz; rc.scale = rc_scale;
            cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj_e, det[e], ux, uy, uz, pe, re, rc);
        } else {
            cvfem_hex8_ns_upwind_residual(rho, mu, adj_e, det[e], ux, uy, uz, pe, re);
        }

        // Elements within a pack share nodes, so this accumulation needs atomics even
        // though the pack's buffer is block-private.
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            double *const acc = &s_out[(ptrdiff_t)ev[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&acc[f], re[a * 4 + f]);
        }
    }
    __syncthreads();

    if (FLUSH == CVFEM_CUDA_FLUSH_TWO_PASS) {
        // Owned nodes have exactly one writing pack in this mode, so a plain store is
        // race-free; ghosts go to their own slot and are gathered afterwards.
        for (ptrdiff_t i = threadIdx.x; i < n_contiguous; i += blockDim.x) {
            const double *const acc = &s_out[i * CVFEM_CUDA_NF];
            double *const       dst = &r[(owned + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] = acc[f];
        }
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x) {
            const double *const acc = &s_out[(n_contiguous + i) * CVFEM_CUDA_NF];
            double *const       dst = &ghost_buf[(gbegin + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] = acc[f];
        }
    } else {
        // One pass. The owned prefix below n_not_shared is touched by no other pack
        // (PACKED_FORMAT.md section 3), so it needs no atomics; everything above it can
        // race with another pack's ghost flush.
        const ptrdiff_t n_not_shared = n_contiguous - n_shared[p];
        for (ptrdiff_t i = threadIdx.x; i < n_not_shared; i += blockDim.x) {
            const double *const acc = &s_out[i * CVFEM_CUDA_NF];
            double *const       dst = &r[(owned + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] += acc[f];
        }
        for (ptrdiff_t i = n_not_shared + threadIdx.x; i < n_contiguous; i += blockDim.x) {
            const double *const acc = &s_out[i * CVFEM_CUDA_NF];
            double *const       dst = &r[(owned + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], acc[f]);
        }
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x) {
            const double *const acc = &s_out[(n_contiguous + i) * CVFEM_CUDA_NF];
            double *const       dst = &r[(ptrdiff_t)ghost_idx[gbegin + i] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], acc[f]);
        }
    }
}

// y = J(u) v. Structurally the residual kernel with a third staged array; the flush is
// identical, so the two share the same ghost-reduce pass.
template <int FLUSH>
__global__ void cvfem_hex8_jacobian_action_pack_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const double rho, const double mu,
        const uint16_t  *const __restrict__ elems,
        const ptrdiff_t *const __restrict__ owned_nodes_ptr,
        const ptrdiff_t *const __restrict__ n_shared,
        const ptrdiff_t *const __restrict__ ghost_ptr,
        const int32_t   *const __restrict__ ghost_idx,
        const double    *const __restrict__ adj,
        const double    *const __restrict__ det,
        const double    *const __restrict__ u,
        const double    *const __restrict__ vin,
        double *const __restrict__ r,
        double *const __restrict__ ghost_buf) {
    extern __shared__ double smem[];

    const ptrdiff_t p            = blockIdx.x;
    const ptrdiff_t owned        = owned_nodes_ptr[p];
    const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned;
    const ptrdiff_t gbegin       = ghost_ptr[p];
    const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - gbegin;
    const ptrdiff_t total_nodes  = n_contiguous + n_ghost;
    const ptrdiff_t stride       = (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;

    double *const s_u   = smem;
    double *const s_v   = smem + stride;
    double *const s_out = smem + 2 * stride;

    for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
        const ptrdiff_t g = (i < n_contiguous) ? (owned + i)
                                               : (ptrdiff_t)ghost_idx[gbegin + i - n_contiguous];
        const double *const su = &u[g * CVFEM_CUDA_NF];
        const double *const sv = &vin[g * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) {
            s_u[i * CVFEM_CUDA_NF + f]   = su[f];
            s_v[i * CVFEM_CUDA_NF + f]   = sv[f];
            s_out[i * CVFEM_CUDA_NF + f] = 0.0;
        }
    }
    __syncthreads();

    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        uint16_t ev[CVFEM_HEX8_N_NODES];
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
        double vx[CVFEM_HEX8_N_NODES], vy[CVFEM_HEX8_N_NODES], vz[CVFEM_HEX8_N_NODES];
        double pe[CVFEM_HEX8_N_NODES], q[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const uint16_t l = elems[(ptrdiff_t)a * nelements + e];
            ev[a] = l;
            const double *const nu = &s_u[(ptrdiff_t)l * CVFEM_CUDA_NF];
            const double *const nv = &s_v[(ptrdiff_t)l * CVFEM_CUDA_NF];
            ux[a] = nu[0]; uy[a] = nu[1]; uz[a] = nu[2]; pe[a] = nu[3];
            vx[a] = nv[0]; vy[a] = nv[1]; vz[a] = nv[2]; q[a]  = nv[3];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        double re[CVFEM_HEX8_N_DOF];
        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj_e, det[e], ux, uy, uz,
                                             vx, vy, vz, q, re);
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            double *const acc = &s_out[(ptrdiff_t)ev[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&acc[f], re[a * 4 + f]);
        }
        (void)pe;
    }
    __syncthreads();

    if (FLUSH == CVFEM_CUDA_FLUSH_TWO_PASS) {
        for (ptrdiff_t i = threadIdx.x; i < n_contiguous; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                r[(owned + i) * CVFEM_CUDA_NF + f] = s_out[i * CVFEM_CUDA_NF + f];
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                ghost_buf[(gbegin + i) * CVFEM_CUDA_NF + f] =
                        s_out[(n_contiguous + i) * CVFEM_CUDA_NF + f];
    } else {
        const ptrdiff_t n_not_shared = n_contiguous - n_shared[p];
        for (ptrdiff_t i = threadIdx.x; i < n_not_shared; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                r[(owned + i) * CVFEM_CUDA_NF + f] += s_out[i * CVFEM_CUDA_NF + f];
        for (ptrdiff_t i = n_not_shared + threadIdx.x; i < n_contiguous; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                atomicAdd(&r[(owned + i) * CVFEM_CUDA_NF + f], s_out[i * CVFEM_CUDA_NF + f]);
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                atomicAdd(&r[(ptrdiff_t)ghost_idx[gbegin + i] * CVFEM_CUDA_NF + f],
                          s_out[(n_contiguous + i) * CVFEM_CUDA_NF + f]);
    }
}

// Gather the staged ghost contributions. Each destination appears in exactly one row,
// so this is race-free and bit-deterministic.
__global__ void cvfem_hex8_ghost_reduce_kernel(
        const ptrdiff_t n_rows,
        const ptrdiff_t *const __restrict__ reduce_ptr,
        const ptrdiff_t *const __restrict__ reduce_idx,
        const int32_t   *const __restrict__ reduce_dest,
        const double    *const __restrict__ ghost_buf,
        double *const __restrict__ r) {
    for (ptrdiff_t row = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; row < n_rows;
         row += (ptrdiff_t)blockDim.x * gridDim.x) {
        double acc[CVFEM_CUDA_NF] = {0.0, 0.0, 0.0, 0.0};
        const ptrdiff_t b = reduce_ptr[row], e = reduce_ptr[row + 1];
        for (ptrdiff_t j = b; j < e; ++j) {
            const double *const g = &ghost_buf[reduce_idx[j] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) acc[f] += g[f];
        }
        double *const dst = &r[(ptrdiff_t)reduce_dest[row] * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] += acc[f];
    }
}

// ---------------------------------------------------------------- assembly

// Element-parallel, grid-stride, writing straight into the global BSR with atomicAdd.
// Both kernel families already accumulate through CVFEM_ATOMIC_ADD, which expands to
// atomicAdd under __CUDA_ARCH__, so the same source serves host and device.
template <int VARIANT>
__global__ void cvfem_hex8_assemble_bsr_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g    = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2]; pe[a] = n[3];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        const int32_t *const es = &slots[e * 64];
        if      constexpr (VARIANT == CVFEM_CUDA_JAC_HANDWRITTEN)
            cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_BLOCK)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_ROW)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        (void)pe;
    }
}

template <int VARIANT>
int launch_assemble_v(cvfem_cuda_ctx *ctx, double rho, double mu, int block, cudaStream_t s) {
    const int grid = (int)((ctx->nelements + block - 1) / block);
    cvfem_hex8_assemble_bsr_kernel<VARIANT><<<grid, block, 0, s>>>(
            ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
            ctx->adj, ctx->det, ctx->u, ctx->values);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

int launch_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu, int flush_mode,
                           int block_size, cudaStream_t s) {
    if (!ctx->v) return 1;
    if (!ctx->jv_optin_done) {
        if (ctx->jv_shmem_bytes > 48u * 1024u) {
            CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ctx->jv_shmem_bytes));
            CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ctx->jv_shmem_bytes));
        }
        ctx->jv_optin_done = true;
    }
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)ctx->n_packs;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->r, 0,
                                     (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double), s));
    if (flush_mode == CVFEM_CUDA_FLUSH_TWO_PASS) {
        cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS>
                <<<grid, block, ctx->jv_shmem_bytes, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->v, ctx->r, ctx->ghost_buf);
        CVFEM_CUDA_CHECK(cudaGetLastError());
        if (ctx->n_ghost_reduce_rows > 0) {
            const int rb = 256, rg = (int)((ctx->n_ghost_reduce_rows + rb - 1) / rb);
            cvfem_hex8_ghost_reduce_kernel<<<rg, rb, 0, s>>>(
                    ctx->n_ghost_reduce_rows, ctx->ghost_reduce_ptr, ctx->ghost_reduce_idx,
                    ctx->ghost_reduce_dest, ctx->ghost_buf, ctx->r);
            CVFEM_CUDA_CHECK(cudaGetLastError());
        }
    } else {
        cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC>
                <<<grid, block, ctx->jv_shmem_bytes, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->v, ctx->r, ctx->ghost_buf);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

int launch_assemble(cvfem_cuda_ctx *ctx, double rho, double mu, int variant,
                    int block_size, cudaStream_t s) {
    if (!ctx->values) return 1;
    const int block = block_size > 0 ? block_size : 128;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    switch (variant) {
        case CVFEM_CUDA_JAC_HANDWRITTEN: return launch_assemble_v<CVFEM_CUDA_JAC_HANDWRITTEN>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY:       return launch_assemble_v<CVFEM_CUDA_JAC_SYMPY>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_BLOCK: return launch_assemble_v<CVFEM_CUDA_JAC_SYMPY_BLOCK>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_ROW:   return launch_assemble_v<CVFEM_CUDA_JAC_SYMPY_ROW>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_FACE:  return launch_assemble_v<CVFEM_CUDA_JAC_SYMPY_FACE>(ctx, rho, mu, block, s);
        default: return 1;
    }
}

// One block per pack of the current colour, accumulating without atomics.
//
// !! CORRECT ONLY WITH blockDim.x == 1. !!
//
// Pack colouring removes *inter*-pack races: two packs of the same colour share no
// nodes, so they cannot write the same BSR block. It does nothing about *intra*-pack
// races, and on a GPU a pack is a whole block of threads -- two threads working on two
// elements of the same pack do share nodes, and do write the same block.
//
// On the CPU this distinction does not exist because a pack is one thread, which is why
// the colored layout is correct there. Measured here: with blockDim.x > 1 the result is
// wrong by a relative 0.54; with blockDim.x == 1 it agrees to 7.7e-16 and runs at
// 1.2 MDOF/s, roughly 200x slower than the atomic variants.
//
// So this kernel does not test the "is assembly atomic-bound?" question. Testing that
// needs an *element*-level colouring, which is a different and much larger colouring
// problem than the pack colouring the code already builds. Kept as an executable
// demonstration of the distinction, not as a performance path.
template <bool USE_SYMPY>
__global__ void cvfem_hex8_assemble_colored_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t color_begin, const double rho, const double mu,
        const ptrdiff_t *const __restrict__ pack_order,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values) {
    const ptrdiff_t p       = pack_order[color_begin + blockIdx.x];
    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g     = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2]; pe[a] = n[3];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        const int32_t *const es = &slots[e * 64];
        if constexpr (USE_SYMPY)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else
            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        (void)pe;
    }
}

template <bool USE_SYMPY>
int launch_colored_v(cvfem_cuda_ctx *ctx, double rho, double mu, int block, cudaStream_t s) {
    for (int c = 0; c < ctx->n_colors; ++c) {
        const ptrdiff_t b = ctx->h_color_ptr[c], e = ctx->h_color_ptr[c + 1];
        const int       n = (int)(e - b);
        if (n <= 0) continue;
        cvfem_hex8_assemble_colored_kernel<USE_SYMPY><<<n, block, 0, s>>>(
                ctx->nelements, ctx->n_elements_per_pack, b, rho, mu, ctx->pack_order,
                ctx->elements_global, ctx->element_slots, ctx->adj, ctx->det,
                ctx->u, ctx->values);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

int launch_colored(cvfem_cuda_ctx *ctx, double rho, double mu, int use_sympy,
                   int block_size, cudaStream_t s) {
    if (!ctx->values || !ctx->pack_order) return 1;
    const int block = block_size > 0 ? block_size : 128;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    return use_sympy ? launch_colored_v<true>(ctx, rho, mu, block, s)
                     : launch_colored_v<false>(ctx, rho, mu, block, s);
}

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

enum BoundaryOp { BOUNDARY_RESIDUAL = 0, BOUNDARY_JV = 1, BOUNDARY_ASSEMBLE = 2 };

template <int OP>
__global__ void cvfem_hex8_boundary_kernel(
        const ptrdiff_t n_boundary, const ptrdiff_t nelements,
        const double rho, const double mu,
        const double Lx, const double Ly, const double Lz,
        const int32_t *const __restrict__ blist,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ px,
        const double  *const __restrict__ py,
        const double  *const __restrict__ pz,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        const double  *const __restrict__ vin,
        double *const __restrict__ r,
        double *const __restrict__ values) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_boundary;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t e = blist[t];
        int32_t ev[CVFEM_HEX8_N_NODES];
        double  x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
        double  pe[CVFEM_HEX8_N_NODES];
        double  vx[CVFEM_HEX8_N_NODES], vy[CVFEM_HEX8_N_NODES], vz[CVFEM_HEX8_N_NODES];
        double  q[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            ev[a] = g;
            x[a] = px[g]; y[a] = py[g]; z[a] = pz[g];
            const double *const nd = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = nd[0]; uy[a] = nd[1]; uz[a] = nd[2]; pe[a] = nd[3];
            if (OP == BOUNDARY_JV) {
                const double *const nv = &vin[(ptrdiff_t)g * CVFEM_CUDA_NF];
                vx[a] = nv[0]; vy[a] = nv[1]; vz[a] = nv[2]; q[a] = nv[3];
            }
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        if (OP == BOUNDARY_ASSEMBLE) {
            // Accumulates straight into the global BSR through CVFEM_ATOMIC_ADD.
            boundary_scs_add_jacobian<true>(rho, mu, 0, adj_e, det[e], Lx, Ly, Lz,
                                            x, y, z, ux, uy, uz, &slots[e * 64], values);
        } else {
            double re[CVFEM_HEX8_N_DOF];
#pragma unroll
            for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) re[i] = 0.0;
            if (OP == BOUNDARY_RESIDUAL)
                boundary_scs_add_residual(rho, mu, 0, adj_e, det[e], Lx, Ly, Lz,
                                          x, y, z, ux, uy, uz, pe, re);
            else
                boundary_scs_add_jacobian_action(rho, mu, 0, adj_e, det[e], Lx, Ly, Lz,
                                                 x, y, z, ux, uy, uz, vx, vy, vz, q, re);
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                double *const dst = &r[(ptrdiff_t)ev[a] * CVFEM_CUDA_NF];
#pragma unroll
                for (int f = 0; f < CVFEM_CUDA_NF; ++f) {
                    const double v = re[a * 4 + f];
                    if (v != 0.0) atomicAdd(&dst[f], v);
                }
            }
        }
    }
}

template <int OP>
int launch_boundary(cvfem_cuda_ctx *ctx, double rho, double mu, int block_size,
                    cudaStream_t s) {
    if (ctx->n_boundary <= 0) return 0;
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->n_boundary + block - 1) / block);
    cvfem_hex8_boundary_kernel<OP><<<grid, block, 0, s>>>(
            ctx->n_boundary, ctx->nelements, rho, mu, ctx->Lx, ctx->Ly, ctx->Lz,
            ctx->boundary_elems, ctx->elements_global, ctx->element_slots,
            ctx->px, ctx->py, ctx->pz, ctx->adj, ctx->det, ctx->u, ctx->v,
            ctx->r, ctx->values);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

int ensure_shmem_optin(cvfem_cuda_ctx *ctx) {
    if (ctx->shmem_optin_done) return 0;
    // Anything above the 48 KB default needs an explicit opt-in per kernel.
    if (ctx->shmem_bytes > 48u * 1024u) {
        CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ctx->shmem_bytes));
        CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ctx->shmem_bytes));
    }
    ctx->shmem_optin_done = true;
    return 0;
}

int launch_residual(cvfem_cuda_ctx *ctx, double rho, double mu, int flush_mode,
                    int block_size, cudaStream_t s) {
    if (ensure_shmem_optin(ctx) != 0) return 1;
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)ctx->n_packs;

    // Both modes accumulate into r, so it must start at zero.
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->r, 0,
                                     (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double), s));

    if (flush_mode == CVFEM_CUDA_FLUSH_TWO_PASS) {
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, false>
                <<<grid, block, ctx->shmem_bytes, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->r, ctx->ghost_buf,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.0);
        CVFEM_CUDA_CHECK(cudaGetLastError());
        if (ctx->n_ghost_reduce_rows > 0) {
            const int rblock = 256;
            const int rgrid  = (int)((ctx->n_ghost_reduce_rows + rblock - 1) / rblock);
            cvfem_hex8_ghost_reduce_kernel<<<rgrid, rblock, 0, s>>>(
                    ctx->n_ghost_reduce_rows, ctx->ghost_reduce_ptr, ctx->ghost_reduce_idx,
                    ctx->ghost_reduce_dest, ctx->ghost_buf, ctx->r);
            CVFEM_CUDA_CHECK(cudaGetLastError());
        }
    } else {
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, false>
                <<<grid, block, ctx->shmem_bytes, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->r, ctx->ghost_buf,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.0);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------- ABI

extern "C" int cvfem_cuda_device_info(int *sm_count, int *max_shmem_per_block,
                                      int *max_optin_shmem_per_block, int *warp_size) {
    int dev = 0;
    CVFEM_CUDA_CHECK(cudaGetDevice(&dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(max_shmem_per_block,
                                            cudaDevAttrMaxSharedMemoryPerBlock, dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(max_optin_shmem_per_block,
                                            cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(warp_size, cudaDevAttrWarpSize, dev));
    return 0;
}

extern "C" size_t cvfem_cuda_residual_shmem_bytes(ptrdiff_t max_pack_nodes) {
    return (size_t)2 * CVFEM_CUDA_NF * (size_t)max_pack_nodes * sizeof(double);
}

extern "C" int cvfem_cuda_create(cvfem_cuda_ctx **out_ctx,
                                 ptrdiff_t nnodes, ptrdiff_t nelements,
                                 ptrdiff_t n_packs, ptrdiff_t n_elements_per_pack,
                                 ptrdiff_t max_pack_nodes,
                                 ptrdiff_t n_ghost_entries, ptrdiff_t n_ghost_reduce_rows,
                                 const uint16_t *elems_flat,
                                 const ptrdiff_t *owned_nodes_ptr, const ptrdiff_t *n_shared,
                                 const ptrdiff_t *ghost_ptr, const int32_t *ghost_idx,
                                 const ptrdiff_t *ghost_reduce_ptr,
                                 const ptrdiff_t *ghost_reduce_idx,
                                 const int32_t *ghost_reduce_dest,
                                 const double *adj_flat, const double *det) {
    cvfem_cuda_ctx *c = new cvfem_cuda_ctx();
    c->nnodes = nnodes; c->nelements = nelements;
    c->n_packs = n_packs; c->n_elements_per_pack = n_elements_per_pack;
    c->max_pack_nodes = max_pack_nodes;
    c->n_ghost_entries = n_ghost_entries; c->n_ghost_reduce_rows = n_ghost_reduce_rows;
    c->shmem_bytes = cvfem_cuda_residual_shmem_bytes(max_pack_nodes);
    c->jv_shmem_bytes = cvfem_cuda_jacobian_action_shmem_bytes(max_pack_nodes);

    if (device_dup(&c->elems, elems_flat, (size_t)8 * nelements) ||
        device_dup(&c->owned_nodes_ptr, owned_nodes_ptr, (size_t)n_packs + 1) ||
        device_dup(&c->n_shared, n_shared, (size_t)n_packs) ||
        device_dup(&c->ghost_ptr, ghost_ptr, (size_t)n_packs + 1) ||
        device_dup(&c->ghost_idx, ghost_idx, (size_t)n_ghost_entries) ||
        device_dup(&c->ghost_reduce_ptr, ghost_reduce_ptr, (size_t)n_ghost_reduce_rows + 1) ||
        device_dup(&c->ghost_reduce_idx, ghost_reduce_idx, (size_t)n_ghost_entries) ||
        device_dup(&c->ghost_reduce_dest, ghost_reduce_dest, (size_t)n_ghost_reduce_rows) ||
        device_dup(&c->adj, adj_flat, (size_t)9 * nelements) ||
        device_dup(&c->det, det, (size_t)nelements)) {
        delete c;
        return 1;
    }
    if (cudaMalloc(&c->u, (size_t)nnodes * CVFEM_CUDA_NF * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&c->r, (size_t)nnodes * CVFEM_CUDA_NF * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&c->v, (size_t)nnodes * CVFEM_CUDA_NF * sizeof(double)) != cudaSuccess) {
        delete c; return 1;
    }
    if (n_ghost_entries > 0 &&
        cudaMalloc(&c->ghost_buf,
                   (size_t)n_ghost_entries * CVFEM_CUDA_NF * sizeof(double)) != cudaSuccess) {
        delete c; return 1;
    }
    *out_ctx = c;
    return 0;
}

extern "C" int cvfem_cuda_destroy(cvfem_cuda_ctx *ctx) {
    if (!ctx) return 0;
    cudaFree(ctx->elems); cudaFree(ctx->owned_nodes_ptr); cudaFree(ctx->n_shared);
    cudaFree(ctx->ghost_ptr); cudaFree(ctx->ghost_idx);
    cudaFree(ctx->ghost_reduce_ptr); cudaFree(ctx->ghost_reduce_idx);
    cudaFree(ctx->ghost_reduce_dest);
    cudaFree(ctx->adj); cudaFree(ctx->det);
    cudaFree(ctx->u); cudaFree(ctx->r); cudaFree(ctx->v); cudaFree(ctx->ghost_buf);
    cudaFree(ctx->elements_global); cudaFree(ctx->element_slots); cudaFree(ctx->values);
    cudaFree(ctx->pack_order); cudaFree(ctx->color_ptr);
    cudaFree(ctx->rowptr); cudaFree(ctx->colidx);
    if (ctx->spdesc) cusparseDestroyMatDescr(ctx->spdesc);
    if (ctx->sp) cusparseDestroy(ctx->sp);
    cudaFree(ctx->boundary_elems); cudaFree(ctx->px); cudaFree(ctx->py); cudaFree(ctx->pz);
    cudaFree(ctx->pgx); cudaFree(ctx->pgy); cudaFree(ctx->pgz); cudaFree(ctx->pgw);
    delete ctx;
    return 0;
}

extern "C" int cvfem_cuda_upload_u(cvfem_cuda_ctx *ctx, const double *u) {
    CVFEM_CUDA_CHECK(cudaMemcpy(ctx->u, u,
                                (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double),
                                cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cvfem_cuda_download_r(cvfem_cuda_ctx *ctx, double *r) {
    CVFEM_CUDA_CHECK(cudaMemcpy(r, ctx->r,
                                (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double),
                                cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" size_t cvfem_cuda_jacobian_action_shmem_bytes(ptrdiff_t max_pack_nodes) {
    return (size_t)3 * CVFEM_CUDA_NF * (size_t)max_pack_nodes * sizeof(double);
}

extern "C" int cvfem_cuda_upload_v(cvfem_cuda_ctx *ctx, const double *v) {
    CVFEM_CUDA_CHECK(cudaMemcpy(ctx->v, v,
                                (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double),
                                cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cvfem_cuda_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu,
                                          int flush_mode, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_jacobian_action(ctx, rho, mu, flush_mode, block_size, s);
}

extern "C" double cvfem_cuda_time_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                  int flush_mode, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_jacobian_action(ctx, rho, mu, flush_mode, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_jacobian_action(ctx, rho, mu, flush_mode, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_residual(cvfem_cuda_ctx *ctx, double rho, double mu,
                                   int flush_mode, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_residual(ctx, rho, mu, flush_mode, block_size, s);
}

extern "C" const char *cvfem_cuda_jac_variant_name(int v) {
    switch (v) {
        case CVFEM_CUDA_JAC_HANDWRITTEN: return "handwritten";
        case CVFEM_CUDA_JAC_SYMPY:       return "sympy";
        case CVFEM_CUDA_JAC_SYMPY_BLOCK: return "sympy_block";
        case CVFEM_CUDA_JAC_SYMPY_ROW:   return "sympy_row";
        case CVFEM_CUDA_JAC_SYMPY_FACE:  return "sympy_face";
        default: return "?";
    }
}

extern "C" int cvfem_cuda_bsr_attach(cvfem_cuda_ctx *ctx, ptrdiff_t nnz,
                                     const int32_t *elements_global_flat,
                                     const int32_t *element_slots,
                                     const int32_t *rowptr, const int32_t *colidx) {
    ctx->nnz = nnz;
    if (device_dup(&ctx->elements_global, elements_global_flat, (size_t)8 * ctx->nelements) ||
        device_dup(&ctx->element_slots, element_slots, (size_t)64 * ctx->nelements) ||
        device_dup(&ctx->rowptr, rowptr, (size_t)ctx->nnodes + 1) ||
        device_dup(&ctx->colidx, colidx, (size_t)nnz))
        return 1;
    CVFEM_CUDA_CHECK(cudaMalloc(&ctx->values, (size_t)nnz * 16 * sizeof(double)));
    return 0;
}

extern "C" int cvfem_cuda_spmv(cvfem_cuda_ctx *ctx, void *stream) {
    if (!ctx->values || !ctx->rowptr) return 1;
    if (!ctx->sp) {
        if (cusparseCreate(&ctx->sp) != CUSPARSE_STATUS_SUCCESS) return 1;
        if (cusparseCreateMatDescr(&ctx->spdesc) != CUSPARSE_STATUS_SUCCESS) return 1;
        cusparseSetMatType(ctx->spdesc, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(ctx->spdesc, CUSPARSE_INDEX_BASE_ZERO);
    }
    if (stream) cusparseSetStream(ctx->sp, *static_cast<cudaStream_t *>(stream));
    const double alpha = 1.0, beta = 0.0;
    const int    mb = (int)ctx->nnodes;   // block rows == nodes; 4 unknowns per node
    const cusparseStatus_t st = cusparseDbsrmv(
            ctx->sp, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
            mb, mb, (int)ctx->nnz, &alpha, ctx->spdesc,
            ctx->values, ctx->rowptr, ctx->colidx, 4,
            ctx->v, &beta, ctx->r);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "cusparseDbsrmv failed: %d\n", (int)st);
        return 1;
    }
    return 0;
}

extern "C" double cvfem_cuda_time_spmv(cvfem_cuda_ctx *ctx, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (cvfem_cuda_spmv(ctx, nullptr) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (cvfem_cuda_spmv(ctx, nullptr) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_assemble(cvfem_cuda_ctx *ctx, double rho, double mu,
                                   int variant, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_assemble(ctx, rho, mu, variant, block_size, s);
}

extern "C" int cvfem_cuda_download_values(cvfem_cuda_ctx *ctx, double *values) {
    CVFEM_CUDA_CHECK(cudaMemcpy(values, ctx->values, (size_t)ctx->nnz * 16 * sizeof(double),
                                cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" double cvfem_cuda_time_assemble(cvfem_cuda_ctx *ctx, double rho, double mu,
                                           int variant, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_assemble(ctx, rho, mu, variant, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_assemble(ctx, rho, mu, variant, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_coloring_attach(cvfem_cuda_ctx *ctx, int n_colors,
                                          const ptrdiff_t *pack_order,
                                          const ptrdiff_t *color_ptr) {
    ctx->n_colors = n_colors;
    ctx->h_color_ptr.assign(color_ptr, color_ptr + n_colors + 1);
    if (device_dup(&ctx->pack_order, pack_order, (size_t)ctx->n_packs) ||
        device_dup(&ctx->color_ptr, color_ptr, (size_t)n_colors + 1))
        return 1;
    return 0;
}

extern "C" int cvfem_cuda_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                           int use_sympy, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_colored(ctx, rho, mu, use_sympy, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                   int use_sympy, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_colored(ctx, rho, mu, use_sympy, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_colored(ctx, rho, mu, use_sympy, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_boundary_attach(cvfem_cuda_ctx *ctx, ptrdiff_t n_boundary,
                                          const int32_t *boundary_elems,
                                          const double *px, const double *py, const double *pz,
                                          double Lx, double Ly, double Lz) {
    ctx->n_boundary = n_boundary;
    ctx->Lx = Lx; ctx->Ly = Ly; ctx->Lz = Lz;
    if (device_dup(&ctx->boundary_elems, boundary_elems, (size_t)n_boundary) ||
        device_dup(&ctx->px, px, (size_t)ctx->nnodes) ||
        device_dup(&ctx->py, py, (size_t)ctx->nnodes) ||
        device_dup(&ctx->pz, pz, (size_t)ctx->nnodes))
        return 1;
    return 0;
}

extern "C" int cvfem_cuda_boundary_residual(cvfem_cuda_ctx *ctx, double rho, double mu,
                                            int block_size, void *stream) {
    if (ctx->n_boundary <= 0) return 0;
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_boundary<BOUNDARY_RESIDUAL>(ctx, rho, mu, block_size, s);
}

extern "C" int cvfem_cuda_boundary_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                   int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_boundary<BOUNDARY_JV>(ctx, rho, mu, block_size, s);
}

extern "C" int cvfem_cuda_boundary_assemble(cvfem_cuda_ctx *ctx, double rho, double mu,
                                            int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_boundary<BOUNDARY_ASSEMBLE>(ctx, rho, mu, block_size, s);
}

extern "C" int cvfem_cuda_nodal_p_grad(cvfem_cuda_ctx *ctx, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    const size_t nb = (size_t)ctx->nnodes * sizeof(double);
    if (!ctx->pgx) {
        if (cudaMalloc(&ctx->pgx, nb) != cudaSuccess || cudaMalloc(&ctx->pgy, nb) != cudaSuccess ||
            cudaMalloc(&ctx->pgz, nb) != cudaSuccess || cudaMalloc(&ctx->pgw, nb) != cudaSuccess)
            return 1;
    }
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->pgx, 0, nb, s));
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->pgy, 0, nb, s));
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->pgz, 0, nb, s));
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->pgw, 0, nb, s));

    const int block = block_size > 0 ? block_size : 128;
    int grid = (int)((ctx->nelements + block - 1) / block);
    cvfem_hex8_nodal_p_grad_accumulate_kernel<<<grid, block, 0, s>>>(
            ctx->nelements, ctx->elements_global, ctx->adj, ctx->det, ctx->u,
            ctx->pgx, ctx->pgy, ctx->pgz, ctx->pgw);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    grid = (int)((ctx->nnodes + block - 1) / block);
    cvfem_hex8_nodal_p_grad_normalize_kernel<<<grid, block, 0, s>>>(
            ctx->nnodes, ctx->pgx, ctx->pgy, ctx->pgz, ctx->pgw);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cvfem_cuda_download_p_grad(cvfem_cuda_ctx *ctx, double *pgx, double *pgy,
                                          double *pgz) {
    const size_t nb = (size_t)ctx->nnodes * sizeof(double);
    CVFEM_CUDA_CHECK(cudaMemcpy(pgx, ctx->pgx, nb, cudaMemcpyDeviceToHost));
    CVFEM_CUDA_CHECK(cudaMemcpy(pgy, ctx->pgy, nb, cudaMemcpyDeviceToHost));
    CVFEM_CUDA_CHECK(cudaMemcpy(pgz, ctx->pgz, nb, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" size_t cvfem_cuda_residual_rc_shmem_bytes(ptrdiff_t max_pack_nodes) {
    // 4 staged fields + 4 accumulated + 6 Rhie-Chow = 14 doubles per node.
    return (size_t)14 * (size_t)max_pack_nodes * sizeof(double);
}

extern "C" int cvfem_cuda_residual_rc(cvfem_cuda_ctx *ctx, double rho, double mu,
                                      double rc_scale, int flush_mode, int block_size,
                                      void *stream) {
    if (!ctx->pgx || !ctx->px) return 1;
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    const size_t need = cvfem_cuda_residual_rc_shmem_bytes(ctx->max_pack_nodes);
    if (need > 48u * 1024u) {
        CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need));
        CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need));
    }
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)ctx->n_packs;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->r, 0,
                                     (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double), s));
    if (flush_mode == CVFEM_CUDA_FLUSH_TWO_PASS) {
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, true><<<grid, block, need, s>>>(
                ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                ctx->adj, ctx->det, ctx->u, ctx->r, ctx->ghost_buf,
                ctx->px, ctx->py, ctx->pz, ctx->pgx, ctx->pgy, ctx->pgz, rc_scale);
        CVFEM_CUDA_CHECK(cudaGetLastError());
        if (ctx->n_ghost_reduce_rows > 0) {
            const int rb = 256, rg = (int)((ctx->n_ghost_reduce_rows + rb - 1) / rb);
            cvfem_hex8_ghost_reduce_kernel<<<rg, rb, 0, s>>>(
                    ctx->n_ghost_reduce_rows, ctx->ghost_reduce_ptr, ctx->ghost_reduce_idx,
                    ctx->ghost_reduce_dest, ctx->ghost_buf, ctx->r);
            CVFEM_CUDA_CHECK(cudaGetLastError());
        }
    } else {
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, true><<<grid, block, need, s>>>(
                ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                ctx->adj, ctx->det, ctx->u, ctx->r, ctx->ghost_buf,
                ctx->px, ctx->py, ctx->pz, ctx->pgx, ctx->pgy, ctx->pgz, rc_scale);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

extern "C" int cvfem_cuda_synchronize(void) {
    CVFEM_CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

extern "C" double cvfem_cuda_time_residual(cvfem_cuda_ctx *ctx, double rho, double mu,
                                           int flush_mode, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_residual(ctx, rho, mu, flush_mode, block_size, 0) != 0) return -1.0;  // warm
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_residual(ctx, rho, mu, flush_mode, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}
