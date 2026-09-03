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

// Geometry model, mirroring the host's GeomKind. Affine reads one precomputed adjugate
// and determinant per element; isoparametric evaluates the trilinear Jacobian at each of
// the 12 sub-control-surface points from the element's node coordinates, which is 12
// 3x3 inversions per element instead of a lookup.
static constexpr int CVFEM_CUDA_GEOM_AFFINE   = 0;
static constexpr int CVFEM_CUDA_GEOM_ISOPARAM = 1;

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
    // Isoparametric needs three more doubles per node for the coordinates, in both the
    // residual and the J*v kernel, so it carries its own sizes and its own opt-in.
    size_t  iso_shmem_bytes{0}, iso_jv_shmem_bytes{0};
    bool    iso_optin_done{false}, iso_jv_optin_done{false};

    // assembled BSR
    ptrdiff_t nnz{0};
    int32_t  *elements_global{nullptr};   // [8 * nelements], GLOBAL ids
    int32_t  *element_slots{nullptr};     // [64 * nelements]
    double   *values{nullptr};            // [16 * nnz]
    double   *values_linear{nullptr};     // [16 * nnz], geometry-only, built once
    double   *diag{nullptr};              // [16 * nnodes], block diagonal
    double   *diag_static{nullptr};       // [16 * nnodes], its viscous part
    int32_t  *nl_blocks{nullptr};         // block ids the nonlinear half writes
    ptrdiff_t n_nl_blocks{0};
    uint16_t *nl_masks{nullptr};          // [nnz], by block id: which entries change
    double   *linear_compact{nullptr};    // [16 * n_nl_blocks], only what gets overwritten
    int32_t  *rowptr{nullptr};            // [nnodes + 1], block rows
    int32_t  *colidx{nullptr};            // [nnz]
    cusparseHandle_t      sp{nullptr};
    cusparseMatDescr_t    spdesc{nullptr};

    ptrdiff_t  n_boundary{0};
    int32_t   *boundary_elems{nullptr};
    double    *px{nullptr}, *py{nullptr}, *pz{nullptr};
    double    *pgx{nullptr}, *pgy{nullptr}, *pgz{nullptr}, *pgw{nullptr};
    double     Lx{0}, Ly{0}, Lz{0};

    int        n_ecolors{0};
    int32_t   *element_order{nullptr};
    std::vector<ptrdiff_t> h_ecolor_ptr;

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

// GEOM selects the geometry model: CVFEM_CUDA_GEOM_AFFINE reads one precomputed
// adjugate and determinant per element, CVFEM_CUDA_GEOM_ISOPARAM evaluates the
// trilinear Jacobian at each of the 12 sub-control-surface points from the element's
// node coordinates. Isoparametric therefore needs the coordinates staged per pack --
// the same three arrays Rhie-Chow already stages, so when both are on they share one
// buffer instead of holding two copies.
template <int FLUSH, bool WITH_RC, int GEOM>
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

    constexpr bool NEED_XYZ = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) || WITH_RC;

    double *const s_u   = smem;
    double *const s_out = smem + (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;
    // Coordinates, staged when the geometry is isoparametric or Rhie-Chow needs them.
    double *const s_xyz = smem + 2 * (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;
    // The nodal pressure gradient sits after them, so the coordinate block has the same
    // address in both cases and one gather serves both consumers.
    double *const s_pg  = s_xyz + (NEED_XYZ ? 3 * total_nodes : 0);

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
        if (NEED_XYZ) {
            double *const q = &s_xyz[i * 3];
            q[0] = px[g]; q[1] = py[g]; q[2] = pz[g];
        }
        if (WITH_RC) {
            double *const q = &s_pg[i * 3];
            q[0] = pgx[g]; q[1] = pgy[g]; q[2] = pgz[g];
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

        // The kernels want element-local arrays of 8, so gather from shared.
        double ex[8], ey[8], ez[8];
        if (NEED_XYZ) {
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const double *const q = &s_xyz[(ptrdiff_t)ev[a] * 3];
                ex[a] = q[0]; ey[a] = q[1]; ez[a] = q[2];
            }
        }

        double adj_e[9];
        if (GEOM == CVFEM_CUDA_GEOM_AFFINE) {
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        }

        double re[CVFEM_HEX8_N_DOF];
        if (WITH_RC) {
            double rgx[8], rgy[8], rgz[8];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const double *const q = &s_pg[(ptrdiff_t)ev[a] * 3];
                rgx[a] = q[0]; rgy[a] = q[1]; rgz[a] = q[2];
            }
            Hex8RhieChowT<double> rc;
            rc.x = ex; rc.y = ey; rc.z = ez;
            rc.pgx = rgx; rc.pgy = rgy; rc.pgz = rgz; rc.scale = rc_scale;
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM)
                cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex, ey, ez, ux, uy, uz, pe, re, rc);
            else
                cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj_e, det[e], ux, uy, uz, pe, re, rc);
        } else if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex, ey, ez, ux, uy, uz, pe, re);
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
template <int FLUSH, int GEOM>
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
        double *const __restrict__ ghost_buf,
        const double    *const __restrict__ px,
        const double    *const __restrict__ py,
        const double    *const __restrict__ pz) {
    extern __shared__ double smem[];

    const ptrdiff_t p            = blockIdx.x;
    const ptrdiff_t owned        = owned_nodes_ptr[p];
    const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned;
    const ptrdiff_t gbegin       = ghost_ptr[p];
    const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - gbegin;
    const ptrdiff_t total_nodes  = n_contiguous + n_ghost;
    const ptrdiff_t stride       = (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;

    constexpr bool NEED_XYZ = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM);

    double *const s_u   = smem;
    double *const s_v   = smem + stride;
    double *const s_out = smem + 2 * stride;
    double *const s_xyz = smem + 3 * stride;

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
        if (NEED_XYZ) {
            double *const c = &s_xyz[i * 3];
            c[0] = px[g]; c[1] = py[g]; c[2] = pz[g];
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
        double re[CVFEM_HEX8_N_DOF];
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            double ex[8], ey[8], ez[8];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const double *const c = &s_xyz[(ptrdiff_t)ev[a] * 3];
                ex[a] = c[0]; ey[a] = c[1]; ez[a] = c[2];
            }
            cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, ex, ey, ez, ux, uy, uz,
                                                          vx, vy, vz, q, re);
        } else {
            double adj_e[9];
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
            cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj_e, det[e], ux, uy, uz,
                                                 vx, vy, vz, q, re);
        }
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

// --------------------------------------------------- packed-mesh assembly
//
// Assembly on the packed mesh, for comparison against the element-parallel form that
// uses global ids. A pack-local BSR cannot be staged -- one element alone produces 64
// blocks x 16 doubles = 8 KiB and a pack holds thousands of blocks -- so what is staged
// is the *read* side: the pack's fields go into shared memory once and the element loop
// gathers them through the packed mesh's uint16 local ids. The write side is unchanged,
// straight into the global BSR through element_slots with atomicAdd.
//
// That makes the comparison a clean one. Both forms write identically, so the difference
// measures exactly what the packed mesh addresses: how the fields are read.
template <int VARIANT, int GEOM>
__global__ void cvfem_hex8_assemble_packed_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const double rho, const double mu,
        const uint16_t  *const __restrict__ elems,
        const ptrdiff_t *const __restrict__ owned_nodes_ptr,
        const ptrdiff_t *const __restrict__ ghost_ptr,
        const int32_t   *const __restrict__ ghost_idx,
        const int32_t   *const __restrict__ slots,
        const double    *const __restrict__ adj,
        const double    *const __restrict__ det,
        const double    *const __restrict__ u,
        double *const __restrict__ values,
        const double    *const __restrict__ px,
        const double    *const __restrict__ py,
        const double    *const __restrict__ pz) {
    extern __shared__ double smem[];
    constexpr bool ISO = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM);

    const ptrdiff_t p            = blockIdx.x;
    const ptrdiff_t owned        = owned_nodes_ptr[p];
    const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned;
    const ptrdiff_t gbegin       = ghost_ptr[p];
    const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - gbegin;
    const ptrdiff_t total_nodes  = n_contiguous + n_ghost;

    double *const s_u   = smem;
    double *const s_xyz = smem + (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;

    for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
        const ptrdiff_t g = (i < n_contiguous) ? (owned + i)
                                               : (ptrdiff_t)ghost_idx[gbegin + i - n_contiguous];
        const double *const src = &u[g * CVFEM_CUDA_NF];
        double *const       dst = &s_u[i * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] = src[f];
        if (ISO) {
            double *const c = &s_xyz[i * 3];
            c[0] = px[g]; c[1] = py[g]; c[2] = pz[g];
        }
    }
    __syncthreads();

    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
        double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const uint16_t l  = elems[(ptrdiff_t)a * nelements + e];
            const double *const nd = &s_u[(ptrdiff_t)l * CVFEM_CUDA_NF];
            ux[a] = nd[0]; uy[a] = nd[1]; uz[a] = nd[2]; pe[a] = nd[3];
            if (ISO) {
                const double *const c = &s_xyz[(ptrdiff_t)l * 3];
                ex[a] = c[0]; ey[a] = c[1]; ez[a] = c[2];
            }
        }
        const int32_t *const es = &slots[e * 64];
        if constexpr (ISO) {
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                    rho, mu, ex, ey, ez, ux, uy, uz, es, values);
        } else {
            double adj_e[9];
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
            if constexpr (VARIANT == CVFEM_CUDA_JAC_HANDWRITTEN)
                cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
            else
                cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        }
        (void)pe;
    }
}

template <int VARIANT, int GEOM>
int launch_assemble_packed(cvfem_cuda_ctx *ctx, double rho, double mu, int block_size,
                           cudaStream_t s) {
    if (!ctx->values || !ctx->element_slots) return 1;
    if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM && !ctx->px) return 1;
    const size_t shmem = (size_t)ctx->max_pack_nodes *
                         (CVFEM_CUDA_NF + (GEOM == CVFEM_CUDA_GEOM_ISOPARAM ? 3 : 0)) *
                         sizeof(double);
    if (shmem > 48u * 1024u)
        CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                cvfem_hex8_assemble_packed_kernel<VARIANT, GEOM>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    const int block = block_size > 0 ? block_size : 128;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    cvfem_hex8_assemble_packed_kernel<VARIANT, GEOM><<<(int)ctx->n_packs, block, shmem, s>>>(
            ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
            ctx->owned_nodes_ptr, ctx->ghost_ptr, ctx->ghost_idx, ctx->element_slots,
            ctx->adj, ctx->det, ctx->u, ctx->values, ctx->px, ctx->py, ctx->pz);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ------------------------------------------- standard-mesh matrix-free baseline
//
// The same residual and J*v computed WITHOUT the packed mesh: one thread per element,
// grid-stride, global node ids, accumulating straight into the global vector with
// atomicAdd. No packs, no shared memory, no ghost machinery -- this is what the
// operators look like on an ordinary element->node connectivity, and it is the
// baseline the block-per-pack kernels have to beat.
//
// The CPU has had this comparison all along, because its `atomic` layout is exactly
// this and its `packed` layout is the pack-based one. The device had only the packed
// form, so what the format was worth here had never been measured.
template <int GEOM>
__global__ void cvfem_hex8_residual_global_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ r,
        const double  *const __restrict__ px,
        const double  *const __restrict__ py,
        const double  *const __restrict__ pz) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        int32_t gid[CVFEM_HEX8_N_NODES];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double  uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
        double  ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            gid[a]          = g;
            const double *const nd = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = nd[0]; uy[a] = nd[1]; uz[a] = nd[2]; pe[a] = nd[3];
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) { ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g]; }
        }

        double re[CVFEM_HEX8_N_DOF];
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex, ey, ez, ux, uy, uz, pe, re);
        } else {
            double adj_e[9];
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
            cvfem_hex8_ns_upwind_residual(rho, mu, adj_e, det[e], ux, uy, uz, pe, re);
        }

#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            double *const dst = &r[(ptrdiff_t)gid[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], re[a * 4 + f]);
        }
    }
}

template <int GEOM>
__global__ void cvfem_hex8_jacobian_action_global_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        const double  *const __restrict__ vin,
        double *const __restrict__ r,
        const double  *const __restrict__ px,
        const double  *const __restrict__ py,
        const double  *const __restrict__ pz) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        int32_t gid[CVFEM_HEX8_N_NODES];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
        double  vx[CVFEM_HEX8_N_NODES], vy[CVFEM_HEX8_N_NODES], vz[CVFEM_HEX8_N_NODES];
        double  q[CVFEM_HEX8_N_NODES];
        double  ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            gid[a]          = g;
            const double *const nu = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            const double *const nv = &vin[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = nu[0]; uy[a] = nu[1]; uz[a] = nu[2];
            vx[a] = nv[0]; vy[a] = nv[1]; vz[a] = nv[2]; q[a] = nv[3];
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) { ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g]; }
        }

        double re[CVFEM_HEX8_N_DOF];
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, ex, ey, ez, ux, uy, uz,
                                                          vx, vy, vz, q, re);
        } else {
            double adj_e[9];
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
            cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj_e, det[e], ux, uy, uz,
                                                 vx, vy, vz, q, re);
        }

#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            double *const dst = &r[(ptrdiff_t)gid[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], re[a * 4 + f]);
        }
    }
}

template <int GEOM, bool JV>
int launch_global_mf(cvfem_cuda_ctx *ctx, double rho, double mu, int block_size, cudaStream_t s) {
    if (!ctx->elements_global) return 1;
    if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM && !ctx->px) return 1;
    if (JV && !ctx->v) return 1;
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->r, 0,
                                     (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double), s));
    if (JV)
        cvfem_hex8_jacobian_action_global_kernel<GEOM><<<grid, block, 0, s>>>(
                ctx->nelements, rho, mu, ctx->elements_global, ctx->adj, ctx->det,
                ctx->u, ctx->v, ctx->r, ctx->px, ctx->py, ctx->pz);
    else
        cvfem_hex8_residual_global_kernel<GEOM><<<grid, block, 0, s>>>(
                ctx->nelements, rho, mu, ctx->elements_global, ctx->adj, ctx->det,
                ctx->u, ctx->r, ctx->px, ctx->py, ctx->pz);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ---------------------------------------------------------------- assembly

// Element-parallel, grid-stride, writing straight into the global BSR with atomicAdd.
// Both kernel families already accumulate through CVFEM_ATOMIC_ADD, which expands to
// atomicAdd under __CUDA_ARCH__, so the same source serves host and device.
template <int VARIANT, int GEOM = CVFEM_CUDA_GEOM_AFFINE, int PART = CVFEM_HEX8_PART_ALL>
__global__ void cvfem_hex8_assemble_bsr_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values,
        const double  *const __restrict__ px = nullptr,
        const double  *const __restrict__ py = nullptr,
        const double  *const __restrict__ pz = nullptr) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
        double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g    = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2]; pe[a] = n[3];
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) { ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g]; }
        }
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            const int32_t *const es = &slots[(ptrdiff_t)e * 64];
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true, PART>(
                    rho, mu, ex, ey, ez, ux, uy, uz, es, values);
            continue;
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

// Same element kernel as the atomic version, but writing with a plain += because the
// colouring guarantees no two threads in flight touch the same matrix block. VARIANT 0
// is the hand-written kernel with Atomic=false; the SymPy *_local_slots family already
// accumulates without atomics.
template <int VARIANT>
__global__ void cvfem_hex8_assemble_ecolored_kernel(
        const ptrdiff_t n_in_color, const ptrdiff_t color_begin, const ptrdiff_t nelements,
        const double rho, const double mu,
        const int32_t *const __restrict__ order,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values,
        const double  *const __restrict__ px = nullptr,
        const double  *const __restrict__ py = nullptr,
        const double  *const __restrict__ pz = nullptr) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_in_color;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t e = order[color_begin + t];
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
        if constexpr (VARIANT == CVFEM_CUDA_JAC_ISOPARAM) {
            // Colouring makes the writes race-free, so this accumulates without atomics
            // exactly as the affine coloured variants do.
            double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const ptrdiff_t g = elements[(ptrdiff_t)a * nelements + e];
                ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
            }
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                    rho, mu, ex, ey, ez, ux, uy, uz, es, values);
            (void)pe;
            continue;
        }
        if      constexpr (VARIANT == CVFEM_CUDA_JAC_HANDWRITTEN)
            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_BLOCK)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_ROW)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        (void)pe;
    }
}

template <int VARIANT>
int launch_ecolored_v(cvfem_cuda_ctx *ctx, double rho, double mu, int block, cudaStream_t s) {
    for (int c = 0; c < ctx->n_ecolors; ++c) {
        const ptrdiff_t b = ctx->h_ecolor_ptr[c], e = ctx->h_ecolor_ptr[c + 1];
        const ptrdiff_t n = e - b;
        if (n <= 0) continue;
        const int grid = (int)((n + block - 1) / block);
        cvfem_hex8_assemble_ecolored_kernel<VARIANT><<<grid, block, 0, s>>>(
                n, b, ctx->nelements, rho, mu, ctx->element_order, ctx->elements_global,
                ctx->element_slots, ctx->adj, ctx->det, ctx->u, ctx->values);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

// Isoparametric split. The viscous half depends only on geometry and mu, so it is built
// once; the convective half is what each Newton step rebuilds. Same decomposition as the
// affine split, but selected out of one kernel body rather than derived separately.
template <int PART>
int launch_assemble_isoparam_part(cvfem_cuda_ctx *ctx, double rho, double mu, double *dst,
                                  bool zero_first, int block_size, cudaStream_t s) {
    if (!ctx->px) return 1;
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    if (zero_first)
        CVFEM_CUDA_CHECK(cudaMemsetAsync(dst, 0, (size_t)ctx->nnz * 16 * sizeof(double), s));
    cvfem_hex8_assemble_bsr_kernel<CVFEM_CUDA_JAC_HANDWRITTEN, CVFEM_CUDA_GEOM_ISOPARAM, PART>
            <<<grid, block, 0, s>>>(
                    ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
                    ctx->adj, ctx->det, ctx->u, dst, ctx->px, ctx->py, ctx->pz);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

int launch_ecolored_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu, int block_size,
                             cudaStream_t s) {
    if (!ctx->px) return 1;
    const int block = block_size > 0 ? block_size : 128;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    for (int c = 0; c < ctx->n_ecolors; ++c) {
        const ptrdiff_t b = ctx->h_ecolor_ptr[c], n = ctx->h_ecolor_ptr[c + 1] - b;
        if (n <= 0) continue;
        const int grid = (int)((n + block - 1) / block);
        cvfem_hex8_assemble_ecolored_kernel<CVFEM_CUDA_JAC_ISOPARAM><<<grid, block, 0, s>>>(
                n, b, ctx->nelements, rho, mu, ctx->element_order, ctx->elements_global,
                ctx->element_slots, ctx->adj, ctx->det, ctx->u, ctx->values,
                ctx->px, ctx->py, ctx->pz);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

int launch_ecolored(cvfem_cuda_ctx *ctx, double rho, double mu, int variant,
                    int block_size, cudaStream_t s) {
    if (!ctx->values || !ctx->element_order) return 1;
    const int block = block_size > 0 ? block_size : 128;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    switch (variant) {
        case CVFEM_CUDA_JAC_HANDWRITTEN: return launch_ecolored_v<CVFEM_CUDA_JAC_HANDWRITTEN>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY:       return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_BLOCK: return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY_BLOCK>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_ROW:   return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY_ROW>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_FACE:  return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY_FACE>(ctx, rho, mu, block, s);
        default: return 1;
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

// Isoparametric assembly. Element-parallel like the affine path, but the coordinates are
// gathered straight from global memory rather than staged: assembly is already
// element-parallel with no pack structure to hang shared memory off, and the 24 extra
// doubles per element are read once against the 64 blocks x 16 doubles it writes.
int launch_assemble_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu, int block_size,
                             cudaStream_t s) {
    if (!ctx->px) return 1;  // coordinates were never uploaded
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    cvfem_hex8_assemble_bsr_kernel<CVFEM_CUDA_JAC_HANDWRITTEN, CVFEM_CUDA_GEOM_ISOPARAM>
            <<<grid, block, 0, s>>>(
                    ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
                    ctx->adj, ctx->det, ctx->u, ctx->values, ctx->px, ctx->py, ctx->pz);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

template <int GEOM>
int launch_jacobian_action_geom(cvfem_cuda_ctx *ctx, double rho, double mu, int flush_mode,
                                int block_size, cudaStream_t s) {
    constexpr bool ISO   = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM);
    const size_t   shmem = ISO ? ctx->iso_jv_shmem_bytes : ctx->jv_shmem_bytes;
    if (!ctx->v) return 1;
    if (ISO && !ctx->px) return 1;  // coordinates were never uploaded

    bool &done = ISO ? ctx->iso_jv_optin_done : ctx->jv_optin_done;
    if (!done) {
        if (shmem > 48u * 1024u) {
            CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, GEOM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
            CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, GEOM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
        }
        done = true;
    }
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)ctx->n_packs;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->r, 0,
                                     (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double), s));
    if (flush_mode == CVFEM_CUDA_FLUSH_TWO_PASS) {
        cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, GEOM>
                <<<grid, block, shmem, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->v, ctx->r, ctx->ghost_buf,
                        ctx->px, ctx->py, ctx->pz);
        CVFEM_CUDA_CHECK(cudaGetLastError());
        if (ctx->n_ghost_reduce_rows > 0) {
            const int rb = 256, rg = (int)((ctx->n_ghost_reduce_rows + rb - 1) / rb);
            cvfem_hex8_ghost_reduce_kernel<<<rg, rb, 0, s>>>(
                    ctx->n_ghost_reduce_rows, ctx->ghost_reduce_ptr, ctx->ghost_reduce_idx,
                    ctx->ghost_reduce_dest, ctx->ghost_buf, ctx->r);
            CVFEM_CUDA_CHECK(cudaGetLastError());
        }
    } else {
        cvfem_hex8_jacobian_action_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, GEOM>
                <<<grid, block, shmem, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->v, ctx->r, ctx->ghost_buf,
                        ctx->px, ctx->py, ctx->pz);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

int launch_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu, int flush_mode,
                           int block_size, cudaStream_t s) {
    return launch_jacobian_action_geom<CVFEM_CUDA_GEOM_AFFINE>(ctx, rho, mu, flush_mode,
                                                               block_size, s);
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

// Geometry-only assembly. Reads no velocity; run once per mesh.
__global__ void cvfem_hex8_assemble_linear_kernel(
        const ptrdiff_t nelements, const double mu,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_linear<true>(mu, adj_e, det[e],
                                                             &slots[e * 64], values);
    }
}

// Velocity-dependent assembly, added on top of the restored linear part.
__global__ void cvfem_hex8_assemble_nonlinear_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g     = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_nonlinear<true>(rho, mu, adj_e, det[e],
                                                                ux, uy, uz,
                                                                &slots[e * 64], values);
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

// Gather the touched blocks out of the full linear matrix into the compact buffer, once.
__global__ void cvfem_hex8_compact_linear_kernel(
        const ptrdiff_t n_blocks,
        const int32_t *const __restrict__ block_ids,
        const double  *const __restrict__ src,
        double *const __restrict__ compact) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_blocks * 16;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t b = t >> 4, k = t & 15;
        compact[t] = src[(ptrdiff_t)block_ids[b] * 16 + k];
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

// The viscous half for the pairs that are written once and never revisited.
__global__ void cvfem_hex8_assemble_static_kernel(
        const ptrdiff_t nelements, const double mu,
        const int32_t  *const __restrict__ slots,
        const uint16_t *const __restrict__ masks,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_static<true>(
                mu, adj_e, det[e], &slots[e * 64], masks, values);
    }
}

// Everything that is rebuilt each iteration: the viscous half for the recomputed pairs,
// plus convection. Runs into blocks that were just zeroed.
__global__ void cvfem_hex8_assemble_dynamic_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t  *const __restrict__ elements,
        const int32_t  *const __restrict__ slots,
        const uint16_t *const __restrict__ masks,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g     = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_dynamic<true>(rho, mu, adj_e, det[e],
                                                              ux, uy, uz, &slots[e * 64],
                                                              masks, values);
    }
}

// DIAG_MODE: 0 = everything, 1 = viscous only (constant), 2 = velocity-dependent only.
template <int DIAG_MODE>
__global__ void cvfem_hex8_assemble_diag_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ diag,
        const double  *const __restrict__ px = nullptr,
        const double  *const __restrict__ py = nullptr,
        const double  *const __restrict__ pz = nullptr) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        // -1 everywhere off the diagonal: those writes are dropped, so only the 8
        // diagonal blocks of this element are touched.
        int32_t sl[64];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
#pragma unroll
            for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) sl[a * 8 + b] = -1;
            sl[a * 8 + a] = g;
            const double *const n = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2];
        }
        if constexpr (DIAG_MODE == 3) {
            // Isoparametric: same negative-slot trick, geometry rebuilt per element.
            double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const int32_t g = elements[(ptrdiff_t)a * nelements + e];
                ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
            }
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                    rho, mu, ex, ey, ez, ux, uy, uz, sl, diag);
            continue;
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        if      constexpr (DIAG_MODE == 0)
            cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj_e, det[e], ux, uy, uz, sl, diag);
        else if constexpr (DIAG_MODE == 1)
            cvfem_hex8_ns_upwind_jacobian_add_slots_linear<true>(mu, adj_e, det[e], sl, diag);
        else
            cvfem_hex8_ns_upwind_jacobian_add_slots_nonlinear<true>(rho, mu, adj_e, det[e],
                                                                    ux, uy, uz, sl, diag);
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

template <int GEOM>
int launch_residual_geom(cvfem_cuda_ctx *ctx, double rho, double mu, int flush_mode,
                         int block_size, cudaStream_t s) {
    constexpr bool ISO = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM);
    const size_t   shmem = ISO ? ctx->iso_shmem_bytes : ctx->shmem_bytes;

    bool &done = ISO ? ctx->iso_optin_done : ctx->shmem_optin_done;
    if (!done) {
        // Anything above the 48 KB default needs an explicit opt-in per kernel.
        if (shmem > 48u * 1024u) {
            CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, false, GEOM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
            CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, false, GEOM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
        }
        done = true;
    }
    if (ISO && !ctx->px) return 1;  // coordinates were never uploaded

    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)ctx->n_packs;

    // Both modes accumulate into r, so it must start at zero.
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->r, 0,
                                     (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double), s));

    if (flush_mode == CVFEM_CUDA_FLUSH_TWO_PASS) {
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, false, GEOM>
                <<<grid, block, shmem, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->r, ctx->ghost_buf,
                        ctx->px, ctx->py, ctx->pz, nullptr, nullptr, nullptr, 0.0);
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
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, false, GEOM>
                <<<grid, block, shmem, s>>>(
                        ctx->nelements, ctx->n_elements_per_pack, rho, mu, ctx->elems,
                        ctx->owned_nodes_ptr, ctx->n_shared, ctx->ghost_ptr, ctx->ghost_idx,
                        ctx->adj, ctx->det, ctx->u, ctx->r, ctx->ghost_buf,
                        ctx->px, ctx->py, ctx->pz, nullptr, nullptr, nullptr, 0.0);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

int launch_residual(cvfem_cuda_ctx *ctx, double rho, double mu, int flush_mode,
                    int block_size, cudaStream_t s) {
    return launch_residual_geom<CVFEM_CUDA_GEOM_AFFINE>(ctx, rho, mu, flush_mode, block_size, s);
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

// Isoparametric stages the node coordinates as well: 64 B/node becomes 88 B/node.
extern "C" size_t cvfem_cuda_residual_isoparam_shmem_bytes(ptrdiff_t max_pack_nodes) {
    return cvfem_cuda_residual_shmem_bytes(max_pack_nodes)
           + (size_t)3 * (size_t)max_pack_nodes * sizeof(double);
}

// 96 B/node becomes 120 B/node, which is what caps the isoparametric pack size.
extern "C" size_t cvfem_cuda_jacobian_action_isoparam_shmem_bytes(ptrdiff_t max_pack_nodes) {
    return cvfem_cuda_jacobian_action_shmem_bytes(max_pack_nodes)
           + (size_t)3 * (size_t)max_pack_nodes * sizeof(double);
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
    c->iso_shmem_bytes = cvfem_cuda_residual_isoparam_shmem_bytes(max_pack_nodes);
    c->iso_jv_shmem_bytes = cvfem_cuda_jacobian_action_isoparam_shmem_bytes(max_pack_nodes);

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
    cudaFree(ctx->rowptr); cudaFree(ctx->colidx); cudaFree(ctx->element_order);
    cudaFree(ctx->values_linear); cudaFree(ctx->nl_blocks); cudaFree(ctx->linear_compact);
    cudaFree(ctx->diag); cudaFree(ctx->diag_static);
    cudaFree(ctx->nl_masks);
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

// ---- isoparametric geometry -------------------------------------------------
//
// The element kernels were already __host__ __device__ and templated after the
// portability phase, so these entry points wire up the geometry the device did not yet
// have rather than introducing new math. Call cvfem_cuda_attach_coords first.

// ---- packed-mesh assembly, for comparison against the element-parallel form -------
//
// `variant` takes CVFEM_CUDA_JAC_HANDWRITTEN or CVFEM_CUDA_JAC_SYMPY; `geom` 0 or 1.
extern "C" int cvfem_cuda_assemble_packed(cvfem_cuda_ctx *ctx, double rho, double mu,
                                          int variant, int geom, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    if (geom == CVFEM_CUDA_GEOM_ISOPARAM)
        return launch_assemble_packed<CVFEM_CUDA_JAC_HANDWRITTEN, CVFEM_CUDA_GEOM_ISOPARAM>(
                ctx, rho, mu, block_size, s);
    if (variant == CVFEM_CUDA_JAC_SYMPY)
        return launch_assemble_packed<CVFEM_CUDA_JAC_SYMPY, CVFEM_CUDA_GEOM_AFFINE>(
                ctx, rho, mu, block_size, s);
    return launch_assemble_packed<CVFEM_CUDA_JAC_HANDWRITTEN, CVFEM_CUDA_GEOM_AFFINE>(
            ctx, rho, mu, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_packed(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                  int variant, int geom, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (cvfem_cuda_assemble_packed(ctx, rho, mu, variant, geom, block_size, nullptr) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (cvfem_cuda_assemble_packed(ctx, rho, mu, variant, geom, block_size, nullptr) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

// ---- standard-mesh matrix-free baseline, for comparison against the packed form ----

extern "C" int cvfem_cuda_residual_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                          int geom, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return geom == CVFEM_CUDA_GEOM_ISOPARAM
                   ? launch_global_mf<CVFEM_CUDA_GEOM_ISOPARAM, false>(ctx, rho, mu, block_size, s)
                   : launch_global_mf<CVFEM_CUDA_GEOM_AFFINE, false>(ctx, rho, mu, block_size, s);
}

extern "C" int cvfem_cuda_jacobian_action_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                 int geom, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return geom == CVFEM_CUDA_GEOM_ISOPARAM
                   ? launch_global_mf<CVFEM_CUDA_GEOM_ISOPARAM, true>(ctx, rho, mu, block_size, s)
                   : launch_global_mf<CVFEM_CUDA_GEOM_AFFINE, true>(ctx, rho, mu, block_size, s);
}

static double time_global_mf(cvfem_cuda_ctx *ctx, double rho, double mu, int geom, bool jv,
                             int block_size, int repeat) {
    auto once = [&]() {
        return jv ? cvfem_cuda_jacobian_action_global(ctx, rho, mu, geom, block_size, nullptr)
                  : cvfem_cuda_residual_global(ctx, rho, mu, geom, block_size, nullptr);
    };
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (once() != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (once() != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" double cvfem_cuda_time_residual_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                  int geom, int block_size, int repeat) {
    return time_global_mf(ctx, rho, mu, geom, false, block_size, repeat);
}

extern "C" double cvfem_cuda_time_jacobian_action_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                         int geom, int block_size, int repeat) {
    return time_global_mf(ctx, rho, mu, geom, true, block_size, repeat);
}

extern "C" int cvfem_cuda_attach_coords(cvfem_cuda_ctx *ctx,
                                        const double *px, const double *py, const double *pz) {
    if (ctx->px) return 0;  // already uploaded, by this or by boundary_attach
    if (device_dup(&ctx->px, px, (size_t)ctx->nnodes) ||
        device_dup(&ctx->py, py, (size_t)ctx->nnodes) ||
        device_dup(&ctx->pz, pz, (size_t)ctx->nnodes))
        return 1;
    return 0;
}

extern "C" int cvfem_cuda_residual_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                            int flush_mode, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_residual_geom<CVFEM_CUDA_GEOM_ISOPARAM>(ctx, rho, mu, flush_mode, block_size, s);
}

extern "C" double cvfem_cuda_time_residual_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                    int flush_mode, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_residual_geom<CVFEM_CUDA_GEOM_ISOPARAM>(ctx, rho, mu, flush_mode, block_size, 0) != 0)
        return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_residual_geom<CVFEM_CUDA_GEOM_ISOPARAM>(ctx, rho, mu, flush_mode, block_size, 0) != 0)
            return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_jacobian_action_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                   int flush_mode, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_jacobian_action_geom<CVFEM_CUDA_GEOM_ISOPARAM>(ctx, rho, mu, flush_mode,
                                                                 block_size, s);
}

extern "C" double cvfem_cuda_time_jacobian_action_isoparam(cvfem_cuda_ctx *ctx, double rho,
                                                           double mu, int flush_mode,
                                                           int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_jacobian_action_geom<CVFEM_CUDA_GEOM_ISOPARAM>(ctx, rho, mu, flush_mode, block_size, 0) != 0)
        return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_jacobian_action_geom<CVFEM_CUDA_GEOM_ISOPARAM>(ctx, rho, mu, flush_mode, block_size, 0) != 0)
            return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_assemble_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                            int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_assemble_isoparam(ctx, rho, mu, block_size, s);
}

extern "C" int cvfem_cuda_assemble_ecolored_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                     int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_ecolored_isoparam(ctx, rho, mu, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_ecolored_isoparam(cvfem_cuda_ctx *ctx, double rho,
                                                             double mu, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_ecolored_isoparam(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_ecolored_isoparam(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

// Build the constant half once into values_linear.
extern "C" int cvfem_cuda_assemble_linear_isoparam(cvfem_cuda_ctx *ctx, double mu,
                                                   int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    if (!ctx->values) return 1;
    const size_t nb = (size_t)ctx->nnz * 16 * sizeof(double);
    if (!ctx->values_linear) CVFEM_CUDA_CHECK(cudaMalloc(&ctx->values_linear, nb));
    return launch_assemble_isoparam_part<CVFEM_HEX8_PART_LINEAR>(ctx, 0.0, mu, ctx->values_linear,
                                                                 true, block_size, s);
}

// Restore it and add only the velocity-dependent half.
extern "C" int cvfem_cuda_assemble_nonlinear_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                      int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    if (!ctx->values_linear) return 1;
    CVFEM_CUDA_CHECK(cudaMemcpyAsync(ctx->values, ctx->values_linear,
                                     (size_t)ctx->nnz * 16 * sizeof(double),
                                     cudaMemcpyDeviceToDevice, s));
    return launch_assemble_isoparam_part<CVFEM_HEX8_PART_NONLINEAR>(ctx, rho, mu, ctx->values,
                                                                    false, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_nonlinear_isoparam(cvfem_cuda_ctx *ctx, double rho,
                                                              double mu, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (cvfem_cuda_assemble_nonlinear_isoparam(ctx, rho, mu, block_size, nullptr) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (cvfem_cuda_assemble_nonlinear_isoparam(ctx, rho, mu, block_size, nullptr) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" double cvfem_cuda_time_assemble_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                    int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_assemble_isoparam(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_assemble_isoparam(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
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

extern "C" int cvfem_cuda_assemble_linear(cvfem_cuda_ctx *ctx, double mu, int block_size,
                                          void *stream) {
    if (!ctx->values) return 1;
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    const size_t nb = (size_t)ctx->nnz * 16 * sizeof(double);
    if (!ctx->values_linear) CVFEM_CUDA_CHECK(cudaMalloc(&ctx->values_linear, nb));
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values_linear, 0, nb, s));
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    cvfem_hex8_assemble_linear_kernel<<<grid, block, 0, s>>>(
            ctx->nelements, mu, ctx->element_slots, ctx->adj, ctx->det, ctx->values_linear);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_assemble_nonlinear(cvfem_cuda_ctx *ctx, double rho, double mu,
                                     int block_size, cudaStream_t s) {
    if (!ctx->values_linear) return 1;
    // Restore the constant part. This is a fully coalesced device-to-device copy, which
    // is a very different access pattern from the scattered accumulation it replaces.
    CVFEM_CUDA_CHECK(cudaMemcpyAsync(ctx->values, ctx->values_linear,
                                     (size_t)ctx->nnz * 16 * sizeof(double),
                                     cudaMemcpyDeviceToDevice, s));
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    cvfem_hex8_assemble_nonlinear_kernel<<<grid, block, 0, s>>>(
            ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
            ctx->adj, ctx->det, ctx->u, ctx->values);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cvfem_cuda_assemble_nonlinear(cvfem_cuda_ctx *ctx, double rho, double mu,
                                             int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_assemble_nonlinear(ctx, rho, mu, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_nonlinear(cvfem_cuda_ctx *ctx, double rho,
                                                     double mu, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_assemble_nonlinear(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_assemble_nonlinear(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_nonlinear_blocks_attach(cvfem_cuda_ctx *ctx, ptrdiff_t n_blocks,
                                                  const int32_t *block_ids,
                                                  const uint16_t *block_masks_by_id) {
    ctx->n_nl_blocks = n_blocks;
    if (device_dup(&ctx->nl_blocks, block_ids, (size_t)n_blocks) != 0) return 1;
    if (device_dup(&ctx->nl_masks, block_masks_by_id, (size_t)ctx->nnz) != 0) return 1;
    if (!ctx->values_linear) return 1;   // needs cvfem_cuda_assemble_linear first

    // Compact the saved linear data down to the blocks that will actually be
    // overwritten, then release the full-size copy. The other 73.5% of blocks are
    // already correct in `values` and are never written again, so nothing needs to hold
    // a second copy of them.
    CVFEM_CUDA_CHECK(cudaMalloc(&ctx->linear_compact, (size_t)n_blocks * 16 * sizeof(double)));
    const int block = 256;
    const int grid  = (int)((n_blocks * 16 + block - 1) / block);
    cvfem_hex8_compact_linear_kernel<<<grid, block>>>(n_blocks, ctx->nl_blocks,
                                                      ctx->values_linear, ctx->linear_compact);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    CVFEM_CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(ctx->values_linear);
    ctx->values_linear = nullptr;
    return 0;
}

extern "C" int cvfem_cuda_assemble_static(cvfem_cuda_ctx *ctx, double mu, int block_size,
                                          void *stream) {
    if (!ctx->values) return 1;
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    if (!ctx->nl_masks) return 1;
    cvfem_hex8_assemble_static_kernel<<<grid, block, 0, s>>>(
            ctx->nelements, mu, ctx->element_slots, ctx->nl_masks, ctx->adj, ctx->det,
            ctx->values);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_assemble_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                   int block_size, cudaStream_t s) {
    if (!ctx->nl_blocks) return 1;
    const int block = block_size > 0 ? block_size : 128;
    {
        const int grid = (int)((ctx->n_nl_blocks * 16 + block - 1) / block);
        cvfem_hex8_zero_blocks_kernel<<<grid, block, 0, s>>>(ctx->n_nl_blocks, ctx->nl_blocks,
                                                             ctx->nl_masks, ctx->values);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    const int grid = (int)((ctx->nelements + block - 1) / block);
    cvfem_hex8_assemble_dynamic_kernel<<<grid, block, 0, s>>>(
            ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
            ctx->nl_masks, ctx->adj, ctx->det, ctx->u, ctx->values);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cvfem_cuda_assemble_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                           int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_assemble_dynamic(ctx, rho, mu, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                   int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_assemble_dynamic(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_assemble_dynamic(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_diag_alloc(cvfem_cuda_ctx *ctx) {
    const size_t nb = (size_t)ctx->nnodes * 16 * sizeof(double);
    if (!ctx->diag) CVFEM_CUDA_CHECK(cudaMalloc(&ctx->diag, nb));
    return 0;
}

static int launch_diag(cvfem_cuda_ctx *ctx, double rho, double mu, int mode,
                       double *dst, int block_size, cudaStream_t s, bool zero_first) {
    if (cvfem_cuda_diag_alloc(ctx) != 0) return 1;
    const size_t nb = (size_t)ctx->nnodes * 16 * sizeof(double);
    if (zero_first) CVFEM_CUDA_CHECK(cudaMemsetAsync(dst, 0, nb, s));
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    switch (mode) {
        case 0: cvfem_hex8_assemble_diag_kernel<0><<<grid, block, 0, s>>>(
                    ctx->nelements, rho, mu, ctx->elements_global, ctx->adj, ctx->det, ctx->u, dst);
                break;
        case 1: cvfem_hex8_assemble_diag_kernel<1><<<grid, block, 0, s>>>(
                    ctx->nelements, rho, mu, ctx->elements_global, ctx->adj, ctx->det, ctx->u, dst);
                break;
        case 2: cvfem_hex8_assemble_diag_kernel<2><<<grid, block, 0, s>>>(
                    ctx->nelements, rho, mu, ctx->elements_global, ctx->adj, ctx->det, ctx->u, dst);
                break;
        default:  // isoparametric
                if (!ctx->px) return 1;
                cvfem_hex8_assemble_diag_kernel<3><<<grid, block, 0, s>>>(
                    ctx->nelements, rho, mu, ctx->elements_global, ctx->adj, ctx->det, ctx->u, dst,
                    ctx->px, ctx->py, ctx->pz);
                break;
    }
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cvfem_cuda_assemble_diag(cvfem_cuda_ctx *ctx, double rho, double mu,
                                        int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    // Allocate before reading ctx->diag: passing it as an argument would capture the
    // null pointer from before launch_diag's own allocation runs.
    if (cvfem_cuda_diag_alloc(ctx) != 0) return 1;
    return launch_diag(ctx, rho, mu, 0, ctx->diag, block_size, s, true);
}

extern "C" int cvfem_cuda_assemble_diag_static(cvfem_cuda_ctx *ctx, double mu, int block_size,
                                               void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    const size_t nb = (size_t)ctx->nnodes * 16 * sizeof(double);
    if (!ctx->diag_static) CVFEM_CUDA_CHECK(cudaMalloc(&ctx->diag_static, nb));
    return launch_diag(ctx, 0.0, mu, 1, ctx->diag_static, block_size, s, true);
}

static int launch_diag_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu, int block_size,
                               cudaStream_t s) {
    if (!ctx->diag_static) return 1;
    if (cvfem_cuda_diag_alloc(ctx) != 0) return 1;
    const ptrdiff_t n = ctx->nnodes * 16;
    const int block = block_size > 0 ? block_size : 256;
    const int grid  = (int)((n + block - 1) / block);
    cvfem_hex8_diag_restore_kernel<<<grid, block, 0, s>>>(n, ctx->diag_static, ctx->diag);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return launch_diag(ctx, rho, mu, 2, ctx->diag, block_size, s, false);
}

extern "C" int cvfem_cuda_assemble_diag_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_diag_dynamic(ctx, rho, mu, block_size, s);
}

extern "C" int cvfem_cuda_download_diag(cvfem_cuda_ctx *ctx, double *diag) {
    if (!ctx->diag) return 1;
    CVFEM_CUDA_CHECK(cudaMemcpy(diag, ctx->diag, (size_t)ctx->nnodes * 16 * sizeof(double),
                                cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int cvfem_cuda_invert_diag(cvfem_cuda_ctx *ctx, int block_size, void *stream) {
    if (!ctx->diag) return 1;
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nnodes + block - 1) / block);
    cvfem_hex8_invert_diag_kernel<<<grid, block, 0, s>>>(ctx->nnodes, nullptr, ctx->diag);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" double cvfem_cuda_time_assemble_diag(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (cvfem_cuda_assemble_diag(ctx, rho, mu, block_size, nullptr) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (cvfem_cuda_assemble_diag(ctx, rho, mu, block_size, nullptr) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_assemble_diag_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                 int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    if (cvfem_cuda_diag_alloc(ctx) != 0) return 1;
    return launch_diag(ctx, rho, mu, 3, ctx->diag, block_size, s, true);
}

extern "C" double cvfem_cuda_time_assemble_diag_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                         int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cvfem_cuda_diag_alloc(ctx) != 0) return -1.0;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_diag(ctx, rho, mu, 3, ctx->diag, block_size, 0, true) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_diag(ctx, rho, mu, 3, ctx->diag, block_size, 0, true) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}


extern "C" double cvfem_cuda_time_assemble_diag_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                        int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_diag_dynamic(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_diag_dynamic(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" size_t cvfem_cuda_linear_side_bytes(cvfem_cuda_ctx *ctx) {
    if (ctx->linear_compact) return (size_t)ctx->n_nl_blocks * 16 * sizeof(double);
    if (ctx->values_linear) return (size_t)ctx->nnz * 16 * sizeof(double);
    return 0;
}

static int launch_assemble_nonlinear_sparse(cvfem_cuda_ctx *ctx, double rho, double mu,
                                            int block_size, cudaStream_t s) {
    if (!ctx->linear_compact || !ctx->nl_blocks) return 1;
    const int block = block_size > 0 ? block_size : 128;
    {
        const ptrdiff_t work = ctx->n_nl_blocks * 4;   // one double4 per thread
        const int       grid = (int)((work + block - 1) / block);
        cvfem_hex8_restore_blocks_kernel<<<grid, block, 0, s>>>(
                ctx->n_nl_blocks, ctx->nl_blocks, ctx->linear_compact, ctx->values);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    const int grid = (int)((ctx->nelements + block - 1) / block);
    cvfem_hex8_assemble_nonlinear_kernel<<<grid, block, 0, s>>>(
            ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
            ctx->adj, ctx->det, ctx->u, ctx->values);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cvfem_cuda_assemble_nonlinear_sparse(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                    int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_assemble_nonlinear_sparse(ctx, rho, mu, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_nonlinear_sparse(cvfem_cuda_ctx *ctx, double rho,
                                                            double mu, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_assemble_nonlinear_sparse(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_assemble_nonlinear_sparse(ctx, rho, mu, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" double cvfem_cuda_time_restore_only(cvfem_cuda_ctx *ctx, int repeat) {
    if (!ctx->values_linear) return -1.0;
    const size_t nb = (size_t)ctx->nnz * 16 * sizeof(double);
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    cudaMemcpy(ctx->values, ctx->values_linear, nb, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        cudaMemcpyAsync(ctx->values, ctx->values_linear, nb, cudaMemcpyDeviceToDevice, 0);
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" double cvfem_cuda_time_nonlinear_only(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                 int block_size, int repeat) {
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        cvfem_hex8_assemble_nonlinear_kernel<<<grid, block>>>(
                ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
                ctx->adj, ctx->det, ctx->u, ctx->values);
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}

extern "C" int cvfem_cuda_element_coloring_attach(cvfem_cuda_ctx *ctx, int n_colors,
                                                 const int32_t *element_order,
                                                 const ptrdiff_t *color_ptr) {
    ctx->n_ecolors = n_colors;
    ctx->h_ecolor_ptr.assign(color_ptr, color_ptr + n_colors + 1);
    return device_dup(&ctx->element_order, element_order, (size_t)ctx->nelements);
}

extern "C" int cvfem_cuda_assemble_ecolored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                            int variant, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_ecolored(ctx, rho, mu, variant, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_ecolored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                    int variant, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_ecolored(ctx, rho, mu, variant, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_ecolored(ctx, rho, mu, variant, block_size, 0) != 0) return -1.0;
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
    if (device_dup(&ctx->boundary_elems, boundary_elems, (size_t)n_boundary)) return 1;
    return cvfem_cuda_attach_coords(ctx, px, py, pz);
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
                cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, true, CVFEM_CUDA_GEOM_AFFINE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need));
        CVFEM_CUDA_CHECK(cudaFuncSetAttribute(
                cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, true, CVFEM_CUDA_GEOM_AFFINE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need));
    }
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)ctx->n_packs;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->r, 0,
                                     (size_t)ctx->nnodes * CVFEM_CUDA_NF * sizeof(double), s));
    if (flush_mode == CVFEM_CUDA_FLUSH_TWO_PASS) {
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_TWO_PASS, true, CVFEM_CUDA_GEOM_AFFINE><<<grid, block, need, s>>>(
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
        cvfem_hex8_residual_pack_kernel<CVFEM_CUDA_FLUSH_ATOMIC, true, CVFEM_CUDA_GEOM_AFFINE><<<grid, block, need, s>>>(
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
