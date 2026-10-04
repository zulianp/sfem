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

#include "support/cvfem_default_types.hpp"

#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"

#include "kernels/microkernels/hex8/generated/cvfem_hex8_ns_upwind_sympy_kernels.hpp"
#ifdef CVFEM_ENABLE_SUBPAR
#include "cvfem_hex8_ns_upwind_sympy_subpar.hpp"
#endif
#include "kernels/microkernels/hex8/cvfem_hex8_boundary_scs.hpp"

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
    // Bit-reproducible flush: per-element scratch and the node->element CSR.
    double    *elem_r{nullptr};      // [32 * nelements]
    ptrdiff_t *n2e_ptr{nullptr};     // [nnodes + 1]
    int32_t   *n2e_enc{nullptr};     // element * 8 + local index
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
#include "kernels/microkernels/hex8/cuda/cvfem_hex8_ns_cuda_nodal.cuh"
#include "kernels/standard/cuda/cvfem_hex8_ns_cuda_standard.cuh"
#include "kernels/packed/cuda/cvfem_hex8_ns_cuda_packed.cuh"
#include "kernels/colored/cuda/cvfem_hex8_ns_cuda_ecolored.cuh"


template <typename T>
int device_dup(T **dst, const T *src, size_t n) {
    if (n == 0) { *dst = nullptr; return 0; }
    CVFEM_CUDA_CHECK(cudaMalloc(dst, n * sizeof(T)));
    CVFEM_CUDA_CHECK(cudaMemcpy(*dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
    return 0;
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

template <int GEOM>
int launch_residual_deterministic(cvfem_cuda_ctx *ctx, double rho, double mu, int block_size,
                                  cudaStream_t s) {
    if (!ctx->elements_global || !ctx->n2e_ptr) return 1;
    if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM && !ctx->px) return 1;
    if (!ctx->elem_r)
        CVFEM_CUDA_CHECK(cudaMalloc(&ctx->elem_r,
                                    (size_t)ctx->nelements * CVFEM_HEX8_N_DOF * sizeof(double)));
    const int block = block_size > 0 ? block_size : 128;
    const int egrid = (int)((ctx->nelements + block - 1) / block);
    const int ngrid = (int)((ctx->nnodes + block - 1) / block);
    cvfem_hex8_residual_store_kernel<GEOM><<<egrid, block, 0, s>>>(
            ctx->nelements, rho, mu, ctx->elements_global, ctx->adj, ctx->det, ctx->u,
            ctx->elem_r, ctx->px, ctx->py, ctx->pz);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    cvfem_hex8_residual_gather_kernel<<<ngrid, block, 0, s>>>(
            ctx->nnodes, ctx->n2e_ptr, ctx->n2e_enc, ctx->elem_r, ctx->r);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
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
        // Element colouring with the hand-written kernel is a measured loss: 166.6 MDOF/s
        // against 218.2 for the same kernel on the atomic path. Colouring buys the right
        // to accumulate with a plain `+=` instead of atomicAdd, and on this device that
        // is the wrong trade -- atomicAdd compiles to a fire-and-forget reduction while
        // `+=` has to wait on the load. The trade only pays for the fused kernels, where
        // there is enough arithmetic per write to hide the dependency: sympy_block gains
        // 18% (277.3 against 235.5) and is the fastest GPU assembly measured.
        //
        // Kept behind the subpar option because it is the evidence for that sentence.
#ifdef CVFEM_ENABLE_SUBPAR
        case CVFEM_CUDA_JAC_HANDWRITTEN: return launch_ecolored_v<CVFEM_CUDA_JAC_HANDWRITTEN>(ctx, rho, mu, block, s);
#endif
        case CVFEM_CUDA_JAC_SYMPY:       return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_BLOCK: return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY_BLOCK>(ctx, rho, mu, block, s);
#ifdef CVFEM_ENABLE_SUBPAR
        case CVFEM_CUDA_JAC_SYMPY_ROW:   return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY_ROW>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_FACE:  return launch_ecolored_v<CVFEM_CUDA_JAC_SYMPY_FACE>(ctx, rho, mu, block, s);
#endif
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
#ifdef CVFEM_ENABLE_SUBPAR
        case CVFEM_CUDA_JAC_SYMPY_ROW:   return launch_assemble_v<CVFEM_CUDA_JAC_SYMPY_ROW>(ctx, rho, mu, block, s);
        case CVFEM_CUDA_JAC_SYMPY_FACE:  return launch_assemble_v<CVFEM_CUDA_JAC_SYMPY_FACE>(ctx, rho, mu, block, s);
#endif
        default: return 1;
    }
}

// Pack-coloured assembly lives in subpar/cuda/cvfem_hex8_ns_cuda_colored.cuh: it is
// correct only with blockDim.x == 1 and is ~200x slower than the atomic path there.
// Build with -DCVFEM_ENABLE_SUBPAR to compile it. The ctx fields it uses (n_colors,
// pack_order, color_ptr, h_color_ptr) stay above -- four pointers, freed nullptr-safely
// by cvfem_cuda_destroy -- so that the quarantined file needs no change to the struct.
#ifdef CVFEM_ENABLE_SUBPAR
#include "cvfem_hex8_ns_cuda_colored.cuh"
#endif


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
    // Box domains with closed faces only, and the refusal is here because nothing else
    // would notice. Every boundary_scs_add_* call in this file passes neither face mask nor
    // boundary data, so it takes the bounding-box coordinate fallback (fmask = -1) and no
    // traction or prescribed pressure. On a domain with a re-entrant face that leaves those
    // control volumes unclosed, and with a named condition it drops it entirely -- in both
    // cases producing a converged, plausible, wrong answer rather than an error.
    //
    // SFEM_BOUNDARY_MASK=1 is exactly the caller saying "the coordinate test is not enough
    // for this domain", so it is the right thing to key on.
    if (const char *bm = std::getenv("SFEM_BOUNDARY_MASK"); bm && std::atoi(bm) != 0) {
        std::fprintf(stderr,
                     "cvfem_cuda_create: SFEM_BOUNDARY_MASK=1 is not supported on the CUDA "
                     "path -- its boundary kernels take no face mask and no boundary data, "
                     "so a non-box domain or a named traction/pressure condition would be "
                     "silently ignored. Use the CPU path.\n");
        return -1;
    }

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
    cudaFree(ctx->elem_r); cudaFree(ctx->n2e_ptr); cudaFree(ctx->n2e_enc);
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
#ifdef CVFEM_ENABLE_SUBPAR
        case CVFEM_CUDA_JAC_SYMPY_ROW:   return "sympy_row";
        case CVFEM_CUDA_JAC_SYMPY_FACE:  return "sympy_face";
#endif
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

// The standard-mesh matrix-free kernels need the global connectivity but not the matrix.
// bsr_attach uploads both together, which is fine when a matrix is wanted and wrong when
// the problem is too large to hold one -- exactly the sizes a saturation sweep reaches.
extern "C" int cvfem_cuda_attach_elements_global(cvfem_cuda_ctx *ctx, const int32_t *elements) {
    if (ctx->elements_global) return 0;
    return device_dup(&ctx->elements_global, elements, (size_t)8 * ctx->nelements);
}

extern "C" int cvfem_cuda_attach_node_to_element(cvfem_cuda_ctx *ctx,
                                                const ptrdiff_t *n2e_ptr,
                                                const int32_t *n2e_enc, ptrdiff_t n_entries) {
    if (ctx->n2e_ptr) return 0;
    return device_dup(&ctx->n2e_ptr, n2e_ptr, (size_t)ctx->nnodes + 1) ||
           device_dup(&ctx->n2e_enc, n2e_enc, (size_t)n_entries);
}

extern "C" int cvfem_cuda_residual_deterministic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                 int geom, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return geom == CVFEM_CUDA_GEOM_ISOPARAM
                   ? launch_residual_deterministic<CVFEM_CUDA_GEOM_ISOPARAM>(ctx, rho, mu, block_size, s)
                   : launch_residual_deterministic<CVFEM_CUDA_GEOM_AFFINE>(ctx, rho, mu, block_size, s);
}

extern "C" double cvfem_cuda_time_residual_deterministic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                         int geom, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (cvfem_cuda_residual_deterministic(ctx, rho, mu, geom, block_size, nullptr) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (cvfem_cuda_residual_deterministic(ctx, rho, mu, geom, block_size, nullptr) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
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

// The generated isoparametric kernel. Same geometry, CSE'd algebra with the twelve sets
// of reference shape derivatives folded in as literals rather than evaluated per element.
extern "C" int cvfem_cuda_assemble_isoparam_sympy(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                  int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    if (!ctx->px) return 1;
    const int block = block_size > 0 ? block_size : 128;
    const int grid  = (int)((ctx->nelements + block - 1) / block);
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    cvfem_hex8_assemble_bsr_kernel<CVFEM_CUDA_JAC_ISOPARAM_SYMPY, CVFEM_CUDA_GEOM_ISOPARAM>
            <<<grid, block, 0, s>>>(
                    ctx->nelements, rho, mu, ctx->elements_global, ctx->element_slots,
                    ctx->adj, ctx->det, ctx->u, ctx->values, ctx->px, ctx->py, ctx->pz);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" double cvfem_cuda_time_assemble_isoparam_sympy(cvfem_cuda_ctx *ctx, double rho,
                                                          double mu, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (cvfem_cuda_assemble_isoparam_sympy(ctx, rho, mu, block_size, nullptr) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (cvfem_cuda_assemble_isoparam_sympy(ctx, rho, mu, block_size, nullptr) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
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

// Times the Rhie-Chow residual. Its absence is why every device throughput number in
// README_alps.md was for a kernel without the Rhie-Chow term while every host number
// included it: cvfem_cuda_residual_rc existed but was only ever verified, never timed,
// and none of the other cvfem_cuda_time_* entry points take rc_scale.
extern "C" double cvfem_cuda_time_residual_rc(cvfem_cuda_ctx *ctx, double rho, double mu,
                                              double rc_scale, int flush_mode, int block_size,
                                              int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (cvfem_cuda_residual_rc(ctx, rho, mu, rc_scale, flush_mode, block_size, nullptr) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (cvfem_cuda_residual_rc(ctx, rho, mu, rc_scale, flush_mode, block_size, nullptr) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
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
