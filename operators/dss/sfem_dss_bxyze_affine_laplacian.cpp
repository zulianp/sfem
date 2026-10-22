#include "sfem_dss_bxyze_affine_laplacian.hpp"

#include "sfem_dss_bxyze_affine_laplacian_inline.hpp"

#ifdef SFEM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace {

template <int N, int Q, int B_EXEC, int B_SAVE = 0>
int volumetric_apply_tpl(const sfem_dss_bxyze_laplacian_affine_volumetric_t *const op,
                         const scalar_t *const S,
                         const scalar_t *const D,
                         const scalar_t *const qw,
                         const scalar_t *const geometry,
                         const real_t *const x,
                         real_t *const y,
                         scalar_t *const workspace) {
    const auto &layout = op->layout;
    constexpr ptrdiff_t ngrad = (ptrdiff_t)Q * Q * Q * B_EXEC;
    static_assert(B_SAVE == 0 || (B_SAVE >= B_EXEC && B_SAVE % B_EXEC == 0), "Invalid storage batch");
    // B_SAVE=0 preserves the general path for storage sizes outside the setup table.
    const int bsave = B_SAVE ? B_SAVE : layout.element_block_size;
    const ptrdiff_t block_stride = B_SAVE ? (ptrdiff_t)B_SAVE * N * N * N : layout.block_stride;
#pragma omp parallel
    {
        int tid = 0;
#ifdef SFEM_ENABLE_OPENMP
        tid = omp_get_thread_num();
#endif
        scalar_t *const gx = workspace + (ptrdiff_t)tid * op->workspace_stride;
        scalar_t *const gy = gx + ngrad;
        scalar_t *const gz = gy + ngrad;
        scalar_t *const scratch = gz + ngrad;

#pragma omp for schedule(static)
        for (ptrdiff_t eb = 0; eb < layout.nblocks; ++eb) {
            for (int lane = 0; lane < bsave; lane += B_EXEC) {
                const real_t *const xb = x + eb * block_stride + lane;
                sfem_dss_bxyze_lagrange_gradient<N, Q, B_EXEC>(bsave, S, D, xb, gx, gy, gz, scratch);
                const scalar_t *const h = geometry + eb * ((ptrdiff_t)6 * bsave) + lane;
                sfem_dss_bxyze_affine_metric_apply<Q, B_EXEC>(bsave, qw, h, gx, gy, gz);
                real_t *const yb = y + eb * block_stride + lane;
                sfem_dss_bxyze_lagrange_gradient_transpose<N, Q, B_EXEC>(bsave, S, D, gx, gy, gz, yb, scratch);
            }
        }
    }
    return SFEM_SUCCESS;
}

template <int N, int Q, int B_EXEC>
sfem_dss_bxyze_laplacian_affine_volumetric_kernel_t select_storage_kernel(const int bs) {
    switch (bs) {
        case 1: if constexpr (B_EXEC <= 1) return volumetric_apply_tpl<N, Q, B_EXEC, 1>; break;
        case 2: if constexpr (B_EXEC <= 2) return volumetric_apply_tpl<N, Q, B_EXEC, 2>; break;
        case 4: if constexpr (B_EXEC <= 4) return volumetric_apply_tpl<N, Q, B_EXEC, 4>; break;
        case 8: if constexpr (B_EXEC <= 8) return volumetric_apply_tpl<N, Q, B_EXEC, 8>; break;
        case 16: return volumetric_apply_tpl<N, Q, B_EXEC, 16>;
        case 32: return volumetric_apply_tpl<N, Q, B_EXEC, 32>;
        case 64: return volumetric_apply_tpl<N, Q, B_EXEC, 64>;
        default: return volumetric_apply_tpl<N, Q, B_EXEC>;
    }
    return nullptr;
}

template <int N, int Q>
sfem_dss_bxyze_laplacian_affine_volumetric_kernel_t select_batch_kernel(const int bs, const int be) {
    switch (be) {
        case 1: return select_storage_kernel<N, Q, 1>(bs);
        case 2: return select_storage_kernel<N, Q, 2>(bs);
        case 4: return select_storage_kernel<N, Q, 4>(bs);
        case 8: return select_storage_kernel<N, Q, 8>(bs);
        case 16: return select_storage_kernel<N, Q, 16>(bs);
        default: return nullptr;
    }
}

template <int N>
sfem_dss_bxyze_laplacian_affine_volumetric_kernel_t select_quadrature_kernel(const int Q, const int bs, const int be) {
    if (Q == N) return select_batch_kernel<N, N>(bs, be);
    if (Q == 3 * (N - 1)) return select_batch_kernel<N, 3 * (N - 1)>(bs, be);
    return nullptr;
}

}  // namespace

ptrdiff_t sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(
        const sfem_dss_bxyze_layout_t *const layout, const int Q) {
    if (!layout || (layout->level != 2 && layout->level != 4 && layout->level != 8)) return -1;
    sfem_dss_bxyze_layout_t expected;
    if (sfem_dss_bxyze_create_layout(layout->level, layout->nelements, layout->element_block_size,
                                    layout->execution_block_size, &expected) != SFEM_SUCCESS ||
        layout->n != expected.n || layout->block_stride != expected.block_stride ||
        layout->execution_stride != expected.execution_stride || layout->nblocks != expected.nblocks ||
        layout->chunks_per_block != expected.chunks_per_block || layout->ntasks != expected.ntasks) return -1;

    const ptrdiff_t n = layout->n;
    const int bs = layout->element_block_size, be = layout->execution_block_size;
    if ((Q != n && Q != 3 * layout->level) ||
        (be != 1 && be != 2 && be != 4 && be != 8 && be != 16) ||
        bs % be != 0 || layout->nelements % bs != 0) return -1;

    const ptrdiff_t values = ((ptrdiff_t)3 * Q * Q * Q + 2 * n * n * Q + 3 * n * Q) * be;
    const ptrdiff_t gap = 128 / sizeof(scalar_t);
    return ((values + gap - 1) / gap + 1) * gap;
}

int sfem_dss_bxyze_laplacian_affine_volumetric_setup(
        const sfem_dss_bxyze_layout_t *const layout,
        const int Q,
        sfem_dss_bxyze_laplacian_affine_volumetric_t *const op) {
    if (!op) return SFEM_FAILURE;
    const ptrdiff_t stride = sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(layout, Q);
    sfem_dss_bxyze_laplacian_affine_volumetric_kernel_t kernel = nullptr;
    if (stride >= 0) {
        const int bs = layout->element_block_size, be = layout->execution_block_size;
        switch (layout->level) {
            case 2: kernel = select_quadrature_kernel<3>(Q, bs, be); break;
            case 4: kernel = select_quadrature_kernel<5>(Q, bs, be); break;
            case 8: kernel = select_quadrature_kernel<9>(Q, bs, be); break;
        }
    }
    if (!kernel) {
        *op = {};
        return SFEM_FAILURE;
    }
    *op = {*layout, Q, stride, kernel};
    return SFEM_SUCCESS;
}

int sfem_dss_bxyze_laplacian_affine_volumetric_apply(
        const sfem_dss_bxyze_laplacian_affine_volumetric_t *const op,
        const scalar_t *const S,
        const scalar_t *const D,
        const scalar_t *const qw,
        const scalar_t *const geometry,
        const real_t *const x_bxyze,
        real_t *const y_bxyze,
        scalar_t *const workspace) {
    if (!op || !op->kernel) return SFEM_FAILURE;
    if (op->layout.nelements == 0) return SFEM_SUCCESS;
    if (!S || !D || !qw || !geometry || !x_bxyze || !y_bxyze || !workspace) return SFEM_FAILURE;
    return op->kernel(op, S, D, qw, geometry, x_bxyze, y_bxyze, workspace);
}
