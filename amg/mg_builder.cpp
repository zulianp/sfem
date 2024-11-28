#include "mg_builder.hpp"
#include <stdio.h>
#include <cstddef>
#include "partitioner.h"
#include "sfem_API.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_LpSmoother.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_pwc_interpolator.hpp"
#include "smoother.h"

std::shared_ptr<sfem::Multigrid<real_t>> builder(
        const real_t coarsening_factor, const std::shared_ptr<sfem::Buffer<mask_t>> bdy_dofs_buff,
        std::shared_ptr<sfem::Buffer<real_t>> near_null,
        std::shared_ptr<sfem::CooSymSpMV<idx_t, real_t>> &fine_mat) {
    std::shared_ptr<sfem::Multigrid<real_t>> amg = sfem::h_mg<real_t>();

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;
    auto bdy_dofs = bdy_dofs_buff->data();

    ptrdiff_t fine_ndofs = fine_mat->rows();
    count_t offdiag_nnz = fine_mat->values->size();

    for (idx_t i = 0; i < fine_ndofs; i++) {
        near_null->data()[i] = 1.0;
    }

    PartitionerWorkspace *ws = create_partition_ws(fine_ndofs, offdiag_nnz);
    // Weighted connectivity graph in COO format
    auto offdiag_row_indices = sfem::h_buffer<idx_t>(offdiag_nnz);
    auto offdiag_col_indices = sfem::h_buffer<idx_t>(offdiag_nnz);
    auto offdiag_values = sfem::h_buffer<real_t>(offdiag_nnz);

    idx_t amg_levels = 1;

    // AMG tunable paramaters
    count_t smoothing_steps = 3;
    count_t coarsest_ndofs = 500;
    count_t max_levels = 20;

    auto prev_mat = fine_mat;
    std::shared_ptr<sfem::PiecewiseConstantInterpolator<idx_t, real_t>> p, pt = nullptr;

    auto diag_smoother = sfem::h_buffer<real_t>(fine_ndofs);
    l2_smoother(fine_ndofs,
                bdy_dofs,
                prev_mat->values->size(),
                prev_mat->diag_values->data(),
                prev_mat->values->data(),
                prev_mat->offdiag_rowidx->data(),
                prev_mat->offdiag_colidx->data(),
                diag_smoother->data());
    auto amg_smoother = sfem::create_shiftable_jacobi(diag_smoother, es);
    amg_smoother->relaxation_parameter = 1.0;

    ptrdiff_t ndofs = fine_ndofs;
    count_t fine_memory = fine_ndofs + offdiag_nnz;
    count_t fine_nnz = fine_ndofs + offdiag_nnz * 2;

    count_t prev_nnz = fine_nnz;
    count_t total_memory = fine_memory;
    count_t total_complexity = fine_nnz;

    printf("\nAMG info:\n level      cf         dofs       offdiag nnz    true nnz   "
           "sparsity "
           "factor\n");
    printf("|%-10d|%-10.2f|%-10td|%-14d|%-10d|%-16.2f|\n",
           1,
           1.0,
           fine_ndofs,
           offdiag_nnz,
           fine_nnz,
           1.0);

    while (amg_levels < max_levels && ndofs > coarsest_ndofs) {
        for (idx_t k = 0; k < offdiag_nnz; k++) {
            offdiag_row_indices->data()[k] = prev_mat->offdiag_rowidx->data()[k];
            offdiag_col_indices->data()[k] = prev_mat->offdiag_colidx->data()[k];
            offdiag_values->data()[k] = prev_mat->values->data()[k];
        }

        ptrdiff_t finer_dim = ndofs;
        int failure = partition(bdy_dofs,
                                coarsening_factor,
                                near_null->data(),
                                offdiag_row_indices->data(),
                                offdiag_col_indices->data(),
                                offdiag_values->data(),
                                &offdiag_nnz,
                                &ndofs,
                                ws);

        if (failure) {
            printf("Failed to add new level, AMG levels: %d\n", amg_levels);
            // Otherwise no AMG :(
            assert(amg_levels > 1);
            break;
        }

        ptrdiff_t coarser_dim = ndofs;
        auto partition_buff = sfem::h_buffer<idx_t>(finer_dim);
        auto weights_buff = sfem::h_buffer<real_t>(finer_dim);

        for (idx_t k = 0; k < finer_dim; k++) {
            partition_buff->data()[k] = ws->partition[k];
            weights_buff->data()[k] = near_null->data()[k];
        }

        auto pt = h_pwc_interp(weights_buff, partition_buff, coarser_dim);
        pt->transpose();

        // Convert matrix?
        std::shared_ptr<sfem::Operator<real_t>> coarse_op = prev_mat;
        auto stat_iter = sfem::create_stationary<real_t>(coarse_op, amg_smoother, es);

        stat_iter->set_max_it(smoothing_steps);
        amg->add_level(coarse_op, stat_iter, p, pt);
        p = h_pwc_interp(weights_buff, partition_buff, coarser_dim);

        amg_levels++;

        auto a_coarse = p->coarsen(prev_mat);
        offdiag_nnz = a_coarse->values->size();
        count_t coarse_nnz = ndofs + (offdiag_nnz * 2);
        printf("|%-10d|%-10.2f|%-10td|%-14d|%-10d|%-16.2f|\n",
               amg_levels,
               (real_t)finer_dim / (real_t)coarser_dim,
               ndofs,
               offdiag_nnz,
               coarse_nnz,
               (real_t)coarse_nnz / (real_t)prev_nnz);
        total_memory += ndofs + offdiag_nnz;
        total_complexity += ndofs + (offdiag_nnz * 2);
        prev_nnz = coarse_nnz;
        prev_mat = a_coarse;
        bdy_dofs = nullptr;

        diag_smoother = sfem::create_buffer<real_t>(ndofs, es);
        l2_smoother(ndofs,
                    bdy_dofs,
                    prev_mat->values->size(),
                    prev_mat->diag_values->data(),
                    prev_mat->values->data(),
                    prev_mat->offdiag_rowidx->data(),
                    prev_mat->offdiag_colidx->data(),
                    diag_smoother->data());
        amg_smoother = sfem::create_shiftable_jacobi(diag_smoother, es);
        amg_smoother->relaxation_parameter = 1.0;
    }

    // Create a coarsest level solver, could also just smooth here if coarsest problem isn't
    // small enough to solve exactly. Direct solver by cholesky is also probably better here
    auto cg = sfem::create_cg<real_t>(prev_mat, es);
    cg->verbose = false;
    cg->set_max_it(10000);  // Keep it large just to be sure!
    cg->set_rtol(1e-12);
    cg->set_preconditioner_op(amg_smoother);
    amg->add_level(prev_mat, cg, p, nullptr);

    if (amg_levels == max_levels) {
        printf("AMG constructed successfully with max levels hit (%d levels)\n", amg_levels);
    } else if (ndofs <= coarsest_ndofs) {
        printf("AMG constructed successfully with coarsest target hit (%td dofs on coarsest)\n",
               ndofs);
    }
    printf("Memory complexity: %.2f\n", (real_t)total_memory / (real_t)fine_memory);
    printf("Operator complexity: %.2f\n\n", (real_t)total_complexity / (real_t)fine_nnz);

    free_partition_ws(ws);

    return amg;
}
