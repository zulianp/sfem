#include "mg_builder.hpp"
#include <stdio.h>
#include <cstddef>
#include "partitioner.h"
#include "sfem_API.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_LpSmoother.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_crs_SpMV.hpp"
#include "sfem_pwc_interpolator.hpp"
#include "smoothed_aggregation.h"
#include "smoother.h"

int csr_to_coosym(ptrdiff_t ndofs,
                  count_t  *rowptr,
                  idx_t    *colidx,
                  real_t   *values,
                  idx_t    *offdiag_row_indices,
                  idx_t    *offdiag_col_indices,
                  real_t   *offdiag_values,
                  real_t   *diag_values) {
    count_t k = 0;
    for (idx_t i = 0; i < ndofs; i++) {
        for (count_t idx = rowptr[i]; idx < rowptr[i + 1]; idx++) {
            idx_t  j   = colidx[idx];
            real_t val = values[idx];
            if (i == j) {
                diag_values[i] = val;
            } else if (j > i) {
                offdiag_row_indices[k] = i;
                offdiag_col_indices[k] = j;
                offdiag_values[k]      = val;
                k++;
            }
        }
    }
    return k;
}

std::shared_ptr<sfem::Multigrid<real_t>> builder_sa(const real_t                                            coarsening_factor,
                                                    const std::shared_ptr<sfem::Buffer<mask_t>>             bdy_dofs_buff,
                                                    std::shared_ptr<sfem::Buffer<real_t>>                   near_null,
                                                    std::shared_ptr<sfem::Buffer<real_t>>                   zeros,
                                                    std::shared_ptr<sfem::CRSSpMV<count_t, idx_t, real_t>> &fine_mat) {
    std::shared_ptr<sfem::Multigrid<real_t>> amg = sfem::h_mg<real_t>();

    sfem::ExecutionSpace es       = sfem::EXECUTION_SPACE_HOST;
    auto                 bdy_dofs = bdy_dofs_buff->data();

    ptrdiff_t fine_ndofs  = fine_mat->rows();
    count_t   offdiag_nnz = (fine_mat->values->size() - fine_ndofs) / 2;

    PartitionerWorkspace *ws = create_partition_ws(fine_ndofs, offdiag_nnz);
    // Weighted connectivity graph in COO format
    auto offdiag_row_indices = sfem::create_host_buffer<idx_t>(offdiag_nnz);
    auto offdiag_col_indices = sfem::create_host_buffer<idx_t>(offdiag_nnz);
    auto offdiag_values      = sfem::create_host_buffer<real_t>(offdiag_nnz);
    auto diag_values         = sfem::create_host_buffer<real_t>(fine_ndofs);

    idx_t amg_levels = 1;

    // AMG tunable paramaters
    count_t smoothing_steps = 3;
    count_t coarsest_ndofs  = 500;
    count_t max_levels      = 20;

    auto prev_mat = fine_mat;

    std::shared_ptr<sfem::CRSSpMV<count_t, idx_t, real_t>> p, pt = nullptr;
    std::shared_ptr<sfem::ShiftableJacobi<double>>         amg_smoother = nullptr;

    ptrdiff_t ndofs = fine_ndofs;

    printf("\nAMG info:\n level      cf         dofs       nnz   "
           "sparsity "
           "factor\n");
    printf("|%-10d|%-10.2f|%-10td|%-10td|%-16.2f|\n", 1, 1.0, fine_ndofs, fine_mat->values->size(), 1.0);
    while (amg_levels < max_levels && ndofs > coarsest_ndofs) {
        count_t *rowptr = prev_mat->row_ptr->data();
        idx_t   *colidx = prev_mat->col_idx->data();
        real_t  *values = prev_mat->values->data();

        count_t offdiag_nnz = (prev_mat->values->size() - ndofs) / 2;
        count_t k           = csr_to_coosym(ndofs,
                                  rowptr,
                                  colidx,
                                  values,
                                  offdiag_row_indices->data(),
                                  offdiag_col_indices->data(),
                                  offdiag_values->data(),
                                  diag_values->data());
        printf("%d == %d\n", k, offdiag_nnz);
        assert(k == offdiag_nnz);

        auto diag_smoother = sfem::create_host_buffer<real_t>(ndofs);
        l1_smoother(ndofs,
                    bdy_dofs,
                    offdiag_nnz,
                    diag_values->data(),
                    offdiag_values->data(),
                    offdiag_row_indices->data(),
                    offdiag_col_indices->data(),
                    diag_smoother->data());
        amg_smoother                       = sfem::create_shiftable_jacobi(diag_smoother, es);
        amg_smoother->relaxation_parameter = 1.0;

        std::shared_ptr<sfem::Operator<real_t>> op        = prev_mat;
        auto                                    stat_iter = sfem::create_stationary<real_t>(op, amg_smoother, es);
        stat_iter->set_max_it(smoothing_steps);
        stat_iter->apply(zeros->data(), near_null->data());

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

        count_t *rowptr_p;
        idx_t   *colidx_p;
        real_t  *values_p;
        count_t *rowptr_pt;
        idx_t   *colidx_pt;
        real_t  *values_pt;
        count_t *rowptr_coarse;
        idx_t   *colidx_coarse;
        real_t  *values_coarse;

        printf("Partition: %td coarse dofs\n", coarser_dim);
        smoothed_aggregation(finer_dim,
                             coarser_dim,
                             0.66,
                             ws->partition,
                             rowptr,
                             colidx,
                             values,
                             near_null->data(),
                             &rowptr_p,
                             &colidx_p,
                             &values_p,
                             &rowptr_pt,
                             &colidx_pt,
                             &values_pt,
                             &rowptr_coarse,
                             &colidx_coarse,
                             &values_coarse);

        assert(rowptr_p[finer_dim] == rowptr_pt[coarser_dim]);
        count_t interp_weights = rowptr_p[finer_dim];

        pt = sfem::h_crs_spmv(coarser_dim,
                              finer_dim,
                              sfem::manage_host_buffer(coarser_dim + 1, rowptr_pt),
                              sfem::manage_host_buffer(interp_weights, colidx_pt),
                              sfem::manage_host_buffer(interp_weights, values_pt),
                              1.0);

        amg->add_level(op, stat_iter, p, pt);
        p = sfem::h_crs_spmv(finer_dim,
                             coarser_dim,
                             sfem::manage_host_buffer(finer_dim + 1, rowptr_p),
                             sfem::manage_host_buffer(interp_weights, colidx_p),
                             sfem::manage_host_buffer(interp_weights, values_p),
                             1.0);

        amg_levels++;

        count_t nnz_coarse = rowptr_coarse[coarser_dim];
        auto    a_coarse   = sfem::h_crs_spmv(coarser_dim,
                                         coarser_dim,
                                         sfem::manage_host_buffer(coarser_dim + 1, rowptr_coarse),
                                         sfem::manage_host_buffer(nnz_coarse, colidx_coarse),
                                         sfem::manage_host_buffer(nnz_coarse, values_coarse),
                                         1.0);

        count_t prev_nnz = prev_mat->values->size();
        prev_mat         = a_coarse;
        bdy_dofs         = nullptr;
        printf("|%-10d|%-10.2f|%-10td|%-10d|%-16.2f|\n",
               amg_levels,
               (real_t)finer_dim / (real_t)coarser_dim,
               ndofs,
               nnz_coarse,
               (real_t)nnz_coarse / (real_t)prev_nnz);
    }

    offdiag_nnz = (prev_mat->values->size() - ndofs) / 2;
    count_t k   = csr_to_coosym(ndofs,
                              prev_mat->row_ptr->data(),
                              prev_mat->col_idx->data(),
                              prev_mat->values->data(),
                              offdiag_row_indices->data(),
                              offdiag_col_indices->data(),
                              offdiag_values->data(),
                              diag_values->data());
    printf("%d == %d\n", k, offdiag_nnz);
    assert(k == offdiag_nnz);

    auto diag_smoother = sfem::create_host_buffer<real_t>(ndofs);
    l1_smoother(ndofs,
                bdy_dofs,
                offdiag_nnz,
                diag_values->data(),
                offdiag_values->data(),
                offdiag_row_indices->data(),
                offdiag_col_indices->data(),
                diag_smoother->data());
    amg_smoother                       = sfem::create_shiftable_jacobi(diag_smoother, es);
    amg_smoother->relaxation_parameter = 1.0;

    // Create a coarsest level solver, could also just smooth here if coarsest problem isn't
    // small enough to solve exactly. Direct solver by cholesky is also probably better here
#if 1
    auto cg     = sfem::create_cg<real_t>(prev_mat, es);
    cg->verbose = false;
    cg->set_max_it(10000);  // Keep it large just to be sure!
    cg->set_rtol(1e-12);
    cg->set_preconditioner_op(amg_smoother);
    amg->add_level(prev_mat, cg, p, nullptr);
#else
    std::shared_ptr<sfem::Operator<real_t>> op        = prev_mat;
    auto                                    stat_iter = sfem::create_stationary<real_t>(op, amg_smoother, es);
    stat_iter->set_max_it(smoothing_steps);
    amg->add_level(prev_mat, stat_iter, p, nullptr);
#endif

    if (amg_levels == max_levels) {
        printf("AMG constructed successfully with max levels hit (%d levels)\n", amg_levels);
    } else if (ndofs <= coarsest_ndofs) {
        printf("AMG constructed successfully with coarsest target hit (%td dofs on coarsest)\n", ndofs);
    }
    // printf("Memory complexity: %.2f\n", (real_t)total_memory / (real_t)fine_memory);
    // printf("Operator complexity: %.2f\n\n", (real_t)total_complexity / (real_t)fine_nnz);

    free_partition_ws(ws);

    return amg;
}

std::shared_ptr<sfem::Multigrid<real_t>> builder_pwc(const real_t                                      coarsening_factor,
                                                     const std::shared_ptr<sfem::Buffer<mask_t>>       bdy_dofs_buff,
                                                     std::shared_ptr<sfem::Buffer<real_t>>             near_null,
                                                     std::shared_ptr<sfem::CooSymSpMV<idx_t, real_t>> &fine_mat) {
    std::shared_ptr<sfem::Multigrid<real_t>> amg = sfem::h_mg<real_t>();

    sfem::ExecutionSpace es       = sfem::EXECUTION_SPACE_HOST;
    auto                 bdy_dofs = bdy_dofs_buff->data();

    ptrdiff_t fine_ndofs  = fine_mat->rows();
    count_t   offdiag_nnz = fine_mat->values->size();

    for (idx_t i = 0; i < fine_ndofs; i++) {
        near_null->data()[i] = 1.0;
    }

    PartitionerWorkspace *ws = create_partition_ws(fine_ndofs, offdiag_nnz);
    // Weighted connectivity graph in COO format
    auto offdiag_row_indices = sfem::create_host_buffer<idx_t>(offdiag_nnz);
    auto offdiag_col_indices = sfem::create_host_buffer<idx_t>(offdiag_nnz);
    auto offdiag_values      = sfem::create_host_buffer<real_t>(offdiag_nnz);

    idx_t amg_levels = 1;

    // AMG tunable paramaters
    count_t smoothing_steps = 3;
    count_t coarsest_ndofs  = 500;
    count_t max_levels      = 20;

    auto                                                                prev_mat = fine_mat;
    std::shared_ptr<sfem::PiecewiseConstantInterpolator<idx_t, real_t>> p, pt = nullptr;

    auto diag_smoother = sfem::create_host_buffer<real_t>(fine_ndofs);
    l1_smoother(fine_ndofs,
                bdy_dofs,
                prev_mat->values->size(),
                prev_mat->diag_values->data(),
                prev_mat->values->data(),
                prev_mat->offdiag_rowidx->data(),
                prev_mat->offdiag_colidx->data(),
                diag_smoother->data());
    auto amg_smoother                  = sfem::create_shiftable_jacobi(diag_smoother, es);
    amg_smoother->relaxation_parameter = 1.0;

    ptrdiff_t ndofs       = fine_ndofs;
    count_t   fine_memory = fine_ndofs + offdiag_nnz;
    count_t   fine_nnz    = fine_ndofs + offdiag_nnz * 2;

    count_t prev_nnz         = fine_nnz;
    count_t total_memory     = fine_memory;
    count_t total_complexity = fine_nnz;

    printf("\nAMG info:\n level      cf         dofs       offdiag nnz    true nnz   "
           "sparsity "
           "factor\n");
    printf("|%-10d|%-10.2f|%-10td|%-14d|%-10d|%-16.2f|\n", 1, 1.0, fine_ndofs, offdiag_nnz, fine_nnz, 1.0);

    while (amg_levels < max_levels && ndofs > coarsest_ndofs) {
        for (idx_t k = 0; k < offdiag_nnz; k++) {
            offdiag_row_indices->data()[k] = prev_mat->offdiag_rowidx->data()[k];
            offdiag_col_indices->data()[k] = prev_mat->offdiag_colidx->data()[k];
            offdiag_values->data()[k]      = prev_mat->values->data()[k];
        }

        ptrdiff_t finer_dim = ndofs;
        int       failure   = partition(bdy_dofs,
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

        ptrdiff_t coarser_dim    = ndofs;
        auto      partition_buff = sfem::create_host_buffer<idx_t>(finer_dim);
        auto      weights_buff   = sfem::create_host_buffer<real_t>(finer_dim);

        for (idx_t k = 0; k < finer_dim; k++) {
            partition_buff->data()[k] = ws->partition[k];
            weights_buff->data()[k]   = near_null->data()[k];
        }

        auto pt = h_pwc_interp(weights_buff, partition_buff, coarser_dim);
        pt->transpose();

        // Convert matrix?
        std::shared_ptr<sfem::Operator<real_t>> coarse_op = prev_mat;
        auto                                    stat_iter = sfem::create_stationary<real_t>(coarse_op, amg_smoother, es);

        stat_iter->set_max_it(smoothing_steps);
        amg->add_level(coarse_op, stat_iter, p, pt);
        p = h_pwc_interp(weights_buff, partition_buff, coarser_dim);

        amg_levels++;

        auto a_coarse      = p->coarsen(prev_mat);
        offdiag_nnz        = a_coarse->values->size();
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
        l1_smoother(ndofs,
                    bdy_dofs,
                    prev_mat->values->size(),
                    prev_mat->diag_values->data(),
                    prev_mat->values->data(),
                    prev_mat->offdiag_rowidx->data(),
                    prev_mat->offdiag_colidx->data(),
                    diag_smoother->data());
        amg_smoother                       = sfem::create_shiftable_jacobi(diag_smoother, es);
        amg_smoother->relaxation_parameter = 1.0;
    }

    // Create a coarsest level solver, could also just smooth here if coarsest problem isn't
    // small enough to solve exactly. Direct solver by cholesky is also probably better here
    auto cg     = sfem::create_cg<real_t>(prev_mat, es);
    cg->verbose = false;
    cg->set_max_it(10000);  // Keep it large just to be sure!
    cg->set_rtol(1e-12);
    cg->set_preconditioner_op(amg_smoother);
    amg->add_level(prev_mat, cg, p, nullptr);

    if (amg_levels == max_levels) {
        printf("AMG constructed successfully with max levels hit (%d levels)\n", amg_levels);
    } else if (ndofs <= coarsest_ndofs) {
        printf("AMG constructed successfully with coarsest target hit (%td dofs on coarsest)\n", ndofs);
    }
    printf("Memory complexity: %.2f\n", (real_t)total_memory / (real_t)fine_memory);
    printf("Operator complexity: %.2f\n\n", (real_t)total_complexity / (real_t)fine_nnz);

    free_partition_ws(ws);

    return amg;
}
