#include <iostream>

#include "sfem_CooSym.hpp"
#include "sfem_Function.hpp"

#include "sfem_API.hpp"
#include "sfem_mask.h"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif
#include "sfem_Stationary.hpp"

#ifdef SFEM_ENABLE_AMG
#include "mg_builder.hpp"
#include "smoother.h"
#endif

#include "sfem_SSMultigrid.hpp"

#include <vector>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    double tick = MPI_Wtime();

    // -------------------------------
    // Read inputs
    // -------------------------------

    const char *folder                    = argv[1];
    const char *output_path               = argv[2];
    const char *SFEM_OPERATOR             = "Laplacian";
    int         SFEM_BLOCK_SIZE           = 1;
    int         SFEM_USE_PRECONDITIONER   = 0;
    int         SFEM_ELEMENT_REFINE_LEVEL = 0;
    int         SFEM_MAX_IT               = 1000;
    bool        SFEM_USE_GPU              = true;
    int         SFEM_USE_AMG              = false;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);
    SFEM_READ_ENV(SFEM_USE_GPU, atoi);
    SFEM_READ_ENV(SFEM_USE_AMG, atoi);

    sfem::ExecutionSpace es = SFEM_USE_GPU ? sfem::EXECUTION_SPACE_DEVICE : sfem::EXECUTION_SPACE_HOST;

    // -------------------------------
    // Create discretization
    // -------------------------------

    auto m  = sfem::Mesh::create_from_file(comm, folder);
    auto fs = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);

    if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    }

    // -------------------------------
    // Create problem
    // -------------------------------

    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    auto conds = sfem::create_dirichlet_conditions_from_env(fs, es);
    auto f     = sfem::Function::create(fs);
    f->add_constraint(conds);
    f->add_operator(op);

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    // -------------------------------
    // Create linear solver
    // -------------------------------

    std::shared_ptr<sfem::Operator<real_t>> solver;

#ifdef SFEM_ENABLE_AMG

    if (SFEM_USE_AMG == 2) {
        auto ssmg = sfem::create_ssmg<sfem::Multigrid<real_t>>(f, es);
        ssmg->set_max_it(30);
        ssmg->set_atol(1e-8);
        solver = ssmg;

    } else if (SFEM_USE_AMG) {
        /* old version with piecewise constant interpolation
        auto crs_graph = f->space()->mesh_ptr()->node_to_node_graph_upper_triangular();

        auto diag_values = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        auto off_diag_values = sfem::create_buffer<real_t>(crs_graph->nnz(), es);
        auto off_diag_rows = sfem::create_buffer<idx_t>(crs_graph->nnz(), es);

        f->hessian_crs_sym(x->data(),
                           crs_graph->rowptr()->data(),
                           crs_graph->colidx()->data(),
                           diag_values->data(),
                           off_diag_values->data());

        count_t *row_ptr = crs_graph->rowptr()->data();
        idx_t *col_indices = crs_graph->colidx()->data();
        for (idx_t i = 0; i < fs->n_dofs(); i++) {
            for (idx_t idx = row_ptr[i]; idx < row_ptr[i + 1]; idx++) {
                off_diag_rows->data()[idx] = i;
                assert(col_indices[idx] > i);
            }
        }

        auto mask = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
        f->constaints_mask(mask->data());
        auto fine_mat = sfem::h_coosym<idx_t, real_t>(
                mask, off_diag_rows, crs_graph->colidx(), off_diag_values, diag_values);

        auto near_null = sfem::create_buffer<real_t>(fs->n_dofs(), es);

        real_t coarsening_factor = 7.5;
        auto amg = builder_pwc(coarsening_factor, mask, near_null, fine_mat);
        */

        // New version with smoothed aggregation
        auto crs_graph = f->space()->mesh_ptr()->node_to_node_graph();
        auto values    = sfem::create_buffer<real_t>(crs_graph->nnz(), es);

        f->hessian_crs(x->data(), crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

        count_t *row_ptr     = crs_graph->rowptr()->data();
        idx_t   *col_indices = crs_graph->colidx()->data();
        auto     mask        = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
        f->constaints_mask(mask->data());
        auto fine_mat = sfem::h_crs_spmv<count_t, idx_t, real_t>(
                fs->n_dofs(), fs->n_dofs(), crs_graph->rowptr(), crs_graph->colidx(), values, 1.0);

        auto near_null = sfem::create_buffer<real_t>(fs->n_dofs(), es);

        real_t coarsening_factor = 7.5;
        auto   amg               = builder_sa(coarsening_factor, mask, near_null, fine_mat);

        if (!amg->test_interp()) {
            printf("tests passed\n");
        } else {
            printf("FAILEDDDDDD\n");
        }

#if 1
        amg->set_max_it(100);
        amg->verbose = true;
        // amg->debug = true;
        solver = amg;

        /*
        auto inv_diag = sfem::create_buffer<real_t>(mask_count(fs->n_dofs()), es);
        l2_smoother(fs->n_dofs(),
                    mask->data(),
                    fine_mat->values->size(),
                    fine_mat->diag_values->data(),
                    fine_mat->values->data(),
                    fine_mat->offdiag_rowidx->data(),
                    fine_mat->offdiag_colidx->data(),
                    inv_diag->data());
        auto l2_smoother = sfem::h_lpsmoother(inv_diag);
        */
#else
        amg->set_max_it(1);
        amg->verbose   = false;
        auto cg        = sfem::create_cg<real_t>(fine_mat, es);
        cg->verbose    = true;
        cg->check_each = 1;
        cg->set_max_it(SFEM_MAX_IT);
        cg->set_op(fine_mat);
        cg->set_preconditioner_op(amg);
        // cg->set_preconditioner_op(l2_smoother);
        solver = cg;
#endif
        /*
        auto stat_iter = sfem::h_stationary<real_t>(fine_mat, l2_smoother);
        stat_iter->set_max_it(100);
        stat_iter->verbose = true;
        solver = stat_iter;
        */
    } else
#endif  // SFEM_ENABLE_AMG
    {
        auto linear_op = sfem::make_linear_op(f);
        auto cg        = sfem::create_cg<real_t>(linear_op, es);
        cg->verbose    = true;
        cg->set_max_it(SFEM_MAX_IT);
        cg->set_op(linear_op);
        solver = cg;
    }

    // -------------------------------
    // Solve
    // -------------------------------

    double solve_tick = MPI_Wtime();

    for (int i = 0; i < fs->n_dofs(); i++) {
        x->data()[i] = 1.0;
    }

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), x->data());

    double solve_tock = MPI_Wtime();

    // -------------------------------
    // Write output
    // -------------------------------

    f->set_output_dir(output_path);
    auto output = f->output();

#ifdef SFEM_ENABLE_CUDA
    auto h_x = sfem::to_host(x);
#else
    auto h_x = x;
#endif
    output->write("x", h_x->data());

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #dofs %ld\n", (long)m->n_elements(), (long)m->n_nodes(), (long)fs->n_dofs());
        printf("TTS:\t\t\t%g seconds (solve: %g)\n", tock - tick, solve_tock - solve_tick);
    }

    return MPI_Finalize();
}
