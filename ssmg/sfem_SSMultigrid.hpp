#ifndef SFEM_SS_MULTIGRID_HPP
#define SFEM_SS_MULTIGRID_HPP

#include "sfem_API.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_ShiftableJacobi.hpp"

namespace sfem {

    template <class MG>
    std::shared_ptr<MG> create_ssmg(const std::shared_ptr<Function> &f,
                                   const enum ExecutionSpace es) {
        if (!f->space()->has_semi_structured_mesh()) {
            fprintf(stderr, "[Error] create_ssmg cannot build MG without a semistructured mesh");
            MPI_Abort(MPI_COMM_WORLD, -1);
            return nullptr;
        }

        auto fs = f->space();
        auto fs_coarse = fs->derefine();
        auto f_coarse = f->derefine(fs_coarse, true);
        // auto linear_op = sfem::create_linear_operator("BSR", f, nullptr, es);
        // auto linear_op_coarse = sfem::create_linear_operator("BCRS_SYM", f_coarse, nullptr, es);

        auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);
        auto linear_op_coarse = sfem::create_linear_operator(fs->block_size() == 1? "COO_SYM" : "BCRS_SYM", f_coarse, nullptr, es);

        // auto smoother = sfem::create_cheb3<real_t>(linear_op, es);
        // smoother->eigen_solver_tol = 1e-2;
        // smoother->init_with_ones();
        // smoother->scale_eig_max = 1.02;
        // smoother->set_max_it(5);
        // smoother->set_initial_guess_zero(false);

        // auto smoother = sfem::create_cg<real_t>(linear_op, es);
        // smoother->set_max_it(10);
        // smoother->verbose = false;

        auto d = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        f->hessian_diag(nullptr, d->data());
        f->set_value_to_constrained_dofs(1, d->data());

        auto sj = sfem::create_shiftable_jacobi(d, es);
        sj->relaxation_parameter = 0.4;
        auto smoother = sfem::create_stationary<real_t>(linear_op, sj, es);
        smoother->set_max_it(3);

        auto solver_coarse = sfem::create_cg<real_t>(linear_op_coarse, es);
        solver_coarse->verbose = false;

        auto restriction = sfem::create_hierarchical_restriction(fs, fs_coarse, es);
        auto prolong_unconstr = sfem::create_hierarchical_prolongation(fs_coarse, fs, es);
        auto prolongation = sfem::make_op<real_t>(
                prolong_unconstr->rows(),
                prolong_unconstr->cols(),
                [=](const real_t *const from, real_t *const to) {
                    prolong_unconstr->apply(from, to);
                    f->apply_zero_constraints(to);
                },
                es);

        auto mg = std::make_shared<MG>();
        mg->set_nlsmooth_steps(5);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            CUDA_BLAS<real_t>::build_blas(mg->blas());
            // TODO cuda_init()
        } else
#endif
        {
            mg->default_init();
        }

        mg->add_level(linear_op, smoother, nullptr, restriction);
        mg->add_level(linear_op_coarse, solver_coarse, prolongation, nullptr);

        // TODO Add algebraic levels
        // TODO all on CPU
        return mg;
    }

}  // namespace sfem

#endif  // SFEM_SS_MULTIGRID_HPP
