#include "sfem_ssgmg.hpp"

#include "sfem_hex8_mesh_graph.h"

#include "sfem_API.hpp"

namespace sfem {

    std::shared_ptr<Multigrid<real_t>> create_ssgmg(const std::shared_ptr<Function> &f,
                                                    const enum ExecutionSpace        es,
                                                    const std::shared_ptr<Input>    &in) {
        SFEM_TRACE_SCOPE("create_ssgmg");

        if (!f->space()->has_semi_structured_mesh()) {
            SFEM_ERROR("[Error] create_ssgmg cannot build MG without a semistructured mesh");
            return nullptr;
        }

        auto      fs     = f->space();
        auto     &ssmesh = f->space()->semi_structured_mesh();
        const int L      = ssmesh.level();

        std::vector<int> levels = ssmesh.derefinement_levels();

        // Order from finest to coarsest!
        std::reverse(levels.begin(), levels.end());
        const int nlevels = levels.size();

        std::vector<std::shared_ptr<Function>> functions;
        functions.push_back(f);

        for (int l = 1; l < nlevels; l++) {
            printf("Derefine %d -> %d\n", levels[l - 1], levels[l]);
            auto f_prev  = functions.back();
            auto fs_prev = f_prev->space();
            auto fs_next = fs_prev->derefine(levels[l]);
            fs_next->n_dofs();
            printf("fs_next->n_dofs() = %ld\n", (long)fs_next->n_dofs());
            functions.push_back(f_prev->derefine(fs_next, true));
        }

        std::vector<std::shared_ptr<Operator<real_t>>> operators;
        std::vector<std::shared_ptr<Operator<real_t>>> smoothers_or_solver;

        for (int i = 0; i < nlevels - 1; i++) {
            auto fi        = functions[i];
            auto linear_op = sfem::create_linear_operator(MATRIX_FREE, fi, nullptr, es);
            operators.push_back(linear_op);

            auto d = sfem::create_buffer<real_t>(fi->space()->n_dofs(), es);
            fi->hessian_diag(nullptr, d->data());
            fi->set_value_to_constrained_dofs(1, d->data());

            auto sj                  = sfem::create_shiftable_jacobi(d, es);
            sj->relaxation_parameter = 1. / fs->block_size();

            auto smoother = sfem::create_stationary<real_t>(linear_op, sj, es);
            smoother->set_max_it(3);
            smoothers_or_solver.push_back(smoother);
        }

        // Coarse level
        auto fi        = functions.back();
        auto linear_op = sfem::create_linear_operator(MATRIX_FREE, fi, nullptr, es);
        operators.push_back(linear_op);

        // Coarse-grid solver
        auto coarse_solver = sfem::create_cg<real_t>(operators.back(), es);
        coarse_solver->set_max_it(10000);
        coarse_solver->verbose = false;
        coarse_solver->set_rtol(1e-6);

        // Fine level
        bool enable_coarse_space_preconditioner = false;
        if (in) in->get("enable_coarse_space_preconditioner", enable_coarse_space_preconditioner);

        if (enable_coarse_space_preconditioner) {
            auto f_coarse = functions.back();
            auto diag     = sfem::create_buffer<real_t>(f_coarse->space()->n_dofs(), es);
            f_coarse->hessian_diag(nullptr, diag->data());
            auto sj_coarse = sfem::create_shiftable_jacobi(diag, es);
            coarse_solver->set_preconditioner_op(sj_coarse);
        }

        smoothers_or_solver.push_back(coarse_solver);

        for (int i = 0; i < nlevels; i++) {
            auto s = functions[i]->space();
            printf("%d) \tL=%d\n", i, s->has_semi_structured_mesh() ? s->semi_structured_mesh().level() : 1);

            functions[i]->describe(std::cout);
        }

        auto mg = std::make_shared<Multigrid<real_t>>();

        {
            SFEM_TRACE_SCOPE("create_ssgmg::construct");
            // Construct actual multigrid

            auto restriction_unconstr = sfem::create_hierarchical_restriction(functions[0]->space(), functions[1]->space(), es);
            auto f_coarse             = functions[1];
            auto restriction          = sfem::make_op<real_t>(
                    restriction_unconstr->rows(),
                    restriction_unconstr->cols(),
                    [=](const real_t *const from, real_t *const to) {
                        restriction_unconstr->apply(from, to);
                        f_coarse->apply_zero_constraints(to);
                    },
                    es);

            mg->add_level(operators[0], smoothers_or_solver[0], nullptr, restriction);

            // Intermediate levels
            for (int i = 1; i < nlevels - 1; i++) {
                auto restriction_unconstr =
                        sfem::create_hierarchical_restriction(functions[i]->space(), functions[i + 1]->space(), es);
                auto f_coarse    = functions[i + 1];
                auto restriction = sfem::make_op<real_t>(
                        restriction_unconstr->rows(),
                        restriction_unconstr->cols(),
                        [=](const real_t *const from, real_t *const to) {
                            restriction_unconstr->apply(from, to);
                            f_coarse->apply_zero_constraints(to);
                        },
                        es);

                auto prolong_unconstr =
                        sfem::create_hierarchical_prolongation(functions[i]->space(), functions[i - 1]->space(), es);

                auto prolongation = sfem::make_op<real_t>(
                        prolong_unconstr->rows(),
                        prolong_unconstr->cols(),
                        [prolong_unconstr, f = functions[i - 1]](const real_t *const from, real_t *const to) {
                            prolong_unconstr->apply(from, to);
                            f->apply_zero_constraints(to);
                        },
                        es);

                mg->add_level(operators[i], smoothers_or_solver[i], prolongation, restriction);
            }

            auto prolong_unconstr =
                    sfem::create_hierarchical_prolongation(functions.back()->space(), functions[nlevels - 2]->space(), es);

            auto prolongation = sfem::make_op<real_t>(
                    prolong_unconstr->rows(),
                    prolong_unconstr->cols(),
                    [prolong_unconstr, f = functions.back()](const real_t *const from, real_t *const to) {
                        prolong_unconstr->apply(from, to);
                        f->apply_zero_constraints(to);
                    },
                    es);

            // Coarse level
            mg->add_level(operators.back(), smoothers_or_solver.back(), prolongation, nullptr);
        }

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            // FIXME this should not be here!
            CUDA_BLAS<real_t>::build_blas(mg->blas());
            mg->execution_space_ = EXECUTION_SPACE_DEVICE;
        } else
#endif
        {
            mg->default_init();
        }

        mg->verbose = true;
        return mg;
    }

}  // namespace sfem
