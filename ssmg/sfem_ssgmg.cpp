#include "sfem_ssgmg.hpp"

#include "sfem_hex8_mesh_graph.h"

#include "sfem_API.hpp"

namespace sfem {

    std::shared_ptr<Multigrid<real_t>> create_ssgmg(const std::shared_ptr<Function> &f,
                                                    const enum ExecutionSpace        es,
                                                    const std::shared_ptr<Input>    &in) {
        if (!f->space()->has_semi_structured_mesh()) {
            SFEM_ERROR("[Error] create_ssgmg cannot build MG without a semistructured mesh");
            return nullptr;
        }

        auto      fs     = f->space();
        auto     &ssmesh = f->space()->semi_structured_mesh();
        const int L      = ssmesh.level();

        // FiXME harcoded for sshex8
        const int nlevels = sshex8_hierarchical_n_levels(L);

        std::vector<int> levels(nlevels);

        // FiXME harcoded for sshex8
        sshex8_hierarchical_mesh_levels(L, nlevels, levels.data());

        std::vector<std::shared_ptr<Function>> functions;
        functions.push_back(f);

        for (int l = L; l >= 1; l--) {
            printf("Derefine %d -> %d\n", levels[l], levels[l - 1]);
            auto f_prev  = functions.back();
            auto fs_prev = f_prev->space();
            auto fs_next = fs_prev->derefine(levels[l - 1]);
            functions.push_back(f_prev->derefine(fs_next, true));
        }

        std::vector<std::shared_ptr<Operator<real_t>>> operators;
        std::vector<std::shared_ptr<Operator<real_t>>> smoothers_or_solver;

        for (int i = 0; i < nlevels - 1; i++) {
            auto fi        = functions[i];
            auto linear_op = sfem::create_linear_operator("MF", fi, nullptr, es);
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

        // Coarse-grid solver
        auto coarse_solver = sfem::create_cg<real_t>(operators[nlevels - 1], es);

        // Fine level
        bool enable_coarse_space_preconditioner = false;
        in->get("enable_coarse_space_preconditioner", enable_coarse_space_preconditioner);

        if (enable_coarse_space_preconditioner) {
            auto f_coarse = functions[nlevels - 1];
            auto diag     = sfem::create_buffer<real_t>(f_coarse->space()->n_dofs(), es);
            f_coarse->hessian_diag(nullptr, diag->data());
            auto sj_coarse = sfem::create_shiftable_jacobi(diag, es);
            coarse_solver->set_preconditioner_op(sj_coarse);
        }

        smoothers_or_solver.push_back(coarse_solver);

        auto mg = std::make_shared<Multigrid<real_t>>();
        {  // Construct actual multigrid
            mg->add_level(operators[0],
                          smoothers_or_solver[0],
                          nullptr,
                          sfem::create_hierarchical_restriction(functions[1]->space(), functions[0]->space(), es));

            // Intermediate levels
            for (int i = 1; i < nlevels - 1; i++) {
                auto restriction = sfem::create_hierarchical_restriction(functions[i]->space(), functions[i + 1]->space(), es);
                
                auto prolong_unconstr =
                        sfem::create_hierarchical_prolongation(functions[i]->space(), functions[i - 1]->space(), es);
                
                auto prolongation = sfem::make_op<real_t>(
                        prolong_unconstr->rows(),
                        prolong_unconstr->cols(),
                        [=](const real_t *const from, real_t *const to) {
                            prolong_unconstr->apply(from, to);
                            f->apply_zero_constraints(to);
                        },
                        es);

                mg->add_level(operators[i], smoothers_or_solver[i], prolongation, restriction);
            }

            // Coarse level
            mg->add_level(
                    operators[nlevels - 1],
                    smoothers_or_solver[nlevels - 1],
                    sfem::create_hierarchical_prolongation(functions[nlevels - 1]->space(), functions[nlevels - 2]->space(), es),
                    nullptr);
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

        return mg;
    }

}  // namespace sfem
