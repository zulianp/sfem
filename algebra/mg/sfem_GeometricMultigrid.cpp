#include "sfem_GeometricMultigrid.hpp"

#include "sfem_aliases.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

#include <algorithm>

namespace sfem {

    std::shared_ptr<MultigridData> create_gmg_data(const std::shared_ptr<Function> &f) {
        SFEM_TRACE_SCOPE("create_gmg_data");

        if (!f->space()->has_semi_structured_mesh()) {
            SFEM_ERROR("[Error] create_ssgmg cannot build MG without a semistructured mesh");
            return nullptr;
        }

        auto data = std::make_shared<MultigridData>();

        auto  es     = f->execution_space();
        auto &ssmesh = f->space()->mesh();

        std::vector<int> levels = smesh::derefinement_levels(ssmesh);
        std::reverse(levels.begin(), levels.end());
        const int nlevels = levels.size();

        data->functions.push_back(f);

        for (int l = 1; l < nlevels; l++) {
            auto f_prev  = data->functions.back();
            auto fs_next = f_prev->space()->derefine(levels[l]);
            data->functions.push_back(f_prev->derefine(fs_next, true));
        }

        auto create_r = [&](const int i) -> std::shared_ptr<Operator<real_t>> {
            auto restriction_unconstr =
                    sfem::create_hierarchical_restriction(data->functions[i]->space(), data->functions[i + 1]->space(), es);
            auto f_coarse = data->functions[i + 1];

            auto restriction = sfem::make_op<real_t>(
                    restriction_unconstr->rows(),
                    restriction_unconstr->cols(),
                    [=](const real_t *const from, real_t *const to) {
                        restriction_unconstr->apply(from, to);
                        f_coarse->apply_zero_constraints(to);
                    },
                    es);
            return restriction;
        };

        auto create_p = [&](const int i) -> std::shared_ptr<Operator<real_t>> {
            auto prolong_unconstr =
                    sfem::create_hierarchical_prolongation(data->functions[i]->space(), data->functions[i - 1]->space(), es);
            auto prolongation = sfem::make_op<real_t>(
                    prolong_unconstr->rows(),
                    prolong_unconstr->cols(),
                    [prolong_unconstr, f = data->functions[i - 1]](const real_t *const from, real_t *const to) {
                        prolong_unconstr->apply(from, to);
                        f->apply_zero_constraints(to);
                    },
                    es);
            return prolongation;
        };

        data->prolongations.push_back(nullptr);

        for (int i = 0; i < nlevels - 1; i++) {
            data->restrictions.push_back(create_r(i));
            data->prolongations.push_back(create_p(i + 1));
        }

        return data;
    }

    std::vector<std::shared_ptr<Operator<real_t>>> create_gmg_operators(const std::shared_ptr<MultigridData> &data,
                                                                        const OperatorType                    op_type) {
        std::vector<std::shared_ptr<Operator<real_t>>> operators;
        for (int i = 0; i < data->functions.size(); i++) {
            auto f  = data->functions[i];
            auto op = sfem::create_linear_operator(op_type, f, nullptr, f->execution_space());
            operators.push_back(op);
        }
        return operators;
    }

    std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> create_gmg_default_smoothers_and_solver(
            const std::shared_ptr<MultigridData>           &data,
            std::vector<std::shared_ptr<Operator<real_t>>> &ops,
            const int                                       smoothing_steps,
            const bool                                      enable_mixed_precision) {
        std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> smoothers;

        if (data->functions.empty()) {
            SFEM_ERROR("[Error] create_gmg_default_smoothers_and_solver cannot build MG without a function");
            return {};
        }

        const int  block_size     = data->functions.front()->space()->block_size();
        const auto es             = data->functions.front()->execution_space();
        const int  nlevels        = data->functions.size();
        const int  sym_block_size = (block_size == 3 ? 6 : 3);

        auto create_jacobi = [&](const std::shared_ptr<Function> &f) -> std::shared_ptr<Operator<real_t>> {
            if (block_size == 1) {
                auto diag = sfem::create_buffer<real_t>(f->space()->n_dofs(), es);
                f->hessian_diag(nullptr, diag->data());
                f->set_value_to_constrained_dofs(1, diag->data());

                auto jacobi                  = sfem::create_shiftable_jacobi(diag, es);
                jacobi->relaxation_parameter = 1.;
                return jacobi;
            } else {
                auto fs   = f->space();
                auto diag = sfem::create_buffer<real_t>(fs->n_dofs() / fs->block_size() * sym_block_size, es);
                auto mask = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);

                f->constraints_mask(mask->data());
                f->hessian_block_diag_sym(nullptr, diag->data());

                std::shared_ptr<sfem::Operator<real_t>> jacobi;
                if (enable_mixed_precision) {
                    auto temp =
                            sfem::create_mixed_precision_shiftable_block_sym_jacobi<real_t, float>(block_size, diag, mask, es);
                    temp->relaxation_parameter = 1. / block_size;
                    jacobi                     = temp;
                } else {
                    auto temp                  = sfem::create_shiftable_block_sym_jacobi(block_size, diag, mask, es);
                    temp->relaxation_parameter = 1. / block_size;
                    jacobi                     = temp;
                }

                return jacobi;
            }
        };

        for (int i = 0; i < nlevels - 1; i++) {
            auto f        = data->functions[i];
            auto smoother = sfem::create_stationary<real_t>(ops[i], create_jacobi(f), es);
            smoother->set_max_it(smoothing_steps);
            smoothers.push_back(smoother);
        }

        auto coarse_solver = sfem::create_cg<real_t>(ops.back(), es);
        coarse_solver->set_max_it(10000);
        coarse_solver->verbose = true;
        coarse_solver->set_rtol(1e-6);

        bool enable_coarse_space_preconditioner = true;
        if (enable_coarse_space_preconditioner) {
            auto f    = data->functions.back();
            auto diag = sfem::create_buffer<real_t>(f->space()->n_dofs(), es);
            f->hessian_diag(nullptr, diag->data());
            auto sj_coarse                  = sfem::create_shiftable_jacobi(diag, es);
            sj_coarse->relaxation_parameter = 1. / block_size;
            coarse_solver->set_preconditioner_op(sj_coarse);

            // This is not working for some reason. BCs?
            // coarse_solver->set_preconditioner_op(create_jacobi(data->functions.back()));
        }

        smoothers.push_back(coarse_solver);

        return smoothers;
    }

}  // namespace sfem
