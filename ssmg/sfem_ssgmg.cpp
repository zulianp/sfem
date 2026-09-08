#include "sfem_ssgmg.hpp"

#include "sfem_GeometricMultigrid.hpp"

#include "sfem_API.hpp"

namespace sfem {

    std::shared_ptr<Multigrid<real_t>> create_ssgmg(const std::shared_ptr<Function> &f, const enum ExecutionSpace es) {
        SFEM_TRACE_SCOPE("create_ssgmg");

        auto data = sfem::create_gmg_data(f);
        if (!data) {
            return nullptr;
        }

        const int nlevels          = data->functions.size();
        const int smoothing_steps  = smesh::Env::read("SFEM_MG_SMOOTHING_STEPS", 5);
        auto      operators        = sfem::create_gmg_operators(data, op_type::MATRIX_FREE);
        auto      smoothers_or_solver =
                sfem::create_gmg_default_smoothers_and_solver(data, operators, smoothing_steps, false);

        auto mg = std::make_shared<Multigrid<real_t>>();

        for (int i = 0; i < nlevels; i++) {
            auto restriction  = (i < nlevels - 1) ? data->restrictions[i] : nullptr;
            auto prolongation = data->prolongations[i];
            mg->add_level(operators[i], smoothers_or_solver[i], prolongation, restriction);
        }

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            // FIXME this should not be here!
            mg->blas() = make_cuda_blas<real_t>();
            mg->execution_space_ = EXECUTION_SPACE_DEVICE;
        } else
#endif
        {
            mg->default_init();
        }

        mg->verbose = true;
        mg->set_max_it(smesh::Env::read("SFEM_MG_MAX_IT", 40));
        mg->set_atol(smesh::Env::read("SFEM_MG_ATOL", real_t(1e-10)));
        return mg;
    }

}  // namespace sfem
