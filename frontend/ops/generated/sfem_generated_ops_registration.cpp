#include "sfem_generated_ops_registration.hpp"

namespace sfem {
    void register_GeneratedLaplace_generated_op();
    void register_GeneratedLinearElasticity_generated_op();
    void register_GeneratedModifiedMooneyRivlin_generated_op();
    void register_GeneratedNavierStokes_generated_op();
    void register_GeneratedNeoHookeanOgden_generated_op();
    void register_GeneratedNeumann_generated_op();
    void register_GeneratedNeumannGeneral_generated_op();
    void register_GeneratedPoroHyperelasticity_generated_op();
    void register_GeneratedSaintVenantKirchhoff_generated_op();
    void register_GeneratedStokes_generated_op();
    void register_GeneratedTwoPhaseFlow_generated_op();

    void register_generated_ops() {
        register_GeneratedLaplace_generated_op();
        register_GeneratedLinearElasticity_generated_op();
        register_GeneratedModifiedMooneyRivlin_generated_op();
        register_GeneratedNavierStokes_generated_op();
        register_GeneratedNeoHookeanOgden_generated_op();
        register_GeneratedNeumann_generated_op();
        register_GeneratedNeumannGeneral_generated_op();
        register_GeneratedPoroHyperelasticity_generated_op();
        register_GeneratedSaintVenantKirchhoff_generated_op();
        register_GeneratedStokes_generated_op();
        register_GeneratedTwoPhaseFlow_generated_op();
    }
}  // namespace sfem
