#include "sfem_generated_ops_registration.hpp"

namespace sfem {
    void register_GeneratedNeoHookeanOgden_generated_op();
    void register_GeneratedPoroHyperelasticity_generated_op();
    void register_GeneratedStokes_generated_op();
    void register_GeneratedTwoPhaseFlow_generated_op();

    void register_generated_ops() {
        register_GeneratedNeoHookeanOgden_generated_op();
        register_GeneratedPoroHyperelasticity_generated_op();
        register_GeneratedStokes_generated_op();
        register_GeneratedTwoPhaseFlow_generated_op();
    }
}  // namespace sfem
