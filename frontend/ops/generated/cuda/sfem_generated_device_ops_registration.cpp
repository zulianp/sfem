#include "sfem_generated_device_ops_registration.hpp"

namespace sfem {
  void register_GeneratedBodyForce_cuda_generated_op();
  void register_GeneratedLaplace_cuda_generated_op();
  void register_GeneratedLinearElasticity_cuda_generated_op();
  void register_GeneratedMooneyRivlinKelvinVoigtNewmark_cuda_generated_op();
  void register_GeneratedNavierStokes_cuda_generated_op();
  void register_GeneratedNeoHookeanOgden_cuda_generated_op();
  void register_GeneratedNeumann_cuda_generated_op();
  void register_GeneratedNeumannGeneral_cuda_generated_op();
  void register_GeneratedTwoPhaseFlow_cuda_generated_op();

  void register_generated_device_ops() {
    register_GeneratedBodyForce_cuda_generated_op();
    register_GeneratedLaplace_cuda_generated_op();
    register_GeneratedLinearElasticity_cuda_generated_op();
    register_GeneratedMooneyRivlinKelvinVoigtNewmark_cuda_generated_op();
    register_GeneratedNavierStokes_cuda_generated_op();
    register_GeneratedNeoHookeanOgden_cuda_generated_op();
    register_GeneratedNeumann_cuda_generated_op();
    register_GeneratedNeumannGeneral_cuda_generated_op();
    register_GeneratedTwoPhaseFlow_cuda_generated_op();
  }
}  // namespace sfem
