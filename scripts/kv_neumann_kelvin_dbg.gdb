set breakpoint pending on
set environment MPICH_GPU_SUPPORT_ENABLED 0
set environment SFEM_EXECUTION_SPACE device

break /users/hyang/ws/sfem_github/sfem/frontend/tests/sfem_NewmarkKVTest.cpp:332 if steps == 1

break /users/hyang/ws/sfem_github/sfem/operators/quadshell4/cuda/cu_quadshell4_integrate_values.cu:82 if threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0
commands
  silent
  printf "\n==== NEUMANN ELEMENT (thread %lld) ====\n", (long long)e
  printf "xe: %g %g %g %g\n", (double)xe[0 * coords_stride], (double)xe[1 * coords_stride], (double)xe[2 * coords_stride], (double)xe[3 * coords_stride]
  printf "ye: %g %g %g %g\n", (double)ye[0 * coords_stride], (double)ye[1 * coords_stride], (double)ye[2 * coords_stride], (double)ye[3 * coords_stride]
  printf "ze: %g %g %g %g\n", (double)ze[0 * coords_stride], (double)ze[1 * coords_stride], (double)ze[2 * coords_stride], (double)ze[3 * coords_stride]
  continue
end

break /users/hyang/ws/sfem_github/sfem/operators/quadshell4/cuda/cu_quadshell4_integrate_values.cu:101 if threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0
commands
  silent
  printf "  element_vector=%g %g %g %g\n", (double)element_vector[0], (double)element_vector[1], (double)element_vector[2], (double)element_vector[3]
  continue
end

break cu_hex8_kelvin_voigt_apply_adj<double,double> if threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && k == 0
commands
  silent
  printf "\n==== KELVIN ELEMENT ====\n"
  printf "denom=%g qw=%g\n", (double)jacobian_determinant, (double)qw
  printf "disp_grad=%g %g %g %g %g %g %g %g %g\n", (double)disp_grad[0], (double)disp_grad[1], (double)disp_grad[2], (double)disp_grad[3], (double)disp_grad[4], (double)disp_grad[5], (double)disp_grad[6], (double)disp_grad[7], (double)disp_grad[8]
  printf "velo_grad=%g %g %g %g %g %g %g %g %g\n", (double)velo_grad[0], (double)velo_grad[1], (double)velo_grad[2], (double)velo_grad[3], (double)velo_grad[4], (double)velo_grad[5], (double)velo_grad[6], (double)velo_grad[7], (double)velo_grad[8]
  continue
end

run
