#include "isolver_lsolve.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  isolver_lsolve_t lsolve;
  lsolve.comm = MPI_COMM_WORLD;

  const isolver_idx_t rowptr[3] = {0, 1, 2};
  const isolver_idx_t colidx[2] = {0, 1};
  const isolver_scalar_t values[2] = {1, 1};
  const isolver_scalar_t rhs[2] = {2, 2};
  isolver_scalar_t x[2] = {0, 0};

  isolver_lsolve_init(&lsolve);

  isolver_lsolve_set_max_iterations(&lsolve, 1);
  isolver_lsolve_set_atol(&lsolve, 1e-16);
  isolver_lsolve_set_stol(&lsolve, 1e-8);
  isolver_lsolve_set_rtol(&lsolve, 1e-16);
  isolver_lsolve_set_verbosity(&lsolve, 1);

  isolver_lsolve_update_crs(&lsolve, 2, 2, rowptr, colidx, values);
  isolver_lsolve_apply(&lsolve, rhs, x);
  
  isolver_lsolve_destroy(&lsolve);

  return MPI_Finalize();
}
