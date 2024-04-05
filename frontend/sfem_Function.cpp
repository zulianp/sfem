#include "sfem_Function.hpp"

namespace sfem {

    int Function::create_matrix_crs(ptrdiff_t *nlocal,
                                    ptrdiff_t *nglobal,
                                    ptrdiff_t *nnz,
                                    isolver_idx_t **rowptr,
                                    isolver_idx_t **colidx) {
    	// TODO
    	return ISOLVER_FUNCTION_SUCCESS;
    }

}  // namespace sfem
