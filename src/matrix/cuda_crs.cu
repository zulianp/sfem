
extern "C" {
#include "sfem_base.hpp"
}

#include "sfem_cuda_base.hpp"


extern "C" void crs_device_create(const ptrdiff_t nnodes,
                                  const ptrdiff_t nnz,
                                  count_t** rowptr,
                                  idx_t** colidx,
                                  real_t** values)

{  
    SFEM_CUDA_CHECK(cudaMalloc(rowptr, (nnodes + 1) * sizeof(count_t)));
    SFEM_CUDA_CHECK(cudaMalloc(colidx, nnz * sizeof(idx_t)));
    SFEM_CUDA_CHECK(cudaMalloc(values, nnz * sizeof(real_t)));
}

extern "C" void crs_graph_device_create(const ptrdiff_t nnodes,
                                  const ptrdiff_t nnz,
                                  count_t** rowptr,
                                  idx_t** colidx)

{  
    SFEM_CUDA_CHECK(cudaMalloc(rowptr, (nnodes + 1) * sizeof(count_t)));
    SFEM_CUDA_CHECK(cudaMalloc(colidx, nnz * sizeof(idx_t)));
}

extern "C" void crs_device_free(count_t* rowptr, idx_t* colidx, real_t* values) {
    SFEM_CUDA_CHECK(cudaFree(rowptr));
    SFEM_CUDA_CHECK(cudaFree(colidx));
    SFEM_CUDA_CHECK(cudaFree(values));
}

extern "C" void crs_graph_device_free(count_t* rowptr, idx_t* colidx) {
    SFEM_CUDA_CHECK(cudaFree(rowptr));
    SFEM_CUDA_CHECK(cudaFree(colidx));
}
