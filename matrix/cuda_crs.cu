
extern "C" {
#include "sfem_base.h"
}

#include "sfem_cuda_base.h"


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

extern "C" void crs_graph_host_to_device(const ptrdiff_t nnodes,
                                         const ptrdiff_t nnz,
                                         const count_t* const SFEM_RESTRICT h_rowptr,
                                         const idx_t* const SFEM_RESTRICT h_colidx,
                                         count_t* const SFEM_RESTRICT d_rowptr,
                                         idx_t* const SFEM_RESTRICT d_colidx) {
    SFEM_CUDA_CHECK(cudaMemcpy(d_rowptr, h_rowptr, (nnodes + 1) * sizeof(count_t), cudaMemcpyHostToDevice));
    SFEM_CUDA_CHECK(cudaMemcpy(d_colidx, h_colidx, nnz * sizeof(idx_t), cudaMemcpyHostToDevice));
}
