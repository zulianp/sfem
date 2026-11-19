#ifndef SFEM_CUDA_CRS_H
#define SFEM_CUDA_CRS_H

#include "sfem_base.hpp"

void crs_device_create(const ptrdiff_t nnodes,
                       const ptrdiff_t nnz,
                       count_t** rowptr,
                       idx_t** colidx,
                       real_t** values);

void crs_graph_device_create(const ptrdiff_t nnodes,
                             const ptrdiff_t nnz,
                             count_t** rowptr,
                             idx_t** colidx);

void crs_device_free(count_t* rowptr, idx_t* colidx, real_t* values);
void crs_graph_device_free(count_t* rowptr, idx_t* colidx);

#endif
