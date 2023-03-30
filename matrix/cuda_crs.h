#ifndef SFEM_CUDA_CRS_H
#define SFEM_CUDA_CRS_H

#include "sfem_base.h"

void crs_device_create(const ptrdiff_t nnodes, const ptrdiff_t nnz, count_t** rowptr, idx_t** colidx, real_t** values);

void crs_device_free(count_t* rowptr, idx_t* colidx, real_t* values);

void crs_graph_host_to_device(const ptrdiff_t nnodes,
                              const ptrdiff_t nnz,
                              const count_t* const SFEM_RESTRICT h_rowptr,
                              const idx_t* const SFEM_RESTRICT h_colidx,
                              count_t* const SFEM_RESTRICT d_rowptr,
                              idx_t* const SFEM_RESTRICT d_colidx);

#endif
