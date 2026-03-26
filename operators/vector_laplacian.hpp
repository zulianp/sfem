#ifndef VECTOR_LAPLACIAN_H
#define VECTOR_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.hpp"
#include "smesh_elem_type.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int vector_laplacian_apply(smesh::ElemType               element_type,
                           const ptrdiff_t               nelements,
                           const ptrdiff_t               nnodes,
                           idx_t **const SFEM_RESTRICT   elements,
                           geom_t **const SFEM_RESTRICT  points,
                           const int                     vector_size,
                           const ptrdiff_t               stride,
                           real_t **const SFEM_RESTRICT  u,
                           real_t **const SFEM_RESTRICT  values);

int vector_laplacian_apply_opt(smesh::ElemType                     element_type,
                               const ptrdiff_t                     nelements,
                               idx_t **const SFEM_RESTRICT           elements,
                               const jacobian_t *const SFEM_RESTRICT fff,
                               const int                           vector_size,
                               const ptrdiff_t                     stride,
                               real_t **const SFEM_RESTRICT          u,
                               real_t **const SFEM_RESTRICT          values);

#ifdef __cplusplus
}
#endif

#endif  // VECTOR_LAPLACIAN_H
