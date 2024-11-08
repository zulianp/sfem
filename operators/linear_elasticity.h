#ifndef LINEAR_ELASTICITY_H
#define LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//////////////////////////
// Structure of arrays
//////////////////////////

int linear_elasticity_assemble_value_soa(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elems,
                                         geom_t **const SFEM_RESTRICT xyz,
                                         const real_t mu,
                                         const real_t lambda,
                                         const real_t **const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT value);

int linear_elasticity_assemble_gradient_soa(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t mu,
                                            const real_t lambda,
                                            const real_t **const SFEM_RESTRICT u,
                                            real_t **const SFEM_RESTRICT values);

int linear_elasticity_crs_soa(const enum ElemType element_type,
                              const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elems,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t mu,
                              const real_t lambda,
                              const count_t *const SFEM_RESTRICT rowptr,
                              const idx_t *const SFEM_RESTRICT colidx,
                              real_t **const SFEM_RESTRICT values);

int linear_elasticity_assemble_diag_aos(const enum ElemType element_type,
                                        const ptrdiff_t nelements,
                                        const ptrdiff_t nnodes,
                                        idx_t **const SFEM_RESTRICT elems,
                                        geom_t **const SFEM_RESTRICT xyz,
                                        const real_t mu,
                                        const real_t lambda,
                                        real_t *const SFEM_RESTRICT values);

int linear_elasticity_apply_soa(const enum ElemType element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t mu,
                                const real_t lambda,
                                const real_t **const SFEM_RESTRICT u,
                                real_t **const SFEM_RESTRICT values);

//////////////////////////
// Array of structures
//////////////////////////

int linear_elasticity_assemble_value_aos(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elems,
                                         geom_t **const SFEM_RESTRICT xyz,
                                         const real_t mu,
                                         const real_t lambda,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT value);

int linear_elasticity_assemble_gradient_aos(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t mu,
                                            const real_t lambda,
                                            const real_t *const SFEM_RESTRICT u,
                                            real_t *const SFEM_RESTRICT values);

int linear_elasticity_crs_aos(const enum ElemType element_type,
                              const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elems,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t mu,
                              const real_t lambda,
                              const count_t *const SFEM_RESTRICT rowptr,
                              const idx_t *const SFEM_RESTRICT colidx,
                              real_t *const SFEM_RESTRICT values);

int linear_elasticity_apply_aos(const enum ElemType element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t mu,
                                const real_t lambda,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values);

// Block sparse row (BSR) https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-storage-formats
int linear_elasticity_bsr(const enum ElemType element_type,
                          const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          const real_t mu,
                          const real_t lambda,
                          const count_t *const SFEM_RESTRICT rowptr,
                          const idx_t *const SFEM_RESTRICT colidx,
                          real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // LINEAR_ELASTICITY_H
