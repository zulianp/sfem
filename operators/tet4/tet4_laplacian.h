#ifndef TET4_LAPLACIAN_H
#define TET4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#include "tet4_fff.h"

// UNTESTED
void tet4_laplacian_assemble_value(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const SFEM_RESTRICT elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t *const SFEM_RESTRICT u,
                                   real_t *const SFEM_RESTRICT value);

void tet4_laplacian_apply(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points,
                          const real_t *const SFEM_RESTRICT u,
                          real_t *const SFEM_RESTRICT values);

/**
 * @brief Assembles the gradient of the Laplacian matrix for tetrahedral elements.
 *
 * Given an array of tetrahedral elements and their corresponding nodal coordinates, as well as a
 * solution vector 'u', computes the gradient of the Laplacian matrix for each element and assembles
 * it into the global gradient vector 'values'.
 *
 * @param[in] nelements Number of tetrahedral elements in the mesh.
 * @param[in] nnodes Number of nodes in the mesh (unused in this function).
 * @param[in] elements Array of size 4xnelements containing the node IDs of each tetrahedral
 * element.
 * @param[in] points Array of size 3xnnodes containing the coordinates of each node.
 * @param[in] u Array of size nnodes containing the solution vector 'u'.
 * @param[out] values Array of size nnodes where the assembled gradient vector will be stored.
 */
void tet4_laplacian_assemble_gradient(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elements,
                                      geom_t **const SFEM_RESTRICT points,
                                      const real_t *const SFEM_RESTRICT u,
                                      real_t *const SFEM_RESTRICT values);

void tet4_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const count_t *const SFEM_RESTRICT rowptr,
                                     const idx_t *const SFEM_RESTRICT colidx,
                                     real_t *const SFEM_RESTRICT values);

void tet4_laplacian_diag(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elements,
                         geom_t **const SFEM_RESTRICT points,
                         real_t *const SFEM_RESTRICT diag);

int tet4_laplacian_apply_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values);

int tet4_laplacian_diag_opt(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            const jacobian_t *const SFEM_RESTRICT fff,
                            real_t *const SFEM_RESTRICT diag);

#endif  // TET4_LAPLACIAN_H
