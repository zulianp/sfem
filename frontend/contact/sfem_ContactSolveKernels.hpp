#ifndef SFEM_CONTACT_SOLVE_KERNELS_HPP
#define SFEM_CONTACT_SOLVE_KERNELS_HPP

#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_macros.hpp"

#include <stddef.h>

namespace sfem {
    void compute_macaulay_term(const int                                              dim,
                               const ptrdiff_t                                        nnodes,
                               const count_t* const SFEM_RESTRICT                     cm_rowptr,
                               const idx_t* const SFEM_RESTRICT                       cm_colidx,
                               const real_t* const SFEM_RESTRICT                      cm_vals,
                               const real_t* const SFEM_RESTRICT                      distances,
                               const real_t* const SFEM_RESTRICT                      agumentation,
                               const real_t* const* const SFEM_RESTRICT               normals,
                               const real_t* const SFEM_RESTRICT                      mass,
                               const real_t                                           penalty,
                               const ptrdiff_t                                        in_stride,
                               const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in_old,
                               const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                               real_t* const                                          macaulay);

    void assemble_contact_gradient(const int                                dim,
                                   const ptrdiff_t                          nnodes,
                                   const real_t                             penalty,
                                   const count_t* const SFEM_RESTRICT       cm_rowptr,
                                   const idx_t* const SFEM_RESTRICT         cm_colidx,
                                   const real_t* const SFEM_RESTRICT        cm_vals,
                                   const real_t* const SFEM_RESTRICT        distances,
                                   const real_t* const SFEM_RESTRICT        agumentation,
                                   const real_t* const* const SFEM_RESTRICT normals,
                                   const real_t* const SFEM_RESTRICT        mass,
                                   const real_t* const SFEM_RESTRICT        macaulay,
                                   real_t* const SFEM_RESTRICT              grad);

    void assemble_contact_hessian_diag_block(const int                                        dim,
                                             const ptrdiff_t                                  nnodes,
                                             const count_t* const SFEM_RESTRICT               cm_rowptr,
                                             const idx_t* const SFEM_RESTRICT                 cm_colidx,
                                             const real_t* const SFEM_RESTRICT                cm_vals,
                                             const real_t* const SFEM_RESTRICT                distances,
                                             const real_t* const SFEM_RESTRICT                agumentation,
                                             const real_t* const* const SFEM_RESTRICT         normals,
                                             const real_t* const SFEM_RESTRICT                mass,
                                             const real_t                                     penalty,
                                             const real_t* const SFEM_RESTRICT                macaulay,
                                             const ptrdiff_t                                  diag_stride,
                                             real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values);

    void contact_hessian_apply(const int                                              dim,
                               const ptrdiff_t                                        nnodes,
                               const count_t* const SFEM_RESTRICT                     cm_rowptr,
                               const idx_t* const SFEM_RESTRICT                       cm_colidx,
                               const real_t* const SFEM_RESTRICT                      cm_vals,
                               const real_t* const* const SFEM_RESTRICT               normals,
                               const real_t* const SFEM_RESTRICT                      mass,
                               const real_t                                           penalty,
                               const real_t* const SFEM_RESTRICT                      macaulay,
                               const ptrdiff_t                                        in_stride,
                               const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                               const ptrdiff_t                                        out_stride,
                               real_t* const SFEM_RESTRICT* const SFEM_RESTRICT       out_values);

    void gather_combine_hessian_diag(const int                                              dim,
                                     const ptrdiff_t                                        n_contact_nodes,
                                     const idx_t* const                                     node_mapping,
                                     const ptrdiff_t                                        elasticity_diag_stride,
                                     const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT elasticity_diag_values,
                                     const ptrdiff_t                                        contact_diag_stride,
                                     real_t* const SFEM_RESTRICT* const SFEM_RESTRICT       contact_diag_values);

    void compute_penetration(const int                                              dim,
                             const ptrdiff_t                                        nnodes,
                             const count_t* const SFEM_RESTRICT                     cm_rowptr,
                             const idx_t* const SFEM_RESTRICT                       cm_colidx,
                             const real_t* const SFEM_RESTRICT                      cm_vals,
                             const real_t* const* SFEM_RESTRICT const SFEM_RESTRICT normals,
                             const real_t* const SFEM_RESTRICT                      gap,
                             const ptrdiff_t                                        in_stride,
                             const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in_old,
                             const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                             real_t* const SFEM_RESTRICT                            penetration);
}  // namespace sfem

#endif  // SFEM_CONTACT_SOLVE_KERNELS_HPP
