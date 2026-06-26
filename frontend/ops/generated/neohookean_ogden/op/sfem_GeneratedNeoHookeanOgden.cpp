#include "sfem_GeneratedNeoHookeanOgden.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

extern "C" {
int generated_neohookean_ogden_tri3_tri3_objective_isoparametric_mesh_soa(ptrdiff_t,
                                                                          ptrdiff_t,
                                                                          idx_t **,
                                                                          const geom_t *const *,
                                                                          const real_t mu,
                                                                          const real_t lmbda,
                                                                          ptrdiff_t,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          real_t *);
int generated_neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                         ptrdiff_t,
                                                                         idx_t **,
                                                                         const geom_t *const *,
                                                                         const real_t mu,
                                                                         const real_t lmbda,
                                                                         ptrdiff_t,
                                                                         const real_t *,
                                                                         const real_t *,
                                                                         ptrdiff_t,
                                                                         real_t *,
                                                                         real_t *);
int generated_neohookean_ogden_tri3_tri3_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                      ptrdiff_t,
                                                                      idx_t **,
                                                                      const geom_t *const *,
                                                                      const real_t mu,
                                                                      const real_t lmbda,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      real_t *,
                                                                      real_t *);
int generated_neohookean_ogden_tri3_tri3_objective_affine_mesh_soa(ptrdiff_t,
                                                                   ptrdiff_t,
                                                                   idx_t **,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t mu,
                                                                   const real_t lmbda,
                                                                   ptrdiff_t,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   real_t *);
int generated_neohookean_ogden_tri3_tri3_gradient_affine_mesh_soa(ptrdiff_t,
                                                                  ptrdiff_t,
                                                                  idx_t **,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t mu,
                                                                  const real_t lmbda,
                                                                  ptrdiff_t,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  ptrdiff_t,
                                                                  real_t *,
                                                                  real_t *);
int generated_neohookean_ogden_tri3_tri3_apply_affine_mesh_soa(ptrdiff_t,
                                                               ptrdiff_t,
                                                               idx_t **,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t mu,
                                                               const real_t lmbda,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               real_t *,
                                                               real_t *);
int generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa(ptrdiff_t,
                                                                          ptrdiff_t,
                                                                          idx_t **,
                                                                          const geom_t *const *,
                                                                          const real_t mu,
                                                                          const real_t lmbda,
                                                                          ptrdiff_t,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          real_t *);
int generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                         ptrdiff_t,
                                                                         idx_t **,
                                                                         const geom_t *const *,
                                                                         const real_t mu,
                                                                         const real_t lmbda,
                                                                         ptrdiff_t,
                                                                         const real_t *,
                                                                         const real_t *,
                                                                         ptrdiff_t,
                                                                         real_t *,
                                                                         real_t *);
int generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                      ptrdiff_t,
                                                                      idx_t **,
                                                                      const geom_t *const *,
                                                                      const real_t mu,
                                                                      const real_t lmbda,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      real_t *,
                                                                      real_t *);
int generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa(ptrdiff_t,
                                                                   ptrdiff_t,
                                                                   idx_t **,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t mu,
                                                                   const real_t lmbda,
                                                                   ptrdiff_t,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   real_t *);
int generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa(ptrdiff_t,
                                                                  ptrdiff_t,
                                                                  idx_t **,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t mu,
                                                                  const real_t lmbda,
                                                                  ptrdiff_t,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  ptrdiff_t,
                                                                  real_t *,
                                                                  real_t *);
int generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa(ptrdiff_t,
                                                               ptrdiff_t,
                                                               idx_t **,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t mu,
                                                               const real_t lmbda,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               real_t *,
                                                               real_t *);
int generated_neohookean_ogden_quad4_quad4_objective_isoparametric_mesh_soa(ptrdiff_t,
                                                                            ptrdiff_t,
                                                                            idx_t **,
                                                                            const geom_t *const *,
                                                                            const real_t mu,
                                                                            const real_t lmbda,
                                                                            ptrdiff_t,
                                                                            const real_t *,
                                                                            const real_t *,
                                                                            real_t *);
int generated_neohookean_ogden_quad4_quad4_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                           ptrdiff_t,
                                                                           idx_t **,
                                                                           const geom_t *const *,
                                                                           const real_t mu,
                                                                           const real_t lmbda,
                                                                           ptrdiff_t,
                                                                           const real_t *,
                                                                           const real_t *,
                                                                           ptrdiff_t,
                                                                           real_t *,
                                                                           real_t *);
int generated_neohookean_ogden_quad4_quad4_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                        ptrdiff_t,
                                                                        idx_t **,
                                                                        const geom_t *const *,
                                                                        const real_t mu,
                                                                        const real_t lmbda,
                                                                        ptrdiff_t,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        ptrdiff_t,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        ptrdiff_t,
                                                                        real_t *,
                                                                        real_t *);
int generated_neohookean_ogden_quad4_quad4_objective_affine_mesh_soa(ptrdiff_t,
                                                                     ptrdiff_t,
                                                                     idx_t **,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t mu,
                                                                     const real_t lmbda,
                                                                     ptrdiff_t,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     real_t *);
int generated_neohookean_ogden_quad4_quad4_gradient_affine_mesh_soa(ptrdiff_t,
                                                                    ptrdiff_t,
                                                                    idx_t **,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t mu,
                                                                    const real_t lmbda,
                                                                    ptrdiff_t,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    ptrdiff_t,
                                                                    real_t *,
                                                                    real_t *);
int generated_neohookean_ogden_quad4_quad4_apply_affine_mesh_soa(ptrdiff_t,
                                                                 ptrdiff_t,
                                                                 idx_t **,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t mu,
                                                                 const real_t lmbda,
                                                                 ptrdiff_t,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 ptrdiff_t,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 ptrdiff_t,
                                                                 real_t *,
                                                                 real_t *);
int generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa(ptrdiff_t,
                                                                          ptrdiff_t,
                                                                          idx_t **,
                                                                          const geom_t *const *,
                                                                          const real_t mu,
                                                                          const real_t lmbda,
                                                                          ptrdiff_t,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          real_t *);
int generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                         ptrdiff_t,
                                                                         idx_t **,
                                                                         const geom_t *const *,
                                                                         const real_t mu,
                                                                         const real_t lmbda,
                                                                         ptrdiff_t,
                                                                         const real_t *,
                                                                         const real_t *,
                                                                         const real_t *,
                                                                         ptrdiff_t,
                                                                         real_t *,
                                                                         real_t *,
                                                                         real_t *);
int generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                      ptrdiff_t,
                                                                      idx_t **,
                                                                      const geom_t *const *,
                                                                      const real_t mu,
                                                                      const real_t lmbda,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      real_t *,
                                                                      real_t *,
                                                                      real_t *);
int generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa(ptrdiff_t,
                                                                   ptrdiff_t,
                                                                   idx_t **,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t mu,
                                                                   const real_t lmbda,
                                                                   ptrdiff_t,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   real_t *);
int generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa(ptrdiff_t,
                                                                  ptrdiff_t,
                                                                  idx_t **,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t mu,
                                                                  const real_t lmbda,
                                                                  ptrdiff_t,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  ptrdiff_t,
                                                                  real_t *,
                                                                  real_t *,
                                                                  real_t *);
int generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa(ptrdiff_t,
                                                               ptrdiff_t,
                                                               idx_t **,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t mu,
                                                               const real_t lmbda,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               real_t *,
                                                               real_t *,
                                                               real_t *);
int generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa(ptrdiff_t,
                                                                            ptrdiff_t,
                                                                            idx_t **,
                                                                            const geom_t *const *,
                                                                            const real_t mu,
                                                                            const real_t lmbda,
                                                                            ptrdiff_t,
                                                                            const real_t *,
                                                                            const real_t *,
                                                                            const real_t *,
                                                                            real_t *);
int generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                           ptrdiff_t,
                                                                           idx_t **,
                                                                           const geom_t *const *,
                                                                           const real_t mu,
                                                                           const real_t lmbda,
                                                                           ptrdiff_t,
                                                                           const real_t *,
                                                                           const real_t *,
                                                                           const real_t *,
                                                                           ptrdiff_t,
                                                                           real_t *,
                                                                           real_t *,
                                                                           real_t *);
int generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                        ptrdiff_t,
                                                                        idx_t **,
                                                                        const geom_t *const *,
                                                                        const real_t mu,
                                                                        const real_t lmbda,
                                                                        ptrdiff_t,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        ptrdiff_t,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        ptrdiff_t,
                                                                        real_t *,
                                                                        real_t *,
                                                                        real_t *);
int generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa(ptrdiff_t,
                                                                     ptrdiff_t,
                                                                     idx_t **,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t mu,
                                                                     const real_t lmbda,
                                                                     ptrdiff_t,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     real_t *);
int generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa(ptrdiff_t,
                                                                    ptrdiff_t,
                                                                    idx_t **,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t mu,
                                                                    const real_t lmbda,
                                                                    ptrdiff_t,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    ptrdiff_t,
                                                                    real_t *,
                                                                    real_t *,
                                                                    real_t *);
int generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa(ptrdiff_t,
                                                                 ptrdiff_t,
                                                                 idx_t **,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t mu,
                                                                 const real_t lmbda,
                                                                 ptrdiff_t,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 ptrdiff_t,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 ptrdiff_t,
                                                                 real_t *,
                                                                 real_t *,
                                                                 real_t *);
int generated_neohookean_ogden_hex8_hex8_objective_isoparametric_mesh_soa(ptrdiff_t,
                                                                          ptrdiff_t,
                                                                          idx_t **,
                                                                          const geom_t *const *,
                                                                          const real_t mu,
                                                                          const real_t lmbda,
                                                                          ptrdiff_t,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          real_t *);
int generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                         ptrdiff_t,
                                                                         idx_t **,
                                                                         const geom_t *const *,
                                                                         const real_t mu,
                                                                         const real_t lmbda,
                                                                         ptrdiff_t,
                                                                         const real_t *,
                                                                         const real_t *,
                                                                         const real_t *,
                                                                         ptrdiff_t,
                                                                         real_t *,
                                                                         real_t *,
                                                                         real_t *);
int generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                      ptrdiff_t,
                                                                      idx_t **,
                                                                      const geom_t *const *,
                                                                      const real_t mu,
                                                                      const real_t lmbda,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      const real_t *,
                                                                      ptrdiff_t,
                                                                      real_t *,
                                                                      real_t *,
                                                                      real_t *);
int generated_neohookean_ogden_hex8_hex8_objective_affine_mesh_soa(ptrdiff_t,
                                                                   ptrdiff_t,
                                                                   idx_t **,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t mu,
                                                                   const real_t lmbda,
                                                                   ptrdiff_t,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   const real_t *,
                                                                   real_t *);
int generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa(ptrdiff_t,
                                                                  ptrdiff_t,
                                                                  idx_t **,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t mu,
                                                                  const real_t lmbda,
                                                                  ptrdiff_t,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  const real_t *,
                                                                  ptrdiff_t,
                                                                  real_t *,
                                                                  real_t *,
                                                                  real_t *);
int generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa(ptrdiff_t,
                                                               ptrdiff_t,
                                                               idx_t **,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t mu,
                                                               const real_t lmbda,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               const real_t *,
                                                               const real_t *,
                                                               const real_t *,
                                                               ptrdiff_t,
                                                               real_t *,
                                                               real_t *,
                                                               real_t *);
int generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa(ptrdiff_t,
                                                                            ptrdiff_t,
                                                                            idx_t **,
                                                                            const geom_t *const *,
                                                                            const real_t mu,
                                                                            const real_t lmbda,
                                                                            ptrdiff_t,
                                                                            const real_t *,
                                                                            const real_t *,
                                                                            const real_t *,
                                                                            real_t *);
int generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                           ptrdiff_t,
                                                                           idx_t **,
                                                                           const geom_t *const *,
                                                                           const real_t mu,
                                                                           const real_t lmbda,
                                                                           ptrdiff_t,
                                                                           const real_t *,
                                                                           const real_t *,
                                                                           const real_t *,
                                                                           ptrdiff_t,
                                                                           real_t *,
                                                                           real_t *,
                                                                           real_t *);
int generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                        ptrdiff_t,
                                                                        idx_t **,
                                                                        const geom_t *const *,
                                                                        const real_t mu,
                                                                        const real_t lmbda,
                                                                        ptrdiff_t,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        ptrdiff_t,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        const real_t *,
                                                                        ptrdiff_t,
                                                                        real_t *,
                                                                        real_t *,
                                                                        real_t *);
int generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa(ptrdiff_t,
                                                                     ptrdiff_t,
                                                                     idx_t **,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t mu,
                                                                     const real_t lmbda,
                                                                     ptrdiff_t,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     const real_t *,
                                                                     real_t *);
int generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa(ptrdiff_t,
                                                                    ptrdiff_t,
                                                                    idx_t **,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t mu,
                                                                    const real_t lmbda,
                                                                    ptrdiff_t,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    const real_t *,
                                                                    ptrdiff_t,
                                                                    real_t *,
                                                                    real_t *,
                                                                    real_t *);
int generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa(ptrdiff_t,
                                                                 ptrdiff_t,
                                                                 idx_t **,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t mu,
                                                                 const real_t lmbda,
                                                                 ptrdiff_t,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 ptrdiff_t,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 const real_t *,
                                                                 ptrdiff_t,
                                                                 real_t *,
                                                                 real_t *,
                                                                 real_t *);
}

namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
            parameters.set_value("mu", 1);
            parameters.set_value("lmbda", 1);
        }

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh, const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("GeneratedNeoHookeanOgden: mesh block pointer not found in mesh.blocks()\n");
            return 0;
        }
    }  // namespace

    class GeneratedNeoHookeanOgden::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]>      element_values;
        ptrdiff_t                      element_capacity{0};
        bool                           objective_uses_affine{false};
        bool                           gradient_uses_affine{false};

        bool apply_uses_affine{false};
    };

    std::unique_ptr<Op> GeneratedNeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != space->mesh_ptr()->spatial_dimension()) {
            SFEM_ERROR("GeneratedNeoHookeanOgden requires block_size=spatial_dimension\n");
            return nullptr;
        }
        auto op = std::make_unique<GeneratedNeoHookeanOgden>(space);
        op->initialize();
        return op;
    }

    GeneratedNeoHookeanOgden::GeneratedNeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedNeoHookeanOgden::~GeneratedNeoHookeanOgden() = default;

    ptrdiff_t GeneratedNeoHookeanOgden::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedNeoHookeanOgden::n_dofs_image() const { return impl_->space->n_dofs(); }

    int GeneratedNeoHookeanOgden::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        auto mesh      = impl_->space->mesh_ptr();
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity           = std::max(impl_->element_capacity, entry.second.block->n_elements());
            const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *entry.second.block);
            auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(mesh, smesh::MEMORY_SPACE_HOST, block_id);
            if (!jacobian) {
                return SFEM_FAILURE;
            }
            entry.second.user_data = std::static_pointer_cast<void>(jacobian);
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        return SFEM_SUCCESS;
    }

    int GeneratedNeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        auto mesh   = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const real_t *const *adjugate    = nullptr;
            const real_t        *determinant = nullptr;
            if (impl_->gradient_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedNeoHookeanOgden affine gradient requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate    = reinterpret_cast<const real_t *const *>(jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(jacobian->jacobian_determinant()->data());
            }
            switch (domain.element_type) {
                case smesh::TRI3:
                    return impl_->gradient_uses_affine ? generated_neohookean_ogden_tri3_tri3_gradient_affine_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 adjugate[0],
                                                                 adjugate[1],
                                                                 adjugate[2],
                                                                 adjugate[3],
                                                                 determinant,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 2,
                                                                 x + 0,
                                                                 x + 1,
                                                                 2,
                                                                 out + 0,
                                                                 out + 1)
                                                       : generated_neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 points,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 2,
                                                                 x + 0,
                                                                 x + 1,
                                                                 2,
                                                                 out + 0,
                                                                 out + 1);
                case smesh::TRI6:
                    return impl_->gradient_uses_affine ? generated_neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 adjugate[0],
                                                                 adjugate[1],
                                                                 adjugate[2],
                                                                 adjugate[3],
                                                                 determinant,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 2,
                                                                 x + 0,
                                                                 x + 1,
                                                                 2,
                                                                 out + 0,
                                                                 out + 1)
                                                       : generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 points,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 2,
                                                                 x + 0,
                                                                 x + 1,
                                                                 2,
                                                                 out + 0,
                                                                 out + 1);
                case smesh::QUAD4:
                    return impl_->gradient_uses_affine ? generated_neohookean_ogden_quad4_quad4_gradient_affine_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 adjugate[0],
                                                                 adjugate[1],
                                                                 adjugate[2],
                                                                 adjugate[3],
                                                                 determinant,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 2,
                                                                 x + 0,
                                                                 x + 1,
                                                                 2,
                                                                 out + 0,
                                                                 out + 1)
                                                       : generated_neohookean_ogden_quad4_quad4_gradient_isoparametric_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 points,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 2,
                                                                 x + 0,
                                                                 x + 1,
                                                                 2,
                                                                 out + 0,
                                                                 out + 1);
                case smesh::TET4:
                    return impl_->gradient_uses_affine ? generated_neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 adjugate[0],
                                                                 adjugate[1],
                                                                 adjugate[2],
                                                                 adjugate[3],
                                                                 adjugate[4],
                                                                 adjugate[5],
                                                                 adjugate[6],
                                                                 adjugate[7],
                                                                 adjugate[8],
                                                                 determinant,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2)
                                                       : generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 points,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2);
                case smesh::TET10:
                    return impl_->gradient_uses_affine ? generated_neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 adjugate[0],
                                                                 adjugate[1],
                                                                 adjugate[2],
                                                                 adjugate[3],
                                                                 adjugate[4],
                                                                 adjugate[5],
                                                                 adjugate[6],
                                                                 adjugate[7],
                                                                 adjugate[8],
                                                                 determinant,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2)
                                                       : generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 points,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2);
                case smesh::HEX8:
                    return impl_->gradient_uses_affine ? generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 adjugate[0],
                                                                 adjugate[1],
                                                                 adjugate[2],
                                                                 adjugate[3],
                                                                 adjugate[4],
                                                                 adjugate[5],
                                                                 adjugate[6],
                                                                 adjugate[7],
                                                                 adjugate[8],
                                                                 determinant,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2)
                                                       : generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 points,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2);
                case smesh::HEX27:
                    return impl_->gradient_uses_affine ? generated_neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 adjugate[0],
                                                                 adjugate[1],
                                                                 adjugate[2],
                                                                 adjugate[3],
                                                                 adjugate[4],
                                                                 adjugate[5],
                                                                 adjugate[6],
                                                                 adjugate[7],
                                                                 adjugate[8],
                                                                 determinant,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2)
                                                       : generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa(
                                                                 domain.block->n_elements(),
                                                                 mesh->n_nodes(),
                                                                 domain.block->elements()->data(),
                                                                 points,
                                                                 domain.parameters->require_real_value("mu"),
                                                                 domain.parameters->require_real_value("lmbda"),
                                                                 3,
                                                                 x + 0,
                                                                 x + 1,
                                                                 x + 2,
                                                                 3,
                                                                 out + 0,
                                                                 out + 1,
                                                                 out + 2);
                default:
                    SFEM_ERROR("GeneratedNeoHookeanOgden does not support element type %d\n", domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedNeoHookeanOgden::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        auto mesh   = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const real_t *const *adjugate    = nullptr;
            const real_t        *determinant = nullptr;
            if (impl_->apply_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedNeoHookeanOgden affine hessian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate    = reinterpret_cast<const real_t *const *>(jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(jacobian->jacobian_determinant()->data());
            }
            switch (domain.element_type) {
                case smesh::TRI3:
                    return impl_->apply_uses_affine ? generated_neohookean_ogden_tri3_tri3_apply_affine_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              adjugate[0],
                                                              adjugate[1],
                                                              adjugate[2],
                                                              adjugate[3],
                                                              determinant,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              2,
                                                              x + 0,
                                                              x + 1,
                                                              2,
                                                              h + 0,
                                                              h + 1,
                                                              2,
                                                              out + 0,
                                                              out + 1)
                                                    : generated_neohookean_ogden_tri3_tri3_apply_isoparametric_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              points,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              2,
                                                              x + 0,
                                                              x + 1,
                                                              2,
                                                              h + 0,
                                                              h + 1,
                                                              2,
                                                              out + 0,
                                                              out + 1);
                case smesh::TRI6:
                    return impl_->apply_uses_affine ? generated_neohookean_ogden_tri6_tri6_apply_affine_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              adjugate[0],
                                                              adjugate[1],
                                                              adjugate[2],
                                                              adjugate[3],
                                                              determinant,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              2,
                                                              x + 0,
                                                              x + 1,
                                                              2,
                                                              h + 0,
                                                              h + 1,
                                                              2,
                                                              out + 0,
                                                              out + 1)
                                                    : generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              points,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              2,
                                                              x + 0,
                                                              x + 1,
                                                              2,
                                                              h + 0,
                                                              h + 1,
                                                              2,
                                                              out + 0,
                                                              out + 1);
                case smesh::QUAD4:
                    return impl_->apply_uses_affine ? generated_neohookean_ogden_quad4_quad4_apply_affine_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              adjugate[0],
                                                              adjugate[1],
                                                              adjugate[2],
                                                              adjugate[3],
                                                              determinant,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              2,
                                                              x + 0,
                                                              x + 1,
                                                              2,
                                                              h + 0,
                                                              h + 1,
                                                              2,
                                                              out + 0,
                                                              out + 1)
                                                    : generated_neohookean_ogden_quad4_quad4_apply_isoparametric_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              points,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              2,
                                                              x + 0,
                                                              x + 1,
                                                              2,
                                                              h + 0,
                                                              h + 1,
                                                              2,
                                                              out + 0,
                                                              out + 1);
                case smesh::TET4:
                    return impl_->apply_uses_affine ? generated_neohookean_ogden_tet4_tet4_apply_affine_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              adjugate[0],
                                                              adjugate[1],
                                                              adjugate[2],
                                                              adjugate[3],
                                                              adjugate[4],
                                                              adjugate[5],
                                                              adjugate[6],
                                                              adjugate[7],
                                                              adjugate[8],
                                                              determinant,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2)
                                                    : generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              points,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2);
                case smesh::TET10:
                    return impl_->apply_uses_affine ? generated_neohookean_ogden_tet10_tet10_apply_affine_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              adjugate[0],
                                                              adjugate[1],
                                                              adjugate[2],
                                                              adjugate[3],
                                                              adjugate[4],
                                                              adjugate[5],
                                                              adjugate[6],
                                                              adjugate[7],
                                                              adjugate[8],
                                                              determinant,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2)
                                                    : generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              points,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2);
                case smesh::HEX8:
                    return impl_->apply_uses_affine ? generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              adjugate[0],
                                                              adjugate[1],
                                                              adjugate[2],
                                                              adjugate[3],
                                                              adjugate[4],
                                                              adjugate[5],
                                                              adjugate[6],
                                                              adjugate[7],
                                                              adjugate[8],
                                                              determinant,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2)
                                                    : generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              points,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2);
                case smesh::HEX27:
                    return impl_->apply_uses_affine ? generated_neohookean_ogden_hex27_hex27_apply_affine_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              adjugate[0],
                                                              adjugate[1],
                                                              adjugate[2],
                                                              adjugate[3],
                                                              adjugate[4],
                                                              adjugate[5],
                                                              adjugate[6],
                                                              adjugate[7],
                                                              adjugate[8],
                                                              determinant,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2)
                                                    : generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa(
                                                              domain.block->n_elements(),
                                                              mesh->n_nodes(),
                                                              domain.block->elements()->data(),
                                                              points,
                                                              domain.parameters->require_real_value("mu"),
                                                              domain.parameters->require_real_value("lmbda"),
                                                              3,
                                                              x + 0,
                                                              x + 1,
                                                              x + 2,
                                                              3,
                                                              h + 0,
                                                              h + 1,
                                                              h + 2,
                                                              3,
                                                              out + 0,
                                                              out + 1,
                                                              out + 2);
                default:
                    SFEM_ERROR("GeneratedNeoHookeanOgden does not support element type %d\n", domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedNeoHookeanOgden::value(const real_t *x, real_t *const out) {
        auto mesh   = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out        = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t      nelements   = domain.block->n_elements();
            const real_t *const *adjugate    = nullptr;
            const real_t        *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("GeneratedNeoHookeanOgden affine objective requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate    = reinterpret_cast<const real_t *const *>(jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const real_t *>(jacobian->jacobian_determinant()->data());
            }
            std::fill(impl_->element_values.get(), impl_->element_values.get() + nelements, 0);
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
                case smesh::TRI3:
                    status = impl_->objective_uses_affine ? generated_neohookean_ogden_tri3_tri3_objective_affine_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    adjugate[0],
                                                                    adjugate[1],
                                                                    adjugate[2],
                                                                    adjugate[3],
                                                                    determinant,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    2,
                                                                    x + 0,
                                                                    x + 1,
                                                                    impl_->element_values.get())
                                                          : generated_neohookean_ogden_tri3_tri3_objective_isoparametric_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    points,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    2,
                                                                    x + 0,
                                                                    x + 1,
                                                                    impl_->element_values.get());
                    break;
                case smesh::TRI6:
                    status = impl_->objective_uses_affine ? generated_neohookean_ogden_tri6_tri6_objective_affine_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    adjugate[0],
                                                                    adjugate[1],
                                                                    adjugate[2],
                                                                    adjugate[3],
                                                                    determinant,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    2,
                                                                    x + 0,
                                                                    x + 1,
                                                                    impl_->element_values.get())
                                                          : generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    points,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    2,
                                                                    x + 0,
                                                                    x + 1,
                                                                    impl_->element_values.get());
                    break;
                case smesh::QUAD4:
                    status = impl_->objective_uses_affine
                                     ? generated_neohookean_ogden_quad4_quad4_objective_affine_mesh_soa(
                                               nelements,
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               adjugate[0],
                                               adjugate[1],
                                               adjugate[2],
                                               adjugate[3],
                                               determinant,
                                               domain.parameters->require_real_value("mu"),
                                               domain.parameters->require_real_value("lmbda"),
                                               2,
                                               x + 0,
                                               x + 1,
                                               impl_->element_values.get())
                                     : generated_neohookean_ogden_quad4_quad4_objective_isoparametric_mesh_soa(
                                               nelements,
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               points,
                                               domain.parameters->require_real_value("mu"),
                                               domain.parameters->require_real_value("lmbda"),
                                               2,
                                               x + 0,
                                               x + 1,
                                               impl_->element_values.get());
                    break;
                case smesh::TET4:
                    status = impl_->objective_uses_affine ? generated_neohookean_ogden_tet4_tet4_objective_affine_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    adjugate[0],
                                                                    adjugate[1],
                                                                    adjugate[2],
                                                                    adjugate[3],
                                                                    adjugate[4],
                                                                    adjugate[5],
                                                                    adjugate[6],
                                                                    adjugate[7],
                                                                    adjugate[8],
                                                                    determinant,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    3,
                                                                    x + 0,
                                                                    x + 1,
                                                                    x + 2,
                                                                    impl_->element_values.get())
                                                          : generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    points,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    3,
                                                                    x + 0,
                                                                    x + 1,
                                                                    x + 2,
                                                                    impl_->element_values.get());
                    break;
                case smesh::TET10:
                    status = impl_->objective_uses_affine
                                     ? generated_neohookean_ogden_tet10_tet10_objective_affine_mesh_soa(
                                               nelements,
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               adjugate[0],
                                               adjugate[1],
                                               adjugate[2],
                                               adjugate[3],
                                               adjugate[4],
                                               adjugate[5],
                                               adjugate[6],
                                               adjugate[7],
                                               adjugate[8],
                                               determinant,
                                               domain.parameters->require_real_value("mu"),
                                               domain.parameters->require_real_value("lmbda"),
                                               3,
                                               x + 0,
                                               x + 1,
                                               x + 2,
                                               impl_->element_values.get())
                                     : generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa(
                                               nelements,
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               points,
                                               domain.parameters->require_real_value("mu"),
                                               domain.parameters->require_real_value("lmbda"),
                                               3,
                                               x + 0,
                                               x + 1,
                                               x + 2,
                                               impl_->element_values.get());
                    break;
                case smesh::HEX8:
                    status = impl_->objective_uses_affine ? generated_neohookean_ogden_hex8_hex8_objective_affine_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    adjugate[0],
                                                                    adjugate[1],
                                                                    adjugate[2],
                                                                    adjugate[3],
                                                                    adjugate[4],
                                                                    adjugate[5],
                                                                    adjugate[6],
                                                                    adjugate[7],
                                                                    adjugate[8],
                                                                    determinant,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    3,
                                                                    x + 0,
                                                                    x + 1,
                                                                    x + 2,
                                                                    impl_->element_values.get())
                                                          : generated_neohookean_ogden_hex8_hex8_objective_isoparametric_mesh_soa(
                                                                    nelements,
                                                                    mesh->n_nodes(),
                                                                    domain.block->elements()->data(),
                                                                    points,
                                                                    domain.parameters->require_real_value("mu"),
                                                                    domain.parameters->require_real_value("lmbda"),
                                                                    3,
                                                                    x + 0,
                                                                    x + 1,
                                                                    x + 2,
                                                                    impl_->element_values.get());
                    break;
                case smesh::HEX27:
                    status = impl_->objective_uses_affine
                                     ? generated_neohookean_ogden_hex27_hex27_objective_affine_mesh_soa(
                                               nelements,
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               adjugate[0],
                                               adjugate[1],
                                               adjugate[2],
                                               adjugate[3],
                                               adjugate[4],
                                               adjugate[5],
                                               adjugate[6],
                                               adjugate[7],
                                               adjugate[8],
                                               determinant,
                                               domain.parameters->require_real_value("mu"),
                                               domain.parameters->require_real_value("lmbda"),
                                               3,
                                               x + 0,
                                               x + 1,
                                               x + 2,
                                               impl_->element_values.get())
                                     : generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa(
                                               nelements,
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               points,
                                               domain.parameters->require_real_value("mu"),
                                               domain.parameters->require_real_value("lmbda"),
                                               3,
                                               x + 0,
                                               x + 1,
                                               x + 2,
                                               impl_->element_values.get());
                    break;
                default:
                    SFEM_ERROR("GeneratedNeoHookeanOgden does not support element type %d\n", domain.element_type);
                    return SFEM_FAILURE;
            }
            if (status != SFEM_SUCCESS) return status;
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t element = 0; element < nelements; ++element) {
                sum += impl_->element_values[element];
            }
            *out += sum;
            return SFEM_SUCCESS;
        });
    }

    int GeneratedNeoHookeanOgden::hessian_crs(const real_t *const, const count_t *const, const idx_t *const, real_t *const) {
        return SFEM_FAILURE;
    }

    void GeneratedNeoHookeanOgden::set_option(const std::string &name, const bool val) {
        if (name == "ASSUME_AFFINE") {
            impl_->objective_uses_affine = val;
            impl_->gradient_uses_affine  = val;
            impl_->apply_uses_affine     = val;
        } else if (name == "ASSUME_AFFINE_OBJECTIVE" || name == "OBJECTIVE_ASSUME_AFFINE") {
            impl_->objective_uses_affine = val;
        } else if (name == "ASSUME_AFFINE_GRADIENT" || name == "GRADIENT_ASSUME_AFFINE") {
            impl_->gradient_uses_affine = val;
        } else if (name == "ASSUME_AFFINE_HESSIAN_ACTION" || name == "HESSIAN_ACTION_ASSUME_AFFINE" ||
                   name == "ASSUME_AFFINE_APPLY" || name == "APPLY_ASSUME_AFFINE") {
            impl_->apply_uses_affine = val;
        }
    }

    void GeneratedNeoHookeanOgden::set_value_in_block(const std::string &block_name,
                                                      const std::string &var_name,
                                                      const real_t       value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }
}  // namespace sfem
