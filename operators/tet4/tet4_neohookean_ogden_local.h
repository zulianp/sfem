#ifndef TET4_NEOHOOKEAN_OGDEN_LOCAL_H
#define TET4_NEOHOOKEAN_OGDEN_LOCAL_H

#include <math.h>
#include "sfem_base.h"

#include "tet4_partial_assembly_neohookean_inline.h"

static SFEM_INLINE void tet4_neohookean_ogden_objective_elemental(const scalar_t *const SFEM_RESTRICT adjugate,
                                                  const scalar_t                      jacobian_determinant,
                                                  const scalar_t                      lmbda,
                                                  const scalar_t                      mu,
                                                  const scalar_t *const SFEM_RESTRICT dispx,
                                                  const scalar_t *const SFEM_RESTRICT dispy,
                                                  const scalar_t *const SFEM_RESTRICT dispz,
                                                  scalar_t *const SFEM_RESTRICT       v) {
    scalar_t F[9];
    tet4_F(adjugate, jacobian_determinant, dispx, dispy, dispz, F);

    // mundane ops: 50 divs: 0 sqrts: 0
    // total ops: 50
    const scalar_t x0 = log(F[0] * F[4] * F[8] - F[0] * F[5] * F[7] - F[1] * F[3] * F[8] + F[1] * F[5] * F[6] +
                            F[2] * F[3] * F[7] - F[2] * F[4] * F[6]);
    v[0] += (1.0 / 6.0) * jacobian_determinant *
            ((1.0 / 2.0) * lmbda * POW2(x0) - mu * x0 +
             (1.0 / 2.0) * mu *
                     (POW2(F[0]) + POW2(F[1]) + POW2(F[2]) + POW2(F[3]) + POW2(F[4]) + POW2(F[5]) + POW2(F[6]) + POW2(F[7]) +
                      POW2(F[8]) - 3));
}

static SFEM_INLINE void tet4_neohookean_ogden_objective_integral(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                 const scalar_t                      jacobian_determinant,
                                                                 const scalar_t                      mu,
                                                                 const scalar_t                      lmbda,
                                                                 const scalar_t *const SFEM_RESTRICT dispx,
                                                                 const scalar_t *const SFEM_RESTRICT dispy,
                                                                 const scalar_t *const SFEM_RESTRICT dispz,
                                                                 scalar_t *const SFEM_RESTRICT       v) {
    tet4_neohookean_ogden_objective_elemental(adjugate, jacobian_determinant, mu, lmbda, dispx, dispy, dispz, v);
}

static SFEM_INLINE void tet4_neohookean_ogden_objective_steps_integral(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                       const scalar_t                      jacobian_determinant,
                                                                       const scalar_t                      mu,
                                                                       const scalar_t                      lmbda,
                                                                       const scalar_t *const SFEM_RESTRICT dispx,
                                                                       const scalar_t *const SFEM_RESTRICT dispy,
                                                                       const scalar_t *const SFEM_RESTRICT dispz,
                                                                       const scalar_t *const SFEM_RESTRICT incx,
                                                                       const scalar_t *const SFEM_RESTRICT incy,
                                                                       const scalar_t *const SFEM_RESTRICT incz,
                                                                       const int                           nsteps,
                                                                       const scalar_t *const SFEM_RESTRICT steps,
                                                                       scalar_t *const SFEM_RESTRICT       v) {
    scalar_t ux[4];
    scalar_t uy[4];
    scalar_t uz[4];

    for (int i = 0; i < nsteps; i++) {
        for (int j = 0; j < 4; j++) {
            ux[j] = dispx[j] + incx[j] * steps[i];
            uy[j] = dispy[j] + incy[j] * steps[i];
            uz[j] = dispz[j] + incz[j] * steps[i];
        }

        tet4_neohookean_ogden_objective_integral(adjugate, jacobian_determinant, mu, lmbda, ux, uy, uz, &v[i]);
    }
}

#endif  // TET4_NEOHOOKEAN_OGDEN_LOCAL_H