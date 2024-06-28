#include "macro_tet4_linear_elasticity.h"

#include "sfem_base.h"

#include "macro_tet4_inline_cpu.h"
#include "tet4_linear_elasticity_inline_cpu.h"

#include <stddef.h>

static const int sub_tets[8][4] = {{0, 4, 6, 7},
                                   {4, 1, 5, 8},
                                   {6, 5, 2, 9},
                                   {7, 8, 9, 3},
                                   {4, 5, 6, 8},
                                   {7, 4, 6, 8},
                                   {6, 5, 9, 8},
                                   {7, 6, 9, 8}};

typedef void (*SubAdjFun)(const scalar_t *const SFEM_RESTRICT, scalar_t *const SFEM_RESTRICT);

static SubAdjFun octahedron_adj_fun[4] = {&tet4_sub_adj_4,
                                          &tet4_sub_adj_5,
                                          &tet4_sub_adj_6,
                                          &tet4_sub_adj_7};

static SFEM_INLINE void subtet_gather(const int i,
                                      const scalar_t *const SFEM_RESTRICT in,
                                      scalar_t *const SFEM_RESTRICT out) {
    const int *g = sub_tets[i];
    for (int v = 0; v < 4; ++v) {
        out[v] = in[g[v]];
    }
}

static SFEM_INLINE void subtet_scatter_add(const int i,
                                           const accumulator_t *const SFEM_RESTRICT in,
                                           accumulator_t *const SFEM_RESTRICT out) {
    const int *s = sub_tets[i];
    for (int v = 0; v < 4; ++v) {
        out[s[v]] += in[v];
    }
}

static SFEM_INLINE void macro_tet4_local_apply_adj(const scalar_t *const SFEM_RESTRICT
                                                           jacobian_adjugate,
                                                   const scalar_t jacobian_determinant,
                                                   const scalar_t mu,
                                                   const scalar_t lambda,
                                                   const scalar_t *const SFEM_RESTRICT ux,
                                                   const scalar_t *const SFEM_RESTRICT uy,
                                                   const scalar_t *const SFEM_RESTRICT uz,
                                                   accumulator_t *const SFEM_RESTRICT outx,
                                                   accumulator_t *const SFEM_RESTRICT outy,
                                                   accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t sub_adjugate[9];

    scalar_t sub_ux[4];
    scalar_t sub_uy[4];
    scalar_t sub_uz[4];

    accumulator_t sub_outx[4];
    accumulator_t sub_outy[4];
    accumulator_t sub_outz[4];


    scalar_t sub_determinant = jacobian_determinant*8.0;

    {  // Corner tests
        tet4_sub_adj_0(jacobian_adjugate, sub_adjugate);

        for (int i = 0; i < 4; i++) {
            subtet_gather(i, ux, sub_ux);
            subtet_gather(i, uy, sub_uy);
            subtet_gather(i, uz, sub_uz);

            tet4_linear_elasticity_apply_adj(sub_adjugate,
                                             sub_determinant,
                                             mu,
                                             lambda,
                                             sub_ux,
                                             sub_uy,
                                             sub_uz,
                                             sub_outx,
                                             sub_outy,
                                             sub_outz);

            subtet_scatter_add(i, sub_outx, outx);
            subtet_scatter_add(i, sub_outy, outy);
            subtet_scatter_add(i, sub_outz, outz);
        }
    }

    {  // Octahedron tets
        for (int i = 0; i < 4; i++) {
            SubAdjFun sub_adj_fun = octahedron_adj_fun[i];

            (*sub_adj_fun)(jacobian_adjugate, sub_adjugate);

            subtet_gather(4 + i, ux, sub_ux);
            subtet_gather(4 + i, uy, sub_uy);
            subtet_gather(4 + i, uz, sub_uz);

            tet4_linear_elasticity_apply_adj(sub_adjugate,
                                             sub_determinant,
                                             mu,
                                             lambda,
                                             sub_ux,
                                             sub_uy,
                                             sub_uz,
                                             sub_outx,
                                             sub_outy,
                                             sub_outz);

            subtet_scatter_add(4 + i, sub_outx, outx);
            subtet_scatter_add(4 + i, sub_outy, outy);
            subtet_scatter_add(4 + i, sub_outz, outz);
        }
    }
}

int macro_tet4_linear_elasticity_apply(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const real_t mu,
                                       const real_t lambda,
                                       const ptrdiff_t u_stride,
                                       const real_t *const SFEM_RESTRICT g_ux,
                                       const real_t *const SFEM_RESTRICT g_uy,
                                       const real_t *const SFEM_RESTRICT g_uz,
                                       const ptrdiff_t out_stride,
                                       real_t *const SFEM_RESTRICT g_outx,
                                       real_t *const SFEM_RESTRICT g_outy,
                                       real_t *const SFEM_RESTRICT g_outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        scalar_t ux[10];
        scalar_t uy[10];
        scalar_t uz[10];
        accumulator_t outx[10] = {0};
        accumulator_t outy[10] = {0};
        accumulator_t outz[10] = {0};
        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            ux[v] = g_ux[ev[v] * u_stride];
            uy[v] = g_uy[ev[v] * u_stride];
            uz[v] = g_uz[ev[v] * u_stride];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
                                x[ev[1]],
                                x[ev[2]],
                                x[ev[3]],
                                // Y-coordinates
                                y[ev[0]],
                                y[ev[1]],
                                y[ev[2]],
                                y[ev[3]],
                                // Z-coordinates
                                z[ev[0]],
                                z[ev[1]],
                                z[ev[2]],
                                z[ev[3]],
                                // Output
                                jacobian_adjugate,
                                &jacobian_determinant);

        macro_tet4_local_apply_adj(
                jacobian_adjugate, jacobian_determinant, mu, lambda, ux, uy, uz, outx, outy, outz);

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outx[ev[v] * out_stride] += outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outy[ev[v] * out_stride] += outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outz[ev[v] * out_stride] += outz[v];
        }
    }
    return 0;
}

int macro_tet4_linear_elasticity_apply_opt(
        const ptrdiff_t nelements,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
        const real_t mu,
        const real_t lambda,
        const ptrdiff_t u_stride,
        const real_t *const SFEM_RESTRICT g_ux,
        const real_t *const SFEM_RESTRICT g_uy,
        const real_t *const SFEM_RESTRICT g_uz,
        const ptrdiff_t out_stride,
        real_t *const SFEM_RESTRICT g_outx,
        real_t *const SFEM_RESTRICT g_outy,
        real_t *const SFEM_RESTRICT g_outz) {
    {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            scalar_t ux[10];
            scalar_t uy[10];
            scalar_t uz[10];
            accumulator_t outx[10] = {0};
            accumulator_t outy[10] = {0};
            accumulator_t outz[10] = {0};

            const scalar_t jacobian_determinant = g_jacobian_determinant[i];
            scalar_t jacobian_adjugate[9];
            for (int k = 0; k < 9; k++) {
                jacobian_adjugate[k] = g_jacobian_adjugate[i * 9 + k];
            }

            // #pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 10; ++v) {
                const ptrdiff_t idx = ev[v] * u_stride;
                ux[v] = g_ux[idx];
                uy[v] = g_uy[idx];
                uz[v] = g_uz[idx];
            }

            macro_tet4_local_apply_adj(jacobian_adjugate,
                                       jacobian_determinant,
                                       mu,
                                       lambda,
                                       ux,
                                       uy,
                                       uz,
                                       outx,
                                       outy,
                                       outz);

            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                g_outx[ev[v] * out_stride] += outx[v];
            }

            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                g_outy[ev[v] * out_stride] += outy[v];
            }

            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                g_outz[ev[v] * out_stride] += outz[v];
            }
        }
    }

    return 0;
}

static SFEM_INLINE void macro_tet4_local_diag_adj(const scalar_t *const SFEM_RESTRICT
                                                          jacobian_adjugate,
                                                  const scalar_t jacobian_determinant,
                                                  const scalar_t mu,
                                                  const scalar_t lambda,
                                                  accumulator_t *const SFEM_RESTRICT outx,
                                                  accumulator_t *const SFEM_RESTRICT outy,
                                                  accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t sub_adjugate[9];
    accumulator_t sub_outx[4];
    accumulator_t sub_outy[4];
    accumulator_t sub_outz[4];

    scalar_t sub_determinant = jacobian_determinant * 8;

    {  // Corner tests
        tet4_sub_adj_0(jacobian_adjugate, sub_adjugate);

        for (int i = 0; i < 4; i++) {
            tet4_linear_elasticity_diag_adj(
                    mu, lambda, sub_adjugate, sub_determinant, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(i, sub_outx, outx);
            subtet_scatter_add(i, sub_outy, outy);
            subtet_scatter_add(i, sub_outz, outz);
        }
    }

    {  // Octahedron tets
        for (int i = 0; i < 4; i++) {
            SubAdjFun sub_adj_fun = octahedron_adj_fun[i];

            (*sub_adj_fun)(jacobian_adjugate, sub_adjugate);

            tet4_linear_elasticity_diag_adj(
                    mu, lambda, sub_adjugate, sub_determinant, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(4 + i, sub_outx, outx);
            subtet_scatter_add(4 + i, sub_outy, outy);
            subtet_scatter_add(4 + i, sub_outz, outz);
        }
    }
}

int macro_tet4_linear_elasticity_diag(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elements,
                                      geom_t **const SFEM_RESTRICT points,
                                      const real_t mu,
                                      const real_t lambda,
                                      const ptrdiff_t out_stride,
                                      real_t *const SFEM_RESTRICT g_outx,
                                      real_t *const SFEM_RESTRICT g_outy,
                                      real_t *const SFEM_RESTRICT g_outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        accumulator_t outx[10] = {0};
        accumulator_t outy[10] = {0};
        accumulator_t outz[10] = {0};
        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
                                x[ev[1]],
                                x[ev[2]],
                                x[ev[3]],
                                // Y-coordinates
                                y[ev[0]],
                                y[ev[1]],
                                y[ev[2]],
                                y[ev[3]],
                                // Z-coordinates
                                z[ev[0]],
                                z[ev[1]],
                                z[ev[2]],
                                z[ev[3]],
                                // Output
                                jacobian_adjugate,
                                &jacobian_determinant);

        macro_tet4_local_diag_adj(
                jacobian_adjugate, jacobian_determinant, mu, lambda, outx, outy, outz);

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outx[ev[v] * out_stride] += outx[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outy[ev[v] * out_stride] += outy[v];
        }

#pragma unroll(10)
        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outz[ev[v] * out_stride] += outz[v];
        }
    }
    return 0;
}

int macro_tet4_linear_elasticity_diag_opt(const ptrdiff_t nelements,
                                          idx_t **const SFEM_RESTRICT elements,
                                          const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                          const jacobian_t *const SFEM_RESTRICT
                                                  g_jacobian_determinant,
                                          const real_t mu,
                                          const real_t lambda,
                                          const ptrdiff_t out_stride,
                                          real_t *const SFEM_RESTRICT g_outx,
                                          real_t *const SFEM_RESTRICT g_outy,
                                          real_t *const SFEM_RESTRICT g_outz) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        accumulator_t outx[10] = {0};
        accumulator_t outy[10] = {0};
        accumulator_t outz[10] = {0};

        const scalar_t jacobian_determinant = g_jacobian_determinant[i];
        scalar_t jacobian_adjugate[9];
        for (int k = 0; k < 9; k++) {
            jacobian_adjugate[k] = g_jacobian_adjugate[i * 9 + k];
        }

        // #pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        macro_tet4_local_diag_adj(
                jacobian_adjugate, jacobian_determinant, mu, lambda, outx, outy, outz);

        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outx[ev[v] * out_stride] += outx[v];
        }

        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outy[ev[v] * out_stride] += outy[v];
        }

        for (int v = 0; v < 10; v++) {
#pragma omp atomic update
            g_outz[ev[v] * out_stride] += outz[v];
        }
    }

    return 0;
}

static SFEM_INLINE void macro_tet4_local_hessian(const idx_t *const SFEM_RESTRICT ev10,
                                                 const scalar_t *const SFEM_RESTRICT
                                                         jacobian_adjugate,
                                                 const scalar_t jacobian_determinant,
                                                 const scalar_t mu,
                                                 const scalar_t lambda,
                                                 const count_t *const SFEM_RESTRICT rowptr,
                                                 const idx_t *const SFEM_RESTRICT colidx,
                                                 real_t *const SFEM_RESTRICT values) {
    scalar_t sub_adjugate[9];
    idx_t ev[4];
    accumulator_t element_matrix[(4 * 3) * (4 * 3)];

    scalar_t sub_determinant = jacobian_determinant*8.0 ;

    {  // Corner tests
        tet4_sub_adj_0(jacobian_adjugate, sub_adjugate);

        // Assemble once and reuse for all corners
        tet4_linear_elasticity_hessian_adj(mu, lambda, sub_adjugate, sub_determinant, element_matrix);

        // [0, 4, 6, 7]
        tet4_gather_idx(ev10, 0, 4, 6, 7, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
        
        // [4, 1, 5, 8] 
        tet4_gather_idx(ev10, 4, 1, 5, 8, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
        
        // [6, 5, 2, 9]
        tet4_gather_idx(ev10, 6, 5, 2, 9, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);

        // [7, 8, 9, 3]
        tet4_gather_idx(ev10, 7, 8, 9, 3, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
    }

    {  // Octahedron tets
        // [4, 5, 6, 8]
        tet4_sub_adj_4(jacobian_adjugate, sub_adjugate);
        tet4_linear_elasticity_hessian_adj(mu, lambda, sub_adjugate, sub_determinant, element_matrix);
        tet4_gather_idx(ev10, 4, 5, 6, 8, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
                
        // [7, 4, 6, 8]
        tet4_sub_adj_5(jacobian_adjugate, sub_adjugate);
        tet4_linear_elasticity_hessian_adj(mu, lambda, sub_adjugate, sub_determinant, element_matrix);
        tet4_gather_idx(ev10, 7, 4, 6, 8, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
            
        // [6, 5, 9, 8]
        tet4_sub_adj_6(jacobian_adjugate, sub_adjugate);
        tet4_linear_elasticity_hessian_adj(mu, lambda, sub_adjugate, sub_determinant, element_matrix);
        tet4_gather_idx(ev10, 6, 5, 9, 8, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);

        // [7, 6, 9, 8]
        tet4_sub_adj_7(jacobian_adjugate, sub_adjugate);
        tet4_linear_elasticity_hessian_adj(mu, lambda, sub_adjugate, sub_determinant, element_matrix);
        tet4_gather_idx(ev10, 7, 6, 9, 8, ev);
        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
    
    }
}

int macro_tet4_linear_elasticity_hessian(const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t mu,
                                         const real_t lambda,
                                         const count_t *const SFEM_RESTRICT rowptr,
                                         const idx_t *const SFEM_RESTRICT colidx,
                                         real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;
        tet4_adjugate_and_det_s(x[ev[0]],
                                x[ev[1]],
                                x[ev[2]],
                                x[ev[3]],
                                // Y-coordinates
                                y[ev[0]],
                                y[ev[1]],
                                y[ev[2]],
                                y[ev[3]],
                                // Z-coordinates
                                z[ev[0]],
                                z[ev[1]],
                                z[ev[2]],
                                z[ev[3]],
                                // Output
                                jacobian_adjugate,
                                &jacobian_determinant);

        macro_tet4_local_hessian(
                ev, jacobian_adjugate, jacobian_determinant, mu, lambda, rowptr, colidx, values);
    }

    return 0;
}
