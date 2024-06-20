#include "tet4_adjugate.h"

#include "tet4_inline_cpu.h"

void tet4_adjugate_fill(const ptrdiff_t nelements,
                        idx_t **const SFEM_RESTRICT elements,
                        geom_t **const SFEM_RESTRICT points,
                        jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                        jacobian_t *const SFEM_RESTRICT jacobian_determinant) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; e++) {
        tet4_adjugate_and_det(points[0][elements[0][e]],
                              points[0][elements[1][e]],
                              points[0][elements[2][e]],
                              points[0][elements[3][e]],
                              points[1][elements[0][e]],
                              points[1][elements[1][e]],
                              points[1][elements[2][e]],
                              points[1][elements[3][e]],
                              points[2][elements[0][e]],
                              points[2][elements[1][e]],
                              points[2][elements[2][e]],
                              points[2][elements[3][e]],
                              &jacobian_adjugate[e * 9],
                              &jacobian_determinant[e]);
    }
}

void tet4_adjugate_create(tet4_adjugate_t *ctx,
                          const ptrdiff_t nelements,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points) {
    jacobian_t *jacobian_adjugate = (jacobian_t *)calloc(9 * nelements, sizeof(jacobian_t));
    jacobian_t *jacobian_determinant = (jacobian_t *)calloc(nelements, sizeof(jacobian_t));

    tet4_adjugate_fill(nelements, elements, points, jacobian_adjugate, jacobian_determinant);

    ctx->nelements = nelements;
    ctx->jacobian_adjugate = jacobian_adjugate;
    ctx->jacobian_determinant = jacobian_determinant;
    ctx->elements = elements;
    ctx->element_type = TET4;
}

void tet4_adjugate_destroy(tet4_adjugate_t *ctx) {
    free(ctx->jacobian_adjugate);
    free(ctx->jacobian_determinant);

    ctx->nelements = 0;
    ctx->elements = 0;
    ctx->element_type = INVALID;
}
