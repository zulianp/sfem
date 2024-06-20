
#include "tet4_fff.h"
#include "tet4_inline_cpu.h"

int tet4_fff_fill(const ptrdiff_t nelements,
                  idx_t **const SFEM_RESTRICT elements,
                  geom_t **const SFEM_RESTRICT points,
                  jacobian_t *const SFEM_RESTRICT fff) {  // Create FFF and store it on device

#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; e++) {
        tet4_fff(points[0][elements[0][e]],
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
                 &fff[e * 6]);
    }

    return 0;
}

int tet4_fff_create(fff_t *ctx,
                    const ptrdiff_t nelements,
                    idx_t **const SFEM_RESTRICT elements,
                    jacobian_t **const SFEM_RESTRICT points) {
    jacobian_t *fff = (jacobian_t *)calloc(6 * nelements, sizeof(jacobian_t));

    tet4_fff_fill(nelements, elements, points, fff);

    ctx->data = fff;
    ctx->elements = elements;
    ctx->nelements = nelements;
    ctx->element_type = TET4;

    return 0;
}

int tet4_fff_destroy(fff_t *ctx) {
    free(ctx->data);

    ctx->nelements = 0;
    ctx->elements = 0;
    ctx->element_type = INVALID;

    return 0;
}
