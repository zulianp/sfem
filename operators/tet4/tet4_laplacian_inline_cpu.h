
static SFEM_INLINE void tet4_laplacian_hessian_fff(const geom_t *const SFEM_RESTRICT fff,
                                                   real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = -fff[0] - fff[1] - fff[2];
    const real_t x1 = -fff[1] - fff[3] - fff[4];
    const real_t x2 = -fff[2] - fff[4] - fff[5];
    element_matrix[0] = fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    element_matrix[1] = x0;
    element_matrix[2] = x1;
    element_matrix[3] = x2;
    element_matrix[4] = x0;
    element_matrix[5] = fff[0];
    element_matrix[6] = fff[1];
    element_matrix[7] = fff[2];
    element_matrix[8] = x1;
    element_matrix[9] = fff[1];
    element_matrix[10] = fff[3];
    element_matrix[11] = fff[4];
    element_matrix[12] = x2;
    element_matrix[13] = fff[2];
    element_matrix[14] = fff[4];
    element_matrix[15] = fff[5];
}

static SFEM_INLINE void tet4_laplacian_diag_fff(const geom_t *const SFEM_RESTRICT fff,
                                                real_t *const SFEM_RESTRICT e0,
                                                real_t *const SFEM_RESTRICT e1,
                                                real_t *const SFEM_RESTRICT e2,
                                                real_t *const SFEM_RESTRICT e3) {
    *e0 += fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    *e1 += fff[0];
    *e2 += fff[3];
    *e3 += fff[5];
}

static SFEM_INLINE void tet4_laplacian_apply_fff(const geom_t *const SFEM_RESTRICT fff,
                                                 const real_t u0,
                                                 const real_t u1,
                                                 const real_t u2,
                                                 const real_t u3,
                                                 real_t *const SFEM_RESTRICT e0,
                                                 real_t *const SFEM_RESTRICT e1,
                                                 real_t *const SFEM_RESTRICT e2,
                                                 real_t *const SFEM_RESTRICT e3) {
    const real_t x0 = fff[0] + fff[1] + fff[2];
    const real_t x1 = fff[1] + fff[3] + fff[4];
    const real_t x2 = fff[2] + fff[4] + fff[5];
    const real_t x3 = fff[1] * u0;
    const real_t x4 = fff[2] * u0;
    const real_t x5 = fff[4] * u0;
    *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}
