// #include "sstet4_sampling.h"

// #include "sstet4_laplacian.h"

// #include <assert.h>

// int sstet4_up_sampling(const int                         level,
//                        const ptrdiff_t                   nelements,
//                        idx_t **const                     tet4_elements,
//                        geom_t **const SFEM_RESTRICT      tet4_points,
//                        const real_t *const SFEM_RESTRICT tet4_u,
//                        real_t *const SFEM_RESTRICT       sstet4_u) {
//     const int nxe = sstet4_nxe(level);
//     const int txe = sstet4_txe(level);

//     const geom_t *const x = tet4_points[0];
//     const geom_t *const y = tet4_points[1];
//     const geom_t *const z = tet4_points[2];
//     const scalar_t      h = 1. / level;

// #define NQP 1
//     static scalar_t qw[NQP] = {(scalar_t)1};
//     static scalar_t qx[NQP] = {(scalar_t)1 / 3};
//     static scalar_t qy[NQP] = {(scalar_t)1 / 3};
//     static scalar_t qz[NQP] = {(scalar_t)1 / 3};

// #pragma omp parallel
//     {
//         scalar_t *tet_qx = malloc(NQP * sizeof(scalar_t));
//         scalar_t *tet_qy = malloc(NQP * sizeof(scalar_t));
//         scalar_t *tet_qz = malloc(NQP * sizeof(scalar_t));

//         scalar_t *grid_qx = malloc(NQP * sizeof(scalar_t));
//         scalar_t *grid_qy = malloc(NQP * sizeof(scalar_t));
//         scalar_t *grid_qz = malloc(NQP * sizeof(scalar_t));

// #pragma omp for
//         for (ptrdiff_t i = 0; i < nelements; ++i) {
//             idx_t ev[4];
// #pragma unroll(4)
//             for (int v = 0; v < 4; ++v) {
//                 ev[v] = tet4_elements[v][i];
//             }

//             // real_t *const element_vector = &sstet4_u[i * nxe];

//             // Element indices
//             const idx_t i0 = ev[0];
//             const idx_t i1 = ev[1];
//             const idx_t i2 = ev[2];
//             const idx_t i3 = ev[3];

//             // Macro Jacobian and translation
//             scalar_t J[3 * 3], J_det;
//             scalar_t b[3];
//             {
//                 b[0] = x[i0];
//                 b[1] = y[i0];
//                 b[2] = z[i0];

//                 const scalar_t x0 = x[i0] - x[i1];
//                 const scalar_t x1 = x[i0] - x[i2];
//                 const scalar_t x2 = x[i0] - x[i3];
//                 const scalar_t x3 = y[i0] - y[i1];
//                 const scalar_t x4 = y[i0] - y[i2];
//                 const scalar_t x5 = y[i0] - y[i3];
//                 const scalar_t x6 = z[i0] - z[i1];
//                 const scalar_t x7 = z[i0] - z[i2];
//                 const scalar_t x8 = z[i0] - z[i3];

//                 J[0]  = -x0;
//                 J[1]  = -x1;
//                 J[2]  = -x2;
//                 J[3]  = -x3;
//                 J[4]  = -x4;
//                 J[5]  = -x5;
//                 J[6]  = -x6;
//                 J[7]  = -x7;
//                 J[8]  = -x8;
//                 J_det = -x0 * x4 * x8 + x0 * x5 * x7 + x1 * x3 * x8 - x1 * x5 * x6 - x2 * x3 * x7 + x2 * x4 * x6;
//             }

//             const scalar_t sub_vol       = (scalar_t)1 / (level * level * level);
//             const scalar_t JxJ_det_micro = J_det / sub_vol;

//             // Micro Jacobian per category
//             scalar_t J_micro[6][3 * 3];
//             {
//                 const scalar_t x0  = J[1] * sub_vol;
//                 const scalar_t x1  = J[2] * sub_vol;
//                 const scalar_t x2  = J[4] * sub_vol;
//                 const scalar_t x3  = J[5] * sub_vol;
//                 const scalar_t x4  = J[7] * sub_vol;
//                 const scalar_t x5  = J[8] * sub_vol;
//                 const scalar_t x6  = sub_vol * (-J[0] + J[1] + J[2]);
//                 const scalar_t x7  = sub_vol * (-J[3] + J[4] + J[5]);
//                 const scalar_t x8  = sub_vol * (-J[6] + J[7] + J[8]);
//                 const scalar_t x9  = -sub_vol * (J[0] - J[1]);
//                 const scalar_t x10 = -J[2];
//                 const scalar_t x11 = -sub_vol * (J[0] + x10);
//                 const scalar_t x12 = -sub_vol * (J[3] - J[4]);
//                 const scalar_t x13 = -J[5];
//                 const scalar_t x14 = -sub_vol * (J[3] + x13);
//                 const scalar_t x15 = -sub_vol * (J[6] - J[7]);
//                 const scalar_t x16 = -J[8];
//                 const scalar_t x17 = -sub_vol * (J[6] + x16);

//                 J_micro[0][0] = J[0] * sub_vol;
//                 J_micro[0][1] = x0;
//                 J_micro[0][2] = x1;
//                 J_micro[0][3] = J[3] * sub_vol;
//                 J_micro[0][4] = x2;
//                 J_micro[0][5] = x3;
//                 J_micro[0][6] = J[6] * sub_vol;
//                 J_micro[0][7] = x4;
//                 J_micro[0][8] = x5;
//                 J_micro[1][0] = -sub_vol * (J[0] + J[2]);
//                 J_micro[1][1] = x1;
//                 J_micro[1][2] = x6;
//                 J_micro[1][3] = -sub_vol * (J[3] + J[5]);
//                 J_micro[1][4] = x3;
//                 J_micro[1][5] = x7;
//                 J_micro[1][6] = -sub_vol * (J[6] + J[8]);
//                 J_micro[1][7] = x5;
//                 J_micro[1][8] = x8;
//                 J_micro[2][0] = x0;
//                 J_micro[2][1] = x6;
//                 J_micro[2][2] = x1;
//                 J_micro[2][3] = x2;
//                 J_micro[2][4] = x7;
//                 J_micro[2][5] = x3;
//                 J_micro[2][6] = x4;
//                 J_micro[2][7] = x8;
//                 J_micro[2][8] = x5;
//                 J_micro[3][0] = x9;
//                 J_micro[3][1] = x11;
//                 J_micro[3][2] = x6;
//                 J_micro[3][3] = x12;
//                 J_micro[3][4] = x14;
//                 J_micro[3][5] = x7;
//                 J_micro[3][6] = x15;
//                 J_micro[3][7] = x17;
//                 J_micro[3][8] = x8;
//                 J_micro[4][0] = -sub_vol * (J[1] + x10);
//                 J_micro[4][1] = x1;
//                 J_micro[4][2] = x11;
//                 J_micro[4][3] = -sub_vol * (J[4] + x13);
//                 J_micro[4][4] = x3;
//                 J_micro[4][5] = x14;
//                 J_micro[4][6] = -sub_vol * (J[7] + x16);
//                 J_micro[4][7] = x5;
//                 J_micro[4][8] = x17;
//                 J_micro[5][0] = x9;
//                 J_micro[5][1] = x6;
//                 J_micro[5][2] = x0;
//                 J_micro[5][3] = x12;
//                 J_micro[5][4] = x7;
//                 J_micro[5][5] = x2;
//                 J_micro[5][6] = x15;
//                 J_micro[5][7] = x8;
//                 J_micro[5][8] = x4;
//             }

//             ///////////////////////////////////
//             // Cat 0
//             ///////////////////////////////////
//             {
//                 // Rotate/Scale/Shear the qp to the macro-tet space (translation is done for micro element)
//                 for (int q = 0; q < NQP; q++) {
//                     const scalar_t *const row0 = &J_micro[0][0 * 3];
//                     const scalar_t *const row1 = &J_micro[0][1 * 3];
//                     const scalar_t *const row2 = &J_micro[0][2 * 3];

//                     grid_qx[d] += row0[0] * qx[q] +  //
//                                   row0[1] * qy[q] +  //
//                                   row0[2] * qz[q];

//                     grid_qy[d] += row1[0] * qx[q] +  //
//                                   row0[1] * qy[q] +  //
//                                   row0[2] * qz[q];

//                     grid_qz[d] += row2[0] * qx[q] +  //
//                                   row0[1] * qy[q] +  //
//                                   row0[2] * qz[q];
//                 }

//                 // Rotate/Scale/Shear the qp to the grid space (translation is done for micro element)
//                 for (int q = 0; q < NQP; q++) {
//                     tet_qx[q] = sub_vol * qx[q];
//                     tet_qy[q] = sub_vol * qy[q];
//                     tet_qz[q] = sub_vol * qz[q];
//                 }

//                 int p = 0;
//                 for (int i = 0; i < level - 1; i++) {
//                     int layer_items = (level - i + 1) * (level - i) / 2;
//                     for (int j = 0; j < level - i - 1; j++) {
//                         for (int k = 0; k < level - i - j - 1; k++) {
//                             ev[0] = p;
//                             ev[1] = p + 1;
//                             ev[2] = p + level - i - j;
//                             ev[3] = p + layer_items - j;

//                             scalar_t b_micro[3]   = {i * h, j * h, k * h};
//                             scalar_t translate[3] = {J[0] * b_micro[0] + J[1] * b_micro[1] + J[2] * b_micro[2] + b[0],
//                                                      J[3] * b_micro[0] + J[4] * b_micro[1] + J[5] * b_micro[2] + b[1],
//                                                      J[6] * b_micro[0] + J[7] * b_micro[1] + J[8] * b_micro[2] + b[2]};

//                             for (int q = 0; q < NQP; q++) {
//                                 // TODO: interpolate values at tet_qp using rq
//                                 scalar_t rqx = tet_qx[q] + b_micro[0];
//                                 scalar_t rqy = tet_qy[q] + b_micro[1];
//                                 scalar_t rqz = tet_qz[q] + b_micro[2];

//                                 // fun(rqx, rqy, rqz, f);

//                                 // TODO: project values to grid
//                                 scalar_t gqx = grid_qx[q] + translate[0];
//                                 scalar_t gqy = grid_qy[q] + translate[1];
//                                 scalar_t gqz = grid_qz[q] + translate[2];

//                                 // ptrdiff_t i = gqx ./
//                                 // scalar_t reg_gqx = gqx - i; 

//                                 // TODO: integral
//                                 // TODO: scatter
//                             }

//                             p++;
//                         }
//                         p++;
//                     }
//                     p++;
//                 }
//             }

//             //     ///////////////////////////////////
//             //     // Cat 1
//             //     ///////////////////////////////////
//             //     {
//             //         sstet4_sub_fff_1(level, &g_fff[e * 6], fff);

//             //         int p = 0;
//             //         for (int i = 0; i < level - 1; i++) {
//             //             int layer_items = (level - i) * (level - i - 1) / 2;
//             //             for (int j = 0; j < level - i - 1; j++) {
//             //                 p++;
//             //                 for (int k = 1; k < level - i - j - 1; k++) {
//             //                     ev[0] = p;
//             //                     ev[1] = p + layer_items + level - i - j - 1;
//             //                     ev[2] = p + layer_items + level - i - j;
//             //                     ev[3] = p + layer_items + level - i - j - 1 + level - i - j - 1;

//             //                     tet4_laplacian_apply_fff(fff,
//             //                                              element_u[ev[0]],
//             //                                              element_u[ev[1]],
//             //                                              element_u[ev[2]],
//             //                                              element_u[ev[3]],
//             //                                              &v[0],
//             //                                              &v[1],
//             //                                              &v[2],
//             //                                              &v[3]);

//             //                     for (int d = 0; d < 4; d++) {
//             //                         element_vector[ev[d]] += v[d];
//             //                     }
//             //                     p++;
//             //                 }
//             //                 p++;
//             //             }
//             //             p++;
//             //         }
//             //     }

//             //     ///////////////////////////////////
//             //     // Cat 2
//             //     ///////////////////////////////////
//             //     {
//             //         sstet4_sub_fff_2(level, &g_fff[e * 6], fff);

//             //         int p = 0;
//             //         for (int i = 0; i < level - 1; i++) {
//             //             int layer_items = (level - i) * (level - i - 1) / 2;
//             //             for (int j = 0; j < level - i - 1; j++) {
//             //                 p++;
//             //                 for (int k = 1; k < level - i - j - 1; k++) {
//             //                     ev[0] = p;
//             //                     ev[1] = p + level - i - j;
//             //                     ev[3] = p + layer_items + level - i - j;
//             //                     ev[2] = p + layer_items + level - i - j - 1 + level - i - j - 1;

//             //                     tet4_laplacian_apply_fff(fff,
//             //                                              element_u[ev[0]],
//             //                                              element_u[ev[1]],
//             //                                              element_u[ev[2]],
//             //                                              element_u[ev[3]],
//             //                                              &v[0],
//             //                                              &v[1],
//             //                                              &v[2],
//             //                                              &v[3]);

//             //                     for (int d = 0; d < 4; d++) {
//             //                         element_vector[ev[d]] += v[d];
//             //                     }

//             //                     p++;
//             //                 }
//             //                 p++;
//             //             }
//             //             p++;
//             //         }
//             //     }

//             //     ///////////////////////////////////
//             //     // Cat 3
//             //     ///////////////////////////////////
//             //     {
//             //         sstet4_sub_fff_3(level, &g_fff[e * 6], fff);

//             //         int p = 0;
//             //         for (int i = 0; i < level - 1; i++) {
//             //             int layer_items = (level - i) * (level - i - 1) / 2;
//             //             for (int j = 0; j < level - i - 1; j++) {
//             //                 p++;
//             //                 for (int k = 1; k < level - i - j - 1; k++) {
//             //                     ev[0] = p;
//             //                     ev[1] = p + level - i - j - 1;
//             //                     ev[2] = p + layer_items + level - i - j - 1;
//             //                     ev[3] = p + layer_items + level - i - j - 1 + level - i - j - 1;

//             //                     tet4_laplacian_apply_fff(fff,
//             //                                              element_u[ev[0]],
//             //                                              element_u[ev[1]],
//             //                                              element_u[ev[2]],
//             //                                              element_u[ev[3]],
//             //                                              &v[0],
//             //                                              &v[1],
//             //                                              &v[2],
//             //                                              &v[3]);

//             //                     for (int d = 0; d < 4; d++) {
//             //                         element_vector[ev[d]] += v[d];
//             //                     }

//             //                     p++;
//             //                 }
//             //                 p++;
//             //             }
//             //             p++;
//             //         }
//             //     }

//             //     ///////////////////////////////////
//             //     // Cat 4
//             //     ///////////////////////////////////
//             //     {
//             //         sstet4_sub_fff_4(level, &g_fff[e * 6], fff);

//             //         int p = 0;
//             //         for (int i = 1; i < level - 1; i++) {
//             //             p               = p + level - i + 1;
//             //             int layer_items = (level - i) * (level - i - 1) / 2;
//             //             for (int j = 0; j < level - i - 1; j++) {
//             //                 p++;
//             //                 for (int k = 1; k < level - i - j - 1; k++) {
//             //                     ev[0] = p;
//             //                     ev[1] = p + layer_items + level - i;
//             //                     ev[2] = p + layer_items + level - i - j + level - i;
//             //                     ev[3] = p + layer_items + level - i - j + level - i - 1;

//             //                     tet4_laplacian_apply_fff(fff,
//             //                                              element_u[ev[0]],
//             //                                              element_u[ev[1]],
//             //                                              element_u[ev[2]],
//             //                                              element_u[ev[3]],
//             //                                              &v[0],
//             //                                              &v[1],
//             //                                              &v[2],
//             //                                              &v[3]);

//             //                     for (int d = 0; d < 4; d++) {
//             //                         element_vector[ev[d]] += v[d];
//             //                     }

//             //                     p++;
//             //                 }
//             //                 p++;
//             //             }
//             //             p++;
//             //         }
//             //     }

//             //     ///////////////////////////////////
//             //     // Cat 5
//             //     ///////////////////////////////////
//             //     {
//             //         sstet4_sub_fff_5(level, &g_fff[e * 6], fff);

//             //         int p = 0;
//             //         for (int i = 0; i < level - 1; i++) {
//             //             int layer_items = (level - i) * (level - i - 1) / 2;
//             //             for (int j = 0; j < level - i - 1; j++) {
//             //                 p++;
//             //                 for (int k = 1; k < level - i - j - 1; k++) {
//             //                     ev[0] = p;
//             //                     ev[1] = p + level - i - j - 1;
//             //                     ev[2] = p + layer_items + level - i - j - 1 + level - i - j - 1;
//             //                     ev[3] = p + level - i - j;

//             //                     tet4_laplacian_apply_fff(fff,
//             //                                              element_u[ev[0]],
//             //                                              element_u[ev[1]],
//             //                                              element_u[ev[2]],
//             //                                              element_u[ev[3]],
//             //                                              &v[0],
//             //                                              &v[1],
//             //                                              &v[2],
//             //                                              &v[3]);

//             //                     for (int d = 0; d < 4; d++) {
//             //                         element_vector[ev[d]] += v[d];
//             //                     }

//             //                     p++;
//             //                 }
//             //                 p++;
//             //             }
//             //             p++;
//             //         }
//             //     }
//         }

//         free(tet_qx);
//         free(tet_qy);
//         free(tet_qz);

//         free(grid_qx);
//         free(grid_qy);
//         free(grid_qz);
//     }

//     return SFEM_SUCCESS;
// }