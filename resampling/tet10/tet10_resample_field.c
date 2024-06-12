#include "tet10_resample_field.h"

// int tet10_resample_field_local(
//         // Mesh
//         const ptrdiff_t nelements,          // number of elements
//         const ptrdiff_t nnodes,             // number of nodes
//         idx_t** const SFEM_RESTRICT elems,  // connectivity
//         geom_t** const SFEM_RESTRICT xyz,   // coordinates
//         // SDF
//         const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
//         const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
//         const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
//         const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
//         const real_t* const SFEM_RESTRICT data,       // SDF
//         // Output
//         real_t* const SFEM_RESTRICT weighted_field) {
//     //
//     printf("============================================================\n");
//     printf("Start: tet4_resample_field_local\n");
//     printf("============================================================\n");
//     //
//     const real_t ox = (real_t)origin[0];
//     const real_t oy = (real_t)origin[1];
//     const real_t oz = (real_t)origin[2];

//     const real_t dx = (real_t)delta[0];
//     const real_t dy = (real_t)delta[1];
//     const real_t dz = (real_t)delta[2];

//     // printf("\nnumber of elements %ld  +++++++++++++++++++++++++++++++++++ \n", nelements);

// #pragma omp parallel
//     {
// #pragma omp for  // nowait

//         /// Loop over the elements of the mesh
//         for (ptrdiff_t i = 0; i < nelements; ++i) {
//             idx_t ev[4];
//             geom_t x[4], y[4], z[4];

//             real_t hex8_f[8];
//             real_t coeffs[8];

//             real_t tet4_f[4];
//             real_t element_field[4];

//             // loop over the 4 vertices of the tetrahedron
//             UNROLL_ZERO
//             for (int v = 0; v < 4; ++v) {
//                 ev[v] = elems[v][i];
//             }

//             // copy the coordinates of the vertices
//             for (int v = 0; v < 4; ++v) {
//                 x[v] = xyz[0][ev[v]];  // x-coordinates
//                 y[v] = xyz[1][ev[v]];  // y-coordinates
//                 z[v] = xyz[2][ev[v]];  // z-coordinates
//             }

//             memset(element_field, 0,
//                    4 * sizeof(real_t));  // set to zero the element field

//             // Area of the tetrahedron
//             const real_t measure = tet4_measure(x[0],
//                                                 x[1],
//                                                 x[2],
//                                                 x[3],
//                                                 //
//                                                 y[0],
//                                                 y[1],
//                                                 y[2],
//                                                 y[3],
//                                                 //
//                                                 z[0],
//                                                 z[1],
//                                                 z[2],
//                                                 z[3]);

//             assert(measure > 0);

//             for (int q = 0; q < TET4_NQP; q++) {  // loop over the quadrature points

//                 real_t g_qx, g_qy, g_qz;
//                 // Transform quadrature point to physical space
//                 // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
//                 // space
//                 tet4_transform(x[0],
//                                x[1],
//                                x[2],
//                                x[3],
//                                //
//                                y[0],
//                                y[1],
//                                y[2],
//                                y[3],
//                                //
//                                z[0],
//                                z[1],
//                                z[2],
//                                z[3],
//                                //
//                                tet4_qx[q],
//                                tet4_qy[q],
//                                tet4_qz[q],
//                                //
//                                &g_qx,
//                                &g_qy,
//                                &g_qz);

// #ifndef SFEM_RESAMPLE_GAP_DUAL
//                 // Standard basis function
//                 {
//                     tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
//                     tet4_f[1] = tet4_qx[q];
//                     tet4_f[2] = tet4_qy[q];
//                     tet4_f[2] = tet4_qz[q];
//                 }
// #else
//                 // DUAL basis function
//                 {
//                     const real_t f0 = 1.0 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
//                     const real_t f1 = tet4_qx[q];
//                     const real_t f2 = tet4_qy[q];
//                     const real_t f3 = tet4_qz[q];

//                     tet4_f[0] = 4 * f0 - f1 - f2 - f3;
//                     tet4_f[1] = -f0 + 4 * f1 - f2 - f3;
//                     tet4_f[2] = -f0 - f1 + 4 * f2 - f3;
//                     tet4_f[3] = -f0 - f1 - f2 + 4 * f3;
//                 }
// #endif
//                 const real_t dV = measure * tet4_qw[q];

//                 const real_t grid_x = (g_qx - ox) / dx;
//                 const real_t grid_y = (g_qy - oy) / dy;
//                 const real_t grid_z = (g_qz - oz) / dz;

//                 const ptrdiff_t i = floor(grid_x);
//                 const ptrdiff_t j = floor(grid_y);
//                 const ptrdiff_t k = floor(grid_z);

//                 // If outside
//                 if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) ||
//                     (k + 1 >= n[2])) {
//                     fprintf(stderr,
//                             "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
//                             "%ld)!\n",
//                             g_qx,
//                             g_qy,
//                             g_qz,
//                             i,
//                             j,
//                             k,
//                             n[0],
//                             n[1],
//                             n[2]);
//                     continue;
//                 }

//                 // Get the reminder [0, 1]
//                 real_t l_x = (grid_x - i);
//                 real_t l_y = (grid_y - j);
//                 real_t l_z = (grid_z - k);

//                 assert(l_x >= -1e-8);
//                 assert(l_y >= -1e-8);
//                 assert(l_z >= -1e-8);

//                 assert(l_x <= 1 + 1e-8);
//                 assert(l_y <= 1 + 1e-8);
//                 assert(l_z <= 1 + 1e-8);

//                 hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);
//                 hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

//                 // Integrate gap function
//                 {
//                     real_t eval_field = 0;
//                     UNROLL_ZERO
//                     for (int edof_j = 0; edof_j < 8; edof_j++) {
//                         eval_field += hex8_f[edof_j] * coeffs[edof_j];
//                     }

//                     UNROLL_ZERO
//                     for (int edof_i = 0; edof_i < 4; edof_i++) {
//                         element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
//                     }  // end edof_i loop
//                 }
//             }  // end quadrature loop

//             UNROLL_ZERO
//             for (int v = 0; v < 4; ++v) {
//                 // Invert sign since distance field is negative insdide and positive outside
// #pragma omp critical
//                 { weighted_field[ev[v]] += element_field[v]; }

//             }  // end vertex loop
//         }      // end element loop
//     }          // end parallel region

//     return 0;
// }