#ifndef __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_H__
#define __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_H__

#include "sfem_resample_field.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_tet4_math.h"
#include "sfem_stack.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "hyteg.h"
#include "hyteg_Jacobian_matrix_real_t.h"

#include "quadratures_rule.h"

typedef struct {
    real_t x, y, z;    // Physical coordinates
    real_t weight;     // Physical weight
    bool   is_inside;  // Containment result
} quadrature_point_result_t;

typedef struct {
    ptrdiff_t i;
    ptrdiff_t j;
    ptrdiff_t k;
    bool      inside_tet;
} ijk_index_t;

typedef enum { TET_QUAD_MIDPOINT_NQP, TET_QUAD_GAUSS_LEGENDRE_NQP } tet_quad_midpoint_nqp_t;

quadrature_point_result_t                                                  //
transform_quadrature_point_n(const int                         q_ijk,      //
                             const real_t* const SFEM_RESTRICT Q_nodes_x,  //
                             const real_t* const SFEM_RESTRICT Q_nodes_y,  //
                             const real_t* const SFEM_RESTRICT Q_nodes_z,  //
                             const real_t* const SFEM_RESTRICT Q_weights,  //
                             const geom_t* const SFEM_RESTRICT origin,     //
                             const geom_t* const SFEM_RESTRICT delta,      //
                             const ptrdiff_t                   i_grid,     //
                             const ptrdiff_t                   j_grid,     //
                             const ptrdiff_t                   k_grid);                      //

quadrature_point_result_t                                                             //
transform_and_check_quadrature_point_n(const int           q_ijk,                     //
                                       const real_t        tet4_faces_normals[4][3],  //
                                       const real_t        faces_centroids[4][3],     //
                                       const real_t* const Q_nodes_x,                 //
                                       const real_t* const Q_nodes_y,                 //
                                       const real_t* const Q_nodes_z,                 //
                                       const real_t* const Q_weights,                 //
                                       const geom_t* const origin,                    //
                                       const geom_t* const delta,                     //
                                       const ptrdiff_t     i_grid,                    //
                                       const ptrdiff_t     j_grid,                    //
                                       const ptrdiff_t     k_grid,                    //
                                       const real_t        tet_vertices_x[4],         //
                                       const real_t        tet_vertices_y[4],         //
                                       const real_t        tet_vertices_z[4]);               //

void tet4_inv_transform_J(const real_t*               J_inv,   //
                          const real_t                pfx,     //
                          const real_t                pfy,     //
                          const real_t                pfz,     //
                          const real_t                px0,     //
                          const real_t                py0,     //
                          const real_t                pz0,     //
                          real_t* const SFEM_RESTRICT out_x,   //
                          real_t* const SFEM_RESTRICT out_y,   //
                          real_t* const SFEM_RESTRICT out_z);  //

ijk_index_t                                                                              //
transfer_weighted_field_tet4_to_hex_ckp(const real_t                wf0,                 //
                                        const real_t                wf1,                 //
                                        const real_t                wf2,                 //
                                        const real_t                wf3,                 //
                                        const real_t                q_phys_x,            //
                                        const real_t                q_phys_y,            //
                                        const real_t                q_phys_z,            //
                                        const real_t                q_ref_x,             //
                                        const real_t                q_ref_y,             //
                                        const real_t                q_ref_z,             //
                                        const real_t                QW_phys_hex,         //
                                        const real_t                ox,                  //
                                        const real_t                oy,                  //
                                        const real_t                oz,                  //
                                        const real_t                dx,                  //
                                        const real_t                dy,                  //
                                        const real_t                dz,                  //
                                        real_t* const SFEM_RESTRICT hex_element_field);  //

void                                 //
tet4_inv_Jacobian(const real_t px0,  //
                  const real_t px1,  //
                  const real_t px2,  //
                  const real_t px3,  //
                  const real_t py0,  //
                  const real_t py1,  //
                  const real_t py2,  //
                  const real_t py3,  //
                  const real_t pz0,  //
                  const real_t pz1,  //
                  const real_t pz2,  //
                  const real_t pz3,  //
                  real_t       J_inv[9]);  //

int                                                                     //
compute_tet_bounding_box_a(const real_t                   x0,           //
                           const real_t                   x1,           //
                           const real_t                   x2,           //
                           const real_t                   x3,           //
                           const real_t                   y0,           //
                           const real_t                   y1,           //
                           const real_t                   y2,           //
                           const real_t                   y3,           //
                           const real_t                   z0,           //
                           const real_t                   z1,           //
                           const real_t                   z2,           //
                           const real_t                   z3,           //
                           const ptrdiff_t                stride0,      //
                           const ptrdiff_t                stride1,      //
                           const ptrdiff_t                stride2,      //
                           const geom_t                   origin0,      //
                           const geom_t                   origin1,      //
                           const geom_t                   origin2,      //
                           const geom_t                   inv_delta0,   //
                           const geom_t                   inv_delta1,   //
                           const geom_t                   inv_delta2,   //
                           ptrdiff_t* const SFEM_RESTRICT min_grid_x,   //
                           ptrdiff_t* const SFEM_RESTRICT max_grid_x,   //
                           ptrdiff_t* const SFEM_RESTRICT min_grid_y,   //
                           ptrdiff_t* const SFEM_RESTRICT max_grid_y,   //
                           ptrdiff_t* const SFEM_RESTRICT min_grid_z,   //
                           ptrdiff_t* const SFEM_RESTRICT max_grid_z);  //

bool                                                //
is_hex_out_of_tet(const real_t inv_J_tet[9],        //
                  const real_t tet_origin_x,        //
                  const real_t tet_origin_y,        //
                  const real_t tet_origin_z,        //
                  const real_t hex_vertices_x[8],   //
                  const real_t hex_vertices_y[8],   //
                  const real_t hex_vertices_z[8]);  //

void                                                           //
is_hex_out_of_tet_step2h(const real_t inv_J_tet[9],            //
                         const real_t tet_origin_x,            //
                         const real_t tet_origin_y,            //
                         const real_t tet_origin_z,            //
                         const real_t hex_vertices_x_arg[16],  //
                         const real_t hex_vertices_y_arg[16],  //
                         const real_t hex_vertices_z_arg[16],  //
                         bool         in_out[2]);                      //

int                                                    //
sfem_quad_rule_3D(const tet_quad_midpoint_nqp_t rule,  //
                  const int                     N,     //
                  real_t*                       qx,    //
                  real_t*                       qy,    //
                  real_t*                       qz,    //
                  real_t*                       qw);                         //

#endif  // __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_H__