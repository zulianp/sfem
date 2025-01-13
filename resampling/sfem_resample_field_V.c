#include "sfem_resample_field.h"

#include "mass.h"
#include "matrixio_array.h"
#include "sfem_resample_V.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "quadratures_rule.h"
#include "sfem_resample_field_vec.h"

#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL
// #define real_type real_t

SFEM_INLINE vec_indices floor_V(const vec_real x) {
    const vec_indices res = __builtin_convertvector(x, vec_indices);
    return res;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_measure_V8 ///////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static vec_real           //
tet4_measure_V(const vec_real px0,    // X-coordinate
               const vec_real px1,    // X-coordinate
               const vec_real px2,    // X-coordinate
               const vec_real px3,    // X-coordinate
               const vec_real py0,    // Y-coordinate
               const vec_real py1,    // Y-coordinate
               const vec_real py2,    // Y-coordinate
               const vec_real py3,    // Y-coordinate
               const vec_real pz0,    // Z-coordinates
               const vec_real pz1,    // Z-coordinates
               const vec_real pz2,    // Z-coordinates
               const vec_real pz3) {  // Z-coordinates
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_t   ref_vol = 1. / 6;
    const vec_real x0      = -pz0 + pz3;
    const vec_real x1      = -py0 + py2;
    const vec_real x2      = -ref_vol * px0 + ref_vol * px1;
    const vec_real x3      = -py0 + py3;
    const vec_real x4      = -pz0 + pz2;
    const vec_real x5      = -py0 + py1;
    const vec_real x6      = -ref_vol * px0 + ref_vol * px2;
    const vec_real x7      = -pz0 + pz1;
    const vec_real x8      = -ref_vol * px0 + ref_vol * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_transform_V8 /////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void                                //
tet4_transform_V(const vec_real                px0,    // X-coordinates
                 const vec_real                px1,    //
                 const vec_real                px2,    //
                 const vec_real                px3,    //
                 const vec_real                py0,    // Y-coordinates
                 const vec_real                py1,    //
                 const vec_real                py2,    //
                 const vec_real                py3,    //
                 const vec_real                pz0,    // Z-coordinates
                 const vec_real                pz1,    //
                 const vec_real                pz2,    //
                 const vec_real                pz3,    //
                 const vec_real                qx,     // Quadrature point
                 const vec_real                qy,     //
                 const vec_real                qz,     //
                 vec_real* const SFEM_RESTRICT out_x,  // Output
                 vec_real* const SFEM_RESTRICT out_y,  //
                 vec_real* const SFEM_RESTRICT out_z) {
    /**
****************************************************************************************
\begin{bmatrix}
out_x \\
out_y \\
out_z
\end{bmatrix}
=
\begin{bmatrix}
px_0 \\
py_0 \\
pz_0
\end{bmatrix}
+
\begin{bmatrix}
px_1 - px_0 & px_2 - px_0 & px_3 - px_0 \\
py_1 - py_0 & py_2 - py_0 & py_3 - py_0 \\
pz_1 - pz_0 & pz_2 - pz_0 & pz_3 - pz_0
\end{bmatrix}
\cdot
\begin{bmatrix}
qx \\
qy \\
qz
\end{bmatrix}
*************************************************************************************************
*/
    //
    //
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}  // end tet4_transform_v2

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void                                //
hex_aa_8_eval_fun_V(const vec_real                x,   //
                    const vec_real                y,   //
                    const vec_real                z,   //
                    vec_real* const SFEM_RESTRICT f0,  // Output
                    vec_real* const SFEM_RESTRICT f1,  //
                    vec_real* const SFEM_RESTRICT f2,  //
                    vec_real* const SFEM_RESTRICT f3,  //
                    vec_real* const SFEM_RESTRICT f4,  //
                    vec_real* const SFEM_RESTRICT f5,  //
                    vec_real* const SFEM_RESTRICT f6,  //
                    vec_real* const SFEM_RESTRICT f7) {
    //

    // Quadrature point (local coordinates)
    // With respect to the hat functions of a cube element
    // In a local coordinate system

    *f0 = (1.0 - x) * (1.0 - y) * (1.0 - z);
    *f1 = x * (1.0 - y) * (1.0 - z);
    *f2 = x * y * (1.0 - z);
    *f3 = (1.0 - x) * y * (1.0 - z);
    *f4 = (1.0 - x) * (1.0 - y) * z;
    *f5 = x * (1.0 - y) * z;
    *f6 = x * y * z;
    *f7 = (1.0 - x) * y * z;
}

#if _VL_ == 8
#define GET_OUT_MACRO(_out, _data, _indx_V)   \
    {                                         \
        _out = (vec_real){_data[_indx_V[0]],  \
                          _data[_indx_V[1]],  \
                          _data[_indx_V[2]],  \
                          _data[_indx_V[3]],  \
                          _data[_indx_V[4]],  \
                          _data[_indx_V[5]],  \
                          _data[_indx_V[6]],  \
                          _data[_indx_V[7]]}; \
    }

#define GET_INDICES(_elems_, _element_i_)                                                                              \
    {                                                                                                                  \
        _elems_[_element_i_ + 0], _elems_[_element_i_ + 1], _elems_[_element_i_ + 2], _elems_[_element_i_ + 3],        \
                _elems_[_element_i_ + 4], _elems_[_element_i_ + 5], _elems_[_element_i_ + 6], _elems_[_element_i_ + 7] \
    }

#define COPY_COORDINATES(__ev__, __xyz_ind__)                                                                                  \
    (vec_real) {                                                                                                               \
        (real_t) xyz[__xyz_ind__][__ev__[0]], (real_t)xyz[__xyz_ind__][__ev__[1]], (real_t)xyz[__xyz_ind__][__ev__[2]],        \
                (real_t)xyz[__xyz_ind__][__ev__[3]], (real_t)xyz[__xyz_ind__][__ev__[4]], (real_t)xyz[__xyz_ind__][__ev__[5]], \
                (real_t)xyz[__xyz_ind__][__ev__[6]], (real_t)xyz[__xyz_ind__][__ev__[7]]                                       \
    }

#define ACCUMULATE_WFIELD(_indx_, _element_fieldN_)                          \
    {                                                                        \
        weighted_field[elems[_indx_][element_i + 0]] += _element_fieldN_[0]; \
        weighted_field[elems[_indx_][element_i + 1]] += _element_fieldN_[1]; \
        weighted_field[elems[_indx_][element_i + 2]] += _element_fieldN_[2]; \
        weighted_field[elems[_indx_][element_i + 3]] += _element_fieldN_[3]; \
        weighted_field[elems[_indx_][element_i + 4]] += _element_fieldN_[4]; \
        weighted_field[elems[_indx_][element_i + 5]] += _element_fieldN_[5]; \
        weighted_field[elems[_indx_][element_i + 6]] += _element_fieldN_[6]; \
        weighted_field[elems[_indx_][element_i + 7]] += _element_fieldN_[7]; \
    }

#elif _VL_ == 4
#define GET_OUT_MACRO(_out, _data, _indx_V) \
    { _out = (vec_real){_data[_indx_V[0]], _data[_indx_V[1]], _data[_indx_V[2]], _data[_indx_V[3]]}; }

#define GET_INDICES(_elems_, _element_i_) \
    { _elems_[_element_i_ + 0], _elems_[_element_i_ + 1], _elems_[_element_i_ + 2], _elems_[_element_i_ + 3] }

#define COPY_COORDINATES(__ev__, __xyz_ind__)                                                                           \
    (vec_real) {                                                                                                        \
        (real_t) xyz[__xyz_ind__][__ev__[0]], (real_t)xyz[__xyz_ind__][__ev__[1]], (real_t)xyz[__xyz_ind__][__ev__[2]], \
                (real_t)xyz[__xyz_ind__][__ev__[3]]                                                                     \
    }

#define ACCUMULATE_WFIELD(_indx_, _element_fieldN_)                          \
    {                                                                        \
        weighted_field[elems[_indx_][element_i + 0]] += _element_fieldN_[0]; \
        weighted_field[elems[_indx_][element_i + 1]] += _element_fieldN_[1]; \
        weighted_field[elems[_indx_][element_i + 2]] += _element_fieldN_[2]; \
        weighted_field[elems[_indx_][element_i + 3]] += _element_fieldN_[3]; \
    }

#elif _VL_ == 16
#define GET_OUT_MACRO(_out, _data, _indx_V)    \
    {                                          \
        _out = (vec_real){_data[_indx_V[0]],   \
                          _data[_indx_V[1]],   \
                          _data[_indx_V[2]],   \
                          _data[_indx_V[3]],   \
                          _data[_indx_V[4]],   \
                          _data[_indx_V[5]],   \
                          _data[_indx_V[6]],   \
                          _data[_indx_V[7]],   \
                          _data[_indx_V[8]],   \
                          _data[_indx_V[9]],   \
                          _data[_indx_V[10]],  \
                          _data[_indx_V[11]],  \
                          _data[_indx_V[12]],  \
                          _data[_indx_V[13]],  \
                          _data[_indx_V[14]],  \
                          _data[_indx_V[15]]}; \
    }

// #define GET_INDICES(_elems_, _element_i_)                                                                                  \
//     {                                                                                                                      \
//         _elems_[_element_i_ + 0], _elems_[_element_i_ + 1], _elems_[_element_i_ + 2], _elems_[_element_i_ + 3],            \
//                 _elems_[_element_i_ + 4], _elems_[_element_i_ + 5], _elems_[_element_i_ + 6], _elems_[_element_i_ + 7],    \
//                 _elems_[_element_i_ + 8], _elems_[_element_i_ + 9], _elems_[_element_i_ + 10], _elems_[_element_i_ + 11],  \
//                 _elems_[_element_i_ + 12], _elems_[_element_i_ + 13], _elems_[_element_i_ + 14], _elems_[_element_i_ + 15] \
//     }

#define GET_INDICES(_elems_, _element_i_)                                                                                  \
    {                                                                                                                      \
        _elems_[_element_i_ + 0], _elems_[_element_i_ + 1], _elems_[_element_i_ + 2], _elems_[_element_i_ + 3],            \
                _elems_[_element_i_ + 4], _elems_[_element_i_ + 5], _elems_[_element_i_ + 6], _elems_[_element_i_ + 7],    \
                _elems_[_element_i_ + 8], _elems_[_element_i_ + 9], _elems_[_element_i_ + 10], _elems_[_element_i_ + 11],  \
                _elems_[_element_i_ + 12], _elems_[_element_i_ + 13], _elems_[_element_i_ + 14], _elems_[_element_i_ + 15] \
    }

#define COPY_COORDINATES(__ev__, __xyz_ind__)                                                                               \
    (vec_real) {                                                                                                            \
        xyz[__xyz_ind__][__ev__[0]], xyz[__xyz_ind__][__ev__[1]], xyz[__xyz_ind__][__ev__[2]], xyz[__xyz_ind__][__ev__[3]], \
                xyz[__xyz_ind__][__ev__[4]], xyz[__xyz_ind__][__ev__[5]], xyz[__xyz_ind__][__ev__[6]],                      \
                xyz[__xyz_ind__][__ev__[7]], xyz[__xyz_ind__][__ev__[8]], xyz[__xyz_ind__][__ev__[9]],                      \
                xyz[__xyz_ind__][__ev__[10]], xyz[__xyz_ind__][__ev__[11]], xyz[__xyz_ind__][__ev__[12]],                   \
                xyz[__xyz_ind__][__ev__[13]], xyz[__xyz_ind__][__ev__[14]], xyz[__xyz_ind__][__ev__[15]]                    \
    }

#define ACCUMULATE_WFIELD(_indx_, _element_fieldN_)                            \
    {                                                                          \
        weighted_field[elems[_indx_][element_i + 0]] += _element_fieldN_[0];   \
        weighted_field[elems[_indx_][element_i + 1]] += _element_fieldN_[1];   \
        weighted_field[elems[_indx_][element_i + 2]] += _element_fieldN_[2];   \
        weighted_field[elems[_indx_][element_i + 3]] += _element_fieldN_[3];   \
        weighted_field[elems[_indx_][element_i + 4]] += _element_fieldN_[4];   \
        weighted_field[elems[_indx_][element_i + 5]] += _element_fieldN_[5];   \
        weighted_field[elems[_indx_][element_i + 6]] += _element_fieldN_[6];   \
        weighted_field[elems[_indx_][element_i + 7]] += _element_fieldN_[7];   \
        weighted_field[elems[_indx_][element_i + 8]] += _element_fieldN_[8];   \
        weighted_field[elems[_indx_][element_i + 9]] += _element_fieldN_[9];   \
        weighted_field[elems[_indx_][element_i + 10]] += _element_fieldN_[10]; \
        weighted_field[elems[_indx_][element_i + 11]] += _element_fieldN_[11]; \
        weighted_field[elems[_indx_][element_i + 12]] += _element_fieldN_[12]; \
        weighted_field[elems[_indx_][element_i + 13]] += _element_fieldN_[13]; \
        weighted_field[elems[_indx_][element_i + 14]] += _element_fieldN_[14]; \
        weighted_field[elems[_indx_][element_i + 15]] += _element_fieldN_[15]; \
    }

#endif

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs_V8 ////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void hex_aa_8_collect_coeffs_V(const vec_indices stride0, const vec_indices stride1, const vec_indices stride2,
                                                  const vec_indices i, const vec_indices j, const vec_indices k,
                                                  // Attention this is geometric data transformed to solver data!
                                                  const real_t* const SFEM_RESTRICT data,
                                                  //
                                                  vec_real* SFEM_RESTRICT out0, vec_real* SFEM_RESTRICT out1,
                                                  vec_real* SFEM_RESTRICT out2, vec_real* SFEM_RESTRICT out3,
                                                  vec_real* SFEM_RESTRICT out4, vec_real* SFEM_RESTRICT out5,
                                                  vec_real* SFEM_RESTRICT out6, vec_real* SFEM_RESTRICT out7) {
    //

    const vec_indices i0 = i * stride0 + j * stride1 + k * stride2;
    const vec_indices i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const vec_indices i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const vec_indices i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const vec_indices i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const vec_indices i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const vec_indices i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const vec_indices i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

    GET_OUT_MACRO(*out0, data, i0);
    GET_OUT_MACRO(*out1, data, i1);
    GET_OUT_MACRO(*out2, data, i2);
    GET_OUT_MACRO(*out3, data, i3);
    GET_OUT_MACRO(*out4, data, i4);
    GET_OUT_MACRO(*out5, data, i5);
    GET_OUT_MACRO(*out6, data, i6);
    GET_OUT_MACRO(*out7, data, i7);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                       //
tet4_resample_field_local_V_aligned(const ptrdiff_t                      start_nelement,  // Mesh
                                    const ptrdiff_t                      end_nelement,    //
                                    const ptrdiff_t                      nnodes,          //
                                    idx_t** const SFEM_RESTRICT          elems,           //
                                    geom_t** const SFEM_RESTRICT         xyz,             //
                                    const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                    const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                    const geom_t* const SFEM_RESTRICT    origin,          //
                                    const geom_t* const SFEM_RESTRICT    delta,           //
                                    const real_t* const SFEM_RESTRICT    data,            //
                                    real_t* const SFEM_RESTRICT          weighted_field) {         // Output
    //
    PRINT_CURRENT_FUNCTION;

    printf("============================================================\n");
    printf("Start: tet4_resample_field_local_V_aligned  V8 [%s:%d] \n", __FILE__, __LINE__);
    printf("============================================================\n");
    //
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    vec_indices stride0 = CONST_VEC(stride[0]);
    vec_indices stride1 = CONST_VEC(stride[1]);
    vec_indices stride2 = CONST_VEC(stride[2]);

    //////////////////////////////////////////////////////////////////////
    // Loop over the elements
    for (ptrdiff_t element_i = start_nelement; element_i < end_nelement; element_i += _VL_) {
        //

        vec_indices ev0 = GET_INDICES(elems[0], element_i);
        vec_indices ev1 = GET_INDICES(elems[1], element_i);
        vec_indices ev2 = GET_INDICES(elems[2], element_i);
        vec_indices ev3 = GET_INDICES(elems[3], element_i);

        const vec_real zeros = ZEROS_VEC();

        // Vertices coordinates of the tetrahedron
        vec_real x0 = zeros, x1 = zeros, x2 = zeros, x3 = zeros;

        vec_real y0 = zeros, y1 = zeros, y2 = zeros, y3 = zeros;
        vec_real z0 = zeros, z1 = zeros, z2 = zeros, z3 = zeros;

        // real_t hex8_f[8];
        vec_real hex8_f0 = zeros, hex8_f1 = zeros, hex8_f2 = zeros, hex8_f3 = zeros, hex8_f4 = zeros, hex8_f5 = zeros,
                 hex8_f6 = zeros, hex8_f7 = zeros;

        // real_t coeffs[8];
        vec_real coeffs0 = zeros, coeffs1 = zeros, coeffs2 = zeros, coeffs3 = zeros, coeffs4 = zeros, coeffs5 = zeros,
                 coeffs6 = zeros, coeffs7 = zeros;

        // real_t tet4_f[4];
        vec_real tet4_f0 = zeros, tet4_f1 = zeros, tet4_f2 = zeros, tet4_f3 = zeros;

        // real_t element_field[4];
        vec_real element_field0 = zeros, element_field1 = zeros, element_field2 = zeros, element_field3 = zeros;

        // copy the coordinates of the vertices
        {
            x0 = COPY_COORDINATES(ev0, 0);
            x1 = COPY_COORDINATES(ev1, 0);
            x2 = COPY_COORDINATES(ev2, 0);
            x3 = COPY_COORDINATES(ev3, 0);

            y0 = COPY_COORDINATES(ev0, 1);
            y1 = COPY_COORDINATES(ev1, 1);
            y2 = COPY_COORDINATES(ev2, 1);
            y3 = COPY_COORDINATES(ev3, 1);

            z0 = COPY_COORDINATES(ev0, 2);
            z1 = COPY_COORDINATES(ev1, 2);
            z2 = COPY_COORDINATES(ev2, 2);
            z3 = COPY_COORDINATES(ev3, 2);

        }  // end copy the coordinates of the vertices

        // Volume of the tetrahedrons (8 at a time)
        const vec_real theta_volume = tet4_measure_V(x0,
                                                     x1,
                                                     x2,
                                                     x3,
                                                     //
                                                     y0,
                                                     y1,
                                                     y2,
                                                     y3,
                                                     //
                                                     z0,
                                                     z1,
                                                     z2,
                                                     z3);

        //////////////////////////////////////////////////////////////////////
        // loop over the quadrature points
        for (int quad_i = 0; quad_i < TET4_NQP; quad_i++) {
            //
            vec_real g_qx, g_qy, g_qz;

            vec_real tet4_qx_v = CONST_VEC(tet4_qx[quad_i]);
            vec_real tet4_qy_v = CONST_VEC(tet4_qy[quad_i]);
            vec_real tet4_qz_v = CONST_VEC(tet4_qz[quad_i]);

            tet4_transform_V(x0,
                             x1,
                             x2,
                             x3,
                             //
                             y0,
                             y1,
                             y2,
                             y3,
                             //
                             z0,
                             z1,
                             z2,
                             z3,
                             //
                             tet4_qx_v,
                             tet4_qy_v,
                             tet4_qz_v,
                             //
                             &g_qx,
                             &g_qy,
                             &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
            // Standard basis function
            {
                // tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
                // tet4_f[1] = tet4_qx[q];
                // tet4_f[2] = tet4_qy[q];
                // tet4_f[2] = tet4_qz[q];
            }
#else
            // DUAL basis function
            {
                const vec_real f0 = 1.0 - tet4_qx_v - tet4_qy_v - tet4_qz_v;
                const vec_real f1 = tet4_qx_v;
                const vec_real f2 = tet4_qy_v;
                const vec_real f3 = tet4_qz_v;

                tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
                tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
                tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
                tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
            }
#endif

            const vec_real grid_x = (g_qx - ox) / dx;
            const vec_real grid_y = (g_qy - oy) / dy;
            const vec_real grid_z = (g_qz - oz) / dz;

            const vec_indices i = floor_V(grid_x);
            const vec_indices j = floor_V(grid_y);
            const vec_indices k = floor_V(grid_z);

            // Get the reminder [0, 1]
            vec_real l_x = (grid_x - __builtin_convertvector(i, vec_real));
            vec_real l_y = (grid_y - __builtin_convertvector(j, vec_real));
            vec_real l_z = (grid_z - __builtin_convertvector(k, vec_real));

            // Critical point
            hex_aa_8_eval_fun_V(l_x, l_y, l_z, &hex8_f0, &hex8_f1, &hex8_f2, &hex8_f3, &hex8_f4, &hex8_f5, &hex8_f6, &hex8_f7);

            hex_aa_8_collect_coeffs_V(stride0,
                                      stride1,
                                      stride2,
                                      i,
                                      j,
                                      k,
                                      data,
                                      &coeffs0,
                                      &coeffs1,
                                      &coeffs2,
                                      &coeffs3,
                                      &coeffs4,
                                      &coeffs5,
                                      &coeffs6,
                                      &coeffs7);

            // Integrate gap function
            {
                vec_real eval_field = ZEROS_VEC();
                // UNROLL_ZERO
                // for (int edof_j = 0; edof_j < 8; edof_j++) {
                //     eval_field += hex8_f[edof_j] * coeffs[edof_j];
                // }
                eval_field += hex8_f0 * coeffs0;
                eval_field += hex8_f1 * coeffs1;
                eval_field += hex8_f2 * coeffs2;
                eval_field += hex8_f3 * coeffs3;
                eval_field += hex8_f4 * coeffs4;
                eval_field += hex8_f5 * coeffs5;
                eval_field += hex8_f6 * coeffs6;
                eval_field += hex8_f7 * coeffs7;

                // eval_field = (vec_real)CONST_VEC(1.0f);

                // UNROLL_ZERO
                // for (int edof_i = 0; edof_i < 4; edof_i++) {
                //     element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
                // }  // end edof_i loop

                vec_real dV = theta_volume * tet4_qw[quad_i];
                // dV = (vec8_t){1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

                element_field0 += eval_field * tet4_f0 * dV;
                element_field1 += eval_field * tet4_f1 * dV;
                element_field2 += eval_field * tet4_f2 * dV;
                element_field3 += eval_field * tet4_f3 * dV;

                // element_field0 = (vec_real) CONST_VEC(1.0f);
                // element_field1 = (vec_real) CONST_VEC(1.0f);
                // element_field2 = (vec_real) CONST_VEC(1.0f);
                // element_field3 = (vec_real) CONST_VEC(1.0f);

            }  // end integrate gap function

        }  // end for over quadrature points

        ACCUMULATE_WFIELD(0, element_field0);
        ACCUMULATE_WFIELD(1, element_field1);
        ACCUMULATE_WFIELD(2, element_field2);
        ACCUMULATE_WFIELD(3, element_field3);

    }  // end for over elements

    RETURN_FROM_FUNCTION(0);
}  // end tet4_resample_field_local_V8_aligned

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_local_V(const ptrdiff_t                      nelements,  // Mesh: number of elements
                                const ptrdiff_t                      nnodes,     // Mesh: number of nodes
                                idx_t** const SFEM_RESTRICT          elems,      // Mesh: connectivity
                                geom_t** const SFEM_RESTRICT         xyz,        // Mesh: coordinates
                                const ptrdiff_t* const SFEM_RESTRICT n,          // SDF: number of nodes in each direction
                                const ptrdiff_t* const SFEM_RESTRICT stride,     // SDF: stride
                                const geom_t* const SFEM_RESTRICT    origin,     // SDF: origin
                                const geom_t* const SFEM_RESTRICT    delta,      // SDF: delta
                                const real_t* const SFEM_RESTRICT    data,       // SDF: data
                                real_t* const SFEM_RESTRICT          weighted_field) {    // Output
    //
    PRINT_CURRENT_FUNCTION;

    const ptrdiff_t nelements_aligned = nelements - (nelements % _VL_);
    const ptrdiff_t nelements_tail    = nelements % _VL_;

    printf("=============================================\n");
    printf("nelements_aligned = %ld\n", nelements_aligned);
    printf("nelements_tail =    %ld\n", nelements_tail);
    printf("=============================================\n");

    int ret = 0;

    ret = tet4_resample_field_local_V_aligned(0,                  // start_nelement
                                              nelements_aligned,  // end_nelement
                                              nnodes,             //
                                              elems,              //
                                              xyz,                //
                                              n,                  //
                                              stride,             //
                                              origin,             //
                                              delta,              //
                                              data,               //
                                              weighted_field);    //

    if (nelements_tail > 0) {
        ret = ret || tet4_resample_field_local_v2(nelements_aligned,  // start_nelement: for the tail
                                                  nelements,          // end_nelement:   for the tail
                                                  nnodes,             //
                                                  elems,              //
                                                  xyz,                //
                                                  n,                  //
                                                  stride,             //
                                                  origin,             //
                                                  delta,              //
                                                  data,               //
                                                  weighted_field);    //
    }

    RETURN_FROM_FUNCTION(ret);
}