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
SFEM_INLINE static vec_real tet4_measure_V(
        // X-coordinates
        const vec_real px0, const vec_real px1, const vec_real px2, const vec_real px3,
        // Y-coordinates
        const vec_real py0, const vec_real py1, const vec_real py2, const vec_real py3,
        // Z-coordinates
        const vec_real pz0, const vec_real pz1, const vec_real pz2, const vec_real pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_t ref_vol = 1. / 6;
    const vec_real x0 = -pz0 + pz3;
    const vec_real x1 = -py0 + py2;
    const vec_real x2 = -ref_vol * px0 + ref_vol * px1;
    const vec_real x3 = -py0 + py3;
    const vec_real x4 = -pz0 + pz2;
    const vec_real x5 = -py0 + py1;
    const vec_real x6 = -ref_vol * px0 + ref_vol * px2;
    const vec_real x7 = -pz0 + pz1;
    const vec_real x8 = -ref_vol * px0 + ref_vol * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_transform_V8 /////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void tet4_transform_V(
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

        // X-coordinates
        const vec_real px0, const vec_real px1, const vec_real px2, const vec_real px3,
        // Y-coordinates
        const vec_real py0, const vec_real py1, const vec_real py2, const vec_real py3,
        // Z-coordinates
        const vec_real pz0, const vec_real pz1, const vec_real pz2, const vec_real pz3,
        // Quadrature point
        const vec_real qx, const vec_real qy, const vec_real qz,
        // Output
        vec_real* const SFEM_RESTRICT out_x, vec_real* const SFEM_RESTRICT out_y,
        vec_real* const SFEM_RESTRICT out_z) {
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
SFEM_INLINE static void hex_aa_8_eval_fun_V(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const vec_real x, const vec_real y, const vec_real z,

        // Output
        vec_real* const SFEM_RESTRICT f0, vec_real* const SFEM_RESTRICT f1,
        vec_real* const SFEM_RESTRICT f2, vec_real* const SFEM_RESTRICT f3,
        vec_real* const SFEM_RESTRICT f4, vec_real* const SFEM_RESTRICT f5,
        vec_real* const SFEM_RESTRICT f6, vec_real* const SFEM_RESTRICT f7) {
    //
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

#define GET_INDICES(_elems_, _element_i_)                                                     \
    {                                                                                         \
        _elems_[_element_i_ + 0], _elems_[_element_i_ + 1], _elems_[_element_i_ + 2],         \
                _elems_[_element_i_ + 3], _elems_[_element_i_ + 4], _elems_[_element_i_ + 5], \
                _elems_[_element_i_ + 6], _elems_[_element_i_ + 7]                            \
    }

#elif _VL_ == 4
#define GET_OUT_MACRO(_out, _data, _indx_V)                                                  \
    {                                                                                        \
        _out = (vec_real){                                                                   \
                _data[_indx_V[0]], _data[_indx_V[1]], _data[_indx_V[2]], _data[_indx_V[3]]}; \
    }

#define GET_INDICES(_elems_, _element_i_)                                             \
    {                                                                                 \
        _elems_[_element_i_ + 0], _elems_[_element_i_ + 1], _elems_[_element_i_ + 2], \
                _elems_[_element_i_ + 3]                                              \
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

#define GET_INDICES(_elems_, _element_i_)                                                        \
    {                                                                                            \
        _elems_[_element_i_ + 0], _elems_[_element_i_ + 1], _elems_[_element_i_ + 2],            \
                _elems_[_element_i_ + 3], _elems_[_element_i_ + 4], _elems_[_element_i_ + 5],    \
                _elems_[_element_i_ + 6], _elems_[_element_i_ + 7], _elems_[_element_i_ + 8],    \
                _elems_[_element_i_ + 9], _elems_[_element_i_ + 10], _elems_[_element_i_ + 11],  \
                _elems_[_element_i_ + 12], _elems_[_element_i_ + 13], _elems_[_element_i_ + 14], \
                _elems_[_element_i_ + 15]                                                        \
    }

#endif

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs_V8 ////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void hex_aa_8_collect_coeffs_V(
        const vec_indices8_t stride0, const vec_indices8_t stride1, const vec_indices8_t stride2,
        const vec_indices8_t i, const vec_indices8_t j, const vec_indices8_t k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data,
        //
        vec_real* SFEM_RESTRICT out0, vec_real* SFEM_RESTRICT out1, vec_real* SFEM_RESTRICT out2,
        vec_real* SFEM_RESTRICT out3, vec_real* SFEM_RESTRICT out4, vec_real* SFEM_RESTRICT out5,
        vec_real* SFEM_RESTRICT out6, vec_real* SFEM_RESTRICT out7) {
    //

    const vec_indices8_t i0 = i * stride0 + j * stride1 + k * stride2;
    const vec_indices8_t i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const vec_indices8_t i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const vec_indices8_t i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const vec_indices8_t i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const vec_indices8_t i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const vec_indices8_t i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const vec_indices8_t i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

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
int tet4_resample_field_local_V8_aligned(
        // Mesh
        const ptrdiff_t start_nelement, const ptrdiff_t end_nelement, const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
    //
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local_V8_aligned  V8 [%s] \n", __FILE__);
    printf("============================================================\n");
    //
    const scalar_t ox = (scalar_t)origin[0];
    const scalar_t oy = (scalar_t)origin[1];
    const scalar_t oz = (scalar_t)origin[2];

    const scalar_t dx = (scalar_t)delta[0];
    const scalar_t dy = (scalar_t)delta[1];
    const scalar_t dz = (scalar_t)delta[2];

    vec_indices8_t stride0 = CONST_VEC(stride[0]);
    vec_indices8_t stride1 = CONST_VEC(stride[1]);
    vec_indices8_t stride2 = CONST_VEC(stride[2]);

    //////////////////////////////////////////////////////////////////////
    // Loop over the elements
    for (ptrdiff_t element_i = start_nelement; element_i < end_nelement; element_i += 8) {
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
        vec_real hex8_f0 = zeros, hex8_f1 = zeros, hex8_f2 = zeros, hex8_f3 = zeros,
                 hex8_f4 = zeros, hex8_f5 = zeros, hex8_f6 = zeros, hex8_f7 = zeros;

        // real_t coeffs[8];
        vec_real coeffs0 = zeros, coeffs1 = zeros, coeffs2 = zeros, coeffs3 = zeros,
                 coeffs4 = zeros, coeffs5 = zeros, coeffs6 = zeros, coeffs7 = zeros;

        // real_t tet4_f[4];
        vec_real tet4_f0 = zeros, tet4_f1 = zeros, tet4_f2 = zeros, tet4_f3 = zeros;

        // real_t element_field[4];
        vec_real element_field0 = zeros, element_field1 = zeros, element_field2 = zeros,
                 element_field3 = zeros;

    }  // end for over elements

    return 0;
}  // end tet4_resample_field_local_V8_aligned