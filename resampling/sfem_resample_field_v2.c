#include "sfem_resample_field.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// #define real_t double

#include "quadratures_rule.h"

#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

/**
 * @brief Transform a quadrature point from reference to physical coordinates.
 *
 * Given the vertices of a tetrahedron (first vertex plus differences),
 * compute the transformed coordinates ([out_x, out_y, out_z]) at the given
 * quadrature point ([qx, qy, qz]). The mapping is defined by:
 *
 *   [out_x; out_y; out_z] = [px0; py0; pz0] +
 *    [px1 - px0, px2 - px0, px3 - px0;
 *     py1 - py0, py2 - py0, py3 - py0;
 *     pz1 - pz0, pz2 - pz0, pz3 - pz0] * [qx; qy; qz]
 *
 * @param px0 X-coordinate of vertex 0.
 * @param px1 X-coordinate of vertex 1.
 * @param px2 X-coordinate of vertex 2.
 * @param px3 X-coordinate of vertex 3.
 * @param py0 Y-coordinate of vertex 0.
 * @param py1 Y-coordinate of vertex 1.
 * @param py2 Y-coordinate of vertex 2.
 * @param py3 Y-coordinate of vertex 3.
 * @param pz0 Z-coordinate of vertex 0.
 * @param pz1 Z-coordinate of vertex 1.
 * @param pz2 Z-coordinate of vertex 2.
 * @param pz3 Z-coordinate of vertex 3.
 * @param qx  Quadrature point coordinate along x in the reference tetrahedron.
 * @param qy  Quadrature point coordinate along y.
 * @param qz  Quadrature point coordinate along z.
 * @param out_x Pointer to store the resulting x-coordinate.
 * @param out_y Pointer to store the resulting y-coordinate.
 * @param out_z Pointer to store the resulting z-coordinate.
 */
SFEM_INLINE static void                                    //
tet4_transform_v2(const real_type                px0,      // X-coordinate
                  const real_type                px1,      //
                  const real_type                px2,      //
                  const real_type                px3,      //
                  const real_type                py0,      // Y-coordinate
                  const real_type                py1,      //
                  const real_type                py2,      //
                  const real_type                py3,      //
                  const real_type                pz0,      // Z-coordinate
                  const real_type                pz1,      //
                  const real_type                pz2,      //
                  const real_type                pz3,      //
                  const real_type                qx,       // Quadrature point
                  const real_type                qy,       //
                  const real_type                qz,       //
                  real_type* const SFEM_RESTRICT out_x,    // Output
                  real_type* const SFEM_RESTRICT out_y,    //
                  real_type* const SFEM_RESTRICT out_z) {  //
    //
    //

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

    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}

/**
 * @brief Compute the scaled volume measure of a tetrahedral element.
 *
 * This function computes the volume of a tetrahedron using the determinant of the
 * 4x4 matrix defined by the tetrahedron's vertex coordinates augmented with a unit column.
 * In particular, the volume is obtained by
 *
 *   V = (1/6) * det(M)
 *
 * where M is defined by:
 *
 *   M = [ px0, py0, pz0, 1 ]
 *       [ px1, py1, pz1, 1 ]
 *       [ px2, py2, pz2, 1 ]
 *       [ px3, py3, pz3, 1 ]
 *
 * Instead of computing the full 4x4 determinant directly, the function uses a set of
 * intermediate expressions (x0 to x8) that encapsulate the required differences and
 * scaling factors. This approach avoids forming the full 4x4 matrix and directly yields
 * the scaled determinant equivalent to the tetrahedron's volume.
 *
 * @param[in] px0 X-coordinate of vertex 0.
 * @param[in] px1 X-coordinate of vertex 1.
 * @param[in] px2 X-coordinate of vertex 2.
 * @param[in] px3 X-coordinate of vertex 3.
 * @param[in] py0 Y-coordinate of vertex 0.
 * @param[in] py1 Y-coordinate of vertex 1.
 * @param[in] py2 Y-coordinate of vertex 2.
 * @param[in] py3 Y-coordinate of vertex 3.
 * @param[in] pz0 Z-coordinate of vertex 0.
 * @param[in] pz1 Z-coordinate of vertex 1.
 * @param[in] pz2 Z-coordinate of vertex 2.
 * @param[in] pz3 Z-coordinate of vertex 3.
 *
 * @return The volume measure of the tetrahedron (V = (1/6) * det(M)).
 */
SFEM_INLINE static real_type  //
tet4_measure_v2(
        // X-coordinates
        const real_type px0, const real_type px1, const real_type px2, const real_type px3,
        // Y-coordinates
        const real_type py0, const real_type py1, const real_type py2, const real_type py3,
        // Z-coordinates
        const real_type pz0, const real_type pz1, const real_type pz2, const real_type pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_type x0 = -pz0 + pz3;
    const real_type x1 = -py0 + py2;
    const real_type x2 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px1;
    const real_type x3 = -py0 + py3;
    const real_type x4 = -pz0 + pz2;
    const real_type x5 = -py0 + py1;
    const real_type x6 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px2;
    const real_type x7 = -pz0 + pz1;
    const real_type x8 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

/**
 * @brief Compute the barycenter (centroid) of a tetrahedral element.
 *
 * This function calculates the coordinates of the barycenter of a tetrahedron by taking
 * the arithmetic mean of its four vertices' coordinates. The barycenter is the point where
 * the four medians of the tetrahedron intersect, and represents the center of mass assuming
 * uniform density.
 *
 * @param[in] px0 X-coordinate of vertex 0.
 * @param[in] px1 X-coordinate of vertex 1.
 * @param[in] px2 X-coordinate of vertex 2.
 * @param[in] px3 X-coordinate of vertex 3.
 * @param[in] py0 Y-coordinate of vertex 0.
 * @param[in] py1 Y-coordinate of vertex 1.
 * @param[in] py2 Y-coordinate of vertex 2.
 * @param[in] py3 Y-coordinate of vertex 3.
 * @param[in] pz0 Z-coordinate of vertex 0.
 * @param[in] pz1 Z-coordinate of vertex 1.
 * @param[in] pz2 Z-coordinate of vertex 2.
 * @param[in] pz3 Z-coordinate of vertex 3.
 * @param[out] bx Pointer to store the x-coordinate of the barycenter.
 * @param[out] by Pointer to store the y-coordinate of the barycenter.
 * @param[out] bz Pointer to store the z-coordinate of the barycenter.
 */
SFEM_INLINE static void                                  //
tet4_barycenter_v2(const real_type                px0,   // X-coordinate
                   const real_type                px1,   //
                   const real_type                px2,   //
                   const real_type                px3,   //
                   const real_type                py0,   // Y-coordinate
                   const real_type                py1,   //
                   const real_type                py2,   //
                   const real_type                py3,   //
                   const real_type                pz0,   // Z-coordinate
                   const real_type                pz1,   //
                   const real_type                pz2,   //
                   const real_type                pz3,   //
                   real_type* const SFEM_RESTRICT bx,    // Output
                   real_type* const SFEM_RESTRICT by,    //
                   real_type* const SFEM_RESTRICT bz) {  //
                                                         //
    *bx = (px0 + px1 + px2 + px3) / 4.0;
    *by = (py0 + py1 + py2 + py3) / 4.0;
    *bz = (pz0 + pz1 + pz2 + pz3) / 4.0;
}

/**
 * @brief Evaluate the shape functions for an 8-node hexahedral element.
 *
 * Computes the value of the eight standard basis (hat) functions associated
 * with a hexahedral (cubic) element evaluated at the local coordinate (x, y, z). This
 * is typically used for interpolation in grid-based methods.
 *
 * @param[in] x Local x-coordinate in the unit cube.
 * @param[in] y Local y-coordinate in the unit cube.
 * @param[in] z Local z-coordinate in the unit cube.
 * @param[out] f0 Pointer to the value of the shape function associated with node 0.
 * @param[out] f1 Pointer to the value of the shape function associated with node 1.
 * @param[out] f2 Pointer to the value of the shape function associated with node 2.
 * @param[out] f3 Pointer to the value of the shape function associated with node 3.
 * @param[out] f4 Pointer to the value of the shape function associated with node 4.
 * @param[out] f5 Pointer to the value of the shape function associated with node 5.
 * @param[out] f6 Pointer to the value of the shape function associated with node 6.
 * @param[out] f7 Pointer to the value of the shape function associated with node 7.
 */
SFEM_INLINE static void                                //
hex_aa_8_eval_fun_V(const real_t                x,     // Local coordinates (in the unit cube)
                    const real_t                y,     //
                    const real_t                z,     //
                    real_t* const SFEM_RESTRICT f0,    // Output
                    real_t* const SFEM_RESTRICT f1,    //
                    real_t* const SFEM_RESTRICT f2,    //
                    real_t* const SFEM_RESTRICT f3,    //
                    real_t* const SFEM_RESTRICT f4,    //
                    real_t* const SFEM_RESTRICT f5,    //
                    real_t* const SFEM_RESTRICT f6,    //
                    real_t* const SFEM_RESTRICT f7) {  //
    // Quadrature point (local coordinates)
    // With respect to the hat functions of a cube element
    // In a local coordinate system
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

/**
 * @brief Collect coefficients for an 8-node hexahedral element.
 *
 * This function extracts eight coefficient values from a flattened input array
 * using the given multidimensional indices and stride information. The coefficients
 * represent the geometric data transformed into the solver's data layout. The
 * extraction is performed by computing a unique index for each node from the grid
 * position defined by (i, j, k) and the given stride, then reading the corresponding
 * value from the input array.
 *
 * The indices for the eight nodes are computed as follows:
 *  - i0 = i * stride[0] + j * stride[1] + k * stride[2]
 *  - i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2]
 *  - i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2]
 *  - i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2]
 *  - i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2]
 *  - i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2]
 *  - i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2]
 *  - i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2]
 *
 * @param[in]  stride  Pointer to an array of stride values for each dimension.
 * @param[in]  i       The index along the first dimension of the element.
 * @param[in]  j       The index along the second dimension of the element.
 * @param[in]  k       The index along the third dimension of the element.
 * @param[in]  data    Pointer to the flattened input data array.
 * @param[out] out0    Pointer to store the coefficient at node 0.
 * @param[out] out1    Pointer to store the coefficient at node 1.
 * @param[out] out2    Pointer to store the coefficient at node 2.
 * @param[out] out3    Pointer to store the coefficient at node 3.
 * @param[out] out4    Pointer to store the coefficient at node 4.
 * @param[out] out5    Pointer to store the coefficient at node 5.
 * @param[out] out6    Pointer to store the coefficient at node 6.
 * @param[out] out7    Pointer to store the coefficient at node 7.
 */
void                                                                    //
hex_aa_8_collect_coeffs_V(const ptrdiff_t* const SFEM_RESTRICT stride,  // Stride
                          const ptrdiff_t                      i,       // Indices of the element
                          const ptrdiff_t                      j,       //
                          const ptrdiff_t                      k,       //
                          const real_t* const SFEM_RESTRICT    data,    // Input
                          real_t* const SFEM_RESTRICT          out0,    // Output
                          real_t* const SFEM_RESTRICT          out1,    //
                          real_t* const SFEM_RESTRICT          out2,    //
                          real_t* const SFEM_RESTRICT          out3,    //
                          real_t* const SFEM_RESTRICT          out4,    //
                          real_t* const SFEM_RESTRICT          out5,    //
                          real_t* const SFEM_RESTRICT          out6,    //
                          real_t* const SFEM_RESTRICT          out7) {           //
    // Attention this is geometric data transformed to solver data!
    //
    const ptrdiff_t i0 = i * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];

    *out0 = data[i0];
    *out1 = data[i1];
    *out2 = data[i2];
    *out3 = data[i3];
    *out4 = data[i4];
    *out5 = data[i5];
    *out6 = data[i6];
    *out7 = data[i7];
}

/**
 * @brief Compute vertex indices for a hexahedral element in a structured grid.
 *
 * This function calculates the global indices for the 8 vertices of a hexahedral
 * element located at position (i,j,k) in a structured 3D grid. The indices are
 * computed using the provided stride values which define the spacing between
 * consecutive elements in each dimension.
 *
 * The vertices are numbered according to the following convention:
 * - i0: (i,j,k)         - Bottom face, lower left
 * - i1: (i+1,j,k)       - Bottom face, lower right
 * - i2: (i+1,j+1,k)     - Bottom face, upper right
 * - i3: (i,j+1,k)       - Bottom face, upper left
 * - i4: (i,j,k+1)       - Top face, lower left
 * - i5: (i+1,j,k+1)     - Top face, lower right
 * - i6: (i+1,j+1,k+1)   - Top face, upper right
 * - i7: (i,j+1,k+1)     - Top face, upper left
 *
 * @param[in]  stride Array of stride values for each dimension [x,y,z]
 * @param[in]  i      Index in the first dimension (x)
 * @param[in]  j      Index in the second dimension (y)
 * @param[in]  k      Index in the third dimension (z)
 * @param[out] i0     Pointer to store the index of vertex 0
 * @param[out] i1     Pointer to store the index of vertex 1
 * @param[out] i2     Pointer to store the index of vertex 2
 * @param[out] i3     Pointer to store the index of vertex 3
 * @param[out] i4     Pointer to store the index of vertex 4
 * @param[out] i5     Pointer to store the index of vertex 5
 * @param[out] i6     Pointer to store the index of vertex 6
 * @param[out] i7     Pointer to store the index of vertex 7
 */
void                                                                            //
hex_aa_8_collect_coeffs_indices_V(const ptrdiff_t* const SFEM_RESTRICT stride,  // Stride
                                  const ptrdiff_t                      i,       // Indices of the element
                                  const ptrdiff_t                      j,       //
                                  const ptrdiff_t                      k,       //
                                  ptrdiff_t* const SFEM_RESTRICT       i0,      //
                                  ptrdiff_t* const SFEM_RESTRICT       i1,      //
                                  ptrdiff_t* const SFEM_RESTRICT       i2,      //
                                  ptrdiff_t* const SFEM_RESTRICT       i3,      //
                                  ptrdiff_t* const SFEM_RESTRICT       i4,      //
                                  ptrdiff_t* const SFEM_RESTRICT       i5,      //
                                  ptrdiff_t* const SFEM_RESTRICT       i6,      //
                                  ptrdiff_t* const SFEM_RESTRICT       i7) {          //

    *i0 = i * stride[0] + j * stride[1] + k * stride[2];
    *i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2];
    *i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2];
    *i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2];
    *i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2];
    *i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2];
    *i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
    *i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                               //
tet4_resample_field_local_v2(const ptrdiff_t                      start_element,  // Mesh
                             const ptrdiff_t                      end_element,    //
                             const ptrdiff_t                      nnodes,         //
                             const idx_t** const SFEM_RESTRICT    elems,          //
                             const geom_t** const SFEM_RESTRICT   xyz,            //
                             const ptrdiff_t* const SFEM_RESTRICT n,              // SDF
                             const ptrdiff_t* const SFEM_RESTRICT stride,         //
                             const geom_t* const SFEM_RESTRICT    origin,         //
                             const geom_t* const SFEM_RESTRICT    delta,          //
                             const real_type* const SFEM_RESTRICT data,           //
                             real_type* const SFEM_RESTRICT       weighted_field) {     // Output
//
#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local  v2 [%s] \n", __FILE__);
    printf("============================================================\n");
#endif
    //
    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // real_type x[4], y[4], z[4];
        // Vertices coordinates of the tetrahedron
        real_type x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
        real_type y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
        real_type z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

        // real_type hex8_f[8];
        real_type hex8_f0 = 0.0, hex8_f1 = 0.0, hex8_f2 = 0.0, hex8_f3 = 0.0, hex8_f4 = 0.0, hex8_f5 = 0.0, hex8_f6 = 0.0,
                  hex8_f7 = 0.0;

        // real_type coeffs[8];
        real_type coeffs0 = 0.0, coeffs1 = 0.0, coeffs2 = 0.0, coeffs3 = 0.0, coeffs4 = 0.0, coeffs5 = 0.0, coeffs6 = 0.0,
                  coeffs7 = 0.0;

        // real_type tet4_f[4];
        real_type tet4_f0 = 0.0, tet4_f1 = 0.0, tet4_f2 = 0.0, tet4_f3 = 0.0;

        // real_type element_field[4];
        real_type element_field0 = 0.0, element_field1 = 0.0, element_field2 = 0.0, element_field3 = 0.0;

        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // copy the coordinates of the vertices
        // for (int v = 0; v < 4; ++v) {
        //     x[v] = xyz[0][ev[v]];  // x-coordinates
        //     y[v] = xyz[1][ev[v]];  // y-coordinates
        //     z[v] = xyz[2][ev[v]];  // z-coordinates
        // }
        {
            x0 = xyz[0][ev[0]];
            x1 = xyz[0][ev[1]];
            x2 = xyz[0][ev[2]];
            x3 = xyz[0][ev[3]];

            y0 = xyz[1][ev[0]];
            y1 = xyz[1][ev[1]];
            y2 = xyz[1][ev[2]];
            y3 = xyz[1][ev[3]];

            z0 = xyz[2][ev[0]];
            z1 = xyz[2][ev[1]];
            z2 = xyz[2][ev[2]];
            z3 = xyz[2][ev[3]];
        }

        // Volume of the tetrahedron
        const real_type theta_volume = tet4_measure_v2(x0,
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

        /////////////////////////////////////////////
        // loop over the quadrature points
        for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

            real_type g_qx, g_qy, g_qz;

            // Transform quadrature point to physical space
            // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical space
            // of the tetrahedral element
            tet4_transform_v2(x0,               // x-coordinates of the vertices
                              x1,               //
                              x2,               //
                              x3,               //
                              y0,               // y-coordinates of the vertices
                              y1,               //
                              y2,               //
                              y3,               //
                              z0,               // z-coordinates of the vertices
                              z1,               //
                              z2,               //
                              z3,               //
                              tet4_qx[quad_i],  // Quadrature point
                              tet4_qy[quad_i],  //
                              tet4_qz[quad_i],  //
                              &g_qx,            // Output coordinates
                              &g_qy,            //
                              &g_qz);           //

#ifndef SFEM_RESAMPLE_GAP_DUAL
            // Standard basis function
            {
                tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
                tet4_f[1] = tet4_qx[q];
                tet4_f[2] = tet4_qy[q];
                tet4_f[2] = tet4_qz[q];
            }
#else

            {
                // DUAL basis function (Shape functions for tetrahedral elements)
                // at the quadrature point
                const real_type f0 = 1.0 - tet4_qx[quad_i] - tet4_qy[quad_i] - tet4_qz[quad_i];
                const real_type f1 = tet4_qx[quad_i];
                const real_type f2 = tet4_qy[quad_i];
                const real_type f3 = tet4_qz[quad_i];

                // Values of the shape functions at the quadrature point
                // In the local coordinate system of the tetrahedral element
                // For each vertex of the tetrahedral element
                tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
                tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
                tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
                tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
            }
#endif

            const real_type grid_x = (g_qx - ox) / dx;
            const real_type grid_y = (g_qy - oy) / dy;
            const real_type grid_z = (g_qz - oz) / dz;

            const ptrdiff_t i = floor(grid_x);
            const ptrdiff_t j = floor(grid_y);
            const ptrdiff_t k = floor(grid_z);

            // printf("i = %ld grid_x = %g\n", i, grid_x);
            // printf("j = %ld grid_y = %g\n", j, grid_y);
            // printf("k = %ld grid_z = %g\n", k, grid_z);

            // If outside the domain of the grid (i.e., the grid is not large enough)
            if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                fprintf(stderr,
                        "WARNING: (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                        "%ld)!\n",
                        g_qx,
                        g_qy,
                        g_qz,
                        i,
                        j,
                        k,
                        n[0],
                        n[1],
                        n[2]);
                exit(1);
            }

            // Get the reminder [0, 1]
            // The local coordinates of the quadrature point in the unit cube
            real_type l_x = (grid_x - (double)i);
            real_type l_y = (grid_y - (double)j);
            real_type l_z = (grid_z - (double)k);

            assert(l_x >= -1e-8);
            assert(l_y >= -1e-8);
            assert(l_z >= -1e-8);

            assert(l_x <= 1 + 1e-8);
            assert(l_y <= 1 + 1e-8);
            assert(l_z <= 1 + 1e-8);

            // Critical point
            // Compute the shape functions of the hexahedral (cubic) element
            // at the quadrature point
            hex_aa_8_eval_fun_V(l_x,        // Local coordinates
                                l_y,        //
                                l_z,        //
                                &hex8_f0,   // Output shape functions
                                &hex8_f1,   //
                                &hex8_f2,   //
                                &hex8_f3,   //
                                &hex8_f4,   //
                                &hex8_f5,   //
                                &hex8_f6,   //
                                &hex8_f7);  //

            // Collect coefficients for the hexahedral element
            // The data at the vertices of the hexahedral element (in the structured grid)
            // are stored in the variables coeffs0, ..., coeffs7
            hex_aa_8_collect_coeffs_V(stride,     // Stride
                                      i,          // Indices of the element (in the grid)
                                      j,          //
                                      k,          //
                                      data,       // Input
                                      &coeffs0,   // Output
                                      &coeffs1,   //
                                      &coeffs2,   //
                                      &coeffs3,   //
                                      &coeffs4,   //
                                      &coeffs5,   //
                                      &coeffs6,   //
                                      &coeffs7);  //

            // Integrate gap function
            {
                real_type eval_field = 0.0;

                // Value of the field at the quadrature point
                // Is a linear combination of the coefficients and the shape functions
                // of the hexahedral element
                eval_field += hex8_f0 * coeffs0;
                eval_field += hex8_f1 * coeffs1;
                eval_field += hex8_f2 * coeffs2;
                eval_field += hex8_f3 * coeffs3;
                eval_field += hex8_f4 * coeffs4;
                eval_field += hex8_f5 * coeffs5;
                eval_field += hex8_f6 * coeffs6;
                eval_field += hex8_f7 * coeffs7;

                // UNROLL_ZERO
                // for (int edof_i = 0; edof_i < 4; edof_i++) {
                //     element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
                // }  // end edof_i loop

                // Update the field at the vertices of the tetrahedral element
                // with the contribution of the quadrature point
                const real_type dV = theta_volume * tet4_qw[quad_i];

                //
                element_field0 += eval_field * tet4_f0 * dV;
                element_field1 += eval_field * tet4_f1 * dV;
                element_field2 += eval_field * tet4_f2 * dV;
                element_field3 += eval_field * tet4_f3 * dV;

            }  // end integrate gap function

        }  // end for quad_i over quadrature points

        // for (int v = 0; v < 4; ++v) {
        //     // Invert sign since distance field is negative insdide and positive outside

        //      weighted_field[ev[v]] += element_field[v];
        // }  // end vertex loop

        // Update the field at the vertices of the tetrahedral element
        // with the contribution of the quadrature points of the element (sum over quadrature points)
        weighted_field[ev[0]] += element_field0;
        weighted_field[ev[1]] += element_field1;
        weighted_field[ev[2]] += element_field2;
        weighted_field[ev[3]] += element_field3;

    }  // end for i over elements

    return 0;
}  // end tet4_resample_field_local_v2

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_adjoint /////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                         //
tet4_resample_tetrahedron_local_adjoint(const real_type                      x0,            // Tetrahedron vertices X-coordinates
                                        const real_type                      x1,            //
                                        const real_type                      x2,            //
                                        const real_type                      x3,            //
                                        const real_type                      y0,            // Tetrahedron vertices Y-coordinates
                                        const real_type                      y1,            //
                                        const real_type                      y2,            //
                                        const real_type                      y3,            //
                                        const real_type                      z0,            // Tetrahedron vertices Z-coordinates
                                        const real_type                      z1,            //
                                        const real_type                      z2,            //
                                        const real_type                      z3,            //
                                        const real_type                      theta_volume,  // Volume of the tetrahedron
                                        const real_type                      wf0,           // Weighted field at the vertices
                                        const real_type                      wf1,           //
                                        const real_type                      wf2,           //
                                        const real_type                      wf3,           //
                                        const real_type                      ox,            // Origin of the grid
                                        const real_type                      oy,            //
                                        const real_type                      oz,            //
                                        const real_type                      dx,            // Spacing of the grid
                                        const real_type                      dy,            //
                                        const real_type                      dz,            //
                                        const ptrdiff_t* const SFEM_RESTRICT stride,        // Stride
                                        const ptrdiff_t* const SFEM_RESTRICT n,             // Size of the grid
                                        real_type* const SFEM_RESTRICT       data) {              // Output

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

        real_type g_qx, g_qy, g_qz;

        // Transform quadrature point to physical space
        // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical space
        // of the tetrahedral element
        tet4_transform_v2(x0,               // x-coordinates of the vertices
                          x1,               //
                          x2,               //
                          x3,               //
                          y0,               // y-coordinates of the vertices
                          y1,               //
                          y2,               //
                          y3,               //
                          z0,               // z-coordinates of the vertices
                          z1,               //
                          z2,               //
                          z3,               //
                          tet4_qx[quad_i],  // Quadrature point
                          tet4_qy[quad_i],  //
                          tet4_qz[quad_i],  //
                          &g_qx,            // Output coordinates
                          &g_qy,            //
                          &g_qz);           //

#ifndef SFEM_RESAMPLE_GAP_DUAL
        // Standard basis function
        {
            tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
            tet4_f[1] = tet4_qx[q];
            tet4_f[2] = tet4_qy[q];
            tet4_f[2] = tet4_qz[q];
        }
#else

        real_type tet4_f0, tet4_f1, tet4_f2, tet4_f3;
        {
            // DUAL basis function (Shape functions for tetrahedral elements)
            // at the quadrature point
            const real_type f0 = 1.0 - tet4_qx[quad_i] - tet4_qy[quad_i] - tet4_qz[quad_i];
            const real_type f1 = tet4_qx[quad_i];
            const real_type f2 = tet4_qy[quad_i];
            const real_type f3 = tet4_qz[quad_i];

            // Values of the shape functions at the quadrature point
            // In the local coordinate system of the tetrahedral element
            // For each vertex of the tetrahedral element
            tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
            tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
            tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
            tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
        }
#endif

        const real_type grid_x = (g_qx - ox) / dx;
        const real_type grid_y = (g_qy - oy) / dy;
        const real_type grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

        // printf("i = %ld grid_x = %g\n", i, grid_x);
        // printf("j = %ld grid_y = %g\n", j, grid_y);
        // printf("k = %ld grid_z = %g\n", k, grid_z);

        // If outside the domain of the grid (i.e., the grid is not large enough)
        if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
            fprintf(stderr,
                    "WARNING: (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                    "%ld)!\n",
                    g_qx,
                    g_qy,
                    g_qz,
                    i,
                    j,
                    k,
                    n[0],
                    n[1],
                    n[2]);
            exit(1);
        }

        // Get the reminder [0, 1]
        // The local coordinates of the quadrature point in the unit cube
        real_type l_x = (grid_x - (double)i);
        real_type l_y = (grid_y - (double)j);
        real_type l_z = (grid_z - (double)k);

        assert(l_x >= -1e-8);
        assert(l_y >= -1e-8);
        assert(l_z >= -1e-8);

        assert(l_x <= 1 + 1e-8);
        assert(l_y <= 1 + 1e-8);
        assert(l_z <= 1 + 1e-8);

        // Critical point
        // Compute the shape functions of the hexahedral (cubic) element
        // at the quadrature point

        real_type hex8_f0, hex8_f1, hex8_f2, hex8_f3, hex8_f4, hex8_f5, hex8_f6, hex8_f7;

        hex_aa_8_eval_fun_V(l_x,        // Local coordinates
                            l_y,        //
                            l_z,        //
                            &hex8_f0,   // Output shape functions
                            &hex8_f1,   //
                            &hex8_f2,   //
                            &hex8_f3,   //
                            &hex8_f4,   //
                            &hex8_f5,   //
                            &hex8_f6,   //
                            &hex8_f7);  //

        // Indices of the vertices of the hexahedral element
        ptrdiff_t i0, i1, i2, i3, i4, i5, i6, i7;
        hex_aa_8_collect_coeffs_indices_V(stride,  // Stride
                                          i,       // Indices of the element
                                          j,       //
                                          k,       //
                                          &i0,     // Output indices
                                          &i1,     //
                                          &i2,     //
                                          &i3,     //
                                          &i4,     //
                                          &i5,     //
                                          &i6,     //
                                          &i7);    //

        // Integrate the values of the field at the vertices of the tetrahedral element
        const real_type dV = theta_volume * tet4_qw[quad_i];
        const real_type It = (tet4_f0 * wf0 + tet4_f1 * wf1 + tet4_f2 * wf2 + tet4_f3 * wf3) * dV;

        const real_type d0 = It * hex8_f0;
        const real_type d1 = It * hex8_f1;
        const real_type d2 = It * hex8_f2;
        const real_type d3 = It * hex8_f3;
        const real_type d4 = It * hex8_f4;
        const real_type d5 = It * hex8_f5;
        const real_type d6 = It * hex8_f6;
        const real_type d7 = It * hex8_f7;

        // Update the data
        data[i0] += d0;
        data[i1] += d1;
        data[i2] += d2;
        data[i3] += d3;
        data[i4] += d4;
        data[i5] += d5;
        data[i6] += d6;
        data[i7] += d7;
    }

    return 0;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_adjoint /////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                     //
tet4_resample_field_local_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                  const ptrdiff_t                      end_element,     //
                                  const ptrdiff_t                      nnodes,          //
                                  const idx_t** const SFEM_RESTRICT    elems,           //
                                  const geom_t** const SFEM_RESTRICT   xyz,             //
                                  const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                  const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                  const geom_t* const SFEM_RESTRICT    origin,          //
                                  const geom_t* const SFEM_RESTRICT    delta,           //
                                  const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                                  real_t* const SFEM_RESTRICT          data) {                   // Output
                                                                                        //
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    const real_type hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local_adjoint  v2: %s:%d \n", __FILE__, __LINE__);
    printf("Heaxahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    // shape functions of the hexahedral element
    real_type hex8_f0 = 0.0, hex8_f1 = 0.0, hex8_f2 = 0.0, hex8_f3 = 0.0,  //
            hex8_f4 = 0.0, hex8_f5 = 0.0, hex8_f6 = 0.0, hex8_f7 = 0.0;

    ptrdiff_t i0, i1, i2, i3, i4, i5, i6, i7;

    // real_type tet4_f[4];
    real_type tet4_f0 = 0.0, tet4_f1 = 0.0, tet4_f2 = 0.0, tet4_f3 = 0.0;

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // Vertices coordinates of the tetrahedron

        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // Read the coordinates of the vertices of the tetrahedron
        const real_type x0 = xyz[0][ev[0]];
        const real_type x1 = xyz[0][ev[1]];
        const real_type x2 = xyz[0][ev[2]];
        const real_type x3 = xyz[0][ev[3]];

        const real_type y0 = xyz[1][ev[0]];
        const real_type y1 = xyz[1][ev[1]];
        const real_type y2 = xyz[1][ev[2]];
        const real_type y3 = xyz[1][ev[3]];

        const real_type z0 = xyz[2][ev[0]];
        const real_type z1 = xyz[2][ev[1]];
        const real_type z2 = xyz[2][ev[2]];
        const real_type z3 = xyz[2][ev[3]];

        // Volume of the tetrahedron
        const real_type theta_volume = tet4_measure_v2(x0,
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

        const real_type wf0 = weighted_field[ev[0]];
        const real_type wf1 = weighted_field[ev[1]];
        const real_type wf2 = weighted_field[ev[2]];
        const real_type wf3 = weighted_field[ev[3]];

        const real_type sampled_volume = hexahedron_volume * (double)(TET_QUAD_NQP);

        tet4_resample_tetrahedron_local_adjoint(x0,            // Tetrahedron vertices X-coordinates
                                                x1,            //
                                                x2,            //
                                                x3,            //
                                                y0,            // Tetrahedron vertices Y-coordinates
                                                y1,            //
                                                y2,            //
                                                y3,            //
                                                z0,            // Tetrahedron vertices Z-coordinates
                                                z1,            //
                                                z2,            //
                                                z3,            //
                                                theta_volume,  // Volume of the tetrahedron
                                                wf0,           // Weighted field at the vertices
                                                wf1,           //
                                                wf2,           //
                                                wf3,           //
                                                ox,            // Origin of the grid
                                                oy,            //
                                                oz,            //
                                                dx,            // Spacing of the grid
                                                dy,            //
                                                dz,            //
                                                stride,        // Stride
                                                n,             // Size of the grid
                                                data);         // Output

        // if (sampled_volume < 8.0 * theta_volume) {
        //     fprintf(stderr, "WARNING: sampled_volume < 8 * theta_volume: %g < %g\n", sampled_volume, 8.0 *
        // theta_volume);
        // }

    }  // end for i over elements

    return ret;
}  // end tet4_resample_field_local_adjoint
