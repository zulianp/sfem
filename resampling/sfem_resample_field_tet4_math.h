#ifndef __SFEM_RESAMPLE_FIELD_TET4_MATH_H__
#define __SFEM_RESAMPLE_FIELD_TET4_MATH_H__

#include "sfem_config.h"

#define real_type real_t

#ifdef __cplusplus
extern "C" {
#endif

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
void                                                      //
tet4_transform_v2(const real_type                px0,     // X-coordinate
                  const real_type                px1,     //
                  const real_type                px2,     //
                  const real_type                px3,     //
                  const real_type                py0,     // Y-coordinate
                  const real_type                py1,     //
                  const real_type                py2,     //
                  const real_type                py3,     //
                  const real_type                pz0,     // Z-coordinate
                  const real_type                pz1,     //
                  const real_type                pz2,     //
                  const real_type                pz3,     //
                  const real_type                qx,      // Quadrature point
                  const real_type                qy,      //
                  const real_type                qz,      //
                  real_type* const SFEM_RESTRICT out_x,   // Output
                  real_type* const SFEM_RESTRICT out_y,   //
                  real_type* const SFEM_RESTRICT out_z);  //

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
void                                                  //
hex_aa_8_eval_fun_V(const real_t                x,    // Local coordinates (in the unit cube)
                    const real_t                y,    //
                    const real_t                z,    //
                    real_t* const SFEM_RESTRICT f0,   // Output
                    real_t* const SFEM_RESTRICT f1,   //
                    real_t* const SFEM_RESTRICT f2,   //
                    real_t* const SFEM_RESTRICT f3,   //
                    real_t* const SFEM_RESTRICT f4,   //
                    real_t* const SFEM_RESTRICT f5,   //
                    real_t* const SFEM_RESTRICT f6,   //
                    real_t* const SFEM_RESTRICT f7);  //

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
                                  ptrdiff_t* const SFEM_RESTRICT       i7);           //

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
                          real_t* const SFEM_RESTRICT          out7);            //

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
real_type                              ////
tet4_measure_v2(const real_type px0,   // X-coordinate 1st vertex
                const real_type px1,   //              2nd vertex
                const real_type px2,   //              3rd vertex
                const real_type px3,   //              4th vertex
                const real_type py0,   // Y-coordinate 1st vertex
                const real_type py1,   //              2nd vertex
                const real_type py2,   //              3rd vertex
                const real_type py3,   //              4th vertex
                const real_type pz0,   // Z-coordinate 1st vertex
                const real_type pz1,   //              2nd vertex
                const real_type pz2,   //              3rd vertex
                const real_type pz3);  //              4th vertex

/**
 * @brief Check if a point lies inside a tetrahedron or on its boundary.
 *
 * This function determines whether a point is inside a tetrahedron or on its boundary
 * by checking if the point is on the same side of all four faces as the opposite vertex.
 * The algorithm works by examining the point's position relative to each of the four
 * triangular faces that make up the tetrahedron.
 *
 * @param[in] px X-coordinate of the point to test.
 * @param[in] py Y-coordinate of the point to test.
 * @param[in] pz Z-coordinate of the point to test.
 * @param[in] v1x X-coordinate of the first tetrahedron vertex.
 * @param[in] v1y Y-coordinate of the first tetrahedron vertex.
 * @param[in] v1z Z-coordinate of the first tetrahedron vertex.
 * @param[in] v2x X-coordinate of the second tetrahedron vertex.
 * @param[in] v2y Y-coordinate of the second tetrahedron vertex.
 * @param[in] v2z Z-coordinate of the second tetrahedron vertex.
 * @param[in] v3x X-coordinate of the third tetrahedron vertex.
 * @param[in] v3y Y-coordinate of the third tetrahedron vertex.
 * @param[in] v3z Z-coordinate of the third tetrahedron vertex.
 * @param[in] v4x X-coordinate of the fourth tetrahedron vertex.
 * @param[in] v4y Y-coordinate of the fourth tetrahedron vertex.
 * @param[in] v4z Z-coordinate of the fourth tetrahedron vertex.
 *
 * @return 1 if the point is inside the tetrahedron or on its boundary, 0 otherwise.
 */
int                                                //
is_point_inside_tetrahedron(const real_type px,    //
                            const real_type py,    //
                            const real_type pz,    //
                            const real_type v1x,   //
                            const real_type v1y,   //
                            const real_type v1z,   //
                            const real_type v2x,   //
                            const real_type v2y,   //
                            const real_type v2z,   //
                            const real_type v3x,   //
                            const real_type v3y,   //
                            const real_type v3z,   //
                            const real_type v4x,   //
                            const real_type v4y,   //
                            const real_type v4z);  //

/**
 * @brief Structure representing the 4 vertices of a tetrahedron
 *
 * This structure stores the coordinates of the four vertices that define a tetrahedron.
 * Each vertex is represented by its x, y, and z coordinates.
 */
struct tet_vertices {
    real_type x0, y0, z0;      // Vertex 0 coordinates
    real_type x1, y1, z1;      // Vertex 1 coordinates
    real_type x2, y2, z2;      // Vertex 2 coordinates
    real_type x3, y3, z3;      // Vertex 3 coordinates
    real_type w0, w1, w2, w3;  // Weights for the vertices
};

/**
 * @brief Performs uniform refinement of a tetrahedron into 8 smaller tetrahedra
 *
 * This function subdivides a tetrahedron into 8 smaller tetrahedra by introducing
 * new vertices at the midpoints of the original edges. The refinement is uniform,
 * meaning all resulting tetrahedra have the same volume (1/8 of the original).
 *
 * @param[in] v0x X-coordinate of the first vertex (vertex 0) of the original tetrahedron
 * @param[in] v0y Y-coordinate of the first vertex (vertex 0) of the original tetrahedron
 * @param[in] v0z Z-coordinate of the first vertex (vertex 0) of the original tetrahedron
 * @param[in] v1x X-coordinate of the second vertex (vertex 1) of the original tetrahedron
 * @param[in] v1y Y-coordinate of the second vertex (vertex 1) of the original tetrahedron
 * @param[in] v1z Z-coordinate of the second vertex (vertex 1) of the original tetrahedron
 * @param[in] v2x X-coordinate of the third vertex (vertex 2) of the original tetrahedron
 * @param[in] v2y Y-coordinate of the third vertex (vertex 2) of the original tetrahedron
 * @param[in] v2z Z-coordinate of the third vertex (vertex 2) of the original tetrahedron
 * @param[in] v3x X-coordinate of the fourth vertex (vertex 3) of the original tetrahedron
 * @param[in] v3y Y-coordinate of the fourth vertex (vertex 3) of the original tetrahedron
 * @param[in] v3z Z-coordinate of the fourth vertex (vertex 3) of the original tetrahedron
 * @param[in] w0 Weight associated with vertex 0, used for field interpolation
 * @param[in] w1 Weight associated with vertex 1, used for field interpolation
 * @param[in] w2 Weight associated with vertex 2, used for field interpolation
 * @param[in] w3 Weight associated with vertex 3, used for field interpolation
 * @param[out] rTets Pointer to an array of 8 tet_vertices structures that will
 *                  store the coordinates and interpolated weights for the refined tetrahedra
 *
 * @details
 * The refinement is achieved by computing midpoints of the edges of the original
 * tetrahedron and using these points, along with the original vertices, to define
 * 8 new tetrahedra that completely fill the volume of the original tetrahedron.
 * Additionally, the function interpolates field values (weights) at the new vertices
 * based on the weights at the original vertices.
 *
 * This is a common operation in adaptive mesh refinement procedures for finite element
 * methods. The implementation uses the method from the paper:
 *   Uniform Refinement of a Tetrahedron
 *   Elizabeth G. Ong, January 1991, CAM Report 91-01
 *
 * @return 0 if the operation was successful, 1 otherwise
 */
int                                                        //
tet_uniform_refinement(const real_t               v0x,     //
                       const real_t               v0y,     //
                       const real_t               v0z,     //
                       const real_t               v1x,     //
                       const real_t               v1y,     //
                       const real_t               v1z,     //
                       const real_t               v2x,     //
                       const real_t               v2y,     //
                       const real_t               v2z,     //
                       const real_t               v3x,     //
                       const real_t               v3y,     //
                       const real_t               v3z,     //
                       const real_t               w0,      //
                       const real_t               w1,      //
                       const real_t               w2,      //
                       const real_t               w3,      //
                       struct tet_vertices* const rTets);  // Output tetrahedra

/**
 * @brief Compute the volume of an array of tetrahedra.
 * return the total volume of the tetrahedra in the array.
 * The volume of each tetrahedron is computed using the determinant of the 4x4 matrix
 * Returns the total volume of the tetrahedra in the array.
 *
 * @param[in] tets Array of tetrahedra
 * @param[in] n Number of tetrahedra
 * @param[out] V Pointer to store the total volume
 * @return real_t Total volume of the tetrahedra
 */
real_t                                                   //
volume_tet_array(const struct tet_vertices* const tets,  // Array of tetrahedra
                 const int                        n,     // Number of tetrahedra
                 real_t* const                    V);                       // Output

/**
 * @brief Calculate the lengths of all edges in a tetrahedron
 *
 * This function computes the lengths of all six edges of a tetrahedron defined by four vertices.
 * The edge lengths are calculated as the Euclidean distances between pairs of vertices:
 *
 * - edge_length[0]: Distance between vertices v0 and v1
 * - edge_length[1]: Distance between vertices v0 and v2
 * - edge_length[2]: Distance between vertices v0 and v3
 * - edge_length[3]: Distance between vertices v1 and v2
 * - edge_length[4]: Distance between vertices v1 and v3
 * - edge_length[5]: Distance between vertices v2 and v3
 *
 * @param[in] v0x X-coordinate of the first vertex (vertex 0)
 * @param[in] v0y Y-coordinate of the first vertex (vertex 0)
 * @param[in] v0z Z-coordinate of the first vertex (vertex 0)
 * @param[in] v1x X-coordinate of the second vertex (vertex 1)
 * @param[in] v1y Y-coordinate of the second vertex (vertex 1)
 * @param[in] v1z Z-coordinate of the second vertex (vertex 1)
 * @param[in] v2x X-coordinate of the third vertex (vertex 2)
 * @param[in] v2y Y-coordinate of the third vertex (vertex 2)
 * @param[in] v2z Z-coordinate of the third vertex (vertex 2)
 * @param[in] v3x X-coordinate of the fourth vertex (vertex 3)
 * @param[in] v3y Y-coordinate of the fourth vertex (vertex 3)
 * @param[in] v3z Z-coordinate of the fourth vertex (vertex 3)
 * @param[out] edge_length Pointer to an array of at least 6 elements to store the calculated edge lengths
 *
 * @details
 * This function is useful for mesh quality assessment, refinement criteria, and various
 * finite element computations where the edge lengths of tetrahedra are needed. The array
 * edge_length must be pre-allocated with sufficient space to store all 6 edge lengths.
 *
 * @return 0 if the operation was successful, a non-zero error code otherwise
 */
int                                          //
tet_edge_length(const real_t  v0x,           //
                const real_t  v0y,           //
                const real_t  v0z,           //
                const real_t  v1x,           //
                const real_t  v1y,           //
                const real_t  v1z,           //
                const real_t  v2x,           //
                const real_t  v2y,           //
                const real_t  v2z,           //
                const real_t  v3x,           //
                const real_t  v3y,           //
                const real_t  v3z,           //
                real_t* const edge_length);  //

/**
 * @brief structure to store the analytic field
 */
struct field_analytic {
    unsigned int n;
    real_t*      alpha;
    real_t*      volume;
};

struct field_analytic                //
field_analytic_create(const int n);  //

void  //
field_analytic_destroy(struct field_analytic* field);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __SFEM_RESAMPLE_FIELD_TET4_MATH_H__