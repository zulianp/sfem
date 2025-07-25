#ifndef TET10_RESAMPLE_FIELD_H
#define TET10_RESAMPLE_FIELD_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * sub-parametric assumes that the mesh geometry is 1st order (i.e., tet4 with no warped
 elements)
 * but approximation space (coefficients) is 2nd order.
 */
int hex8_to_subparametric_tet10_resample_field_local(
        // Mesh
        const ptrdiff_t              nelements,  // number of elements
        const ptrdiff_t              nnodes,     // number of nodes
        idx_t** const SFEM_RESTRICT  elems,      // connectivity
        geom_t** const SFEM_RESTRICT xyz,        // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT    origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT    delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT    data,    // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

// int subparametric_tet10_assemble_dual_mass_vector(const ptrdiff_t nelements,
//                                                   const ptrdiff_t nnodes,
//                                                   idx_t** const SFEM_RESTRICT elems,
//                                                   geom_t** const SFEM_RESTRICT xyz,
//                                                   real_t* const values);

int tet10_assemble_dual_mass_vector(const ptrdiff_t nelements, const ptrdiff_t nnodes, idx_t** const SFEM_RESTRICT elems,
                                    geom_t** const SFEM_RESTRICT xyz, real_t* const values);

int hex8_to_tet10_resample_field_local(
        // Mesh
        const ptrdiff_t              nelements,  // number of elements
        const ptrdiff_t              nnodes,     // number of nodes
        idx_t** const SFEM_RESTRICT  elems,      // connectivity
        geom_t** const SFEM_RESTRICT xyz,        // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT    origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT    delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT    data,    // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

int hex8_to_isoparametric_tet10_resample_field_local_cube1(
        /// Mesh
        const ptrdiff_t              nelements,  // number of elements
        const ptrdiff_t              nnodes,     // number of nodes
        idx_t** const SFEM_RESTRICT  elems,      // connectivity
        geom_t** const SFEM_RESTRICT xyz,        // coordinates
        /// SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT    origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT    delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT    data,    // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

real_t hex_aa_8_eval_weno4_3D_Unit(                 //
        const real_t                      x_unit,   //
        const real_t                      y_unit,   //
        const real_t                      z_unit,   //
        const real_t                      ox_unit,  // Coordinates of the origin of the grid in the unitary space
        const real_t                      oy_unit,  // for the structured grid
        const real_t                      oz_unit,  // X, Y and Z
        const ptrdiff_t                   i,        // it must be the absolute index
        const ptrdiff_t                   j,        // Used to get the data
        const ptrdiff_t                   k,        // From the data array
        const ptrdiff_t*                  stride,   //
        const real_t* const SFEM_RESTRICT data);

int hex8_to_tet10_resample_field_local_CUDA_wrapper(  //
        const int mpi_size,                           // MPI size
        const int mpi_rank,                           // MPI rank
                                                      // Mesh
        mesh_t* mesh,                                 // Mesh
        int*    bool_assemble_dual_mass_vector,       // assemble dual mass vector
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host);

int hex8_to_tet10_resample_field_local_CUDA(                          // it is for the host memory
                                                                      // Mesh
        const ptrdiff_t              nelements,                       // number of elements
        const ptrdiff_t              nnodes,                          // number of nodes
        const int                    bool_assemble_dual_mass_vector,  // assemble dual mass vector
        idx_t** const SFEM_RESTRICT  elems,                           // connectivity
        geom_t** const SFEM_RESTRICT xyz,                             // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host);

int hex8_to_tet10_resample_field_local_CUDA_Unified(
        // Mesh
        const ptrdiff_t              nelements,                       // number of elements
        const ptrdiff_t              nnodes,                          // number of nodes
        const int                    bool_assemble_dual_mass_vector,  // assemble dual mass vector
        idx_t** const SFEM_RESTRICT  elems,                           // connectivity
        geom_t** const SFEM_RESTRICT xyz,                             // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host);

int hex8_to_tet10_resample_field_local_CUDA_Managed(
        // Mesh
        const ptrdiff_t              nelements,                       // number of elements
        const ptrdiff_t              nnodes,                          // number of nodes
        const int                    bool_assemble_dual_mass_vector,  // assemble dual mass vector
        idx_t** const SFEM_RESTRICT  elems,                           // connectivity
        geom_t** const SFEM_RESTRICT xyz,                             // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host);

/**
 * @brief Resample a field from a structured grid to a quadratic (10-node) tetrahedral mesh using subparametric mapping.
 *
 * @param start_element
 * @param end_element
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param weighted_field
 * @param data
 * @return int
 */
int                                                                                                      //
hex8_to_isoparametric_tet10_resample_field_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                                   const ptrdiff_t                      end_element,     //
                                                   const ptrdiff_t                      nnodes,          //
                                                   const idx_t** const SFEM_RESTRICT    elems,           //
                                                   const geom_t** const SFEM_RESTRICT   xyz,             //
                                                   const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                   const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                   const geom_t* const SFEM_RESTRICT    origin,          //
                                                   const geom_t* const SFEM_RESTRICT    delta,           //
                                                   const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
                                                   real_t* const SFEM_RESTRICT          data);                    // Output

/**
 * @brief Resamples a field from a 10-node tetrahedral mesh back to a structured hexahedral grid with adaptive refinement.
 *
 * This function performs the adjoint operation of resampling, transferring data from a second-order
 * (quadratic) tetrahedral mesh to a structured hexahedral grid. Unlike the standard adjoint version,
 * this function includes an adaptive refinement strategy controlled by the alpha_th parameter.
 *
 * The alpha threshold is used to determine which elements should be refined based on the ratio of
 * the edges of the tetrahedral mesh compared  to the edge of the hexahedral mesh.
 *
 * @param start_element First element index to process (for parallelization)
 * @param end_element Last element index to process (exclusive)
 * @param nnodes Total number of nodes in the tetrahedral mesh
 * @param elems Element connectivity array for the tetrahedral mesh (10 indices per element)
 * @param xyz Nodal coordinates array for the tetrahedral mesh
 * @param n Number of nodes in each direction of the structured grid
 * @param stride Stride values for navigating the structured grid
 * @param origin Origin coordinates of the structured grid
 * @param delta Grid spacing of the structured grid
 * @param weighted_field Input field values from the tetrahedral mesh (already weighted)
 * @param alpha_th Threshold for refinement - elements with gradients exceeding this value will be refined
 * @param data Output field values resampled to the structured grid
 *
 * @return 0 on success, non-zero on failure
 */
int                                                                                                             //
hex8_to_isoparametric_tet10_resample_field_refine_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                                          const ptrdiff_t                      end_element,     //
                                                          const ptrdiff_t                      nnodes,          //
                                                          const idx_t** const SFEM_RESTRICT    elems,           //
                                                          const geom_t** const SFEM_RESTRICT   xyz,             //
                                                          const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                          const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                          const geom_t* const SFEM_RESTRICT    origin,          //
                                                          const geom_t* const SFEM_RESTRICT    delta,           //
                                                          const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
                                                          const real_t                         alpha_th,  // Threshold for alpha
                                                          real_t* const SFEM_RESTRICT          data);              // Output

/**
 * @brief
 *
 * @param start_element
 * @param end_element
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param weighted_field
 * @param alpha_th
 * @param data
 * @return int
 */
int                                                                                                                    //
hex8_to_isoparametric_tet10_resample_field_iterative_ref_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                                                 const ptrdiff_t                      end_element,     //
                                                                 const ptrdiff_t                      nnodes,          //
                                                                 const idx_t** const SFEM_RESTRICT    elems,           //
                                                                 const geom_t** const SFEM_RESTRICT   xyz,             //
                                                                 const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                                 const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                                 const geom_t* const SFEM_RESTRICT    origin,          //
                                                                 const geom_t* const SFEM_RESTRICT    delta,           //
                                                                 const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
                                                                 const real_t                alpha_th,  // Threshold for alpha
                                                                 real_t* const SFEM_RESTRICT data);     // Output

/**
 * @struct tet10_vertices
 * @brief Holds the coordinates and weight for a 10-node tetrahedral element.
 *
 * This structure provides separate arrays for x, y, z, and w values, each
 * containing data for the 10 nodes of a second-order tetrahedron.
 *
 * @var tet10_vertices::x
 *     The x-coordinates of the 10 nodes.
 * @var tet10_vertices::y
 *     The y-coordinates of the 10 nodes.
 * @var tet10_vertices::z
 *     The z-coordinates of the 10 nodes.
 * @var tet10_vertices::w
 *     Additional weighting or field data for the 10 nodes.
 */
struct tet10_vertices {
    real_t x[10];
    real_t y[10];
    real_t z[10];
    real_t w[10];
};

/**
 * @brief Uniformly refine a tetrahedral 2nd order mesh.
 *
 * Refines the provided 10-node tetrahedron by updating coordinates and weights.
 * @note The ordering of the vertices are the one form User Manual for EXODUS II Mesh Converter.
 * at page 14 Figure 8 TETRA 10-NODED ELEMENT (no central node).
 *
 * @param x     Input array of x-coordinates (size: 10).
 * @param y     Input array of y-coordinates (size: 10).
 * @param z     Input array of z-coordinates (size: 10).
 * @param w     Input array of weights or field data (size: 10).
 * @param rTets Output struct containing refined element data.
 *
 * @return 0 on success, non-zero otherwise.
 */
int                                                            //
tet10_uniform_refinement(const real_t* const          x,       //
                         const real_t* const          y,       //
                         const real_t* const          z,       //
                         const real_t* const          w,       //
                         struct tet10_vertices* const rTets);  //

/**
 * @brief
 *
 * @param x
 * @param y
 * @param z
 * @param w
 * @param vertex_a
 * @param vertex_b
 * @param rTets
 * @return int
 */
int                                                                  //
tet10_refine_two_edge_vertex(const real_t* const          x,         //
                             const real_t* const          y,         //
                             const real_t* const          z,         //
                             const real_t* const          w,         //
                             const int                    vertex_a,  //
                             const int                    vertex_b,  //
                             struct tet10_vertices* const rTets);    //

/**
 * @brief Compute the volumes of the 10-node tetrahedra in the array rTets.
 *
 * @param rTets
 * @param N
 * @param V
 * @param M
 * @return the total volume of the tetrahedra.
 */
real_t                                                   //
tet10_volumes(const struct tet10_vertices* const rTets,  //
              const int                          N,      //
              real_t* const                      V);                          //

/**
 * @brief Calculate the lengths of all edges in a tetrahedron defined by the 10 nodes of a tet10.
 *
 * @param x Array of x-coordinates for the 10 nodes.
 * @param y Array of y-coordinates for the 10 nodes.
 * @param z Array of z-coordinates for the 10 nodes.
 * @param array of edge lengths for the 6 edges.
 * @return real_t Maximum edge length.
 */
real_t                                           //
tet10_edge_lengths(const real_t* x,              //
                   const real_t* y,              //
                   const real_t* z,              //
                   int*          vertex_a,       //
                   int*          vertex_b,       //
                   real_t* const edge_lengths);  //

void  //
tet10_transform(const geom_t* const SFEM_RESTRICT x, const geom_t* const SFEM_RESTRICT y, const geom_t* const SFEM_RESTRICT z,
                // Quadrature point
                const real_t qx, const real_t qy, const real_t qz,
                // Output
                real_t* const SFEM_RESTRICT out_x, real_t* const SFEM_RESTRICT out_y, real_t* const SFEM_RESTRICT out_z);

void                                                         //
tet10_transform_real_t(const real_t* const SFEM_RESTRICT x,  //
                       const real_t* const SFEM_RESTRICT y,  //
                       const real_t* const SFEM_RESTRICT z,  //
                       // Quadrature point
                       const real_t qx,  //
                       const real_t qy,  //
                       const real_t qz,  //
                       // Output
                       real_t* const SFEM_RESTRICT out_x, real_t* const SFEM_RESTRICT out_y, real_t* const SFEM_RESTRICT out_z);

void  //
tet10_dual_basis_hrt(const real_t qx, const real_t qy, const real_t qz, real_t* const f);

void  //
tet10_Lagrange_basis(const real_t qx, const real_t qy, const real_t qz, real_t* const f);

void                                                                   //
hex_aa_8_collect_indices(const ptrdiff_t* const SFEM_RESTRICT stride,  //
                         const ptrdiff_t                      i,       //
                         const ptrdiff_t                      j,       //
                         const ptrdiff_t                      k,       //
                         ptrdiff_t* const SFEM_RESTRICT       indices);

void  //
hex_aa_8_eval_fun(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const real_t x, const real_t y, const real_t z,
        // Output
        real_t* const SFEM_RESTRICT f);

real_t                                               //
tet10_measure(const geom_t* const SFEM_RESTRICT x,   //
              const geom_t* const SFEM_RESTRICT y,   //
              const geom_t* const SFEM_RESTRICT z,   //
              const real_t                      qx,  // Quadrature point
              const real_t                      qy,  //
              const real_t                      qz);

real_t                                                      //
tet10_measure_real_t(const real_t* const SFEM_RESTRICT x,   //
                     const real_t* const SFEM_RESTRICT y,   //
                     const real_t* const SFEM_RESTRICT z,   //
                     const real_t                      qx,  // Quadrature point
                     const real_t                      qy,  //
                     const real_t                      qz);                      //

#ifdef __cplusplus
}
#endif

// TODO (full 2nd order FEM)
// For the structured grid we can decide to go directy high-order splines instead of tri-linear
// hexas int hex{8,?}_to_isoparametric_tet10_resample_field_local(...) int
// isoparametric_tet10_assemble_dual_mass_vector(...)

#endif  // TET10_RESAMPLE_FIELD_H
