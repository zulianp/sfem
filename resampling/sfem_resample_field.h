#ifndef SFEM_RESAMPLE_FIELD_H
#define SFEM_RESAMPLE_FIELD_H

#include <mpi.h>
#include <stddef.h>

#include "bit_array.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

// typedef enum {
//     TET4,
//     TET10
// } element_type;

typedef struct {
    ptrdiff_t        quad_nodes_cnt;   // Number of quadrature points
    ptrdiff_t        nelements;        // Number of elements
    enum ElemType    element_type;     // Element type
    AcceleratorsType use_accelerator;  // Use accelerator such as CUDA
} sfem_resample_field_info;

/**
 * @brief
 *
 * @param element_type
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param wg
 * @param info
 * @return int
 */
int                                                                      //
resample_field_local(const enum ElemType                  element_type,  // Mesh
                     const ptrdiff_t                      nelements,     // Mesh
                     const ptrdiff_t                      nnodes,        // Mesh
                     idx_t** const SFEM_RESTRICT          elems,         // Mesh
                     geom_t** const SFEM_RESTRICT         xyz,           // Mesh
                     const ptrdiff_t* const SFEM_RESTRICT n,             // SDF
                     const ptrdiff_t* const SFEM_RESTRICT stride,        // SDF
                     const geom_t* const SFEM_RESTRICT    origin,        // SDF
                     const geom_t* const SFEM_RESTRICT    delta,         // SDF
                     const real_t* const SFEM_RESTRICT    data,          // SDF
                     real_t* const SFEM_RESTRICT          wg,            //  Output
                     sfem_resample_field_info*            info);                    // Info

/**
 * @brief
 *
 * @param element_type
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param field
 * @param info
 * @return int
 */
int                                                                //
resample_field(const enum ElemType                  element_type,  // // Mesh
               const ptrdiff_t                      nelements,     //
               const ptrdiff_t                      nnodes,        //
               idx_t** const SFEM_RESTRICT          elems,         //
               geom_t** const SFEM_RESTRICT         xyz,           //
               const ptrdiff_t* const SFEM_RESTRICT n,             // // SDF
               const ptrdiff_t* const SFEM_RESTRICT stride,        //
               const geom_t* const SFEM_RESTRICT    origin,        //
               const geom_t* const SFEM_RESTRICT    delta,         //
               const real_t* const SFEM_RESTRICT    data,          //
               real_t* const SFEM_RESTRICT          field,         // // Output
               sfem_resample_field_info*            info);                    //

/**
 * @brief
 *
 * @param mpi_size
 * @param mpi_rank
 * @param mesh
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param field
 * @param g
 * @param info
 * @return int
 */
int                                                                       //
resample_field_mesh_tet10(const int                            mpi_size,  // MPI size
                          const int                            mpi_rank,  // MPI rank
                          const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh
                          const ptrdiff_t* const SFEM_RESTRICT n,         // SDF: n[3]
                          const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                          const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                          const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                          const real_t* const SFEM_RESTRICT    data,      // SDF: data
                          real_t* const SFEM_RESTRICT          g,         // Output
                          sfem_resample_field_info*            info);                // info

/**
 * @brief Resamples a field from a tetrahedral mesh to a structured grid using MPI.
 *
 * This function resamples a field defined on a tetrahedral mesh onto a structured grid,
 * utilizing MPI for parallel processing. It's designed to distribute the computational
 * load across multiple processes for efficiency.
 *
 * @param[in] mpi_size The total number of MPI processes.
 * @param[in] mpi_rank The rank of the current MPI process (0 to mpi_size - 1).
 * @param[in] mesh A pointer to the mesh_t struct, containing the tetrahedral mesh data.
 * @param[in] n Number of grid points in each dimension of the structured grid (nx, ny, nz).
 * @param[in] stride Stride values for the structured grid, defining memory offsets
 *                   between grid points in each dimension.
 * @param[in] origin Origin of the structured grid (coordinates of the grid's corner).
 * @param[in] delta Grid spacing (dx, dy, dz) in each dimension of the structured grid.
 * @param[in] data Input field defined on the structured grid.
 * @param[out] g Output field, where the resampled values from the tetrahedral mesh
 *               will be accumulated.
 * @param[in,out] info A pointer to the sfem_resample_field_info struct, containing
 *                   information about the resampling process (e.g., quadrature nodes count).
 *
 * @details
 * This function distributes the tetrahedral mesh elements across MPI processes. Each process
 * then performs a local resampling operation, integrating field values at quadrature points
 * within its assigned elements and updating the corresponding locations on the structured grid.
 * MPI is used to manage the parallel execution and potentially gather results (though the
 * specific details of result aggregation would depend on the broader context of how 'g' is used).
 *
 * @return 0 if the operation is successful.
 */
int                                                                      //
resample_field_mesh_tet4(const int                            mpi_size,  // MPI size
                         const int                            mpi_rank,  // MPI rank
                         const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh: mesh_t struct
                         const ptrdiff_t* const SFEM_RESTRICT n,         // SDF: n[3]
                         const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                         const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                         const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                         const real_t* const SFEM_RESTRICT    data,      // SDF: data
                         real_t* const SFEM_RESTRICT          g,         // Output
                         sfem_resample_field_info*            info);                //

/**
 * @brief Determines which points in a structured grid are inside a tetrahedral mesh.
 *
 * This function creates a binary mask (bit array) indicating which points in a structured grid
 * are located inside a tetrahedral (TET4) mesh. The determination is done in parallel using MPI,
 * with the computational load distributed across multiple processes.
 *
 * @param[in] mpi_size  The total number of MPI processes.
 * @param[in] mpi_rank  The rank of the current MPI process (0 to mpi_size - 1).
 * @param[in] mesh      A pointer to the mesh_t struct, containing the tetrahedral mesh data.
 * @param[in] n         Number of grid points in each dimension of the structured grid (nx, ny, nz).
 * @param[in] stride    Stride values for the structured grid, defining memory offsets
 *                      between grid points in each dimension.
 * @param[in] origin    Origin of the structured grid (coordinates of the grid's corner).
 * @param[in] delta     Grid spacing (dx, dy, dz) in each dimension of the structured grid.
 * @param[out] bit_array Output bit array where each bit indicates whether the corresponding grid point
 *                      is inside the mesh (1) or outside the mesh (0).
 * @param[in,out] info  A pointer to the sfem_resample_field_info struct, containing information
 *                      about the process configuration (e.g., use of accelerators).
 *
 * @details
 * The function distributes the structured grid across MPI processes. Each process determines
 * whether its assigned grid points are inside or outside the tetrahedral mesh. This information
 * is encoded in the bit_array, where a set bit (1) indicates that the corresponding grid point
 * is inside the mesh, and a cleared bit (0) indicates that it's outside.
 *
 * This inside/outside determination is useful for subsequent operations that need to know which
 * grid points should be processed (e.g., for level set computations, domain masking, or
 * selective field resampling).
 *
 * @return 0 if the operation is successful.
 */
int                                                          //
in_out_tet4(const int                            mpi_size,   // MPI size
            const int                            mpi_rank,   // MPI rank
            const mesh_t* const SFEM_RESTRICT    mesh,       // Mesh: mesh_t struct
            const ptrdiff_t* const SFEM_RESTRICT n,          // SDF: n[3]
            const ptrdiff_t* const SFEM_RESTRICT stride,     // SDF: stride[3]
            const geom_t* const SFEM_RESTRICT    origin,     // SDF: origin[3]
            const geom_t* const SFEM_RESTRICT    delta,      // SDF: delta[3]
            BitArray                             bit_array,  // Output
            sfem_resample_field_info*            info);                 //

/**
 * @brief Resamples a field from a tetrahedral mesh to a structured grid (adjoint version).
 *
 * This function performs an adjoint (reverse) resampling operation using MPI, transferring a
 * field from a tetrahedral mesh to a structured grid. It distributes the values from the
 * tetrahedral mesh to the structured grid, acting as the transpose of a forward resampling.
 *
 * @param[in] mpi_size  The total number of MPI processes.
 * @param[in] mpi_rank  The rank of the current MPI process (0 to mpi_size - 1).
 * @param[in] mesh      A pointer to the mesh_t struct, containing the tetrahedral mesh data.
 * @param[in] n         Number of grid points in each dimension of the structured grid (nx, ny, nz).
 * @param[in] stride    Stride values for the structured grid, defining memory offsets between grid points in each dimension.
 * @param[in] origin    Origin of the structured grid (coordinates of the grid's corner).
 * @param[in] delta     Grid spacing (dx, dy, dz) in each dimension of the structured grid.
 * @param[in] g         Input field (weighted field) defined on the tetrahedral mesh.
 * @param[out] data     Output array representing the structured grid, where the resampled field values will be stored.
 * @param[in,out] info  A pointer to the sfem_resample_field_info struct, containing information about the resampling process.
 *
 * @details
 * This function distributes the tetrahedral mesh elements across MPI processes. Each process then performs a
 * local adjoint resampling operation, transferring values from its assigned elements to the structured grid.
 * The "adjoint" nature means that it distributes values from the tetrahedral mesh to the structured grid,
 * rather than interpolating values from the structured grid to the tetrahedral mesh.
 *
 * @return 0 if the operation is successful.
 */
int                                                                         //
resample_field_adjoint_tet4(const int                            mpi_size,  // MPI size
                            const int                            mpi_rank,  // MPI rank
                            const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh: mesh_t struct
                            const ptrdiff_t* const SFEM_RESTRICT n,         // SDF: n[3]
                            const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                            const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                            const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                            const real_t* const SFEM_RESTRICT    g,         // Weighted field
                            real_t* const SFEM_RESTRICT          data,      // SDF: data (output)
                            unsigned int*                        data_cnt,  // SDF: data count (output)
                            sfem_resample_field_info*            info);                // Info struct with options and flags

/// @brief  DEBUG code for testing the adjoint resampling operation
/// @param mpi_size
/// @param mpi_rank
/// @param mesh
/// @param n
/// @param stride
/// @param origin
/// @param delta
/// @param in_data
/// @param out_data
/// @param g
/// @param info
/// @return
int                                                                              //
resample_field_TEST_adjoint_tet4(const int                            mpi_size,  // MPI size
                                 const int                            mpi_rank,  // MPI rank
                                 const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh: mesh_t struct
                                 const ptrdiff_t* const SFEM_RESTRICT n,         // SDF: n[3]
                                 const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                                 const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                                 const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                                 const real_t* const SFEM_RESTRICT    in_data,   // Weighted field
                                 real_t* const SFEM_RESTRICT          out_data,  // SDF: data (output)
                                 real_t* const SFEM_RESTRICT          g,         // Weighted field (output)
                                 sfem_resample_field_info*            info);

int                                                             //
interpolate_field(const ptrdiff_t                      nnodes,  //
                  geom_t** const SFEM_RESTRICT         xyz,     //
                  const ptrdiff_t* const SFEM_RESTRICT n,       // SDF
                  const ptrdiff_t* const SFEM_RESTRICT stride,  //
                  const geom_t* const SFEM_RESTRICT    origin,  //
                  const geom_t* const SFEM_RESTRICT    delta,   //
                  const real_t* const SFEM_RESTRICT    data,    //
                  real_t* const SFEM_RESTRICT          field);           // Output

int                                                            //
field_view(MPI_Comm                             comm,          //
           const ptrdiff_t                      nnodes,        //
           const geom_t* SFEM_RESTRICT          z_coordinate,  //
           const ptrdiff_t* const               nlocal,        //
           const ptrdiff_t* const SFEM_RESTRICT nglobal,       //
           const ptrdiff_t* const SFEM_RESTRICT stride,        //
           const geom_t* const                  origin,        //
           const geom_t* const SFEM_RESTRICT    delta,         //
           const real_t* const                  field,         //
           real_t**                             field_out,     //
           ptrdiff_t*                           z_nlocal_out,  //
           geom_t* const SFEM_RESTRICT          z_origin_out);          //

int                                                                          //
field_view_ensure_margin(MPI_Comm                             comm,          //
                         const ptrdiff_t                      nnodes,        //
                         const geom_t* SFEM_RESTRICT          z_coordinate,  //
                         const ptrdiff_t* const               nlocal,        //
                         const ptrdiff_t* const SFEM_RESTRICT nglobal,       //
                         const ptrdiff_t* const SFEM_RESTRICT stride,        //
                         const geom_t* const                  origin,        //
                         const geom_t* const SFEM_RESTRICT    delta,         //
                         const real_t* const                  field,         //
                         const ptrdiff_t                      z_margin,      //
                         real_t**                             field_out,     //
                         ptrdiff_t*                           z_nlocal_out,  //
                         geom_t* const SFEM_RESTRICT          z_origin_out);          //

/**
 * @brief Resample a field from a tetrahedral mesh to a structured grid (version 2).
 *
 * This function resamples a field defined on a tetrahedral mesh onto a structured grid.
 * It iterates through a set of tetrahedral elements and, for each element, integrates
 * the field values at quadrature points within the element. The resulting values are
 * then used to update the corresponding locations on the structured grid.
 *
 * @param[in] start_element Index of the first tetrahedral element to process.
 * @param[in] end_element   Index of the last tetrahedral element to process.
 * @param[in] nnodes        Total number of nodes in the tetrahedral mesh.
 * @param[in] elems         Array of element connectivity: `elems[v][element_i]` is the global
 *                          index of the v-th vertex of the element_i-th tetrahedron.
 * @param[in] xyz           Array of vertex coordinates: `xyz[d][i]` is the d-th coordinate
 *                          (d=0 for x, d=1 for y, d=2 for z) of the i-th vertex.
 * @param[in] n             Number of grid points in each dimension of the structured grid (nx, ny, nz).
 * @param[in] stride        Stride values for the structured grid, defining memory offsets
 *                          between grid points in each dimension.
 * @param[in] origin        Origin of the structured grid (coordinates of the grid's corner).
 * @param[in] delta         Grid spacing (dx, dy, dz) in each dimension of the structured grid.
 * @param[in] data          Input field defined on the structured grid.
 * @param[out] weighted_field Output field, where the resampled values from the tetrahedral
 *                           mesh will be accumulated.
 *
 * @details
 * The function transforms quadrature points from the reference tetrahedron to physical space
 * and determines the corresponding location on the structured grid. It then uses shape
 * functions to interpolate the field values from the tetrahedral mesh to the grid location,
 * effectively transferring the field from the mesh to the grid.
 *
 * @return 0 if the operation is successful.
 */
int                                                                               //
tet4_resample_field_local_v2(const ptrdiff_t                      start_element,  // Mesh
                             const ptrdiff_t                      end_element,    // Mesh
                             const ptrdiff_t                      nnodes,         // Mesh
                             const idx_t** const SFEM_RESTRICT    elems,          // Mesh
                             const geom_t** const SFEM_RESTRICT   xyz,            // Mesh
                             const ptrdiff_t* const SFEM_RESTRICT n,              // SDF
                             const ptrdiff_t* const SFEM_RESTRICT stride,         // SDF
                             const geom_t* const SFEM_RESTRICT    origin,         // SDF
                             const geom_t* const SFEM_RESTRICT    delta,          // SDF
                             const real_t* const SFEM_RESTRICT    data,           // SDF
                             real_t* const SFEM_RESTRICT          weighted_field);         // Output

/**
 * @brief Adjoint resampling: Transfers data from a tetrahedral mesh to a hexahedral grid.
 *
 * This function performs an adjoint (reverse) resampling operation, transferring a field
 * defined on a tetrahedral mesh to a structured hexahedral grid. It distributes the
 * values from the tetrahedral mesh to the hexahedral grid, acting as the transpose
 * of a forward resampling operation.
 *
 * @param[in] start_element Index of the first tetrahedral element to process.
 * @param[in] end_element   Index of the last tetrahedral element to process.
 * @param[in] nnodes        Total number of nodes in the tetrahedral mesh.
 * @param[in] elems         Array of element connectivity: `elems[v][element_i]` is the global
 *                          index of the v-th vertex of the element_i-th tetrahedron.
 * @param[in] xyz           Array of vertex coordinates: `xyz[d][i]` is the d-th coordinate
 *                          (d=0 for x, d=1 for y, d=2 for z) of the i-th vertex.
 * @param[in] n             Number of grid points in each dimension of the hexahedral grid (nx, ny, nz).
 * @param[in] stride        Stride values for the hexahedral grid, defining memory offsets
 *                          between grid points in each dimension.
 * @param[in] origin        Origin of the hexahedral grid (coordinates of the grid's corner).
 * @param[in] delta         Grid spacing (dx, dy, dz) in each dimension of the hexahedral grid.
 * @param[in] weighted_field Input field defined on the vertices of the tetrahedral mesh.
 * @param[out] data          Output array representing the structured hexahedral grid, where
 *                          the resampled field values will be stored.
 *
 * @details
 * The function iterates through each tetrahedral element and quadrature point within that element.
 * It transforms the quadrature point to physical space and determines which hexahedral grid
 * cell contains that point. The value from the tetrahedral mesh (weighted by shape functions
 * and quadrature weights) is then distributed to the vertices of the containing hexahedral
 * grid cell. This process effectively projects the tetrahedral field onto the hexahedral grid.
 *
 * The "adjoint" nature of this function means that it's the reverse of a typical
 * interpolation; instead of pulling values from the hexahedral grid to the tetrahedron,
 * it pushes values from the tetrahedron to the hexahedral grid.
 *
 * @return 0 if the operation is successful.
 */
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
                                  real_t* const SFEM_RESTRICT          data);                    // Output

/**
 * @brief Count how many tetrahedral elements contribute to each grid point during adjoint resampling.
 *
 * This function traverses tetrahedral elements and counts how many times each point in the
 * structured grid is updated during the adjoint resampling process. This information can be
 * used for proper normalization of accumulated field values.
 *
 * @param[in] start_element Index of the first tetrahedral element to process.
 * @param[in] end_element   Index of the last tetrahedral element to process.
 * @param[in] nnodes        Total number of nodes in the tetrahedral mesh.
 * @param[in] elems         Array of element connectivity: `elems[v][element_i]` is the global
 *                          index of the v-th vertex of the element_i-th tetrahedron.
 * @param[in] xyz           Array of vertex coordinates: `xyz[d][i]` is the d-th coordinate
 *                          (d=0 for x, d=1 for y, d=2 for z) of the i-th vertex.
 * @param[in] n             Number of grid points in each dimension of the structured grid (nx, ny, nz).
 * @param[in] stride        Stride values for the structured grid, defining memory offsets
 *                          between grid points in each dimension.
 * @param[in] origin        Origin of the structured grid (coordinates of the grid's corner).
 * @param[in] delta         Grid spacing (dx, dy, dz) in each dimension of the structured grid.
 * @param[in] weighted_field Input field defined on the tetrahedral mesh (used only for quadrature logic).
 * @param[out] data_cnt     Output array that counts how many times each grid point is updated
 *                          during the adjoint resampling operation.
 *
 * @details
 * This function mimics the traversal pattern of the adjoint resampling operation but instead
 * of accumulating field values, it increments a counter for each grid point that would be
 * updated. The resulting count map can be used to normalize field values in cases where
 * multiple tetrahedral elements contribute to the same grid point.
 *
 * @return 0 if the operation is successful.
 */
int                                                                         //
tet4_cnt_mesh_adjoint(const ptrdiff_t                      start_element,   // Mesh
                      const ptrdiff_t                      end_element,     //
                      const ptrdiff_t                      nnodes,          //
                      const idx_t** const SFEM_RESTRICT    elems,           //
                      const geom_t** const SFEM_RESTRICT   xyz,             //
                      const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                      const ptrdiff_t* const SFEM_RESTRICT stride,          //
                      const geom_t* const SFEM_RESTRICT    origin,          //
                      const geom_t* const SFEM_RESTRICT    delta,           //
                      const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                      unsigned int* const SFEM_RESTRICT    data_cnt);          // Output

/** @brief Function pointer type for a function of three variables. */
typedef real_t (*function_XYZ_t)(real_t x, real_t y, real_t z);

/**
 * @brief Apply a function to a mesh
 *
 * This function applies a given function to a mesh within the specified range of elements.
 *
 * @param start_element [in] The starting element index of the mesh to apply the function to.
 * @param end_element [in] The ending element index of the mesh to apply the function to.
 * @param nnodes [in] Number of nodes in the mesh.
 * @param elems [in] Pointer to the array of element indices in the mesh.
 * @param xyz [in] Pointer to the array of geometric coordinates of the nodes in the mesh.
 * @param fun [in] The function to be applied to the mesh.
 * @param weighted_field [out] Pointer to the output array where the weighted field will be stored.
 */
int                                                                  //
apply_fun_to_mesh(const ptrdiff_t                    start_element,  // Mesh
                  const ptrdiff_t                    end_element,    // Mesh
                  const ptrdiff_t                    nnodes,         // Mesh
                  const idx_t** const SFEM_RESTRICT  elems,          // Mesh
                  const geom_t** const SFEM_RESTRICT xyz,            // Mesh
                  const function_XYZ_t               fun,            // Function
                  real_t* const SFEM_RESTRICT        weighted_field);       //   Output (weighted field)

#ifdef __cplusplus
}
#endif

#endif  // SFEM_RESAMPLE_FIELD_H
