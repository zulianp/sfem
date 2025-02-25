#ifndef SFEM_RESAMPLE_FIELD_H
#define SFEM_RESAMPLE_FIELD_H

#include <mpi.h>
#include <stddef.h>

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
                                  real_t* const SFEM_RESTRICT          data);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_RESAMPLE_FIELD_H
