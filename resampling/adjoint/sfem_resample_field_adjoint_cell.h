#ifndef __SFEM_RESAMPLE_FIELD_ADJOINT_CELL_H__
#define __SFEM_RESAMPLE_FIELD_ADJOINT_CELL_H__

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cell_list_3d_map_mesh.h"
#include "cell_list_bench.h"
#include "cell_tet2box.h"
#include "resampling_utils.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"

/**
 * @brief Build bounding boxes and mesh geometry for the given mesh
 * @param mesh The input mesh
 * @param boxes Pointer to store the generated boxes data structure
 * @param mesh_geom Pointer to store the generated mesh geometry data structure
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int                                                           //
build_bounding_boxes_mesh_geom(const mesh_t     *mesh,        //
                               boxes_t         **boxes,       //
                               mesh_tet_geom_t **mesh_geom);  //

/**
 * @brief Build bounding box statistics and side length histograms for the given boxes
 * @param boxes The input boxes data structure
 * @param bins Number of bins to use for the side length histograms
 * @param stats Pointer to store the generated bounding box statistics
 * @param histograms Pointer to store the generated side length histograms
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int                                                                    //
build_bounding_box_statistics(const boxes_t              *boxes,       //
                              const int                   bins,        //
                              bounding_box_statistics_t **stats,       //
                              side_length_histograms_t  **histograms);  //

/**
 * @brief Update hex field values at a single quadrature node based on tet field values
 * @param mpi_size MPI size
 * @param mpi_rank MPI rank
 * @param x Physical x coordinate of the quadrature point
 * @param y Physical y coordinate of the quadrature point
 * @param z Physical z coordinate of the quadrature point
 * @param phys_w Quadrature weight for the quadrature point
 * @param index_tet The index of the tet containing the quadrature point
 * @param mesh Mesh structure
 * @param mesh_geom Mesh geometry data structure
 * @param n SDF dimensions n[3]
 * @param stride SDF stride[3]
 * @param origin SDF origin[3]
 * @param delta SDF delta[3]
 * @param weighted_field Weighted field values at mesh nodes
 * @param hex_element_field Output field values for the 8 hex nodes
 * @return int 0 on success
 */
int                                                                        //
update_hex_quad_node(const int                            mpi_size,        //
                     const int                            mpi_rank,        //
                     const real_t                         x,               //
                     const real_t                         y,               //
                     const real_t                         z,               //
                     const real_t                         phys_w,          //
                     const ptrdiff_t                      index_tet,       //
                     const mesh_t *const SFEM_RESTRICT    mesh,            //
                     mesh_tet_geom_t                     *mesh_geom,       //
                     const ptrdiff_t *const SFEM_RESTRICT n,               //
                     const ptrdiff_t *const SFEM_RESTRICT stride,          //
                     const geom_t *const SFEM_RESTRICT    origin,          //
                     const geom_t *const SFEM_RESTRICT    delta,           //
                     const real_t *const SFEM_RESTRICT    weighted_field,  //
                     real_t *const SFEM_RESTRICT          hex_element_field);

/**
 * @brief Update hex field by resampling from tet mesh at a given grid location
 * @param mpi_size MPI size
 * @param mpi_rank MPI rank
 * @param split_map Cell list split map data structure
 * @param boxes Boxes data structure
 * @param mesh_geom Mesh geometry data structure
 * @param i_grid The i index of the grid point in the hex mesh
 * @param j_grid The j index of the grid point in the hex mesh
 * @param mesh Mesh structure
 * @param n SDF dimensions n[3]
 * @param stride SDF stride[3]
 * @param origin SDF origin[3]
 * @param delta SDF delta[3]
 * @param weighted_field Weighted field values at mesh nodes
 * @param hex_field Output field for the hex cell containing (x,y,z)
 * @return int 0 on success
 */
int                                                                    //
update_hex_field(const int                            mpi_size,        //
                 const int                            mpi_rank,        //
                 cell_list_split_3d_2d_map_t         *split_map,       //
                 boxes_t                             *boxes,           //
                 mesh_tet_geom_t                     *mesh_geom,       //
                 const ptrdiff_t                      i_grid,          //
                 const ptrdiff_t                      j_grid,          //
                 const mesh_t *const SFEM_RESTRICT    mesh,            //
                 const ptrdiff_t *const SFEM_RESTRICT n,               //
                 const ptrdiff_t *const SFEM_RESTRICT stride,          //
                 const geom_t *const SFEM_RESTRICT    origin,          //
                 const geom_t *const SFEM_RESTRICT    delta,           //
                 const real_t *const SFEM_RESTRICT    weighted_field,  //
                 real_t *const SFEM_RESTRICT          hex_field);               //

/**
 * @brief Update hex field values at multiple quadrature nodes based on tet field values, optimized for multiple z values in the
 * same tet
 * @param mpi_size MPI size
 * @param mpi_rank MPI rank
 * @param x Physical x coordinate of the quadrature point
 * @param y Physical y coordinate of the quadrature point
 * @param z Array of physical z coordinates of the quadrature points (multiple z values to process)
 * @param z_size Size of the z array (number of z values to process)
 * @param phys_w Quadrature weight for the quadrature point
 * @param index_tet The index of the tet containing the quadrature points (same for all z values)
 * @param mesh Mesh structure
 * @param mesh_geom Mesh geometry data structure
 * @param n SDF dimensions n[3]
 * @param stride SDF stride[3]
 * @param origin SDF origin[3]
 * @param delta SDF delta[3]
 * @param weighted_field Weighted field values at mesh nodes
 * @param hex_element_field Output field values for the 8 hex nodes
 * @return int 0 on success
 */
int                                                                      //
update_hex_quad_node_vz(const int                            mpi_size,   // MPI size
                        const int                            mpi_rank,   // MPI rank
                        const real_t                         x,          // Physical x coordinate of the quadrature point
                        const real_t                         y,          // Physical y coordinate of the quadrature point
                        const real_t                        *z,          // Physical z coordinate of the quadrature point
                        const ptrdiff_t                      z_size,     // Size of the z array (number of z values to process)
                        const real_t                         phys_w,     // Quadrature weight for the quadrature point
                        const ptrdiff_t                      index_tet,  // The index of the tet containing the quadrature point
                        const mesh_t *const SFEM_RESTRICT    mesh,       // Mesh: mesh_t struct
                        mesh_tet_geom_t                     *mesh_geom,  // Mesh geometry data structure
                        const ptrdiff_t *const SFEM_RESTRICT n,          // SDF: n[3]
                        const ptrdiff_t *const SFEM_RESTRICT stride,     // SDF: stride[3]
                        const geom_t *const SFEM_RESTRICT    origin,     // SDF: origin[3]
                        const geom_t *const SFEM_RESTRICT    delta,      // SDF: delta[3]
                        const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                        real_t *const SFEM_RESTRICT          hex_element_field);       //

/**
 * @brief Transfer field values from tet mesh to hex mesh by iterating over hex grid points and resampling from the tet mesh
 * @param mpi_size MPI size
 * @param mpi_rank MPI rank
 * @param split_map Cell list split map data structure
 * @param boxes Boxes data structure
 * @param mesh_geom Mesh geometry data structure
 * @param mesh Mesh structure
 * @param n SDF dimensions n[3]
 * @param stride SDF stride[3]
 * @param origin SDF origin[3]
 * @param delta SDF delta[3]
 * @param weighted_field Weighted field values at mesh nodes
 * @param hex_field Output field for the hex mesh after resampling
 * @return int 0 on success
 */
int                                                                                   //
transfer_to_hex_field_cell_tet4(const int                            mpi_size,        // MPI size
                                const int                            mpi_rank,        // MPI rank
                                cell_list_split_3d_2d_map_t         *split_map,       // Cell list split map data structure
                                boxes_t                             *boxes,           // Boxes data structure
                                mesh_tet_geom_t                     *mesh_geom,       // Mesh geometry data structure
                                const mesh_t *const SFEM_RESTRICT    mesh,            // Mesh: mesh_t struct
                                const ptrdiff_t *const SFEM_RESTRICT n,               // SDF: n[3]
                                const ptrdiff_t *const SFEM_RESTRICT stride,          // SDF: stride[3]
                                const geom_t *const SFEM_RESTRICT    origin,          // SDF: origin[3]
                                const geom_t *const SFEM_RESTRICT    delta,           // SDF: delta[3]
                                const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                                real_t *const SFEM_RESTRICT          hex_field);               //

/**
 * @brief Compress and reorder key-value pairs by removing entries with key -1 and shifting valid entries to the front of
 * the arrays
 * @param keyArray Array of keys to be compressed and reordered
 * @param valArray Array of values corresponding to the keys, to be reordered in the same way as keyArray
 * @param n Size of the input arrays keyArray and valArray
 * @return int The count of valid items after compression and reordering
 */
int                                     //
compress_and_reorder(int    *keyArray,  //
                     real_t *valArray,  //
                     int     n);            //

#endif  // __SFEM_RESAMPLE_FIELD_ADJOINT_CELL_H__