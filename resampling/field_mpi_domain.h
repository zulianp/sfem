#ifndef __FIELD_MPI_DOMAIN_H__
#define __FIELD_MPI_DOMAIN_H__

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "mesh_aura.h"
#include "read_mesh.h"
#include "sfem_mesh_write.h"
#include "sfem_resample_field.h"
#include "sfem_resample_field_tet4_math.h"

#include "mass.h"
#include "mesh_utils.h"
#include "quadratures_rule.h"
#include "tet10_resample_field.h"

struct field_mpi_domain {
    int    mpi_rank;          // MPI rank associated with this domain
    int    n_zyx;             // Total number of elements in the z, y, and x directions
    int    start_indices[3];  // Start indices for the z, y, and x directions (with respect to the global grid)
    int    nlocal[3];         // Number of local elements in the z, y, and x directions
    geom_t origin[3];         // Local origin coordinates in the z, y, and x directions
};

typedef struct field_mpi_domain field_mpi_domain_t;

/**
 * @brief Prints the field_mpi_domain_t structure.
 *
 * @param domain
 */
void print_field_mpi_domain(const field_mpi_domain_t* const domain);  //

int test_field_mpi_domain(int argc, char* argv[]);

/**
 * @brief Builds a field_mpi_domain_t structure.
 *
 * @param mpi_rank The MPI rank associated with this domain.
 * @param n_zyx Total number of elements in the z, y, and x directions for this rank.
 * @param nlocal Number of local elements in the z, y, and x directions.
 * @param origin Local origin coordinates in the z, y, and x directions.
 * @param delta Grid spacing in the z, y, and x directions.
 * @return field_mpi_domain_t The populated structure.
 */
field_mpi_domain_t                                 //
make_field_mpi_domain(const int        mpi_rank,  //
                       const ptrdiff_t  n_zyx,     //
                       const ptrdiff_t* nlocal,    //
                       const geom_t*    origin,    //
                       const geom_t*    delta);       //

/**
 * @brief Calculates the subset of the Z domain that overlaps between two field_mpi_domain_t structures.
 *
 * @param domain_a
 * @param domain_b
 * @param overlapped_domain
 */
int                                                                  //
calculate_subset_Z_domain(const field_mpi_domain_t* const domain_a,  //
                          const field_mpi_domain_t* const domain_b,  //
                          field_mpi_domain_t* const       sub_domain);     //


int is_point_in_domain(const field_mpi_domain_t* const domain, const geom_t point[3]);

#endif  // __FIELD_MPI_DOMAIN_H__
