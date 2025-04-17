
#include "field_mpi_domain.h"

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
field_mpi_domain_t build_field_mpi_domain(const int        mpi_rank,  //
                                          const ptrdiff_t  n_zyx,     //
                                          const ptrdiff_t* nlocal,    //
                                          const geom_t*    origin,    //
                                          const geom_t*    delta) {      //
    field_mpi_domain_t domain;

    domain.mpi_rank = mpi_rank;
    domain.n_zyx    = n_zyx;

    memcpy(domain.nlocal, nlocal, 3 * sizeof(ptrdiff_t));
    memcpy(domain.origin, origin, 3 * sizeof(geom_t));

    // Calculate start indices based on the logic from print_rank_info
    domain.start_indices[0] = 0;  // Assuming x index starts at 0 for all ranks
    domain.start_indices[1] = 0;  // Assuming y index starts at 0 for all ranks
    // Ensure delta[2] is not zero to avoid division by zero
    if (delta[2] != 0) {
        // Use round to get the nearest integer index, handle potential floating point inaccuracies
        domain.start_indices[2] = (int)round(origin[2] / delta[2]);
    } else {
        // Handle the case where delta[2] is zero, perhaps set to 0 or report an error
        domain.start_indices[2] = 0;
        // Optionally print an error or warning
        // fprintf(stderr, "Warning: delta[2] is zero, cannot calculate start_index_z accurately.\n");
    }

    return domain;
}