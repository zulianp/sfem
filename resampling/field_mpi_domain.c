
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
field_mpi_domain_t make_field_mpi_domain(const int        mpi_rank,  //
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

// struct field_mpi_domain {
//     int    mpi_rank;          // MPI rank associated with this domain
//     int    n_zyx;             // Total number of elements in the z, y, and x directions
//     int    start_indices[3];  // Start indices for the z, y, and x directions (with respect to the global grid)
//     int    nlocal[3];         // Number of local elements in the z, y, and x directions
//     geom_t origin[3];         // Local origin coordinates in the z, y, and x directions
// };

/////////////////////////////////////////////////////////////////////////////////
// print_field_mpi_domain
////////////////////////////////////////////////////////////////////////////////
void print_field_mpi_domain(const field_mpi_domain_t* const domain) {
    printf("rank %d: n_zyx = %d, nlocal = (%d, %d, %d), start_indices = (%d, %d, %d), origin = (%f, %f, %f)\n",
           domain->mpi_rank,
           domain->n_zyx,
           domain->nlocal[0],
           domain->nlocal[1],
           domain->nlocal[2],
           domain->start_indices[0],
           domain->start_indices[1],
           domain->start_indices[2],
           domain->origin[0],
           domain->origin[1],
           domain->origin[2]);
}

/////////////////////////////////////////////////////////////////////////////////
// calculate_subset_Z_domain
////////////////////////////////////////////////////////////////////////////////
int                                                                  //
calculate_subset_Z_domain(const field_mpi_domain_t* const domain_a,  //
                          const field_mpi_domain_t* const domain_b,  //
                          field_mpi_domain_t* const       sub_domain) {    //

    // Calculate the overlapped domain based on the start indices and nlocal values
    // This function find the portion of domain_b that is a subset of domain_a

    sub_domain->mpi_rank  = domain_a->mpi_rank;
    sub_domain->n_zyx     = domain_a->n_zyx;
    sub_domain->nlocal[0] = domain_a->nlocal[0];
    sub_domain->nlocal[1] = domain_a->nlocal[1];
    sub_domain->origin[0] = domain_a->origin[0];
    sub_domain->origin[1] = domain_a->origin[1];

    const int start_index_z_a = domain_a->start_indices[2];
    const int end_index_z_a   = start_index_z_a + domain_a->nlocal[2] - 1;

    const int start_index_z_b = domain_b->start_indices[2];
    const int end_index_z_b   = start_index_z_b + domain_b->nlocal[2] - 1;

    int overlap_start = 0;
    int overlap_end   = 0;

    if (start_index_z_a > end_index_z_b || start_index_z_b > end_index_z_a) {
        // No overlap

        sub_domain->nlocal[2]        = 0;
        sub_domain->start_indices[2] = 0;
        sub_domain->n_zyx            = 0;
        sub_domain->origin[2]        = 0.0;

        return 0;
    } else {
        // Calculate the overlap
        overlap_start = (start_index_z_a > start_index_z_b) ? start_index_z_a : start_index_z_b;
        overlap_end   = (end_index_z_a < end_index_z_b) ? end_index_z_a : end_index_z_b;

        sub_domain->nlocal[2] = overlap_end - overlap_start + 1;
    }

    // Calculate the start index for the overlapped domain
    sub_domain->start_indices[2] = overlap_start;

    return 1;  // Return 1 to indicate that the overlap was found (TRUE case)
}

/**
 * @brief Initializes a field_mpi_domain_t structure with specified parameters.
 *
 * @param mpi_rank The MPI rank for the domain.
 * @param start_z The starting index in the Z dimension.
 * @param nlocal_x Number of local elements in the X dimension.
 * @param nlocal_y Number of local elements in the Y dimension.
 * @param nlocal_z Number of local elements in the Z dimension.
 * @param delta The grid spacing (assumed uniform in x, y, z for origin calculation).
 * @return field_mpi_domain_t The initialized domain structure.
 */
field_mpi_domain_t initialize_field_mpi_domain(const int mpi_rank, const int start_z, const int nlocal_x, const int nlocal_y,
                                               const int nlocal_z, const float_t delta) {
    field_mpi_domain_t domain;

    domain.mpi_rank = mpi_rank;
    // Assuming start indices for x and y are always 0 for this initialization pattern
    domain.start_indices[0] = 0;
    domain.start_indices[1] = 0;
    domain.start_indices[2] = start_z;

    assert(nlocal_x > 0);
    assert(nlocal_y > 0);
    assert(nlocal_z > 0);

    domain.nlocal[0] = nlocal_x;
    domain.nlocal[1] = nlocal_y;
    domain.nlocal[2] = nlocal_z;

    // Use ptrdiff_t for calculation to avoid potential overflow before casting
    ptrdiff_t nx = nlocal_x;
    ptrdiff_t ny = nlocal_y;
    ptrdiff_t nz = nlocal_z;
    domain.n_zyx = (int)(nx * ny * nz);  // Cast the final result

    domain.origin[0] = domain.start_indices[0] * delta;
    domain.origin[1] = domain.start_indices[1] * delta;
    domain.origin[2] = domain.start_indices[2] * delta;

    return domain;
}

///////////////////////////////////////////////////////////////////////////////////
// tets_field_mpi_domain
///////////////////////////////////////////////////////////////////////////////////
int test_field_mpi_domain(int argc, char* argv[]) {  //

    field_mpi_domain_t domain_a, domain_b, overlapped_domain;

    float_t delta = 1.0;

    domain_a = initialize_field_mpi_domain(0,       // mpi_rank
                                           200,     // start_z
                                           100,     // nlocal_x
                                           100,     // nlocal_y
                                           300,     // nlocal_z
                                           delta);  //

    domain_b = initialize_field_mpi_domain(1,       // mpi_rank
                                           250,     // start_z
                                           100,     // nlocal_x
                                           100,     // nlocal_y
                                           300,     // nlocal_z
                                           delta);  //

    print_field_mpi_domain(&domain_a);
    print_field_mpi_domain(&domain_b);

    return 0;
}