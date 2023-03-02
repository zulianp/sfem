#include "sfem_mesh_write.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/utils.h"

int mesh_write(const char *path, const mesh_t *mesh) {
    // TODO
    MPI_Comm comm = mesh->comm;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    static const char *str_xyz = "xyzt";

    char output_path[2048];

    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0700);
    }

    if (size == 1) {
        for (int d = 0; d < mesh->spatial_dim; ++d) {
            sprintf(output_path, "%s/%c.raw", path, str_xyz[d]);
            array_write(comm, output_path, SFEM_MPI_GEOM_T, mesh->points[d], mesh->nnodes, mesh->nnodes);
        }

        for (int d = 0; d < mesh->element_type; ++d) {
            sprintf(output_path, "%s/i%d.raw", path, d);
            array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->elements[d], mesh->nelements, mesh->nelements);
        }

        if (mesh->node_mapping) {
            sprintf(output_path, "%s/node_mapping.raw", path);
            array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->node_mapping, mesh->nnodes, mesh->nnodes);
        }

        if (mesh->element_mapping) {
            sprintf(output_path, "%s/element_mapping.raw", path);
            array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->element_mapping, mesh->nelements, mesh->nelements);
        }

        return 0;
    } else {
        // TODO
        assert(0);
        return 1;
    }
}

int write_mapped_field(MPI_Comm comm,
                       const char *folder,
                       const ptrdiff_t n_local,
                       const ptrdiff_t n_global,
                       const idx_t *const mapping,
                       MPI_Datatype data_type,
                       void *const data) {
    int rank, size;
    MPI_Comm_size(comm, &rank);
    MPI_Comm_size(comm, &size);

    const ptrdiff_t local_file_size = n_global / size;
    const ptrdiff_t begin = (n_global / size) * rank;

    idx_t * send_count = (idx_t * )malloc((size + 1) * sizeof(idx_t));
    memset(send_count, 0, (size + 1) * sizeof(idx_t));

    for(ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx = mapping[i];
        int dest_rank = idx / local_file_size;
        assert(dest_rank < size);
        send_count[dest_rank]++;
    }

    idx_t * recv_count = (idx_t * )malloc((size + 1) * sizeof(idx_t));
    // memset(recv_count, 0, (size + 1) * sizeof(idx_t));

    CATCH_MPI_ERROR(
                MPI_Alltoall(send_count, 1, SFEM_MPI_IDX_T, recv_count, 1, SFEM_MPI_IDX_T, comm));

    free(send_count);
    return 0;
}
