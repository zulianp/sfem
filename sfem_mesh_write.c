#include "sfem_mesh_write.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <stdio.h>
#include <assert.h>

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

        return 0;
    } else {
        // TODO
        assert(0);
        return 1;
    }
}
