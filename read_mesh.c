
#include "read_mesh.h"

#include "../matrix.io/utils.h"
#include "../matrix.io/matrixio_array.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

static ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
}

int read_mesh(MPI_Comm comm, const char *folder, mesh_t *mesh) {
	///////////////////////////////////////////////////////////////
    // FIXME check from folder
    int nnodesxelem = 4;
    int ndims = 3;

    MPI_Datatype mpi_geom_t = MPI_FLOAT;
    MPI_Datatype mpi_idx_t = MPI_INT;

    ///////////////////////////////////////////////////////////////

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    ptrdiff_t n_local_elements = 0, n_elements = 0;
    ptrdiff_t n_local_nodes = 0, n_nodes = 0;

    char path[1024 * 10];

    idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);

    {
        idx_t *id = 0;

        for (int d = 0; d < nnodesxelem; ++d) {
            sprintf(path, "%s/i%d.raw", folder, d);
            array_read(comm, path, mpi_idx_t, (void **)&id, &n_local_elements, &n_elements);
            elems[d] = id;
        }
    }

    const ptrdiff_t nbytes = sizeof(ptrdiff_t) * (size + 1);
    ptrdiff_t * pcount = (ptrdiff_t *)malloc(nbytes);
    memset(pcount, 0, nbytes);

    long * distro = (ptrdiff_t *)malloc(sizeof(ptrdiff_t) * (size + 1));
    memset(pcount, 0, nbytes);

    MPI_Allreduce(MPI_IN_PLACE, &distro[1], size, MPI_LONG, MPI_SUM, comm);

    for(int i = 0; i < size; ++i) {
    	distro[i+1] += distro[i];
    }

    // ALGO 1

    // We want
    // Local nodes
    // Ghost nodes
    // (NEXT) What about aura and ghost elements?

    // For all elements e
    // - For all nodes in e
    //   - find bucket and increase count
    // Allocate space for indices (local / remote separate)
    // Store indices in allocated space
    // Sort buffers and create unique sorted lists
    // Count nodes owned by remote processs
    // Create table from rank to offset
    // Owned count and ghost count can be computed now
    // Create local crs and remote crs graphs, crs we have to account for the different offsets
    // - Local crs 
    // - Remote crs, remote idx are sorted based on rank
    // 

    // ALGO 2

    // Read xyz
    geom_t **xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

    {
        geom_t *x = 0;
        char coord_names[4] = {'x', 'y', 'z', 't'};

        for (int d = 0; d < ndims; ++d) {
            sprintf(path, "%s/%c.raw", folder, coord_names[d]);
            array_read(comm, path, mpi_geom_t, (void **)&x, &n_local_nodes, &n_nodes);
            xyz[d] = x;
        }
    }

    return 0;
}

int serial_read_tet_mesh(const char *folder, ptrdiff_t *nelements, idx_t *elems[4], ptrdiff_t *nnodes, geom_t *xyz[4]) {
    char path[1024 * 10];

    {
        sprintf(path, "%s/x.raw", folder);
        ptrdiff_t x_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[0]);

        sprintf(path, "%s/y.raw", folder);
        ptrdiff_t y_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[1]);

        sprintf(path, "%s/z.raw", folder);
        ptrdiff_t z_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[2]);

        assert(x_nnodes == y_nnodes);
        assert(x_nnodes == z_nnodes);

        x_nnodes /= sizeof(geom_t);
        assert(x_nnodes * sizeof(geom_t) == y_nnodes);
        *nnodes = x_nnodes;
    }

    {
        sprintf(path, "%s/i0.raw", folder);
        ptrdiff_t nindex0 = read_file(MPI_COMM_SELF, path, (void **)&elems[0]);

        sprintf(path, "%s/i1.raw", folder);
        ptrdiff_t nindex1 = read_file(MPI_COMM_SELF, path, (void **)&elems[1]);

        sprintf(path, "%s/i2.raw", folder);
        ptrdiff_t nindex2 = read_file(MPI_COMM_SELF, path, (void **)&elems[2]);

        sprintf(path, "%s/i3.raw", folder);
        ptrdiff_t nindex3 = read_file(MPI_COMM_SELF, path, (void **)&elems[3]);

        assert(nindex0 == nindex1);
        assert(nindex3 == nindex2);

        nindex0 /= sizeof(idx_t);
        assert(nindex0 * sizeof(idx_t) == nindex1);
        *nelements = nindex0;
    }

    return 0;
}
