
#include "read_mesh.h"

#include "../matrix.io/utils.h"

#include <assert.h>
#include <stdio.h>

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

int serial_read_tet_mesh(
	const char*folder, 
	ptrdiff_t *nelements,
	idx_t *elems[4],
	ptrdiff_t *nnodes,
	geom_t *xyz[4]
	)
{
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
