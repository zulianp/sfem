#include "grid.h"

#include "sfem_base.h"

#include "utils.h"

#include <stdio.h>

void gridz_create(gridz_t *const g,
                  MPI_Comm comm,
                  ptrdiff_t nx,
                  ptrdiff_t ny,
                  ptrdiff_t nz,
                  int overlap) {
    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    ptrdiff_t local_nz = nz / size;
    ptrdiff_t remainder = (nz - (local_nz * size));
    local_nz += rank < remainder;

    ptrdiff_t z_margin_left = (rank > 0) * overlap;
    ptrdiff_t z_margin_right = (rank < size - 1) * overlap;
    ptrdiff_t z_begin = (nz / size) * rank + MIN(rank, remainder);
    ptrdiff_t offset = z_begin * nx * ny;

    ptrdiff_t local_size_with_margins = nx * ny * (local_nz + z_margin_left + z_margin_right);
    ptrdiff_t stride_z = ny * nx;
    ptrdiff_t stride_y = nx;

    g->comm = comm;
    g->extent[0] = nx;
    g->extent[1] = ny;
    g->extent[2] = local_nz;

    g->stride[0] = 1;
    g->stride[1] = stride_y;
    g->stride[2] = stride_z;

    g->z_global_extent = nz;

    g->z_margin_left = z_margin_left;
    g->z_margin_right = z_margin_right;

    g->z_begin = z_begin;
    g->offset = offset;

    g->size = nx * ny * nz;
    g->local_size = nx * ny * local_nz;
    g->local_size_with_margins = local_size_with_margins;
}

void gridz_synchronize_field(const gridz_t *const g, MPI_Datatype data_type, char *const data) {
    int type_size;
    CATCH_MPI_ERROR(MPI_Type_size(data_type, &type_size));
    grid1_synchronize(g->comm,
                      g->extent[2],
                      g->stride[2] * type_size,
                      g->z_margin_left,
                      g->z_margin_right,
                      data_type,
                      data);
}

void grid1_synchronize(MPI_Comm comm,
                       ptrdiff_t ln,
                       ptrdiff_t stride,
                       ptrdiff_t lmargin,
                       ptrdiff_t rmargin,
                       MPI_Datatype datatype,
                       char *const data) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size == 1) return;

    int isodd = (rank & 1);
    for (int m = 0; m < 2; ++m) {
        if (m == isodd) {
            if (lmargin) {
                int lneigh = rank - 1;
                lneigh = (lneigh < 0) ? (lneigh + size) : lneigh;

                ptrdiff_t sendoffset = lmargin;
                ptrdiff_t recvoffset = 0;

                int recvtag = m;
                int sendtag = m;

                CATCH_MPI_ERROR(MPI_Sendrecv(
                    // send
                    &data[sendoffset * stride],
                    stride * lmargin,
                    datatype,
                    lneigh,
                    sendtag,
                    // recv
                    &data[recvoffset * stride],
                    stride * lmargin,
                    datatype,
                    lneigh,
                    recvtag,
                    //
                    comm,
                    MPI_STATUS_IGNORE));
            }

        } else {
            if (rmargin) {
                int rneigh = rank + 1;
                rneigh = (rneigh >= size) ? (rneigh - size) : rneigh;

                ptrdiff_t sendoffset = (lmargin + ln - rmargin);
                ptrdiff_t recvoffset = (lmargin + ln);

                int recvtag = m;
                int sendtag = m;
                CATCH_MPI_ERROR(MPI_Sendrecv(
                    // send
                    &data[sendoffset * stride],
                    stride * rmargin,
                    datatype,
                    rneigh,
                    sendtag,
                    // recv
                    &data[recvoffset * stride],
                    stride * rmargin,
                    datatype,
                    rneigh,
                    recvtag,
                    //
                    comm,
                    MPI_STATUS_IGNORE));
            }
        }
    }
}

int read_raw_file(MPI_Comm comm,
                  const char *path,
                  ptrdiff_t z_begin,
                  ptrdiff_t ln,
                  MPI_Datatype datatype,
                  void *const data) {
    MPI_File handle;
    MPI_Offset offset = z_begin;
    MPI_Status status;

    if (MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &handle) != MPI_SUCCESS) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        if (!rank) {
            fprintf(stderr, "Unable to read %s\n", path);
        }

        return 1;
    }

    CATCH_MPI_ERROR(MPI_File_read_at_all(handle, offset, data, ln, datatype, &status));

    CATCH_MPI_ERROR(MPI_File_close(&handle));
    return 0;
}

int write_raw_file(MPI_Comm comm,
                   const char *path,
                   ptrdiff_t n,
                   ptrdiff_t z_begin,
                   ptrdiff_t ln,
                   MPI_Datatype datatype,
                   const void *const data) {
    MPI_File handle;
    MPI_Offset offset = z_begin;
    MPI_Status status;

    if (MPI_File_open(comm, path, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &handle) !=
        MPI_SUCCESS) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        if (!rank) {
            fprintf(stderr, "Unable to write %s\n", path);
        }

        return 1;
    }

    CATCH_MPI_ERROR(MPI_File_set_size(handle, n));

    CATCH_MPI_ERROR(MPI_File_write_at_all(handle, offset, data, ln, datatype, &status));

    CATCH_MPI_ERROR(MPI_File_close(&handle));
    return 0;
}

void gridz_read_field(const gridz_t *const g,
                      const char *path,
                      MPI_Datatype data_type,
                      void *const data) {
    int type_size;
    CATCH_MPI_ERROR(MPI_Type_size(data_type, &type_size));

    ptrdiff_t array_offset = g->z_margin_left * g->stride[2] * type_size;

    read_raw_file(
        g->comm, path, g->offset, g->local_size, data_type, &((char *)data)[array_offset]);
}

void gridz_write_field(const gridz_t *const g,
                       const char *path,
                       MPI_Datatype data_type,
                       const void *const data) {
    int type_size;
    CATCH_MPI_ERROR(MPI_Type_size(data_type, &type_size));

    ptrdiff_t array_offset = g->z_margin_left * g->stride[2] * type_size;

    write_raw_file(
        g->comm, path, g->size, g->offset, g->local_size, data_type, &((char *)data)[array_offset]);
}

void gridz_z_ownership_ranges(gridz_t *const g, ptrdiff_t *const ranges) {
    int size;
    MPI_Comm_size(g->comm, &size);

    ptrdiff_t local_nz = g->z_global_extent / size;
    ptrdiff_t remainder = (g->z_global_extent - (local_nz * size));

    ranges[0] = 0;

    for (int rank = 0; rank < size; ++rank) {
        ptrdiff_t z_begin = (g->z_global_extent / size) * rank + MIN(rank, remainder);
        ranges[rank + 1] = z_begin;
    }
}
