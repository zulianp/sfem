#ifndef SFEM_GRID_H
#define SFEM_GRID_H

#include <mpi.h>
#include <stddef.h>

typedef struct {
    MPI_Comm comm;

    ptrdiff_t extent[3];
    ptrdiff_t stride[3];

    ptrdiff_t z_global_extent;
    ptrdiff_t z_margin_left;
    ptrdiff_t z_margin_right;
    ptrdiff_t z_begin;
    
    ptrdiff_t offset;

    ptrdiff_t size;
    ptrdiff_t local_size;
    ptrdiff_t local_size_with_margins;
} gridz_t;




void gridz_create(gridz_t *const g, MPI_Comm comm, ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, int overlap);

void gridz_read_field(const gridz_t *const g, const char *path, MPI_Datatype data_type, void *const data);
void gridz_write_field(const gridz_t *const g, const char *path, MPI_Datatype data_type, const void *const data);

void gridz_synchronize_field(const gridz_t *const g, MPI_Datatype data_type, char *const data);

void grid1_synchronize(MPI_Comm comm,
                       ptrdiff_t ln,
                       ptrdiff_t stride,
                       ptrdiff_t lmargin,
                       ptrdiff_t rmargin,
                       MPI_Datatype datatype,
                       char *const data);

int read_raw_file(MPI_Comm comm,
                  const char *path,
                  ptrdiff_t begin,
                  ptrdiff_t ln,
                  MPI_Datatype datatype,
                  void *const data);

int write_raw_file(MPI_Comm comm,
                   const char *path,
                   ptrdiff_t n,
                   ptrdiff_t begin,
                   ptrdiff_t ln,
                   MPI_Datatype datatype,
                   const void *const data);

void gridz_z_ownership_ranges(gridz_t *const g, ptrdiff_t *const ranges);

#endif  // SFEM_GRID_H
