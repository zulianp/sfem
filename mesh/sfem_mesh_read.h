#ifndef SFEM_MESH_READ_H
#define SFEM_MESH_READ_H

#include "sfem_defs.h"

#include <mpi.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int read_mapped_field(MPI_Comm           comm,        //
                      const char        *input_path,  //
                      const ptrdiff_t    n_local,     //
                      const ptrdiff_t    n_global,    //
                      const idx_t *const mapping,     //
                      MPI_Datatype       data_type,   //
                      void *const        data_out);          //

int array_read(MPI_Comm     comm,    //
               const char  *path,    //
               MPI_Datatype type,    //
               void        *data,    //
               ptrdiff_t    nlocal,  //
               ptrdiff_t    nglobal);   //

#ifdef __cplusplus
}
#endif

#endif  // SFEM_MESH_READ_H