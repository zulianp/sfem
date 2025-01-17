#include "sfem_base.h"

#include <mpi.h>

void sfem_abort(){ 
	MPI_Abort(MPI_COMM_WORLD, -1);
}
