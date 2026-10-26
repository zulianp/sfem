#include "sfem_base.hpp"

#include <mpi.h>

void sfem_abort(){ 
	MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);
}
