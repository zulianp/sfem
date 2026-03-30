#ifndef __CELL_BUILD_TET_GEOM_CUH__
#define __CELL_BUILD_TET_GEOM_CUH__

#include "cell_list_cuda.cuh"
#include "cell_list_resampling_gpu.h"
#include "resample_field_adjoint_cell_cuda.cuh"
#include "resample_field_adjoint_cell_cuda_shm.cuh"

__global__ void                                                                    //
mesh_tet_geometry_compute_inv_Jacobian_gpu(mesh_tet_geom_device_t geom_device,     //
                                           elems_tet4_device      elems_device) {  //
}

#endif  // __CELL_BUILD_TET_GEOM_CUH__