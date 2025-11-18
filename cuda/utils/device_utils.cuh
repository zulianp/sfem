#ifndef __CUDA_UTILS_DEVICE_UTILS_CUH__
#define __CUDA_UTILS_DEVICE_UTILS_CUH__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* acc_get_device_properties(const int device_id);

int getSMCount();

#endif  // __CUDA_UTILS_DEVICE_UTILS_CUH__