#ifndef SFEM_HIP_COMPAT_CUB_DEVICE_SCAN_CUH
#define SFEM_HIP_COMPAT_CUB_DEVICE_SCAN_CUH

// CUB's device scan under HIP: hipCUB, whose cub::DeviceScan calls take the same arguments in the
// same order.
#include <hipcub/device/device_scan.hpp>

namespace cub = hipcub;

#endif  // SFEM_HIP_COMPAT_CUB_DEVICE_SCAN_CUH
