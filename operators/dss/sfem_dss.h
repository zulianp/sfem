#pragma once

#include <stddef.h>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif

#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

#ifdef __cplusplus
#include "sfem_dss_bxyze.hpp"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int sfem_dss_apply_real32(const ptrdiff_t n, const float *const input, float *const output);
int sfem_dss_apply_real64(const ptrdiff_t n, const double *const input, double *const output);

#ifdef __cplusplus
}
#endif
