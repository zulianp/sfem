#include "sfem_base.h"


#include <math.h>

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif 

{CONSTANTS}

{UTILITIES}

SFEM_DEVICE_FUNCTION void {NAME}_mk_jacobian(
{COORDINATES}const count_t stride_jacobian,
real_t *const SFEM_RESTRICT jacobian
)
{{
{JACOBIAN}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_jacobian_inverse(
{COORDINATES}const count_t stride_jacobian_inverse,
real_t *const SFEM_RESTRICT jacobian_inverse
)
{{
{JACOBIAN_INVERSE}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_jacobian_determinant_and_inverse(
{COORDINATES} //arrays
const count_t stride_jacobian_determinant,
real_t * const SFEM_RESTRICT jacobian_determinant,
const count_t stride_jacobian_inverse,
real_t *const SFEM_RESTRICT jacobian_inverse
)
{{
{JACOBIAN_DETERMINANT_AND_INVERSE}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_jacobian_determinant(
{COORDINATES} // arrays
const count_t stride_jacobian_determinant,
real_t * const SFEM_RESTRICT jacobian_determinant
)
{{
{JACOBIAN_DETERMINANT}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_fun(
{QUADRATURE_POINT} // arrays
const int stride_fun,
real_t * const SFEM_RESTRICT f
)
{{
{FUN}
}}

SFEM_DEVICE_FUNCTION real_t {NAME}_mk_interpolate(
{QUADRATURE_POINT} // arrays
const int stride_coeff,
const real_t * const SFEM_RESTRICT c
)
{{
return {INTERPOLATE};
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_grad_interpolate(
{QUADRATURE_POINT} // arrays
const count_t stride_jacobian_inverse,
const real_t *const SFEM_RESTRICT jacobian_inverse,
const int stride_coeff,
const real_t * const SFEM_RESTRICT c,
const int stride_grad,
real_t * const SFEM_RESTRICT grad
)
{{
{GRAD_INTERPOLATE}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_partial_x(
{QUADRATURE_POINT} // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gx
)
{{
{PARTIAL_X}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_partial_y(
{QUADRATURE_POINT} // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gy
)
{{
{PARTIAL_Y}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_partial_z(
{QUADRATURE_POINT} // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gz
) 
{{
{PARTIAL_Z}
}}
