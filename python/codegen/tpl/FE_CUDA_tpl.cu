#include "sfem_base.h"

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif 

{CONSTANTS}

SFEM_DEVICE_FUNCTION void {NAME}_mk_jacobian(
{COORDINATES}const count_t stride_jacobian,
real_t *jacobian
)
{{
{JACOBIAN}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_jacobian_inverse(
{COORDINATES}const count_t stride_jacobian_inverse,
real_t *jacobian_inverse
)
{{
{JACOBIAN_INVERSE}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_jacobian_determinant(
{COORDINATES} // arrays
const count_t stride_jacobian_determinant,
real_t *jacobian_determinant
)
{{
{JACOBIAN_DETERMINANT}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_fun(
{QUADRATURE_POINT} // arrays
const int stride_fun,
real_t * SFEM_RESTRICT f
)
{{
{FUN}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_partial_x(
{QUADRATURE_POINT} // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gx
)
{{
{PARTIAL_X}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_partial_y(
{QUADRATURE_POINT} // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gy
)
{{
{PARTIAL_Y}
}}

SFEM_DEVICE_FUNCTION void {NAME}_mk_partial_z(
{QUADRATURE_POINT} // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gz
) 
{{
{PARTIAL_Z}
}}
