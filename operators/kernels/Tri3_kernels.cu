#include "sfem_base.h"


#include <math.h>

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif 

static const int fe_spatial_dim = 2;
static const int fe_manifold_dim = 2;
static const int fe_n_nodes = 3;
static const char * fe_name = "Tri3";
static const int fe_n_nodes_for_jacobian = 3;
static const int fe_subparam_n_nodes = 3;
static const float fe_reference_measure = 1.0/2.0;


static SFEM_CUDA_INLINE real_t Tri3_mk_det_2(const count_t stride,
 const real_t *const SFEM_RESTRICT a
){
return a[0*stride]*a[3*stride] - a[1*stride]*a[2*stride];
}


SFEM_DEVICE_FUNCTION void Tri3_mk_jacobian(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t py0,
const real_t py1,
const real_t py2,
const count_t stride_jacobian,
real_t *const SFEM_RESTRICT jacobian
)
{
jacobian[0*stride_jacobian] = -px0 + px1;
jacobian[1*stride_jacobian] = -px0 + px2;
jacobian[2*stride_jacobian] = -py0 + py1;
jacobian[3*stride_jacobian] = -py0 + py2;
}

SFEM_DEVICE_FUNCTION void Tri3_mk_jacobian_inverse(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t py0,
const real_t py1,
const real_t py2,
const count_t stride_jacobian_inverse,
real_t *const SFEM_RESTRICT jacobian_inverse
)
{
const real_t x0 = -py0 + py2;
const real_t x1 = -px0 + px1;
const real_t x2 = px0 - px2;
const real_t x3 = py0 - py1;
const real_t x4 = 1.0/(x0*x1 - x2*x3);
jacobian_inverse[0*stride_jacobian_inverse] = x0*x4;
jacobian_inverse[1*stride_jacobian_inverse] = x2*x4;
jacobian_inverse[2*stride_jacobian_inverse] = x3*x4;
jacobian_inverse[3*stride_jacobian_inverse] = x1*x4;
}

SFEM_DEVICE_FUNCTION void Tri3_mk_jacobian_determinant_and_inverse(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t py0,
const real_t py1,
const real_t py2,
 //arrays
const count_t stride_jacobian_determinant,
real_t * const SFEM_RESTRICT jacobian_determinant,
const count_t stride_jacobian_inverse,
real_t *const SFEM_RESTRICT jacobian_inverse
)
{
const real_t x0 = -px0 + px1;
const real_t x1 = -py0 + py2;
const real_t x2 = px0 - px2;
const real_t x3 = py0 - py1;
const real_t x4 = x0*x1 - x2*x3;
const real_t x5 = 1.0/x4;
jacobian_determinant[0] = x4;
jacobian_inverse[0*stride_jacobian_inverse] = x1*x5;
jacobian_inverse[1*stride_jacobian_inverse] = x2*x5;
jacobian_inverse[2*stride_jacobian_inverse] = x3*x5;
jacobian_inverse[3*stride_jacobian_inverse] = x0*x5;
}

SFEM_DEVICE_FUNCTION void Tri3_mk_jacobian_determinant(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t py0,
const real_t py1,
const real_t py2,
 // arrays
const count_t stride_jacobian_determinant,
real_t * const SFEM_RESTRICT jacobian_determinant
)
{
jacobian_determinant[0] = (-px0 + px1)*(-py0 + py2) - (-px0 + px2)*(-py0 + py1);
}

SFEM_DEVICE_FUNCTION void Tri3_mk_fun(
const real_t qx,
const real_t qy,
 // arrays
const int stride_fun,
real_t * const SFEM_RESTRICT f
)
{
f[0*stride_fun] = -qx - qy + 1;
f[1*stride_fun] = qx;
f[2*stride_fun] = qy;
}

SFEM_DEVICE_FUNCTION real_t Tri3_mk_interpolate(
const real_t qx,
const real_t qy,
 // arrays
const int stride_coeff,
const real_t * const SFEM_RESTRICT c
)
{
return c[0*stride_coeff]*(-qx - qy + 1) + c[1*stride_coeff]*qx + c[2*stride_coeff]*qy;
}

SFEM_DEVICE_FUNCTION void Tri3_mk_grad_interpolate(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t *const SFEM_RESTRICT jacobian_inverse,
const int stride_coeff,
const real_t * const SFEM_RESTRICT c,
const int stride_grad,
real_t * const SFEM_RESTRICT grad
)
{
grad[0*stride_grad] = c[0*stride_coeff]*(-jacobian_inverse[0*stride_jacobian_inverse] - jacobian_inverse[2*stride_jacobian_inverse]) + c[1*stride_coeff]*jacobian_inverse[0*stride_jacobian_inverse] + c[2*stride_coeff]*jacobian_inverse[2*stride_jacobian_inverse];
grad[1*stride_grad] = c[0*stride_coeff]*(-jacobian_inverse[1*stride_jacobian_inverse] - jacobian_inverse[3*stride_jacobian_inverse]) + c[1*stride_coeff]*jacobian_inverse[1*stride_jacobian_inverse] + c[2*stride_coeff]*jacobian_inverse[3*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void Tri3_mk_partial_x(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gx
)
{
gx[0*stride_grad] = -jacobian_inverse[0*stride_jacobian_inverse] - jacobian_inverse[2*stride_jacobian_inverse];
gx[1*stride_grad] = jacobian_inverse[0*stride_jacobian_inverse];
gx[2*stride_grad] = jacobian_inverse[2*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void Tri3_mk_partial_y(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gy
)
{
gy[0*stride_grad] = -jacobian_inverse[1*stride_jacobian_inverse] - jacobian_inverse[3*stride_jacobian_inverse];
gy[1*stride_grad] = jacobian_inverse[1*stride_jacobian_inverse];
gy[2*stride_grad] = jacobian_inverse[3*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void Tri3_mk_partial_z(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gz
) 
{
//TODO

}
