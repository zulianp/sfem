#include "sfem_base.h"


#include <math.h>

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif 

static const int fe_spatial_dim = 3;
static const int fe_manifold_dim = 3;
static const int fe_n_nodes = 4;
static const char * fe_name = "Tet4";
static const int fe_n_nodes_for_jacobian = 4;
static const int fe_subparam_n_nodes = 4;
static const float fe_reference_measure = 1.0/6.0;


static SFEM_CUDA_INLINE real_t Tet4_mk_det_3(const count_t stride,
 const real_t *const SFEM_RESTRICT a
){
return a[0*stride]*a[4*stride]*a[8*stride] - a[0*stride]*a[5*stride]*a[7*stride] - a[1*stride]*a[3*stride]*a[8*stride] + a[1*stride]*a[5*stride]*a[6*stride] + a[2*stride]*a[3*stride]*a[7*stride] - a[2*stride]*a[4*stride]*a[6*stride];
}


SFEM_DEVICE_FUNCTION void Tet4_mk_jacobian(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t px3,
const real_t py0,
const real_t py1,
const real_t py2,
const real_t py3,
const real_t pz0,
const real_t pz1,
const real_t pz2,
const real_t pz3,
const count_t stride_jacobian,
real_t *const SFEM_RESTRICT jacobian
)
{
jacobian[0*stride_jacobian] = -px0 + px1;
jacobian[1*stride_jacobian] = -px0 + px2;
jacobian[2*stride_jacobian] = -px0 + px3;
jacobian[3*stride_jacobian] = -py0 + py1;
jacobian[4*stride_jacobian] = -py0 + py2;
jacobian[5*stride_jacobian] = -py0 + py3;
jacobian[6*stride_jacobian] = -pz0 + pz1;
jacobian[7*stride_jacobian] = -pz0 + pz2;
jacobian[8*stride_jacobian] = -pz0 + pz3;
}

SFEM_DEVICE_FUNCTION void Tet4_mk_jacobian_inverse(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t px3,
const real_t py0,
const real_t py1,
const real_t py2,
const real_t py3,
const real_t pz0,
const real_t pz1,
const real_t pz2,
const real_t pz3,
const count_t stride_jacobian_inverse,
real_t *const SFEM_RESTRICT jacobian_inverse
)
{
const real_t x0 = -py0 + py2;
const real_t x1 = -pz0 + pz3;
const real_t x2 = x0*x1;
const real_t x3 = -py0 + py3;
const real_t x4 = -pz0 + pz2;
const real_t x5 = x3*x4;
const real_t x6 = -px0 + px1;
const real_t x7 = -pz0 + pz1;
const real_t x8 = -px0 + px2;
const real_t x9 = x3*x8;
const real_t x10 = -py0 + py1;
const real_t x11 = -px0 + px3;
const real_t x12 = x1*x8;
const real_t x13 = x0*x11;
const real_t x14 = 1.0/(x10*x11*x4 - x10*x12 - x13*x7 + x2*x6 - x5*x6 + x7*x9);
jacobian_inverse[0*stride_jacobian_inverse] = x14*(x2 - x5);
jacobian_inverse[1*stride_jacobian_inverse] = x14*(x11*x4 - x12);
jacobian_inverse[2*stride_jacobian_inverse] = x14*(-x13 + x9);
jacobian_inverse[3*stride_jacobian_inverse] = x14*(-x1*x10 + x3*x7);
jacobian_inverse[4*stride_jacobian_inverse] = x14*(x1*x6 - x11*x7);
jacobian_inverse[5*stride_jacobian_inverse] = x14*(x10*x11 - x3*x6);
jacobian_inverse[6*stride_jacobian_inverse] = x14*(-x0*x7 + x10*x4);
jacobian_inverse[7*stride_jacobian_inverse] = x14*(-x4*x6 + x7*x8);
jacobian_inverse[8*stride_jacobian_inverse] = x14*(x0*x6 - x10*x8);
}

SFEM_DEVICE_FUNCTION void Tet4_mk_jacobian_determinant_and_inverse(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t px3,
const real_t py0,
const real_t py1,
const real_t py2,
const real_t py3,
const real_t pz0,
const real_t pz1,
const real_t pz2,
const real_t pz3,
 //arrays
const count_t stride_jacobian_determinant,
real_t * const SFEM_RESTRICT jacobian_determinant,
const count_t stride_jacobian_inverse,
real_t *const SFEM_RESTRICT jacobian_inverse
)
{
const real_t x0 = -px0 + px1;
const real_t x1 = -py0 + py2;
const real_t x2 = -pz0 + pz3;
const real_t x3 = x1*x2;
const real_t x4 = -pz0 + pz1;
const real_t x5 = -px0 + px2;
const real_t x6 = -py0 + py3;
const real_t x7 = x5*x6;
const real_t x8 = -py0 + py1;
const real_t x9 = -px0 + px3;
const real_t x10 = -pz0 + pz2;
const real_t x11 = x10*x6;
const real_t x12 = x2*x5;
const real_t x13 = x1*x9;
const real_t x14 = -x0*x11 + x0*x3 + x10*x8*x9 - x12*x8 - x13*x4 + x4*x7;
const real_t x15 = 1.0/x14;
jacobian_determinant[0] = x14;
jacobian_inverse[0*stride_jacobian_inverse] = x15*(-x11 + x3);
jacobian_inverse[1*stride_jacobian_inverse] = x15*(x10*x9 - x12);
jacobian_inverse[2*stride_jacobian_inverse] = x15*(-x13 + x7);
jacobian_inverse[3*stride_jacobian_inverse] = x15*(-x2*x8 + x4*x6);
jacobian_inverse[4*stride_jacobian_inverse] = x15*(x0*x2 - x4*x9);
jacobian_inverse[5*stride_jacobian_inverse] = x15*(-x0*x6 + x8*x9);
jacobian_inverse[6*stride_jacobian_inverse] = x15*(-x1*x4 + x10*x8);
jacobian_inverse[7*stride_jacobian_inverse] = x15*(-x0*x10 + x4*x5);
jacobian_inverse[8*stride_jacobian_inverse] = x15*(x0*x1 - x5*x8);
}

SFEM_DEVICE_FUNCTION void Tet4_mk_jacobian_determinant(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t px3,
const real_t py0,
const real_t py1,
const real_t py2,
const real_t py3,
const real_t pz0,
const real_t pz1,
const real_t pz2,
const real_t pz3,
 // arrays
const count_t stride_jacobian_determinant,
real_t * const SFEM_RESTRICT jacobian_determinant
)
{
const real_t x0 = -px0 + px1;
const real_t x1 = -py0 + py2;
const real_t x2 = -pz0 + pz3;
const real_t x3 = -px0 + px2;
const real_t x4 = -py0 + py3;
const real_t x5 = -pz0 + pz1;
const real_t x6 = -px0 + px3;
const real_t x7 = -py0 + py1;
const real_t x8 = -pz0 + pz2;
jacobian_determinant[0] = x0*x1*x2 - x0*x4*x8 - x1*x5*x6 - x2*x3*x7 + x3*x4*x5 + x6*x7*x8;
}

SFEM_DEVICE_FUNCTION void Tet4_mk_fun(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const int stride_fun,
real_t * const SFEM_RESTRICT f
)
{
f[0*stride_fun] = -qx - qy - qz + 1;
f[1*stride_fun] = qx;
f[2*stride_fun] = qy;
f[3*stride_fun] = qz;
}

SFEM_DEVICE_FUNCTION real_t Tet4_mk_interpolate(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const int stride_coeff,
const real_t * const SFEM_RESTRICT c
)
{
return c[0*stride_coeff]*(-qx - qy - qz + 1) + c[1*stride_coeff]*qx + c[2*stride_coeff]*qy + c[3*stride_coeff]*qz;
}

SFEM_DEVICE_FUNCTION void Tet4_mk_grad_interpolate(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const count_t stride_jacobian_inverse,
const real_t *const SFEM_RESTRICT jacobian_inverse,
const int stride_coeff,
const real_t * const SFEM_RESTRICT c,
const int stride_grad,
real_t * const SFEM_RESTRICT grad
)
{
grad[0*stride_grad] = c[0*stride_coeff]*(-jacobian_inverse[0*stride_jacobian_inverse] - jacobian_inverse[3*stride_jacobian_inverse] - jacobian_inverse[6*stride_jacobian_inverse]) + c[1*stride_coeff]*jacobian_inverse[0*stride_jacobian_inverse] + c[2*stride_coeff]*jacobian_inverse[3*stride_jacobian_inverse] + c[3*stride_coeff]*jacobian_inverse[6*stride_jacobian_inverse];
grad[1*stride_grad] = c[0*stride_coeff]*(-jacobian_inverse[1*stride_jacobian_inverse] - jacobian_inverse[4*stride_jacobian_inverse] - jacobian_inverse[7*stride_jacobian_inverse]) + c[1*stride_coeff]*jacobian_inverse[1*stride_jacobian_inverse] + c[2*stride_coeff]*jacobian_inverse[4*stride_jacobian_inverse] + c[3*stride_coeff]*jacobian_inverse[7*stride_jacobian_inverse];
grad[2*stride_grad] = c[0*stride_coeff]*(-jacobian_inverse[2*stride_jacobian_inverse] - jacobian_inverse[5*stride_jacobian_inverse] - jacobian_inverse[8*stride_jacobian_inverse]) + c[1*stride_coeff]*jacobian_inverse[2*stride_jacobian_inverse] + c[2*stride_coeff]*jacobian_inverse[5*stride_jacobian_inverse] + c[3*stride_coeff]*jacobian_inverse[8*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void Tet4_mk_partial_x(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gx
)
{
gx[0*stride_grad] = -jacobian_inverse[0*stride_jacobian_inverse] - jacobian_inverse[3*stride_jacobian_inverse] - jacobian_inverse[6*stride_jacobian_inverse];
gx[1*stride_grad] = jacobian_inverse[0*stride_jacobian_inverse];
gx[2*stride_grad] = jacobian_inverse[3*stride_jacobian_inverse];
gx[3*stride_grad] = jacobian_inverse[6*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void Tet4_mk_partial_y(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gy
)
{
gy[0*stride_grad] = -jacobian_inverse[1*stride_jacobian_inverse] - jacobian_inverse[4*stride_jacobian_inverse] - jacobian_inverse[7*stride_jacobian_inverse];
gy[1*stride_grad] = jacobian_inverse[1*stride_jacobian_inverse];
gy[2*stride_grad] = jacobian_inverse[4*stride_jacobian_inverse];
gy[3*stride_grad] = jacobian_inverse[7*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void Tet4_mk_partial_z(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * const SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * const SFEM_RESTRICT gz
) 
{
gz[0*stride_grad] = -jacobian_inverse[2*stride_jacobian_inverse] - jacobian_inverse[5*stride_jacobian_inverse] - jacobian_inverse[8*stride_jacobian_inverse];
gz[1*stride_grad] = jacobian_inverse[2*stride_jacobian_inverse];
gz[2*stride_grad] = jacobian_inverse[5*stride_jacobian_inverse];
gz[3*stride_grad] = jacobian_inverse[8*stride_jacobian_inverse];
}
