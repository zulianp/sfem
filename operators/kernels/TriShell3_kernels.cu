#include "sfem_base.h"

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif 

static const int fe_spatial_dim = 3;
static const int fe_manifold_dim = 2;
static const int fe_n_nodes = 3;
static const char * fe_name = "TriShell3";
static const int fe_n_nodes_for_jacobian = 3;
static const int fe_subparam_n_nodes = 3;


SFEM_DEVICE_FUNCTION void TriShell3_mk_jacobian(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t py0,
const real_t py1,
const real_t py2,
const real_t pz0,
const real_t pz1,
const real_t pz2,
const count_t stride_jacobian,
real_t *jacobian
)
{
jacobian[0*stride_jacobian] = -px0 + px1;
jacobian[1*stride_jacobian] = -px0 + px2;
jacobian[2*stride_jacobian] = -py0 + py1;
jacobian[3*stride_jacobian] = -py0 + py2;
jacobian[4*stride_jacobian] = -pz0 + pz1;
jacobian[5*stride_jacobian] = -pz0 + pz2;
}

SFEM_DEVICE_FUNCTION void TriShell3_mk_jacobian_inverse(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t py0,
const real_t py1,
const real_t py2,
const real_t pz0,
const real_t pz1,
const real_t pz2,
const count_t stride_jacobian_inverse,
real_t *jacobian_inverse
)
{
const real_t x0 = -px0 + px1;
const real_t x1 = -px0 + px2;
const real_t x2 = -py0 + py2;
const real_t x3 = -pz0 + pz2;
const real_t x4 = pow(x1, 2) + pow(x2, 2) + pow(x3, 2);
const real_t x5 = -py0 + py1;
const real_t x6 = -pz0 + pz1;
const real_t x7 = x0*x1 + x2*x5 + x3*x6;
const real_t x8 = pow(x0, 2) + pow(x5, 2) + pow(x6, 2);
const real_t x9 = 1.0/(x4*x8 - pow(x7, 2));
const real_t x10 = x4*x9;
const real_t x11 = -x7*x9;
const real_t x12 = x8*x9;
jacobian_inverse[0*stride_jacobian_inverse] = x0*x10 + x1*x11;
jacobian_inverse[1*stride_jacobian_inverse] = x10*x5 + x11*x2;
jacobian_inverse[2*stride_jacobian_inverse] = x10*x6 + x11*x3;
jacobian_inverse[3*stride_jacobian_inverse] = x0*x11 + x1*x12;
jacobian_inverse[4*stride_jacobian_inverse] = x11*x5 + x12*x2;
jacobian_inverse[5*stride_jacobian_inverse] = x11*x6 + x12*x3;
}

SFEM_DEVICE_FUNCTION void TriShell3_mk_jacobian_determinant(
const real_t px0,
const real_t px1,
const real_t px2,
const real_t py0,
const real_t py1,
const real_t py2,
const real_t pz0,
const real_t pz1,
const real_t pz2,
 // arrays
const count_t stride_jacobian_determinant,
real_t *jacobian_determinant
)
{
const real_t x0 = -px0 + px1;
const real_t x1 = -px0 + px2;
const real_t x2 = -py0 + py1;
const real_t x3 = -py0 + py2;
const real_t x4 = -pz0 + pz1;
const real_t x5 = -pz0 + pz2;
jacobian_determinant[0] = sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2))*(pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) - pow(x0*x1 + x2*x3 + x4*x5, 2));
}

SFEM_DEVICE_FUNCTION void TriShell3_mk_fun(
const real_t qx,
const real_t qy,
 // arrays
const int stride_fun,
real_t * SFEM_RESTRICT f
)
{
f[0*stride_fun] = -qx - qy + 1;
f[1*stride_fun] = qx;
f[2*stride_fun] = qy;
}

SFEM_DEVICE_FUNCTION void TriShell3_mk_partial_x(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gx
)
{
gx[0*stride_grad] = -jacobian_inverse[0*stride_jacobian_inverse] - jacobian_inverse[3*stride_jacobian_inverse];
gx[1*stride_grad] = jacobian_inverse[0*stride_jacobian_inverse];
gx[2*stride_grad] = jacobian_inverse[3*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void TriShell3_mk_partial_y(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gy
)
{
gy[0*stride_grad] = -jacobian_inverse[1*stride_jacobian_inverse] - jacobian_inverse[4*stride_jacobian_inverse];
gy[1*stride_grad] = jacobian_inverse[1*stride_jacobian_inverse];
gy[2*stride_grad] = jacobian_inverse[4*stride_jacobian_inverse];
}

SFEM_DEVICE_FUNCTION void TriShell3_mk_partial_z(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gz
) 
{
gz[0*stride_grad] = -jacobian_inverse[2*stride_jacobian_inverse] - jacobian_inverse[5*stride_jacobian_inverse];
gz[1*stride_grad] = jacobian_inverse[2*stride_jacobian_inverse];
gz[2*stride_grad] = jacobian_inverse[5*stride_jacobian_inverse];
}
