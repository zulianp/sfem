#include "sfem_base.h"

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif 

static const int fe_spatial_dim = 3;
static const int fe_manifold_dim = 3;
static const int fe_n_nodes = 10;
static const char * fe_name = "Tet10";
static const int fe_n_nodes_for_jacobian = 4;
static const int fe_subparam_n_nodes = 4;


SFEM_DEVICE_FUNCTION void Tet10_mk_jacobian(
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
real_t *jacobian
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

SFEM_DEVICE_FUNCTION void Tet10_mk_jacobian_inverse(
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
real_t *jacobian_inverse
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

SFEM_DEVICE_FUNCTION void Tet10_mk_jacobian_determinant(
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
real_t *jacobian_determinant
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

SFEM_DEVICE_FUNCTION void Tet10_mk_fun(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const int stride_fun,
real_t * SFEM_RESTRICT f
)
{
const real_t x0 = 2*qy;
const real_t x1 = 2*qz;
const real_t x2 = 2*qx - 1;
const real_t x3 = 4*qx;
const real_t x4 = 4*qy;
const real_t x5 = -4*qz - x3 - x4 + 4;
f[0*stride_fun] = (-x0 - x1 - x2)*(-qx - qy - qz + 1);
f[1*stride_fun] = qx*x2;
f[2*stride_fun] = qy*(x0 - 1);
f[3*stride_fun] = qz*(x1 - 1);
f[4*stride_fun] = qx*x5;
f[5*stride_fun] = qy*x3;
f[6*stride_fun] = qy*x5;
f[7*stride_fun] = qz*x5;
f[8*stride_fun] = qz*x3;
f[9*stride_fun] = qz*x4;
}

SFEM_DEVICE_FUNCTION void Tet10_mk_partial_x(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gx
)
{
const real_t x0 = 4*qx;
const real_t x1 = 4*qy;
const real_t x2 = 4*qz;
const real_t x3 = x1 + x2;
const real_t x4 = x0 + x3 - 3;
const real_t x5 = jacobian_inverse[3*stride_jacobian_inverse]*x0;
const real_t x6 = jacobian_inverse[6*stride_jacobian_inverse]*x0;
const real_t x7 = jacobian_inverse[0*stride_jacobian_inverse]*x1;
const real_t x8 = jacobian_inverse[6*stride_jacobian_inverse]*x1;
const real_t x9 = x0 - 4;
const real_t x10 = jacobian_inverse[0*stride_jacobian_inverse]*x2;
const real_t x11 = jacobian_inverse[3*stride_jacobian_inverse]*x2;
gx[0*stride_grad] = jacobian_inverse[0*stride_jacobian_inverse]*x4 + jacobian_inverse[3*stride_jacobian_inverse]*x4 + jacobian_inverse[6*stride_jacobian_inverse]*x4;
gx[1*stride_grad] = jacobian_inverse[0*stride_jacobian_inverse]*(x0 - 1);
gx[2*stride_grad] = jacobian_inverse[3*stride_jacobian_inverse]*(x1 - 1);
gx[3*stride_grad] = jacobian_inverse[6*stride_jacobian_inverse]*(x2 - 1);
gx[4*stride_grad] = jacobian_inverse[0*stride_jacobian_inverse]*(-8*qx - x3 + 4) - x5 - x6;
gx[5*stride_grad] = x5 + x7;
gx[6*stride_grad] = jacobian_inverse[3*stride_jacobian_inverse]*(-8*qy - x2 - x9) - x7 - x8;
gx[7*stride_grad] = jacobian_inverse[6*stride_jacobian_inverse]*(-8*qz - x1 - x9) - x10 - x11;
gx[8*stride_grad] = x10 + x6;
gx[9*stride_grad] = x11 + x8;
}

SFEM_DEVICE_FUNCTION void Tet10_mk_partial_y(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gy
)
{
const real_t x0 = 4*qx;
const real_t x1 = 4*qy;
const real_t x2 = 4*qz;
const real_t x3 = x1 + x2;
const real_t x4 = x0 + x3 - 3;
const real_t x5 = jacobian_inverse[4*stride_jacobian_inverse]*x0;
const real_t x6 = jacobian_inverse[7*stride_jacobian_inverse]*x0;
const real_t x7 = jacobian_inverse[1*stride_jacobian_inverse]*x1;
const real_t x8 = jacobian_inverse[7*stride_jacobian_inverse]*x1;
const real_t x9 = x0 - 4;
const real_t x10 = jacobian_inverse[1*stride_jacobian_inverse]*x2;
const real_t x11 = jacobian_inverse[4*stride_jacobian_inverse]*x2;
gy[0*stride_grad] = jacobian_inverse[1*stride_jacobian_inverse]*x4 + jacobian_inverse[4*stride_jacobian_inverse]*x4 + jacobian_inverse[7*stride_jacobian_inverse]*x4;
gy[1*stride_grad] = jacobian_inverse[1*stride_jacobian_inverse]*(x0 - 1);
gy[2*stride_grad] = jacobian_inverse[4*stride_jacobian_inverse]*(x1 - 1);
gy[3*stride_grad] = jacobian_inverse[7*stride_jacobian_inverse]*(x2 - 1);
gy[4*stride_grad] = jacobian_inverse[1*stride_jacobian_inverse]*(-8*qx - x3 + 4) - x5 - x6;
gy[5*stride_grad] = x5 + x7;
gy[6*stride_grad] = jacobian_inverse[4*stride_jacobian_inverse]*(-8*qy - x2 - x9) - x7 - x8;
gy[7*stride_grad] = jacobian_inverse[7*stride_jacobian_inverse]*(-8*qz - x1 - x9) - x10 - x11;
gy[8*stride_grad] = x10 + x6;
gy[9*stride_grad] = x11 + x8;
}

SFEM_DEVICE_FUNCTION void Tet10_mk_partial_z(
const real_t qx,
const real_t qy,
const real_t qz,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gz
) 
{
const real_t x0 = 4*qx;
const real_t x1 = 4*qy;
const real_t x2 = 4*qz;
const real_t x3 = x1 + x2;
const real_t x4 = x0 + x3 - 3;
const real_t x5 = jacobian_inverse[5*stride_jacobian_inverse]*x0;
const real_t x6 = jacobian_inverse[8*stride_jacobian_inverse]*x0;
const real_t x7 = jacobian_inverse[2*stride_jacobian_inverse]*x1;
const real_t x8 = jacobian_inverse[8*stride_jacobian_inverse]*x1;
const real_t x9 = x0 - 4;
const real_t x10 = jacobian_inverse[2*stride_jacobian_inverse]*x2;
const real_t x11 = jacobian_inverse[5*stride_jacobian_inverse]*x2;
gz[0*stride_grad] = jacobian_inverse[2*stride_jacobian_inverse]*x4 + jacobian_inverse[5*stride_jacobian_inverse]*x4 + jacobian_inverse[8*stride_jacobian_inverse]*x4;
gz[1*stride_grad] = jacobian_inverse[2*stride_jacobian_inverse]*(x0 - 1);
gz[2*stride_grad] = jacobian_inverse[5*stride_jacobian_inverse]*(x1 - 1);
gz[3*stride_grad] = jacobian_inverse[8*stride_jacobian_inverse]*(x2 - 1);
gz[4*stride_grad] = jacobian_inverse[2*stride_jacobian_inverse]*(-8*qx - x3 + 4) - x5 - x6;
gz[5*stride_grad] = x5 + x7;
gz[6*stride_grad] = jacobian_inverse[5*stride_jacobian_inverse]*(-8*qy - x2 - x9) - x7 - x8;
gz[7*stride_grad] = jacobian_inverse[8*stride_jacobian_inverse]*(-8*qz - x1 - x9) - x10 - x11;
gz[8*stride_grad] = x10 + x6;
gz[9*stride_grad] = x11 + x8;
}
