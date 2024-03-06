#include "sfem_base.h"

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif 

static const int fe_spatial_dim = 2;
static const int fe_manifold_dim = 2;
static const int fe_n_nodes = 4;
static const char * fe_name = "Quad4";
static const int fe_n_nodes_for_jacobian = 2;
static const int fe_subparam_n_nodes = 4;


SFEM_DEVICE_FUNCTION void Quad4_mk_jacobian(
const real_t px0,
const real_t px2,
const real_t py0,
const real_t py2,
const count_t stride_jacobian,
real_t *jacobian
)
{
jacobian[0*stride_jacobian] = -px0 + px2;
jacobian[1*stride_jacobian] = 0;
jacobian[2*stride_jacobian] = 0;
jacobian[3*stride_jacobian] = -py0 + py2;
}

SFEM_DEVICE_FUNCTION void Quad4_mk_jacobian_inverse(
const real_t px0,
const real_t px2,
const real_t py0,
const real_t py2,
const count_t stride_jacobian_inverse,
real_t *jacobian_inverse
)
{
jacobian_inverse[0*stride_jacobian_inverse] = 1.0/(-px0 + px2);
jacobian_inverse[1*stride_jacobian_inverse] = 0;
jacobian_inverse[2*stride_jacobian_inverse] = 0;
jacobian_inverse[3*stride_jacobian_inverse] = 1.0/(-py0 + py2);
}

SFEM_DEVICE_FUNCTION void Quad4_mk_jacobian_determinant(
const real_t px0,
const real_t px2,
const real_t py0,
const real_t py2,
 // arrays
const count_t stride_jacobian_determinant,
real_t *jacobian_determinant
)
{
jacobian_determinant[0] = (-px0 + px2)*(-py0 + py2);
}

SFEM_DEVICE_FUNCTION void Quad4_mk_fun(
const real_t qx,
const real_t qy,
 // arrays
const int stride_fun,
real_t * SFEM_RESTRICT f
)
{
const real_t x0 = 1 - qx;
const real_t x1 = 1 - qy;
f[0*stride_fun] = x0*x1;
f[1*stride_fun] = qx*x1;
f[2*stride_fun] = qx*qy;
f[3*stride_fun] = qy*x0;
}

SFEM_DEVICE_FUNCTION void Quad4_mk_partial_x(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gx
)
{
const real_t x0 = qy - 1;
const real_t x1 = jacobian_inverse[0*stride_jacobian_inverse]*qy;
gx[0*stride_grad] = jacobian_inverse[0*stride_jacobian_inverse]*x0;
gx[1*stride_grad] = -jacobian_inverse[0*stride_jacobian_inverse]*x0;
gx[2*stride_grad] = x1;
gx[3*stride_grad] = -x1;
}

SFEM_DEVICE_FUNCTION void Quad4_mk_partial_y(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gy
)
{
const real_t x0 = qx - 1;
const real_t x1 = jacobian_inverse[3*stride_jacobian_inverse]*qx;
gy[0*stride_grad] = jacobian_inverse[3*stride_jacobian_inverse]*x0;
gy[1*stride_grad] = -x1;
gy[2*stride_grad] = x1;
gy[3*stride_grad] = -jacobian_inverse[3*stride_jacobian_inverse]*x0;
}

SFEM_DEVICE_FUNCTION void Quad4_mk_partial_z(
const real_t qx,
const real_t qy,
 // arrays
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gz
) 
{
//TODO

}
