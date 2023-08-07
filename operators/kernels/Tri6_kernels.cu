#include "sfem_base.h"

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE __device__ __host__
#else
#define SFEM_DEVICE_FUNCTION static SFEM_INLINE
#endif

static const int fe_spatial_dim = 2;
static const int fe_manifold_dim = 2;
static const int fe_n_nodes = 6;
static const char *fe_name = "Tri6";
static const int fe_n_nodes_for_jacobian = 3;
static const int fe_subparam_n_nodes = 3;

SFEM_DEVICE_FUNCTION void Tri6_mk_jacobian(const real_t px0,
                                           const real_t px1,
                                           const real_t px2,
                                           const real_t py0,
                                           const real_t py1,
                                           const real_t py2,
                                           const count_t stride_jacobian,
                                           real_t *jacobian) {
    jacobian[0 * stride_jacobian] = -px0 + px1;
    jacobian[1 * stride_jacobian] = -px0 + px2;
    jacobian[2 * stride_jacobian] = -py0 + py1;
    jacobian[3 * stride_jacobian] = -py0 + py2;
}

SFEM_DEVICE_FUNCTION void Tri6_mk_jacobian_inverse(const real_t px0,
                                                   const real_t px1,
                                                   const real_t px2,
                                                   const real_t py0,
                                                   const real_t py1,
                                                   const real_t py2,
                                                   const count_t stride_jacobian_inverse,
                                                   real_t *jacobian_inverse) {
    const real_t x0 = -py0 + py2;
    const real_t x1 = -px0 + px1;
    const real_t x2 = px0 - px2;
    const real_t x3 = py0 - py1;
    const real_t x4 = 1.0 / (x0 * x1 - x2 * x3);
    jacobian_inverse[0 * stride_jacobian_inverse] = x0 * x4;
    jacobian_inverse[1 * stride_jacobian_inverse] = x2 * x4;
    jacobian_inverse[2 * stride_jacobian_inverse] = x3 * x4;
    jacobian_inverse[3 * stride_jacobian_inverse] = x1 * x4;
}

SFEM_DEVICE_FUNCTION void Tri6_mk_jacobian_determinant(const real_t px0,
                                                       const real_t px1,
                                                       const real_t px2,
                                                       const real_t py0,
                                                       const real_t py1,
                                                       const real_t py2,
                                                       // arrays
                                                       const count_t stride_jacobian_determinant,
                                                       real_t *jacobian_determinant) {
    jacobian_determinant[0] = (-px0 + px1) * (-py0 + py2) - (-px0 + px2) * (-py0 + py1);
}

SFEM_DEVICE_FUNCTION void Tri6_mk_fun(const real_t qx,
                                      const real_t qy,
                                      // arrays
                                      const int stride_fun,
                                      real_t *SFEM_RESTRICT f) {
    const real_t x0 = 4 * qx * qy;
    const real_t x1 = pow(qx, 2);
    const real_t x2 = 2 * x1;
    const real_t x3 = pow(qy, 2);
    const real_t x4 = 2 * x3;
    f[0 * stride_fun] = -3 * qx - 3 * qy + x0 + x2 + x4 + 1;
    f[1 * stride_fun] = -qx + x2;
    f[2 * stride_fun] = -qy + x4;
    f[3 * stride_fun] = 4 * qx - x0 - 4 * x1;
    f[4 * stride_fun] = x0;
    f[5 * stride_fun] = 4 * qy - x0 - 4 * x3;
}

SFEM_DEVICE_FUNCTION void Tri6_mk_partial_x(const real_t qx,
                                            const real_t qy,
                                            // arrays
                                            const count_t stride_jacobian_inverse,
                                            const real_t *SFEM_RESTRICT jacobian_inverse,
                                            const count_t stride_grad,
                                            real_t *SFEM_RESTRICT gx) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = x0 + x1 - 3;
    const real_t x3 = jacobian_inverse[2 * stride_jacobian_inverse] * x0;
    const real_t x4 = jacobian_inverse[0 * stride_jacobian_inverse] * x1;
    gx[0 * stride_grad] = jacobian_inverse[0 * stride_jacobian_inverse] * x2 +
                          jacobian_inverse[2 * stride_jacobian_inverse] * x2;
    gx[1 * stride_grad] = jacobian_inverse[0 * stride_jacobian_inverse] * (x0 - 1);
    gx[2 * stride_grad] = jacobian_inverse[2 * stride_jacobian_inverse] * (x1 - 1);
    gx[3 * stride_grad] = jacobian_inverse[0 * stride_jacobian_inverse] * (-8 * qx - x1 + 4) - x3;
    gx[4 * stride_grad] = x3 + x4;
    gx[5 * stride_grad] = jacobian_inverse[2 * stride_jacobian_inverse] * (-8 * qy - x0 + 4) - x4;
}

SFEM_DEVICE_FUNCTION void Tri6_mk_partial_y(const real_t qx,
                                            const real_t qy,
                                            // arrays
                                            const count_t stride_jacobian_inverse,
                                            const real_t *SFEM_RESTRICT jacobian_inverse,
                                            const count_t stride_grad,
                                            real_t *SFEM_RESTRICT gy) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = x0 + x1 - 3;
    const real_t x3 = jacobian_inverse[3 * stride_jacobian_inverse] * x0;
    const real_t x4 = jacobian_inverse[1 * stride_jacobian_inverse] * x1;
    gy[0 * stride_grad] = jacobian_inverse[1 * stride_jacobian_inverse] * x2 +
                          jacobian_inverse[3 * stride_jacobian_inverse] * x2;
    gy[1 * stride_grad] = jacobian_inverse[1 * stride_jacobian_inverse] * (x0 - 1);
    gy[2 * stride_grad] = jacobian_inverse[3 * stride_jacobian_inverse] * (x1 - 1);
    gy[3 * stride_grad] = jacobian_inverse[1 * stride_jacobian_inverse] * (-8 * qx - x1 + 4) - x3;
    gy[4 * stride_grad] = x3 + x4;
    gy[5 * stride_grad] = jacobian_inverse[3 * stride_jacobian_inverse] * (-8 * qy - x0 + 4) - x4;
}

SFEM_DEVICE_FUNCTION void Tri6_mk_partial_z(const real_t qx,
                                            const real_t qy,
                                            // arrays
                                            const count_t stride_jacobian_inverse,
                                            const real_t *SFEM_RESTRICT jacobian_inverse,
                                            const count_t stride_grad,
                                            real_t *SFEM_RESTRICT gz) {
    // TODO
}
