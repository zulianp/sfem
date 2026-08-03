#include "../../op/sfem_GeneratedNeumannGeneral_c_abi.hpp"

extern "C" int neumann_general_proteus_quad4_edgeshell2_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_010,
        const real_t t1_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1
);
extern "C" int neumann_general_proteus_quad4_edgeshell2_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_010,
        const float t1_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1
);

extern "C" int neumann_general_quad4_edgeshell2_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_010,
        const real_t t1_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return neumann_general_proteus_quad4_edgeshell2_boundary_residual_sideset_soa(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t0_010, t0_100, t1, t1_010, t1_100, out_stride, out0, out1);
}

extern "C" int neumann_general_quad4_edgeshell2_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_010,
        const float t1_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return neumann_general_proteus_quad4_edgeshell2_boundary_residual_sideset_soa_float(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t0_010, t0_100, t1, t1_010, t1_100, out_stride, out0, out1);
}
