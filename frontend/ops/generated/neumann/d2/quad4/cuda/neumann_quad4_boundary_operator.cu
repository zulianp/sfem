#include "../../../op/cuda/sfem_GeneratedNeumann_cuda_c_abi.hpp"

extern "C" int cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        void *const stream
);
extern "C" int cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const float t0,
        const float t1,
        const int out_stride,
        float *const RSTR out0,
        float *const RSTR out1,
        void *const stream
);

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        void *const stream
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t1, out_stride, out0, out1, stream);
}

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const float t0,
        const float t1,
        const int out_stride,
        float *const RSTR out0,
        float *const RSTR out1,
        void *const stream
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t1, out_stride, out0, out1, stream);
}
