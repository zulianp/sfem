#include "../../op/sfem_GeneratedNeoHookeanOgden_c_abi.hpp"

extern "C" int neohookean_ogden_proteus_quad4_inexact_apply_compressed_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const RSTR tangent,
        const scaling_t *const RSTR scaling,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);
extern "C" int neohookean_ogden_proteus_quad4_inexact_apply_stored_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);
extern "C" int neohookean_ogden_proteus_quad4_inexact_apply_tangent_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const RSTR tangent
);

extern "C" int neohookean_ogden_quad4_inexact_apply_compressed_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const RSTR tangent,
        const scaling_t *const RSTR scaling,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return neohookean_ogden_proteus_quad4_inexact_apply_compressed_a_msoa(scalar_bytes, nelements, proteus_elements, tangent_component_stride, tangent, scaling, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_quad4_inexact_apply_stored_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return neohookean_ogden_proteus_quad4_inexact_apply_stored_a_msoa(scalar_bytes, nelements, proteus_elements, tangent_component_stride, tangent, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_quad4_inexact_apply_tangent_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const RSTR tangent
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return neohookean_ogden_proteus_quad4_inexact_apply_tangent_a_msoa(scalar_bytes, nelements, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, ux, uy, tangent_component_stride, tangent);
}
