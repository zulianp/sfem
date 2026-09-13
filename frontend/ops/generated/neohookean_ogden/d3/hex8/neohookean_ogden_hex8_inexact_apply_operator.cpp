#include "../../op/sfem_GeneratedNeoHookeanOgden_c_abi.hpp"

extern "C" int neohookean_ogden_proteus_hex8_inexact_apply_compressed_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const RSTR tangent,
        const scaling_t *const RSTR scaling,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);
extern "C" int neohookean_ogden_proteus_hex8_inexact_apply_stored_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);
extern "C" int neohookean_ogden_proteus_hex8_inexact_apply_tangent_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const RSTR tangent
);
extern "C" int neohookean_ogden_proteus_hex8_inexact_apply_stored_packed_two_pass_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        void *const RSTR ghost_buf,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int neohookean_ogden_hex8_inexact_apply_compressed_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const RSTR tangent,
        const scaling_t *const RSTR scaling,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return neohookean_ogden_proteus_hex8_inexact_apply_compressed_a_msoa(scalar_bytes, nelements, proteus_elements, tangent_component_stride, tangent, scaling, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_stored_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return neohookean_ogden_proteus_hex8_inexact_apply_stored_a_msoa(scalar_bytes, nelements, proteus_elements, tangent_component_stride, tangent, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_tangent_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const RSTR tangent
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return neohookean_ogden_proteus_hex8_inexact_apply_tangent_a_msoa(scalar_bytes, nelements, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, tangent_component_stride, tangent);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_stored_packed_two_pass_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        void *const RSTR ghost_buf,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return neohookean_ogden_proteus_hex8_inexact_apply_stored_packed_two_pass_a_msoa(scalar_bytes, n_packs, n_elements_per_pack, nelements, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_ghost_entries, n_ghost_reduce_rows, ghost_ptr, ghost_idx, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, tangent_component_stride, tangent, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}
