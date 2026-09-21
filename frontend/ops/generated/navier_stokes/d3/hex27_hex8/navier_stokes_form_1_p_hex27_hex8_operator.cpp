#include "../../op/sfem_GeneratedNavierStokes_c_abi.hpp"

extern "C" int navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out
);
extern "C" int navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
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
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out
);
extern "C" const sfem::codegen::KernelDiagnostics * navier_stokes_form_1_p_proteus_hex27_proteus_hex8_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_esoa_diagnostics(
        void
);

extern "C" int navier_stokes_form_1_p_hex27_hex8_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out
) {
    idx_t *proteus_elements[27] = {
        elements[0],
        elements[8],
        elements[1],
        elements[11],
        elements[24],
        elements[9],
        elements[3],
        elements[10],
        elements[2],
        elements[16],
        elements[20],
        elements[17],
        elements[23],
        elements[26],
        elements[21],
        elements[19],
        elements[22],
        elements[18],
        elements[4],
        elements[12],
        elements[5],
        elements[15],
        elements[25],
        elements[13],
        elements[7],
        elements[14],
        elements[6]
    };
    return navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_1_p_hex27_hex8_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
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
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out
) {
    idx_t *proteus_elements[27] = {
        elements[0],
        elements[8],
        elements[1],
        elements[11],
        elements[24],
        elements[9],
        elements[3],
        elements[10],
        elements[2],
        elements[16],
        elements[20],
        elements[17],
        elements[23],
        elements[26],
        elements[21],
        elements[19],
        elements[22],
        elements[18],
        elements[4],
        elements[12],
        elements[5],
        elements[15],
        elements[25],
        elements[13],
        elements[7],
        elements[14],
        elements[6]
    };
    return navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

extern "C" const sfem::codegen::KernelDiagnostics * navier_stokes_form_1_p_hex27_hex8_jacobian_action_esoa_diagnostics(
        void
) {
    return navier_stokes_form_1_p_proteus_hex27_proteus_hex8_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * navier_stokes_form_1_p_hex27_hex8_residual_esoa_diagnostics(
        void
) {
    return navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_esoa_diagnostics();
}
