#include "sfem_GeneratedVectorMass_c_abi.hpp"

extern "C" int vector_mass_proteus_hex27_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int vector_mass_proteus_hex27_jacobian_action_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR direction,
        void *const RSTR output
);
extern "C" int vector_mass_proteus_hex27_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_hex27_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        void *const RSTR output
);
extern "C" int vector_mass_proteus_hex27_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_hex27_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_hex27_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);
extern "C" const sfem::codegen::KernelDiagnostics * vector_mass_proteus_hex27_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * vector_mass_proteus_hex27_residual_esoa_diagnostics(
        void
);

extern "C" int vector_mass_hex27_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
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
    return vector_mass_proteus_hex27_hessian_bsr_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, rowptr, colidx, values);
}

extern "C" int vector_mass_hex27_jacobian_action_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR direction,
        void *const RSTR output
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
    return vector_mass_proteus_hex27_jacobian_action_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, direction, output);
}

extern "C" int vector_mass_hex27_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
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
    return vector_mass_proteus_hex27_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int vector_mass_hex27_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        void *const RSTR output
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
    return vector_mass_proteus_hex27_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, current, output);
}

extern "C" int vector_mass_hex27_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
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
    return vector_mass_proteus_hex27_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int vector_mass_hex27_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
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
    return vector_mass_proteus_hex27_jacobian_action_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int vector_mass_hex27_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
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
    return vector_mass_proteus_hex27_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out);
}

extern "C" const sfem::codegen::KernelDiagnostics * vector_mass_hex27_jacobian_action_esoa_diagnostics(
        void
) {
    return vector_mass_proteus_hex27_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * vector_mass_hex27_residual_esoa_diagnostics(
        void
) {
    return vector_mass_proteus_hex27_residual_esoa_diagnostics();
}
