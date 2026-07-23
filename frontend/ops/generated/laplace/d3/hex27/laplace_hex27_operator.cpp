#include "../../op/sfem_GeneratedLaplace_c_abi.hpp"

extern "C" void laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" double laplace_proteus_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" void laplace_proteus_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" double laplace_proteus_hex27_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_hex27_jacobian_u_u_diagnostics(
        void
);
extern "C" void laplace_proteus_hex27_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" double laplace_proteus_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_hex27_residual_element_soa_diagnostics(
        void
);
extern "C" void laplace_proteus_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" void laplace_proteus_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
);
extern "C" int laplace_proteus_hex27_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_packed_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_packed_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" void laplace_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double laplace_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return laplace_proteus_hex27_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_hex27_jacobian_action_element_soa_diagnostics(
        void
) {
    return laplace_proteus_hex27_jacobian_action_element_soa_diagnostics();
}

extern "C" void laplace_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double laplace_hex27_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return laplace_proteus_hex27_jacobian_u_u_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_hex27_jacobian_u_u_diagnostics(
        void
) {
    return laplace_proteus_hex27_jacobian_u_u_diagnostics();
}

extern "C" void laplace_hex27_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_u_u_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_jacobian_u_u_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double laplace_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return laplace_proteus_hex27_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_hex27_residual_element_soa_diagnostics(
        void
) {
    return laplace_proteus_hex27_residual_element_soa_diagnostics();
}

extern "C" void laplace_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void laplace_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    laplace_proteus_hex27_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int laplace_hex27_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
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
    return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, direction, output);
}

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
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
    return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, direction, output);
}

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    uint16_t *proteus_elements[27] = {
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
    return laplace_proteus_hex27_jacobian_action_packed_affine_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    uint16_t *proteus_elements[27] = {
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
    return laplace_proteus_hex27_jacobian_action_packed_affine_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_packed_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    uint16_t *proteus_elements[27] = {
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
    return laplace_proteus_hex27_jacobian_action_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_packed_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    uint16_t *proteus_elements[27] = {
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
    return laplace_proteus_hex27_jacobian_action_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_hex27_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
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
    return laplace_proteus_hex27_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, output);
}

extern "C" int laplace_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
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
    return laplace_proteus_hex27_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, output);
}

extern "C" int laplace_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
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
    return laplace_proteus_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, kappa, current_stride, u, out_stride, u_out);
}
