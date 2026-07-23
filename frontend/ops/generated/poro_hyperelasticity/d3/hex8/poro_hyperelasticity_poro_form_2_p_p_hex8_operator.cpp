#include "../../op/sfem_GeneratedPoroHyperelasticity_c_abi.hpp"

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
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
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, dt, hydraulic_conductivity, storage, out_stride, rows, cols, values);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
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
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, dt, hydraulic_conductivity, storage, out_stride, rows, cols, values);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa(
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
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_out
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
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_out
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
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_out
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
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_out
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
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * poro_hyperelasticity_poro_form_2_p_p_hex8_matrix_assembly_variant(
        const int variant
) {
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_matrix_assembly_variant(variant);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_matrix_assembly_variant_count(
        void
) {
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_matrix_assembly_variant_count();
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics(
        void
) {
    return poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_element_soa_diagnostics();
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    poro_hyperelasticity_poro_form_2_p_p_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}
