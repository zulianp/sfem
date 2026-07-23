#include "../../op/sfem_GeneratedTwoPhaseFlow_c_abi.hpp"

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa(nelems, geometry_stride, determinant, output);
}

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_float(nelems, geometry_stride, determinant, output);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    two_phase_flow_form_1_p_c_proteus_quad4_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * two_phase_flow_form_1_p_c_quad4_matrix_assembly_variant(
        const int variant
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_matrix_assembly_variant(variant);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_matrix_assembly_variant_count(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_matrix_assembly_variant_count();
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT previous[8],
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa(nelems, geometry_stride, determinant, adjugate, current, previous, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
}

extern "C" double two_phase_flow_form_1_p_c_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_quad4_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT previous[8],
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, previous, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa(nelems, geometry_stride, determinant, output);
}

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_float(nelems, geometry_stride, determinant, output);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    two_phase_flow_form_1_p_w_proteus_quad4_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * two_phase_flow_form_1_p_w_quad4_matrix_assembly_variant(
        const int variant
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_matrix_assembly_variant(variant);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_matrix_assembly_variant_count(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_matrix_assembly_variant_count();
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT previous[8],
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa(nelems, geometry_stride, determinant, adjugate, current, previous, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" double two_phase_flow_form_1_p_w_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_quad4_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT previous[8],
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, previous, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT direction[8],
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa(nelems, geometry_stride, determinant, adjugate, current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
}

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT direction[8],
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * two_phase_flow_form_2_p_c_p_c_quad4_matrix_assembly_variant(
        const int variant
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_matrix_assembly_variant(variant);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_matrix_assembly_variant_count(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_matrix_assembly_variant_count();
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa(nelems, geometry_stride, determinant, output);
}

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_float(nelems, geometry_stride, determinant, output);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT direction[8],
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa(nelems, geometry_stride, determinant, adjugate, current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
}

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT direction[8],
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_matrix_assembly_variant(
        const int variant
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_matrix_assembly_variant(variant);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_matrix_assembly_variant_count(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_matrix_assembly_variant_count();
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa(nelems, geometry_stride, determinant, output);
}

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_float(nelems, geometry_stride, determinant, output);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT direction[8],
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa(nelems, geometry_stride, determinant, adjugate, current, direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT direction[8],
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * two_phase_flow_form_2_p_w_p_c_quad4_matrix_assembly_variant(
        const int variant
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_matrix_assembly_variant(variant);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_matrix_assembly_variant_count(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_matrix_assembly_variant_count();
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa(nelems, geometry_stride, determinant, output);
}

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_float(nelems, geometry_stride, determinant, output);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT direction[8],
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa(nelems, geometry_stride, determinant, adjugate, current, direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT direction[8],
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics();
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * two_phase_flow_form_2_p_w_p_w_quad4_matrix_assembly_variant(
        const int variant
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_matrix_assembly_variant(variant);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_matrix_assembly_variant_count(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_matrix_assembly_variant_count();
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa(nelems, geometry_stride, determinant, output);
}

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_diagnostics();
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_float(nelems, geometry_stride, determinant, output);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_quad4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_quad4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_quad4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT direction[8],
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_proteus_quad4_jacobian_action_element_soa(nelems, geometry_stride, determinant, adjugate, current, direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" double two_phase_flow_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_quad4_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_proteus_quad4_jacobian_action_element_soa_diagnostics();
}

extern "C" int two_phase_flow_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT direction[8],
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_proteus_quad4_jacobian_action_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" void two_phase_flow_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_action_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_action_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_quad4_jacobian_p_c_p_c_diagnostics(
        void
) {
    return two_phase_flow_proteus_quad4_jacobian_p_c_p_c_diagnostics();
}

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_c_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_c_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_quad4_jacobian_p_c_p_w_diagnostics(
        void
) {
    return two_phase_flow_proteus_quad4_jacobian_p_c_p_w_diagnostics();
}

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_c_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_c_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_quad4_jacobian_p_w_p_c_diagnostics(
        void
) {
    return two_phase_flow_proteus_quad4_jacobian_p_w_p_c_diagnostics();
}

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_w_p_c_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_w_p_c_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" double two_phase_flow_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_quad4_jacobian_p_w_p_w_diagnostics(
        void
) {
    return two_phase_flow_proteus_quad4_jacobian_p_w_p_w_diagnostics();
}

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_w_p_w_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_jacobian_p_w_p_w_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_matrix_assembly_print_variant(
        const int variant,
        const ptrdiff_t nelements
) {
    two_phase_flow_proteus_quad4_matrix_assembly_print_variant(variant, nelements);
}

extern "C" const sfem_MatrixAssemblyDiagnostics * two_phase_flow_quad4_matrix_assembly_variant(
        const int variant
) {
    return two_phase_flow_proteus_quad4_matrix_assembly_variant(variant);
}

extern "C" int two_phase_flow_quad4_matrix_assembly_variant_count(
        void
) {
    return two_phase_flow_proteus_quad4_matrix_assembly_variant_count();
}

extern "C" int two_phase_flow_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_residual_affine_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_residual_affine_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT previous[8],
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_proteus_quad4_residual_element_soa(nelems, geometry_stride, determinant, adjugate, current, previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" double two_phase_flow_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes
) {
    return two_phase_flow_proteus_quad4_residual_element_soa_arithmetic_intensity(nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_quad4_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_proteus_quad4_residual_element_soa_diagnostics();
}

extern "C" int two_phase_flow_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT previous[8],
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[8]
) {
    return two_phase_flow_proteus_quad4_residual_element_soa_float(nelems, geometry_stride, determinant, adjugate, current, previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
}

extern "C" void two_phase_flow_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_residual_element_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_residual_element_soa_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" void two_phase_flow_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(elapsed, nelements, ndofs, repeat);
}

extern "C" void two_phase_flow_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat
) {
    two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(elapsed, nelements, ndofs, repeat);
}
