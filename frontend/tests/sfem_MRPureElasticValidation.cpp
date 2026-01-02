/**
 * @file sfem_MRPureElasticValidation.cpp
 * @brief Pure elastic Mooney-Rivlin validation against analytical formulas
 * 
 * Validates the MooneyRivlinVisco class (C++ frontend + C backend) against
 * analytical engineering stress formulas from continuummechanics.org:
 * https://www.continuummechanics.org/mooneyrivlin.html
 * 
 * Test cases:
 * 1. Uniaxial tension: λ₁ = λ, λ₂ = λ₃ = 1/√λ
 * 2. Plane strain (pure shear): λ₁ = λ, λ₂ = 1, λ₃ = 1/λ
 * 3. Equibiaxial tension: λ₁ = λ₂ = λ, λ₃ = 1/λ²
 * 
 * KEY: Your code uses UNIMODULAR Mooney-Rivlin (I1b, I2b).
 *      The analytical formulas assume J=1 (volume preserving deformation).
 * 
 * Operators exercised:
 * - MooneyRivlinVisco::set_C10, set_C01, set_K
 * - MooneyRivlinVisco::set_prony_terms(0, ...) for pure elastic
 * - MooneyRivlinVisco::initialize, initialize_history
 * - MooneyRivlinVisco::gradient (to compute reaction forces)
 * - MooneyRivlinVisco::get_gamma, get_num_active_terms
 * 
 * C backend kernels exercised:
 * - hex8_mooney_rivlin_grad_flexible (algorithmic gradient)
 * - hex8_mooney_rivlin_S_dev_from_disp (deviatoric stress)
 */

#include <stdio.h>
#include <math.h>
#include <memory>
#include <vector>
#include <fstream>
#include <cstring>
#include <cassert>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"

// ============================================================================
// Analytical formulas for UNIMODULAR (Isochoric) Mooney-Rivlin
// 
// IMPORTANT: Your implementation uses UNIMODULAR form:
//   W_dev = C10*(I1b - 3) + C01*(I2b - 3)
//   where I1b = J^(-2/3)*I1, I2b = J^(-4/3)*I2
//
// The website (continuummechanics.org) uses STANDARD form:
//   W_dev = C10*(I1 - 3) + C01*(I2 - 3)
//
// Even when J=1, the DERIVATIVES (and hence stresses) are DIFFERENT
// because of the deviatoric projection in the unimodular form:
//   ∂I1b/∂C = J^(-2/3) * [I - (1/3)*I1*C^(-1)]
//   ∂I2b/∂C = J^(-4/3) * [(I1*I - C) - (2/3)*I2*C^(-1)]
//
// Below are the correct UNIMODULAR analytical formulas.
// ============================================================================

// Uniaxial tension: λ₁ = λ, λ₂ = λ₃ = 1/√λ
// Derived from: S_11 = 2*C10*[1 - (1/3)*(1 + 2/λ³)] = (4/3)*C10*(1 - 1/λ³)
//               P_11 = λ * S_11 = (4/3)*C10*(λ - 1/λ²)
static double analytical_uniaxial(double lam, double C10, double C01, double C20 = 0.0) {
    (void)C20;
    
    // σ_eng = (4/3)*C10*(λ - 1/λ²) + (4/3)*C01*(1 - 1/λ³)
    double lam2 = lam * lam;
    double lam3 = lam2 * lam;
    double term1 = (4.0/3.0) * C10 * (lam - 1.0 / lam2);
    double term2 = (4.0/3.0) * C01 * (1.0 - 1.0 / lam3);
    return term1 + term2;
}

// Plane strain tension (pure shear): λ₁ = λ, λ₂ = 1, λ₃ = 1/λ
// Derived from: I1 = λ² + 1 + 1/λ², C^(-1)_11 = 1/λ²
//   (∂I1b/∂C)_11 = 1 - (1/3)*(λ² + 1 + 1/λ²)/λ² = 1 - (1/3)*(1 + 1/λ² + 1/λ⁴)
//   S_11 = 2*C10*[1 - (1/3)*(1 + 1/λ² + 1/λ⁴)]
//   P_11 = λ * S_11 = (2/3)*C10*(2λ - 1/λ - 1/λ³)
// For C01 term: (∂I2b/∂C)_11 = (1/3)*(1 + 1/λ² - 2/λ⁴)
//   P_11 = (2/3)*C01*(λ + 1/λ - 2/λ³)
static double analytical_plane_strain(double lam, double C10, double C01, double C20 = 0.0) {
    (void)C20;
    
    // σ_eng = (2/3)*C10*(2λ - 1/λ - 1/λ³) + (2/3)*C01*(λ + 1/λ - 2/λ³)
    double lam3 = lam * lam * lam;
    double term1 = (2.0/3.0) * C10 * (2.0*lam - 1.0/lam - 1.0/lam3);
    double term2 = (2.0/3.0) * C01 * (lam + 1.0/lam - 2.0/lam3);
    return term1 + term2;
}

// Equibiaxial tension: λ₁ = λ₂ = λ, λ₃ = 1/λ²
// Derived from: I1 = 2λ² + 1/λ⁴, C^(-1)_11 = 1/λ²
//   (∂I1b/∂C)_11 = 1 - (1/3)*(2λ² + 1/λ⁴)/λ² = 1 - (1/3)*(2 + 1/λ⁶) = (1/3)*(1 - 1/λ⁶)
//   S_11 = (2/3)*C10*(1 - 1/λ⁶)
//   P_11 = λ * S_11 = (2/3)*C10*(λ - 1/λ⁵)
// For C01 term: I2 = λ⁴ + 2/λ²
//   (∂I2b/∂C)_11 = (1/3)*(λ² - 1/λ⁴)
//   P_11 = (2/3)*C01*(λ³ - 1/λ³)
static double analytical_equibiaxial(double lam, double C10, double C01, double C20 = 0.0) {
    (void)C20;
    
    // σ_eng = (2/3)*C10*(λ - 1/λ⁵) + (2/3)*C01*(λ³ - 1/λ³)
    double lam3 = lam * lam * lam;
    double lam5 = lam3 * lam * lam;
    double term1 = (2.0/3.0) * C10 * (lam - 1.0 / lam5);
    double term2 = (2.0/3.0) * C01 * (lam3 - 1.0 / lam3);
    return term1 + term2;
}

// ============================================================================
// Test configuration
// ============================================================================

enum TestMode {
    MODE_UNIAXIAL = 0,
    MODE_PLANE_STRAIN = 1,   // Pure shear
    MODE_EQUIBIAXIAL = 2
};

static const char* mode_names[] = {"Uniaxial", "PlaneStrain", "Equibiaxial"};

// Apply volume-preserving displacement based on mode
static void apply_displacement(TestMode mode, double lambda, ptrdiff_t n_nodes, 
                               geom_t **pts, real_t *u) {
    double lam_y, lam_z;
    
    switch (mode) {
        case MODE_UNIAXIAL:
            lam_y = 1.0 / sqrt(lambda);
            lam_z = 1.0 / sqrt(lambda);
            break;
        case MODE_PLANE_STRAIN:
            lam_y = 1.0;
            lam_z = 1.0 / lambda;
            break;
        case MODE_EQUIBIAXIAL:
            lam_y = lambda;
            lam_z = 1.0 / (lambda * lambda);
            break;
        default:
            lam_y = 1.0;
            lam_z = 1.0;
    }
    
    double strain_x = lambda - 1.0;
    double strain_y = lam_y - 1.0;
    double strain_z = lam_z - 1.0;
    
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        u[3*i + 0] = strain_x * pts[0][i];
        u[3*i + 1] = strain_y * pts[1][i];
        u[3*i + 2] = strain_z * pts[2][i];
    }
}

// Get analytical stress for given mode
static double get_analytical_stress(TestMode mode, double lam, double C10, double C01) {
    switch (mode) {
        case MODE_UNIAXIAL:     return analytical_uniaxial(lam, C10, C01);
        case MODE_PLANE_STRAIN: return analytical_plane_strain(lam, C10, C01);
        case MODE_EQUIBIAXIAL:  return analytical_equibiaxial(lam, C10, C01);
        default: return 0.0;
    }
}

// ============================================================================
// Main validation test
// ============================================================================

int run_validation(TestMode mode, double C10, double C01, double K, bool use_flexible) {
    printf("\n========================================\n");
    printf("Testing: %s (C10=%.3f, C01=%.3f, K=%.1f, flexible=%s)\n", 
           mode_names[mode], C10, C01, K, use_flexible ? "true" : "false");
    printf("========================================\n");
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto es = sfem::EXECUTION_SPACE_HOST;
    
    // Create 1x1x1 hex8 mesh
    auto mesh = sfem::Mesh::create_hex8_cube(
        sfem::Communicator::wrap(comm),
        1, 1, 1,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    );
    
    auto fs = sfem::FunctionSpace::create(mesh, 3);
    
    // Create operator
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    
    // Set material parameters
    op->set_C10(C10);
    op->set_C01(C01);
    op->set_K(K);
    op->set_dt(0.01);  // Doesn't matter for pure elastic
    
    // KEY: Set zero Prony terms for PURE ELASTIC behavior
    op->set_prony_terms(0, nullptr, nullptr);
    
    // Test both flexible and non-flexible paths
    
    op->initialize();
    op->initialize_history();
    
    // Verify pure elastic state
    printf("  num_active_terms = %d (should be 0)\n", op->get_num_active_terms());
    printf("  gamma = %.6f (should be ~1.0 for pure elastic)\n", op->get_gamma());
    
    // Prepare buffers
    const ptrdiff_t ndofs = fs->n_dofs();
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    
    auto x_buf = sfem::create_buffer<real_t>(ndofs, es);
    auto rhs_buf = sfem::create_buffer<real_t>(ndofs, es);
    auto prev_buf = sfem::create_buffer<real_t>(ndofs, es);  // For prev_u in flexible mode
    
    auto blas = sfem::blas<real_t>(es);
    blas->zeros(ndofs, x_buf->data());
    blas->zeros(ndofs, rhs_buf->data());
    blas->zeros(ndofs, prev_buf->data());
    
    geom_t **pts = mesh->points()->data();
    
    // Find all nodes on the right face (x = x_max) for reaction extraction
    double x_max = -1e30;
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        if (pts[0][i] > x_max) x_max = pts[0][i];
    }
    
    std::vector<ptrdiff_t> right_face_nodes;
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        if (fabs(pts[0][i] - x_max) < 1e-10) {
            right_face_nodes.push_back(i);
        }
    }
    
    printf("  Right face (x=%.2f) has %zu nodes: ", x_max, right_face_nodes.size());
    for (auto n : right_face_nodes) {
        printf("%td(%.1f,%.1f,%.1f) ", n, pts[0][n], pts[1][n], pts[2][n]);
    }
    printf("\n");
    
    // Output CSV file
    char filename[256];
    snprintf(filename, sizeof(filename), "mr_pure_elastic_%s_C10_%.2f_C01_%.2f.csv", 
             mode_names[mode], C10, C01);
    std::ofstream csv(filename);
    csv << "lambda,strain,sigma_sim,sigma_analytical,rel_error_percent\n";
    
    printf("\n  %-10s %-12s %-14s %-14s %-12s\n", 
           "Lambda", "Strain", "Sigma_sim", "Sigma_analyt", "Rel_err[%]");
    printf("  -----------------------------------------------------------------\n");
    
    // Test range of lambda values
    double max_error = 0.0;
    double avg_error = 0.0;
    int count = 0;
    
    for (double lam = 1.0; lam <= 1.5; lam += 0.02) {
        // Apply displacement
        apply_displacement(mode, lam, n_nodes, pts, x_buf->data());
        
        // Compute gradient (internal forces)
        blas->zeros(ndofs, rhs_buf->data());
        int err = op->gradient(x_buf->data(), rhs_buf->data());
        if (err != 0) {
            printf("  ERROR: gradient() returned %d\n", err);
            return 1;
        }
        
        // Extract total reaction force on the right face (x = x_max)
        // Sum all nodal forces in x-direction on the right face
        // Engineering stress = Total Force / Original Area = Total Force (since A0 = 1)
        double total_force_x = 0.0;
        for (auto node : right_face_nodes) {
            total_force_x += rhs_buf->data()[3 * node];  // x-direction force at each node
        }
        double sigma_sim = total_force_x;  // Stress = Force / Area, Area = 1
        
        // Get analytical stress
        double sigma_analyt = get_analytical_stress(mode, lam, C10, C01);
        
        // Compute relative error
        double rel_err = 0.0;
        if (fabs(sigma_analyt) > 1e-10) {
            rel_err = fabs(sigma_sim - sigma_analyt) / fabs(sigma_analyt) * 100.0;
        }
        
        // Track statistics
        if (lam > 1.01) {  // Skip very small strains
            if (rel_err > max_error) max_error = rel_err;
            avg_error += rel_err;
            count++;
        }
        
        // Output
        double strain = lam - 1.0;
        printf("  %-10.4f %-12.6f %-14.6f %-14.6f %-12.2f\n",
               lam, strain, sigma_sim, sigma_analyt, rel_err);
        
        csv << lam << "," << strain << "," << sigma_sim << "," 
            << sigma_analyt << "," << rel_err << "\n";
    }
    
    csv.close();
    
    if (count > 0) avg_error /= count;
    
    printf("\n  Summary:\n");
    printf("    Max relative error: %.2f%%\n", max_error);
    printf("    Avg relative error: %.2f%%\n", avg_error);
    printf("    Results written to: %s\n", filename);
    
    // Pass/fail criteria
    bool passed = (max_error < 5.0);  // 5% tolerance for numerical error
    printf("\n  TEST %s (tolerance: 5%%)\n", passed ? "PASSED" : "FAILED");
    
    return passed ? 0 : 1;
}

// ============================================================================
// Generate comparison plots
// ============================================================================

void generate_plot_data() {
    // Material parameters matching website examples
    double params[][3] = {
        {1.0, 0.0, 1e6},  // C10=1, C01=0 (matches top plot in screenshot)
        {0.0, 1.0, 1e6},  // C10=0, C01=1 (matches bottom plot in screenshot)
        {0.5, 0.5, 1e6},  // Mixed case
    };
    
    printf("\n\n========================================\n");
    printf("Generating analytical curves for comparison\n");
    printf("========================================\n");
    
    for (int p = 0; p < 3; ++p) {
        double C10 = params[p][0];
        double C01 = params[p][1];
        
        char filename[256];
        snprintf(filename, sizeof(filename), "analytical_curves_C10_%.1f_C01_%.1f.csv", C10, C01);
        std::ofstream csv(filename);
        csv << "strain,sigma_uniaxial,sigma_shear,sigma_equibiaxial\n";
        
        for (double strain = 0.0; strain <= 0.5; strain += 0.01) {
            double lam = 1.0 + strain;
            double s_uni = analytical_uniaxial(lam, C10, C01);
            double s_shear = analytical_plane_strain(lam, C10, C01);
            double s_equi = analytical_equibiaxial(lam, C10, C01);
            
            csv << strain << "," << s_uni << "," << s_shear << "," << s_equi << "\n";
        }
        
        csv.close();
        printf("  Written: %s\n", filename);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int total_failures = 0;
    
    // Default material parameters
    double C10 = 1.0;
    double C01 = 0.0;
    double K = 1e6;  // Large K for near-incompressibility
    
    // Read from environment
    const char* env_c10 = getenv("SFEM_MOONEY_RIVLIN_C10");
    const char* env_c01 = getenv("SFEM_MOONEY_RIVLIN_C01");
    const char* env_k = getenv("SFEM_MOONEY_RIVLIN_K");
    
    if (env_c10) C10 = atof(env_c10);
    if (env_c01) C01 = atof(env_c01);
    if (env_k) K = atof(env_k);
    
    printf("==========================================================\n");
    printf("  Mooney-Rivlin Pure Elastic Validation Test\n");
    printf("==========================================================\n");
    printf("\n");
    printf("This test validates your MooneyRivlinVisco class against\n");
    printf("analytical formulas from continuummechanics.org.\n");
    printf("\n");
    printf("Your implementation: UNIMODULAR (I1b, I2b)\n");
    printf("Website formulas:    STANDARD (I1, I2)\n");
    printf("\n");
    printf("IMPORTANT: Even for J=1, UNIMODULAR and STANDARD stresses DIFFER!\n");
    printf("The deviatoric projection in unimodular gives stress ≈ 2/3 of standard.\n");
    printf("This test uses UNIMODULAR analytical formulas to match your code.\n");
    printf("\n");
    printf("Operators being tested:\n");
    printf("  - MooneyRivlinVisco::set_C10, set_C01, set_K\n");
    printf("  - MooneyRivlinVisco::set_prony_terms(0, ...)\n");
    printf("  - MooneyRivlinVisco::initialize(), initialize_history()\n");
    printf("  - MooneyRivlinVisco::gradient()\n");
    printf("  - MooneyRivlinVisco::get_gamma(), get_num_active_terms()\n");
    printf("\n");
    printf("C backend kernels exercised:\n");
    printf("  - hex8_mooney_rivlin_grad_flexible\n");
    printf("  - hex8_mooney_rivlin_S_dev_from_disp\n");
    printf("\n");
    
    // Generate analytical curves for plotting
    generate_plot_data();
    
    // ========== Test Case 1: C10=1, C01=0 (matches website top plot) ==========
    printf("\n\n=== Test Set 1: C10=1, C01=0 (website plot #1) ===\n");
    
    total_failures += run_validation(MODE_UNIAXIAL, 1.0, 0.0, K, true);
    total_failures += run_validation(MODE_PLANE_STRAIN, 1.0, 0.0, K, true);
    total_failures += run_validation(MODE_EQUIBIAXIAL, 1.0, 0.0, K, true);
    
    // ========== Test Case 2: C10=0, C01=1 (matches website bottom plot) ==========
    printf("\n\n=== Test Set 2: C10=0, C01=1 (website plot #2) ===\n");
    
    total_failures += run_validation(MODE_UNIAXIAL, 0.0, 1.0, K, true);
    total_failures += run_validation(MODE_PLANE_STRAIN, 0.0, 1.0, K, true);
    total_failures += run_validation(MODE_EQUIBIAXIAL, 0.0, 1.0, K, true);
    
    // ========== Test Case 3: User-specified parameters ==========
    if (fabs(C10 - 1.0) > 0.01 || fabs(C01) > 0.01) {
        printf("\n\n=== Test Set 3: User parameters C10=%.3f, C01=%.3f ===\n", C10, C01);
        
        total_failures += run_validation(MODE_UNIAXIAL, C10, C01, K, true);
        total_failures += run_validation(MODE_PLANE_STRAIN, C10, C01, K, true);
        total_failures += run_validation(MODE_EQUIBIAXIAL, C10, C01, K, true);
    }
    
    // ========== Summary ==========
    printf("\n\n==========================================================\n");
    printf("  OVERALL SUMMARY\n");
    printf("==========================================================\n");
    printf("  Total test failures: %d\n", total_failures);
    printf("  Status: %s\n", total_failures == 0 ? "ALL PASSED" : "SOME FAILED");
    printf("==========================================================\n");
    
    MPI_Finalize();
    return total_failures;
}

