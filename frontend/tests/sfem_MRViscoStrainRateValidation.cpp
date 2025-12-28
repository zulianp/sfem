/**
 * @file sfem_MRViscoStrainRateValidation.cpp
 * @brief Strain-rate dependent validation for hyper-viscoelastic model
 * 
 * Based on: "Polymer Testing 133 (2024) 108375"
 * Section 5.5: Numerical validation of the proposed hyper-viscoelastic model
 * 
 * Validates strain-rate dependent behavior by comparing stress-strain curves
 * at different strain rates (from 0.001 to 5000 s^-1).
 * 
 * Material parameters from the paper:
 * - Hyperelastic: A01=1.7549, A10=-0.5341, A11=0.0844 MPa (Table 1)
 *   Note: A11 term is IGNORED in this test (standard Mooney-Rivlin approximation)
 * - Prony series: 10 terms with g_inf=0.0225 (Table 3)
 * 
 * Test: Uniaxial compression at constant strain rate
 */

#include <stdio.h>
#include <math.h>
#include <memory>
#include <vector>
#include <fstream>
#include <cstring>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"

// ============================================================================
// Material parameters from Polymer Testing 133 (2024) 108375
// ============================================================================

// Hyperelastic parameters
// Table 1 gives LONG TERM parameters: Ā₁₀ = -0.5341, Ā₀₁ = 1.7549
// From Eq.45: A = Ā / g_inf (Short Term = Long Term / 0.0225)
//
// Using Short Term parameters + gamma = g_inf + Σβ_i
// This is mathematically correct for our FEM (tensor-based) formulation
//
// Note: ~4/3 factor difference vs paper's analytical formula is expected
// (different stress derivation: FEM ∂W/∂C vs paper's ∂W/∂λ)
static double PAPER_C10 = -0.5341 / 0.0225;  // MPa (Short Term A_10)
static double PAPER_C01 = 1.7549 / 0.0225;   // MPa (Short Term A_01)
// Paper assumes J=1 (incompressible). For near-incompressible:
// K/μ should be >> 100, where μ ≈ 2(C10+C01) ≈ 108 MPa
// So K should be ~10,000+ MPa for ν ≈ 0.499
static double PAPER_K = 100000.0;            // MPa (large K for near-incompressible)

// Prony series parameters (Table 3)
// 10 terms, logarithmically spaced from 10^-7 to 10^2 s
static const int NUM_PRONY = 10;

// Original g_i values from Table 3
static const double prony_g_original[] = {
    0.2874,     // tau = 10^-7
    0.2671,     // tau = 10^-6
    0.2064,     // tau = 10^-5
    0.1292,     // tau = 10^-4
    0.0660,     // tau = 10^-3
    0.0288,     // tau = 10^-2
    0.0111,     // tau = 10^-1
    0.0033,     // tau = 10^0
    5.296e-4,   // tau = 10^1
    1.795e-5    // tau = 10^2
};

static const double prony_tau[] = {
    1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
    1e-2, 1e-1, 1e0,  1e1,  1e2
};

// g_infinity from paper
static const double G_INF_PAPER = 0.0225;

// Normalized g_i values (will be computed)
static double prony_g_normalized[NUM_PRONY];

// ============================================================================
// Test strain rates from Fig. 16
// ============================================================================

struct StrainRateTest {
    double strain_rate;  // s^-1
    double max_strain;   // maximum strain to reach
    const char* label;
};

static const StrainRateTest strain_rate_tests[] = {
    {100000.0, 0.50, "pure_elastic"},  // Very high rate triggers pure elastic mode
    {5000.0,  0.50, "5000"},
    {2000.0,  0.50, "2000"},
    {1000.0,  0.50, "1000"},
    {0.1,     0.55, "0.1"},
    {0.01,    0.55, "0.01"},
    {0.001,   0.55, "0.001"}
};
static const int NUM_STRAIN_RATES = 7;

// ============================================================================
// Helper functions
// ============================================================================

void normalize_prony_series() {
    // The paper gives g_inf = 0.0225
    // We need sum(g_i) = 1 - g_inf = 0.9775
    
    double sum_original = 0.0;
    for (int i = 0; i < NUM_PRONY; i++) {
        sum_original += prony_g_original[i];
    }
    
    double target_sum = 1.0 - G_INF_PAPER;  // = 0.9775
    double scale = target_sum / sum_original;
    
    printf("Prony series normalization:\n");
    printf("  Original sum(g_i) = %.6f\n", sum_original);
    printf("  Target sum(g_i) = %.6f (1 - g_inf)\n", target_sum);
    printf("  Scale factor = %.6f\n", scale);
    
    for (int i = 0; i < NUM_PRONY; i++) {
        prony_g_normalized[i] = prony_g_original[i] * scale;
    }
    
    // Verify
    double check_sum = 0.0;
    for (int i = 0; i < NUM_PRONY; i++) {
        check_sum += prony_g_normalized[i];
    }
    printf("  Normalized sum(g_i) = %.6f\n", check_sum);
    printf("  Resulting g_inf = %.6f\n\n", 1.0 - check_sum);
}

// ============================================================================
// Paper's analytical stress formula (Eq. 41, ignoring A11)
// Π⁰₁₁(λ) = 2(λ - λ⁻²)[A₁₀λ + A₀₁]  -- uses SHORT TERM parameters
// Total stress (Eq. 46): Π = gamma * Π⁰  (in gamma approximation)
// where gamma = g_inf + Σβ_i
// ============================================================================
double paper_analytical_stress(double lambda, double gamma) {
    // SHORT TERM parameters (A = Ā / g_inf)
    const double A10 = -0.5341 / 0.0225;  // = -23.74 MPa
    const double A01 = 1.7549 / 0.0225;   // = 78.0 MPa
    
    // Paper's simplified formula (Eq. 41 without A11)
    double factor = 2.0 * (lambda - 1.0/(lambda*lambda));
    double inner = A10 * lambda + A01;
    double Pi_elastic = factor * inner;
    
    // Total stress with viscoelastic factor
    // gamma = g_inf + Σβ_i (same as SFEM)
    return fabs(gamma * Pi_elastic);
}

// Apply uniaxial COMPRESSION displacement (lambda < 1)
void apply_compression_displacement(double strain, ptrdiff_t n_nodes, 
                                    geom_t **pts, real_t *u) {
    // For uniaxial compression: lambda = 1 - |strain|
    // Volume preserving: lambda_y = lambda_z = 1/sqrt(lambda)
    
    double lambda = 1.0 - fabs(strain);  // Compression: lambda < 1
    if (lambda < 0.4) lambda = 0.4;      // Safety limit
    
    double lambda_transverse = 1.0 / sqrt(lambda);
    
    double strain_x = lambda - 1.0;              // Negative (compression)
    double strain_y = lambda_transverse - 1.0;   // Positive (expansion)
    double strain_z = lambda_transverse - 1.0;   // Positive (expansion)
    
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        u[3*i + 0] = strain_x * pts[0][i];
        u[3*i + 1] = strain_y * pts[1][i];
        u[3*i + 2] = strain_z * pts[2][i];
    }
}

// ============================================================================
// Run test for single strain rate
// ============================================================================

int run_strain_rate_test(double strain_rate, double max_strain, const char* label,
                         std::ofstream& combined_csv) {
    printf("\n========================================\n");
    printf("Strain Rate Test: ε̇ = %g s⁻¹\n", strain_rate);
    printf("========================================\n");
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto es = sfem::EXECUTION_SPACE_HOST;
    
    // Create mesh (1x1x1 cube)
    auto mesh = sfem::Mesh::create_hex8_cube(
        sfem::Communicator::wrap(comm),
        1, 1, 1,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    );
    
    auto fs = sfem::FunctionSpace::create(mesh, 3);
    const ptrdiff_t ndofs = fs->n_dofs();
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    
    // Create operator
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    
    // Calculate appropriate time step
    // We want strain increment per step to be small enough
    double d_strain = 0.001;  // 0.1% strain per step (smaller for accuracy)
    double dt = d_strain / strain_rate;
    
    // For very high strain rates, limit minimum dt
    if (dt < 1e-9) {
        dt = 1e-9;
        d_strain = dt * strain_rate;
    }
    
    printf("  dt = %.6e s (strain increment = %.4f per step)\n", dt, d_strain);
    
    // Set material parameters
    op->set_C10(PAPER_C10);
    op->set_C01(PAPER_C01);
    op->set_K(PAPER_K);
    op->set_dt(dt);
    op->set_use_flexible(true);
    
    // Option to test pure elastic (for debugging)
    bool pure_elastic = (strain_rate >= 10000.0);  // Use very high rate as "pure elastic" test
    
    if (pure_elastic) {
        // Pure elastic: no Prony terms, gamma = 1
        op->set_prony_terms(0, nullptr, nullptr);
        printf("  [DEBUG] Pure elastic mode (num_prony=0, gamma=1)\n");
    } else {
        // Set Prony series (normalized)
        op->set_prony_terms(NUM_PRONY, prony_g_normalized, prony_tau);
    }
    
    op->initialize();
    op->initialize_history();
    
    printf("  C10 = %.4f, C01 = %.4f MPa\n", PAPER_C10, PAPER_C01);
    printf("  Prony terms: %d, Active: %d, Gamma: %.6f\n",
           NUM_PRONY, op->get_num_active_terms(), op->get_gamma());
    
    // Buffers
    auto x_buf = sfem::create_buffer<real_t>(ndofs, es);
    auto rhs_buf = sfem::create_buffer<real_t>(ndofs, es);
    auto blas = sfem::blas<real_t>(es);
    
    blas->zeros(ndofs, x_buf->data());
    
    geom_t **pts = mesh->points()->data();
    
    // Find all nodes on the loaded face (x = x_max) for stress extraction
    double x_max = -1e30;
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        if (pts[0][i] > x_max) x_max = pts[0][i];
    }
    
    std::vector<ptrdiff_t> loaded_face_nodes;
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        if (fabs(pts[0][i] - x_max) < 1e-10) {
            loaded_face_nodes.push_back(i);
        }
    }
    
    // Output file for this strain rate
    char filename[256];
    snprintf(filename, sizeof(filename), "strain_rate_%s_results.csv", label);
    std::ofstream csv(filename);
    csv << "time,strain,stress_MPa\n";
    
    // Time loop
    double t = 0.0;
    double t_end = max_strain / strain_rate;
    int step = 0;
    int output_interval = std::max(1, (int)(t_end / dt / 50));  // ~50 output points
    
    // Get gamma for analytical comparison
    // Both SFEM and Paper use: gamma = g_inf + Σβ_i with Short Term parameters
    double gamma = op->get_gamma();
    
    printf("\n  gamma = %.6f (g_inf + Σβ_i)\n", gamma);
    printf("\n  %-12s %-12s %-14s %-14s %-10s\n", "Time[s]", "Strain", "SFEM[MPa]", "Paper[MPa]", "Ratio");
    printf("  ------------------------------------------------------------------------\n");
    
    std::vector<double> strains_out, stresses_out;
    
    while (t <= t_end + 1e-12) {
        // Current strain (compression)
        double current_strain = strain_rate * t;
        if (current_strain > max_strain) current_strain = max_strain;
        
        // Apply displacement
        apply_compression_displacement(current_strain, n_nodes, pts, x_buf->data());
        
        // Compute internal forces
        blas->zeros(ndofs, rhs_buf->data());
        op->gradient(x_buf->data(), rhs_buf->data());
        
        // Extract stress: sum of reaction forces on loaded face / area
        // Area = 1 for unit cube, but with transverse expansion it changes
        // For engineering stress (force / original area), just sum forces
        double total_force = 0.0;
        for (auto node : loaded_face_nodes) {
            total_force += rhs_buf->data()[3 * node];  // x-direction force
        }
        
        // Engineering stress (force / original area = 1)
        // Note: compression gives negative force, take absolute value
        double stress_sfem = fabs(total_force);
        
        // Paper's analytical stress (Short Term params + gamma)
        double lambda = 1.0 - fabs(current_strain);
        double stress_paper = paper_analytical_stress(lambda, gamma);
        
        strains_out.push_back(current_strain);
        stresses_out.push_back(stress_sfem);
        
        // Output
        double ratio = (stress_paper > 0.01) ? stress_sfem / stress_paper : 0.0;
        if (step % output_interval == 0 || t >= t_end - dt/2) {
            printf("  %-12.4e %-12.4f %-14.4f %-14.4f %-10.4f\n", 
                   t, current_strain, stress_sfem, stress_paper, ratio);
            csv << t << "," << current_strain << "," << stress_sfem << "," << stress_paper << "\n";
            combined_csv << strain_rate << "," << current_strain << "," << stress_sfem << "\n";
        }
        
        // Update history
        op->update_history(x_buf->data());
        
        t += dt;
        step++;
    }
    
    csv.close();
    printf("\n  Results written to: %s\n", filename);
    printf("  Total steps: %d\n", step);
    
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    printf("==========================================================\n");
    printf("  Hyper-Viscoelastic Strain Rate Validation\n");
    printf("  Based on: Polymer Testing 133 (2024) 108375\n");
    printf("==========================================================\n\n");
    
    printf("Material parameters (Table 1, approximated):\n");
    printf("  C10 (A_10) = %.4f MPa\n", PAPER_C10);
    printf("  C01 (A_01) = %.4f MPa\n", PAPER_C01);
    printf("  Note: A_11 = 0.0844 MPa term is IGNORED\n\n");
    
    // Read optional overrides
    SFEM_READ_ENV(PAPER_C10, atof);
    SFEM_READ_ENV(PAPER_C01, atof);
    SFEM_READ_ENV(PAPER_K, atof);
    
    // Normalize Prony series
    normalize_prony_series();
    
    printf("Prony series (Table 3, normalized):\n");
    printf("  %-12s %-12s\n", "tau_i [s]", "g_i");
    for (int i = 0; i < NUM_PRONY; i++) {
        printf("  %-12.2e %-12.4f\n", prony_tau[i], prony_g_normalized[i]);
    }
    printf("  g_inf = %.4f\n\n", G_INF_PAPER);
    
    // Combined output file
    std::ofstream combined_csv("strain_rate_all_results.csv");
    combined_csv << "strain_rate,strain,stress_MPa\n";
    
    // Run tests for each strain rate
    printf("Testing %d strain rates from Fig. 16...\n", NUM_STRAIN_RATES);
    
    for (int i = 0; i < NUM_STRAIN_RATES; i++) {
        run_strain_rate_test(
            strain_rate_tests[i].strain_rate,
            strain_rate_tests[i].max_strain,
            strain_rate_tests[i].label,
            combined_csv
        );
    }
    
    combined_csv.close();
    
    printf("\n==========================================================\n");
    printf("  All tests completed!\n");
    printf("  Combined results: strain_rate_all_results.csv\n");
    printf("==========================================================\n");
    
    MPI_Finalize();
    return 0;
}

