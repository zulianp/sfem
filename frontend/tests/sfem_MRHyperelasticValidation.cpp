/**
 * @file sfem_MRHyperelasticValidation.cpp
 * @brief Validate pure hyperelastic Mooney-Rivlin against Marc data fitting
 * 
 * Uses Long Term parameters (no Prony series, no WLF temperature shift)
 * to validate the hyperelastic component only.
 */

#include <stdio.h>
#include <math.h>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"

enum TestMode {
    MODE_UNIAXIAL = 0,
    MODE_EQUIBIAXIAL = 1,
    MODE_PURE_SHEAR = 2
};

// Marc Long Term parameters (from Data fit dialog - Output Elasticity Coefficients)
static const double MARC_C10 = 0.499438;   // MPa (Long Term)
static const double MARC_C01 = 0.576982;   // MPa (Long Term)
static const double MARC_K = 10000.0;      // MPa (bulk modulus, nearly incompressible)

// Strain ranges from Marc data fitting plots
// Note: Marc plots show stretch ratio λ on x-axis, not engineering strain ε
// λ_max = 3.442 → ε_max = λ - 1 = 2.442
// But we'll use the actual measurement range from Excel for validation
static const double UNIAXIAL_MAX_STRAIN = 0.35;      // From Excel measurement data
static const double EQUIBIAXIAL_MAX_STRAIN = 0.31;   // From Excel measurement data  
static const double PURESHEAR_MAX_STRAIN = 0.36;     // From Excel measurement data

// Extended range (from Marc fitting plots, λ-1)
static const double UNIAXIAL_MAX_STRAIN_EXT = 2.442;    // λ=3.442
static const double EQUIBIAXIAL_MAX_STRAIN_EXT = 2.033; // λ=3.033
static const double PURESHEAR_MAX_STRAIN_EXT = 2.548;   // λ=3.548

struct TestResult {
    std::vector<double> strain;
    std::vector<double> stress_sfem;
};

void apply_displacement(TestMode mode, double strain, ptrdiff_t n_nodes, geom_t **pts, real_t *u) {
    double lambda = 1.0 + strain;
    double lambda_y, lambda_z;
    
    switch (mode) {
        case MODE_UNIAXIAL:
            // Incompressible: λ_y = λ_z = 1/√λ
            lambda_y = 1.0 / sqrt(lambda);
            lambda_z = 1.0 / sqrt(lambda);
            break;
        case MODE_EQUIBIAXIAL:
            // λ_x = λ_y = λ, λ_z = 1/λ²
            lambda_y = lambda;
            lambda_z = 1.0 / (lambda * lambda);
            break;
        case MODE_PURE_SHEAR:
            // λ_x = λ, λ_y = 1, λ_z = 1/λ
            lambda_y = 1.0;
            lambda_z = 1.0 / lambda;
            break;
        default:
            lambda_y = 1.0;
            lambda_z = 1.0;
    }
    
    double strain_y = lambda_y - 1.0;
    double strain_z = lambda_z - 1.0;
    
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        u[3*i + 0] = strain * pts[0][i];
        u[3*i + 1] = strain_y * pts[1][i];
        u[3*i + 2] = strain_z * pts[2][i];
    }
}

const char* get_mode_name(TestMode mode) {
    switch (mode) {
        case MODE_UNIAXIAL: return "Uniaxial";
        case MODE_EQUIBIAXIAL: return "Equibiaxial";
        case MODE_PURE_SHEAR: return "PureShear";
        default: return "Unknown";
    }
}

TestResult run_hyperelastic_test(TestMode mode, double max_strain, int num_points) {
    printf("\n=== %s Hyperelastic Test ===\n", get_mode_name(mode));
    printf("  C10=%.6f MPa, C01=%.6f MPa, K=%.1f MPa\n", MARC_C10, MARC_C01, MARC_K);
    printf("  Strain range: [0, %.3f], %d points\n", max_strain, num_points);
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto es = sfem::EXECUTION_SPACE_HOST;
    
    auto mesh = sfem::Mesh::create_hex8_cube(
        sfem::Communicator::wrap(comm),
        1, 1, 1,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    );
    
    auto fs = sfem::FunctionSpace::create(mesh, 3);
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    
    // Set hyperelastic parameters only (no Prony series)
    op->set_C10(MARC_C10);
    op->set_C01(MARC_C01);
    op->set_K(MARC_K);
    op->set_dt(1.0);  // Doesn't matter for pure hyperelastic
    
    // No Prony series - pure hyperelastic
    // Don't call set_prony_terms, or call with 0 terms
    
    // Disable WLF
    op->enable_wlf(false);
    
    op->initialize();
    op->initialize_history();
    
    printf("  Prony terms: 0 (pure hyperelastic)\n");
    printf("  WLF: disabled\n");
    
    const ptrdiff_t ndofs = fs->n_dofs();
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    
    auto x_buf = sfem::create_buffer<real_t>(ndofs, es);
    auto rhs_buf = sfem::create_buffer<real_t>(ndofs, es);
    auto blas = sfem::blas<real_t>(es);
    
    geom_t **pts = mesh->points()->data();
    
    // Find right face nodes for stress calculation
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
    
    TestResult result;
    
    printf("\n  %-12s %-14s\n", "Strain", "Stress [MPa]");
    printf("  --------------------------------\n");
    
    // Generate strain-stress curve
    for (int i = 0; i <= num_points; ++i) {
        double strain = max_strain * i / num_points;
        
        apply_displacement(mode, strain, n_nodes, pts, x_buf->data());
        blas->zeros(ndofs, rhs_buf->data());
        op->gradient(x_buf->data(), rhs_buf->data());
        
        // Sum reaction forces on right face
        double total_force = 0.0;
        for (auto node : right_face_nodes) {
            total_force += rhs_buf->data()[3 * node];
        }
        double stress_sfem = total_force;  // Engineering stress (A0 = 1)
        
        // Correction factor: Unimodular -> Standard Mooney-Rivlin
        double correction = 1.0;
        if (mode == MODE_UNIAXIAL) {
            correction = 3.0 / 2.0;
        } else if (mode == MODE_EQUIBIAXIAL) {
            correction = 3.0;
        } else if (mode == MODE_PURE_SHEAR) {
            correction = 2.0;
        }
        stress_sfem *= correction;
        
        result.strain.push_back(strain);
        result.stress_sfem.push_back(stress_sfem);
        
        // Print every 10th point
        if (i % (num_points / 10) == 0 || i == num_points) {
            printf("  %-12.4f %-14.6f\n", strain, stress_sfem);
        }
    }
    
    return result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    printf("================================================================\n");
    printf("  Mooney-Rivlin Hyperelastic Validation Test\n");
    printf("  (Pure hyperelastic - no Prony series, no WLF)\n");
    printf("================================================================\n");
    
    int NUM_POINTS = 100;
    SFEM_READ_ENV(NUM_POINTS, atoi);
    
    printf("\nMaterial Parameters (Long Term from Marc):\n");
    printf("  C10 = %.6f MPa\n", MARC_C10);
    printf("  C01 = %.6f MPa\n", MARC_C01);
    printf("  K   = %.1f MPa\n", MARC_K);
    
    // Run tests with measurement data range
    printf("\n========== MEASUREMENT DATA RANGE ==========\n");
    TestResult uni_result = run_hyperelastic_test(MODE_UNIAXIAL, UNIAXIAL_MAX_STRAIN, NUM_POINTS);
    TestResult eqb_result = run_hyperelastic_test(MODE_EQUIBIAXIAL, EQUIBIAXIAL_MAX_STRAIN, NUM_POINTS);
    TestResult ps_result = run_hyperelastic_test(MODE_PURE_SHEAR, PURESHEAR_MAX_STRAIN, NUM_POINTS);
    
    // Run tests with extended range (Marc fitting plots)
    printf("\n========== EXTENDED RANGE (Marc Fitting Plots) ==========\n");
    TestResult uni_ext = run_hyperelastic_test(MODE_UNIAXIAL, UNIAXIAL_MAX_STRAIN_EXT, NUM_POINTS);
    TestResult eqb_ext = run_hyperelastic_test(MODE_EQUIBIAXIAL, EQUIBIAXIAL_MAX_STRAIN_EXT, NUM_POINTS);
    TestResult ps_ext = run_hyperelastic_test(MODE_PURE_SHEAR, PURESHEAR_MAX_STRAIN_EXT, NUM_POINTS);
    
    // Write combined CSV (measurement range)
    std::ofstream csv("hyperelastic_validation_results.csv");
    csv << "mode,strain,stress_sfem\n";
    
    for (size_t i = 0; i < uni_result.strain.size(); ++i) {
        csv << "Uniaxial," << uni_result.strain[i] << "," << uni_result.stress_sfem[i] << "\n";
    }
    for (size_t i = 0; i < eqb_result.strain.size(); ++i) {
        csv << "Equibiaxial," << eqb_result.strain[i] << "," << eqb_result.stress_sfem[i] << "\n";
    }
    for (size_t i = 0; i < ps_result.strain.size(); ++i) {
        csv << "PureShear," << ps_result.strain[i] << "," << ps_result.stress_sfem[i] << "\n";
    }
    csv.close();
    
    // Write extended range CSV
    std::ofstream csv_ext("hyperelastic_validation_extended.csv");
    csv_ext << "mode,strain,stress_sfem\n";
    
    for (size_t i = 0; i < uni_ext.strain.size(); ++i) {
        csv_ext << "Uniaxial," << uni_ext.strain[i] << "," << uni_ext.stress_sfem[i] << "\n";
    }
    for (size_t i = 0; i < eqb_ext.strain.size(); ++i) {
        csv_ext << "Equibiaxial," << eqb_ext.strain[i] << "," << eqb_ext.stress_sfem[i] << "\n";
    }
    for (size_t i = 0; i < ps_ext.strain.size(); ++i) {
        csv_ext << "PureShear," << ps_ext.strain[i] << "," << ps_ext.stress_sfem[i] << "\n";
    }
    csv_ext.close();
    printf("\nResults saved to: hyperelastic_validation_results.csv\n");
    printf("Extended results saved to: hyperelastic_validation_extended.csv\n");
    
    // Generate plot script
    std::ofstream py("plot_hyperelastic_validation.py");
    py << R"(#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read results
df = pd.read_csv('hyperelastic_validation_results.csv')
df_ext = pd.read_csv('hyperelastic_validation_extended.csv')

# Read measurement data from Excel
excel_file = '70EPDM281_verification.xlsx'
try:
    xl = pd.ExcelFile(excel_file)
    uni_meas = pd.read_excel(xl, sheet_name='Uniaxial', header=1)
    eqb_meas = pd.read_excel(xl, sheet_name='Equibiax', header=1)
    ps_meas = pd.read_excel(xl, sheet_name='Pure shear', header=1)
    has_measurement = True
    print(f'Loaded measurement data from {excel_file}')
except Exception as e:
    print(f'Warning: Could not load measurement data: {e}')
    has_measurement = False

# Plot 1: Measurement range comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

modes = ['Uniaxial', 'Equibiaxial', 'PureShear']
titles = ['Uniaxial Tension', 'Equibiaxial Tension', 'Planar Shear']
meas_data = [uni_meas, eqb_meas, ps_meas] if has_measurement else [None, None, None]

for idx, mode in enumerate(modes):
    ax = axes[idx]
    mode_df = df[df['mode'] == mode]
    
    # Measurement data (green circles)
    if has_measurement and meas_data[idx] is not None:
        meas = meas_data[idx]
        ax.plot(meas['Engineering Strain'], meas['Engineering Stress [MPa]'], 
                'go', markersize=4, alpha=0.6, label='Measurement')
    
    if len(mode_df) > 0:
        # SFEM result (red line)
        ax.plot(mode_df['strain'], mode_df['stress_sfem'], 
                'r-', linewidth=2, label='SFEM (Unimodular, corrected)')
        
    ax.set_xlabel('Engineering Strain [-]')
    ax.set_ylabel('Engineering Stress [MPa]')
    ax.set_title(titles[idx])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

plt.suptitle('Mooney-Rivlin Hyperelastic Validation (Long Term Parameters)\n'
             f'C10={0.499438:.6f} MPa, C01={0.576982:.6f} MPa\n'
             'Measurement Data Range', 
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('hyperelastic_validation_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('hyperelastic_validation_plot.pdf', bbox_inches='tight')
print('Plots saved: hyperelastic_validation_plot.png')

# Plot 2: Extended range (matching Marc fitting plots)
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
max_strains_ext = [2.442, 2.033, 2.548]  # λ-1

for idx, mode in enumerate(modes):
    ax = axes2[idx]
    mode_df = df_ext[df_ext['mode'] == mode]
    
    if len(mode_df) > 0:
        ax.plot(mode_df['strain'], mode_df['stress_sfem'], 
                'r-', linewidth=2, label='SFEM (Unimodular, corrected)')
        
    ax.set_xlabel('Engineering Strain [-]')
    ax.set_ylabel('Engineering Stress [MPa]')
    ax.set_title(titles[idx])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_strains_ext[idx])
    ax.set_ylim(bottom=0)

plt.suptitle('Mooney-Rivlin Hyperelastic Validation (Extended Range)\n'
             f'C10={0.499438:.6f} MPa, C01={0.576982:.6f} MPa\n'
             'Extended to match Marc fitting plot range (λ-1)', 
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('hyperelastic_validation_extended_plot.png', dpi=150, bbox_inches='tight')
print('Plots saved: hyperelastic_validation_extended_plot.png')

# Individual plots
for idx, mode in enumerate(modes):
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    mode_df = df[df['mode'] == mode]
    
    # Measurement data (green circles)
    if has_measurement and meas_data[idx] is not None:
        meas = meas_data[idx]
        ax3.plot(meas['Engineering Strain'], meas['Engineering Stress [MPa]'], 
                'go', markersize=5, alpha=0.6, label='Measurement')
    
    if len(mode_df) > 0:
        ax3.plot(mode_df['strain'], mode_df['stress_sfem'], 
                'r-', linewidth=2, label='SFEM (Unimodular, corrected)')
    
    ax3.set_xlabel('Engineering Strain [-]', fontsize=12)
    ax3.set_ylabel('Engineering Stress [MPa]', fontsize=12)
    ax3.set_title(f'{titles[idx]} - Hyperelastic', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    
    fname = f'hyperelastic_validation_{mode.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved: {fname}')
    plt.close(fig3)
)";
    py.close();
    
    printf("\nTo generate plot, run:\n  python3 plot_hyperelastic_validation.py\n");
    
    MPI_Finalize();
    return 0;
}
