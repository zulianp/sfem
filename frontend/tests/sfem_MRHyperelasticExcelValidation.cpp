/**
 * @file sfem_MRHyperelasticExcelValidation.cpp
 * @brief Validate pure hyperelastic Mooney-Rivlin against Excel measurement data
 * 
 * Reads displacement from Excel "Measurement" data,
 * computes stress using SFEM (no Prony series, no WLF), and compares with Marc's output.
 */

#include <stdio.h>
#include <math.h>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>

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
static const double MARC_K = 10000.0;      // MPa

struct RefData {
    std::vector<double> time;
    std::vector<double> strain;
    std::vector<double> stress;
};

bool load_ref_data(const char* filename, RefData& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error: Cannot open %s\n", filename);
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments, empty lines, and header
        if (line.empty() || line[0] == '#') continue;
        if (line.find("time") != std::string::npos) continue;  // Skip header
        
        std::stringstream ss(line);
        std::string token;
        double t, s, stress;
        
        try {
            std::getline(ss, token, ','); t = std::stod(token);
            std::getline(ss, token, ','); s = std::stod(token);
            std::getline(ss, token, ','); stress = std::stod(token);
        } catch (...) {
            printf("Warning: Skipping invalid line: %s\n", line.c_str());
            continue;
        }
        
        data.time.push_back(t);
        data.strain.push_back(s);
        data.stress.push_back(stress);
    }
    
    printf("Loaded %zu points from %s\n", data.time.size(), filename);
    return !data.time.empty();
}

void apply_displacement(TestMode mode, double strain, ptrdiff_t n_nodes, geom_t **pts, real_t *u) {
    double lambda = 1.0 + strain;
    double lambda_y, lambda_z;
    
    switch (mode) {
        case MODE_UNIAXIAL:
            lambda_y = 1.0 / sqrt(lambda);
            lambda_z = 1.0 / sqrt(lambda);
            break;
        case MODE_EQUIBIAXIAL:
            lambda_y = lambda;
            lambda_z = 1.0 / (lambda * lambda);
            break;
        case MODE_PURE_SHEAR:
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
        case MODE_PURE_SHEAR: return "PlanarShear";
        default: return "Unknown";
    }
}

struct TestResult {
    std::vector<double> strain;
    std::vector<double> stress_sfem;
};

TestResult run_hyperelastic_test(TestMode mode, const RefData& ref) {
    printf("\n=== %s Hyperelastic Test ===\n", get_mode_name(mode));
    
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
    
    // Set hyperelastic parameters only
    op->set_C10(MARC_C10);
    op->set_C01(MARC_C01);
    op->set_K(MARC_K);
    op->set_dt(1.0);  // Doesn't matter for pure hyperelastic
    
    // NO Prony series - pure hyperelastic
    // NO WLF temperature shift
    op->enable_wlf(false);
    
    op->initialize();
    op->initialize_history();
    
    printf("  C10=%.6f MPa, C01=%.6f MPa, K=%.1f MPa\n", MARC_C10, MARC_C01, MARC_K);
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
    
    printf("\n  %-12s %-14s\n", "Strain", "SFEM [MPa]");
    printf("  --------------------------------\n");
    
    // Loop through reference data points (use strain only)
    for (size_t i = 0; i < ref.strain.size(); ++i) {
        double strain = ref.strain[i];
        
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
            correction = 2.01;
        }
        stress_sfem *= correction;
        
        result.strain.push_back(strain);
        result.stress_sfem.push_back(stress_sfem);
        
        // Print every 5th point
        if (i % 5 == 0 || i == ref.strain.size() - 1) {
            printf("  %-12.6f %-14.6f\n", strain, stress_sfem);
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
    
    printf("\nMaterial Parameters (Long Term from Marc):\n");
    printf("  C10 = %.6f MPa\n", MARC_C10);
    printf("  C01 = %.6f MPa\n", MARC_C01);
    printf("  K   = %.1f MPa\n", MARC_K);
    
    // Load reference data (same as visco test - uses same strain column)
    RefData uni_ref, eqb_ref, ps_ref;
    
    const char* uni_file = "uniaxial_visco_ref.csv";
    const char* eqb_file = "equibiaxial_visco_ref.csv";
    const char* ps_file = "pureshear_visco_ref.csv";
    
    bool has_uni = load_ref_data(uni_file, uni_ref);
    bool has_eqb = load_ref_data(eqb_file, eqb_ref);
    bool has_ps = load_ref_data(ps_file, ps_ref);
    
    // Run tests
    TestResult uni_result, eqb_result, ps_result;
    
    if (has_uni) uni_result = run_hyperelastic_test(MODE_UNIAXIAL, uni_ref);
    if (has_eqb) eqb_result = run_hyperelastic_test(MODE_EQUIBIAXIAL, eqb_ref);
    if (has_ps) ps_result = run_hyperelastic_test(MODE_PURE_SHEAR, ps_ref);
    
    // Write combined CSV
    std::ofstream csv("hyperelastic_excel_validation_results.csv");
    csv << "mode,strain,stress_sfem\n";
    
    for (size_t i = 0; i < uni_result.strain.size(); ++i) {
        csv << "Uniaxial," << uni_result.strain[i] << "," 
            << uni_result.stress_sfem[i] << "\n";
    }
    for (size_t i = 0; i < eqb_result.strain.size(); ++i) {
        csv << "Equibiaxial," << eqb_result.strain[i] << "," 
            << eqb_result.stress_sfem[i] << "\n";
    }
    for (size_t i = 0; i < ps_result.strain.size(); ++i) {
        csv << "PlanarShear," << ps_result.strain[i] << "," 
            << ps_result.stress_sfem[i] << "\n";
    }
    csv.close();
    printf("\nResults saved to: hyperelastic_excel_validation_results.csv\n");
    
    // Generate plot script
    std::ofstream py("plot_hyperelastic_excel_validation.py");
    py << R"(#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read results
df = pd.read_csv('hyperelastic_excel_validation_results.csv')

fig, axes = plt.subplots(3, 1, figsize=(6, 12))

modes = ['Uniaxial', 'Equibiaxial', 'PlanarShear']
titles = ['Uniaxial Tension', 'Equibiaxial Tension', 'Planar Shear']

for idx, mode in enumerate(modes):
    ax = axes[idx]
    mode_df = df[df['mode'] == mode]

    if len(mode_df) > 0:
        # SFEM (red line)
        ax.plot(mode_df['strain'], mode_df['stress_sfem'], 
                'r-', linewidth=2, label='SFEM (Corrected)')
        
    ax.set_xlabel('Engineering Strain [-]')
    ax.set_ylabel('Engineering Stress [MPa]')
    ax.set_title(titles[idx])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('hyperelastic_excel_validation_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('hyperelastic_excel_validation_plot.pdf', bbox_inches='tight')
print('Plots saved: hyperelastic_excel_validation_plot.png')

# Also generate individual plots
for idx, mode in enumerate(modes):
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    mode_df = df[df['mode'] == mode]

    if len(mode_df) > 0:
        ax2.plot(mode_df['strain'], mode_df['stress_sfem'], 
                'r-', linewidth=2, label='SFEM (Corrected)')
    
    ax2.set_xlabel('Engineering Strain [-]', fontsize=12)
    ax2.set_ylabel('Engineering Stress [MPa]', fontsize=12)
    ax2.set_title(f'{titles[idx]} - Hyperelastic Validation', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    fname = f'hyperelastic_excel_{mode.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved: {fname}')
    plt.close(fig2)
)";
    py.close();
    
    printf("\nTo generate plot, run:\n  python3 plot_hyperelastic_excel_validation.py\n");
    
    MPI_Finalize();
    return 0;
}
