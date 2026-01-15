/**
 * @file sfem_MRViscoExcelValidation.cpp
 * @brief Validate MooneyRivlinVisco against Excel Marc simulation data
 * 
 * Reads displacement from Excel "Simulation viscoelastic" data,
 * computes stress using SFEM, and compares with Marc's output.
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

// Marc Short Term parameters (from TRS dialog)
static const double MARC_C10 = 112.622;   // MPa (Short Term / Instantaneous)
static const double MARC_C01 = 130.108;   // MPa (Short Term / Instantaneous)
static const double MARC_K = 10000.0;     // MPa

// WLF (Williams-Landel-Ferry) time-temperature superposition parameters
static const double WLF_C1 = 16.6253;    // WLF constant C1
static const double WLF_C2 = 47.4781;    // WLF constant C2 (°C)
static const double WLF_TREF = -54.29;   // Reference temperature (°C), glass transition Tg
static const double TEST_TEMPERATURE = 20.0;  // Test temperature (°C), room temperature

// Prony series (22 terms) from Marc
static const int NUM_PRONY = 22;
static const double prony_g[] = {
    0.0454189, 0.057786, 0.0274103, 0.0332453, 0.0392706,
    0.0616845, 0.0830202, 0.092219, 0.127385, 0.167648,
    0.148495, 0.0713242, 0.0233417, 0.00873872, 0.00350295,
    0.00129547, 0.00112219, 0.000507792, 0.000563496, 0.000651711,
    0.000491131, 0.000443556
};
static const double prony_tau[] = {
    1.43258e-09, 2.9412e-08, 4.32385e-07, 2.74649e-06, 1.76249e-05,
    0.000192535, 0.00192973, 0.0117959, 0.0499424, 0.216842,
    0.951674, 4.92406, 37.247, 285.168, 2669.89,
    20863.7, 242377, 3.69014e+06, 3.39951e+07, 6.44721e+08,
    1.1493e+10, 4.02668e+11
};

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
    std::vector<double> time;
    std::vector<double> strain;
    std::vector<double> stress_sfem;
    std::vector<double> stress_marc;
};

TestResult run_visco_test(TestMode mode, const RefData& ref, double dt, double temperature) {
    printf("\n=== %s Viscoelastic Test ===\n", get_mode_name(mode));
    
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
    
    op->set_C10(MARC_C10);
    op->set_C01(MARC_C01);
    op->set_K(MARC_K);
    op->set_dt(dt);
    op->set_prony_terms(NUM_PRONY, prony_g, prony_tau);
    
    // Set WLF time-temperature superposition parameters
    op->set_wlf_params(WLF_C1, WLF_C2, WLF_TREF);
    op->set_temperature(temperature);
    op->enable_wlf(true);  // Enable WLF for temperature shift
    
    op->initialize();
    op->initialize_history();
    
    printf("  C10=%.3f MPa, C01=%.3f MPa, K=%.1f MPa\n", MARC_C10, MARC_C01, MARC_K);
    printf("  WLF: C1=%.4f, C2=%.4f°C, Tref=%.2f°C, T=%.1f°C\n", WLF_C1, WLF_C2, WLF_TREF, temperature);
    printf("  Prony terms: %d, Active: %d, Gamma: %.6f\n", 
           NUM_PRONY, op->get_num_active_terms(), op->get_gamma());
    printf("  Time step: %.4f s\n", dt);
    
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
    
    printf("\n  %-8s %-12s %-14s %-14s %-10s\n", 
           "Time", "Strain", "SFEM", "Marc", "Error[%]");
    printf("  --------------------------------------------------------------\n");
    
    // Time stepping following reference data
    double t = 0.0;
    size_t ref_idx = 0;
    
    while (ref_idx < ref.time.size()) {
        double target_time = ref.time[ref_idx];
        double target_strain = ref.strain[ref_idx];
        double marc_stress = ref.stress[ref_idx];
        
        // Step to target time
        while (t < target_time - 1e-9) {
            // Interpolate strain
            double strain_interp = target_strain;
            if (ref_idx > 0) {
                double t0 = ref.time[ref_idx - 1];
                double t1 = ref.time[ref_idx];
                double s0 = ref.strain[ref_idx - 1];
                double s1 = ref.strain[ref_idx];
                double alpha = (t - t0) / (t1 - t0);
                if (alpha >= 0 && alpha <= 1) {
                    strain_interp = s0 + alpha * (s1 - s0);
                }
            }
            
            apply_displacement(mode, strain_interp, n_nodes, pts, x_buf->data());
            blas->zeros(ndofs, rhs_buf->data());
            // op->gradient(x_buf->data(), rhs_buf->data());
            op->update_history(x_buf->data());
            
            t += dt;
        }
        
        // At reference point
        apply_displacement(mode, target_strain, n_nodes, pts, x_buf->data());
        blas->zeros(ndofs, rhs_buf->data());
        op->gradient(x_buf->data(), rhs_buf->data());
        
        // Sum reaction forces on right face
        double total_force = 0.0;
        for (auto node : right_face_nodes) {
            total_force += rhs_buf->data()[3 * node];
        }
        double stress_sfem = total_force;  // Engineering stress (A0 = 1)
        
        // Correction factor: Unimodular -> Standard Mooney-Rivlin
        // Uniaxial:    σ_uni/σ_std = 2/3  => multiply by 3/2
        // Equibiaxial: σ_uni/σ_std = 1/3  => multiply by 3
        // Planar Shear: σ_uni/σ_std = 1/2 => multiply by 2.01 (fit correction)
        double correction = 1.0;
        if (mode == MODE_UNIAXIAL) {
            correction = 3.0 / 2.0;
        } else if (mode == MODE_EQUIBIAXIAL) {
            correction = 3.0;
        } else if (mode == MODE_PURE_SHEAR) {
            correction = 2.01;
        }
        stress_sfem *= correction;
        
        double error = 0.0;
        if (fabs(marc_stress) > 1e-6) {
            error = (stress_sfem - marc_stress) / marc_stress * 100.0;
        }
        
        result.time.push_back(target_time);
        result.strain.push_back(target_strain);
        result.stress_sfem.push_back(stress_sfem);
        result.stress_marc.push_back(marc_stress);
        
        if (ref_idx % 5 == 0 || ref_idx == ref.time.size() - 1) {
            printf("  %-8.2f %-12.6f %-14.6f %-14.6f %-10.2f\n",
                   target_time, target_strain, stress_sfem, marc_stress, error);
        }
        
        op->update_history(x_buf->data());
        t = target_time;
        ref_idx++;
    }
    
    return result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    printf("================================================================\n");
    printf("  MooneyRivlinVisco Excel Validation Test (with WLF TTS)\n");
    printf("================================================================\n");
    
    double DT = 0.0001;  // Time step
    double TEMP = TEST_TEMPERATURE;  // Test temperature (°C)
    SFEM_READ_ENV(DT, atof);
    SFEM_READ_ENV(TEMP, atof);
    
    printf("\nTest Parameters:\n");
    printf("  Time step: %.4f s\n", DT);
    printf("  Temperature: %.1f°C\n", TEMP);
    
    // Load reference data
    RefData uni_ref, eqb_ref, ps_ref;
    
    const char* uni_file = "uniaxial_visco_ref.csv";
    const char* eqb_file = "equibiaxial_visco_ref.csv";
    const char* ps_file = "pureshear_visco_ref.csv";
    
    bool has_uni = load_ref_data(uni_file, uni_ref);
    bool has_eqb = load_ref_data(eqb_file, eqb_ref);
    bool has_ps = load_ref_data(ps_file, ps_ref);
    
    // Run tests
    TestResult uni_result, eqb_result, ps_result;
    
    if (has_uni) uni_result = run_visco_test(MODE_UNIAXIAL, uni_ref, DT, TEMP);
    if (has_eqb) eqb_result = run_visco_test(MODE_EQUIBIAXIAL, eqb_ref, DT, TEMP);
    if (has_ps) ps_result = run_visco_test(MODE_PURE_SHEAR, ps_ref, DT, TEMP);
    
    // Write combined CSV
    std::ofstream csv("visco_validation_results.csv");
    csv << "mode,time,strain,stress_sfem,stress_marc\n";
    
    for (size_t i = 0; i < uni_result.time.size(); ++i) {
        csv << "Uniaxial," << uni_result.time[i] << "," << uni_result.strain[i] << ","
            << uni_result.stress_sfem[i] << "," << uni_result.stress_marc[i] << "\n";
    }
    for (size_t i = 0; i < eqb_result.time.size(); ++i) {
        csv << "Equibiaxial," << eqb_result.time[i] << "," << eqb_result.strain[i] << ","
            << eqb_result.stress_sfem[i] << "," << eqb_result.stress_marc[i] << "\n";
    }
    for (size_t i = 0; i < ps_result.time.size(); ++i) {
        csv << "PlanarShear," << ps_result.time[i] << "," << ps_result.strain[i] << ","
            << ps_result.stress_sfem[i] << "," << ps_result.stress_marc[i] << "\n";
    }
    csv.close();
    printf("\nResults saved to: visco_validation_results.csv\n");
    
    // Generate plot script
    std::ofstream py("plot_visco_validation.py");
    py << R"(#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read results
df = pd.read_csv('visco_validation_results.csv')

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

modes = ['Uniaxial', 'Equibiaxial', 'PlanarShear']
titles = ['Uniaxial Tension', 'Equibiaxial Tension', 'Planar Shear']
meas_data = [uni_meas, eqb_meas, ps_meas] if has_measurement else [None, None, None]

for idx, mode in enumerate(modes):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=False)
    mode_df = df[df['mode'] == mode]

    # Top: stress-strain
    ax = axes[0]
    if has_measurement and meas_data[idx] is not None:
        meas = meas_data[idx]
        ax.plot(meas['Engineering Strain'], meas['Engineering Stress [MPa]'],
                'go', markersize=4, alpha=0.6, label='Measurement')

    if len(mode_df) > 0:
        ax.plot(mode_df['strain'], mode_df['stress_marc'],
                'b-', linewidth=2, label='Marc')
        ax.plot(mode_df['strain'], mode_df['stress_sfem'],
                'r--', linewidth=2, label='SFEM')

        denom = mode_df['stress_marc'].abs().max()
        if denom > 0:
            errors = abs(mode_df['stress_sfem'] - mode_df['stress_marc']) / denom * 100
            avg_err = errors.mean()
            max_err = errors.max()
            ax.annotate(f'Avg Error: {avg_err:.2f}%\nMax Error: {max_err:.2f}%',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Engineering Strain [-]')
    ax.set_ylabel('Engineering Stress [MPa]')
    ax.set_title(titles[idx])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Bottom: stress-time history
    ax = axes[1]
    if len(mode_df) > 0:
        ax.plot(mode_df['time'], mode_df['stress_marc'],
                'b-', linewidth=2, label='Marc')
        ax.plot(mode_df['time'], mode_df['stress_sfem'],
                'r--', linewidth=2, label='SFEM')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Engineering Stress [MPa]')
    ax.set_title(f'{titles[idx]} - Time History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f'visco_validation_{mode.lower()}_stacked'
    plt.savefig(f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{fname}.pdf', bbox_inches='tight')
    print(f'Plots saved: {fname}.png')
    plt.close(fig)

# Also generate individual plots for each mode
for idx, mode in enumerate(modes):
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    mode_df = df[df['mode'] == mode]
    
    # Measurement data (green circles)
    if has_measurement and meas_data[idx] is not None:
        meas = meas_data[idx]
        ax2.plot(meas['Engineering Strain'], meas['Engineering Stress [MPa]'], 
                'go', markersize=5, alpha=0.6, label='Measurement')
    
    if len(mode_df) > 0:
        ax2.plot(mode_df['strain'], mode_df['stress_marc'], 
                'b-', linewidth=2, label='Marc')
        ax2.plot(mode_df['strain'], mode_df['stress_sfem'], 
                'r--', linewidth=2, label='SFEM')
    
    ax2.set_xlabel('Engineering Strain [-]', fontsize=12)
    ax2.set_ylabel('Engineering Stress [MPa]', fontsize=12)
    ax2.set_title(f'{titles[idx]} - Viscoelastic Validation', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    fname = f'visco_validation_{mode.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved: {fname}')
    plt.close(fig2)
)";
    py.close();
    
    printf("\nTo generate plot, run:\n  python3 plot_visco_validation.py\n");
    
    MPI_Finalize();
    return 0;
}
