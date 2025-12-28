/**
 * @file sfem_MRViscoMultiModeTest.cpp
 * @brief Multi-mode validation test for Mooney-Rivlin viscoelastic model
 * 
 * Supports three test modes:
 * - UNIAXIAL: λ₁ = λ, λ₂ = λ₃ = 1/√λ
 * - EQUIBIAXIAL: λ₁ = λ₂ = λ, λ₃ = 1/λ²
 * - PURE_SHEAR: λ₁ = λ, λ₂ = 1, λ₃ = 1/λ
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
#include "sfem_bsr_SpMV.hpp"
#include "sfem_cg.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_test.h"

enum TestMode {
    MODE_UNIAXIAL = 0,
    MODE_EQUIBIAXIAL = 1,
    MODE_PURE_SHEAR = 2
};

// EPDM material parameters
// g_inf = 1 - sum(g_i) = 0.004434 for the 22-term Prony series
// Short Term = Long Term / g_inf
// Long Term: C10 = 0.499, C01 = 0.577 MPa
// Short Term: C10 = 112.53, C01 = 130.12 MPa
static const double EPDM_G_INF = 0.004434;
static double EPDM_C10 = 0.499 / EPDM_G_INF;   // MPa (Short Term)
static double EPDM_C01 = 0.577 / EPDM_G_INF;   // MPa (Short Term)
static double EPDM_K = 10000.0;                // MPa

// Prony series (22 terms) - original full set
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

static const double prony_g_3[] = {
    0.20, 0.20, 0.15   // sum = 0.55, g_inf = 0.45
};
static const double prony_tau_3[] = {
    0.05, 0.5, 5.0     // span 2 decades around dt=0.01~0.05s
};

// WLF parameters
static const double WLF_C1 = 16.6253;
static const double WLF_C2 = 47.4781;
static const double WLF_T_REF = -54.29;  // °C

// Reference data arrays (will be loaded from file)
static std::vector<double> ref_times;
static std::vector<double> ref_strains;
static std::vector<double> ref_stresses;

// Linear interpolation helper
double interpolate(const std::vector<double>& times, const std::vector<double>& values, double t) {
    if (times.empty()) return 0.0;
    if (t <= times[0]) return values[0];
    if (t >= times.back()) return values.back();
    
    for (size_t i = 0; i < times.size() - 1; ++i) {
        if (t >= times[i] && t < times[i+1]) {
            double alpha = (t - times[i]) / (times[i+1] - times[i]);
            return values[i] + alpha * (values[i+1] - values[i]);
        }
    }
    return values.back();
}

// Calculate displacement field based on test mode
void apply_displacement(TestMode mode, double strain, ptrdiff_t n_nodes, geom_t **pts, real_t *u) {
    // Stretch ratio in primary direction
    double lambda = 1.0 + strain;
    
    double lambda_y, lambda_z;
    
    switch (mode) {
        case MODE_UNIAXIAL:
            // λ₁ = λ, λ₂ = λ₃ = 1/√λ (volume preserving)
            lambda_y = 1.0 / sqrt(lambda);
            lambda_z = 1.0 / sqrt(lambda);
            break;
            
        case MODE_EQUIBIAXIAL:
            // λ₁ = λ₂ = λ, λ₃ = 1/λ² (volume preserving)
            lambda_y = lambda;
            lambda_z = 1.0 / (lambda * lambda);
            break;
            
        case MODE_PURE_SHEAR:
            // λ₁ = λ, λ₂ = 1, λ₃ = 1/λ (volume preserving)
            lambda_y = 1.0;
            lambda_z = 1.0 / lambda;
            break;
            
        default:
            lambda_y = 1.0;
            lambda_z = 1.0;
    }
    
    // Convert stretch ratios to strains
    double strain_y = lambda_y - 1.0;
    double strain_z = lambda_z - 1.0;
    
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        double node_x = pts[0][i];
        double node_y = pts[1][i];
        double node_z = pts[2][i];
        
        u[3*i + 0] = strain * node_x;
        u[3*i + 1] = strain_y * node_y;
        u[3*i + 2] = strain_z * node_z;
    }
}

// Load reference data from CSV
bool load_reference_data(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Warning: Could not open reference file: %s\n", filename);
        return false;
    }
    
    ref_times.clear();
    ref_strains.clear();
    ref_stresses.clear();
    
    std::string line;
    std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        double t, strain, stress;
        if (sscanf(line.c_str(), "%lf,%lf,%lf", &t, &strain, &stress) == 3) {
            ref_times.push_back(t);
            ref_strains.push_back(strain);
            ref_stresses.push_back(stress);
        }
    }
    
    printf("Loaded %zu reference points from %s\n", ref_times.size(), filename);
    return !ref_times.empty();
}

// Get test mode name
const char* get_mode_name(TestMode mode) {
    switch (mode) {
        case MODE_UNIAXIAL: return "Uniaxial";
        case MODE_EQUIBIAXIAL: return "Equibiaxial";
        case MODE_PURE_SHEAR: return "Pure Shear";
        default: return "Unknown";
    }
}

int run_test(TestMode mode) {
    printf("\n=== EPDM %s Validation Test ===\n", get_mode_name(mode));
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto es = sfem::EXECUTION_SPACE_HOST;
    
    // Configuration
    double TEST_TEMPERATURE = 20.0;
    int USE_WLF = 0;
    double DT = 0.01;
    int VERBOSE = 1;
    
    SFEM_READ_ENV(TEST_TEMPERATURE, atof);
    SFEM_READ_ENV(USE_WLF, atoi);
    SFEM_READ_ENV(DT, atof);
    SFEM_READ_ENV(VERBOSE, atoi);
    SFEM_READ_ENV(EPDM_K, atof);
    
    // Using original Long Term parameters without scaling
    // prony_gamma = g_inf formula is restored
    
    printf("Configuration:\n");
    printf("  Mode: %s\n", get_mode_name(mode));
    printf("  Temperature: %.1f °C\n", TEST_TEMPERATURE);
    printf("  WLF enabled: %s\n", USE_WLF ? "yes" : "no");
    printf("  Time step: %.4f s\n", DT);
    printf("  Material: C10=%.3f, C01=%.3f, K=%.1f MPa\n", EPDM_C10, EPDM_C01, EPDM_K);
    
    // Create mesh (1x1x1 cube)
    auto mesh = sfem::Mesh::create_hex8_cube(
        sfem::Communicator::wrap(comm),
        1, 1, 1,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    );
    
    auto fs = sfem::FunctionSpace::create(mesh, 3);
    auto f = sfem::Function::create(fs);
    
    const ptrdiff_t ndofs = fs->n_dofs();
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    
    // Create operator
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    
    op->set_C10(EPDM_C10);
    op->set_C01(EPDM_C01);
    op->set_K(EPDM_K);
    op->set_dt(DT);
    op->set_use_flexible(true);
    
    op->set_prony_terms(NUM_PRONY, prony_g, prony_tau);
    
    if (USE_WLF) {
        op->set_wlf_params(WLF_C1, WLF_C2, WLF_T_REF);
        op->set_temperature(TEST_TEMPERATURE);
        op->enable_wlf(true);
    }
    
    op->initialize();
    op->initialize_history();
    f->add_operator(op);
    
    printf("  Prony terms: %d, Active: %d, Gamma: %.6f\n", 
           NUM_PRONY, op->get_num_active_terms(), op->get_gamma());
    
    // Buffers
    auto x = sfem::create_buffer<real_t>(ndofs, es);
    auto rhs = sfem::create_buffer<real_t>(ndofs, es);
    auto blas = sfem::blas<real_t>(es);
    
    blas->zeros(ndofs, x->data());
    
    // Get mesh points
    geom_t **pts = mesh->points()->data();

    // Pick the right-face corner node (max x; tie-break max y then max z) to mimic Excel post-process (ReactionForce(Node7)*4)
    ptrdiff_t right_node = 0;
    {
        double best_x = -1e30, best_y = -1e30, best_z = -1e30;
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            double x = pts[0][i], y = pts[1][i], z = pts[2][i];
            if (x > best_x + 1e-12 ||
               (fabs(x - best_x) < 1e-12 && (y > best_y + 1e-12 ||
               (fabs(y - best_y) < 1e-12 && z > best_z + 1e-12)))) {
                best_x = x; best_y = y; best_z = z;
                right_node = i;
            }
        }
        if (VERBOSE) {
            printf("  Using node %td for stress (max x corner), coords=(%g,%g,%g)\n",
                   right_node, pts[0][right_node], pts[1][right_node], pts[2][right_node]);
        }
    }
    
    // Output file
    std::string output_filename = std::string(get_mode_name(mode)) + "_validation_results.csv";
    // Replace spaces
    for (char& c : output_filename) if (c == ' ') c = '_';
    std::ofstream csv_out(output_filename);
    csv_out << "time,strain,stress_sfem,stress_marc,error_percent\n";
    
    // Time loop
    double t = 0.0;
    double t_end = ref_times.empty() ? 334.0 : ref_times.back();
    double T_MAX = t_end;
    SFEM_READ_ENV(T_MAX, atof);
    if (T_MAX < t_end) t_end = T_MAX;
    
    int step = 0;
    int output_interval = (int)(5.0 / DT);
    if (output_interval < 1) output_interval = 1;
    
    printf("\n%-10s %-12s %-14s %-14s %-10s\n", 
           "Time[s]", "Strain", "Stress_SFEM", "Stress_Marc", "Error[%]");
    printf("--------------------------------------------------------------\n");
    
    std::vector<double> sim_times, sim_strains, sim_stresses;
    
    while (t <= t_end + 1e-6) {
        // Get target strain
        double target_strain = ref_strains.empty() ? 0.0 : interpolate(ref_times, ref_strains, t);
        
        // Apply displacement based on mode
        apply_displacement(mode, target_strain, n_nodes, pts, x->data());
        
        // Compute internal forces
        blas->zeros(ndofs, rhs->data());
        op->gradient(x->data(), rhs->data());
        
        // Mimic Excel: take reaction X at selected corner node and multiply by 4
        double reaction_corner = rhs->data()[3 * right_node];
        double stress_sim = reaction_corner * 4.0;
        double stress_ref = ref_stresses.empty() ? 0.0 : interpolate(ref_times, ref_stresses, t);
        double error = (stress_ref > 0.01) ? 100.0 * fabs(stress_sim - stress_ref) / stress_ref : 0.0;
        
        sim_times.push_back(t);
        sim_strains.push_back(target_strain);
        sim_stresses.push_back(stress_sim);
        
        if (step % output_interval == 0 || t >= t_end - 1e-6) {
            printf("%-10.2f %-12.6f %-14.6f %-14.6f %-10.2f\n",
                   t, target_strain, stress_sim, stress_ref, error);
        }
        
        csv_out << t << "," << target_strain << "," << stress_sim << "," 
                << stress_ref << "," << error << "\n";
        
        // Update history
        op->update_history(x->data());
        
        t += DT;
        step++;
    }
    
    csv_out.close();
    printf("\nResults written to %s\n", output_filename.c_str());
    
    // Compute error metrics
    double max_error = 0.0, avg_error = 0.0;
    int count = 0;
    for (size_t i = 0; i < sim_stresses.size(); ++i) {
        double ref = interpolate(ref_times, ref_stresses, sim_times[i]);
        if (ref > 0.01 && sim_times[i] < 80.0) {  // Loading phase
            double err = fabs(sim_stresses[i] - ref) / ref * 100.0;
            if (err > max_error) max_error = err;
            avg_error += err;
            count++;
        }
    }
    if (count > 0) avg_error /= count;
    
    printf("\nError (loading phase):\n  Max: %.2f%%, Avg: %.2f%%\n", max_error, avg_error);
    
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    // Parse test mode from command line or env
    TestMode mode = MODE_UNIAXIAL;
    
    const char* mode_str = getenv("TEST_MODE");
    if (mode_str) {
        if (strcmp(mode_str, "equibiaxial") == 0 || strcmp(mode_str, "EQUIBIAXIAL") == 0) {
            mode = MODE_EQUIBIAXIAL;
        } else if (strcmp(mode_str, "pureshear") == 0 || strcmp(mode_str, "PURESHEAR") == 0 ||
                   strcmp(mode_str, "pure_shear") == 0 || strcmp(mode_str, "PURE_SHEAR") == 0) {
            mode = MODE_PURE_SHEAR;
        }
    }
    
    // Also check command line
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--equibiaxial") == 0) mode = MODE_EQUIBIAXIAL;
        if (strcmp(argv[i], "--pureshear") == 0) mode = MODE_PURE_SHEAR;
        if (strcmp(argv[i], "--uniaxial") == 0) mode = MODE_UNIAXIAL;
    }
    
    // Load reference data based on mode
    const char* ref_file = nullptr;
    switch (mode) {
        case MODE_UNIAXIAL:
            ref_file = "uniaxial_ref.csv";
            break;
        case MODE_EQUIBIAXIAL:
            ref_file = "equibiaxial_ref.csv";
            break;
        case MODE_PURE_SHEAR:
            ref_file = "pureshear_ref.csv";
            break;
    }
    
    // Try to load from env or default
    const char* ref_file_env = getenv("REF_FILE");
    if (ref_file_env) ref_file = ref_file_env;
    
    load_reference_data(ref_file);
    
    int err = run_test(mode);
    
    MPI_Finalize();
    return err;
}

