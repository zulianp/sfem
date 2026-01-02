#include "sfem_MooneyRivlinVisco.hpp"

#include "mooney_rivlin_visco.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_Env.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_API.hpp"

#include <math.h>
#include <mpi.h>
#include <vector>
#include <sstream>
#include <string>

namespace sfem {

    // Helper function to parse comma-separated values from string
    static std::vector<real_t> parse_csv_values(const std::string& str) {
        std::vector<real_t> values;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                token = token.substr(start, end - start + 1);
                values.push_back(std::stod(token));
            }
        }
        return values;
    }

    class MooneyRivlinVisco::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        
        // Material parameters
        real_t C10{1}, K{1}, C01{1}, dt{0.1};
        
        // Prony Series parameters (at reference temperature)
        int num_prony_terms{0};
        std::vector<real_t> prony_g;
        std::vector<real_t> prony_tau;  // tau at T_ref
        
        // WLF (Williams-Landel-Ferry) temperature shift parameters
        // log10(a_T) = C1 * (T - T_ref) / (C2 + T - T_ref)
        // When T > T_ref: a_T > 1, tau_eff = tau/a_T < tau (faster relaxation)
        // Default values from EPDM rubber validation data (T_g ≈ -54°C)
        real_t wlf_C1{16.6253};      // From EPDM fitting
        real_t wlf_C2{47.4781};      // °C (same unit as temperature input)
        real_t wlf_T_ref{-54.29};    // Reference temperature (°C), glass transition
        real_t current_T{20.0};      // Current temperature (°C), room temperature
        bool use_wlf{false};         // Enable WLF shift
        

        std::vector<real_t> prony_alpha;  // exp(-dt/tau_eff_i) for active terms
        std::vector<real_t> prony_beta;   // g_i * (1 - alpha_i) / (dt/tau_eff_i) for active terms
        real_t prony_gamma{1.0};          // g_inf + sum(g_i for relaxed) + sum(beta_i for active)
        int num_active_terms{0};          // Number of active Prony terms
        
        // History buffer
        std::shared_ptr<Buffer<real_t>> history_buffer;
        std::shared_ptr<Buffer<real_t>> new_history_buffer;
        
        // Previous displacement
        std::shared_ptr<Buffer<real_t>> prev_u_buffer;

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        ~Impl();
        
        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
        
        // Compute WLF shift factor a_T
        // log10(a_T) = C1 * (T - T_ref) / (C2 + T - T_ref)
        // tau_eff = tau_ref / a_T
        real_t compute_wlf_shift() const {
            if (!use_wlf) return 1.0;
            
            real_t dT = current_T - wlf_T_ref;
            real_t denom = wlf_C2 + dT;
            
            // Prevent division by zero
            if (fabs(denom) < 1e-10) {
                return 1.0;
            }
            
            // log10(a_T) = C1 * dT / (C2 + dT)
            real_t log10_aT = wlf_C1 * dT / denom;
            real_t aT = pow(10.0, log10_aT);
            
            return aT;
        }
        
        void compute_prony_coefficients() {
            // Compute WLF shift factor
            real_t aT = compute_wlf_shift();
            
            // g_inf = 1 - sum(g_i)
            // For pure elastic (num_prony_terms = 0): g_inf = 1.0
            // For viscoelastic: g_inf = 1 - sum(g_i) is the long-term modulus ratio
            real_t sum_g = 0;
            for (int i = 0; i < num_prony_terms; ++i) {
                sum_g += prony_g[i];
            }
            real_t g_inf = 1.0 - sum_g;
            

            const real_t relax_threshold = 100.0;
            
            // First pass: count active terms and compute gamma
            // For Short Term params: gamma = g_inf + Σβ_i
            prony_gamma = g_inf;  
            num_active_terms = 0;
            
            for (int i = 0; i < num_prony_terms; ++i) {
                real_t tau_eff = prony_tau[i] / aT;
                real_t x = dt / tau_eff;
                
                if (x > relax_threshold) {
                    // Fully relaxed: term has equilibrated within one time step
                    // At equilibrium, Prony term contribution → 0 (NOT g_i!)
                    // Prony series represents deviation from equilibrium
                    // Do NOT add g_i to gamma!
                } else {
                    // Active term: will contribute alpha, beta
                    num_active_terms++;
                }
            }
            
            printf("[compute_prony_coefficients] aT=%g dt=%g relax_threshold=%g num_prony_terms=%d\\n",
                   (double)aT, (double)dt, (double)relax_threshold, num_prony_terms);
            for (int i = 0; i < num_prony_terms; ++i) {
                real_t tau_eff = prony_tau[i] / aT;
                real_t x = dt / tau_eff;
                bool active = (x <= relax_threshold);
                printf("  term %2d: tau=%g tau_eff=%g dt/tau_eff=%g g=%g active=%s\n",
                       i+1, (double)prony_tau[i], (double)tau_eff, (double)x, (double)prony_g[i], active ? "YES" : "NO");
            }
            
            // Second pass: compute alpha, beta for active terms only
            prony_alpha.resize(num_active_terms);
            prony_beta.resize(num_active_terms);
            
            int active_idx = 0;
            for (int i = 0; i < num_prony_terms; ++i) {
                real_t tau_eff = prony_tau[i] / aT;
                real_t x = dt / tau_eff;
                
                if (x <= relax_threshold) {
                    // Active term
                    prony_alpha[active_idx] = exp(-x);
                    prony_beta[active_idx] = prony_g[i] * (1.0 - prony_alpha[active_idx]) / x;
                    // Add beta to gamma for correct S_total = gamma_eff * S_curr + S_hist
                    prony_gamma += prony_beta[active_idx];
                    active_idx++;
                }
            }
            printf("[compute_prony_coefficients] num_active_terms=%d prony_gamma=%g\\n", num_active_terms, (double)prony_gamma);
        }
        
        ptrdiff_t history_per_qp() const {
            // Store H_i for active terms only
            return num_active_terms * 6;
        }
        
        void allocate_history_buffers() {
            ptrdiff_t total_elements = 0;
            iterate([&](const OpDomain &domain) -> int {
                total_elements += domain.block->n_elements();
                return SFEM_SUCCESS;
            });

            const int n_qp = 8; 
            const ptrdiff_t total_size = total_elements * n_qp * history_per_qp();
            
            history_buffer = create_buffer<real_t>(total_size, sfem::EXECUTION_SPACE_HOST);
            new_history_buffer = create_buffer<real_t>(total_size, sfem::EXECUTION_SPACE_HOST);
            
            auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
            blas->zeros(total_size, history_buffer->data());
            blas->zeros(total_size, new_history_buffer->data());
            
            // Allocate prev_u buffer
            ptrdiff_t ndofs = space->mesh_ptr()->n_nodes() * 3;
            prev_u_buffer = create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
            blas->zeros(ndofs, prev_u_buffer->data());
        }
        
        void swap_history_buffers() {
            std::swap(history_buffer, new_history_buffer);
        }
        
        void save_prev_u(const real_t *x) {
            if (prev_u_buffer) {
                ptrdiff_t ndofs = space->mesh_ptr()->n_nodes() * 3;
                memcpy(prev_u_buffer->data(), x, ndofs * sizeof(real_t));
            }
        }
    };

    std::unique_ptr<Op> MooneyRivlinVisco::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::create");
        auto ret = std::make_unique<MooneyRivlinVisco>(space);
        
        // Material parameters
        ret->impl_->C10 = sfem::Env::read("SFEM_MOONEY_RIVLIN_C10", ret->impl_->C10);
        ret->impl_->K   = sfem::Env::read("SFEM_MOONEY_RIVLIN_K", ret->impl_->K);
        ret->impl_->C01 = sfem::Env::read("SFEM_MOONEY_RIVLIN_C01", ret->impl_->C01);
        ret->impl_->dt  = sfem::Env::read("SFEM_DT", ret->impl_->dt);
        
        // WLF temperature shift parameters (all temperatures in °C)
        ret->impl_->wlf_C1    = sfem::Env::read("SFEM_WLF_C1", ret->impl_->wlf_C1);
        ret->impl_->wlf_C2    = sfem::Env::read("SFEM_WLF_C2", ret->impl_->wlf_C2);
        ret->impl_->wlf_T_ref = sfem::Env::read("SFEM_WLF_T_REF", ret->impl_->wlf_T_ref);
        ret->impl_->current_T = sfem::Env::read("SFEM_TEMPERATURE", ret->impl_->current_T);
        ret->impl_->use_wlf   = sfem::Env::read("SFEM_USE_WLF", 0) != 0;
        
        printf("[MooneyRivlinVisco::create] WLF params: C1=%g, C2=%g, T_ref=%g, T=%g, use_wlf=%d\n",
               (double)ret->impl_->wlf_C1, (double)ret->impl_->wlf_C2, 
               (double)ret->impl_->wlf_T_ref, (double)ret->impl_->current_T,
               ret->impl_->use_wlf ? 1 : 0);
        
        // Prony series parameters (comma-separated values)
        // Example: SFEM_PRONY_G="0.15,0.15,0.10,0.05" SFEM_PRONY_TAU="0.1,1.0,10.0,100.0"
        std::string prony_g_str = sfem::Env::read("SFEM_PRONY_G", std::string(""));
        std::string prony_tau_str = sfem::Env::read("SFEM_PRONY_TAU", std::string(""));
        
        if (!prony_g_str.empty() && !prony_tau_str.empty()) {
            std::vector<real_t> g_values = parse_csv_values(prony_g_str);
            std::vector<real_t> tau_values = parse_csv_values(prony_tau_str);
            
            if (g_values.size() == tau_values.size() && !g_values.empty()) {
                ret->impl_->num_prony_terms = static_cast<int>(g_values.size());
                ret->impl_->prony_g = std::move(g_values);
                ret->impl_->prony_tau = std::move(tau_values);
                ret->impl_->compute_prony_coefficients();
                printf("[MooneyRivlinVisco::create] Loaded %d Prony terms from environment\n", 
                       ret->impl_->num_prony_terms);
            } else {
                SFEM_ERROR("SFEM_PRONY_G and SFEM_PRONY_TAU must have the same number of values!");
            }
        }
        
        return ret;
    }

    MooneyRivlinVisco::MooneyRivlinVisco(const std::shared_ptr<FunctionSpace> &space) 
        : impl_(std::make_unique<Impl>(space)) {}

    MooneyRivlinVisco::~MooneyRivlinVisco() = default;

    int MooneyRivlinVisco::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        return SFEM_SUCCESS;
    }
    
    void MooneyRivlinVisco::initialize_history() {
        // Compute coefficients first (need num_active_terms for buffer size)
        if (impl_->num_prony_terms > 0) {
            impl_->compute_prony_coefficients();
        }
        impl_->allocate_history_buffers();
    }

    int MooneyRivlinVisco::hessian_crs(const real_t *const x,
                                       const count_t *const rowptr,
                                       const idx_t *const colidx,
                                       real_t *const values) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::hessian_crs");
        return hessian_bsr(x, rowptr, colidx, values);
    }

    int MooneyRivlinVisco::hessian_bsr(const real_t *const x,
                                       const count_t *const rowptr,
                                       const idx_t *const colidx,
                                       real_t *const values) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::hessian_bsr");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer) {
            SFEM_ERROR("History buffer not initialized! Call initialize_history() first.");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        const int n_qp = 8; 
        const ptrdiff_t history_stride = n_qp * impl_->history_per_qp();

        return impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            int ret = mooney_rivlin_visco_bsr_flexible(
                domain.element_type,
                nelements,
                mesh->n_nodes(),
                domain.block->elements()->data(),
                mesh->points()->data(),
                impl_->C10,
                impl_->C01,
                impl_->K,
                impl_->num_active_terms,
                impl_->prony_alpha.data(),
                impl_->prony_beta.data(),
                impl_->prony_gamma,
                history_stride,
                impl_->history_buffer->data() + history_offset,
                3, 
                &impl_->prev_u_buffer->data()[0],
                &impl_->prev_u_buffer->data()[1],
                &impl_->prev_u_buffer->data()[2],
                &x[0], &x[1], &x[2],
                rowptr, colidx, values);
                
            history_offset += nelements * history_stride;
            return ret;
        });
    }

    int MooneyRivlinVisco::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::hessian_diag");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer) {
            SFEM_ERROR("History buffer not initialized! Call initialize_history() first.");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        const int n_qp = 8; 
        const ptrdiff_t history_stride = n_qp * impl_->history_per_qp();

        return impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            int ret = mooney_rivlin_visco_hessian_diag_flexible(
                domain.element_type,
                nelements,
                mesh->n_nodes(),
                domain.block->elements()->data(),
                mesh->points()->data(),
                impl_->C10,
                impl_->C01,
                impl_->K,
                impl_->num_active_terms,
                impl_->prony_alpha.data(),
                impl_->prony_beta.data(),
                impl_->prony_gamma,
                history_stride,
                impl_->history_buffer->data() + history_offset,
                3,
                &impl_->prev_u_buffer->data()[0],
                &impl_->prev_u_buffer->data()[1],
                &impl_->prev_u_buffer->data()[2],
                &x[0], &x[1], &x[2],
                out);
                
            history_offset += nelements * history_stride;
            return ret;
        });
    }

    int MooneyRivlinVisco::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::gradient");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer) {
            SFEM_ERROR("History buffer not initialized! Call initialize_history() first.");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        const int n_qp = 8; 
        const ptrdiff_t history_stride = n_qp * impl_->history_per_qp();

        return impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            int ret = mooney_rivlin_visco_gradient_flexible(
                domain.element_type,
                nelements,
                mesh->n_nodes(),
                domain.block->elements()->data(),
                mesh->points()->data(),
                impl_->C10,
                impl_->C01,
                impl_->K,
                impl_->num_active_terms,
                impl_->prony_alpha.data(),
                impl_->prony_beta.data(),
                impl_->prony_gamma,
                history_stride,
                impl_->history_buffer->data() + history_offset,
                3,
                &impl_->prev_u_buffer->data()[0],
                &impl_->prev_u_buffer->data()[1],
                &impl_->prev_u_buffer->data()[2],
                &x[0], &x[1], &x[2],
                out);
                
            history_offset += nelements * history_stride;
            return ret;
        });
    }

    int MooneyRivlinVisco::update_history(const real_t *const x) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::update_history");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer || !impl_->new_history_buffer) {
            SFEM_ERROR("History buffers not initialized!");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        const int n_qp = 8; 
        const ptrdiff_t history_stride = n_qp * impl_->history_per_qp();

        int ret = impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            int r = mooney_rivlin_visco_update_history_flexible(
                domain.element_type,
                nelements,
                mesh->n_nodes(),
                domain.block->elements()->data(),
                mesh->points()->data(),
                impl_->C10,
                impl_->C01,
                impl_->K,
                impl_->num_active_terms,
                impl_->prony_alpha.data(),
                impl_->prony_beta.data(),
                history_stride,
                impl_->history_buffer->data() + history_offset,
                impl_->new_history_buffer->data() + history_offset,
                3,
                &impl_->prev_u_buffer->data()[0],
                &impl_->prev_u_buffer->data()[1],
                &impl_->prev_u_buffer->data()[2],
                &x[0], &x[1], &x[2]);
                
            history_offset += nelements * history_stride;
            return r;
        });
        
        if(ret == SFEM_SUCCESS) {
            impl_->swap_history_buffers();
            impl_->save_prev_u(x);
        }
        return ret;
    }

    void MooneyRivlinVisco::set_C10(const real_t val) { impl_->C10 = val; }
    void MooneyRivlinVisco::set_K(const real_t val) { impl_->K = val; }
    void MooneyRivlinVisco::set_C01(const real_t val) { impl_->C01 = val; }
    
    void MooneyRivlinVisco::set_dt(const real_t val) { 
        impl_->dt = val;
    }
    
    void MooneyRivlinVisco::set_prony_terms(const int n, const real_t *g, const real_t *tau) {
        impl_->num_prony_terms = n;
        impl_->prony_g.assign(g, g + n);
        impl_->prony_tau.assign(tau, tau + n);
    }
    
    void MooneyRivlinVisco::set_wlf_params(real_t C1, real_t C2, real_t T_ref) {
        impl_->wlf_C1 = C1;
        impl_->wlf_C2 = C2;
        impl_->wlf_T_ref = T_ref;
    }
    
    void MooneyRivlinVisco::set_temperature(real_t T) {
        impl_->current_T = T;
    }
    
    void MooneyRivlinVisco::enable_wlf(bool enable) {
        impl_->use_wlf = enable;
    }
    
    real_t MooneyRivlinVisco::get_gamma() const {
        return impl_->prony_gamma;
    }
    
    int MooneyRivlinVisco::get_num_active_terms() const {
        return impl_->num_active_terms;
    }
    
    void MooneyRivlinVisco::set_prony_coefficients(int n_active, const real_t* alpha, const real_t* beta, real_t gamma) {
        // Directly set precomputed coefficients (bypasses internal WLF/filtering)
        impl_->num_active_terms = n_active;
        impl_->prony_alpha.resize(n_active);
        impl_->prony_beta.resize(n_active);
        
        for (int i = 0; i < n_active; ++i) {
            impl_->prony_alpha[i] = alpha[i];
            impl_->prony_beta[i] = beta[i];
        }
        impl_->prony_gamma = gamma;
        
        // Reinitialize history buffer with new number of active terms
        if (impl_->history_buffer) {
            impl_->allocate_history_buffers();
        }
    }
    
    void MooneyRivlinVisco::reinitialize() {
        // Recompute coefficients with current parameters
        // Call this after changing temperature, WLF params, dt, etc.
        if (impl_->num_prony_terms > 0) {
            int old_active = impl_->num_active_terms;
            impl_->compute_prony_coefficients();
            
            // Reallocate history buffer if num_active_terms changed
            if (impl_->history_buffer && old_active != impl_->num_active_terms) {
                printf("[reinitialize] num_active_terms changed %d -> %d, reallocating history\n", 
                       old_active, impl_->num_active_terms);
                impl_->allocate_history_buffers();
            }
        }
    }

    MooneyRivlinVisco::Impl::~Impl() {}

}  // namespace sfem

