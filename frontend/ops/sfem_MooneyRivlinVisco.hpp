#pragma once
#include "sfem_Op.hpp"

namespace sfem {
    class MooneyRivlinVisco final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);
        
        const char *name() const override { return "MooneyRivlinVisco"; }
        inline bool is_linear() const override { return false; }
        int initialize(const std::vector<std::string> &block_names = {}) override;
        MooneyRivlinVisco(const std::shared_ptr<FunctionSpace> &space);
        ~MooneyRivlinVisco();
        
        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
                        
        int hessian_diag(const real_t *const x, real_t *const out) override;
        int gradient(const real_t *const x, real_t *const out) override;
        
        // Viscoelasticity specific: history update
        int update_history(const real_t *const x);
        
        // Setters for material parameters
        void set_C10(const real_t val);
        void set_K(const real_t val);
        void set_C01(const real_t val);
        void set_dt(const real_t val);
        void set_prony_terms(const int n, const real_t *g, const real_t *tau);
        
        // Initialize history buffer
        void initialize_history();
        
        // Set flexible mode:
        // false = FIXED (stores S_dev, hardcoded 10 Prony terms)
        // true  = FLEXIBLE (stores only H_i, uses precomputed alpha/beta/gamma, optimized)
        void set_use_flexible(bool flexible);
        
        // WLF (Williams-Landel-Ferry) time-temperature superposition
        // Formula: log10(a_T) = C1 * (T - T_ref) / (C2 + T - T_ref)
        // Effect: tau_eff = tau_ref / a_T
        //   When T > T_ref: a_T > 1, tau_eff < tau_ref (faster relaxation)
        //   When T < T_ref: a_T < 1, tau_eff > tau_ref (slower relaxation)
        // Note: All temperatures in °C
        // Default EPDM values: C1=16.6253, C2=47.4781°C, T_ref=-54.29°C
        void set_wlf_params(real_t C1, real_t C2, real_t T_ref);  // C2 and T_ref in °C
        void set_temperature(real_t T);  // Current temperature in °C
        void enable_wlf(bool enable);

        int hessian_bsr(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            return SFEM_FAILURE;
        }

        int value(const real_t *x, real_t *const out) override {
            return SFEM_FAILURE;
        }

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

