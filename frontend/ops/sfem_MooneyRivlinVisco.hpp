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
        // Updates history variables based on current displacement x
        int update_history(const real_t *const x);
        
        // Setters for material parameters
        void set_C10(const real_t val);
        void set_K(const real_t val);
        void set_C01(const real_t val);
        void set_dt(const real_t val);
        void set_prony_terms(const int n, const real_t *g, const real_t *tau);
        
        // Initialize history buffer
        void initialize_history();

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
        // Updates history variables based on current displacement x
        int update_history(const real_t *const x);
        
        // Setters for material parameters
        void set_C10(const real_t val);
        void set_K(const real_t val);
        void set_C01(const real_t val);
        void set_dt(const real_t val);
        void set_prony_terms(const int n, const real_t *g, const real_t *tau);
        
        // Initialize history buffer
        void initialize_history();

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


