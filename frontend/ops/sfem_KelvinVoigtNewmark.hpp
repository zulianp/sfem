#ifndef SFEM_KELVIN_VOIGT_NEWMARK_HPP
#define SFEM_KELVIN_VOIGT_NEWMARK_HPP

#include "sfem_Op.hpp"

namespace sfem {

    /**
     * @brief Kelvin-Voigt viscoelastic operator with Newmark time integration
     */
    class KelvinVoigtNewmark final : public Op {
    public:
        const char *name() const override { return "KelvinVoigtNewmark"; }
        inline bool is_linear() const override { return true; }

        // Factory
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        // LOR / derefine
        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;

        // Lifecycle
        int initialize(const std::vector<std::string> &block_names = {}) override;
        ~KelvinVoigtNewmark();

        // Assembly / actions
        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_bcrs_sym(const real_t *const  x,
                             const count_t *const rowptr,
                             const idx_t *const   colidx,
                             const ptrdiff_t      block_stride,
                             real_t **const       diag_values,
                             real_t **const       off_diag_values) override;

        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override;
        int hessian_block_diag_sym_soa(const real_t *const x, real_t **const values) override { return SFEM_FAILURE; }
        int hessian_diag(const real_t *const x, real_t *const values) override;

        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override { return nullptr; }

        void set_field(const char* name, const std::shared_ptr<Buffer<real_t>>& vel, int component) override;
        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override;

        // Accessors for stability/debug control
        real_t get_k() const;
        void   set_k(real_t val);
        real_t get_K() const;
        void   set_K(real_t val);
        real_t get_eta() const;
        void   set_eta(real_t val);
        real_t get_dt() const;
        void   set_dt(real_t val);
        real_t get_gamma() const;
        void   set_gamma(real_t val);
        real_t get_beta() const;
        void   set_beta(real_t val);
        real_t get_rho() const;
        void   set_rho(real_t val);

        // Ctors
        KelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    // Convenience factory for places expecting a free function
    std::unique_ptr<Op> create_kelvin_voigt_newmark(const std::shared_ptr<FunctionSpace> &space);
}

#endif // SFEM_KELVIN_VOIGT_NEWMARK_HPP
