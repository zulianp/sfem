#include "sfem_Merit.hpp"

#include <cmath>
#include <limits>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

namespace sfem {

    namespace {
        void non_finite_to_infinity(const int n, real_t *const out) {
            for (int i = 0; i < n; ++i) {
                if (!std::isfinite(out[i])) out[i] = std::numeric_limits<real_t>::infinity();
            }
        }

        class EnergyMerit final : public Merit {
        public:
            explicit EnergyMerit(const std::shared_ptr<Function> &f)
                : f_(f), blas_(blas<real_t>(f->execution_space())), ndofs_(f->space()->n_dofs()) {}

            const char *name() const override { return "energy"; }

            int evaluate(const real_t *const x,
                         const real_t *const h,
                         const int           nsteps,
                         const real_t *const steps,
                         real_t *const       out) override {
                for (int i = 0; i < nsteps; ++i) out[i] = 0;
                if (f_->energy_merit(x, h, nsteps, steps, out) != SFEM_SUCCESS) {
                    for (int i = 0; i < nsteps; ++i) out[i] = std::numeric_limits<real_t>::infinity();
                }
                non_finite_to_infinity(nsteps, out);
                return SFEM_SUCCESS;
            }

            real_t at(const real_t *const x, const real_t *const) override {
                const real_t zero = 0;
                real_t       m    = 0;
                evaluate(x, x, 1, &zero, &m);
                return m;
            }

            real_t slope(const real_t *const, const real_t *const r, const real_t *const h, const real_t) override {
                return blas_->dot(ndofs_, r, h);
            }

            bool batched() const override { return true; }

        private:
            std::shared_ptr<Function>     f_;
            std::shared_ptr<BLAS<real_t>> blas_;
            ptrdiff_t                     ndofs_;
        };

        class ResidualMerit final : public Merit {
        public:
            explicit ResidualMerit(const std::shared_ptr<Function> &f)
                : f_(f), blas_(blas<real_t>(f->execution_space())), ndofs_(f->space()->n_dofs()) {}

            const char *name() const override { return "residual"; }

            int evaluate(const real_t *const x,
                         const real_t *const h,
                         const int           nsteps,
                         const real_t *const steps,
                         real_t *const       out) override {
                for (int i = 0; i < nsteps; ++i) out[i] = 0;
                if (f_->residual_merit(x, h, nsteps, steps, out) != SFEM_SUCCESS) {
                    for (int i = 0; i < nsteps; ++i) out[i] = std::numeric_limits<real_t>::infinity();
                }
                non_finite_to_infinity(nsteps, out);
                return SFEM_SUCCESS;
            }

            real_t at(const real_t *const, const real_t *const r) override {
                const real_t norm = blas_->norm2(ndofs_, r);
                return real_t(0.5) * norm * norm;
            }

            real_t slope(const real_t *const, const real_t *const, const real_t *const, const real_t m0) override {
                return -2 * m0;
            }

            bool batched() const override { return true; }

        private:
            std::shared_ptr<Function>     f_;
            std::shared_ptr<BLAS<real_t>> blas_;
            ptrdiff_t                     ndofs_;
        };
    }  // namespace

    std::shared_ptr<Merit> create_energy_merit(const std::shared_ptr<Function> &f) {
        return std::make_shared<EnergyMerit>(f);
    }

    std::shared_ptr<Merit> create_residual_merit(const std::shared_ptr<Function> &f) {
        return std::make_shared<ResidualMerit>(f);
    }

    std::shared_ptr<Merit> create_merit(const std::shared_ptr<Function> &f, const std::string &choice) {
        if (choice == "energy") {
            if (!f->has_energy_merit()) {
                fprintf(stderr, "create_merit: \"energy\" requested, but an operator of this function has no potential\n");
                return nullptr;
            }
            return create_energy_merit(f);
        }
        if (choice == "residual") return create_residual_merit(f);
        if (choice == "auto") return f->has_energy_merit() ? create_energy_merit(f) : create_residual_merit(f);

        fprintf(stderr, "create_merit: unknown merit \"%s\" (energy, residual or auto)\n", choice.c_str());
        return nullptr;
    }

}  // namespace sfem
