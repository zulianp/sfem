#pragma once

#include <memory>
#include <vector>

#include "sfem_HistoryStorage.hpp"
#include "sfem_PronySeries.hpp"
#include "sfem_QuadratureState.hpp"
#include "sfem_RelaxationIntegrator.hpp"
#include "sfem_TimeTemperatureShift.hpp"

namespace sfem {

    /// The internal variables of a Prony series on one block of elements, and
    /// the scheme that steps them.
    ///
    /// Material independent.  A Simo-type viscoelastic material splits its
    /// energy into an equilibrium part and a relaxed part psi_r, and within a
    /// step its algorithmic second Piola-Kirchhoff stress is
    ///
    ///     S = S_e + gamma S_r + S_hist,
    ///     S_hist = sum_i (alpha_i H_i - b_i S_r^n),   gamma = g_inf + sum_i b_i,
    ///
    /// with S_hist and gamma fixed for the whole step.  That makes the step the
    /// minimisation of psi_e + gamma psi_r + 1/2 S_hist : (C - I), which is
    /// what the generated kernels compute: they read `gamma()` and `history()`
    /// and know nothing else about the series.  This class owns everything
    /// they do not -- the spectrum, the time-temperature shift, the
    /// integrator and H_i -- and two calls move it:
    ///
    ///  - `begin_step(t, dt, S_n)` computes the step's weights and fills S_hist
    ///    from the committed H_i.  Everything a solve reads is fixed here, so a
    ///    line search that re-assembles at every trial step reads one history.
    ///  - `advance(S_n, S_new)` commits the step.
    ///
    /// H_i is the only stored state.  The relaxed stress S_r^n is not stored:
    /// it is a function of the converged displacement, so the caller evaluates
    /// it from that displacement whenever it is needed.  That costs one
    /// evaluation per step boundary and buys two things -- the memory of a
    /// tensor per point, and exactness: S_r^n is then bit for bit the S_r the
    /// previous `advance` committed, so the increments telescope, and the
    /// precision H_i is stored in never touches the stress it is driven by.
    /// It assumes S_r depends on the displacement and on parameters that do
    /// not change between steps.
    ///
    /// Before the first `begin_step`, gamma = 1 and S_hist = 0: the
    /// instantaneous response, which is also the dt -> 0 limit of a step from
    /// a stress-free state.  A series with no terms reads no stress, and its
    /// callers may pass null.
    ///
    /// Arrays are laid out [component][qp][element] like `QuadratureState`;
    /// the integration-point order is the kernels' and is never interpreted
    /// here.
    class PronyRelaxation {
    public:
        struct Options {
            std::shared_ptr<RelaxationIntegrator> integrator;         ///< default: LinearInterpolationIntegrator
            std::shared_ptr<TimeTemperatureShift> shift;              ///< default: NoShift
            real_t                                temperature{0};     ///< read through `shift` at every begin_step
            HistoryStorage                        history_storage{HistoryStorage::FULL};  ///< for H_i
            ExecutionSpace                        execution_space{EXECUTION_SPACE_HOST};  ///< where the state lives
            void                                 *stream{nullptr};  ///< the device loops' stream
        };

        PronyRelaxation(PronySeries series, const int n_components, const int n_qp, const ptrdiff_t n_elements);
        PronyRelaxation(PronySeries     series,
                        const int       n_components,
                        const int       n_qp,
                        const ptrdiff_t n_elements,
                        Options         options);

        /// Takes effect at the next `begin_step`.
        void   set_temperature(const real_t temperature);
        real_t temperature() const { return temperature_; }

        /// Open a step of length `dt`: compute the weights for the reduced step
        /// dt / a_T and fill the frozen history from H_i and `relaxed_stress_n`,
        /// the relaxed stress at the converged state the step starts from.
        /// May be called again to retry the same step with another `dt`.
        void begin_step(const real_t t, const real_t dt, const real_t *const relaxed_stress_n);

        /// Commit the open step: `relaxed_stress_n` at the state it started
        /// from, `relaxed_stress` at the converged state it ends at.
        void advance(const real_t *const relaxed_stress_n, const real_t *const relaxed_stress);

        /// Restart from rest: H_i = 0.  The relaxed stress of the state at rest
        /// is the caller's to evaluate at the next `begin_step`, as always.
        void reset();

        /// What the kernels read.
        real_t        gamma() const { return gamma_; }
        const real_t *history() const { return history_->data(); }

        const PronySeries          &series() const { return series_; }
        const Options              &options() const { return options_; }
        void                        set_stream(void *const stream) { options_.stream = stream; }
        const RelaxationIntegrator &integrator() const { return *options_.integrator; }
        int                         n_components() const { return n_components_; }
        int                         n_qp() const { return n_qp_; }
        ptrdiff_t                   n_elements() const { return n_elements_; }
        bool                        step_open() const { return step_open_; }

        /// The weights of the open step.
        real_t alpha(const int i) const { return alpha_[i]; }
        real_t b(const int i) const { return b_[i]; }

        /// The committed state: tensor i of `internal_variables()` is H_i.
        const QuadratureState &internal_variables() const { return H_; }

        /// Bytes of state held, frozen history included.
        std::size_t bytes() const;

    private:
        void zero_history();

        PronySeries                     series_;
        Options                         options_;
        int                             n_components_;
        int                             n_qp_;
        ptrdiff_t                       n_elements_;
        real_t                          temperature_;
        QuadratureState                 H_;
        std::shared_ptr<Buffer<real_t>> history_;
        std::vector<real_t>             alpha_;
        std::vector<real_t>             b_;
        real_t                          gamma_{1};
        bool                            step_open_{false};
    };

}  // namespace sfem
