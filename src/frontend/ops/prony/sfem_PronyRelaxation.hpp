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
    /// A term that relaxes completely within the step -- alpha_i below half a
    /// unit of real_t's precision, so alpha_i H_i is below the rounding of H_i
    /// itself -- contributes only b_i to gamma, and its update is
    /// H_i^{n+1} = b_i (S_r^{n+1} - S_r^n): the same tensor for every such
    /// term, up to a scalar.  Those terms are therefore not stored: they hold
    /// a coefficient c_i on the last step's increment, H_i = c_i (S_r^n -
    /// S_r^{n-1}), which the caller's two stress slots hold, and it is
    /// materialised into a slot only if a shorter step brings the term back
    /// (`begin_step`'s `relaxed_stress_prev`).  Every term starts this way, at
    /// c_i = 0.  A spectrum fitted over many decades -- most of whose terms
    /// relax inside any step a solver takes -- then stores only the terms that
    /// are still relaxing.
    ///
    /// Remainder form.  A reduced history format (not FULL) rounds each stored
    /// tensor relative to its largest value, every step.  A term slower than
    /// the run (tau_i a_T > `Options::remainder_horizon`) barely relaxes over
    /// it, so H_i stays close to g_i (S_r^n - R), R the relaxed stress its
    /// history was last zero at -- the start of the run or the last `reset`.
    /// Such a term is stored as the remainder G_i = H_i - g_i (S_r^n - R), much
    /// smaller than H_i, and its codes resolve it that much finer.  R is held
    /// once, in the stress slots' precision.  The identity H_i = G_i + g_i (S_r^n
    /// - R) holds for the R stored, but G_i starts at g_i times R's rounding, and
    /// that must stay below the remainder: in int8 it costs most of the gain, in
    /// float32 nothing measurable.
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
            int history_elements_per_mean{0};  ///< mean-deviation storage: QuadratureState's elements_per_mean
            //! The relaxed-stress slots `RelaxationGroups` keeps for this group: FULL (real_t) or
            //! FLOAT32. A float32 slot holds the stress rounded once, and both ends of every
            //! step read the rounded values, so the increments still telescope.
            HistoryStorage stress_storage{HistoryStorage::FULL};
            //! The remainder form's horizon, in seconds: the terms with tau_i a_T above it are
            //! stored as remainders when `history_storage` is not FULL. 0: none; negative: not
            //! set (none here; `RelaxationGroups::set_default_remainder_horizon` sets it).
            real_t remainder_horizon{-1};
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
        /// Takes effect at the next `begin_step`.
        void   set_remainder_horizon(const real_t horizon) { options_.remainder_horizon = horizon; }
        real_t temperature() const { return temperature_; }

        /// Open a step of length `dt`: compute the weights for the reduced step
        /// dt / a_T and fill the frozen history from H_i and `relaxed_stress_n`,
        /// the relaxed stress at the converged state the step starts from.
        /// `relaxed_stress_prev` is the one the last committed step started
        /// from, which a term a shorter step brings back is rebuilt from; it
        /// may be null while no step has been committed since the start or the
        /// last `reset`.  May be called again to retry the same step with
        /// another `dt`.
        void begin_step(const real_t        t,
                        const real_t        dt,
                        const real_t *const relaxed_stress_n,
                        const real_t *const relaxed_stress_prev = nullptr);
#if SFEM_REAL_T_IS_FLOAT64
        /// The same from float32 stress slots (`Options::stress_storage`); with a float real_t
        /// the overload above is that.
        void begin_step(const real_t       t,
                        const real_t       dt,
                        const float *const relaxed_stress_n,
                        const float *const relaxed_stress_prev = nullptr);
#endif

        /// Commit the open step: `relaxed_stress_n` at the state it started
        /// from, `relaxed_stress` at the converged state it ends at.
        void advance(const real_t *const relaxed_stress_n, const real_t *const relaxed_stress);
#if SFEM_REAL_T_IS_FLOAT64
        void advance(const float *const relaxed_stress_n, const float *const relaxed_stress);
#endif

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

        /// The committed H_i, decoded into `values` (laid out [c][q][e]), given
        /// the relaxed stresses the next step starts from (`relaxed_stress_n`)
        /// and the last one started from (`relaxed_stress_prev`, read only for a
        /// term that is not stored), host arrays as `begin_step` takes them.  A
        /// host convenience.
        void decode_history(const int           i,
                            real_t *const       values,
                            const real_t *const relaxed_stress_n,
                            const real_t *const relaxed_stress_prev) const;

        /// Whether term `i` is stored as a remainder (the open or last step's choice).
        bool remainder(const int i) const { return slot_[i] >= 0 && slot_beta_[slot_[i]] != 0; }

        /// How many terms hold a stored tensor.
        int n_stored_terms() const { return n_live_; }

        /// Bytes of state held, frozen history included: the sum of the three below.
        std::size_t bytes() const;

        /// The stored terms H_i, scales included.
        std::size_t history_bytes() const;
        /// The remainder form's reference stress R.
        std::size_t reference_bytes() const;
        /// The frozen history the kernels read.
        std::size_t frozen_bytes() const;

    private:
        template <typename S_t>
        void begin_step_typed(const real_t dt, const S_t *const relaxed_stress_n, const S_t *const relaxed_stress_prev);
        template <typename S_t>
        void advance_typed(const S_t *const relaxed_stress_n, const S_t *const relaxed_stress);

        void zero_history();
        void reserve(const int n_slots);
        template <typename S_t>
        void store(const int i, const S_t *const relaxed_stress_n, const S_t *const relaxed_stress_prev);
        void unstore(const int i);

        /// `prony::advance` (or its device version) on slots [first, first + count).
        template <typename S_t>
        void update_slots(const int           first,
                          const int           count,
                          const real_t *const alpha,
                          const real_t *const b,
                          const real_t *const beta,
                          const S_t *const    S_n,
                          const S_t *const    S_new);

        /// R, in the stress slots' type, or null before it is taken.
        template <typename S_t>
        const S_t *reference() const;
        template <typename S_t>
        void take_reference(const S_t *const relaxed_stress_n);

        PronySeries                     series_;
        Options                         options_;
        int                             n_components_;
        int                             n_qp_;
        ptrdiff_t                       n_elements_;
        real_t                          temperature_;
        //! The stored terms, one tensor per slot; slots [0, n_live_) are in use.
        std::shared_ptr<QuadratureState> H_;
        int                              n_live_{0};
        std::vector<int>                 slot_;          ///< per term, -1 if not stored
        std::vector<int>                 term_of_slot_;  ///< per slot
        //! H_i = coefficient_[i] (S_r^n - S_r^{n-1}) for a term that is not stored.
        std::vector<real_t>              coefficient_;
        //! Per slot: the beta its store is in (G = H - beta (S_r^n - R)), 0 for H itself.
        std::vector<real_t>              slot_beta_;
        //! Per term, for the open step: the beta it is to be stored in.
        std::vector<real_t>              beta_;
        //! R, in real_t or float as the stress slots are.
        std::shared_ptr<Buffer<real_t>>  reference_full_;
        std::shared_ptr<Buffer<float>>   reference_narrow_;
        //! Per term: relaxes completely within the open step.
        std::vector<char>                relaxed_;
        //! The open step's weights per slot, the relaxed slots' alpha zeroed.
        std::vector<real_t>              slot_alpha_;
        std::vector<real_t>              slot_b_;
        std::shared_ptr<Buffer<real_t>>  history_;
        std::vector<real_t>              alpha_;
        std::vector<real_t>              b_;
        real_t                           gamma_{1};
        bool                             step_open_{false};
    };

}  // namespace sfem
