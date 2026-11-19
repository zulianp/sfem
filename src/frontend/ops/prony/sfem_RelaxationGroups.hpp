#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "sfem_PronyRelaxation.hpp"

#ifdef SFEM_ENABLE_RYAML
#include <ryml.hpp>
#endif  // SFEM_ENABLE_RYAML

namespace sfem {

    /// What a generated material declares about one relaxation group: the
    /// names its kernels use for the group's history input and stress output,
    /// and the number of stored components (6 in 3D, 3 in 2D).
    struct RelaxationGroupDeclaration {
        std::string group;
        std::string history;
        std::string stress;
        int         n_components{0};
    };

    /// The relaxation state of one operator: a `PronyRelaxation` per block and
    /// per relaxation group the material declares.
    ///
    /// Material independent.  A generated operator whose energy relaxes holds
    /// one of these, asks it for a group's step weight and frozen history, and
    /// hands it, once, its evaluation of the relaxed stresses at a displacement
    /// into a stress slot (`set_stress_evaluation`).  At the step boundaries the
    /// relaxed stresses are evaluated from the displacements the caller holds:
    ///
    ///     begin_step(t, dt, x_n):  S_r(x_n) -> slot 0; open the step
    ///     advance(x_n, x):         S_r(x_n) -> slot 0, S_r(x) -> slot 1; commit
    ///     reset(x):                H_i = 0; the next step starts from x
    ///
    /// The slots exist only while a call needs them, from the stream-ordered pool
    /// on the device, so between the calls the relaxed stresses take no memory.
    /// Before the first `advance` or `reset`, a step starts from the reference
    /// configuration, the zero displacement, whatever x_n is.
    ///
    /// The evaluation is a pointwise loop, so the same displacement gives the
    /// same tensor bit for bit at either end of a step, and the increments
    /// telescope as if each stress had been held.  A float32 slot
    /// (`PronyRelaxation::Options::stress_storage`) rounds the stress once, the
    /// same way at both ends.
    ///
    /// The one displacement the caller does not hold is the one the last
    /// committed step started from: a term that relaxed completely in that step
    /// is held as a coefficient on its increment, and is read, or brought back by
    /// a shorter step, from S_r(x_{n-1}).  That displacement is copied at
    /// `advance` while a term is held so, and released otherwise.  Configuration
    /// (spectra, time-temperature shift,
    /// integrator, storage of H_i) arrives through `set_series` or YAML; a
    /// group never configured has the empty series, gamma = 1 and a zero frozen
    /// history, which is the parent hyperelastic material.
    ///
    /// YAML, at the operator's top level (all blocks) or inside a block:
    ///
    ///     relaxation:
    ///       iso:                         # group name
    ///         prony: [{g: 0.3, tau: 0.1}, {g: 0.2, tau: 10}]
    ///         integrator: linear         # or midpoint
    ///         wlf: {C1: 8.86, C2: 101.6, T_ref: 0}
    ///         temperature: 10
    ///         history_storage: full      # full, float32, float16, float16_scaled,
    ///                                    # int16_scaled, int8_scaled, float16_md, int8_md, quant_grid,
    ///                                    # quant_block
    ///         history_elements_per_mean: 16  # *_md on one-point elements (QuadratureState)
    ///         history_tolerance: 1e-5    # quant_grid, quant_block: relative to each term's running maximum
    ///         history_rounding: nearest  # integer formats: nearest or stochastic
    ///         stress_storage: full       # the relaxed stresses and R: full or float32
    ///
    /// An entry overrides only the keys it names, so a block can change the
    /// temperature and keep the spectrum the top level gave it.  Any other key
    /// is an error.
    class RelaxationGroups {
    public:
        /// `es` is where the operator's vectors live: the history, the frozen
        /// history the kernels read and the stress slots are all allocated
        /// there.
        RelaxationGroups(std::vector<RelaxationGroupDeclaration> declarations,
                         const ptrdiff_t                         n_dofs,
                         const ExecutionSpace                    es = EXECUTION_SPACE_HOST);

        /// The operator's evaluation of every group's relaxed stress at the
        /// displacement `x` into stress slot `slot` (`stress_scratch`).
        using StressEvaluation = std::function<int(const real_t *const x, const int slot)>;

        /// The operator's evaluation, set once by the operator that holds this state.
        void set_stress_evaluation(StressEvaluation evaluate) { evaluate_ = std::move(evaluate); }

        const std::vector<RelaxationGroupDeclaration> &declarations() const { return declarations_; }

        /// Register a block before any configuration reaches it.
        void add_block(const std::string &block, const ptrdiff_t n_elements, const int n_qp);

        /// Give `group` on `block` a spectrum ("" for every block), replacing
        /// its history with one at rest.
        void set_series(const std::string              &block,
                        const std::string              &group,
                        const PronySeries              &series,
                        const PronyRelaxation::Options &options = PronyRelaxation::Options{});

        void set_temperature(const std::string &block, const std::string &group, const real_t temperature);

        /// The remainder form's horizon (`PronyRelaxation::Options::remainder_horizon`) of every
        /// group that has none set: the span of time the caller is about to simulate.
        void set_default_remainder_horizon(const real_t horizon);

        /// Term `i` of `group`'s committed history on `block`, decoded on the host into
        /// `values` (laid out [c][q][e]); `x_n` is the displacement the next step starts from,
        /// the relaxed stresses are evaluated from it.
        void decode_history(const std::string &block,
                            const std::string &group,
                            const int          i,
                            const real_t *const x_n,
                            real_t *const       values);

        /// Every stored term of every group on every block, decoded on the host as decode_history
        /// does, from one evaluation of the relaxed stresses at `x_n`: `emit(block, group, i,
        /// values)` per term, `values` valid for the call. A stored term reads only the stress at
        /// `x_n`, so nothing else is evaluated.
        void decode_stored_histories(
                const real_t *const                                                                          x_n,
                const std::function<void(const std::string &, const std::string &, int, const real_t *)> &emit);

        /// Per term of `group` on `block`, PronyRelaxation::squared_norms of the committed history,
        /// where it lives: n_terms values into the host array `out`.  Computed by the last
        /// `advance` if `track_squared_norms` asked for it, else from the relaxed stresses at `x_n`,
        /// the displacement the next step starts from.
        void squared_norms(const std::string &block,
                           const std::string &group,
                           const int          n_diagonal,
                           const real_t *const x_n,
                           real_t *const       out);

        /// Have every `advance` compute the squared norms while it holds the relaxed stresses, so
        /// that a probe reading them every step costs no evaluation of its own.
        void track_squared_norms() { track_norms_ = true; }

        /// The blocks registered, in name order.
        std::vector<std::string> block_names() const;
        PronyRelaxation       &relaxation(const std::string &block, const std::string &group);
        const PronyRelaxation &relaxation(const std::string &block, const std::string &group) const;

        /// The kernels' inputs, looked up by the names they use.
        real_t        gamma(const std::string &block, const std::string &group) const;
        const real_t *history(const std::string &block, const std::string &history_name) const;

        /// The stream the device loops run on, the operator's own: the stresses
        /// they read are written by kernels on it.
        void set_stream(void *const stream);

        /// Where the operator writes stress slot `slot` of a group, laid out like
        /// the history: slot 0 is the relaxed stress a step starts from, slot 1 the
        /// one it ends at (or, in `begin_step`, the one the last step started
        /// from).  Valid only while the stress evaluation runs.  A group whose
        /// slots are float32 (`PronyRelaxation::Options::stress_storage`) is
        /// written to a real_t scratch and rounded into the slot after it.
        real_t *stress_scratch(const std::string &block, const std::string &stress_name, const int slot);

        /// Open a step of `dt` from the converged displacement `x_n`.
        void begin_step(const real_t t, const real_t dt, const real_t *const x_n);

        /// Commit every group of every block from `x_n` to the converged displacement `x`.
        void advance(const real_t *const x_n, const real_t *const x);

        /// Restart every group at rest at the displacement `x`, which the next step starts from.
        void reset(const real_t *const x);

        /// The committed state, every block's groups (PronyRelaxation::write_state) and the
        /// displacement the last step started from if a term needs it, into the folder `dir` and
        /// back (a restart).  The relaxed stresses are evaluated again from the displacements.
        int write_state(const std::string &dir);
        int read_state(const std::string &dir);

        /// Whether any group has a non-empty series; an operator with none
        /// need not evaluate stresses at all.
        bool relaxes() const;

        /// Persistent bytes: the history, and the previous displacement while it is held.
        std::size_t bytes() const;

        /// The persistent bytes by what holds them, summed over blocks and groups.
        struct Bytes {
            std::size_t history{0};    ///< the stored terms H_i, scales included
            std::size_t reference{0};  ///< the remainder form's reference stress R
            std::size_t frozen{0};     ///< the frozen history the kernels read
            std::size_t previous{0};   ///< the displacement the last step started from, while held
            //! What the history's integer codes would take packed (not held; 0 if not integer).
            std::size_t history_packed{0};
            std::size_t total() const { return history + reference + frozen + previous; }
        };
        Bytes bytes_breakdown() const;

#ifdef SFEM_ENABLE_RYAML
        /// Read the `relaxation` entry of `node` into `block` ("" for every block).
        void configure_from_yaml(const std::string &block, const ryml::ConstNodeRef &node);
#endif  // SFEM_ENABLE_RYAML

    private:
        struct Block {
            ptrdiff_t                                     n_elements{0};
            int                                           n_qp{0};
            std::vector<std::shared_ptr<PronyRelaxation>> groups;
            //! A stress slot: real_t, or float32 when the group's stress_storage is.
            struct Slot {
                std::shared_ptr<Buffer<real_t>> full;
                std::shared_ptr<Buffer<float>>  narrow;
            };
            //! [slot][group]: slot 0 the step's start, slot 1 its end; while a call needs them.
            std::vector<std::vector<Slot>> stress;
            //! [group]: where a float32 group's stress is evaluated, during `evaluate` only.
            std::vector<std::shared_ptr<Buffer<real_t>>> scratch;
            //! [group][term]: the committed history's squared norms, from the last `advance`.
            std::vector<std::vector<real_t>> norms;
        };

        int           group_index(const std::string &group) const;
        int           history_index(const std::string &history) const;
        int           stress_index(const std::string &stress) const;
        Block        &block(const std::string &name);
        const Block  &block(const std::string &name) const;
        bool          narrow(const Block &b, const size_t group) const;
        //! Every group's relaxed stress at `x` into slot `slot`, held until `release_stress`.
        void          evaluate(const real_t *const x, const int slot);
        void          release_stress();
        //! The displacement a step from `x_n` starts from: the reference configuration (in
        //! `zeros`) before any step or reset.
        const real_t *start(const real_t *const x_n, std::shared_ptr<Buffer<real_t>> &zeros) const;
        //! Hold `x_start`, the displacement the step just committed started from, while a term
        //! is held on that step's increment; release it otherwise.
        void          keep_previous(const real_t *const x_start);
        const real_t *previous() const;
        void          group_norms(const Block &b, const size_t group, const int slot_n, const int slot_prev, real_t *const out) const;

        std::vector<RelaxationGroupDeclaration> declarations_;
        std::map<std::string, Block>            blocks_;
        ptrdiff_t                               n_dofs_;
        ExecutionSpace                          es_;
        void                                   *stream_{nullptr};
        StressEvaluation                        evaluate_;
        //! No step or reset yet: a step starts from the reference configuration.
        bool start_reference_{true};
        //! The displacement the last committed step started from, while a term needs it.
        std::shared_ptr<Buffer<real_t>> previous_;
        bool                            track_norms_{false};
        //! Every block's `norms` are the committed history's.
        bool norms_current_{false};
    };

}  // namespace sfem
