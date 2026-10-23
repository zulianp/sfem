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
    /// per relaxation group the material declares, and the relaxed stress the
    /// next step starts from.
    ///
    /// Material independent.  A generated operator whose energy relaxes holds
    /// one of these, asks it for a group's step weight and frozen history, and
    /// at the step boundaries evaluates the relaxed stresses into its two
    /// stress slots:
    ///
    ///     begin_step(t, dt, S_r):  reads slot 0, S_r^n
    ///     advance(x, S_r):         S_r(x) -> slot 1; commit; slot 1 becomes
    ///                              slot 0, the next step's S_r^n
    ///     reset(x, S_r):           H_i = 0; S_r(x) -> slot 0
    ///
    /// where `S_r` is the operator's evaluation of the relaxed stresses into a
    /// slot.  Before the first `advance` or `reset`, slot 0 is evaluated at the
    /// reference configuration, the zero displacement.
    ///
    /// Each relaxed stress is evaluated once, at the end of the step it closes,
    /// and the next step starts from that very tensor, so the increments
    /// telescope bit for bit.  The slots are allocated once, at the first
    /// evaluation, and held: `advance` needs both at the same time anyway, so
    /// holding them between steps does not raise the peak, and no step
    /// allocates.  Configuration (spectra, time-temperature shift,
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
    ///                                    # int16_scaled, int8_scaled
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
        /// displacement `x` into stress slot `slot`.
        using StressEvaluation = std::function<int(const real_t *const x, const int slot)>;

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

        /// Stress slot `slot` of a group, laid out like the history: slot 0 is
        /// the relaxed stress the open or next step starts from, slot 1 the one
        /// `advance` commits.  Allocated, for every block and group, at first use.
        real_t *stress_scratch(const std::string &block, const std::string &stress_name, const int slot);

        /// Open a step at the relaxed stresses in slot 0, evaluating them at the
        /// reference configuration first if no step or reset has set them.
        void begin_step(const real_t t, const real_t dt, const StressEvaluation &evaluate);

        /// Commit every group of every block at the converged displacement
        /// `x`: its relaxed stress goes to slot 1, the step is committed from
        /// slot 0 to slot 1, and the slots swap, so slot 0 is the next step's
        /// start.
        void advance(const real_t *const x, const StressEvaluation &evaluate);

        /// Restart every group at rest at the displacement `x`.
        void reset(const real_t *const x, const StressEvaluation &evaluate);

        /// Whether any group has a non-empty series; an operator with none
        /// need not evaluate stresses at all.
        bool relaxes() const;

        /// Persistent bytes: the history and the stress slots.
        std::size_t bytes() const;

#ifdef SFEM_ENABLE_RYAML
        /// Read the `relaxation` entry of `node` into `block` ("" for every block).
        void configure_from_yaml(const std::string &block, const ryml::ConstNodeRef &node);
#endif  // SFEM_ENABLE_RYAML

    private:
        struct Block {
            ptrdiff_t                                     n_elements{0};
            int                                           n_qp{0};
            std::vector<std::shared_ptr<PronyRelaxation>> groups;
            //! [slot][group]: slot 0 the step's start, slot 1 its end.
            std::vector<std::vector<std::shared_ptr<Buffer<real_t>>>> stress;
        };

        int          group_index(const std::string &group) const;
        int          history_index(const std::string &history) const;
        int          stress_index(const std::string &stress) const;
        Block       &block(const std::string &name);
        const Block &block(const std::string &name) const;
        const real_t *slot(const Block &b, const int slot, const size_t group) const;
        void          allocate_stress();
        void          evaluate(const real_t *const x, const int slot, const StressEvaluation &evaluate);

        std::vector<RelaxationGroupDeclaration> declarations_;
        std::map<std::string, Block>            blocks_;
        ptrdiff_t                               n_dofs_;
        ExecutionSpace                          es_;
        void                                   *stream_{nullptr};
        //! Slot 0 holds the stress of the state the next step starts from.
        bool start_stress_current_{false};
        //! That state was committed without its stress: not the reference
        //! configuration, and no longer known.
        bool start_unknown_{false};
    };

}  // namespace sfem
