#pragma once

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
    /// per relaxation group the material declares, and the converged
    /// displacement the next step starts from.
    ///
    /// Material independent.  A generated operator whose energy relaxes holds
    /// one of these, asks it for a group's step weight and frozen history, and
    /// at the step boundaries evaluates the relaxed stresses it needs into the
    /// workspaces it lends:
    ///
    ///     begin_step:  acquire_workspace(1); S_r(previous_state()) -> slot 0;
    ///                  begin_step(t, dt); release_workspace()
    ///     advance(x):  acquire_workspace(2); S_r(previous_state()) -> slot 0,
    ///                  S_r(x) -> slot 1; advance(); release_workspace();
    ///                  set_previous_state(x)
    ///
    /// The relaxed stress S_r^n is never stored.  It is evaluated from the
    /// previous displacement each time, so the persistent state is H_i, the
    /// frozen history the kernels read, and one nodal vector.  The workspaces
    /// exist only for the length of one of these calls: three tensors' worth
    /// of memory per element is not held between steps, at the price of an
    /// allocation per call.  Configuration (spectra, time-temperature shift,
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
        /// history the kernels read, the previous displacement and the
        /// workspaces are all allocated there.
        RelaxationGroups(std::vector<RelaxationGroupDeclaration> declarations,
                         const ptrdiff_t                         n_dofs,
                         const ExecutionSpace                    es = EXECUTION_SPACE_HOST);

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

        PronyRelaxation       &relaxation(const std::string &block, const std::string &group);
        const PronyRelaxation &relaxation(const std::string &block, const std::string &group) const;

        /// The kernels' inputs, looked up by the names they use.
        real_t        gamma(const std::string &block, const std::string &group) const;
        const real_t *history(const std::string &block, const std::string &history_name) const;

        /// The stream the device loops run on, the operator's own: the stresses
        /// they read are written by kernels on it.
        void set_stream(void *const stream);

        /// The converged displacement the open or next step starts from.
        /// Zero -- the reference configuration -- until the first `advance`.
        const real_t *previous_state() const { return previous_state_->data(); }
        void          set_previous_state(const real_t *const x);

        /// Lend `n_slots` stress workspaces per block and group, laid out like
        /// the history.  Held until `release_workspace`.
        void    acquire_workspace(const int n_slots);
        void    release_workspace();
        real_t *stress_scratch(const std::string &block, const std::string &stress_name, const int slot);

        /// Open a step at the relaxed stresses in slot 0.
        void begin_step(const real_t t, const real_t dt);

        /// Commit every group of every block: slot 0 holds the stress the step
        /// started from, slot 1 the one it ends at.
        void advance();

        /// Restart every group at rest at the displacement `x`.
        void reset(const real_t *const x);

        /// Whether any group has a non-empty series; an operator with none
        /// need not evaluate stresses at all.
        bool relaxes() const;

        /// Persistent bytes: the history and the previous displacement.
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
            //! [slot][group], allocated only while a workspace is held.
            std::vector<std::vector<std::shared_ptr<Buffer<real_t>>>> workspace;
        };

        int          group_index(const std::string &group) const;
        int          history_index(const std::string &history) const;
        int          stress_index(const std::string &stress) const;
        Block       &block(const std::string &name);
        const Block &block(const std::string &name) const;
        const real_t *slot(const Block &b, const int slot, const size_t group) const;

        std::vector<RelaxationGroupDeclaration> declarations_;
        std::map<std::string, Block>            blocks_;
        ExecutionSpace                          es_;
        std::shared_ptr<Buffer<real_t>>         previous_state_;
        void                                   *stream_{nullptr};
        int                                     n_slots_{0};
    };

}  // namespace sfem
