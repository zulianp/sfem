#pragma once

#include "sfem_aliases.hpp"
#include "smesh_buffer.hpp"

#include <functional>
#include <memory>

namespace smesh {
    class Mesh;
    class Path;
}

namespace sfem {
    struct TwoPhaseFlowBalance {
        real_t left[2]{0, 0};
        real_t right[2]{0, 0};
        real_t interior[2]{0, 0};
        real_t total[2]{0, 0};
    };

    struct TwoPhaseFlowTimeConfig {
        real_t initial_water_pressure{15e6};
        real_t initial_co2_pressure{15.1e6};
        real_t injection_co2_pressure{20e6};
        real_t ramp_duration{1};
    };

    class TwoPhaseFlowTimeIntegration {
    public:
        using StepSolver = std::function<int(const real_t *previous,
                                             real_t *trial,
                                             real_t time,
                                             real_t dt)>;

        TwoPhaseFlowTimeIntegration(const std::shared_ptr<smesh::Mesh> &mesh,
                                    const TwoPhaseFlowTimeConfig &config);
        ~TwoPhaseFlowTimeIntegration();

        int initialize();
        int advance(real_t dt, const StepSolver &solver);
        int save_restart(const smesh::Path &folder) const;
        int load_restart(const smesh::Path &folder);

        void apply_boundary(real_t time, real_t *state) const;
        void constrain_residual(real_t *residual) const;
        void constrain_direction(real_t *direction) const;
        void constrain_linear(const real_t *direction, real_t *output) const;
        TwoPhaseFlowBalance balance(const real_t *unconstrained_residual) const;

        const std::shared_ptr<Buffer<real_t>> &accepted() const;
        const std::shared_ptr<Buffer<real_t>> &trial() const;
        real_t time() const;
        ptrdiff_t step() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
