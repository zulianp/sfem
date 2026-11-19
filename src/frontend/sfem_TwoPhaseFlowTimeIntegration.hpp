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
    class DirichletConditions;

    class TwoPhaseFlowTimeIntegration {
    public:
        using StepSolver = std::function<int(const real_t *previous,
                                             real_t *trial,
                                             real_t time,
                                             real_t dt)>;
        using BoundaryUpdate =
                std::function<void(real_t time, DirichletConditions &conditions)>;

        TwoPhaseFlowTimeIntegration(const std::shared_ptr<smesh::Mesh> &mesh,
                                    const std::shared_ptr<Buffer<real_t>> &initial_state,
                                    const std::shared_ptr<DirichletConditions> &dirichlet,
                                    BoundaryUpdate boundary_update = {});
        ~TwoPhaseFlowTimeIntegration();

        int initialize();
        int advance(real_t dt, const StepSolver &solver);
        int save_restart(const smesh::Path &folder) const;
        int load_restart(const smesh::Path &folder);

        void apply_boundary(real_t time, real_t *state) const;
        void constrain_residual(const real_t *state, real_t *residual) const;
        void constrain_direction(real_t *direction) const;
        void constrain_linear(const real_t *direction, real_t *output) const;

        const std::shared_ptr<Buffer<real_t>> &accepted() const;
        const std::shared_ptr<Buffer<real_t>> &trial() const;
        real_t time() const;
        ptrdiff_t step() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
