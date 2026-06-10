#ifndef SFEM_ROTATE_HPP
#define SFEM_ROTATE_HPP

#include "sfem_DirichletConditions.hpp"
#include "sfem_ForwardDeclarations.hpp"

#include "smesh_env.hpp"
#include "smesh_sideset.hpp"

#include <cmath>

#ifdef SFEM_ENABLE_RYAML
#include <ryml.hpp>
#endif

namespace sfem {

    template <int Axis0, int Axis1>
    struct Rotate : public Constraint {
        std::shared_ptr<FunctionSpace> space;
        int                            steps;
        real_t                         angle;
        std::shared_ptr<Sideset>       sideset;
        SharedBuffer<idx_t>            nodeset;
        SharedBuffer<real_t>           u0;
        SharedBuffer<real_t>           u1;
        ExecutionSpace                 execution_space;
        real_t                         rcenter[3] = {0, 0, 0};
        bool                           verbose{false};
        std::shared_ptr<Constraint>    constraint;

        Rotate(const std::shared_ptr<FunctionSpace> &space,
               const int                             steps,
               const real_t                          angle,
               const std::shared_ptr<Sideset>       &sideset,
               const SharedBuffer<idx_t>            &nodeset,
               const SharedBuffer<real_t>           &u0,
               const SharedBuffer<real_t>           &u1,
               const ExecutionSpace                  execution_space)
            : space(space),
              steps(steps),
              angle(angle),
              sideset(sideset),
              nodeset(nodeset),
              u0(u0),
              u1(u1),
              execution_space(execution_space) {}

        std::shared_ptr<Constraint> create_constraint() {
            if (constraint) {
                return constraint;
            }
            DirichletConditions::Condition rot0{
                    .sidesets = {sideset}, .nodeset = nodeset, .values = u0, .value = 0, .component = Axis0};
            DirichletConditions::Condition rot1{
                    .sidesets = {sideset}, .nodeset = nodeset, .values = u1, .value = 0, .component = Axis1};
            auto dc = DirichletConditions::create(space, {rot0, rot1});
#ifdef SFEM_ENABLE_CUDA
            if (execution_space == EXECUTION_SPACE_DEVICE) {
                constraint = to_device(dc);
            } else
#endif
            {
                constraint = dc;
            }
            return constraint;
        }

        int apply(real_t *const x) override { return create_constraint()->apply(x); }

        int apply_value(const real_t value, real_t *const x) override { return create_constraint()->apply_value(value, x); }

        int gradient(const real_t *const x, real_t *const g) override { return create_constraint()->gradient(x, g); }

        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override {
            return create_constraint()->copy_constrained_dofs(src, dest);
        }

        int mask(mask_t *m) override { return create_constraint()->mask(m); }

        int value(const real_t *const x, real_t *const out) override { return create_constraint()->value(x, out); }

        int value_steps(const real_t       *x,
                        const real_t       *h,
                        const int           nsteps,
                        const real_t *const steps,
                        real_t *const       out) override {
            return create_constraint()->value_steps(x, h, nsteps, steps, out);
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            return create_constraint()->hessian_crs(x, rowptr, colidx, values);
        }

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            return create_constraint()->hessian_bsr(x, rowptr, colidx, values);
        }

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                             const bool                            as_zero) const override {
            assert(constraint);
            return constraint->derefine(coarse_space, as_zero);
        }

        std::shared_ptr<Constraint> lor() const override {
            assert(constraint);
            return constraint->lor();
        }

        void update(int step) {
            auto         points        = space->points()->data();
            const real_t current_angle = step * angle / steps;

            if (verbose) {
                printf("%d) current_angle = %g\n", step, current_angle);
            }

            const real_t c = std::cos(current_angle);
            const real_t s = std::sin(current_angle);

            auto       nodes   = nodeset->data();
            auto       out0    = u0->data();
            auto       out1    = u1->data();
            const auto p0      = points[Axis0];
            const auto p1      = points[Axis1];
            const auto center0 = rcenter[Axis0];
            const auto center1 = rcenter[Axis1];
            const auto n_nodes = nodeset->size();

            if (execution_space == EXECUTION_SPACE_DEVICE) {
                SFEM_ERROR("IMPLEMENT ME!\n");
                return;
            }

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                const ptrdiff_t dof = nodes[i];

                const geom_t pos0 = p0[dof] - center0;
                const geom_t pos1 = p1[dof] - center1;

                out0[i] = c * pos0 - s * pos1 - pos0;
                out1[i] = s * pos0 + c * pos1 - pos1;
            }
        }

        static std::shared_ptr<Rotate<Axis0, Axis1>> create(const std::shared_ptr<FunctionSpace> &space,
                                                            const std::shared_ptr<Sideset>       &sideset,
                                                            const int                             steps,
                                                            const real_t                          angle,
                                                            const ExecutionSpace                  execution_space) {
            auto mesh_for_sideset = space->mesh_ptr();
            auto nodeset          = smesh::create_nodeset_from_sideset(mesh_for_sideset, sideset);

            auto u0  = create_buffer<real_t>(nodeset->size(), EXECUTION_SPACE_HOST);
            auto u1  = create_buffer<real_t>(nodeset->size(), EXECUTION_SPACE_HOST);
            auto ret = std::make_shared<Rotate<Axis0, Axis1>>(space, steps, angle, sideset, nodeset, u0, u1, execution_space);
            ret->create_constraint();
            return ret;
        }

        static std::shared_ptr<Rotate<Axis0, Axis1>> create_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                     const ExecutionSpace                  execution_space) {
            const real_t      angle        = smesh::Env::read("SFEM_ROTATE_ANGLE", 0.0);
            const std::string sideset_path = smesh::Env::read_string("SFEM_ROTATE_SIDESET", "");
            const int         steps        = smesh::Env::read("SFEM_ROTATE_STEPS", 10);
            const bool        verbose      = smesh::Env::read("SFEM_ROTATE_VERBOSE", false);

            if (!sideset_path.empty()) {
                if (verbose) {
                    printf("Rotating sideset %s with angle %g\n", sideset_path.c_str(), angle);
                }
                auto sideset    = Sideset::create_from_file(space->mesh_ptr()->comm(), smesh::Path(sideset_path));
                auto ret        = Rotate<Axis0, Axis1>::create(space, sideset, steps, angle, execution_space);
                ret->verbose    = verbose;
                ret->rcenter[0] = smesh::Env::read("SFEM_ROTATE_RCENTER_X", 0.0);
                ret->rcenter[1] = smesh::Env::read("SFEM_ROTATE_RCENTER_Y", 0.0);
                ret->rcenter[2] = smesh::Env::read("SFEM_ROTATE_RCENTER_Z", 0.0);
                ret->create_constraint();
                return ret;
            }
            return nullptr;
        }

#ifdef SFEM_ENABLE_RYAML
        static std::shared_ptr<Rotate<Axis0, Axis1>> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                      const ryml::NodeRef                  &node,
                                                                      const ExecutionSpace                  execution_space) {
            real_t      angle{0};
            std::string sideset_path;
            int         steps{10};
            bool        verbose{false};

            node["angle"] >> angle;
            node["sideset"] >> sideset_path;

            if (node["steps"].readable()) {
                node["steps"] >> steps;
            }

            if (node["verbose"].readable()) {
                node["verbose"] >> verbose;
            }

            if (!sideset_path.empty()) {
                if (verbose) {
                    printf("Rotating sideset %s with angle %g\n", sideset_path.c_str(), angle);
                }
                auto sideset = Sideset::create_from_file(space->mesh_ptr()->comm(), smesh::Path(sideset_path));
                auto ret     = Rotate<Axis0, Axis1>::create(space, sideset, steps, angle, execution_space);

                auto rcenter = node["rotation_center"];
                if (rcenter.is_seq()) {
                    rcenter[0] >> ret->rcenter[0];
                    rcenter[1] >> ret->rcenter[1];
                    rcenter[2] >> ret->rcenter[2];
                } else {
                    SFEM_ERROR("Rotation center must be a sequence [x, y, z]\n");
                }

                ret->verbose = verbose;
                ret->create_constraint();
                return ret;
            }

            return nullptr;
        }
#endif
    };

    using RotateXY = Rotate<0, 1>;
    using RotateXZ = Rotate<0, 2>;
    using RotateYZ = Rotate<1, 2>;

}  // namespace sfem

#endif
