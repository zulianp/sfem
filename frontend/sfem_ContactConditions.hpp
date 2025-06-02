#ifndef SFEM_CONTACT_CONDITIONS_HPP
#define SFEM_CONTACT_CONDITIONS_HPP

#include "sfem_Function.hpp"
#include "sfem_Grid.hpp"

namespace sfem {

    // This is for now just a copy and past of dirichlet conditions
    // it will change in the near future
    class AxisAlignedContactConditions final : public Constraint {
    public:
        AxisAlignedContactConditions(const std::shared_ptr<FunctionSpace> &space);
        ~AxisAlignedContactConditions();

        std::shared_ptr<FunctionSpace> space();

        static std::shared_ptr<AxisAlignedContactConditions> create_from_env(const std::shared_ptr<FunctionSpace> &space);
        int                                                  apply(real_t *const x) override;
        int                                                  apply_value(const real_t value, real_t *const x) override;
        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override;
        int mask(mask_t *mask) override;

        int gradient(const real_t *const x, real_t *const g) override;

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const    idx,
                           const int       component,
                           real_t *const   values);

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const    idx,
                           const int       component,
                           const real_t    value);

        int   n_conditions() const;
        void *impl_conditions();

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                             const bool                            as_zero) const override;
        std::shared_ptr<Constraint> lor() const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class ContactConditions {
    public:
        ContactConditions(const std::shared_ptr<FunctionSpace> &space);
        ~ContactConditions();

        std::shared_ptr<FunctionSpace> space();

        const std::shared_ptr<Buffer<idx_t>> node_mapping();

        static std::shared_ptr<ContactConditions> create_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                  const enum ExecutionSpace             es);

        static std::shared_ptr<ContactConditions> create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                   const std::string                    &path,
                                                                   const enum ExecutionSpace             es);

        static std::shared_ptr<ContactConditions> create(const std::shared_ptr<FunctionSpace> &space,
                                                         const std::shared_ptr<Grid<geom_t>>  &sdf,
                                                         const std::shared_ptr<Sideset>       &sideset,
                                                         const enum ExecutionSpace             es);

        // int copy_constrained_dofs(const real_t *const src, real_t *const dest);
        int mask(mask_t *mask);
        // int gradient(const real_t *const x, real_t *const g);
        int signed_distance(const real_t *const x, real_t *const g);

        // Call update(x) before calling this
        int update_signed_distance(const real_t *const u, real_t *const g);
        int signed_distance(real_t *const g);

        int hessian_crs(const real_t *const x, const count_t *const rowptr, const idx_t *const colidx, real_t *const values);

        int hessian_block_diag_sym(const real_t *const x, real_t *const values);

        int init();

        int                               update(const real_t *const x);
        std::shared_ptr<Operator<real_t>> linear_constraints_op();
        std::shared_ptr<Operator<real_t>> linear_constraints_op_transpose();

        int normal_project(const real_t *const h, real_t *const out);
        int distribute_contact_forces(const real_t *const f, real_t *const out);

        ptrdiff_t n_constrained_dofs() const;

        int signed_distance_for_mesh_viz(const real_t *const x, real_t *const g) const;

        int full_apply_boundary_mass_inverse(const real_t *const r, real_t *const s);

        std::shared_ptr<Buffer<idx_t *>> ss_sides();
        // std::shared_ptr<Sideset>         sideset();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_CONTACT_CONDITIONS_HPP
