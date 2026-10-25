#ifndef SFEM_CONTACT_CONDITIONS_HPP
#define SFEM_CONTACT_CONDITIONS_HPP

#include "sfem_Function.hpp"
#include "smesh_grid.hpp"

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

        const SharedBuffer<idx_t> node_mapping();
        /// The obstacle's normal at each contact node (component-major, dim rows), pointing towards
        /// the obstacle: the constraint is n . u <= gap + n . u_k. Sampled by init / update.
        SharedBuffer<real_t *> normals() const;
        /// The normals' rows (dim of them; the rest null), readable on the host whatever the
        /// execution space: a device two-level buffer keeps its row pointers on the device.
        void normal_rows(real_t **const rows) const;
        /// Each contact node's share of the surface measure (lumped), in the execution space: the
        /// contact force at node i is m_i times its multiplier.
        SharedBuffer<real_t> surface_mass() const;

        /// How the gap and its normals are taken from the obstacle at the contact nodes; both are
        /// linearised by the nodal normal projection n . u. NODAL (the default) samples the signed
        /// distance at each node, the normal being its gradient there. DUAL_WEIGHTED integrates it over
        /// each node's surface patch against the dual basis, biorthogonal to the nodal one (so n . u is
        /// still the linearisation), and mixes the faces that meet at a sharp edge. Set before init().
        enum class Gap { NODAL, DUAL_WEIGHTED };
        void set_gap(const Gap gap);
        Gap  gap() const;
        /// The curvature of the nodal gap's constraint at the contact nodes as last updated, minus the
        /// Hessian of the gap -sdf / |grad sdf|, into 6 packed rows [xx, xy, xz, yy, yz, zz] of
        /// n_constrained_dofs values in the execution space. Its multiple mu_i C_i is the contact
        /// force's share of a Newton tangent. The NODAL gap only.
        int curvature(real_t **const C);

        static std::shared_ptr<ContactConditions> create_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                  const enum ExecutionSpace             es);

        static std::shared_ptr<ContactConditions> create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                   const std::string                    &path,
                                                                   const enum ExecutionSpace             es);

        static std::shared_ptr<ContactConditions> create(const std::shared_ptr<FunctionSpace> &space,
                                                         const std::shared_ptr<smesh::Grid<geom_t>>  &sdf,
                                                         const std::vector<std::shared_ptr<Sideset>> &sidesets,
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

        SharedBuffer<idx_t *> ss_sides();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// AxisAlignedContactConditions::create_from_env, moved to the device for EXECUTION_SPACE_DEVICE.
    std::shared_ptr<Constraint> create_contact_conditions_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                   const ExecutionSpace                  es);
}  // namespace sfem

#endif  // SFEM_CONTACT_CONDITIONS_HPP
