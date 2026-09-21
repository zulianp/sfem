#ifndef SFEM_FUNCTION_HPP
#define SFEM_FUNCTION_HPP

#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "sfem_mask.hpp"

#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"

#include "sfem_DirichletConditions.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_NeumannConditions.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_output.hpp"

// Operator includes
#include "sfem_Op.hpp"
#include "sfem_OpFactory.hpp"

#include "sfem_Constraint.hpp"

namespace sfem {

    // using Output = smesh::Output;
    class Output {
    public:
        Output(const std::shared_ptr<FunctionSpace> &space);
        ~Output();
        void set_output_dir(const smesh::Path &path);
        int  write(const char *name, const real_t *const x);
        int  write_time_step(const char *name, const real_t t, const real_t *const x);
        void enable_AoS_to_SoA(const bool val);
        void clear();

        void log_time(const real_t t);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Function final {
    public:
        Function(const std::shared_ptr<FunctionSpace> &space);
        ~Function();

        std::shared_ptr<Function> derefine(const bool dirichlet_as_zero);
        std::shared_ptr<Function> derefine(const std::shared_ptr<FunctionSpace> &space, const bool dirichlet_as_zero);

        inline static std::shared_ptr<Function> create(const std::shared_ptr<FunctionSpace> &space) {
            return std::make_shared<Function>(space);
        }

        std::shared_ptr<FunctionSpace> space();

        std::shared_ptr<Function> lor();
        std::shared_ptr<Function> lor(const std::shared_ptr<FunctionSpace> &space);

        void remove_operator(const std::shared_ptr<Op> &op);
        void add_operator(const std::shared_ptr<Op> &op);
        void add_constraint(const std::shared_ptr<Constraint> &c);
        void clear_constraints();
        void add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c);

        std::shared_ptr<CRSGraph> crs_graph() const;

        int hessian_crs(const real_t *const x, const count_t *const rowptr, const idx_t *const colidx, real_t *const values);

        int hessian_bsr(const real_t *const x, const count_t *const rowptr, const idx_t *const colidx, real_t *const values);

        int hessian_dia(const real_t *const x, const int *const diag_offsets, const ptrdiff_t ndiag, real_t *const values);

        int hessian_bcrs_sym(const real_t *const  x,
                             const count_t *const rowptr,
                             const idx_t *const   colidx,
                             const ptrdiff_t      block_stride,
                             real_t **const       diag_values,
                             real_t **const       off_diag_values);

        int hessian_crs_sym(const real_t *const  x,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        diag_values,
                            real_t *const        off_diag_values);

        int hessian_diag(const real_t *const x, real_t *const values);

        int hessian_block_diag_sym(const real_t *const x, real_t *const values);

        int update(const real_t *const x);

        /**
         * @brief Whether any operator here can have its tangent stored and reapplied
         *
         * Any, not all. A function is a sum of operators and the split is worth
         * taking wherever it is offered: an operator that does not support it
         * keeps its exact apply, which is the same operator either way.
         */
        bool inexact_supported() const;

        /**
         * @brief Assemble the stored tangent of every operator that has one
         *
         * Explicit, like the per-operator call it forwards to: the tangent stays
         * valid until the next call, so a Newton step assembles once here and
         * then applies for every Krylov iteration. Nothing invalidates it
         * implicitly.
         */
        int inexact_update(const real_t *const x);

        /**
         * @brief Apply the stored tangent, exactly where an operator has none
         *
         * Takes no state. An operator without a stored tangent is applied
         * exactly, at the state its own `update` last saw -- which is why this
         * is only correct when `inexact_update` and `update` are driven from
         * the same `x`, as a Newton step does.
         */
        int inexact_apply(const real_t *const h, real_t *const out);
        int gradient(const real_t *const x, real_t *const out, const ElementScope scope = ElementScope::ALL);
        int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const       out,
                  const ElementScope  scope = ElementScope::ALL);
        /**
         * @brief The sum of the operators' potentials at each trial step.
         *
         * Exists only when every operator is `energy_or_potential_based()`;
         * otherwise it refuses, naming the operator that has no potential,
         * because a squared residual norm and an energy are not terms of one
         * sum.  Appending a potential -- an inertia beside a static energy --
         * is how a material without transient terms becomes transient, and
         * this is the merit that stays available when you do.
         */
        int energy_merit(const real_t       *x,
                         const real_t       *h,
                         const int           nsteps,
                         const real_t *const steps,
                         real_t *const       out);

        /**
         * @brief `1/2 * ||R||^2` over the residual this Function assembles, at
         *        each trial step.
         *
         * Always available: every system has a residual.  The operators whose
         * `gradient` ignores the state are assembled **once** into an
         * accumulator, and the one operator that moves with the state is handed
         * that accumulator and asked to finish the sum at every step -- so a
         * twelve-point sampling line search costs close to one classical step
         * rather than twelve assemblies of everything.
         *
         * The overload taking `accumulator` lets a caller own the buffer; the
         * other allocates one lazily and keeps it.  It must hold `n_dofs()` and
         * live in this Function's execution space.
         */
        int residual_merit(const real_t       *x,
                           const real_t       *h,
                           const int           nsteps,
                           const real_t *const steps,
                           real_t *const       out);

        int residual_merit(const real_t       *x,
                           const real_t       *h,
                           const int           nsteps,
                           const real_t *const steps,
                           real_t *const       accumulator,
                           real_t *const       out);

        //! The merit at `x` itself: one trial step of length zero.  Both
        //! accumulate into `out` rather than assigning, as every 0-form here
        //! does, so a caller may sum several.
        /**
         * @brief Whether every operator has a potential, so `energy_merit` exists.
         *
         * A question, not a default: the caller still names the merit it wants.
         * `residual_merit` needs no such guard -- every system has a residual.
         */
        bool has_energy_merit() const;

        int energy_merit(const real_t *x, real_t *const out);
        int residual_merit(const real_t *x, real_t *const out);

        int apply_constraints(real_t *const x);
        int constraints_gradient(const real_t *const x, real_t *const g);
        int apply_zero_constraints(real_t *const x);
        int set_value_to_constrained_dofs(const real_t val, real_t *const x);
        int copy_constrained_dofs(const real_t *const src, real_t *const dest);
        int report_solution(const real_t *const x);
        int initial_guess(real_t *const x);
        int constraints_mask(mask_t *mask);

        int set_output_dir(const smesh::Path &path);

        std::shared_ptr<Output> output();
        ExecutionSpace          execution_space() const;

        std::shared_ptr<Operator<real_t>> linear_op_variant(const std::vector<std::pair<std::string, int>> &opts);

        double flops() const;
        double flops_value() const;
        double flops_gradient() const;
        double flops_apply() const;

        size_t memory_traffic_bytes() const;
        size_t memory_traffic_bytes_value() const;
        size_t memory_traffic_bytes_gradient() const;
        size_t memory_traffic_bytes_apply() const;

        void describe(std::ostream &os) const;

        /// True iff every attached Op is linear (empty ops count as linear).
        bool is_linear() const;

    private:
        friend class ParallelMatrixFreeOperator;

        /// Flat subrange within @p scope (parallel overlap); block layout resolved inside ops.
        int apply_scope_flat_range(const real_t *const x,
                                   const real_t *const h,
                                   real_t *const       out,
                                   const ElementScope  scope,
                                   const ptrdiff_t     flat_begin,
                                   const ptrdiff_t     flat_end);

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::pair<smesh::ElemType, std::shared_ptr<Buffer<idx_t *>>> create_surface_from_sideset(
            const std::shared_ptr<FunctionSpace> &space,
            const std::shared_ptr<Sideset>       &sideset);

    SharedBuffer<idx_t *> mesh_connectivity_from_file(const std::shared_ptr<Communicator> &comm, const char *folder);

}  // namespace sfem

#endif  // SFEM_FUNCTION_HPP
