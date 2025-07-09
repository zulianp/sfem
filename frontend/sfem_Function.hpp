#ifndef SFEM_FUNCTION_HPP
#define SFEM_FUNCTION_HPP

#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sfem_base.h"
#include "sfem_defs.h"

#include "sfem_mask.h"

// #include "isolver_function.h"

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_glob.hpp"
#include "sfem_Sideset.hpp"
#include "sfem_NeumannConditions.hpp"
#include "sfem_DirichletConditions.hpp"

// Operator includes
#include "sfem_Op.hpp"
#include "sfem_OpFactory.hpp"

#include "sfem_Constraint.hpp"

namespace sfem {

    class Output {
    public:
        Output(const std::shared_ptr<FunctionSpace> &space);
        ~Output();
        void set_output_dir(const char *path);
        int  write(const char *name, const real_t *const x);
        int  write_time_step(const char *name, const real_t t, const real_t *const x);
        void enable_AoS_to_SoA(const bool val);
        void clear();

        void log_time(const real_t t);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Function final /* : public isolver::Function */ {
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

        int gradient(const real_t *const x, real_t *const out);
        int apply(const real_t *const x, const real_t *const h, real_t *const out);

        int value(const real_t *x, real_t *const out);

        int apply_constraints(real_t *const x);
        int constraints_gradient(const real_t *const x, real_t *const g);
        int apply_zero_constraints(real_t *const x);
        int set_value_to_constrained_dofs(const real_t val, real_t *const x);
        int copy_constrained_dofs(const real_t *const src, real_t *const dest);
        int report_solution(const real_t *const x);
        int initial_guess(real_t *const x);
        int constaints_mask(mask_t *mask);

        int set_output_dir(const char *path);

        std::shared_ptr<Output> output();
        ExecutionSpace          execution_space() const;

        std::shared_ptr<Operator<real_t>> linear_op_variant(const std::vector<std::pair<std::string, int>> &opts);

        void describe(std::ostream &os) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::pair<enum ElemType, std::shared_ptr<Buffer<idx_t *>>> create_surface_from_sideset(
            const std::shared_ptr<FunctionSpace> &space,
            const std::shared_ptr<Sideset>       &sideset);

    SharedBuffer<idx_t *> mesh_connectivity_from_file(const std::shared_ptr<Communicator>& comm, const char *folder);

} // namespace sfem

#endif //SFEM_FUNCTION_HPP