#ifndef SFEM_FUNCTION_HPP
#define SFEM_FUNCTION_HPP

#include <mpi.h>
#include <algorithm>
#include <memory>
#include <string>

#include "isolver_function.h"

namespace sfem {

    enum ExecutionSpace { EXECUTION_SPACE_HOST = 0, EXECUTION_SPACE_DEVICE = 1 };

    class Function;
    class Mesh;
    class FunctionSpace;
    class Op;

    class DirichletConditions;
    class NeumannBoundaryConditions;

    class Mesh final {
    public:
        Mesh();
        Mesh(MPI_Comm comm);
        ~Mesh();

        friend class FunctionSpace;
        friend class Op;
        // friend class NeumannBoundaryConditions;

        int read(const char *path);
        int initialize_node_to_node_graph();
        int convert_to_macro_element_mesh();

        const isolver_idx_t *node_to_node_rowptr() const;
        const isolver_idx_t *node_to_node_colidx() const;

        void *impl_mesh();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class FunctionSpace final {
    public:
        FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size = 1);
        ~FunctionSpace();

        int create_crs_graph(ptrdiff_t *nlocal,
                             ptrdiff_t *nglobal,
                             ptrdiff_t *nnz,
                             isolver_idx_t **rowptr,
                             isolver_idx_t **colidx);

        int destroy_crs_graph(isolver_idx_t *rowptr, isolver_idx_t *colidx);

        int create_vector(ptrdiff_t *nlocal, ptrdiff_t *nglobal, isolver_scalar_t **values);
        int destroy_vector(isolver_scalar_t *values);

        Mesh &mesh();
        int block_size() const;
        ptrdiff_t n_dofs() const;

        friend class Op;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Op {
    public:
        virtual ~Op() = default;

        virtual const char *name() const = 0;

        virtual int initialize() { return ISOLVER_FUNCTION_SUCCESS; }
        virtual int hessian_crs(const isolver_scalar_t *const x,
                                const isolver_idx_t *const rowptr,
                                const isolver_idx_t *const colidx,
                                isolver_scalar_t *const values) = 0;

        virtual int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) = 0;
        virtual int apply(const isolver_scalar_t *const x,
                          const isolver_scalar_t *const h,
                          isolver_scalar_t *const out) = 0;

        virtual int value(const isolver_scalar_t *x, isolver_scalar_t *const out) = 0;
        virtual int report(const isolver_scalar_t *const /*x*/) { return ISOLVER_FUNCTION_SUCCESS; }
        virtual ExecutionSpace execution_space() const { return EXECUTION_SPACE_HOST; }
    };

    class NeumannBoundaryConditions final : public Op {
    public:
        static std::unique_ptr<NeumannBoundaryConditions> create_from_env(
            const std::shared_ptr<FunctionSpace> &space);

        const char *name() const override;

        NeumannBoundaryConditions(const std::shared_ptr<FunctionSpace> &space);
        ~NeumannBoundaryConditions();

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override;

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override;
        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override;

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Constraint {
    public:
        virtual ~Constraint() = default;
        virtual int apply_constraints(isolver_scalar_t *const x) = 0;
        virtual int apply_zero_constraints(isolver_scalar_t *const x) = 0;
        virtual int copy_constrained_dofs(const isolver_scalar_t *const src,
                                          isolver_scalar_t *const dest) = 0;

        virtual int hessian_crs(const isolver_scalar_t *const x,
                                const isolver_idx_t *const rowptr,
                                const isolver_idx_t *const colidx,
                                isolver_scalar_t *const values) = 0;
    };

    class DirichletConditions final : public Constraint {
    public:
        DirichletConditions(const std::shared_ptr<FunctionSpace> &space);
        ~DirichletConditions();

        static std::unique_ptr<DirichletConditions> create_from_env(
            const std::shared_ptr<FunctionSpace> &space);
        int apply_constraints(isolver_scalar_t *const x) override;
        int apply_zero_constraints(isolver_scalar_t *const x) override;
        int copy_constrained_dofs(const isolver_scalar_t *const src,
                                  isolver_scalar_t *const dest) override;

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override;

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           isolver_idx_t *const idx,
                           const int component,
                           isolver_scalar_t *const values);

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           isolver_idx_t *const idx,
                           const int component,
                           const isolver_scalar_t value);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Function final /* : public isolver::Function */ {
    public:
        Function(const std::shared_ptr<FunctionSpace> &space);
        ~Function();

        void add_operator(const std::shared_ptr<Op> &op);
        void add_constraint(const std::shared_ptr<Constraint> &c);
        void add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c);

        int create_crs_graph(ptrdiff_t *nlocal,
                             ptrdiff_t *nglobal,
                             ptrdiff_t *nnz,
                             isolver_idx_t **rowptr,
                             isolver_idx_t **colidx);

        int destroy_crs_graph(isolver_idx_t *rowptr, isolver_idx_t *colidx);

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values);

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out);
        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out);

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out);

        int apply_constraints(isolver_scalar_t *const x);
        int apply_zero_constraints(isolver_scalar_t *const x);
        int copy_constrained_dofs(const isolver_scalar_t *const src, isolver_scalar_t *const dest);
        int report_solution(const isolver_scalar_t *const x);
        int initial_guess(isolver_scalar_t *const x);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Factory {
    public:
        using FactoryFunction =
            std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &)>;

        static void register_op(const std::string &name, FactoryFunction factory_function);
        static std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space,
                                             const char *name);

    private:
        static Factory &instance();

        Factory();
        ~Factory();

        class Impl;
        std::unique_ptr<Impl> impl_;

        void private_register_op(const std::string &name, FactoryFunction factory_function);
    };
}  // namespace sfem

#endif  // SFEM_FUNCTION_HPP
