#ifndef SFEM_FUNCTION_HPP
#define SFEM_FUNCTION_HPP

#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "sfem_base.h"
#include "sfem_defs.h"

// #include "isolver_function.h"

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"

namespace sfem {
    class Function;
    class Mesh;
    class FunctionSpace;
    class Op;
    class CRSGraph;

    class DirichletConditions;
    class NeumannConditions;

    class CRSGraph final {
    public:
        CRSGraph();
        ~CRSGraph();

        CRSGraph(const std::shared_ptr<Buffer<count_t>> &rowptr,
                 const std::shared_ptr<Buffer<idx_t>> &colidx);

        friend class Mesh;

        ptrdiff_t n_nodes() const;
        ptrdiff_t nnz() const;
        std::shared_ptr<Buffer<count_t>> rowptr() const;
        std::shared_ptr<Buffer<idx_t>> colidx() const;
        std::shared_ptr<CRSGraph> block_to_scalar(const int block_size);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Mesh final {
    public:
        Mesh();
        Mesh(MPI_Comm comm);
        ~Mesh();

        Mesh(int spatial_dim,
             enum ElemType element_type,
             ptrdiff_t nelements,
             idx_t **elements,
             ptrdiff_t nnodes,
             geom_t **points);

        friend class FunctionSpace;
        friend class Op;
        // friend class NeumannConditions;

        int read(const char *path);
        int write(const char *path) const;
        int initialize_node_to_node_graph();
        int convert_to_macro_element_mesh();

        int spatial_dimension() const;
        int n_nodes_per_elem() const;
        ptrdiff_t n_nodes() const;
        ptrdiff_t n_elements() const;

        std::shared_ptr<CRSGraph> node_to_node_graph();
        std::shared_ptr<CRSGraph> create_node_to_node_graph(const enum ElemType element_type);

        std::shared_ptr<Buffer<count_t>> node_to_node_rowptr() const;
        std::shared_ptr<Buffer<idx_t>> node_to_node_colidx() const;

        const geom_t *const points(const int coord) const;
        const idx_t *const idx(const int node_num) const;

        void *impl_mesh();

        inline static std::shared_ptr<Mesh> create_from_file(MPI_Comm comm, const char *path) {
            auto ret = std::make_shared<Mesh>(comm);
            ret->read(path);
            return ret;
        }

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class FunctionSpace final {
    public:
        FunctionSpace(const std::shared_ptr<Mesh> &mesh,
                      const int block_size = 1,
                      const enum ElemType element_type = INVALID);
        ~FunctionSpace();

        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<Mesh> &mesh,
                                                     const int block_size = 1,
                                                     const enum ElemType element_type = INVALID) {
            return std::make_shared<FunctionSpace>(mesh, block_size, element_type);
        }

        int create_vector(ptrdiff_t *nlocal, ptrdiff_t *nglobal, real_t **values);
        int destroy_vector(real_t *values);

        Mesh &mesh();
        int block_size() const;
        ptrdiff_t n_dofs() const;

        enum ElemType element_type() const;

        std::shared_ptr<FunctionSpace> derefine() const;
        std::shared_ptr<FunctionSpace> lor() const;

        std::shared_ptr<CRSGraph> dof_to_dof_graph();
        std::shared_ptr<CRSGraph> node_to_node_graph();

        friend class Op;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Op {
    public:
        virtual ~Op() = default;

        virtual const char *name() const = 0;

        virtual bool is_linear() const = 0;
        virtual int initialize() { return SFEM_SUCCESS; }
        virtual int hessian_crs(const real_t *const x,
                                const count_t *const rowptr,
                                const idx_t *const colidx,
                                real_t *const values) = 0;

        virtual int hessian_diag(const real_t *const /*x*/, real_t *const /*values*/) {
            return SFEM_FAILURE;
        }

        virtual int gradient(const real_t *const x, real_t *const out) = 0;
        virtual int apply(const real_t *const x, const real_t *const h, real_t *const out) = 0;

        virtual int value(const real_t *x, real_t *const out) = 0;
        virtual int report(const real_t *const /*x*/) { return SFEM_SUCCESS; }
        virtual ExecutionSpace execution_space() const { return EXECUTION_SPACE_HOST; }

        virtual void set_field(const char * /*name*/, const int /*component*/, real_t * /*x*/) {
            assert(0);
        }

        /// Make low-order-refinement operator
        virtual std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &) {
            assert(false);
            return nullptr;
        }

        virtual std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &) {
            assert(false);
            return nullptr;
        }
    };

    class NeumannConditions final : public Op {
    public:
        static std::shared_ptr<NeumannConditions> create_from_env(
                const std::shared_ptr<FunctionSpace> &space);

        const char *name() const override;

        NeumannConditions(const std::shared_ptr<FunctionSpace> &space);
        ~NeumannConditions();

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;

        int gradient(const real_t *const x, real_t *const out) override;

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;

        int value(const real_t *x, real_t *const out) override;

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const idx,
                           const int component,
                           real_t *const values);

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const idx,
                           const int component,
                           const real_t value);

        inline bool is_linear() const override { return true; }

        int n_conditions() const;
        void *impl_conditions();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Constraint {
    public:
        virtual ~Constraint() = default;
        virtual int apply(real_t *const x) = 0;
        virtual int apply_value(const real_t value, real_t *const x) = 0;
        virtual int apply_zero(real_t *const x);
        virtual int gradient(const real_t *const x, real_t *const g) = 0;
        virtual int copy_constrained_dofs(const real_t *const src, real_t *const dest) = 0;

        virtual int hessian_crs(const real_t *const x,
                                const count_t *const rowptr,
                                const idx_t *const colidx,
                                real_t *const values) = 0;

        virtual std::shared_ptr<Constraint> derefine(
                const std::shared_ptr<FunctionSpace> &coarse_space,
                const bool as_zero) const = 0;
        virtual std::shared_ptr<Constraint> lor() const = 0;
    };

    class DirichletConditions final : public Constraint {
    public:
        DirichletConditions(const std::shared_ptr<FunctionSpace> &space);
        ~DirichletConditions();

        std::shared_ptr<FunctionSpace> space();

        static std::shared_ptr<DirichletConditions> create_from_env(
                const std::shared_ptr<FunctionSpace> &space);
        int apply(real_t *const x) override;
        int apply_value(const real_t value, real_t *const x) override;
        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override;

        int gradient(const real_t *const x, real_t *const g) override;

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const idx,
                           const int component,
                           real_t *const values);

        void add_condition(const ptrdiff_t local_size,
                           const ptrdiff_t global_size,
                           idx_t *const idx,
                           const int component,
                           const real_t value);

        int n_conditions() const;
        void *impl_conditions();

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                             const bool as_zero) const override;
        std::shared_ptr<Constraint> lor() const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Output {
    public:
        Output(const std::shared_ptr<FunctionSpace> &space);
        ~Output();
        void set_output_dir(const char *path);
        int write(const char *name, const real_t *const x);
        int write_time_step(const char *name, const real_t t, const real_t *const x);

        void clear();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Function final /* : public isolver::Function */ {
    public:
        Function(const std::shared_ptr<FunctionSpace> &space);
        ~Function();

        std::shared_ptr<Function> derefine(const bool dirichlet_as_zero);
        std::shared_ptr<Function> derefine(const std::shared_ptr<FunctionSpace> &space,
                                           const bool dirichlet_as_zero);

        inline static std::shared_ptr<Function> create(
                const std::shared_ptr<FunctionSpace> &space) {
            return std::make_shared<Function>(space);
        }

        std::shared_ptr<FunctionSpace> space();

        std::shared_ptr<Function> lor();
        std::shared_ptr<Function> lor(const std::shared_ptr<FunctionSpace> &space);

        void add_operator(const std::shared_ptr<Op> &op);
        void add_constraint(const std::shared_ptr<Constraint> &c);
        void add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c);

        std::shared_ptr<CRSGraph> crs_graph() const;

        // int create_crs_graph(ptrdiff_t *nlocal,
        //                      ptrdiff_t *nglobal,
        //                      ptrdiff_t *nnz,
        //                      count_t **rowptr,
        //                      idx_t **colidx);

        // int destroy_crs_graph(count_t *rowptr, idx_t *colidx);

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values);

        int hessian_diag(const real_t *const x, real_t *const values);

        int gradient(const real_t *const x, real_t *const out);
        int apply(const real_t *const x, const real_t *const h, real_t *const out);

        int value(const real_t *x, real_t *const out);

        int apply_constraints(real_t *const x);
        int constraints_gradient(const real_t *const x, real_t *const g);
        int apply_zero_constraints(real_t *const x);
        int copy_constrained_dofs(const real_t *const src, real_t *const dest);
        int report_solution(const real_t *const x);
        int initial_guess(real_t *const x);

        int set_output_dir(const char *path);

        std::shared_ptr<Output> output();

        std::shared_ptr<Operator<real_t>> hierarchical_restriction();
        std::shared_ptr<Operator<real_t>> hierarchical_prolongation();

        ExecutionSpace execution_space() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Factory {
    public:
        using FactoryFunction =
                std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &)>;

        using FactoryFunctionBoundary =
                std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &, const std::shared_ptr<Buffer<idx_t *>> &)>;

        static void register_op(const std::string &name, FactoryFunction factory_function);
        static std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space,
                                             const char *name);

        static std::shared_ptr<Op> create_op_gpu(const std::shared_ptr<FunctionSpace> &space,
                                                 const char *name);

        static std::shared_ptr<Op> create_boundary_op(
                const std::shared_ptr<FunctionSpace> &space,
                const std::shared_ptr<Buffer<idx_t *>> &boundary_elements,
                const char *name);

    private:
        static Factory &instance();

        Factory();
        ~Factory();

        class Impl;
        std::unique_ptr<Impl> impl_;

        void private_register_op(const std::string &name, FactoryFunction factory_function);
    };

    std::string d_op_str(const std::string &name);
    std::shared_ptr<Buffer<idx_t *>> mesh_connectivity_from_file(MPI_Comm comm,
                                                                 const char *folder);
}  // namespace sfem

#endif  // SFEM_FUNCTION_HPP
