#ifndef SFEM_FUNCTION_HPP
#define SFEM_FUNCTION_HPP

#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <iostream>

#include "sfem_base.h"
#include "sfem_defs.h"

#include "isolver_function.h"

namespace sfem {

    enum ExecutionSpace { EXECUTION_SPACE_HOST = 0, EXECUTION_SPACE_DEVICE = 1 };

    enum MemorySpace {
        MEMORY_SPACE_HOST = EXECUTION_SPACE_HOST,
        MEMORY_SPACE_DEVICE = EXECUTION_SPACE_DEVICE
    };

    class Function;
    class Mesh;
    class FunctionSpace;
    class Op;

    class DirichletConditions;
    class NeumannConditions;

    template <typename T>
    class Buffer {
    public:
        Buffer(const size_t n,
               T *const ptr,
               std::function<void(void *)> destroy,
               MemorySpace mem_space)
            : n_(n), ptr_(ptr), destroy_(destroy), mem_space_(mem_space) {}

        ~Buffer() {
            if (destroy_) {
                destroy_((void *)ptr_);
            }
        }

        inline T *const data() { return ptr_; }
        inline const T *const data() const { return ptr_; }
        inline size_t size() const { return n_; }
        inline MemorySpace mem_space() const { return mem_space_; }

        void print(std::ostream &os) {
            if (mem_space_ == MEMORY_SPACE_DEVICE) {
                os << "On the device!\n";
                return;
            } else {
                for (std::ptrdiff_t i = 0; i < n_; i++) {
                    os << ptr_[i] << " ";
                }
                os << "\n";
            }
        }

    private:
        size_t n_{0};
        T *ptr_{nullptr};
        std::function<void(void *)> destroy_;
        MemorySpace mem_space_;
    };

    template <typename T>
    std::shared_ptr<Buffer<T>> h_buffer(const std::ptrdiff_t n) {
        auto ret =
            std::make_shared<Buffer<T>>(n, (T *)calloc(n, sizeof(T)), &free, MEMORY_SPACE_HOST);
        return ret;
    }

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

        const isolver_idx_t *node_to_node_rowptr() const;
        const isolver_idx_t *node_to_node_colidx() const;

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
        FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size = 1);
        ~FunctionSpace();

        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<Mesh> &mesh,
                                                     const int block_size = 1) {
            return std::make_shared<FunctionSpace>(mesh, block_size);
        }

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

        virtual bool is_linear() const = 0;
        virtual int initialize() { return ISOLVER_FUNCTION_SUCCESS; }
        virtual int hessian_crs(const isolver_scalar_t *const x,
                                const isolver_idx_t *const rowptr,
                                const isolver_idx_t *const colidx,
                                isolver_scalar_t *const values) = 0;

        virtual int hessian_diag(const isolver_scalar_t *const /*x*/,
                                 isolver_scalar_t *const /*values*/) {
            return ISOLVER_FUNCTION_FAILURE;
        }

        virtual int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) = 0;
        virtual int apply(const isolver_scalar_t *const x,
                          const isolver_scalar_t *const h,
                          isolver_scalar_t *const out) = 0;

        virtual int value(const isolver_scalar_t *x, isolver_scalar_t *const out) = 0;
        virtual int report(const isolver_scalar_t *const /*x*/) { return ISOLVER_FUNCTION_SUCCESS; }
        virtual ExecutionSpace execution_space() const { return EXECUTION_SPACE_HOST; }

        virtual void set_field(const char * /*name*/,
                               const int /*component*/,
                               isolver_scalar_t * /*x*/) {
            assert(0);
        }
    };

    class NeumannConditions final : public Op {
    public:
        static std::shared_ptr<NeumannConditions> create_from_env(
            const std::shared_ptr<FunctionSpace> &space);

        const char *name() const override;

        NeumannConditions(const std::shared_ptr<FunctionSpace> &space);
        ~NeumannConditions();

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override;

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override;

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override;

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override;

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
        virtual int apply(isolver_scalar_t *const x) = 0;
        virtual int apply_value(const isolver_scalar_t value, isolver_scalar_t *const x) = 0;
        virtual int apply_zero(isolver_scalar_t *const x);
        virtual int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const g) = 0;
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

        std::shared_ptr<FunctionSpace> space();

        static std::shared_ptr<DirichletConditions> create_from_env(
            const std::shared_ptr<FunctionSpace> &space);
        int apply(isolver_scalar_t *const x) override;
        int apply_value(const isolver_scalar_t value, isolver_scalar_t *const x) override;
        int copy_constrained_dofs(const isolver_scalar_t *const src,
                                  isolver_scalar_t *const dest) override;

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const g) override;

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

        int n_conditions() const;
        void *impl_conditions();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Output {
    public:
        Output(const std::shared_ptr<FunctionSpace> &space);
        ~Output();
        void set_output_dir(const char *path);
        int write(const char *name, const isolver_scalar_t *const x);
        int write_time_step(const char *name,
                            const isolver_scalar_t t,
                            const isolver_scalar_t *const x);

        void clear();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Function final /* : public isolver::Function */ {
    public:
        Function(const std::shared_ptr<FunctionSpace> &space);
        ~Function();

        inline static std::shared_ptr<Function> create(
            const std::shared_ptr<FunctionSpace> &space) {
            return std::make_shared<Function>(space);
        }

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

        int hessian_diag(const isolver_scalar_t *const x, isolver_scalar_t *const values);

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out);
        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out);

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out);

        int apply_constraints(isolver_scalar_t *const x);
        int constraints_gradient(const isolver_scalar_t *const x, isolver_scalar_t *const g);
        int apply_zero_constraints(isolver_scalar_t *const x);
        int copy_constrained_dofs(const isolver_scalar_t *const src, isolver_scalar_t *const dest);
        int report_solution(const isolver_scalar_t *const x);
        int initial_guess(isolver_scalar_t *const x);

        int set_output_dir(const char *path);

        std::shared_ptr<Output> output();

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

        static std::shared_ptr<Op> create_op_gpu(const std::shared_ptr<FunctionSpace> &space,
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
}  // namespace sfem

#endif  // SFEM_FUNCTION_HPP
