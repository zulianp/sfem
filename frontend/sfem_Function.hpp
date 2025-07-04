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

namespace sfem {

    class Sideset final {
    public:
        int                                    read(MPI_Comm comm, const char *path);
        std::shared_ptr<Buffer<element_idx_t>> parent();
        std::shared_ptr<Buffer<int16_t>>       lfi();
        static std::shared_ptr<Sideset>        create_from_file(MPI_Comm comm, const char *path);
        ptrdiff_t                              size() const;
        MPI_Comm                               comm() const;

        Sideset(MPI_Comm comm, const std::shared_ptr<Buffer<element_idx_t>> &parent, const std::shared_ptr<Buffer<int16_t>> &lfi);
        Sideset();
        ~Sideset();

        static std::shared_ptr<Sideset> create(MPI_Comm                                      comm,
                                               const std::shared_ptr<Buffer<element_idx_t>> &parent,
                                               const std::shared_ptr<Buffer<int16_t>>       &lfi);

        static std::shared_ptr<Sideset> create_from_selector(
                const std::shared_ptr<Mesh>                                         &mesh,
                const std::function<bool(const geom_t, const geom_t, const geom_t)> &selector);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class FunctionSpace final {
    public:
        FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size = 1, const enum ElemType element_type = INVALID);
        ~FunctionSpace();

        int promote_to_semi_structured(const int level);

        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<Mesh> &mesh,
                                                     const int                    block_size   = 1,
                                                     const enum ElemType          element_type = INVALID) {
            return std::make_shared<FunctionSpace>(mesh, block_size, element_type);
        }

        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<SemiStructuredMesh> &mesh, const int block_size = 1);

        int create_vector(ptrdiff_t *nlocal, ptrdiff_t *nglobal, real_t **values);
        int destroy_vector(real_t *values);

        void                                   set_device_elements(const std::shared_ptr<sfem::Buffer<idx_t *>> &elems);
        std::shared_ptr<sfem::Buffer<idx_t *>> device_elements();

        Mesh                 &mesh();
        std::shared_ptr<Mesh> mesh_ptr() const;

        bool                has_semi_structured_mesh() const;
        SemiStructuredMesh &semi_structured_mesh();

        int       block_size() const;
        ptrdiff_t n_dofs() const;

        enum ElemType element_type() const;

        std::shared_ptr<FunctionSpace> derefine(const int to_level = 1);
        std::shared_ptr<FunctionSpace> lor() const;

        std::shared_ptr<CRSGraph> dof_to_dof_graph();
        std::shared_ptr<CRSGraph> node_to_node_graph();

        friend class Op;

        // private
        FunctionSpace();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Op {
    public:
        virtual ~Op() = default;

        virtual const char *name() const = 0;

        virtual bool is_linear() const = 0;
        virtual int  initialize() { return SFEM_SUCCESS; }
        virtual int  hessian_crs(const real_t *const  x,
                                 const count_t *const rowptr,
                                 const idx_t *const   colidx,
                                 real_t *const        values) = 0;

        virtual int hessian_bsr(const real_t *const /*x*/,
                                const count_t *const /*rowptr*/,
                                const idx_t *const /*colidx*/,
                                real_t *const /*values*/) {
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }

        virtual int hessian_bcrs_sym(const real_t *const /*x*/,
                                     const count_t *const /*rowidx*/,
                                     const idx_t *const /*colidx*/,
                                     const ptrdiff_t /*block_stride*/,
                                     real_t **const /*diag_values*/,
                                     real_t **const /*off_diag_values*/) {
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }

        virtual int hessian_crs_sym(const real_t *const  x,
                                    const count_t *const rowptr,
                                    const idx_t *const   colidx,
                                    real_t *const        diag_values,
                                    real_t *const        off_diag_values) {
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }

        virtual int hessian_diag(const real_t *const /*x*/, real_t *const /*values*/) {
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }

        virtual int hessian_block_diag_sym(const real_t *const x, real_t *const values) {
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }

        virtual int hessian_block_diag_sym_soa(const real_t *const x, real_t **const values) {
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }

        virtual int gradient(const real_t *const x, real_t *const out)                     = 0;
        virtual int apply(const real_t *const x, const real_t *const h, real_t *const out) = 0;

        virtual int            value(const real_t *x, real_t *const out) = 0;
        virtual int            report(const real_t *const /*x*/) { return SFEM_SUCCESS; }
        virtual ExecutionSpace execution_space() const { return EXECUTION_SPACE_HOST; }

        virtual void set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &x, const int component = 0) {
            SFEM_UNUSED(name);
            SFEM_UNUSED(x);
            SFEM_UNUSED(component);
            assert(false);
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

        virtual bool is_no_op() const { return false; }

        virtual void                set_option(const std::string                &/*name*/, bool /*val*/) {}
        virtual std::shared_ptr<Op> clone() const {
            assert(false);
            return nullptr;
        }
    };

    class NoOp final : public Op {
    public:
        const char *name() const override { return "NoOp"; }

        bool is_linear() const override { return true; }
        int  hessian_crs(const real_t *const /*x*/,
                         const count_t *const /*rowptr*/,
                         const idx_t *const /*colidx*/,
                         real_t *const /*values*/) override {
            return SFEM_SUCCESS;
        }
        int  gradient(const real_t *const x, real_t *const out) override { return SFEM_SUCCESS; }
        int  apply(const real_t *const x, const real_t *const h, real_t *const out) override { return SFEM_SUCCESS; }
        bool is_no_op() const override { return true; }
        int  value(const real_t  */*x*/, real_t *const /*out*/) override { return SFEM_SUCCESS; }
    };

    inline std::shared_ptr<NoOp> no_op() { return std::make_shared<NoOp>(); }

    class NeumannConditions final : public Op {
    public:
        struct Condition {
            enum ElemType                    element_type { INVALID };
            std::shared_ptr<Sideset>         sideset;  /// Maybe undefined in certain cases
            std::shared_ptr<Buffer<idx_t *>> surface;
            std::shared_ptr<Buffer<real_t>>  values;
            real_t                           value{0};
            int                              component{0};
        };

        static std::shared_ptr<NeumannConditions> create_from_env(const std::shared_ptr<FunctionSpace> &space);

        static std::shared_ptr<NeumannConditions> create(const std::shared_ptr<FunctionSpace> &space,
                                                         const std::vector<struct Condition>  &conditions);

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override;

        const char *name() const override;

        NeumannConditions(const std::shared_ptr<FunctionSpace> &space);
        ~NeumannConditions();

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int gradient(const real_t *const x, real_t *const out) override;

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;

        int value(const real_t *x, real_t *const out) override;

        inline bool is_linear() const override { return true; }

        int                            n_conditions() const;
        std::shared_ptr<FunctionSpace> space();
        std::vector<struct Condition> &conditions();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Constraint {
    public:
        virtual ~Constraint()                                        = default;
        virtual int apply(real_t *const x)                           = 0;
        virtual int apply_value(const real_t value, real_t *const x) = 0;
        virtual int apply_zero(real_t *const x);
        virtual int gradient(const real_t *const x, real_t *const g)                   = 0;
        virtual int copy_constrained_dofs(const real_t *const src, real_t *const dest) = 0;
        virtual int mask(mask_t *mask)                                                 = 0;

        virtual int hessian_crs(const real_t *const  x,
                                const count_t *const rowptr,
                                const idx_t *const   colidx,
                                real_t *const        values) = 0;

        virtual int hessian_bsr(const real_t *const /*x*/,
                                const count_t *const /*rowptr*/,
                                const idx_t *const /*colidx*/,
                                real_t *const /*values*/) {
            assert(false);
            return SFEM_FAILURE;
        }

        virtual std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                                     const bool                            as_zero) const = 0;
        virtual std::shared_ptr<Constraint> lor() const                        = 0;
    };

    class DirichletConditions final : public Constraint {
    public:
        struct Condition {
            std::shared_ptr<Sideset>        sideset;  /// Maybe undefined in certain cases
            std::shared_ptr<Buffer<idx_t>>  nodeset;
            std::shared_ptr<Buffer<real_t>> values;
            real_t                          value{0};
            int                             component{0};
        };

        DirichletConditions(const std::shared_ptr<FunctionSpace> &space);
        ~DirichletConditions();

        std::shared_ptr<FunctionSpace> space();
        std::vector<struct Condition> &conditions();

        static std::shared_ptr<DirichletConditions> create_from_env(const std::shared_ptr<FunctionSpace> &space);
        static std::shared_ptr<DirichletConditions> create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                     const std::string                    &path);

        static std::shared_ptr<DirichletConditions> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                     std::string                           yaml);

        static std::shared_ptr<DirichletConditions> create(const std::shared_ptr<FunctionSpace> &space,
                                                           const std::vector<struct Condition>  &conditions);

        int apply(real_t *const x) override;
        int apply_value(const real_t value, real_t *const x) override;
        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override;
        int mask(mask_t *mask) override;

        int gradient(const real_t *const x, real_t *const g) override;

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_bsr(const real_t *const  x,
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

        void add_operator(const std::shared_ptr<Op> &op);
        void add_constraint(const std::shared_ptr<Constraint> &c);
        void clear_constraints();
        void add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c);

        std::shared_ptr<CRSGraph> crs_graph() const;

        // int create_crs_graph(ptrdiff_t *nlocal,
        //                      ptrdiff_t *nglobal,
        //                      ptrdiff_t *nnz,
        //                      count_t **rowptr,
        //                      idx_t **colidx);

        // int destroy_crs_graph(count_t *rowptr, idx_t *colidx);

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

    class Factory {
    public:
        using FactoryFunction = std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &)>;

        using FactoryFunctionBoundary = std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &,
                                                                          const std::shared_ptr<Buffer<idx_t *>> &)>;

        static void                register_op(const std::string &name, FactoryFunction factory_function);
        static std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space, const char *name);

        static std::shared_ptr<Op> create_op_gpu(const std::shared_ptr<FunctionSpace> &space, const char *name);

        static std::shared_ptr<Op> create_boundary_op(const std::shared_ptr<FunctionSpace>   &space,
                                                      const std::shared_ptr<Buffer<idx_t *>> &boundary_elements,
                                                      const char                             *name);

    private:
        static Factory &instance();

        Factory();
        ~Factory();

        class Impl;
        std::unique_ptr<Impl> impl_;

        void private_register_op(const std::string &name, FactoryFunction factory_function);
    };

    std::string                      d_op_str(const std::string &name);
    std::shared_ptr<Buffer<idx_t *>> mesh_connectivity_from_file(MPI_Comm comm, const char *folder);

    std::shared_ptr<Buffer<idx_t>> create_nodeset_from_sideset(const std::shared_ptr<FunctionSpace> &space,
                                                               const std::shared_ptr<Sideset>       &sideset);
}  // namespace sfem

#endif  // SFEM_FUNCTION_HPP
