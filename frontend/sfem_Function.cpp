#include "sfem_Function.hpp"

#include "matrixio_array.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "boundary_condition.h"
#include "boundary_condition_io.h"

#include "dirichlet.h"
#include "neumann.h"

#include <functional>
#include <iostream>
#include <map>
#include <vector>

#include "linear_elasticity.h"

namespace sfem {

    class Mesh::Impl {
    public:
        MPI_Comm comm;
        mesh_t mesh;

        isolver_idx_t *node_to_node_rowptr{nullptr};
        isolver_idx_t *node_to_node_colidx{nullptr};

        ~Impl() {
            mesh_destroy(&mesh);
            free(node_to_node_rowptr);
            free(node_to_node_colidx);
        }
    };

    Mesh::Mesh(MPI_Comm comm) : impl_(std::make_unique<Impl>()) {
        impl_->comm = comm;
        mesh_init(&impl_->mesh);
    }

    Mesh::~Mesh() = default;

    int Mesh::read(const char *path) {
        if (mesh_read(impl_->comm, path, &impl_->mesh)) {
            return ISOLVER_FUNCTION_FAILURE;
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Mesh::initialize_node_to_node_graph() {
        auto mesh = &impl_->mesh;

        if (!impl_->node_to_node_rowptr) {
            build_crs_graph_for_elem_type(mesh->element_type,
                                          mesh->nelements,
                                          mesh->nnodes,
                                          mesh->elements,
                                          &impl_->node_to_node_rowptr,
                                          &impl_->node_to_node_colidx);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Mesh::convert_to_macro_element_mesh() {
        impl_->mesh.element_type = macro_type_variant((enum ElemType)impl_->mesh.element_type);
        return ISOLVER_FUNCTION_SUCCESS;
    }

    void *Mesh::impl_mesh() { return (void *)&impl_->mesh; }

    const isolver_idx_t *Mesh::node_to_node_rowptr() const { return impl_->node_to_node_rowptr; }
    const isolver_idx_t *Mesh::node_to_node_colidx() const { return impl_->node_to_node_colidx; }

    class FunctionSpace::Impl {
    public:
        std::shared_ptr<Mesh> mesh;
        int block_size{1};

        // CRS graph
        ptrdiff_t nlocal{0};
        ptrdiff_t nglobal{0};
        ptrdiff_t nnz{0};
        isolver_idx_t *rowptr{nullptr};
        isolver_idx_t *colidx{nullptr};

        ~Impl() {
            free(rowptr);
            free(colidx);
        }
    };

    FunctionSpace::FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size)
        : impl_(std::make_unique<Impl>()) {
        impl_->mesh = mesh;
        impl_->block_size = block_size;
    }

    FunctionSpace::~FunctionSpace() = default;

    Mesh &FunctionSpace::mesh() { return *impl_->mesh; }

    int FunctionSpace::block_size() const { return impl_->block_size; }

    int FunctionSpace::create_crs_graph(ptrdiff_t *nlocal,
                                        ptrdiff_t *nglobal,
                                        ptrdiff_t *nnz,
                                        isolver_idx_t **rowptr,
                                        isolver_idx_t **colidx) {
        auto &mesh = *impl_->mesh;
        auto c_mesh = &mesh.impl_->mesh;

        // This is for nodal discretizations (CG)
        mesh.initialize_node_to_node_graph();
        if (impl_->block_size == 1) {
            *rowptr = mesh.impl_->node_to_node_rowptr;
            *colidx = mesh.impl_->node_to_node_colidx;

            *nlocal = c_mesh->nnodes;
            *nglobal = c_mesh->nnodes;
            *nnz = (*rowptr)[c_mesh->nnodes];
        } else {
            if (!impl_->rowptr) {
                impl_->rowptr =
                    (count_t *)malloc((c_mesh->nnodes + 1) * impl_->block_size * sizeof(count_t));
                impl_->colidx =
                    (idx_t *)malloc(mesh.impl_->node_to_node_rowptr[c_mesh->nnodes] *
                                    impl_->block_size * impl_->block_size * sizeof(idx_t));

                crs_graph_block_to_scalar(c_mesh->nnodes,
                                          impl_->block_size,
                                          mesh.impl_->node_to_node_rowptr,
                                          mesh.impl_->node_to_node_colidx,
                                          impl_->rowptr,
                                          impl_->colidx);

                impl_->nlocal = c_mesh->nnodes * impl_->block_size;
                impl_->nglobal = c_mesh->nnodes * impl_->block_size;
                impl_->nnz = mesh.impl_->node_to_node_rowptr[c_mesh->nnodes] *
                             (impl_->block_size * impl_->block_size);
            }

            *rowptr = impl_->rowptr;
            *colidx = impl_->colidx;

            *nlocal = impl_->nlocal;
            *nglobal = impl_->nglobal;
            *nnz = impl_->nnz;
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int FunctionSpace::destroy_crs_graph(isolver_idx_t *rowptr, isolver_idx_t *colidx) {
        if (rowptr == impl_->rowptr) {
            impl_->rowptr = nullptr;
        }

        free(rowptr);

        if (colidx == impl_->colidx) {
            impl_->colidx = nullptr;
        }
        free(colidx);

        return ISOLVER_FUNCTION_SUCCESS;
    }

    class NeumannBoundaryConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;

        ~Impl() {
            if (neumann_conditions) {
                for (int i = 0; i < n_neumann_conditions; i++) {
                    free(neumann_conditions[i].idx);
                }

                free(neumann_conditions);
            }
        }

        int n_neumann_conditions{0};
        boundary_condition_t *neumann_conditions{nullptr};
    };

    NeumannBoundaryConditions::NeumannBoundaryConditions(
        const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::unique_ptr<NeumannBoundaryConditions> NeumannBoundaryConditions::create_from_env(
        const std::shared_ptr<FunctionSpace> &space) {
        //
        auto nc = std::make_unique<NeumannBoundaryConditions>(space);

        char *SFEM_NEUMANN_SIDESET = 0;
        char *SFEM_NEUMANN_VALUE = 0;
        char *SFEM_NEUMANN_COMPONENT = 0;
        SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );
        SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );
        SFEM_READ_ENV(SFEM_NEUMANN_COMPONENT, );

        auto mesh = (mesh_t *)space->mesh().impl_mesh();
        read_neumann_conditions(mesh,
                                SFEM_NEUMANN_SIDESET,
                                SFEM_NEUMANN_VALUE,
                                SFEM_NEUMANN_COMPONENT,
                                &nc->impl_->neumann_conditions,
                                &nc->impl_->n_neumann_conditions);

        return nc;
    }

    NeumannBoundaryConditions::~NeumannBoundaryConditions() = default;

    int NeumannBoundaryConditions::hessian_crs(const isolver_scalar_t *const /*x*/,
                                               const isolver_idx_t *const /*rowptr*/,
                                               const isolver_idx_t *const /*colidx*/,
                                               isolver_scalar_t *const /*values*/) {
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int NeumannBoundaryConditions::gradient(const isolver_scalar_t *const x,
                                            isolver_scalar_t *const out) {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();

        for (int i = 0; i < impl_->n_neumann_conditions; i++) {
            surface_forcing_function_vec(side_type((enum ElemType)mesh->element_type),
                                         impl_->neumann_conditions[i].local_size,
                                         impl_->neumann_conditions[i].idx,
                                         mesh->points,
                                         -  // Use negative sign since we are on LHS
                                         impl_->neumann_conditions[i].value,
                                         impl_->space->block_size(),
                                         impl_->neumann_conditions[i].component,
                                         out);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }
    int NeumannBoundaryConditions::apply(const isolver_scalar_t *const /*x*/,
                                         const isolver_scalar_t *const /*h*/,
                                         isolver_scalar_t *const /*out*/) {
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int NeumannBoundaryConditions::value(const isolver_scalar_t *x, isolver_scalar_t *const out) {
        // TODO
        return ISOLVER_FUNCTION_SUCCESS;
    }

    class DirichletConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;

        ~Impl() {
            if (dirichlet_conditions) {
                for (int i = 0; i < n_dirichlet_conditions; i++) {
                    free(dirichlet_conditions[i].idx);
                }

                free(dirichlet_conditions);
            }
        }

        int n_dirichlet_conditions{0};
        boundary_condition_t *dirichlet_conditions{nullptr};
    };

    std::unique_ptr<DirichletConditions> DirichletConditions::create_from_env(
        const std::shared_ptr<FunctionSpace> &space) {
        //
        auto dc = std::make_unique<DirichletConditions>(space);

        char *SFEM_DIRICHLET_NODESET = 0;
        char *SFEM_DIRICHLET_VALUE = 0;
        char *SFEM_DIRICHLET_COMPONENT = 0;
        SFEM_READ_ENV(SFEM_DIRICHLET_NODESET, );
        SFEM_READ_ENV(SFEM_DIRICHLET_VALUE, );
        SFEM_READ_ENV(SFEM_DIRICHLET_COMPONENT, );

        auto mesh = (mesh_t *)space->mesh().impl_mesh();
        read_dirichlet_conditions(mesh,
                                  SFEM_DIRICHLET_NODESET,
                                  SFEM_DIRICHLET_VALUE,
                                  SFEM_DIRICHLET_COMPONENT,
                                  &dc->impl_->dirichlet_conditions,
                                  &dc->impl_->n_dirichlet_conditions);

        return dc;
    }

    int DirichletConditions::apply_constraints(isolver_scalar_t *const x) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_nodes_to_value_vec(impl_->dirichlet_conditions[i].local_size,
                                          impl_->dirichlet_conditions[i].idx,
                                          impl_->space->block_size(),
                                          impl_->dirichlet_conditions[i].component,
                                          impl_->dirichlet_conditions[i].value,
                                          x);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int DirichletConditions::apply_zero_constraints(isolver_scalar_t *const x) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_nodes_to_value_vec(impl_->dirichlet_conditions[i].local_size,
                                          impl_->dirichlet_conditions[i].idx,
                                          impl_->space->block_size(),
                                          impl_->dirichlet_conditions[i].component,
                                          0,
                                          x);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }
    int DirichletConditions::copy_constrained_dofs(const isolver_scalar_t *const src,
                                                   isolver_scalar_t *const dest) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_nodes_copy_vec(impl_->dirichlet_conditions[i].local_size,
                                      impl_->dirichlet_conditions[i].idx,
                                      impl_->space->block_size(),
                                      impl_->dirichlet_conditions[i].component,
                                      src,
                                      dest);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int DirichletConditions::hessian_crs(const isolver_scalar_t *const x,
                                         const isolver_idx_t *const rowptr,
                                         const isolver_idx_t *const colidx,
                                         isolver_scalar_t *const values) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            crs_constraint_nodes_to_identity_vec(impl_->dirichlet_conditions[i].local_size,
                                                 impl_->dirichlet_conditions[i].idx,
                                                 impl_->space->block_size(),
                                                 impl_->dirichlet_conditions[i].component,
                                                 1,
                                                 rowptr,
                                                 colidx,
                                                 values);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    class Function::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::vector<std::shared_ptr<Op>> ops;
        std::string output_dir;
    };

    Function::Function(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;

        const char *SFEM_OUTPUT_DIR = "./sfem_output";
        SFEM_READ_ENV(SFEM_OUTPUT_DIR, );
        impl_->output_dir = SFEM_OUTPUT_DIR;
    }

    Function::~Function() {}

    void Function::add_operator(const std::shared_ptr<Op> &op) { impl_->ops.push_back(op); }

    int Function::create_crs_graph(ptrdiff_t *nlocal,
                                   ptrdiff_t *nglobal,
                                   ptrdiff_t *nnz,
                                   isolver_idx_t **rowptr,
                                   isolver_idx_t **colidx) {
        return impl_->space->create_crs_graph(nlocal, nglobal, nnz, rowptr, colidx);
    }

    int Function::destroy_crs_graph(isolver_idx_t *rowptr, isolver_idx_t *colidx) {
        return impl_->space->destroy_crs_graph(rowptr, colidx);
    }

    int Function::hessian_crs(const isolver_scalar_t *const x,
                              const isolver_idx_t *const rowptr,
                              const isolver_idx_t *const colidx,
                              isolver_scalar_t *const values) {
        for (auto &op : impl_->ops) {
            if (op->hessian_crs(x, rowptr, colidx, values)) {
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) {
        for (auto &op : impl_->ops) {
            if (op->gradient(x, out)) {
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::apply(const isolver_scalar_t *const x,
                        const isolver_scalar_t *const h,
                        isolver_scalar_t *const out) {
        for (auto &op : impl_->ops) {
            if (op->apply(x, h, out)) {
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::value(const isolver_scalar_t *x, isolver_scalar_t *const out) {
        for (auto &op : impl_->ops) {
            if (op->value(x, out)) {
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::apply_constraints(isolver_scalar_t *const x) { return ISOLVER_FUNCTION_SUCCESS; }

    int Function::apply_zero_constraints(isolver_scalar_t *const x) {
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::copy_constrained_dofs(const isolver_scalar_t *const src,
                                        isolver_scalar_t *const dest) {
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::report_solution(const isolver_scalar_t *const x) {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();

        char path[2048];
        sprintf(path, "%s/out.raw", impl_->output_dir);

        if (array_write(mesh->comm,
                        path,
                        SFEM_MPI_REAL_T,
                        x,
                        mesh->nnodes * impl_->space->block_size(),
                        mesh->nnodes * impl_->space->block_size())) {
            return ISOLVER_FUNCTION_FAILURE;
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::initial_guess(isolver_scalar_t *const x) { return ISOLVER_FUNCTION_SUCCESS; }

    class FunctionPtrOp final : public Op {
    public:
        using HessianCrs_t = std::function<int(const isolver_scalar_t *const,
                                               const isolver_idx_t *const,
                                               const isolver_idx_t *const,
                                               isolver_scalar_t *const)>;

        using Gradient_t =
            std::function<int(const isolver_scalar_t *const, isolver_scalar_t *const)>;

        using Apply_t = std::function<int(const isolver_scalar_t *const,
                                          const isolver_scalar_t *const,
                                          isolver_scalar_t *const)>;

        using Value_t = std::function<int(const isolver_scalar_t *const, isolver_scalar_t *const)>;

        using Report_t = std::function<int(const isolver_scalar_t *const)>;

        HessianCrs_t hessian_crs_;
        Gradient_t gradient_;
        Apply_t apply_;
        Value_t value_;
        Report_t report_;

        FunctionPtrOp(HessianCrs_t hessian_crs,
                      Gradient_t gradient,
                      Apply_t apply,
                      Value_t value,
                      Report_t report = Report_t())
            : hessian_crs_(hessian_crs),
              gradient_(gradient),
              apply_(apply),
              value_(value),
              report_(report) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) {
            if (hessian_crs_) return hessian_crs_(x, rowptr, colidx, values);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) {
            if (gradient_) return gradient_(x, out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) {
            if (apply_) return apply_(x, h, out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) {
            if (value_) return value_(x, out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int report(const isolver_scalar_t *const x) {
            if (report_) return report_(x);
            return ISOLVER_FUNCTION_SUCCESS;
        }
    };

    class LinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;

        real_t mu{1}, lambda{1};

        static std::unique_ptr<Op> New(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(mesh.spatial_dim == space.block_size());

            auto ret = std::make_unique<LinearElasticity>(space);

            real_t SFEM_SHEAR_MODULUS = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            ret->mu = SFEM_SHEAR_MODULUS;
            ret->lambda = SFEM_FIRST_LAME_PARAMETER;
            return ret;
        }

        int initialize() override {return ISOLVER_FUNCTION_SUCCESS;}

        LinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_assemble_hessian_aos((enum ElemType)mesh->element_type,
                                                   mesh->nelements,
                                                   mesh->nnodes,
                                                   mesh->elements,
                                                   mesh->points,
                                                   this->mu,
                                                   this->lambda,
                                                   space->mesh().node_to_node_rowptr(),
                                                   space->mesh().node_to_node_colidx(),
                                                   values);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_assemble_gradient_aos((enum ElemType)mesh->element_type,
                                                    mesh->nelements,
                                                    mesh->nnodes,
                                                    mesh->elements,
                                                    mesh->points,
                                                    this->mu,
                                                    this->lambda,
                                                    x,
                                                    out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_apply_aos((enum ElemType)mesh->element_type,
                                        mesh->nelements,
                                        mesh->nnodes,
                                        mesh->elements,
                                        mesh->points,
                                        this->mu,
                                        this->lambda,
                                        h,
                                        out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_assemble_value_aos((enum ElemType)mesh->element_type,
                                                 mesh->nelements,
                                                 mesh->nnodes,
                                                 mesh->elements,
                                                 mesh->points,
                                                 this->mu,
                                                 this->lambda,
                                                 x,
                                                 out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
    };

    std::unique_ptr<Op> Factory::create_op(const std::shared_ptr<FunctionSpace> &space,
                                           const char *name) {
        using FactoryFunction =
            std::function<std::unique_ptr<Op>(const std::shared_ptr<FunctionSpace> &)>;
        static std::map<std::string, FactoryFunction> name_to_create;

        if (name_to_create.empty()) {
            name_to_create["LinearElasticity"] = LinearElasticity::New;
        }

        auto it = name_to_create.find(name);
        if (it == name_to_create.end()) {
            std::cerr << "Unable to find op " << name << "\n";
            return nullptr;
        }

        return it->second(space);
    }

}  // namespace sfem
