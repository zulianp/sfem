#include "sfem_Function.hpp"
#include <stddef.h>

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

// Ops

#include "cvfem_operators.h"
#include "laplacian.h"
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

    Mesh::Mesh() : impl_(std::make_unique<Impl>()) {
        impl_->comm = MPI_COMM_WORLD;
        mesh_init(&impl_->mesh);
    }

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

    ptrdiff_t FunctionSpace::n_dofs() const
    {
        auto &mesh = *impl_->mesh;
        auto c_mesh = &mesh.impl_->mesh;
        return c_mesh->nnodes * block_size();
    }

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

    const char *NeumannBoundaryConditions::name() const { return "NeumannBoundaryConditions"; }

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

    DirichletConditions::DirichletConditions(
        const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    DirichletConditions::~DirichletConditions() = default;

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
        std::vector<std::shared_ptr<Constraint>> constraints;
        std::string output_dir;
    };

    Function::Function(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;

        const char *SFEM_OUTPUT_DIR = "./";
        SFEM_READ_ENV(SFEM_OUTPUT_DIR, );
        impl_->output_dir = SFEM_OUTPUT_DIR;
    }

    Function::~Function() {}

    void Function::add_operator(const std::shared_ptr<Op> &op) { 
        printf("Adding operator %s\n", op->name());
        impl_->ops.push_back(op); }
    void Function::add_constraint(const std::shared_ptr<Constraint> &c) {
        impl_->constraints.push_back(c);
    }

    void Function::add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c)
    {
        add_constraint(c);
    }

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
            printf("Calling apply on %s\n", op->name());
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

    int Function::apply_constraints(isolver_scalar_t *const x) {
        for (auto &c : impl_->constraints) {
            c->apply_constraints(x);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::apply_zero_constraints(isolver_scalar_t *const x) {
        for (auto &c : impl_->constraints) {
            c->apply_zero_constraints(x);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::copy_constrained_dofs(const isolver_scalar_t *const src,
                                        isolver_scalar_t *const dest) {
        for (auto &c : impl_->constraints) {
            c->copy_constrained_dofs(src, dest);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::report_solution(const isolver_scalar_t *const x) {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();

        char path[2048];
        sprintf(path, "%s/out.raw", impl_->output_dir.c_str());

        printf("report_solution %s\n", path);

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

    class LinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;

        real_t mu{1}, lambda{1};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(mesh->spatial_dim == space->block_size());

            auto ret = std::make_unique<LinearElasticity>(space);

            real_t SFEM_SHEAR_MODULUS = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            ret->mu = SFEM_SHEAR_MODULUS;
            ret->lambda = SFEM_FIRST_LAME_PARAMETER;
            return ret;
        }

        const char *name() const override { return "LinearElasticity"; }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

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

    class Laplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;

        const char *name() const override { return "Laplacian"; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(1 == space->block_size());

            auto ret = std::make_unique<Laplacian>(space);
            return ret;
        }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

        Laplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            laplacian_assemble_hessian((enum ElemType)mesh->element_type,
                                       mesh->nelements,
                                       mesh->nnodes,
                                       mesh->elements,
                                       mesh->points,
                                       space->mesh().node_to_node_rowptr(),
                                       space->mesh().node_to_node_colidx(),
                                       values);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            laplacian_assemble_gradient((enum ElemType)mesh->element_type,
                                        mesh->nelements,
                                        mesh->nnodes,
                                        mesh->elements,
                                        mesh->points,
                                        x,
                                        out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            laplacian_apply((enum ElemType)mesh->element_type,
                            mesh->nelements,
                            mesh->nnodes,
                            mesh->elements,
                            mesh->points,
                            h,
                            out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            laplacian_assemble_value((enum ElemType)mesh->element_type,
                                     mesh->nelements,
                                     mesh->nnodes,
                                     mesh->elements,
                                     mesh->points,
                                     x,
                                     out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
    };

    class CVFEMConvection final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        real_t *vel[3];

        const char *name() const override { return "CVFEMConvection"; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(1 == space->block_size());

            const char *SFEM_VELX = nullptr;
            const char *SFEM_VELY = nullptr;
            const char *SFEM_VELZ = nullptr;

            SFEM_READ_ENV(SFEM_VELX, );
            SFEM_READ_ENV(SFEM_VELY, );
            SFEM_READ_ENV(SFEM_VELZ, );

            if (!SFEM_VELX || !SFEM_VELY || (!SFEM_VELZ && mesh->spatial_dim == 3)) {
                fprintf(stderr,
                        "Missing input velocity SFEM_VELX=%s\n,SFEM_VELY=%s\n,SFEM_VELZ=%s\n",
                        SFEM_VELX,
                        SFEM_VELY,
                        SFEM_VELZ);
                assert(0);
                return nullptr;
            }

            auto ret = std::make_unique<CVFEMConvection>(space);

            ptrdiff_t nlocal, nglobal;
            if (array_create_from_file(mesh->comm,
                                       SFEM_VELX,
                                       SFEM_MPI_REAL_T,
                                       (void **)&ret->vel[0],
                                       &nlocal,
                                       &nglobal) ||
                array_create_from_file(mesh->comm,
                                       SFEM_VELY,
                                       SFEM_MPI_REAL_T,
                                       (void **)&ret->vel[1],
                                       &nlocal,
                                       &nglobal) ||
                array_create_from_file(mesh->comm,
                                       SFEM_VELZ,
                                       SFEM_MPI_REAL_T,
                                       (void **)&ret->vel[2],
                                       &nlocal,
                                       &nglobal)) {
                fprintf(stderr, "Unable to read input velocity\n");
                assert(0);
                return nullptr;
            }

            return ret;
        }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

        CVFEMConvection(const std::shared_ptr<FunctionSpace> &space) : space(space) {
            vel[0] = nullptr;
            vel[1] = nullptr;
            vel[2] = nullptr;
        }

        ~CVFEMConvection() {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // cvfem_convection_assemble_hessian((enum ElemType)mesh->element_type,
            //                            mesh->nelements,
            //                            mesh->nnodes,
            //                            mesh->elements,
            //                            mesh->points,
            //                            space->mesh().node_to_node_rowptr(),
            //                            space->mesh().node_to_node_colidx(),
            //                            values);

            // return ISOLVER_FUNCTION_SUCCESS;

            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // cvfem_convection_assemble_gradient((enum ElemType)mesh->element_type,
            //                             mesh->nelements,
            //                             mesh->nnodes,
            //                             mesh->elements,
            //                             mesh->points,
            //                             x,
            //                             out);

            // return ISOLVER_FUNCTION_SUCCESS;

            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            cvfem_convection_apply((enum ElemType)mesh->element_type,
                                   mesh->nelements,
                                   mesh->nnodes,
                                   mesh->elements,
                                   mesh->points,
                                   vel,
                                   h,
                                   out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // cvfem_convection_assemble_value((enum ElemType)mesh->element_type,
            //                          mesh->nelements,
            //                          mesh->nnodes,
            //                          mesh->elements,
            //                          mesh->points,
            //                          x,
            //                          out);

            // return ISOLVER_FUNCTION_SUCCESS;

            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
    };

    class Factory::Impl {
    public:
        std::map<std::string, FactoryFunction> name_to_create;
    };

    Factory::Factory() : impl_(std::make_unique<Impl>()) {}

    Factory::~Factory() = default;

    Factory &Factory::instance() {
        static Factory instance_;
        
        if(instance_.impl_->name_to_create.empty()) {
            instance_.private_register_op("LinearElasticity", LinearElasticity::create);
            instance_.private_register_op("Laplacian", Laplacian::create);
            instance_.private_register_op("CVFEMConvection", CVFEMConvection::create);
        }

        return instance_;
    }

    void Factory::private_register_op(const std::string &name, FactoryFunction factory_function)
    {
        impl_->name_to_create[name] = factory_function;
    }

    void Factory::register_op(const std::string &name, FactoryFunction factory_function) {
        instance().private_register_op(name, factory_function);
    }

    std::shared_ptr<Op> Factory::create_op(const std::shared_ptr<FunctionSpace> &space,
                                           const char *name) {
        assert(instance().impl_);

        auto &ntc = instance().impl_->name_to_create;
        auto it = ntc.find(name);

        if (it == ntc.end()) {
            std::cerr << "Unable to find op " << name << "\n";
            return nullptr;
        }

        return it->second(space);
    }
}  // namespace sfem
