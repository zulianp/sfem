#include "sfem_Function.hpp"
#include <stddef.h>

#include "matrixio_array.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"
#include "sfem_mesh_write.h"

#include "boundary_condition.h"
#include "boundary_condition_io.h"

#include "dirichlet.h"
#include "neumann.h"

#include <sys/stat.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

// Ops

#include "cvfem_operators.h"
#include "laplacian.h"
#include "linear_elasticity.h"
#include "mass.h"

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

    Mesh::Mesh(int spatial_dim,
               enum ElemType element_type,
               ptrdiff_t nelements,
               idx_t **elements,
               ptrdiff_t nnodes,
               geom_t **points)
        : impl_(std::make_unique<Impl>()) {
        mesh_create_serial(
            &impl_->mesh, spatial_dim, element_type, nelements, elements, nnodes, points);
    }

    int Mesh::spatial_dimension() const { return impl_->mesh.spatial_dim; }
    int Mesh::n_nodes_per_elem() const {
        return elem_num_nodes((enum ElemType)impl_->mesh.element_type);
    }
    ptrdiff_t Mesh::n_nodes() const { return impl_->mesh.nnodes; }
    ptrdiff_t Mesh::n_elements() const { return impl_->mesh.nelements; }

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

    int Mesh::write(const char *path) const {
        if (mesh_write(path, &impl_->mesh)) {
            return ISOLVER_FUNCTION_FAILURE;
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    const geom_t *const Mesh::points(const int coord) const {
        assert(coord < spatial_dimension());
        assert(coord >= 0);
        return impl_->mesh.points[coord];
    }

    const idx_t *const Mesh::idx(const int node_num) const {
        assert(node_num < n_nodes_per_elem());
        assert(node_num >= 0);
        return impl_->mesh.elements[node_num];
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

    ptrdiff_t FunctionSpace::n_dofs() const {
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

    class NeumannConditions::Impl {
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

    int NeumannConditions::n_conditions() const { return impl_->n_neumann_conditions; }
    void *NeumannConditions::impl_conditions() { return (void *)impl_->neumann_conditions; }

    const char *NeumannConditions::name() const { return "NeumannConditions"; }

    NeumannConditions::NeumannConditions(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<NeumannConditions> NeumannConditions::create_from_env(
        const std::shared_ptr<FunctionSpace> &space) {
        //
        auto nc = std::make_unique<NeumannConditions>(space);

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

    NeumannConditions::~NeumannConditions() = default;

    int NeumannConditions::hessian_crs(const isolver_scalar_t *const /*x*/,
                                       const isolver_idx_t *const /*rowptr*/,
                                       const isolver_idx_t *const /*colidx*/,
                                       isolver_scalar_t *const /*values*/) {
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int NeumannConditions::gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) {
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
    int NeumannConditions::apply(const isolver_scalar_t *const /*x*/,
                                 const isolver_scalar_t *const /*h*/,
                                 isolver_scalar_t *const /*out*/) {
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int NeumannConditions::value(const isolver_scalar_t *x, isolver_scalar_t *const out) {
        // TODO
        return ISOLVER_FUNCTION_SUCCESS;
    }

    void NeumannConditions::add_condition(const ptrdiff_t local_size,
                                          const ptrdiff_t global_size,
                                          isolver_idx_t *const idx,
                                          const int component,
                                          const isolver_scalar_t value) {
        impl_->neumann_conditions = (boundary_condition_t *)realloc(
            impl_->neumann_conditions,
            (impl_->n_neumann_conditions + 1) * sizeof(boundary_condition_t));

        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        enum ElemType stype = side_type((enum ElemType)mesh->element_type);
        int nns = elem_num_nodes(stype);

        assert((local_size / nns) * nns == local_size);
        assert((global_size / nns) * nns == global_size);

        boundary_condition_create(&impl_->neumann_conditions[impl_->n_neumann_conditions],
                                  local_size / nns,
                                  global_size / nns,
                                  idx,
                                  component,
                                  value,
                                  nullptr);

        impl_->n_neumann_conditions++;
    }

    void NeumannConditions::add_condition(const ptrdiff_t local_size,
                                          const ptrdiff_t global_size,
                                          isolver_idx_t *const idx,
                                          const int component,
                                          isolver_scalar_t *const values) {
        impl_->neumann_conditions = (boundary_condition_t *)realloc(
            impl_->neumann_conditions,
            (impl_->n_neumann_conditions + 1) * sizeof(boundary_condition_t));

        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        enum ElemType stype = side_type((enum ElemType)mesh->element_type);
        int nns = elem_num_sides(stype);

        boundary_condition_create(&impl_->neumann_conditions[impl_->n_neumann_conditions],
                                  local_size / nns,
                                  global_size / nns,
                                  idx,
                                  component,
                                  0,
                                  values);

        impl_->n_neumann_conditions++;
    }

    int Constraint::apply_zero(isolver_scalar_t *const x) { return apply_value(0, x); }

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

    std::shared_ptr<FunctionSpace> DirichletConditions::space() { return impl_->space; }

    int DirichletConditions::n_conditions() const { return impl_->n_dirichlet_conditions; }
    void *DirichletConditions::impl_conditions() { return (void *)impl_->dirichlet_conditions; }

    DirichletConditions::DirichletConditions(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    DirichletConditions::~DirichletConditions() = default;

    void DirichletConditions::add_condition(const ptrdiff_t local_size,
                                            const ptrdiff_t global_size,
                                            isolver_idx_t *const idx,
                                            const int component,
                                            const isolver_scalar_t value) {
        impl_->dirichlet_conditions = (boundary_condition_t *)realloc(
            impl_->dirichlet_conditions,
            (impl_->n_dirichlet_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(&impl_->dirichlet_conditions[impl_->n_dirichlet_conditions],
                                  local_size,
                                  global_size,
                                  idx,
                                  component,
                                  value,
                                  nullptr);

        impl_->n_dirichlet_conditions++;
    }

    void DirichletConditions::add_condition(const ptrdiff_t local_size,
                                            const ptrdiff_t global_size,
                                            isolver_idx_t *const idx,
                                            const int component,
                                            isolver_scalar_t *const values) {
        impl_->dirichlet_conditions = (boundary_condition_t *)realloc(
            impl_->dirichlet_conditions,
            (impl_->n_dirichlet_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(&impl_->dirichlet_conditions[impl_->n_dirichlet_conditions],
                                  local_size,
                                  global_size,
                                  idx,
                                  component,
                                  0,
                                  values);

        impl_->n_dirichlet_conditions++;
    }

    std::shared_ptr<DirichletConditions> DirichletConditions::create_from_env(
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

    int DirichletConditions::apply(isolver_scalar_t *const x) {
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

    int DirichletConditions::gradient(const isolver_scalar_t *const x, isolver_scalar_t *const g) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_gradient_nodes_to_value_vec(impl_->dirichlet_conditions[i].local_size,
                                                   impl_->dirichlet_conditions[i].idx,
                                                   impl_->space->block_size(),
                                                   impl_->dirichlet_conditions[i].component,
                                                   impl_->dirichlet_conditions[i].value,
                                                   x,
                                                   g);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int DirichletConditions::apply_value(const isolver_scalar_t value, isolver_scalar_t *const x) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_nodes_to_value_vec(impl_->dirichlet_conditions[i].local_size,
                                          impl_->dirichlet_conditions[i].idx,
                                          impl_->space->block_size(),
                                          impl_->dirichlet_conditions[i].component,
                                          value,
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

    class Timings {
    public:
        static double tick() { return MPI_Wtime(); }

        class Scoped {
        public:
            double tick_{0};
            double *value_;
            Scoped(double *value) : value_(value) { tick_ = tick(); }

            ~Scoped() { *value_ += tick() - tick_; }
        };

        double create_crs_graph{0};
        double destroy_crs_graph{0};
        double hessian_crs{0};
        double hessian_diag{0};
        double gradient{0};
        double apply{0};
        double value{0};
        double apply_constraints{0};
        double constraints_gradient{0};
        double apply_zero_constraints{0};
        double copy_constrained_dofs{0};
        double report_solution{0};
        double initial_guess{0};

        void clear() {
            create_crs_graph = 0;
            destroy_crs_graph = 0;
            hessian_crs = 0;
            hessian_diag = 0;
            gradient = 0;
            apply = 0;
            value = 0;
            apply_constraints = 0;
            constraints_gradient = 0;
            apply_zero_constraints = 0;
            copy_constrained_dofs = 0;
            report_solution = 0;
            initial_guess = 0;
        }

        void describe(std::ostream &os) const {
            os << "function,seconds\n";
            os << "create_crs_graph," << create_crs_graph << "\n";
            os << "destroy_crs_graph," << destroy_crs_graph << "\n";
            os << "hessian_crs," << hessian_crs << "\n";
            os << "hessian_diag," << hessian_diag << "\n";
            os << "gradient," << gradient << "\n";
            os << "apply," << apply << "\n";
            os << "value," << value << "\n";
            os << "apply_constraints," << apply_constraints << "\n";
            os << "constraints_gradient," << constraints_gradient << "\n";
            os << "apply_zero_constraints," << apply_zero_constraints << "\n";
            os << "copy_constrained_dofs," << copy_constrained_dofs << "\n";
            os << "report_solution," << report_solution << "\n";
            os << "initial_guess," << initial_guess << "\n";
        }
    };

#define SFEM_FUNCTION_SCOPED_TIMING(acc) Timings::Scoped scoped_(&(acc))

    class Output::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::string output_dir{"."};
        std::string file_format{"%s/%s.raw"};
        std::string time_dependent_file_format{"%s/%s.%09d.raw"};
        size_t export_counter{0};
        logger_t time_logger;
        Impl() { log_init(&time_logger); }
        ~Impl() { log_destroy(&time_logger); }
    };

    Output::Output(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space = space;

        const char *SFEM_OUTPUT_DIR = ".";
        SFEM_READ_ENV(SFEM_OUTPUT_DIR, );
        impl_->output_dir = SFEM_OUTPUT_DIR;
    }

    Output::~Output() = default;

    void Output::clear() { impl_->export_counter = 0; }

    void Output::set_output_dir(const char *path) { impl_->output_dir = path; }

    int Output::write(const char *name, const isolver_scalar_t *const x) {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();

        {
            struct stat st = {0};
            if (stat(impl_->output_dir.c_str(), &st) == -1) {
                mkdir(impl_->output_dir.c_str(), 0700);
            }
        }

        char path[2048];
        sprintf(path, impl_->file_format.c_str(), impl_->output_dir.c_str(), name);
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

    int Output::write_time_step(const char *name,
                                const isolver_scalar_t t,
                                const isolver_scalar_t *const x) {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();

        {
            struct stat st = {0};
            if (stat(impl_->output_dir.c_str(), &st) == -1) {
                mkdir(impl_->output_dir.c_str(), 0700);
            }
        }

        char path[2048];
        sprintf(path,
                impl_->time_dependent_file_format.c_str(),
                impl_->output_dir.c_str(),
                name,
                impl_->export_counter++);
        if (array_write(mesh->comm,
                        path,
                        SFEM_MPI_REAL_T,
                        x,
                        mesh->nnodes * impl_->space->block_size(),
                        mesh->nnodes * impl_->space->block_size())) {
            return ISOLVER_FUNCTION_FAILURE;
        }

        if (log_is_empty(&impl_->time_logger)) {
            sprintf(path, "%s/time.txt", impl_->output_dir.c_str());
            log_create_file(&impl_->time_logger, path, "w");
        }

        log_write_double(&impl_->time_logger, t);
        return ISOLVER_FUNCTION_SUCCESS;
    }

    class Function::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::vector<std::shared_ptr<Op>> ops;
        std::vector<std::shared_ptr<Constraint>> constraints;
        Timings timings;

        std::shared_ptr<Output> output;
        bool handle_constraints{true};
    };

    Function::Function(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
        impl_->output = std::make_shared<Output>(space);
    }

    Function::~Function() {
        std::ofstream os;
        os.open("perf.csv");
        if (!os.good()) return;

        impl_->timings.describe(os);
        os.close();
    }

    void Function::add_operator(const std::shared_ptr<Op> &op) { impl_->ops.push_back(op); }
    void Function::add_constraint(const std::shared_ptr<Constraint> &c) {
        impl_->constraints.push_back(c);
    }

    void Function::add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c) {
        add_constraint(c);
    }

    int Function::create_crs_graph(ptrdiff_t *nlocal,
                                   ptrdiff_t *nglobal,
                                   ptrdiff_t *nnz,
                                   isolver_idx_t **rowptr,
                                   isolver_idx_t **colidx) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.create_crs_graph);

        return impl_->space->create_crs_graph(nlocal, nglobal, nnz, rowptr, colidx);
    }

    int Function::destroy_crs_graph(isolver_idx_t *rowptr, isolver_idx_t *colidx) {
        return impl_->space->destroy_crs_graph(rowptr, colidx);
    }

    int Function::hessian_crs(const isolver_scalar_t *const x,
                              const isolver_idx_t *const rowptr,
                              const isolver_idx_t *const colidx,
                              isolver_scalar_t *const values) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.hessian_crs);

        for (auto &op : impl_->ops) {
            if (op->hessian_crs(x, rowptr, colidx, values) != ISOLVER_FUNCTION_SUCCESS) {
                std::cerr << "Failed hessian_crs in op: " << op->name() << "\n";
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->hessian_crs(x, rowptr, colidx, values);
            }
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::hessian_diag(const isolver_scalar_t *const x, isolver_scalar_t *const values) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.hessian_diag);

        for (auto &op : impl_->ops) {
            if (op->hessian_diag(x, values) != ISOLVER_FUNCTION_SUCCESS) {
                std::cerr << "Failed hessian_diag in op: " << op->name() << "\n";
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->apply_value(1, values);
            }
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.gradient);

        for (auto &op : impl_->ops) {
            if (op->gradient(x, out) != ISOLVER_FUNCTION_SUCCESS) {
                std::cerr << "Failed gradient in op: " << op->name() << "\n";
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            constraints_gradient(x, out);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::apply(const isolver_scalar_t *const x,
                        const isolver_scalar_t *const h,
                        isolver_scalar_t *const out) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.apply);

        for (auto &op : impl_->ops) {
            if (op->apply(x, h, out) != ISOLVER_FUNCTION_SUCCESS) {
                std::cerr << "Failed apply in op: " << op->name() << "\n";
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            copy_constrained_dofs(h, out);
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::value(const isolver_scalar_t *x, isolver_scalar_t *const out) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.value);

        for (auto &op : impl_->ops) {
            if (op->value(x, out) != ISOLVER_FUNCTION_SUCCESS) {
                std::cerr << "Failed value in op: " << op->name() << "\n";
                return ISOLVER_FUNCTION_FAILURE;
            }
        }

        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::apply_constraints(isolver_scalar_t *const x) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.apply_constraints);

        for (auto &c : impl_->constraints) {
            c->apply(x);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::constraints_gradient(const isolver_scalar_t *const x, isolver_scalar_t *const g) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.constraints_gradient);

        for (auto &c : impl_->constraints) {
            c->gradient(x, g);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::apply_zero_constraints(isolver_scalar_t *const x) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.apply_zero_constraints);

        for (auto &c : impl_->constraints) {
            c->apply_zero(x);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::copy_constrained_dofs(const isolver_scalar_t *const src,
                                        isolver_scalar_t *const dest) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.copy_constrained_dofs);

        for (auto &c : impl_->constraints) {
            c->copy_constrained_dofs(src, dest);
        }
        return ISOLVER_FUNCTION_SUCCESS;
    }

    int Function::report_solution(const isolver_scalar_t *const x) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.report_solution);

        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        return impl_->output->write("out", x);
    }

    int Function::initial_guess(isolver_scalar_t *const x) { return ISOLVER_FUNCTION_SUCCESS; }

    int Function::set_output_dir(const char *path) {
        impl_->output->set_output_dir(path);
        return ISOLVER_FUNCTION_SUCCESS;
    }

    std::shared_ptr<Output> Function::output() { return impl_->output; }

    class LinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType element_type { INVALID };

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
            ret->element_type = (enum ElemType)mesh->element_type;
            return ret;
        }

        std::shared_ptr<Op> lor_op() override {
            auto ret = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_type_variant(element_type);
            ret->mu = mu;
            ret->lambda = lambda;
            return ret;
        }

        std::shared_ptr<Op> coarsen_op() override {
            auto ret = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->mu = mu;
            ret->lambda = lambda;
            return ret;
        }

        const char *name() const override { return "LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

        LinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_assemble_hessian_aos(element_type,
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

        int hessian_diag(const isolver_scalar_t *const, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_assemble_diag_aos(element_type,
                                                mesh->nelements,
                                                mesh->nnodes,
                                                mesh->elements,
                                                mesh->points,
                                                this->mu,
                                                this->lambda,
                                                out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_assemble_gradient_aos(element_type,
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

            linear_elasticity_apply_aos(element_type,
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

            linear_elasticity_assemble_value_aos(element_type,
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
        enum ElemType element_type { INVALID };

        const char *name() const override { return "Laplacian"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(1 == space->block_size());

            auto ret = std::make_unique<Laplacian>(space);
            ret->element_type = (enum ElemType)mesh->element_type;
            return ret;
        }

        std::shared_ptr<Op> lor_op() override {
            auto ret = std::make_shared<Laplacian>(space);
            ret->element_type = macro_type_variant(element_type);
            return ret;
        }

        std::shared_ptr<Op> coarsen_op() override {
            auto ret = std::make_shared<Laplacian>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

        Laplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            laplacian_assemble_hessian(element_type,
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

            laplacian_assemble_gradient(
                element_type, mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, x, out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            laplacian_apply(
                element_type, mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, h, out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            laplacian_assemble_value(
                element_type, mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, x, out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
    };

    class Mass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType element_type { INVALID };

        const char *name() const override { return "Mass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            assert(1 == space->block_size());

            auto ret = std::make_unique<Mass>(space);
            ret->element_type = (enum ElemType)mesh->element_type;
            return ret;
        }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

        Mass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assemble_mass(element_type,
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

            apply_mass(
                element_type, mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, x, out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            apply_mass(
                element_type, mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, h, out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            // auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // mass_assemble_value((enum ElemType)mesh->element_type,
            //                     mesh->nelements,
            //                     mesh->nnodes,
            //                     mesh->elements,
            //                     mesh->points,
            //                     x,
            //                     out);

            // return ISOLVER_FUNCTION_SUCCESS;

            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
    };

    class LumpedMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType element_type { INVALID };

        const char *name() const override { return "LumpedMass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            auto ret = std::make_unique<LumpedMass>(space);
            ret->element_type = (enum ElemType)mesh->element_type;
            return ret;
        }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

        LumpedMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_diag(const isolver_scalar_t *const /*x*/,
                         isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            if (space->block_size() == 1) {
                assemble_lumped_mass(element_type,
                                     mesh->nelements,
                                     mesh->nnodes,
                                     mesh->elements,
                                     mesh->points,
                                     values);
            } else {
                real_t *temp = (real_t *)calloc(mesh->nnodes, sizeof(real_t));
                assemble_lumped_mass(element_type,
                                     mesh->nelements,
                                     mesh->nnodes,
                                     mesh->elements,
                                     mesh->points,
                                     temp);

                int bs = space->block_size();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {
                    for (int b = 0; b < bs; b++) {
                        values[i * bs + b] += temp[i];
                    }
                }

                free(temp);
            }

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
    };

    class CVFEMMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType element_type { INVALID };

        const char *name() const override { return "CVFEMMass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            assert(1 == space->block_size());

            auto ret = std::make_unique<CVFEMMass>(space);
            ret->element_type = (enum ElemType)mesh->element_type;
            return ret;
        }

        int initialize() override { return ISOLVER_FUNCTION_SUCCESS; }

        CVFEMMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_diag(const isolver_scalar_t *const /*x*/,
                         isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            cvfem_cv_volumes(
                element_type, mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, values);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
    };

    class CVFEMUpwindConvection final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        real_t *vel[3];
        enum ElemType element_type { INVALID };

        const char *name() const override { return "CVFEMUpwindConvection"; }
        inline bool is_linear() const override { return true; }

        void set_field(const char * /* name  = velocity */,
                       const int component,
                       isolver_scalar_t *v) override {
            if (vel[component]) {
                free(vel[component]);
            }

            vel[component] = v;
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(1 == space->block_size());

            auto ret = std::make_unique<CVFEMUpwindConvection>(space);
            ret->vel[0] = nullptr;
            ret->vel[1] = nullptr;
            ret->vel[2] = nullptr;

            const char *SFEM_VELX = nullptr;
            const char *SFEM_VELY = nullptr;
            const char *SFEM_VELZ = nullptr;

            SFEM_READ_ENV(SFEM_VELX, );
            SFEM_READ_ENV(SFEM_VELY, );
            SFEM_READ_ENV(SFEM_VELZ, );

            if (!SFEM_VELX || !SFEM_VELY || (!SFEM_VELZ && mesh->spatial_dim == 3)) {
                // fprintf(stderr,
                //         "No input velocity in env: SFEM_VELX=%s\n,SFEM_VELY=%s\n,SFEM_VELZ=%s\n",
                //         SFEM_VELX,
                //         SFEM_VELY,
                //         SFEM_VELZ);
                ret->element_type = (enum ElemType)mesh->element_type;
                return ret;
            }

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

        CVFEMUpwindConvection(const std::shared_ptr<FunctionSpace> &space) : space(space) {
            vel[0] = nullptr;
            vel[1] = nullptr;
            vel[2] = nullptr;
        }

        ~CVFEMUpwindConvection() {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // cvfem_convection_assemble_hessian(element_type,
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

            cvfem_convection_apply(element_type,
                                   mesh->nelements,
                                   mesh->nnodes,
                                   mesh->elements,
                                   mesh->points,
                                   vel,
                                   x,
                                   out);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            cvfem_convection_apply(element_type,
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

            // cvfem_convection_assemble_value(element_type,
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

        if (instance_.impl_->name_to_create.empty()) {
            instance_.private_register_op("LinearElasticity", LinearElasticity::create);
            instance_.private_register_op("Laplacian", Laplacian::create);
            instance_.private_register_op("CVFEMUpwindConvection", CVFEMUpwindConvection::create);
            instance_.private_register_op("Mass", Mass::create);
            instance_.private_register_op("CVFEMMass", CVFEMMass::create);
            instance_.private_register_op("LumpedMass", LumpedMass::create);
        }

        return instance_;
    }

    void Factory::private_register_op(const std::string &name, FactoryFunction factory_function) {
        impl_->name_to_create[name] = factory_function;
    }

    void Factory::register_op(const std::string &name, FactoryFunction factory_function) {
        instance().private_register_op(name, factory_function);
    }

    std::shared_ptr<Op> Factory::create_op_gpu(const std::shared_ptr<FunctionSpace> &space,
                                               const char *name) {
        return Factory::create_op(space, d_op_str(name).c_str());
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

    std::string d_op_str(const std::string &name) { return "gpu:" + name; }

}  // namespace sfem
