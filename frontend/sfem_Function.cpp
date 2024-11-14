#include "sfem_Function.hpp"
#include <glob.h>
#include <stddef.h>

#include "matrixio_array.h"
#include "utils.h"

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
// #include <sys/wait.h>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

// Ops

#include "cvfem_operators.h"
#include "laplacian.h"
#include "linear_elasticity.h"
#include "mass.h"
#include "neohookean_ogden.h"
#include "proteus_hex8_laplacian.h"
#include "proteus_hex8_linear_elasticity.h"

// Mesh
#include "proteus_hex8.h"
#include "sfem_hex8_mesh_graph.h"

// Multigrid
#include "sfem_prolongation_restriction.h"

namespace sfem {

    class CRSGraph::Impl {
    public:
        std::shared_ptr<Buffer<count_t>> rowptr;
        std::shared_ptr<Buffer<idx_t>> colidx;
        ~Impl() {}
    };

    CRSGraph::CRSGraph(const std::shared_ptr<Buffer<count_t>> &rowptr,
                       const std::shared_ptr<Buffer<idx_t>> &colidx)
        : impl_(std::make_unique<Impl>()) {
        impl_->rowptr = rowptr;
        impl_->colidx = colidx;
    }

    void CRSGraph::print(std::ostream &os) const {
        auto rowptr = this->rowptr()->data();
        auto colidx = this->colidx()->data();
        const ptrdiff_t nnodes = this->n_nodes();
        const ptrdiff_t nnz = this->nnz();

        os << "CRSGraph (" << nnodes << " nodes, " << nnz << " nnz)\n";
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            os << i << " (" << (rowptr[i + 1] - rowptr[i]) << "): ";
            for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
                assert(j < nnz);
                os << colidx[j] << " ";
            }
            os << "\n";
        }
    }

    CRSGraph::CRSGraph() : impl_(std::make_unique<Impl>()) {}
    CRSGraph::~CRSGraph() = default;

    ptrdiff_t CRSGraph::n_nodes() const { return rowptr()->size() - 1; }
    ptrdiff_t CRSGraph::nnz() const { return colidx()->size(); }

    std::shared_ptr<Buffer<count_t>> CRSGraph::rowptr() const { return impl_->rowptr; }
    std::shared_ptr<Buffer<idx_t>> CRSGraph::colidx() const { return impl_->colidx; }

    std::shared_ptr<CRSGraph> CRSGraph::block_to_scalar(const int block_size) {
        auto rowptr = h_buffer<count_t>(this->n_nodes() * block_size + 1);
        auto colidx = h_buffer<idx_t>(this->nnz() * block_size * block_size);

        crs_graph_block_to_scalar(this->n_nodes(),
                                  block_size,
                                  this->rowptr()->data(),
                                  this->colidx()->data(),
                                  rowptr->data(),
                                  colidx->data());

        return std::make_shared<CRSGraph>(rowptr, colidx);
    }

    class Mesh::Impl {
    public:
        MPI_Comm comm;
        mesh_t mesh;

        std::shared_ptr<CRSGraph> crs_graph;
        std::shared_ptr<CRSGraph> crs_graph_upper_triangular;

        ~Impl() { mesh_destroy(&mesh); }
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
            return SFEM_FAILURE;
        }

        int SFEM_USE_MACRO = 0;
        SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

        if (SFEM_USE_MACRO) {
            impl_->mesh.element_type = macro_type_variant((enum ElemType)impl_->mesh.element_type);
        }

        return SFEM_SUCCESS;
    }

    int Mesh::write(const char *path) const {
        if (mesh_write(path, &impl_->mesh)) {
            return SFEM_FAILURE;
        }

        return SFEM_SUCCESS;
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

    std::shared_ptr<CRSGraph> Mesh::node_to_node_graph() {
        initialize_node_to_node_graph();
        return impl_->crs_graph;
    }

    std::shared_ptr<CRSGraph> Mesh::create_node_to_node_graph(const enum ElemType element_type) {
        auto mesh = &impl_->mesh;
        if (mesh->element_type == element_type) {
            return node_to_node_graph();
        }

        const ptrdiff_t n_nodes = max_node_id(element_type, mesh->nelements, mesh->elements) + 1;

        count_t *rowptr{nullptr};
        idx_t *colidx{nullptr};
        build_crs_graph_for_elem_type(
                element_type, mesh->nelements, n_nodes, mesh->elements, &rowptr, &colidx);

        auto crs_graph = std::make_shared<CRSGraph>(
                Buffer<count_t>::own(n_nodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                Buffer<idx_t>::own(rowptr[n_nodes], colidx, free, MEMORY_SPACE_HOST));

        return crs_graph;
    }

    int Mesh::initialize_node_to_node_graph() {
        if (impl_->crs_graph) {
            return SFEM_SUCCESS;
        }

        impl_->crs_graph = std::make_shared<CRSGraph>();

        auto mesh = &impl_->mesh;

        count_t *rowptr{nullptr};
        idx_t *colidx{nullptr};
        build_crs_graph_for_elem_type(mesh->element_type,
                                      mesh->nelements,
                                      mesh->nnodes,
                                      mesh->elements,
                                      &rowptr,
                                      &colidx);

        impl_->crs_graph = std::make_shared<CRSGraph>(
                Buffer<count_t>::own(mesh->nnodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                Buffer<idx_t>::own(rowptr[mesh->nnodes], colidx, free, MEMORY_SPACE_HOST));

        return SFEM_SUCCESS;
    }

    std::shared_ptr<CRSGraph> Mesh::node_to_node_graph_upper_triangular() {
        if (!impl_->crs_graph_upper_triangular) {
            auto mesh = &impl_->mesh;

            count_t *rowptr{nullptr};
            idx_t *colidx{nullptr};
            build_crs_graph_upper_triangular_from_element(
                    mesh->nelements,
                    mesh->nnodes,
                    elem_num_nodes((enum ElemType)mesh->element_type),
                    mesh->elements,
                    &rowptr,
                    &colidx);

            impl_->crs_graph_upper_triangular = std::make_shared<CRSGraph>(
                    Buffer<count_t>::own(mesh->nnodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                    Buffer<idx_t>::own(rowptr[mesh->nnodes], colidx, free, MEMORY_SPACE_HOST));

            // impl_->crs_graph_upper_triangular->print(std::cout);
        }

        return impl_->crs_graph_upper_triangular;
    }

    int Mesh::convert_to_macro_element_mesh() {
        impl_->mesh.element_type = macro_type_variant((enum ElemType)impl_->mesh.element_type);
        return SFEM_SUCCESS;
    }

    void *Mesh::impl_mesh() { return (void *)&impl_->mesh; }

    std::shared_ptr<Buffer<count_t>> Mesh::node_to_node_rowptr() const {
        return impl_->crs_graph->rowptr();
    }
    std::shared_ptr<Buffer<idx_t>> Mesh::node_to_node_colidx() const {
        return impl_->crs_graph->colidx();
    }

    class SemiStructuredMesh::Impl {
    public:
        std::shared_ptr<Mesh> macro_mesh;
        int level;

        std::shared_ptr<Buffer<idx_t *>> elements;
        ptrdiff_t n_unique_nodes{-1}, interior_start{-1};
        std::shared_ptr<CRSGraph> node_to_node_graph;

        void init(const std::shared_ptr<Mesh> macro_mesh, const int level) {
            this->macro_mesh = macro_mesh;
            this->level = level;

            const int nxe = proteus_hex8_nxe(level);
            auto elements = (idx_t **)malloc(nxe * sizeof(idx_t *));
            for (int d = 0; d < nxe; d++) {
                elements[d] = (idx_t *)malloc(macro_mesh->n_elements() * sizeof(idx_t));
            }

#ifndef NDEBUG
            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < macro_mesh->n_elements(); i++) {
                    elements[d][i] = -1;
                }
            }
#endif

            proteus_hex8_create_full_idx(level,
                                         (mesh_t *)macro_mesh->impl_mesh(),
                                         elements,
                                         &this->n_unique_nodes,
                                         &this->interior_start);

            this->elements = std::make_shared<Buffer<idx_t *>>(
                    nxe,
                    macro_mesh->n_elements(),
                    elements,
                    [](int n, void **data) {
                        for (int i = 0; i < n; i++) {
                            free(data[i]);
                        }

                        free(data);
                    },
                    MEMORY_SPACE_HOST);
        }

        Impl() {}
        ~Impl() {}
    };

    std::shared_ptr<CRSGraph> SemiStructuredMesh::node_to_node_graph() {
        // printf("SemiStructuredMesh::node_to_node_graph\n");
        if (impl_->node_to_node_graph) {
            return impl_->node_to_node_graph;
        }

        count_t *rowptr{nullptr};
        idx_t *colidx{nullptr};

        proteus_hex8_crs_graph(impl_->level,
                               this->n_elements(),
                               this->n_nodes(),
                               this->element_data(),
                               &rowptr,
                               &colidx);

        impl_->node_to_node_graph = std::make_shared<CRSGraph>(
                Buffer<count_t>::own(this->n_nodes() + 1, rowptr, free, MEMORY_SPACE_HOST),
                Buffer<idx_t>::own(rowptr[this->n_nodes()], colidx, free, MEMORY_SPACE_HOST));

        return impl_->node_to_node_graph;
    }

    int SemiStructuredMesh::n_nodes_per_element() const { return proteus_hex8_nxe(impl_->level); }

    idx_t **SemiStructuredMesh::element_data() { return impl_->elements->data(); }
    geom_t **SemiStructuredMesh::point_data() {
        return ((mesh_t *)(impl_->macro_mesh->impl_mesh()))->points;
    }
    ptrdiff_t SemiStructuredMesh::interior_start() const { return impl_->interior_start; }

    SemiStructuredMesh::SemiStructuredMesh(const std::shared_ptr<Mesh> macro_mesh, const int level)
        : impl_(std::make_unique<Impl>()) {
        impl_->init(macro_mesh, level);
    }

    SemiStructuredMesh::~SemiStructuredMesh() {}

    ptrdiff_t SemiStructuredMesh::n_nodes() const { return impl_->n_unique_nodes; }
    int SemiStructuredMesh::level() const { return impl_->level; }
    ptrdiff_t SemiStructuredMesh::n_elements() const { return impl_->macro_mesh->n_elements(); }

    class FunctionSpace::Impl {
    public:
        std::shared_ptr<Mesh> mesh;
        int block_size{1};
        enum ElemType element_type { INVALID };

        // Number of nodes of function-space (TODO)
        ptrdiff_t nlocal{0};
        ptrdiff_t nglobal{0};

        // CRS graph
        std::shared_ptr<CRSGraph> node_to_node_graph;
        std::shared_ptr<CRSGraph> dof_to_dof_graph;
        std::shared_ptr<sfem::Buffer<idx_t>> device_elements;

        // Data-structures for semistructured mesh
        std::shared_ptr<SemiStructuredMesh> semi_structured_mesh;

        ~Impl() {}

        int initialize_dof_to_dof_graph(const int block_size) {
            if (semi_structured_mesh) {
                // printf("SemiStructuredMesh::node_to_node_graph (in FunctionSpace)\n");
                if (!node_to_node_graph) {
                    node_to_node_graph = semi_structured_mesh->node_to_node_graph();
                }
                // FIXME
                dof_to_dof_graph = node_to_node_graph;
                return SFEM_SUCCESS;
            }

            // This is for nodal discretizations (CG)
            if (!node_to_node_graph) {
                node_to_node_graph = mesh->create_node_to_node_graph(element_type);
            }

            if (block_size == 1) {
                dof_to_dof_graph = node_to_node_graph;
            } else {
                if (!dof_to_dof_graph) {
                    dof_to_dof_graph = node_to_node_graph->block_to_scalar(block_size);
                }
            }

            return SFEM_SUCCESS;
        }
    };

    void FunctionSpace::set_device_elements(const std::shared_ptr<sfem::Buffer<idx_t>> &elems) {
        impl_->device_elements = elems;
    }

    std::shared_ptr<sfem::Buffer<idx_t>> FunctionSpace::device_elements() {
        return impl_->device_elements;
    }

    std::shared_ptr<CRSGraph> FunctionSpace::dof_to_dof_graph() {
        impl_->initialize_dof_to_dof_graph(this->block_size());
        return impl_->dof_to_dof_graph;
    }

    std::shared_ptr<CRSGraph> FunctionSpace::node_to_node_graph() {
        impl_->initialize_dof_to_dof_graph(this->block_size());

        return impl_->node_to_node_graph;
    }

    enum ElemType FunctionSpace::element_type() const {
        assert(impl_->element_type != INVALID);
        return impl_->element_type;
    }

    std::shared_ptr<FunctionSpace> FunctionSpace::derefine() const {
        // if (has_semi_structured_mesh()) {

        // }

        {
            // FIXME the number of nodes in mesh does not change, will lead to bugs
            return std::make_shared<FunctionSpace>(
                    impl_->mesh, impl_->block_size, macro_base_elem(impl_->element_type));
        }
    }

    FunctionSpace::FunctionSpace(const std::shared_ptr<Mesh> &mesh,
                                 const int block_size,
                                 const enum ElemType element_type)
        : impl_(std::make_unique<Impl>()) {
        impl_->mesh = mesh;
        impl_->block_size = block_size;
        assert(block_size > 0);

        if (element_type == INVALID) {
            impl_->element_type = (enum ElemType)mesh->impl_->mesh.element_type;
        } else {
            impl_->element_type = element_type;
        }

        auto c_mesh = &mesh->impl_->mesh;
        if (impl_->element_type == c_mesh->element_type) {
            impl_->nlocal = c_mesh->nnodes * block_size;
            impl_->nglobal = c_mesh->nnodes * block_size;
        } else {
            // FIXME in parallel it will not work
            impl_->nlocal =
                    (max_node_id(impl_->element_type, c_mesh->nelements, c_mesh->elements) + 1) *
                    block_size;
            impl_->nglobal = impl_->nlocal;

            // MPI_CATCH_ERROR(
            //     MPI_Allreduce(MPI_IN_PLACE, &impl_->nglobal, 1, MPI_LONG, MPI_SUM,
            //     c_mesh->comm));
        }
    }

    int FunctionSpace::promote_to_semi_structured(const int level) {
        if (impl_->element_type == HEX8) {
            impl_->semi_structured_mesh = std::make_shared<SemiStructuredMesh>(impl_->mesh, level);
            impl_->element_type = PROTEUS_HEX8;
            impl_->nlocal = impl_->semi_structured_mesh->n_nodes() * impl_->block_size;
            impl_->nglobal = impl_->nlocal;
            return SFEM_SUCCESS;
        }

        return SFEM_FAILURE;
    }

    FunctionSpace::~FunctionSpace() = default;

    bool FunctionSpace::has_semi_structured_mesh() const {
        return static_cast<bool>(impl_->semi_structured_mesh);
    }

    Mesh &FunctionSpace::mesh() { return *impl_->mesh; }

    std::shared_ptr<Mesh> FunctionSpace::mesh_ptr() const { return impl_->mesh; }

    SemiStructuredMesh &FunctionSpace::semi_structured_mesh() {
        return *impl_->semi_structured_mesh;
    }

    int FunctionSpace::block_size() const { return impl_->block_size; }

    ptrdiff_t FunctionSpace::n_dofs() const { return impl_->nlocal; }

    std::shared_ptr<FunctionSpace> FunctionSpace::lor() const {
        return std::make_shared<FunctionSpace>(
                impl_->mesh, impl_->block_size, macro_type_variant(impl_->element_type));
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

    int NeumannConditions::hessian_crs(const real_t *const /*x*/,
                                       const count_t *const /*rowptr*/,
                                       const idx_t *const /*colidx*/,
                                       real_t *const /*values*/) {
        return SFEM_SUCCESS;
    }

    int NeumannConditions::gradient(const real_t *const x, real_t *const out) {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();

        for (int i = 0; i < impl_->n_neumann_conditions; i++) {
            surface_forcing_function_vec(side_type((enum ElemType)impl_->space->element_type()),
                                         impl_->neumann_conditions[i].local_size,
                                         impl_->neumann_conditions[i].idx,
                                         mesh->points,
                                         -  // Use negative sign since we are on LHS
                                         impl_->neumann_conditions[i].value,
                                         impl_->space->block_size(),
                                         impl_->neumann_conditions[i].component,
                                         out);
        }

        return SFEM_SUCCESS;
    }
    int NeumannConditions::apply(const real_t *const /*x*/,
                                 const real_t *const /*h*/,
                                 real_t *const /*out*/) {
        return SFEM_SUCCESS;
    }

    int NeumannConditions::value(const real_t *x, real_t *const out) {
        // TODO
        return SFEM_SUCCESS;
    }

    void NeumannConditions::add_condition(const ptrdiff_t local_size,
                                          const ptrdiff_t global_size,
                                          idx_t *const idx,
                                          const int component,
                                          const real_t value) {
        impl_->neumann_conditions = (boundary_condition_t *)realloc(
                impl_->neumann_conditions,
                (impl_->n_neumann_conditions + 1) * sizeof(boundary_condition_t));

        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        enum ElemType stype = side_type((enum ElemType)impl_->space->element_type());
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
                                          idx_t *const idx,
                                          const int component,
                                          real_t *const values) {
        impl_->neumann_conditions = (boundary_condition_t *)realloc(
                impl_->neumann_conditions,
                (impl_->n_neumann_conditions + 1) * sizeof(boundary_condition_t));

        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        enum ElemType stype = side_type((enum ElemType)impl_->space->element_type());
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

    int Constraint::apply_zero(real_t *const x) { return apply_value(0, x); }

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

    std::shared_ptr<Constraint> DirichletConditions::derefine(
            const std::shared_ptr<FunctionSpace> &coarse_space,
            const bool as_zero) const {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        auto et = (enum ElemType)impl_->space->element_type();

        const ptrdiff_t max_coarse_idx =
                max_node_id(coarse_space->element_type(), mesh->nelements, mesh->elements);

        auto coarse = std::make_shared<DirichletConditions>(coarse_space);

        coarse->impl_->dirichlet_conditions = (boundary_condition_t *)malloc(
                impl_->n_dirichlet_conditions * sizeof(boundary_condition_t));
        coarse->impl_->n_dirichlet_conditions = 0;

        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            ptrdiff_t coarse_local_size = 0;
            idx_t *coarse_indices = nullptr;
            real_t *coarse_values = nullptr;
            hierarchical_create_coarse_indices(max_coarse_idx,
                                               impl_->dirichlet_conditions[i].local_size,
                                               impl_->dirichlet_conditions[i].idx,
                                               &coarse_local_size,
                                               &coarse_indices);

            if (!as_zero && impl_->dirichlet_conditions[i].values) {
                coarse_values = (real_t *)malloc(coarse_local_size * sizeof(real_t));

                hierarchical_collect_coarse_values(max_coarse_idx,
                                                   impl_->dirichlet_conditions[i].local_size,
                                                   impl_->dirichlet_conditions[i].idx,
                                                   impl_->dirichlet_conditions[i].values,
                                                   coarse_values);
            }

            long coarse_global_size = coarse_local_size;

            // MPI_CATCH_ERROR(
            // MPI_Allreduce(MPI_IN_PLACE, &coarse_global_size, 1, MPI_LONG, MPI_SUM, mesh->comm));

            if (as_zero) {
                boundary_condition_create(
                        &coarse->impl_
                                 ->dirichlet_conditions[coarse->impl_->n_dirichlet_conditions++],
                        coarse_local_size,
                        coarse_global_size,
                        coarse_indices,
                        impl_->dirichlet_conditions[i].component,
                        0,
                        nullptr);

            } else {
                boundary_condition_create(
                        &coarse->impl_
                                 ->dirichlet_conditions[coarse->impl_->n_dirichlet_conditions++],
                        coarse_local_size,
                        coarse_global_size,
                        coarse_indices,
                        impl_->dirichlet_conditions[i].component,
                        impl_->dirichlet_conditions[i].value,
                        coarse_values);
            }
        }

        return coarse;
    }

    std::shared_ptr<Constraint> DirichletConditions::lor() const {
        assert(false);
        return nullptr;
    }

    DirichletConditions::~DirichletConditions() = default;

    void DirichletConditions::add_condition(const ptrdiff_t local_size,
                                            const ptrdiff_t global_size,
                                            idx_t *const idx,
                                            const int component,
                                            const real_t value) {
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
                                            idx_t *const idx,
                                            const int component,
                                            real_t *const values) {
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

    int DirichletConditions::apply(real_t *const x) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_nodes_to_value_vec(impl_->dirichlet_conditions[i].local_size,
                                          impl_->dirichlet_conditions[i].idx,
                                          impl_->space->block_size(),
                                          impl_->dirichlet_conditions[i].component,
                                          impl_->dirichlet_conditions[i].value,
                                          x);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::gradient(const real_t *const x, real_t *const g) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_gradient_nodes_to_value_vec(impl_->dirichlet_conditions[i].local_size,
                                                   impl_->dirichlet_conditions[i].idx,
                                                   impl_->space->block_size(),
                                                   impl_->dirichlet_conditions[i].component,
                                                   impl_->dirichlet_conditions[i].value,
                                                   x,
                                                   g);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::apply_value(const real_t value, real_t *const x) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_nodes_to_value_vec(impl_->dirichlet_conditions[i].local_size,
                                          impl_->dirichlet_conditions[i].idx,
                                          impl_->space->block_size(),
                                          impl_->dirichlet_conditions[i].component,
                                          value,
                                          x);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            constraint_nodes_copy_vec(impl_->dirichlet_conditions[i].local_size,
                                      impl_->dirichlet_conditions[i].idx,
                                      impl_->space->block_size(),
                                      impl_->dirichlet_conditions[i].component,
                                      src,
                                      dest);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::hessian_crs(const real_t *const x,
                                         const count_t *const rowptr,
                                         const idx_t *const colidx,
                                         real_t *const values) {
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

        return SFEM_SUCCESS;
    }

    int DirichletConditions::hessian_bsr(const real_t *const x,
                                         const count_t *const rowptr,
                                         const idx_t *const colidx,
                                         real_t *const values) {
        for (int i = 0; i < impl_->n_dirichlet_conditions; i++) {
            bsr_constraint_nodes_to_identity_vec(impl_->dirichlet_conditions[i].local_size,
                                                 impl_->dirichlet_conditions[i].idx,
                                                 impl_->space->block_size(),
                                                 impl_->dirichlet_conditions[i].component,
                                                 1,
                                                 rowptr,
                                                 colidx,
                                                 values);
        }

        return SFEM_SUCCESS;
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
        double hessian_bsr{0};
        double hessian_bcrs_sym{0};
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
            hessian_bsr = 0;
            hessian_bcrs_sym = 0;
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
            os << "hessian_bsr," << hessian_bsr << "\n";
            os << "hessian_bcrs_sym," << hessian_bcrs_sym << "\n";
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

    int Output::write(const char *name, const real_t *const x) {
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
                        impl_->space->n_dofs(),
                        impl_->space->n_dofs())) {
            return SFEM_FAILURE;
        }

        return SFEM_SUCCESS;
    }

    int Output::write_time_step(const char *name, const real_t t, const real_t *const x) {
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
                        impl_->space->n_dofs(),
                        impl_->space->n_dofs())) {
            return SFEM_FAILURE;
        }

        if (log_is_empty(&impl_->time_logger)) {
            sprintf(path, "%s/time.txt", impl_->output_dir.c_str());
            log_create_file(&impl_->time_logger, path, "w");
        }

        log_write_double(&impl_->time_logger, t);
        return SFEM_SUCCESS;
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

    ExecutionSpace Function::execution_space() const {
        ExecutionSpace ret = EXECUTION_SPACE_INVALID;

        for (auto op : impl_->ops) {
            assert(ret == EXECUTION_SPACE_INVALID || ret == op->execution_space());
            ret = op->execution_space();
        }

        return ret;
    }

    Function::Function(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
        impl_->output = std::make_shared<Output>(space);
    }

    std::shared_ptr<FunctionSpace> Function::space() { return impl_->space; }

    Function::~Function() {
        std::ofstream os;
        os.open("perf_" + std::to_string(space()->n_dofs()) + ".csv");
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

    std::shared_ptr<CRSGraph> Function::crs_graph() const {
        return impl_->space->dof_to_dof_graph();
    }

    int Function::hessian_crs(const real_t *const x,
                              const count_t *const rowptr,
                              const idx_t *const colidx,
                              real_t *const values) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.hessian_crs);

        for (auto &op : impl_->ops) {
            if (op->hessian_crs(x, rowptr, colidx, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_crs in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->hessian_crs(x, rowptr, colidx, values);
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::hessian_bsr(const real_t *const x,
                              const count_t *const rowptr,
                              const idx_t *const colidx,
                              real_t *const values) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.hessian_bsr);

        for (auto &op : impl_->ops) {
            if (op->hessian_bsr(x, rowptr, colidx, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_bsr in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->hessian_bsr(x, rowptr, colidx, values);
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::hessian_bcrs_sym(const real_t *const x,
                                   const count_t *const rowptr,
                                   const idx_t *const colidx,
                                   const ptrdiff_t block_stride,
                                   real_t **const diag_values,
                                   real_t **const off_diag_values) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.hessian_bcrs_sym);

        for (auto &op : impl_->ops) {
            if (op->hessian_bcrs_sym(
                        x, rowptr, colidx, block_stride, diag_values, off_diag_values) !=
                SFEM_SUCCESS) {
                std::cerr << "Failed hessian_bcrs_sym in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }
        return SFEM_SUCCESS;
    }

    int Function::hessian_diag(const real_t *const x, real_t *const values) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.hessian_diag);

        for (auto &op : impl_->ops) {
            if (op->hessian_diag(x, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_diag in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            for (auto &c : impl_->constraints) {
                c->apply_value(1, values);
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::gradient(const real_t *const x, real_t *const out) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.gradient);

        for (auto &op : impl_->ops) {
            if (op->gradient(x, out) != SFEM_SUCCESS) {
                std::cerr << "Failed gradient in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            constraints_gradient(x, out);
        }

        return SFEM_SUCCESS;
    }

    int Function::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.apply);

        for (auto &op : impl_->ops) {
            if (op->apply(x, h, out) != SFEM_SUCCESS) {
                std::cerr << "Failed apply in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        if (impl_->handle_constraints) {
            copy_constrained_dofs(h, out);
        }

        return SFEM_SUCCESS;
    }

    std::shared_ptr<Operator<real_t>> Function::linear_op_variant(
            const std::vector<std::pair<std::string, int>> &options) {
        std::vector<std::shared_ptr<Op>> cloned_ops;

        for (auto &op : impl_->ops) {
            auto c = op->clone();

            for (auto p : options) {
                c->set_option(p.first, p.second);
            }

            cloned_ops.push_back(c);
        }

        return sfem::make_op<real_t>(
                this->space()->n_dofs(),
                this->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) {
                    SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.apply);

                    for (auto op : cloned_ops) {
                        if (op->apply(nullptr, x, y) != SFEM_SUCCESS) {
                            std::cerr << "Failed apply in op: " << op->name() << "\n";
                            assert(false);
                        }
                    }

                    if (impl_->handle_constraints) {
                        copy_constrained_dofs(x, y);
                    }
                },
                this->execution_space());
    }

    int Function::value(const real_t *x, real_t *const out) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.value);

        for (auto &op : impl_->ops) {
            if (op->value(x, out) != SFEM_SUCCESS) {
                std::cerr << "Failed value in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::apply_constraints(real_t *const x) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.apply_constraints);

        for (auto &c : impl_->constraints) {
            c->apply(x);
        }
        return SFEM_SUCCESS;
    }

    int Function::constraints_gradient(const real_t *const x, real_t *const g) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.constraints_gradient);

        for (auto &c : impl_->constraints) {
            c->gradient(x, g);
        }
        return SFEM_SUCCESS;
    }

    int Function::apply_zero_constraints(real_t *const x) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.apply_zero_constraints);

        for (auto &c : impl_->constraints) {
            c->apply_zero(x);
        }
        return SFEM_SUCCESS;
    }

    int Function::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.copy_constrained_dofs);

        for (auto &c : impl_->constraints) {
            c->copy_constrained_dofs(src, dest);
        }
        return SFEM_SUCCESS;
    }

    int Function::report_solution(const real_t *const x) {
        SFEM_FUNCTION_SCOPED_TIMING(impl_->timings.report_solution);

        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        return impl_->output->write("out", x);
    }

    int Function::initial_guess(real_t *const x) { return SFEM_SUCCESS; }

    int Function::set_output_dir(const char *path) {
        impl_->output->set_output_dir(path);
        return SFEM_SUCCESS;
    }

    std::shared_ptr<Output> Function::output() { return impl_->output; }

    std::shared_ptr<Function> Function::derefine(const bool dirichlet_as_zero) {
        return derefine(impl_->space->derefine(), dirichlet_as_zero);
    }

    std::shared_ptr<Function> Function::derefine(const std::shared_ptr<FunctionSpace> &space,
                                                 const bool dirichlet_as_zero) {
        auto ret = std::make_shared<Function>(space);

        for (auto &o : impl_->ops) {
            ret->impl_->ops.push_back(o->derefine_op(space));
        }

        for (auto &c : impl_->constraints) {
            ret->impl_->constraints.push_back(c->derefine(space, dirichlet_as_zero));
        }

        ret->impl_->handle_constraints = impl_->handle_constraints;

        return ret;
    }

    std::shared_ptr<Function> Function::lor() { return lor(impl_->space->lor()); }
    std::shared_ptr<Function> Function::lor(const std::shared_ptr<FunctionSpace> &space) {
        auto ret = std::make_shared<Function>(space);

        for (auto &o : impl_->ops) {
            ret->impl_->ops.push_back(o->lor_op(space));
        }

        for (auto &c : impl_->constraints) {
            ret->impl_->constraints.push_back(c);
        }

        ret->impl_->handle_constraints = impl_->handle_constraints;

        return ret;
    }

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
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_type_variant(element_type);
            ret->mu = mu;
            ret->lambda = lambda;
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->mu = mu;
            ret->lambda = lambda;
            return ret;
        }

        const char *name() const override { return "LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        LinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            auto graph = space->node_to_node_graph();

            linear_elasticity_crs_aos(element_type,
                                      mesh->nelements,
                                      mesh->nnodes,
                                      mesh->elements,
                                      mesh->points,
                                      this->mu,
                                      this->lambda,
                                      graph->rowptr()->data(),
                                      graph->colidx()->data(),
                                      values);

            return SFEM_SUCCESS;
        }

        int hessian_bsr(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            auto graph = space->node_to_node_graph();

            linear_elasticity_bsr(element_type,
                                  mesh->nelements,
                                  mesh->nnodes,
                                  mesh->elements,
                                  mesh->points,
                                  this->mu,
                                  this->lambda,
                                  graph->rowptr()->data(),
                                  graph->colidx()->data(),
                                  values);

            return SFEM_SUCCESS;
        }

        int hessian_bcrs_sym(const real_t *const x,
                             const count_t *const rowptr,
                             const idx_t *const colidx,
                             const ptrdiff_t block_stride,
                             real_t **const diag_values,
                             real_t **const off_diag_values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_bcrs_sym(element_type,
                                       mesh->nelements,
                                       mesh->nnodes,
                                       mesh->elements,
                                       mesh->points,
                                       this->mu,
                                       this->lambda,
                                       rowptr,
                                       colidx,
                                       block_stride,
                                       diag_values,
                                       off_diag_values);
            return SFEM_SUCCESS;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            linear_elasticity_assemble_diag_aos(element_type,
                                                mesh->nelements,
                                                mesh->nnodes,
                                                mesh->elements,
                                                mesh->points,
                                                this->mu,
                                                this->lambda,
                                                out);
            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const out) override {
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

            return SFEM_SUCCESS;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
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

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
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

            return SFEM_SUCCESS;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class SemiStructuredLinearElasticity : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType element_type { INVALID };

        real_t mu{1}, lambda{1};
        bool use_affine_approximation{false};
        long calls{0};
        double total_time{0};

        ~SemiStructuredLinearElasticity() {
            if (calls) {
                printf("SemiStructuredLinearElasticity::apply(%s) called %ld times. Total: %g [s], "
                       "Avg: %g [s]\n",
                       use_affine_approximation ? "affine" : "isoparametric",
                       calls,
                       total_time,
                       total_time / calls);
            }
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(space->has_semi_structured_mesh());
            if (!space->has_semi_structured_mesh()) {
                fprintf(stderr,
                        "[Error] SemiStructuredLinearElasticity::create requires space with "
                        "semi_structured_mesh!\n");
                return nullptr;
            }

            assert(space->element_type() == PROTEUS_HEX8);  // REMOVEME once generalized approach
            auto ret = std::make_unique<SemiStructuredLinearElasticity>(space);

            real_t SFEM_SHEAR_MODULUS = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            ret->mu = SFEM_SHEAR_MODULUS;
            ret->lambda = SFEM_FIRST_LAME_PARAMETER;
            ret->element_type = (enum ElemType)space->element_type();

            int SFEM_HEX8_ASSUME_AFFINE = ret->use_affine_approximation;
            SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
            ret->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

            return ret;
        }

        void set_option(const std::string &name, bool val) override {
            if (name == "ASSUME_AFFINE") {
                use_affine_approximation = val;
            }
        }

        std::shared_ptr<Op> clone() const override {
            auto ret = std::make_shared<SemiStructuredLinearElasticity>(space);
            *ret = *this;
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            assert(false);
            fprintf(stderr, "[Error] ss:LinearElasticity::lor_op NOT IMPLEMENTED!\n");
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            assert(space->element_type() == macro_base_elem(element_type));
            auto ret = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->mu = mu;
            ret->lambda = lambda;
            return ret;
        }

        const char *name() const override { return "ss:LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        SemiStructuredLinearElasticity(const std::shared_ptr<FunctionSpace> &space)
            : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int hessian_bsr(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            auto &ssm = space->semi_structured_mesh();

            return proteus_affine_hex8_elasticity_bsr(ssm.level(),
                                                      ssm.n_elements(),
                                                      ssm.interior_start(),
                                                      ssm.element_data(),
                                                      ssm.point_data(),
                                                      this->mu,
                                                      this->lambda,
                                                      rowptr,
                                                      colidx,
                                                      values);
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            assert(element_type == PROTEUS_HEX8);  // REMOVEME once generalized approach

            auto &ssm = space->semi_structured_mesh();

            calls++;

            double tick = MPI_Wtime();
            int err;
            if (use_affine_approximation) {
                err = proteus_affine_hex8_linear_elasticity_apply(ssm.level(),
                                                                  ssm.n_elements(),
                                                                  ssm.interior_start(),
                                                                  ssm.element_data(),
                                                                  ssm.point_data(),
                                                                  mu,
                                                                  lambda,
                                                                  3,
                                                                  &h[0],
                                                                  &h[1],
                                                                  &h[2],
                                                                  3,
                                                                  &out[0],
                                                                  &out[1],
                                                                  &out[2]);

            } else {
                err = proteus_hex8_linear_elasticity_apply(ssm.level(),
                                                           ssm.n_elements(),
                                                           ssm.interior_start(),
                                                           ssm.element_data(),
                                                           ssm.point_data(),
                                                           mu,
                                                           lambda,
                                                           3,
                                                           &h[0],
                                                           &h[1],
                                                           &h[2],
                                                           3,
                                                           &out[0],
                                                           &out[1],
                                                           &out[2]);
            }

            double tock = MPI_Wtime();
            total_time += (tock - tick);

            return err;
        }

        int value(const real_t *x, real_t *const out) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
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
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret = std::make_shared<Laplacian>(space);
            ret->element_type = macro_type_variant(element_type);
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret = std::make_shared<Laplacian>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        Laplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            auto graph = space->dof_to_dof_graph();

            return laplacian_crs(element_type,
                                 mesh->nelements,
                                 mesh->nnodes,
                                 mesh->elements,
                                 mesh->points,
                                 graph->rowptr()->data(),
                                 graph->colidx()->data(),
                                 values);
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return laplacian_diag(element_type,
                                  mesh->nelements,
                                  mesh->nnodes,
                                  mesh->elements,
                                  mesh->points,
                                  values);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return laplacian_assemble_gradient(element_type,
                                               mesh->nelements,
                                               mesh->nnodes,
                                               mesh->elements,
                                               mesh->points,
                                               x,
                                               out);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return laplacian_apply(element_type,
                                   mesh->nelements,
                                   mesh->nnodes,
                                   mesh->elements,
                                   mesh->points,
                                   h,
                                   out);
        }

        int value(const real_t *x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return laplacian_assemble_value(element_type,
                                            mesh->nelements,
                                            mesh->nnodes,
                                            mesh->elements,
                                            mesh->points,
                                            x,
                                            out);
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class SemiStructuredLaplacian : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType element_type { INVALID };
        bool use_affine_approximation{false};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(space->has_semi_structured_mesh());
            if (!space->has_semi_structured_mesh()) {
                fprintf(stderr,
                        "[Error] SemiStructuredLaplacian::create requires space with "
                        "semi_structured_mesh!\n");
                return nullptr;
            }

            assert(space->element_type() == PROTEUS_HEX8);  // REMOVEME once generalized approach
            auto ret = std::make_unique<SemiStructuredLaplacian>(space);

            ret->element_type = (enum ElemType)space->element_type();

            int SFEM_HEX8_ASSUME_AFFINE = ret->use_affine_approximation;
            SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
            ret->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            assert(false);
            fprintf(stderr, "[Error] ss:Laplacian::lor_op NOT IMPLEMENTED!\n");
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            assert(space->element_type() == macro_base_elem(element_type));
            auto ret = std::make_shared<Laplacian>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }

        const char *name() const override { return "ss:Laplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        SemiStructuredLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            assert(element_type == PROTEUS_HEX8);  // REMOVEME once generalized approach

            auto &ssm = space->semi_structured_mesh();

            if (use_affine_approximation) {
                return proteus_affine_hex8_laplacian_apply(ssm.level(),
                                                           ssm.n_elements(),
                                                           ssm.interior_start(),
                                                           ssm.element_data(),
                                                           ssm.point_data(),
                                                           h,
                                                           out);

            } else {
                return proteus_hex8_laplacian_apply(ssm.level(),
                                                    ssm.n_elements(),
                                                    ssm.interior_start(),
                                                    ssm.element_data(),
                                                    ssm.point_data(),
                                                    h,
                                                    out);
            }
        }

        int value(const real_t *x, real_t *const out) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
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
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        Mass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            auto graph = space->dof_to_dof_graph();

            assemble_mass(element_type,
                          mesh->nelements,
                          mesh->nnodes,
                          mesh->elements,
                          mesh->points,
                          graph->rowptr()->data(),
                          graph->colidx()->data(),
                          values);

            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            apply_mass(element_type,
                       mesh->nelements,
                       mesh->nnodes,
                       mesh->elements,
                       mesh->points,
                       1,
                       x,
                       1,
                       out);

            return SFEM_SUCCESS;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            apply_mass(element_type,
                       mesh->nelements,
                       mesh->nnodes,
                       mesh->elements,
                       mesh->points,
                       1,
                       h,
                       1,
                       out);

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
            // auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // mass_assemble_value((enum ElemType)space->element_type(),
            //                     mesh->nelements,
            //                     mesh->nnodes,
            //                     mesh->elements,
            //                     mesh->points,
            //                     x,
            //                     out);

            // return SFEM_SUCCESS;

            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
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
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        LumpedMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
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

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int value(const real_t *x, real_t *const out) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
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
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        CVFEMMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            cvfem_cv_volumes(element_type,
                             mesh->nelements,
                             mesh->nnodes,
                             mesh->elements,
                             mesh->points,
                             values);

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int value(const real_t *x, real_t *const out) override {
            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
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
                       real_t *v) override {
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
                ret->element_type = (enum ElemType)space->element_type();
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

        int initialize() override { return SFEM_SUCCESS; }

        CVFEMUpwindConvection(const std::shared_ptr<FunctionSpace> &space) : space(space) {
            vel[0] = nullptr;
            vel[1] = nullptr;
            vel[2] = nullptr;
        }

        ~CVFEMUpwindConvection() {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // auto graph = space->dof_to_dof_graph();

            // cvfem_convection_assemble_hessian(element_type,
            //                            mesh->nelements,
            //                            mesh->nnodes,
            //                            mesh->elements,
            //                            mesh->points,
            //                            graph->rowptr()->data(),
            //                            graph->colidx()->data(),
            //                            values);

            // return SFEM_SUCCESS;

            assert(0);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            cvfem_convection_apply(element_type,
                                   mesh->nelements,
                                   mesh->nnodes,
                                   mesh->elements,
                                   mesh->points,
                                   vel,
                                   x,
                                   out);

            return SFEM_SUCCESS;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            cvfem_convection_apply(element_type,
                                   mesh->nelements,
                                   mesh->nnodes,
                                   mesh->elements,
                                   mesh->points,
                                   vel,
                                   h,
                                   out);

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // cvfem_convection_assemble_value(element_type,
            //                          mesh->nelements,
            //                          mesh->nnodes,
            //                          mesh->elements,
            //                          mesh->points,
            //                          x,
            //                          out);

            // return SFEM_SUCCESS;

            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    //

    class NeoHookeanOgden final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType element_type { INVALID };

        real_t mu{1}, lambda{1};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(mesh->spatial_dim == space->block_size());

            auto ret = std::make_unique<NeoHookeanOgden>(space);

            real_t SFEM_SHEAR_MODULUS = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            ret->mu = SFEM_SHEAR_MODULUS;
            ret->lambda = SFEM_FIRST_LAME_PARAMETER;
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret = std::make_shared<NeoHookeanOgden>(space);
            ret->element_type = macro_type_variant(element_type);
            ret->mu = mu;
            ret->lambda = lambda;
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret = std::make_shared<NeoHookeanOgden>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->mu = mu;
            ret->lambda = lambda;
            return ret;
        }

        const char *name() const override { return "NeoHookeanOgden"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        NeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            auto graph = space->node_to_node_graph();

            return neohookean_ogden_hessian_aos(element_type,
                                                mesh->nelements,
                                                mesh->nnodes,
                                                mesh->elements,
                                                mesh->points,
                                                this->mu,
                                                this->lambda,
                                                x,
                                                graph->rowptr()->data(),
                                                graph->colidx()->data(),
                                                values);
        }

        int hessian_diag(const real_t *const x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return neohookean_ogden_diag_aos(element_type,
                                             mesh->nelements,
                                             mesh->nnodes,
                                             mesh->elements,
                                             mesh->points,
                                             this->mu,
                                             this->lambda,
                                             x,
                                             out);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return neohookean_ogden_gradient_aos(element_type,
                                                 mesh->nelements,
                                                 mesh->nnodes,
                                                 mesh->elements,
                                                 mesh->points,
                                                 this->mu,
                                                 this->lambda,
                                                 x,
                                                 out);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return neohookean_ogden_apply_aos(element_type,
                                              mesh->nelements,
                                              mesh->nnodes,
                                              mesh->elements,
                                              mesh->points,
                                              this->mu,
                                              this->lambda,
                                              x,
                                              h,
                                              out);
        }

        int value(const real_t *x, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            return neohookean_ogden_value_aos(element_type,
                                              mesh->nelements,
                                              mesh->nnodes,
                                              mesh->elements,
                                              mesh->points,
                                              this->mu,
                                              this->lambda,
                                              x,
                                              out);
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class BoundaryMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Buffer<idx_t *>> boundary_elements;
        enum ElemType element_type { INVALID };

        const char *name() const override { return "BoundaryMass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(
                const std::shared_ptr<FunctionSpace> &space,
                const std::shared_ptr<Buffer<idx_t *>> &boundary_elements) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            auto ret = std::make_unique<BoundaryMass>(space);
            ret->element_type = shell_type(side_type((enum ElemType)space->element_type()));
            ret->boundary_elements = boundary_elements;
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        BoundaryMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            // auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // auto graph = space->dof_to_dof_graph();

            // assemble_mass(element_type,
            //               boundary_elements->extent(1),
            //               mesh->nnodes,
            //               boundary_elements->data(),
            //               mesh->points,
            //               graph->rowptr()->data(),
            //               graph->colidx()->data(),
            //               values);

            // return SFEM_SUCCESS;

            assert(0);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            // auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // assert(1 == space->block_size());

            // apply_mass(element_type,
            //            boundary_elements->extent(1),
            //            mesh->nnodes,
            //            boundary_elements->data(),
            //            mesh->points,
            //            x,
            //            out);

            // return SFEM_SUCCESS;

            assert(0);
            return SFEM_FAILURE;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            int block_size = space->block_size();
            auto data = boundary_elements->data();

            for (int d = 0; d < block_size; d++) {
                apply_mass(element_type,
                           boundary_elements->extent(1),
                           mesh->nnodes,
                           boundary_elements->data(),
                           mesh->points,
                           block_size,
                           &h[d],
                           block_size,
                           &out[d]);
            }

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
            // auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // mass_assemble_value((enum ElemType)space->element_type(),
            //                     mesh->nelements,
            //                     mesh->nnodes,
            //                     mesh->elements,
            //                     mesh->points,
            //                     x,
            //                     out);

            // return SFEM_SUCCESS;

            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class Factory::Impl {
    public:
        std::map<std::string, FactoryFunction> name_to_create;
        std::map<std::string, FactoryFunctionBoundary> name_to_create_boundary;
    };

    Factory::Factory() : impl_(std::make_unique<Impl>()) {}

    Factory::~Factory() = default;

    Factory &Factory::instance() {
        static Factory instance_;

        if (instance_.impl_->name_to_create.empty()) {
            instance_.private_register_op("LinearElasticity", LinearElasticity::create);
            instance_.private_register_op("ss:LinearElasticity",
                                          SemiStructuredLinearElasticity::create);
            instance_.private_register_op("Laplacian", Laplacian::create);
            instance_.private_register_op("ss:Laplacian", SemiStructuredLaplacian::create);
            instance_.private_register_op("CVFEMUpwindConvection", CVFEMUpwindConvection::create);
            instance_.private_register_op("Mass", Mass::create);
            instance_.private_register_op("CVFEMMass", CVFEMMass::create);
            instance_.private_register_op("LumpedMass", LumpedMass::create);
            instance_.private_register_op("NeoHookeanOgden", NeoHookeanOgden::create);

            instance_.impl_->name_to_create_boundary["BoundaryMass"] = BoundaryMass::create;
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

        std::string m_name = name;

        if (space->has_semi_structured_mesh()) {
            m_name = "ss:" + m_name;
        }

        auto &ntc = instance().impl_->name_to_create;
        auto it = ntc.find(m_name);

        if (it == ntc.end()) {
            std::cerr << "Unable to find op " << m_name << "\n";
            return nullptr;
        }

        return it->second(space);
    }

    std::shared_ptr<Op> Factory::create_boundary_op(
            const std::shared_ptr<FunctionSpace> &space,
            const std::shared_ptr<Buffer<idx_t *>> &boundary_elements,
            const char *name) {
        assert(instance().impl_);

        auto &ntc = instance().impl_->name_to_create_boundary;
        auto it = ntc.find(name);

        if (it == ntc.end()) {
            std::cerr << "Unable to find op " << name << "\n";
            return nullptr;
        }

        return it->second(space, boundary_elements);
    }

    std::string d_op_str(const std::string &name) { return "gpu:" + name; }

    std::shared_ptr<Buffer<idx_t *>> mesh_connectivity_from_file(MPI_Comm comm,
                                                                 const char *folder) {
        char pattern[1024 * 10];
        sprintf(pattern, "%s/i*.raw", folder);

        std::shared_ptr<Buffer<idx_t *>> ret;

        glob_t gl;
        glob(pattern, GLOB_MARK, NULL, &gl);

        int n_files = gl.gl_pathc;

        idx_t **data = (idx_t **)malloc(n_files * sizeof(idx_t *));

        ptrdiff_t local_size = -1;
        ptrdiff_t size = -1;

        printf("n_files (%d):\n", n_files);
        int err = 0;
        for (int np = 0; np < n_files; np++) {
            printf("%s\n", gl.gl_pathv[np]);

            idx_t *idx = 0;
            err |= array_create_from_file(
                    comm, gl.gl_pathv[np], SFEM_MPI_IDX_T, (void **)&idx, &local_size, &size);

            data[np] = idx;
        }

        globfree(&gl);

        ret = std::make_shared<Buffer<idx_t *>>(
                n_files,
                local_size,
                data,
                [](int n, void **data) {
                    for (int i = 0; i < n; i++) {
                        free(data[i]);
                    }

                    free(data);
                },
                MEMORY_SPACE_HOST);

        assert(!err);

        return ret;
    }

}  // namespace sfem
