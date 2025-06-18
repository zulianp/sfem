#include "sfem_Function.hpp"

#include <stddef.h>

#include "matrixio_array.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

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
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

// Ops

#include "cvfem_operators.h"
#include "hex8_laplacian.h"
#include "hex8_vector_laplacian.h"
#include "integrate_values.h"
#include "laplacian.h"
#include "linear_elasticity.h"
#include "mass.h"
#include "neohookean_ogden.h"
#include "spectral_hex_laplacian.h"
#include "sshex8_laplacian.h"
#include "sshex8_linear_elasticity.h"
#include "sshex8_mass.h"
#include "sshex8_stencil_element_matrix_apply.h"
#include "sshex8_vector_laplacian.h"

// Mesh
#include "adj_table.h"
#include "hex8_fff.h"
#include "hex8_jacobian.h"
#include "sfem_hex8_mesh_graph.h"
#include "sshex8.h"
#include "sshex8_mesh.h"

// Multigrid
#include "sfem_prolongation_restriction.h"

// C++ includes
#include "sfem_CRSGraph.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#ifdef SFEM_ENABLE_RYAML

#if defined(RYML_SINGLE_HEADER)  // using the single header directly in the executable
#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>
#elif defined(RYML_SINGLE_HEADER_LIB)  // using the single header from a library
#include <ryml_all.hpp>
#else
#include <ryml.hpp>
// <ryml_std.hpp> is needed if interop with std containers is
// desired; ryml itself does not use any STL container.
// For this sample, we will be using std interop, so...
#include <c4/format.hpp>  // needed for the examples below
#include <ryml_std.hpp>   // optional header, provided for std:: interop
#endif

#include <sstream>
#endif

#include <map>

namespace sfem {

    class Sideset::Impl final {
    public:
        MPI_Comm                               comm;
        std::shared_ptr<Buffer<element_idx_t>> parent;
        std::shared_ptr<Buffer<int16_t>>       lfi;
    };

    Sideset::Sideset(MPI_Comm                                      comm,
                     const std::shared_ptr<Buffer<element_idx_t>> &parent,
                     const std::shared_ptr<Buffer<int16_t>>       &lfi)
        : impl_(std::make_unique<Impl>()) {
        impl_->comm   = comm;
        impl_->parent = parent;
        impl_->lfi    = lfi;
    }

    Sideset::Sideset() : impl_(std::make_unique<Impl>()) {}
    Sideset::~Sideset() = default;

    ptrdiff_t Sideset::size() const { return impl_->parent->size(); }

    MPI_Comm Sideset::comm() const { return impl_->comm; }

    std::shared_ptr<Sideset> Sideset::create(MPI_Comm                                      comm,
                                             const std::shared_ptr<Buffer<element_idx_t>> &parent,
                                             const std::shared_ptr<Buffer<int16_t>>       &lfi) {
        return std::make_shared<Sideset>(comm, parent, lfi);
    }

    std::shared_ptr<Sideset> Sideset::create_from_file(MPI_Comm comm, const char *path) {
        auto ret = std::make_shared<Sideset>();
        if (ret->read(comm, path) != SFEM_SUCCESS) return nullptr;
        return ret;
    }

    std::shared_ptr<Sideset> Sideset::create_from_selector(
            const std::shared_ptr<Mesh>                                         &mesh,
            const std::function<bool(const geom_t, const geom_t, const geom_t)> &selector) {
        SFEM_TRACE_SCOPE("Sideset::create_from_selector");

        const ptrdiff_t nelements = mesh->n_elements();
        const ptrdiff_t nnodes    = mesh->n_nodes();
        const int       dim       = mesh->spatial_dimension();

        auto elements = mesh->elements()->data();
        auto points   = mesh->points()->data();

        enum ElemType element_type = mesh->element_type();

        const enum ElemType st   = side_type(element_type);
        const int           nnxs = elem_num_nodes(st);
        const int           ns   = elem_num_sides(element_type);

        std::vector<int> local_side_table(ns * nnxs);
        fill_local_side_table(mesh->element_type(), local_side_table.data());

        std::list<element_idx_t> parent_list;
        std::list<int16_t>       lfi_list;

        for (ptrdiff_t e = 0; e < nelements; e++) {
            for (int s = 0; s < ns; s++) {
                // Barycenter of face
                double p[3] = {0, 0, 0};

                for (int ln = 0; ln < nnxs; ln++) {
                    const idx_t node = elements[local_side_table[s * nnxs + ln]][e];

                    for (int d = 0; d < dim; d++) {
                        p[d] += points[d][node];
                    }
                }

                for (int d = 0; d < dim; d++) {
                    p[d] /= nnxs;
                }

                if (selector(p[0], p[1], p[2])) {
                    parent_list.push_back(e);
                    lfi_list.push_back(s);
                }
            }
        }

        const ptrdiff_t nparents = parent_list.size();
        auto            parent   = create_host_buffer<element_idx_t>(nparents);
        auto            lfi      = create_host_buffer<int16_t>(nparents);

        {
            ptrdiff_t idx = 0;
            for (auto p : parent_list) {
                parent->data()[idx++] = p;
            }
        }

        {
            ptrdiff_t idx = 0;
            for (auto l : lfi_list) {
                lfi->data()[idx++] = l;
            }
        }

        return std::make_shared<Sideset>(mesh->comm(), parent, lfi);
    }

    int Sideset::read(MPI_Comm comm, const char *folder) {
        SFEM_TRACE_SCOPE("Sideset::read");

        impl_->comm = comm;

        std::string    folder_ = folder;
        ptrdiff_t      nlocal{0}, nglobal{0}, ncheck{0};
        element_idx_t *parent{nullptr};
        int16_t       *lfi{nullptr};

        if (array_create_from_file(
                    comm, (folder_ + "/parent.raw").c_str(), SFEM_MPI_ELEMENT_IDX_T, (void **)&parent, &nlocal, &nglobal) ||
            array_create_from_file(comm, (folder_ + "/lfi.int16.raw").c_str(), MPI_SHORT, (void **)&lfi, &ncheck, &nglobal)) {
            return SFEM_FAILURE;
        }

        impl_->parent = sfem::manage_host_buffer(nlocal, parent);
        impl_->lfi    = sfem::manage_host_buffer(nlocal, lfi);

        if (ncheck != nlocal) {
            SFEM_ERROR("Inconsistent array sizes in sideset at %s\n", folder);
            return SFEM_FAILURE;
        }

        return SFEM_SUCCESS;
    }

    std::shared_ptr<Buffer<element_idx_t>> Sideset::parent() { return impl_->parent; }
    std::shared_ptr<Buffer<int16_t>>       Sideset::lfi() { return impl_->lfi; }

    class FunctionSpace::Impl {
    public:
        std::shared_ptr<Mesh> mesh;
        int                   block_size{1};
        enum ElemType         element_type { INVALID };

        // Number of nodes of function-space (TODO)
        ptrdiff_t nlocal{0};
        ptrdiff_t nglobal{0};

        // CRS graph
        std::shared_ptr<CRSGraph>              node_to_node_graph;
        std::shared_ptr<CRSGraph>              dof_to_dof_graph;
        std::shared_ptr<sfem::Buffer<idx_t *>> device_elements;

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

    void FunctionSpace::set_device_elements(const std::shared_ptr<sfem::Buffer<idx_t *>> &elems) {
        impl_->device_elements = elems;
    }

    std::shared_ptr<sfem::Buffer<idx_t *>> FunctionSpace::device_elements() { return impl_->device_elements; }

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

    std::shared_ptr<FunctionSpace> FunctionSpace::derefine(const int to_level) {
        if (to_level == 1) {
            // FIXME the number of nodes in mesh does not change, will lead to bugs
            return std::make_shared<FunctionSpace>(impl_->mesh, impl_->block_size, macro_base_elem(impl_->element_type));
        }

        assert(has_semi_structured_mesh());
        return create(semi_structured_mesh().derefine(to_level), block_size());
    }

    FunctionSpace::FunctionSpace() : impl_(std::make_unique<Impl>()) {}

    std::shared_ptr<FunctionSpace> FunctionSpace::create(const std::shared_ptr<SemiStructuredMesh> &mesh, const int block_size) {
        auto ret                         = std::make_shared<FunctionSpace>();
        ret->impl_->mesh                 = mesh->macro_mesh();
        ret->impl_->block_size           = block_size;
        ret->impl_->element_type         = SSHEX8;
        ret->impl_->semi_structured_mesh = mesh;
        ret->impl_->nlocal               = mesh->n_nodes() * block_size;
        ret->impl_->nglobal              = ret->impl_->nlocal;
        return ret;
    }

    FunctionSpace::FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size, const enum ElemType element_type)
        : impl_(std::make_unique<Impl>()) {
        impl_->mesh       = mesh;
        impl_->block_size = block_size;
        assert(block_size > 0);

        if (element_type == INVALID) {
            impl_->element_type = mesh->element_type();
        } else {
            impl_->element_type = element_type;
        }

        if (impl_->element_type == mesh->element_type()) {
            impl_->nlocal  = mesh->n_nodes() * block_size;
            impl_->nglobal = mesh->n_nodes() * block_size;
        } else {
            // FIXME in parallel it will not work
            impl_->nlocal  = (max_node_id(impl_->element_type, mesh->n_elements(), mesh->elements()->data()) + 1) * block_size;
            impl_->nglobal = impl_->nlocal;
        }
    }

    int FunctionSpace::promote_to_semi_structured(const int level) {
        if (impl_->element_type == HEX8) {
            impl_->semi_structured_mesh = std::make_shared<SemiStructuredMesh>(impl_->mesh, level);
            impl_->element_type         = SSHEX8;
            impl_->nlocal               = impl_->semi_structured_mesh->n_nodes() * impl_->block_size;
            impl_->nglobal              = impl_->nlocal;
            return SFEM_SUCCESS;
        }

        return SFEM_FAILURE;
    }

    FunctionSpace::~FunctionSpace() = default;

    bool FunctionSpace::has_semi_structured_mesh() const { return static_cast<bool>(impl_->semi_structured_mesh); }

    Mesh &FunctionSpace::mesh() { return *impl_->mesh; }

    std::shared_ptr<Mesh> FunctionSpace::mesh_ptr() const { return impl_->mesh; }

    SemiStructuredMesh &FunctionSpace::semi_structured_mesh() { return *impl_->semi_structured_mesh; }

    int FunctionSpace::block_size() const { return impl_->block_size; }

    ptrdiff_t FunctionSpace::n_dofs() const { return impl_->nlocal; }

    std::shared_ptr<FunctionSpace> FunctionSpace::lor() const {
        return std::make_shared<FunctionSpace>(impl_->mesh, impl_->block_size, macro_type_variant(impl_->element_type));
    }

    std::shared_ptr<Buffer<idx_t>> create_nodeset_from_sideset(const std::shared_ptr<FunctionSpace> &space,
                                                               const std::shared_ptr<Sideset>       &sideset) {
        ptrdiff_t n_nodes{0};
        idx_t    *nodes{nullptr};
        if (space->has_semi_structured_mesh()) {
            auto &&ss = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE("sshex8_extract_nodeset_from_sideset");
            if (sshex8_extract_nodeset_from_sideset(ss.level(),
                                                    ss.element_data(),
                                                    sideset->parent()->size(),
                                                    sideset->parent()->data(),
                                                    sideset->lfi()->data(),
                                                    &n_nodes,
                                                    &nodes) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract nodeset from sideset!\n");
            }
        } else {
            if (extract_nodeset_from_sideset(space->element_type(),
                                             space->mesh_ptr()->elements()->data(),
                                             sideset->parent()->size(),
                                             sideset->parent()->data(),
                                             sideset->lfi()->data(),
                                             &n_nodes,
                                             &nodes) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract nodeset from sideset!\n");
            }
        }

        return sfem::manage_host_buffer(n_nodes, nodes);
    }

    std::pair<enum ElemType, std::shared_ptr<Buffer<idx_t *>>> create_surface_from_sideset(
            const std::shared_ptr<FunctionSpace> &space,
            const std::shared_ptr<Sideset>       &sideset) {
        if (space->has_semi_structured_mesh()) {
            auto &&ssmesh = space->semi_structured_mesh();
            auto   ss_sides =
                    sfem::create_host_buffer<idx_t>((ssmesh.level() + 1) * (ssmesh.level() + 1), sideset->parent()->size());

            if (sshex8_extract_surface_from_sideset(ssmesh.level(),
                                                    ssmesh.element_data(),
                                                    sideset->parent()->size(),
                                                    sideset->parent()->data(),
                                                    sideset->lfi()->data(),
                                                    ss_sides->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }

            idx_t           *idx          = nullptr;
            ptrdiff_t        n_contiguous = SFEM_PTRDIFF_INVALID;
            std::vector<int> levels(sshex8_hierarchical_n_levels(ssmesh.level()));

            // FiXME harcoded for sshex8
            sshex8_hierarchical_mesh_levels(ssmesh.level(), levels.size(), levels.data());

            const int nnxs    = 4;
            const int nexs    = ssmesh.level() * ssmesh.level();
            auto      surface = sfem::create_host_buffer<idx_t>(nnxs, sideset->parent()->size() * nexs);

            ssquad4_to_standard_quad4_mesh(ssmesh.level(), sideset->parent()->size(), ss_sides->data(), surface->data());
            return {QUADSHELL4, surface};
        } else {
            auto st   = shell_type(side_type(space->element_type()));
            int  nnxs = elem_num_nodes(st);

            auto surface = sfem::create_host_buffer<idx_t>(nnxs, sideset->parent()->size());
            auto mesh    = space->mesh_ptr();
            if (extract_surface_from_sideset(space->element_type(),
                                             mesh->elements()->data(),
                                             sideset->parent()->size(),
                                             sideset->parent()->data(),
                                             sideset->lfi()->data(),
                                             surface->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to create surface from sideset!");
            }
            return {st, surface};
        }
    }

    class NeumannConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::vector<struct Condition>  conditions;
        ~Impl() {}
    };

    std::shared_ptr<FunctionSpace>                    NeumannConditions::space() { return impl_->space; }
    std::vector<struct NeumannConditions::Condition> &NeumannConditions::conditions() { return impl_->conditions; }

    int NeumannConditions::n_conditions() const { return impl_->conditions.size(); }

    const char *NeumannConditions::name() const { return "NeumannConditions"; }

    NeumannConditions::NeumannConditions(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<NeumannConditions> NeumannConditions::create_from_env(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeumannConditions::create_from_env");

        auto  neumann_conditions     = std::make_unique<NeumannConditions>(space);
        char *SFEM_NEUMANN_SURFACE   = 0;
        char *SFEM_NEUMANN_SIDESET   = 0;
        char *SFEM_NEUMANN_VALUE     = 0;
        char *SFEM_NEUMANN_COMPONENT = 0;

        SFEM_READ_ENV(SFEM_NEUMANN_SURFACE, );
        SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );
        SFEM_READ_ENV(SFEM_NEUMANN_COMPONENT, );
        SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );

        assert(!SFEM_NEUMANN_SURFACE || !SFEM_NEUMANN_SIDESET);

        if (!SFEM_NEUMANN_SURFACE && !SFEM_NEUMANN_SIDESET) return neumann_conditions;

        MPI_Comm comm = space->mesh_ptr()->comm();
        int      rank;
        MPI_Comm_rank(comm, &rank);

        auto &conds = neumann_conditions->impl_->conditions;

        char       *sets     = SFEM_NEUMANN_SIDESET ? SFEM_NEUMANN_SIDESET : SFEM_NEUMANN_SURFACE;
        const char *splitter = ",";
        int         count    = 1;
        {
            int i = 0;
            while (sets[i]) {
                count += (sets[i++] == splitter[0]);
                assert(i <= strlen(sets));
            }
        }

        auto st = shell_type(side_type(space->element_type()));

        printf("conds = %d, splitter=%c\n", count, splitter[0]);

        // NODESET/SIDESET
        {
            const char *pch = strtok(sets, splitter);
            int         i   = 0;
            while (pch != NULL) {
                printf("Reading file (%d/%d): %s\n", ++i, count, pch);
                struct Condition cneumann_conditions;
                cneumann_conditions.value     = 0;
                cneumann_conditions.component = 0;

                if (SFEM_NEUMANN_SURFACE) {
                    std::string pattern = pch;
                    pattern += "/i*.raw";
                    std::vector<std::string> paths = find_files(pattern);

                    int nnxs = elem_num_nodes(st);
                    if (int(paths.size()) != nnxs) {
                        SFEM_ERROR("Incorrect number of sides!");
                    }

                    idx_t **surface{nullptr};

                    ptrdiff_t nse = 0;
                    {
                        surface = (idx_t **)malloc(nnxs * sizeof(idx_t *));
                        int k   = 0;
                        for (auto &p : paths) {
                            idx_t    *ii{nullptr};
                            ptrdiff_t lsize{0}, gsize{0};
                            if (array_create_from_file(comm, pch, SFEM_MPI_IDX_T, (void **)&ii, &lsize, &gsize)) {
                                SFEM_ERROR("Failed to read file %s\n", pch);
                                break;
                            }

                            if (!nse || nse != lsize) {
                                assert(!nse || nse == lsize);
                                SFEM_ERROR("Inconsistent lenghts between files!\n");
                            }

                            nse          = lsize;
                            surface[k++] = ii;
                        }
                    }

                    cneumann_conditions.element_type = st;
                    cneumann_conditions.surface      = manage_host_buffer(nnxs, nse, surface);

                } else {
                    auto sideset                = Sideset::create_from_file(comm, pch);
                    cneumann_conditions.sideset = sideset;

                    auto surface                     = create_surface_from_sideset(space, sideset);
                    cneumann_conditions.element_type = surface.first;
                    cneumann_conditions.surface      = surface.second;
                }

                conds.push_back(cneumann_conditions);

                pch = strtok(NULL, splitter);
            }
        }

        if (SFEM_NEUMANN_COMPONENT) {
            const char *pch = strtok(SFEM_NEUMANN_COMPONENT, splitter);
            int         i   = 0;
            while (pch != NULL) {
                printf("Parsing comps (%d/%d): %s\n", i + 1, count, pch);
                conds[i].component = atoi(pch);
                i++;

                pch = strtok(NULL, splitter);
            }
        }

        if (SFEM_NEUMANN_VALUE) {
            static const char *path_key     = "path:";
            const int          path_key_len = strlen(path_key);

            const char *pch = strtok(SFEM_NEUMANN_VALUE, splitter);
            int         i   = 0;
            while (pch != NULL) {
                printf("Parsing  values (%d/%d): %s\n", i + 1, count, pch);
                assert(i < count);

                if (strncmp(pch, path_key, path_key_len) == 0) {
                    conds[i].value = 0;

                    real_t   *values{nullptr};
                    ptrdiff_t lsize, gsize;
                    if (array_create_from_file(comm, pch + path_key_len, SFEM_MPI_REAL_T, (void **)&values, &lsize, &gsize)) {
                        SFEM_ERROR("Failed to read file %s\n", pch + path_key_len);
                    }

                    if (conds[i].surface->extent(1) != lsize) {
                        if (!rank) {
                            SFEM_ERROR(
                                    "read_boundary_conditions: len(idx) != len(values) (%ld != "
                                    "%ld)\nfile:%s\n",
                                    (long)conds[i].surface->extent(1),
                                    (long)lsize,
                                    pch + path_key_len);
                        }
                    }

                    conds[i].value = 1;

                } else {
                    conds[i].value = atof(pch);
                }
                i++;

                pch = strtok(NULL, splitter);
            }
        }

        return neumann_conditions;
    }

    NeumannConditions::~NeumannConditions() = default;

    int NeumannConditions::hessian_crs(const real_t *const /*x*/,
                                       const count_t *const /*rowptr*/,
                                       const idx_t *const /*colidx*/,
                                       real_t *const /*values*/) {
        return SFEM_SUCCESS;
    }

    int NeumannConditions::gradient(const real_t *const /*x*/, real_t *const out) {
        SFEM_TRACE_SCOPE("NeumannConditions::gradient");

        auto space = impl_->space;
        auto mesh  = space->mesh_ptr();

        auto points = mesh->points();
        if (space->has_semi_structured_mesh()) {
            points = space->semi_structured_mesh().points();
        }

        int err = 0;
        for (auto &c : impl_->conditions) {
            if (c.values) {
                err |= integrate_values(c.element_type,
                                        c.surface->extent(1),
                                        mesh->n_nodes(),
                                        c.surface->data(),
                                        points->data(),
                                        // Use negative sign since we are on LHS
                                        -c.value,
                                        c.values->data(),
                                        space->block_size(),
                                        c.component,
                                        out);
            } else {
                err |= integrate_value(c.element_type,
                                       c.surface->extent(1),
                                       mesh->n_nodes(),
                                       c.surface->data(),
                                       points->data(),
                                       // Use negative sign since we are on LHS
                                       -c.value,
                                       space->block_size(),
                                       c.component,
                                       out);
            }
        }

        return err;
    }

    int NeumannConditions::apply(const real_t *const /*x*/, const real_t *const /*h*/, real_t *const /*out*/) {
        // No-Op
        return SFEM_SUCCESS;
    }

    int NeumannConditions::value(const real_t *x, real_t *const out) {
        // TODO
        return SFEM_SUCCESS;
    }

    std::shared_ptr<NeumannConditions> NeumannConditions::create(const std::shared_ptr<FunctionSpace> &space,
                                                                 const std::vector<struct Condition>  &conditions) {
        auto nc               = std::make_unique<NeumannConditions>(space);
        nc->impl_->conditions = conditions;

        std::map<  //
                std::shared_ptr<Sideset>,
                std::pair<enum ElemType, std::shared_ptr<Buffer<idx_t *>>>>
                sideset_to_surface;

        for (auto &c : nc->impl_->conditions) {
            if (!c.surface) {
                auto it = sideset_to_surface.find(c.sideset);
                if (it == sideset_to_surface.end()) {
                    auto surface                  = create_surface_from_sideset(space, c.sideset);
                    c.element_type                = surface.first;
                    c.surface                     = surface.second;
                    sideset_to_surface[c.sideset] = surface;
                } else {
                    c.element_type = it->second.first;
                    c.surface      = it->second.second;
                }
            }
        }

        return nc;
    }

    std::shared_ptr<Op> NeumannConditions::derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) {
        // std::map<  //
        //         std::shared_ptr<Sideset>,
        //         std::shared_ptr<Buffer<idx_t *>>>
        //         sideset_to_surface;

        // auto  coarse = std::make_shared<NeumannConditions>(derefined_space);
        // auto &conds  = impl_->conditions;

        // for (auto &c : nc->impl_->conditions) {
        //     auto it = sideset_to_surface.find(c.sideset);

        //     struct Condition cc = {.element_type = c.element_type,
        //                            .sideset      = c.sideset,
        //                            .surface      = nullptr,
        //                            .values       = nullptr,
        //                            .value        = c.value,
        //                            .component    = c.component};

        //     if (it == sideset_to_surface.end()) {
        //         // Derefine surface
        //     } else {
        //         // Reuse derefined surface
        //         cc.surface = it->second;
        //     }

        //     if (c.values) {
        //         // Restrict values
        //         SFEM_ERROR("IMPLEMENT ME!\n");
        //     }

        //     conds.push_back(cc);
        // }

        // auto coarse_sides = sfem::ssquad4_derefine_element_connectivity(level, coarse_level, fine_sides);

        // SFEM_ERROR("NOT NEEDED FOR NEUMANN!\n")
        return no_op();
    }

    int Constraint::apply_zero(real_t *const x) { return apply_value(0, x); }

    class DirichletConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::vector<struct Condition>  conditions;
    };

    std::shared_ptr<DirichletConditions> DirichletConditions::create(const std::shared_ptr<FunctionSpace> &space,
                                                                     const std::vector<struct Condition>  &conditions) {
        auto dc               = std::make_unique<DirichletConditions>(space);
        dc->impl_->conditions = conditions;

        for (auto &c : dc->impl_->conditions) {
            if (!c.nodeset) {
                c.nodeset = create_nodeset_from_sideset(space, c.sideset);
            }
        }

        return dc;
    }

    std::shared_ptr<FunctionSpace>                      DirichletConditions::space() { return impl_->space; }
    std::vector<struct DirichletConditions::Condition> &DirichletConditions::conditions() { return impl_->conditions; }

    int DirichletConditions::n_conditions() const { return impl_->conditions.size(); }

    DirichletConditions::DirichletConditions(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<Constraint> DirichletConditions::derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                                              const bool                            as_zero) const {
        SFEM_TRACE_SCOPE("DirichletConditions::derefine");

        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        auto et   = (enum ElemType)impl_->space->element_type();

        ptrdiff_t max_coarse_idx = -1;
        auto      coarse         = std::make_shared<DirichletConditions>(coarse_space);
        auto     &conds          = impl_->conditions;

        std::map<std::shared_ptr<Sideset>, std::shared_ptr<Buffer<idx_t>>> sideset_to_nodeset;
        for (size_t i = 0; i < conds.size(); i++) {
            ptrdiff_t coarse_num_nodes = 0;
            idx_t    *coarse_nodeset   = nullptr;

            struct Condition cdc;
            cdc.sideset   = conds[i].sideset;
            cdc.component = conds[i].component;
            cdc.value     = as_zero ? 0 : conds[i].value;

            if (!cdc.sideset) {
                if (max_coarse_idx == -1)
                    max_coarse_idx = max_node_id(coarse_space->element_type(), mesh->nelements, mesh->elements);

                hierarchical_create_coarse_indices(
                        max_coarse_idx, conds[i].nodeset->size(), conds[i].nodeset->data(), &coarse_num_nodes, &coarse_nodeset);
                cdc.nodeset = sfem::manage_host_buffer<idx_t>(coarse_num_nodes, coarse_nodeset);

                if (!as_zero && conds[i].values) {
                    cdc.values = create_host_buffer<real_t>(coarse_num_nodes);
                    hierarchical_collect_coarse_values(max_coarse_idx,
                                                       conds[i].nodeset->size(),
                                                       conds[i].nodeset->data(),
                                                       conds[i].values->data(),
                                                       cdc.values->data());
                }

            } else {
                assert(as_zero);

                auto it = sideset_to_nodeset.find(conds[i].sideset);
                if (it == sideset_to_nodeset.end()) {
                    auto nodeset                         = create_nodeset_from_sideset(coarse_space, cdc.sideset);
                    cdc.nodeset                          = nodeset;
                    sideset_to_nodeset[conds[i].sideset] = nodeset;

                } else {
                    cdc.nodeset = it->second;
                }
            }

            coarse->impl_->conditions.push_back(cdc);
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
                                            idx_t *const    idx,
                                            const int       component,
                                            const real_t    value) {
        struct Condition cdc;
        cdc.component = component;
        cdc.value     = value;
        cdc.nodeset   = manage_host_buffer<idx_t>(local_size, idx);
        impl_->conditions.push_back(cdc);
    }

    void DirichletConditions::add_condition(const ptrdiff_t local_size,
                                            const ptrdiff_t global_size,
                                            idx_t *const    idx,
                                            const int       component,
                                            real_t *const   values) {
        struct Condition cdc;
        cdc.component = component;
        cdc.value     = 0;
        cdc.nodeset   = manage_host_buffer<idx_t>(local_size, idx);
        cdc.values    = manage_host_buffer<real_t>(local_size, values);
        impl_->conditions.push_back(cdc);
    }

    // FIXME check for duplicate sidesets read from disk!
    std::shared_ptr<DirichletConditions> DirichletConditions::create_from_env(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("DirichletConditions::create_from_env");

        auto  dc                       = std::make_unique<DirichletConditions>(space);
        char *SFEM_DIRICHLET_NODESET   = 0;
        char *SFEM_DIRICHLET_SIDESET   = 0;
        char *SFEM_DIRICHLET_VALUE     = 0;
        char *SFEM_DIRICHLET_COMPONENT = 0;

        SFEM_READ_ENV(SFEM_DIRICHLET_NODESET, );
        SFEM_READ_ENV(SFEM_DIRICHLET_VALUE, );
        SFEM_READ_ENV(SFEM_DIRICHLET_COMPONENT, );
        SFEM_READ_ENV(SFEM_DIRICHLET_SIDESET, );

        assert(!SFEM_DIRICHLET_NODESET || !SFEM_DIRICHLET_SIDESET);

        if (!SFEM_DIRICHLET_NODESET && !SFEM_DIRICHLET_SIDESET) return dc;

        MPI_Comm comm = space->mesh_ptr()->comm();
        int      rank;
        MPI_Comm_rank(comm, &rank);

        auto &conds = dc->impl_->conditions;

        char       *sets     = SFEM_DIRICHLET_SIDESET ? SFEM_DIRICHLET_SIDESET : SFEM_DIRICHLET_NODESET;
        const char *splitter = ",";
        int         count    = 1;
        {
            int i = 0;
            while (sets[i]) {
                count += (sets[i++] == splitter[0]);
                assert(i <= strlen(sets));
            }
        }

        printf("conds = %d, splitter=%c\n", count, splitter[0]);

        // NODESET/SIDESET
        {
            const char *pch = strtok(sets, splitter);
            int         i   = 0;
            while (pch != NULL) {
                printf("Reading file (%d/%d): %s\n", ++i, count, pch);
                struct Condition cdc;
                cdc.value     = 0;
                cdc.component = 0;

                if (SFEM_DIRICHLET_NODESET) {
                    idx_t    *this_set{nullptr};
                    ptrdiff_t lsize{0}, gsize{0};
                    if (array_create_from_file(comm, pch, SFEM_MPI_IDX_T, (void **)&this_set, &lsize, &gsize)) {
                        SFEM_ERROR("Failed to read file %s\n", pch);
                        break;
                    }

                    cdc.nodeset = manage_host_buffer<idx_t>(lsize, this_set);
                } else {
                    cdc.sideset = Sideset::create_from_file(comm, pch);
                    cdc.nodeset = create_nodeset_from_sideset(space, cdc.sideset);
                }

                conds.push_back(cdc);

                pch = strtok(NULL, splitter);
            }
        }

        if (SFEM_DIRICHLET_COMPONENT) {
            const char *pch = strtok(SFEM_DIRICHLET_COMPONENT, splitter);
            int         i   = 0;
            while (pch != NULL) {
                printf("Parsing comps (%d/%d): %s\n", i + 1, count, pch);
                conds[i].component = atoi(pch);
                i++;

                pch = strtok(NULL, splitter);
            }
        }

        if (SFEM_DIRICHLET_VALUE) {
            static const char *path_key     = "path:";
            const int          path_key_len = strlen(path_key);

            const char *pch = strtok(SFEM_DIRICHLET_VALUE, splitter);
            int         i   = 0;
            while (pch != NULL) {
                printf("Parsing  values (%d/%d): %s\n", i + 1, count, pch);
                assert(i < count);

                if (strncmp(pch, path_key, path_key_len) == 0) {
                    conds[i].value = 0;

                    real_t   *values{nullptr};
                    ptrdiff_t lsize, gsize;
                    if (array_create_from_file(comm, pch + path_key_len, SFEM_MPI_REAL_T, (void **)&values, &lsize, &gsize)) {
                        SFEM_ERROR("Failed to read file %s\n", pch + path_key_len);
                    }

                    if (conds[i].nodeset->size() != lsize) {
                        if (!rank) {
                            SFEM_ERROR(
                                    "read_boundary_conditions: len(idx) != len(values) (%ld != "
                                    "%ld)\nfile:%s\n",
                                    (long)conds[i].nodeset->size(),
                                    (long)lsize,
                                    pch + path_key_len);
                        }
                    }

                } else {
                    conds[i].value = atof(pch);
                }
                i++;

                pch = strtok(NULL, splitter);
            }
        }

        return dc;
    }

    std::shared_ptr<DirichletConditions> DirichletConditions::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                               std::string                           yaml) {
        SFEM_TRACE_SCOPE("DirichletConditions::create_from_yaml");

#ifdef SFEM_ENABLE_RYAML
        auto dc = std::make_unique<DirichletConditions>(space);

        ryml::Tree tree  = ryml::parse_in_place(ryml::to_substr(yaml));
        auto       conds = tree["dirichlet_conditions"];

        MPI_Comm comm = space->mesh_ptr()->comm();

        for (auto c : conds.children()) {
            std::shared_ptr<Sideset>       sideset;
            std::shared_ptr<Buffer<idx_t>> nodeset;

            const bool is_sideset = c["type"].readable() && c["type"].val() == "sideset";
            const bool is_file    = c["format"].readable() && c["format"].val() == "file";
            const bool is_expr    = c["format"].readable() && c["format"].val() == "expr";

            assert(is_file || is_expr);

            if (is_sideset) {
                if (is_file) {
                    std::string path;
                    c["path"] >> path;
                    sideset = Sideset::create_from_file(space->mesh_ptr()->comm(), path.c_str());
                } else if (is_expr) {
                    assert(c["parent"].is_seq());
                    assert(c["lfi"].is_seq());

                    ptrdiff_t size   = c["parent"].num_children();
                    auto      parent = create_host_buffer<element_idx_t>(size);
                    auto      lfi    = create_host_buffer<int16_t>(size);

                    ptrdiff_t parent_count = 0;
                    for (auto p : c["parent"]) {
                        p >> parent->data()[parent_count++];
                    }

                    ptrdiff_t lfi_count = 0;
                    for (auto p : c["lfi"]) {
                        p >> lfi->data()[lfi_count++];
                    }

                    assert(lfi_count == parent_count);
                    sideset = std::make_shared<Sideset>(space->mesh_ptr()->comm(), parent, lfi);
                }

                ptrdiff_t n_nodes{0};
                idx_t    *nodes{nullptr};
                if (space->has_semi_structured_mesh()) {
                    auto &&ss = space->semi_structured_mesh();
                    SFEM_TRACE_SCOPE("sshex8_extract_nodeset_from_sideset");
                    if (sshex8_extract_nodeset_from_sideset(ss.level(),
                                                            ss.element_data(),
                                                            sideset->parent()->size(),
                                                            sideset->parent()->data(),
                                                            sideset->lfi()->data(),
                                                            &n_nodes,
                                                            &nodes) != SFEM_SUCCESS) {
                        SFEM_ERROR("Unable to extract nodeset from sideset!\n");
                    }
                } else {
                    if (extract_nodeset_from_sideset(space->element_type(),
                                                     space->mesh_ptr()->elements()->data(),
                                                     sideset->parent()->size(),
                                                     sideset->parent()->data(),
                                                     sideset->lfi()->data(),
                                                     &n_nodes,
                                                     &nodes) != SFEM_SUCCESS) {
                        SFEM_ERROR("Unable to extract nodeset from sideset!\n");
                    }
                }

                nodeset = sfem::manage_host_buffer(n_nodes, nodes);
            } else {
                if (is_file) {
                    std::string path;
                    c["path"] >> path;
                    idx_t    *arr{nullptr};
                    ptrdiff_t lsize, gsize;
                    if (!array_create_from_file(comm, path.c_str(), SFEM_MPI_IDX_T, (void **)&arr, &lsize, &gsize)) {
                        SFEM_ERROR("Unable to read file %s!\n", path.c_str());
                    }

                    nodeset = manage_host_buffer<idx_t>(lsize, arr);
                } else {
                    ptrdiff_t size  = c["nodes"].num_children();
                    nodeset         = create_host_buffer<idx_t>(size);
                    ptrdiff_t count = 0;
                    for (auto p : c["nodes"]) {
                        p >> nodeset->data()[count++];
                    }
                }
            }

            std::vector<int>    component;
            std::vector<real_t> value;
            auto                node_value     = c["value"];
            auto                node_component = c["component"];

            assert(node_value.readable());
            assert(node_component.readable());

            if (node_value.is_seq()) {
                node_value >> value;
            } else {
                value.resize(1);
                node_value >> value[0];
            }

            if (node_component.is_seq()) {
                node_component >> component;
            } else {
                component.resize(1);
                node_component >> component[0];
            }

            if (component.size() != value.size()) {
                SFEM_ERROR("Inconsistent sizes for component (%d) and value (%d)\n", (int)component.size(), (int)value.size());
            }

            for (size_t i = 0; i < component.size(); i++) {
                struct Condition cdc;
                cdc.component = component[i];
                cdc.value     = value[i];
                cdc.sideset   = sideset;
                cdc.nodeset   = nodeset;
                dc->impl_->conditions.push_back(cdc);
            }
        }

        return dc;
#else
        SFEM_ERROR("This functionaly requires -DSFEM_ENABLE_RYAML=ON\n");
        return nullptr;
#endif
    }

    std::shared_ptr<DirichletConditions> DirichletConditions::create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                               const std::string                    &path) {
        std::ifstream is(path);
        if (!is.good()) {
            SFEM_ERROR("Unable to read file %s\n", path.c_str());
        }

        std::ostringstream contents;
        contents << is.rdbuf();
        auto yaml = contents.str();
        is.close();

        return create_from_yaml(space, std::move(yaml));
    }

    int DirichletConditions::apply(real_t *const x) {
        SFEM_TRACE_SCOPE("DirichletConditions::apply");

        for (auto &c : impl_->conditions) {
            if (c.values) {
                constraint_nodes_to_values_vec(
                        c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, c.values->data(), x);
            } else {
                constraint_nodes_to_value_vec(
                        c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, c.value, x);
            }
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::gradient(const real_t *const x, real_t *const g) {
        SFEM_TRACE_SCOPE("DirichletConditions::gradient");

        for (auto &c : impl_->conditions) {
            constraint_gradient_nodes_to_value_vec(
                    c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, c.value, x, g);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::apply_value(const real_t value, real_t *const x) {
        SFEM_TRACE_SCOPE("DirichletConditions::apply_value");

        for (auto &c : impl_->conditions) {
            constraint_nodes_to_value_vec(
                    c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, value, x);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        SFEM_TRACE_SCOPE("DirichletConditions::copy_constrained_dofs");

        for (auto &c : impl_->conditions) {
            constraint_nodes_copy_vec(c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, src, dest);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::hessian_crs(const real_t *const  x,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         real_t *const        values) {
        SFEM_TRACE_SCOPE("DirichletConditions::hessian_crs");

        for (auto &c : impl_->conditions) {
            crs_constraint_nodes_to_identity_vec(
                    c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, 1, rowptr, colidx, values);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::hessian_bsr(const real_t *const  x,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         real_t *const        values) {
        SFEM_TRACE_SCOPE("DirichletConditions::hessian_bsr");

        for (auto &c : impl_->conditions) {
            bsr_constraint_nodes_to_identity_vec(
                    c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, 1, rowptr, colidx, values);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::mask(mask_t *mask) {
        SFEM_TRACE_SCOPE("DirichletConditions::mask");

        const int block_size = impl_->space->block_size();
        for (auto &c : impl_->conditions) {
            auto nodeset = c.nodeset->data();
            for (ptrdiff_t node = 0; node < c.nodeset->size(); node++) {
                const ptrdiff_t idx = nodeset[node] * block_size + c.component;
                mask_set(idx, mask);
            }
        }

        return SFEM_SUCCESS;
    }

    class Output::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        bool                           AoS_to_SoA{false};
        std::string                    output_dir{"."};
        std::string                    file_format{"%s/%s.raw"};
        std::string                    time_dependent_file_format{"%s/%s.%09d.raw"};
        size_t                         export_counter{0};
        logger_t                       time_logger;
        Impl() { log_init(&time_logger); }
        ~Impl() { log_destroy(&time_logger); }
    };

    void Output::enable_AoS_to_SoA(const bool val) { impl_->AoS_to_SoA = val; }

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
        SFEM_TRACE_SCOPE("Output::write");

        MPI_Comm comm = impl_->space->mesh_ptr()->comm();
        sfem::create_directory(impl_->output_dir.c_str());

        const int block_size = impl_->space->block_size();
        if (impl_->AoS_to_SoA && block_size > 1) {
            ptrdiff_t n_blocks = impl_->space->n_dofs() / block_size;

            auto buff = create_host_buffer<real_t>(n_blocks);
            auto bb   = buff->data();

            char path[2048];
            for (int b = 0; b < block_size; b++) {
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    bb[i] = x[i * block_size + b];
                }

                char b_name[1024];
                sprintf(b_name, "%s.%d", name, b);
                sprintf(path, impl_->file_format.c_str(), impl_->output_dir.c_str(), b_name);
                if (array_write(comm, path, SFEM_MPI_REAL_T, buff->data(), n_blocks, n_blocks)) {
                    return SFEM_FAILURE;
                }
            }

        } else {
            char path[2048];
            sprintf(path, impl_->file_format.c_str(), impl_->output_dir.c_str(), name);
            if (array_write(comm, path, SFEM_MPI_REAL_T, x, impl_->space->n_dofs(), impl_->space->n_dofs())) {
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    void Output::log_time(const real_t t) {
        if (log_is_empty(&impl_->time_logger)) {
            char path[2048];
            sprintf(path, "%s/time.txt", impl_->output_dir.c_str());
            log_create_file(&impl_->time_logger, path, "w");
        }

        log_write_double(&impl_->time_logger, t);
    }

    int Output::write_time_step(const char *name, const real_t t, const real_t *const x) {
        SFEM_TRACE_SCOPE("Output::write_time_step");

        auto      mesh       = (mesh_t *)impl_->space->mesh().impl_mesh();
        const int block_size = impl_->space->block_size();

        char path[2048];

        if (impl_->AoS_to_SoA && block_size > 1) {
            ptrdiff_t n_blocks = impl_->space->n_dofs() / block_size;

            auto buff = create_host_buffer<real_t>(n_blocks);
            auto bb   = buff->data();

            for (int b = 0; b < block_size; b++) {
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    bb[i] = x[i * block_size + b];
                }

                char b_name[1024];
                sprintf(b_name, "%s.%d", name, b);
                sprintf(path,
                        impl_->time_dependent_file_format.c_str(),
                        impl_->output_dir.c_str(),
                        b_name,
                        impl_->export_counter++);

                if (array_write(mesh->comm, path, SFEM_MPI_REAL_T, buff->data(), n_blocks, n_blocks)) {
                    return SFEM_FAILURE;
                }
            }

        } else {
            sfem::create_directory(impl_->output_dir.c_str());

            sprintf(path, impl_->time_dependent_file_format.c_str(), impl_->output_dir.c_str(), name, impl_->export_counter++);

            if (array_write(mesh->comm, path, SFEM_MPI_REAL_T, x, impl_->space->n_dofs(), impl_->space->n_dofs())) {
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    class Function::Impl {
    public:
        std::shared_ptr<FunctionSpace>           space;
        std::vector<std::shared_ptr<Op>>         ops;
        std::vector<std::shared_ptr<Constraint>> constraints;

        std::shared_ptr<Output> output;
        bool                    handle_constraints{true};
    };

    ExecutionSpace Function::execution_space() const {
        ExecutionSpace ret = EXECUTION_SPACE_INVALID;

        for (auto op : impl_->ops) {
            assert(ret == EXECUTION_SPACE_INVALID || ret == op->execution_space());
            ret = op->execution_space();
        }

        return ret;
    }

    void Function::describe(std::ostream &os) const {
        os << "n_dofs: " << impl_->space->n_dofs() << "\n";
        os << "n_ops: " << impl_->ops.size() << "\n";
        os << "n_constraints: " << impl_->constraints.size() << "\n";
    }

    Function::Function(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space  = space;
        impl_->output = std::make_shared<Output>(space);
    }

    std::shared_ptr<FunctionSpace> Function::space() { return impl_->space; }

    Function::~Function() {}

    void Function::add_operator(const std::shared_ptr<Op> &op) { impl_->ops.push_back(op); }
    void Function::add_constraint(const std::shared_ptr<Constraint> &c) { impl_->constraints.push_back(c); }

    void Function::clear_constraints() { impl_->constraints.clear(); }

    void Function::add_dirichlet_conditions(const std::shared_ptr<DirichletConditions> &c) { add_constraint(c); }

    int Function::constaints_mask(mask_t *mask) {
        SFEM_TRACE_SCOPE("Function::constaints_mask");

        for (auto &c : impl_->constraints) {
            c->mask(mask);
        }

        return SFEM_FAILURE;
    }

    std::shared_ptr<CRSGraph> Function::crs_graph() const { return impl_->space->dof_to_dof_graph(); }

    int Function::hessian_crs(const real_t *const  x,
                              const count_t *const rowptr,
                              const idx_t *const   colidx,
                              real_t *const        values) {
        SFEM_TRACE_SCOPE("Function::hessian_crs");

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

    int Function::hessian_bsr(const real_t *const  x,
                              const count_t *const rowptr,
                              const idx_t *const   colidx,
                              real_t *const        values) {
        SFEM_TRACE_SCOPE("Function::hessian_bsr");

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

    int Function::hessian_bcrs_sym(const real_t *const  x,
                                   const count_t *const rowptr,
                                   const idx_t *const   colidx,
                                   const ptrdiff_t      block_stride,
                                   real_t **const       diag_values,
                                   real_t **const       off_diag_values) {
        SFEM_TRACE_SCOPE("Function::hessian_bcrs_sym");
        for (auto &op : impl_->ops) {
            if (op->hessian_bcrs_sym(x, rowptr, colidx, block_stride, diag_values, off_diag_values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_bcrs_sym in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }
        return SFEM_SUCCESS;
    }

    int Function::hessian_crs_sym(const real_t *const  x,
                                  const count_t *const rowptr,
                                  const idx_t *const   colidx,
                                  real_t *const        diag_values,
                                  real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("Function::hessian_crs_sym");
        for (auto &op : impl_->ops) {
            if (op->hessian_crs_sym(x, rowptr, colidx, diag_values, off_diag_values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_crs_sym in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }
        return SFEM_SUCCESS;
    }

    int Function::hessian_diag(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("Function::hessian_diag");
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

    int Function::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("Function::hessian_block_diag_sym");

        for (auto &op : impl_->ops) {
            if (op->hessian_block_diag_sym(x, values) != SFEM_SUCCESS) {
                std::cerr << "Failed hessian_diag in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Function::gradient");

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
        SFEM_TRACE_SCOPE("Function::apply");

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

    std::shared_ptr<Operator<real_t>> Function::linear_op_variant(const std::vector<std::pair<std::string, int>> &options) {
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
        SFEM_TRACE_SCOPE("Function::value");

        for (auto &op : impl_->ops) {
            if (op->value(x, out) != SFEM_SUCCESS) {
                std::cerr << "Failed value in op: " << op->name() << "\n";
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    int Function::apply_constraints(real_t *const x) {
        SFEM_TRACE_SCOPE("Function::apply_constraints");

        for (auto &c : impl_->constraints) {
            c->apply(x);
        }
        return SFEM_SUCCESS;
    }

    int Function::constraints_gradient(const real_t *const x, real_t *const g) {
        SFEM_TRACE_SCOPE("Function::constraints_gradient");

        for (auto &c : impl_->constraints) {
            c->gradient(x, g);
        }
        return SFEM_SUCCESS;
    }

    int Function::apply_zero_constraints(real_t *const x) {
        SFEM_TRACE_SCOPE("Function::apply_zero_constraints");

        for (auto &c : impl_->constraints) {
            c->apply_zero(x);
        }
        return SFEM_SUCCESS;
    }

    int Function::set_value_to_constrained_dofs(const real_t val, real_t *const x) {
        SFEM_TRACE_SCOPE("Function::set_value_to_constrained_dofs");

        for (auto &c : impl_->constraints) {
            c->apply_value(val, x);
        }
        return SFEM_SUCCESS;
    }

    int Function::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        SFEM_TRACE_SCOPE("Function::copy_constrained_dofs");

        for (auto &c : impl_->constraints) {
            c->copy_constrained_dofs(src, dest);
        }
        return SFEM_SUCCESS;
    }

    int Function::report_solution(const real_t *const x) {
        SFEM_TRACE_SCOPE("Function::report_solution");

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

    std::shared_ptr<Function> Function::derefine(const std::shared_ptr<FunctionSpace> &space, const bool dirichlet_as_zero) {
        SFEM_TRACE_SCOPE("Function::derefine");
        auto ret = std::make_shared<Function>(space);

        for (auto &o : impl_->ops) {
            auto dop = o->derefine_op(space);
            if (!dop->is_no_op()) {
                ret->impl_->ops.push_back(dop);
            }
        }

        for (auto &c : impl_->constraints) {
            ret->impl_->constraints.push_back(c->derefine(space, dirichlet_as_zero));
        }

        ret->impl_->handle_constraints = impl_->handle_constraints;

        return ret;
    }

    std::shared_ptr<Function> Function::lor() { return lor(impl_->space->lor()); }
    std::shared_ptr<Function> Function::lor(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Function::lor");

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
        enum ElemType                  element_type { INVALID };

        real_t mu{1}, lambda{1};

        long   calls{0};
        double total_time{0};

        class Jacobians {
        public:
            std::shared_ptr<Buffer<jacobian_t>> adjugate;
            std::shared_ptr<Buffer<jacobian_t>> determinant;

            Jacobians(const ptrdiff_t n_elements, const int size_adjugate)
                : adjugate(sfem::create_host_buffer<jacobian_t>(n_elements * size_adjugate)),
                  determinant(sfem::create_host_buffer<jacobian_t>(n_elements)) {}
        };

        std::shared_ptr<Jacobians> jacobians;

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("LinearElasticity::create");

            auto mesh = space->mesh_ptr();
            assert(mesh->spatial_dimension() == space->block_size());

            auto ret = std::make_unique<LinearElasticity>(space);

            real_t SFEM_SHEAR_MODULUS        = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            ret->mu           = SFEM_SHEAR_MODULUS;
            ret->lambda       = SFEM_FIRST_LAME_PARAMETER;
            ret->element_type = (enum ElemType)space->element_type();

            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("LinearElasticity::lor_op");

            auto ret          = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_type_variant(element_type);
            ret->mu           = mu;
            ret->lambda       = lambda;
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("LinearElasticity::derefine_op");

            auto ret          = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->mu           = mu;
            ret->lambda       = lambda;
            return ret;
        }

        const char *name() const override { return "LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            auto mesh = space->mesh_ptr();

            if (element_type == HEX8) {
                jacobians = std::make_shared<Jacobians>(mesh->n_elements(),
                                                        mesh->spatial_dimension() * elem_manifold_dim(element_type));

                hex8_adjugate_and_det_fill(mesh->n_elements(),
                                           mesh->elements()->data(),
                                           mesh->points()->data(),
                                           jacobians->adjugate->data(),
                                           jacobians->determinant->data());
            }

            return SFEM_SUCCESS;
        }

        LinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        ~LinearElasticity() {
            if (calls) {
                printf("LinearElasticity::apply called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_crs");

            auto mesh = space->mesh_ptr();

            auto graph = space->node_to_node_graph();

            linear_elasticity_crs_aos(element_type,
                                      mesh->n_elements(),
                                      mesh->n_nodes(),
                                      mesh->elements()->data(),
                                      mesh->points()->data(),
                                      this->mu,
                                      this->lambda,
                                      graph->rowptr()->data(),
                                      graph->colidx()->data(),
                                      values);

            return SFEM_SUCCESS;
        }

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_bsr");

            auto mesh  = space->mesh_ptr();
            auto graph = space->node_to_node_graph();

            linear_elasticity_bsr(element_type,
                                  mesh->n_elements(),
                                  mesh->n_nodes(),
                                  mesh->elements()->data(),
                                  mesh->points()->data(),
                                  this->mu,
                                  this->lambda,
                                  graph->rowptr()->data(),
                                  graph->colidx()->data(),
                                  values);

            return SFEM_SUCCESS;
        }

        int hessian_bcrs_sym(const real_t *const  x,
                             const count_t *const rowptr,
                             const idx_t *const   colidx,
                             const ptrdiff_t      block_stride,
                             real_t **const       diag_values,
                             real_t **const       off_diag_values) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_bcrs_sym");

            auto mesh = space->mesh_ptr();

            linear_elasticity_bcrs_sym(element_type,
                                       mesh->n_elements(),
                                       mesh->n_nodes(),
                                       mesh->elements()->data(),
                                       mesh->points()->data(),
                                       this->mu,
                                       this->lambda,
                                       rowptr,
                                       colidx,
                                       block_stride,
                                       diag_values,
                                       off_diag_values);
            return SFEM_SUCCESS;
        }

        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym");

            auto mesh = space->mesh_ptr();
            return linear_elasticity_block_diag_sym_aos(element_type,
                                                        mesh->n_elements(),
                                                        mesh->n_nodes(),
                                                        mesh->elements()->data(),
                                                        mesh->points()->data(),
                                                        this->mu,
                                                        this->lambda,
                                                        values);
        }

        int hessian_block_diag_sym_soa(const real_t *const x, real_t **const values) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym_soa");
            auto mesh = space->mesh_ptr();
            return linear_elasticity_block_diag_sym_soa(element_type,
                                                        mesh->n_elements(),
                                                        mesh->n_nodes(),
                                                        mesh->elements()->data(),
                                                        mesh->points()->data(),
                                                        this->mu,
                                                        this->lambda,
                                                        values);
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_diag");

            auto mesh = space->mesh_ptr();

            linear_elasticity_assemble_diag_aos(element_type,
                                                mesh->n_elements(),
                                                mesh->n_nodes(),
                                                mesh->elements()->data(),
                                                mesh->points()->data(),
                                                this->mu,
                                                this->lambda,
                                                out);
            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("LinearElasticity::gradient");

            auto mesh = space->mesh_ptr();

            linear_elasticity_assemble_gradient_aos(element_type,
                                                    mesh->n_elements(),
                                                    mesh->n_nodes(),
                                                    mesh->elements()->data(),
                                                    mesh->points()->data(),
                                                    this->mu,
                                                    this->lambda,
                                                    x,
                                                    out);

            return SFEM_SUCCESS;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("LinearElasticity::apply");

            auto mesh = space->mesh_ptr();

            double tick = MPI_Wtime();

            if (jacobians) {
                SFEM_TRACE_SCOPE("linear_elasticity_apply_adjugate_aos");
                linear_elasticity_apply_adjugate_aos(element_type,
                                                     mesh->n_elements(),
                                                     mesh->n_nodes(),
                                                     mesh->elements()->data(),
                                                     jacobians->adjugate->data(),
                                                     jacobians->determinant->data(),
                                                     this->mu,
                                                     this->lambda,
                                                     h,
                                                     out);
            } else {
                SFEM_TRACE_SCOPE("linear_elasticity_apply_aos");
                linear_elasticity_apply_aos(element_type,
                                            mesh->n_elements(),
                                            mesh->n_nodes(),
                                            mesh->elements()->data(),
                                            mesh->points()->data(),
                                            this->mu,
                                            this->lambda,
                                            h,
                                            out);
            }

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_TRACE_SCOPE("LinearElasticity::value");

            auto mesh = space->mesh_ptr();

            linear_elasticity_assemble_value_aos(element_type,
                                                 mesh->n_elements(),
                                                 mesh->n_nodes(),
                                                 mesh->elements()->data(),
                                                 mesh->points()->data(),
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
        enum ElemType                  element_type { INVALID };

        real_t mu{1}, lambda{1};
        bool   use_affine_approximation{true};  // FIXME the iso-parametric version has probably a bug
        long   calls{0};
        double total_time{0};

        ~SemiStructuredLinearElasticity() {
            if (calls) {
                printf("SemiStructuredLinearElasticity[%d]::apply(%s) called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       space->semi_structured_mesh().level(),
                       use_affine_approximation ? "affine" : "isoparametric",
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("SemiStructuredLinearElasticity::create");

            assert(space->has_semi_structured_mesh());
            if (!space->has_semi_structured_mesh()) {
                fprintf(stderr,
                        "[Error] SemiStructuredLinearElasticity::create requires space with "
                        "semi_structured_mesh!\n");
                return nullptr;
            }

            assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
            auto ret = std::make_unique<SemiStructuredLinearElasticity>(space);

            real_t SFEM_SHEAR_MODULUS        = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            ret->mu           = SFEM_SHEAR_MODULUS;
            ret->lambda       = SFEM_FIRST_LAME_PARAMETER;
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
            *ret     = *this;
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            assert(false);
            fprintf(stderr, "[Error] ss:LinearElasticity::lor_op NOT IMPLEMENTED!\n");
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("SemiStructuredLinearElasticity::derefine_op");

            assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));

            if (space->has_semi_structured_mesh()) {
                auto ret                      = std::make_shared<SemiStructuredLinearElasticity>(space);
                ret->element_type             = element_type;
                ret->use_affine_approximation = use_affine_approximation;
                ret->mu                       = mu;
                ret->lambda                   = lambda;
                // ret->initialize();
                return ret;
            } else {
                assert(space->element_type() == macro_base_elem(element_type));
                auto ret          = std::make_shared<LinearElasticity>(space);
                ret->element_type = macro_base_elem(element_type);
                ret->mu           = mu;
                ret->lambda       = lambda;
                ret->initialize();
                return ret;
            }
        }

        const char *name() const override { return "ss:LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        SemiStructuredLinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            assert(false);
            return SFEM_FAILURE;
        }

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::hessian_bsr", ssm.level());

            return affine_sshex8_elasticity_bsr(ssm.level(),
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
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::hessian_diag", ssm.level());

            return affine_sshex8_linear_elasticity_diag(ssm.level(),
                                                        ssm.n_elements(),
                                                        ssm.interior_start(),
                                                        ssm.element_data(),
                                                        ssm.point_data(),
                                                        mu,
                                                        lambda,
                                                        3,
                                                        &out[0],
                                                        &out[1],
                                                        &out[2]);
        }

        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::hessian_block_diag_sym", ssm.level());

            return affine_sshex8_linear_elasticity_block_diag_sym(ssm.level(),
                                                                  ssm.n_elements(),
                                                                  ssm.interior_start(),
                                                                  ssm.element_data(),
                                                                  ssm.point_data(),
                                                                  mu,
                                                                  lambda,
                                                                  6,
                                                                  &values[0],
                                                                  &values[1],
                                                                  &values[2],
                                                                  &values[3],
                                                                  &values[4],
                                                                  &values[5]);
        }

        int gradient(const real_t *const x, real_t *const out) override { return apply(nullptr, x, out); }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::apply", ssm.level());

            assert(element_type == SSHEX8);  // REMOVEME once generalized approach

            calls++;

            double tick = MPI_Wtime();
            int    err;
            if (use_affine_approximation) {
                err = affine_sshex8_linear_elasticity_apply(ssm.level(),
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
                err = sshex8_linear_elasticity_apply(ssm.level(),
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
        enum ElemType                  element_type { INVALID };

        long   calls{0};
        double total_time{0};

        const char *name() const override { return "Laplacian"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("Laplacian::create");

            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            assert(1 == space->block_size());

            auto ret          = std::make_unique<Laplacian>(space);
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret          = std::make_shared<Laplacian>(space);
            ret->element_type = macro_type_variant(element_type);
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret          = std::make_shared<Laplacian>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        Laplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        ~Laplacian() {
            if (calls) {
                printf("Laplacian::apply called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("Laplacian::hessian_crs");

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

        int hessian_crs_sym(const real_t *const  x,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        diag_values,
                            real_t *const        off_diag_values) override {
            SFEM_TRACE_SCOPE("Laplacian::hessian_crs_sym");

            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // auto graph = space->node_to_node_graph_upper_triangular();

            return laplacian_crs_sym(element_type,
                                     mesh->nelements,
                                     mesh->nnodes,
                                     mesh->elements,
                                     mesh->points,
                                     rowptr,
                                     colidx,
                                     diag_values,
                                     off_diag_values);
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            SFEM_TRACE_SCOPE("Laplacian::hessian_diag");

            auto mesh = space->mesh_ptr();
            return laplacian_diag(
                    element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), values);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("Laplacian::gradient");

            auto mesh = space->mesh_ptr();
            return laplacian_assemble_gradient(
                    element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), x, out);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("Laplacian::apply");

            auto   mesh = space->mesh_ptr();
            double tick = MPI_Wtime();

            int err = laplacian_apply(
                    element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), h, out);

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;
            return err;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_TRACE_SCOPE("Laplacian::value");

            auto mesh = space->mesh_ptr();
            return laplacian_assemble_value(
                    element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), x, out);
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class VectorLaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };

        std::shared_ptr<Buffer<jacobian_t>> fff;

        long   calls{0};
        double total_time{0};

        const char *name() const override { return "VectorLaplacian"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("VectorLaplacian::create");

            assert(1 != space->block_size());
            auto ret          = std::make_unique<VectorLaplacian>(space);
            ret->element_type = (enum ElemType)space->element_type();

            int SFEM_VECTOR_LAPLACIAN_FFF = 1;
            SFEM_READ_ENV(SFEM_VECTOR_LAPLACIAN_FFF, atoi);

            if (SFEM_VECTOR_LAPLACIAN_FFF) {
                ret->fff = create_host_buffer<jacobian_t>(space->mesh_ptr()->n_elements() * 6);

                if (SFEM_SUCCESS != hex8_fff_fill(space->mesh_ptr()->n_elements(),
                                                  space->mesh_ptr()->elements()->data(),
                                                  space->mesh_ptr()->points()->data(),
                                                  ret->fff->data())) {
                    SFEM_ERROR("Unable to create fff");
                }
            }

            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret          = std::make_shared<VectorLaplacian>(space);
            ret->element_type = macro_type_variant(element_type);
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            auto ret          = std::make_shared<VectorLaplacian>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        VectorLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        ~VectorLaplacian() {
            if (calls) {
                printf("VectorLaplacian::apply called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs");
            SFEM_ERROR("IMPLEMENT ME!\n");
            return SFEM_FAILURE;
        }

        int hessian_crs_sym(const real_t *const  x,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        diag_values,
                            real_t *const        off_diag_values) override {
            SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs_sym");
            SFEM_ERROR("IMPLEMENT ME!\n");
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            SFEM_TRACE_SCOPE("VectorLaplacian::hessian_diag");
            SFEM_ERROR("IMPLEMENT ME!\n");
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("VectorLaplacian::gradient");
            SFEM_ERROR("IMPLEMENT ME!\n");
            return SFEM_FAILURE;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("VectorLaplacian::apply");

            double tick = MPI_Wtime();

            const int             block_size = space->block_size();
            std::vector<real_t *> vec_in(block_size), vec_out(block_size);

            // AoS
            for (int d = 0; d < block_size; d++) {
                vec_in[d]  = const_cast<real_t *>(&h[d]);
                vec_out[d] = &out[d];
            }

            auto mesh = space->mesh_ptr();
            int  err;
            if (this->fff) {
                err = affine_hex8_vector_laplacian_apply_fff(
                        // element_type,
                        mesh->n_elements(),
                        mesh->elements()->data(),
                        this->fff->data(),
                        block_size,
                        block_size,
                        vec_in.data(),
                        vec_out.data());
            } else {
                err = affine_hex8_vector_laplacian_apply(
                        // element_type,
                        mesh->n_elements(),
                        mesh->n_nodes(),
                        mesh->elements()->data(),
                        mesh->points()->data(),
                        block_size,
                        block_size,
                        vec_in.data(),
                        vec_out.data());
            }

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;
            return err;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_TRACE_SCOPE("VectorLaplacian::value");
            SFEM_ERROR("IMPLEMENT ME!\n");
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class SemiStructuredVectorLaplacian : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };

        std::shared_ptr<Buffer<jacobian_t>> fff;

        long   calls{0};
        double total_time{0};

        ~SemiStructuredVectorLaplacian() {
            if (calls) {
                printf("SemiStructuredVectorLaplacian[%d]::apply called %ld times. "
                       "Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       space->semi_structured_mesh().level(),
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("SemiStructuredVectorLaplacian::create");

            assert(space->has_semi_structured_mesh());
            if (!space->has_semi_structured_mesh()) {
                fprintf(stderr,
                        "[Error] SemiStructuredVectorLaplacian::create requires space with "
                        "semi_structured_mesh!\n");
                return nullptr;
            }

            assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
            auto ret = std::make_unique<SemiStructuredVectorLaplacian>(space);

            ret->element_type = (enum ElemType)space->element_type();

            ret->fff = create_host_buffer<jacobian_t>(space->mesh_ptr()->n_elements() * 6);

            if (SFEM_SUCCESS != hex8_fff_fill(space->mesh_ptr()->n_elements(),
                                              space->mesh_ptr()->elements()->data(),
                                              space->mesh_ptr()->points()->data(),
                                              ret->fff->data())) {
                SFEM_ERROR("Unable to create fff");
            }

            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            fprintf(stderr, "[Error] ss:Laplacian::lor_op NOT IMPLEMENTED!\n");
            assert(false);
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("SemiStructuredVectorLaplacian::derefine_op");

            assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
            if (space->has_semi_structured_mesh()) {
                auto ret          = std::make_shared<SemiStructuredVectorLaplacian>(space);
                ret->element_type = element_type;
                return ret;
            } else {
                auto ret          = std::make_shared<VectorLaplacian>(space);
                ret->element_type = macro_base_elem(element_type);
                return ret;
            }
        }

        const char *name() const override { return "ss:Laplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        SemiStructuredVectorLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_ERROR("[Error] ss:Laplacian::hessian_crs NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_ERROR("[Error] ss:Laplacian::hessian_diag NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_ERROR("[Error] ss:Laplacian::gradient NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("SemiStructuredVectorLaplacian::apply");

            assert(element_type == SSHEX8);  // REMOVEME once generalized approach

            auto &ssm = space->semi_structured_mesh();

            double tick = MPI_Wtime();

            const int             block_size = space->block_size();
            std::vector<real_t *> vec_in(block_size), vec_out(block_size);

            // AoS
            for (int d = 0; d < block_size; d++) {
                vec_in[d]  = const_cast<real_t *>(&h[d]);
                vec_out[d] = &out[d];
            }

            int err = affine_sshex8_vector_laplacian_apply_fff(ssm.level(),
                                                               ssm.n_elements(),
                                                               ssm.element_data(),
                                                               this->fff->data(),
                                                               block_size,
                                                               block_size,
                                                               vec_in.data(),
                                                               vec_out.data());

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;
            return err;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_ERROR("[Error] ss:Laplacian::value NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class SemiStructuredLaplacian : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };
        bool                           use_affine_approximation{true};
        bool                           use_stencil{true};

        std::shared_ptr<Buffer<jacobian_t>> fff;

        long   calls{0};
        double total_time{0};

        ~SemiStructuredLaplacian() {
            if (calls) {
                printf("SemiStructuredLaplacian[%d]::apply(%s) called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       space->semi_structured_mesh().level(),
                       use_affine_approximation ? (use_stencil ? "stencil" : "affine") : "isoparametric",
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("SemiStructuredLaplacian::create");

            assert(space->has_semi_structured_mesh());
            if (!space->has_semi_structured_mesh()) {
                fprintf(stderr,
                        "[Error] SemiStructuredLaplacian::create requires space with "
                        "semi_structured_mesh!\n");
                return nullptr;
            }

            assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
            auto ret = std::make_unique<SemiStructuredLaplacian>(space);

            ret->element_type = (enum ElemType)space->element_type();

            int SFEM_HEX8_ASSUME_AFFINE = ret->use_affine_approximation;
            SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
            ret->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

            int SFEM_ENABLE_HEX8_STENCIL = ret->use_stencil;
            SFEM_READ_ENV(SFEM_ENABLE_HEX8_STENCIL, atoi);
            ret->use_stencil = SFEM_ENABLE_HEX8_STENCIL;

            int SFEM_SS_LAPLACIAN_FFF = 1;
            SFEM_READ_ENV(SFEM_SS_LAPLACIAN_FFF, atoi);

            if (SFEM_SS_LAPLACIAN_FFF) {
                ret->fff = create_host_buffer<jacobian_t>(space->mesh_ptr()->n_elements() * 6);

                if (SFEM_SUCCESS != hex8_fff_fill(space->mesh_ptr()->n_elements(),
                                                  space->mesh_ptr()->elements()->data(),
                                                  space->mesh_ptr()->points()->data(),
                                                  ret->fff->data())) {
                    SFEM_ERROR("Unable to create fff");
                }
            }

            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            fprintf(stderr, "[Error] ss:Laplacian::lor_op NOT IMPLEMENTED!\n");
            assert(false);
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("SemiStructuredLaplacian::derefine_op");

            assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
            if (space->has_semi_structured_mesh()) {
                auto ret                      = std::make_shared<SemiStructuredLaplacian>(space);
                ret->element_type             = element_type;
                ret->use_affine_approximation = use_affine_approximation;
                ret->use_stencil              = use_stencil;
                return ret;
            } else {
                auto ret          = std::make_shared<Laplacian>(space);
                ret->element_type = macro_base_elem(element_type);
                return ret;
            }
        }

        const char *name() const override { return "ss:Laplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        SemiStructuredLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_ERROR("[Error] ss:Laplacian::hessian_crs NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_TRACE_SCOPE("SemiStructuredLaplacian::hessian_diag");

            auto &ssm = space->semi_structured_mesh();
            return affine_sshex8_laplacian_diag(
                    ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), out);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("SemiStructuredLaplacian::gradient");
            return apply(nullptr, x, out);
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("SemiStructuredLaplacian::apply");

            assert(element_type == SSHEX8);  // REMOVEME once generalized approach

            auto &ssm = space->semi_structured_mesh();

            double tick = MPI_Wtime();

            int err = 0;

            if (this->fff) {
                SFEM_TRACE_SCOPE("affine_sshex8_laplacian_stencil_apply_fff");
                affine_sshex8_laplacian_stencil_apply_fff(
                        ssm.level(), ssm.n_elements(), ssm.element_data(), this->fff->data(), h, out);
            } else {
                if (use_stencil) {
                    err = affine_sshex8_laplacian_stencil_apply(
                            ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);
                } else if (use_affine_approximation) {
                    err = affine_sshex8_laplacian_apply(
                            ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);

                } else {
                    err = sshex8_laplacian_apply(
                            ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);
                }
            }

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;
            return err;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_ERROR("[Error] ss:Laplacian::value NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class SpectralElementLaplacian : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };

        long   calls{0};
        double total_time{0};

        ~SpectralElementLaplacian() {
            if (calls) {
                printf("SpectralElementLaplacian[%d]::apply called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       space->semi_structured_mesh().level(),
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("SpectralElementLaplacian::create");

            assert(space->has_semi_structured_mesh());
            if (!space->has_semi_structured_mesh()) {
                fprintf(stderr,
                        "[Error] SpectralElementLaplacian::create requires space with "
                        "semi_structured_mesh!\n");
                return nullptr;
            }

            assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
            auto ret          = std::make_unique<SpectralElementLaplacian>(space);
            ret->element_type = (enum ElemType)space->element_type();

            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            fprintf(stderr, "[Error] SpectralElementLaplacian::lor_op NOT IMPLEMENTED!\n");
            assert(false);
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("SpectralElementLaplacian::derefine_op");

            assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
            if (space->has_semi_structured_mesh()) {
                auto ret          = std::make_shared<SpectralElementLaplacian>(space);
                ret->element_type = element_type;
                return ret;
            } else {
                auto ret          = std::make_shared<Laplacian>(space);
                ret->element_type = macro_base_elem(element_type);
                return ret;
            }
        }

        const char *name() const override { return "ss:SpectralElementLaplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        SpectralElementLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_ERROR("[Error] SpectralElementLaplacian::hessian_crs NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_TRACE_SCOPE("SpectralElementLaplacian::hessian_diag");
            SFEM_ERROR("[Error] SpectralElementLaplacian::hessian_diag NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_ERROR("[Error] SpectralElementLaplacian::gradient NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("SpectralElementLaplacian::apply");

            assert(element_type == SSHEX8);  // REMOVEME once generalized approach

            auto &ssm = space->semi_structured_mesh();

            double tick = MPI_Wtime();

            int err = spectral_hex_laplacian_apply(
                    ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;
            return err;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_ERROR("[Error] SpectralElementLaplacian::value NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class SemiStructuredEMLaplacian : public Op {
    public:
        std::shared_ptr<FunctionSpace>    space;
        enum ElemType                     element_type { INVALID };
        std::shared_ptr<Buffer<scalar_t>> element_matrix;

        long   calls{0};
        double total_time{0};

        ~SemiStructuredEMLaplacian() {
            if (calls) {
                printf("SemiStructuredEMLaplacian[%d]::apply() called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       space->semi_structured_mesh().level(),
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::create");

            assert(space->has_semi_structured_mesh());
            if (!space->has_semi_structured_mesh()) {
                fprintf(stderr,
                        "[Error] SemiStructuredEMLaplacian::create requires space with "
                        "semi_structured_mesh!\n");
                return nullptr;
            }

            assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
            auto ret          = std::make_unique<SemiStructuredEMLaplacian>(space);
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            fprintf(stderr, "[Error] SemiStructuredEMLaplacian::lor_op NOT IMPLEMENTED!\n");
            assert(false);
            return nullptr;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::derefine_op");

            assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
            if (space->has_semi_structured_mesh()) {
                auto ret          = std::make_shared<SemiStructuredEMLaplacian>(space);
                ret->element_type = element_type;
                // FIXME every level stores a variatin of it with different scaling
                // It woud be usefull to revisit
                // ret->element_matrix = element_matrix;
                ret->initialize();
                return ret;
            } else {
                auto ret          = std::make_shared<Laplacian>(space);
                ret->element_type = macro_base_elem(element_type);
                return ret;
            }
        }

        const char *name() const override { return "ss:em:Laplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            auto &ssm      = space->semi_structured_mesh();
            auto  mesh     = space->mesh_ptr();
            element_matrix = sfem::create_host_buffer<real_t>(mesh->n_elements() * 64);
            return sshex8_laplacian_element_matrix(ssm.level(),
                                                   mesh->n_elements(),
                                                   mesh->n_nodes(),
                                                   mesh->elements()->data(),
                                                   mesh->points()->data(),
                                                   element_matrix->data());

            // element_matrix->print(std::cout);
            return SFEM_SUCCESS;
        }

        SemiStructuredEMLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_ERROR("[Error] ss:em:Laplacian::hessian_crs NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_TRACE_SCOPE("SemiStructuredLaplacian::hessian_diag");

            auto &ssm = space->semi_structured_mesh();
            return affine_sshex8_laplacian_diag(
                    ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), out);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_ERROR("[Error] ss:em:Laplacian::gradient NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::apply");

            assert(element_type == SSHEX8);  // REMOVEME once generalized approach

            auto &ssm = space->semi_structured_mesh();

            double tick = MPI_Wtime();

            int err = sshex8_stencil_element_matrix_apply(
                    ssm.level(), ssm.n_elements(), ssm.element_data(), element_matrix->data(), h, out);

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;
            return err;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_ERROR("[Error] ss:em:Laplacian::value NOT IMPLEMENTED!\n");
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class Mass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };

        const char *name() const override { return "Mass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("Mass::create");

            assert(1 == space->block_size());

            auto ret          = std::make_unique<Mass>(space);
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        Mass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("Mass::hessian_crs");

            auto mesh  = space->mesh_ptr();
            auto graph = space->dof_to_dof_graph();

            assemble_mass(element_type,
                          mesh->n_elements(),
                          mesh->n_nodes(),
                          mesh->elements()->data(),
                          mesh->points()->data(),
                          graph->rowptr()->data(),
                          graph->colidx()->data(),
                          values);

            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("Mass::gradient");

            auto mesh = space->mesh_ptr();

            apply_mass(element_type,
                       mesh->n_elements(),
                       mesh->n_nodes(),
                       mesh->elements()->data(),
                       mesh->points()->data(),
                       1,
                       x,
                       1,
                       out);

            return SFEM_SUCCESS;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("Mass::apply");

            auto mesh = space->mesh_ptr();

            apply_mass(element_type,
                       mesh->n_elements(),
                       mesh->n_nodes(),
                       mesh->elements()->data(),
                       mesh->points()->data(),
                       1,
                       h,
                       1,
                       out);

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
            // auto mesh = space->mesh_ptr();

            // mass_assemble_value((enum ElemType)space->element_type(),
            //                     mesh->n_elements(),
            //                     mesh->n_nodes(),
            //                     mesh->elements()->data(),
            //                     mesh->points()->data(),
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
        enum ElemType                  element_type { INVALID };

        const char *name() const override { return "LumpedMass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("LumpedMass::create");

            auto ret          = std::make_unique<LumpedMass>(space);
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        LumpedMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            SFEM_TRACE_SCOPE("LumpedMass::hessian_diag");

            auto mesh = space->mesh_ptr();

            if (space->block_size() == 1) {
                assemble_lumped_mass(element_type,
                                     mesh->n_elements(),
                                     mesh->n_nodes(),
                                     mesh->elements()->data(),
                                     mesh->points()->data(),
                                     values);
            } else {
                const ptrdiff_t n_nodes = mesh->n_nodes();
                real_t         *temp    = (real_t *)calloc(n_nodes, sizeof(real_t));
                assemble_lumped_mass(
                        element_type, mesh->n_elements(), n_nodes, mesh->elements()->data(), mesh->points()->data(), temp);

                int bs = space->block_size();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_nodes; i++) {
                    for (int b = 0; b < bs; b++) {
                        values[i * bs + b] += temp[i];
                    }
                }

                free(temp);
            }

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
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

    class SemiStructuredLumpedMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };

        const char *name() const override { return "ss:LumpedMass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("SemiStructuredLumpedMass::create");

            if (!space->has_semi_structured_mesh()) {
                SFEM_ERROR(
                        "[Error] SemiStructuredLumpedMass::create requires space with "
                        "semi_structured_mesh!\n");
            }

            assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
            auto ret          = std::make_unique<SemiStructuredLumpedMass>(space);
            ret->element_type = (enum ElemType)space->element_type();

            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("SemiStructuredLumpedMass::derefine_op");

            assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
            if (space->has_semi_structured_mesh()) {
                auto ret          = std::make_shared<SemiStructuredLumpedMass>(space);
                ret->element_type = element_type;
                return ret;
            } else {
                auto ret          = std::make_shared<LumpedMass>(space);
                ret->element_type = macro_base_elem(element_type);
                return ret;
            }
        }

        int initialize() override { return SFEM_SUCCESS; }

        SemiStructuredLumpedMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            SFEM_TRACE_SCOPE("SemiStructuredLumpedMass::hessian_diag");

            auto &ssm = space->semi_structured_mesh();
            if (space->block_size() == 1) {
                return affine_sshex8_mass_lumped(
                        ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), values);
            } else {
                const ptrdiff_t n = space->n_dofs() / space->block_size();

                auto    buff = create_host_buffer<real_t>(n);
                real_t *temp = buff->data();
                int     err  = affine_sshex8_mass_lumped(
                        ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), temp);

                if (err) SFEM_ERROR("Failure in affine_sshex8_mass_lumped\n");

                int bs = space->block_size();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    for (int b = 0; b < bs; b++) {
                        values[i * bs + b] += temp[i];
                    }
                }
            }

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
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
        enum ElemType                  element_type { INVALID };

        const char *name() const override { return "CVFEMMass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(1 == space->block_size());

            auto ret          = std::make_unique<CVFEMMass>(space);
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        CVFEMMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            auto mesh = space->mesh_ptr();
            cvfem_cv_volumes(
                    element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), values);

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
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
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<Buffer<real_t>> vel[3];
        enum ElemType                   element_type { INVALID };

        const char *name() const override { return "CVFEMUpwindConvection"; }
        inline bool is_linear() const override { return true; }

        void set_field(const char * /* name  = velocity */,
                       const std::shared_ptr<Buffer<real_t>> &v,
                       const int                              component) override {
            vel[component] = v;
        }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = space->mesh_ptr();

            assert(1 == space->block_size());

            auto ret = std::make_unique<CVFEMUpwindConvection>(space);

            const char *SFEM_VELX = nullptr;
            const char *SFEM_VELY = nullptr;
            const char *SFEM_VELZ = nullptr;

            SFEM_READ_ENV(SFEM_VELX, );
            SFEM_READ_ENV(SFEM_VELY, );
            SFEM_READ_ENV(SFEM_VELZ, );

            if (!SFEM_VELX || !SFEM_VELY || (!SFEM_VELZ && mesh->spatial_dimension() == 3)) {
                // fprintf(stderr,
                //         "No input velocity in env: SFEM_VELX=%s\n,SFEM_VELY=%s\n,SFEM_VELZ=%s\n",
                //         SFEM_VELX,
                //         SFEM_VELY,
                //         SFEM_VELZ);
                ret->element_type = (enum ElemType)space->element_type();
                return ret;
            }

            ptrdiff_t nlocal, nglobal;

            real_t *vel0, *vel1, *vel2;
            if (array_create_from_file(mesh->comm(), SFEM_VELX, SFEM_MPI_REAL_T, (void **)&vel0, &nlocal, &nglobal) ||
                array_create_from_file(mesh->comm(), SFEM_VELY, SFEM_MPI_REAL_T, (void **)&vel1, &nlocal, &nglobal) ||
                array_create_from_file(mesh->comm(), SFEM_VELZ, SFEM_MPI_REAL_T, (void **)&vel2, &nlocal, &nglobal)) {
                fprintf(stderr, "Unable to read input velocity\n");
                assert(0);
                return nullptr;
            }

            ret->vel[0] = sfem::manage_host_buffer<real_t>(nlocal, vel0);
            ret->vel[1] = sfem::manage_host_buffer<real_t>(nlocal, vel1);
            ret->vel[2] = sfem::manage_host_buffer<real_t>(nlocal, vel2);

            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        CVFEMUpwindConvection(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        ~CVFEMUpwindConvection() {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            // auto mesh = space->mesh_ptr();

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

            SFEM_ERROR("IMPLEMENT ME");
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            return apply(nullptr, x, out);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto mesh = space->mesh_ptr();

            real_t *vel_[3] = {nullptr, nullptr, nullptr};

            for (int i = 0; i < 3; i++) {
                if (vel[i]) {
                    vel_[i] = vel[i]->data();
                }
            }

            cvfem_convection_apply(element_type,
                                   mesh->n_elements(),
                                   mesh->n_nodes(),
                                   mesh->elements()->data(),
                                   mesh->points()->data(),
                                   vel_,
                                   h,
                                   out);

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
            // auto mesh = space->mesh_ptr();

            // cvfem_convection_assemble_value(element_type,
            //                          mesh->nelements,
            //                          mesh->nnodes,
            //                          mesh->elements,
            //                          mesh->points,
            //                          x,
            //                          out);

            // return SFEM_SUCCESS;

            SFEM_ERROR("IMPLEMENT ME");
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    //

    class NeoHookeanOgden final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };

        real_t mu{1}, lambda{1};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::create");

            assert(space->mesh_ptr()->spatial_dimension() == space->block_size());

            auto ret = std::make_unique<NeoHookeanOgden>(space);

            real_t SFEM_SHEAR_MODULUS        = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            ret->mu           = SFEM_SHEAR_MODULUS;
            ret->lambda       = SFEM_FIRST_LAME_PARAMETER;
            ret->element_type = (enum ElemType)space->element_type();
            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::lor_op");

            auto ret          = std::make_shared<NeoHookeanOgden>(space);
            ret->element_type = macro_type_variant(element_type);
            ret->mu           = mu;
            ret->lambda       = lambda;
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::derefine_op");

            auto ret          = std::make_shared<NeoHookeanOgden>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->mu           = mu;
            ret->lambda       = lambda;
            return ret;
        }

        const char *name() const override { return "NeoHookeanOgden"; }
        inline bool is_linear() const override { return true; }

        int initialize() override { return SFEM_SUCCESS; }

        NeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_crs");

            auto mesh  = space->mesh_ptr();
            auto graph = space->node_to_node_graph();

            return neohookean_ogden_hessian_aos(element_type,
                                                mesh->n_elements(),
                                                mesh->n_nodes(),
                                                mesh->elements()->data(),
                                                mesh->points()->data(),
                                                this->mu,
                                                this->lambda,
                                                x,
                                                graph->rowptr()->data(),
                                                graph->colidx()->data(),
                                                values);
        }

        int hessian_diag(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_diag");

            auto mesh = space->mesh_ptr();

            return neohookean_ogden_diag_aos(element_type,
                                             mesh->n_elements(),
                                             mesh->n_nodes(),
                                             mesh->elements()->data(),
                                             mesh->points()->data(),
                                             this->mu,
                                             this->lambda,
                                             x,
                                             out);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::gradient");

            auto mesh = space->mesh_ptr();

            return neohookean_ogden_gradient_aos(element_type,
                                                 mesh->n_elements(),
                                                 mesh->n_nodes(),
                                                 mesh->elements()->data(),
                                                 mesh->points()->data(),
                                                 this->mu,
                                                 this->lambda,
                                                 x,
                                                 out);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::apply");

            auto mesh = space->mesh_ptr();

            return neohookean_ogden_apply_aos(element_type,
                                              mesh->n_elements(),
                                              mesh->n_nodes(),
                                              mesh->elements()->data(),
                                              mesh->points()->data(),
                                              this->mu,
                                              this->lambda,
                                              x,
                                              h,
                                              out);
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_TRACE_SCOPE("NeoHookeanOgden::value");

            auto mesh = space->mesh_ptr();

            return neohookean_ogden_value_aos(element_type,
                                              mesh->n_elements(),
                                              mesh->n_nodes(),
                                              mesh->elements()->data(),
                                              mesh->points()->data(),
                                              this->mu,
                                              this->lambda,
                                              x,
                                              out);
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    class BoundaryMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace>   space;
        std::shared_ptr<Buffer<idx_t *>> boundary_elements;
        enum ElemType                    element_type { INVALID };

        const char *name() const override { return "BoundaryMass"; }
        inline bool is_linear() const override { return true; }

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace>   &space,
                                          const std::shared_ptr<Buffer<idx_t *>> &boundary_elements) {
            SFEM_TRACE_SCOPE("BoundaryMass::create");

            auto ret          = std::make_unique<BoundaryMass>(space);
            auto element_type = (enum ElemType)space->element_type();
            ret->element_type = shell_type(side_type(element_type));
            if (ret->element_type == INVALID) {
                std::cerr << "Invalid element type for BoundaryMass, Bulk element type: " << type_to_string(element_type) << "\n";
                return nullptr;
            }
            ret->boundary_elements = boundary_elements;
            return ret;
        }

        int initialize() override { return SFEM_SUCCESS; }

        BoundaryMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
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
            SFEM_TRACE_SCOPE("BoundaryMass::apply");

            auto mesh = space->mesh_ptr();

            int  block_size = space->block_size();
            auto data       = boundary_elements->data();

            for (int d = 0; d < block_size; d++) {
                apply_mass(element_type,
                           boundary_elements->extent(1),
                           mesh->n_nodes(),
                           boundary_elements->data(),
                           mesh->points()->data(),
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
        std::map<std::string, FactoryFunction>         name_to_create;
        std::map<std::string, FactoryFunctionBoundary> name_to_create_boundary;
    };

    Factory::Factory() : impl_(std::make_unique<Impl>()) {}

    Factory::~Factory() = default;

    Factory &Factory::instance() {
        static Factory instance_;

        if (instance_.impl_->name_to_create.empty()) {
            instance_.private_register_op("KelvinVoigtNewmark", create_kelvin_voigt_newmark);
            instance_.private_register_op("LinearElasticity", LinearElasticity::create);
            instance_.private_register_op("ss:LinearElasticity", SemiStructuredLinearElasticity::create);
            instance_.private_register_op("Laplacian", Laplacian::create);
            instance_.private_register_op("VectorLaplacian", VectorLaplacian::create);
            instance_.private_register_op("ss:VectorLaplacian", SemiStructuredVectorLaplacian::create);
            instance_.private_register_op("ss:Laplacian", SemiStructuredLaplacian::create);
            instance_.private_register_op("ss:LumpedMass", SemiStructuredLumpedMass::create);
            instance_.private_register_op("ss:em:Laplacian", SemiStructuredEMLaplacian::create);
            instance_.private_register_op("ss:SpectralElementLaplacian", SpectralElementLaplacian::create);
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

    std::shared_ptr<Op> Factory::create_op_gpu(const std::shared_ptr<FunctionSpace> &space, const char *name) {
        return Factory::create_op(space, d_op_str(name).c_str());
    }

    std::shared_ptr<Op> Factory::create_op(const std::shared_ptr<FunctionSpace> &space, const char *name) {
        assert(instance().impl_);

        std::string m_name = name;

        if (space->has_semi_structured_mesh()) {
            m_name = "ss:" + m_name;
        }

        auto &ntc = instance().impl_->name_to_create;
        auto  it  = ntc.find(m_name);

        if (it == ntc.end()) {
            std::cerr << "Unable to find op " << m_name << "\n";
            return nullptr;
        }

        return it->second(space);
    }

    std::shared_ptr<Op> Factory::create_boundary_op(const std::shared_ptr<FunctionSpace>   &space,
                                                    const std::shared_ptr<Buffer<idx_t *>> &boundary_elements,
                                                    const char                             *name) {
        assert(instance().impl_);

        auto &ntc = instance().impl_->name_to_create_boundary;
        auto  it  = ntc.find(name);

        if (it == ntc.end()) {
            std::cerr << "Unable to find op " << name << "\n";
            return nullptr;
        }

        return it->second(space, boundary_elements);
    }

    std::string d_op_str(const std::string &name) { return "gpu:" + name; }

    std::shared_ptr<Buffer<idx_t *>> mesh_connectivity_from_file(MPI_Comm comm, const char *folder) {
        char pattern[1024 * 10];
        sprintf(pattern, "%s/i*.raw", folder);

        std::shared_ptr<Buffer<idx_t *>> ret;

        auto files   = sfem::find_files(pattern);
        int  n_files = files.size();

        idx_t **data = (idx_t **)malloc(n_files * sizeof(idx_t *));

        ptrdiff_t local_size = SFEM_PTRDIFF_INVALID;
        ptrdiff_t size       = SFEM_PTRDIFF_INVALID;

        printf("n_files (%d):\n", n_files);
        int err = 0;
        for (int np = 0; np < n_files; np++) {
            printf("%s\n", files[np].c_str());

            char path[1024 * 10];
            sprintf(path, "%s/i%d.raw", folder, np);

            idx_t *idx = 0;
            err |= array_create_from_file(comm, path, SFEM_MPI_IDX_T, (void **)&idx, &local_size, &size);

            data[np] = idx;
        }

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
