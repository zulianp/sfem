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
#include "integrate_values.h"
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

        auto space = impl_->space;
        auto mesh  = space->mesh_ptr();
        auto et    = (enum ElemType)space->element_type();

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
                    max_coarse_idx = max_node_id(coarse_space->element_type(), mesh->n_elements(), mesh->elements()->data());

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

        auto comm = space->mesh_ptr()->comm();
        int      rank = comm->rank();

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
                    if (array_create_from_file(comm->get(), pch, SFEM_MPI_IDX_T, (void **)&this_set, &lsize, &gsize)) {
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
                    if (array_create_from_file(comm->get(), pch + path_key_len, SFEM_MPI_REAL_T, (void **)&values, &lsize, &gsize)) {
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

        MPI_Comm comm = space->mesh_ptr()->comm()->get();

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
                    if (!array_create_from_file(comm->get(), path.c_str(), SFEM_MPI_IDX_T, (void **)&arr, &lsize, &gsize)) {
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

        MPI_Comm comm = impl_->space->mesh_ptr()->comm()->get();
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
                snprintf(b_name, sizeof(b_name), "%s.%d", name, b);
                snprintf(path, sizeof(path), impl_->file_format.c_str(), impl_->output_dir.c_str(), b_name);
                if (array_write(comm, path, SFEM_MPI_REAL_T, buff->data(), n_blocks, n_blocks)) {
                    return SFEM_FAILURE;
                }
            }

        } else {
            char path[2048];
            snprintf(path, sizeof(path), impl_->file_format.c_str(), impl_->output_dir.c_str(), name);
            if (array_write(comm, path, SFEM_MPI_REAL_T, x, impl_->space->n_dofs(), impl_->space->n_dofs())) {
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    void Output::log_time(const real_t t) {
        if (log_is_empty(&impl_->time_logger)) {
            char path[2048];
            snprintf(path, sizeof(path), "%s/time.txt", impl_->output_dir.c_str());
            log_create_file(&impl_->time_logger, path, "w");
        }

        log_write_double(&impl_->time_logger, t);
    }

    int Output::write_time_step(const char *name, const real_t t, const real_t *const x) {
        SFEM_TRACE_SCOPE("Output::write_time_step");

        auto      space      = impl_->space;
        auto      mesh       = space->mesh_ptr();
        const int block_size = space->block_size();

        char path[2048];

        if (impl_->AoS_to_SoA && block_size > 1) {
            ptrdiff_t n_blocks = space->n_dofs() / block_size;

            auto buff = create_host_buffer<real_t>(n_blocks);
            auto bb   = buff->data();

            for (int b = 0; b < block_size; b++) {
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    bb[i] = x[i * block_size + b];
                }

                char b_name[1024];
                snprintf(b_name, sizeof(b_name), "%s.%d", name, b);
                snprintf(path,
                         sizeof(path),
                         impl_->time_dependent_file_format.c_str(),
                         impl_->output_dir.c_str(),
                         b_name,
                         impl_->export_counter++);

                if (array_write(mesh->comm()->get(), path, SFEM_MPI_REAL_T, buff->data(), n_blocks, n_blocks)) {
                    return SFEM_FAILURE;
                }
            }

        } else {
            sfem::create_directory(impl_->output_dir.c_str());

            snprintf(path,
                     sizeof(path),
                     impl_->time_dependent_file_format.c_str(),
                     impl_->output_dir.c_str(),
                     name,
                     impl_->export_counter++);

            if (array_write(mesh->comm()->get(), path, SFEM_MPI_REAL_T, x, space->n_dofs(), space->n_dofs())) {
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

    std::shared_ptr<Buffer<idx_t *>> mesh_connectivity_from_file(const std::shared_ptr<Communicator>& comm, const char *folder) {
        char pattern[1024 * 10];
        snprintf(pattern, sizeof(pattern), "%s/i*.raw", folder);

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
            snprintf(path, sizeof(path), "%s/i%d.raw", folder, np);

            idx_t *idx = 0;
            err |= array_create_from_file(comm->get(), path, SFEM_MPI_IDX_T, (void **)&idx, &local_size, &size);

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
