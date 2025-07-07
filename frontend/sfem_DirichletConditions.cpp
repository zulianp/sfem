#include "sfem_DirichletConditions.hpp"

#include <stddef.h>

#include "matrixio_array.h"
#include "utils.h"
#include "sfem_Function.hpp"
#include "boundary_condition.h"
#include "operators/hierarchical/sfem_prolongation_restriction.h"
#include "operators/boundary_conditions/dirichlet.h"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include <sys/stat.h>
#include <cstddef>
#include <fstream>
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

}  // namespace sfem 