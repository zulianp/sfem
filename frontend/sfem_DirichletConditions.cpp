#include "sfem_DirichletConditions.hpp"

#include "sfem_config.h"

#include <stddef.h>

#include "boundary_condition.hpp"
#include "operators/boundary_conditions/dirichlet.hpp"
#include "sfem_Function.hpp"
#include "smesh_prolongation.hpp"
#include "smesh_restriction.hpp"
#include "utils.h"

#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_mesh.hpp"
#include "smesh_path.hpp"
#include "smesh_sideset.hpp"

#include <sys/stat.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

// Mesh

#include "hex8_fff.hpp"
#include "hex8_jacobian.hpp"
//
#include "sshex8.hpp"

// C++ includes
//
// #include "sfem_Communicator.hpp"
// #include "smesh_semistructured.hpp"

#include "smesh_glob.hpp"

#include "smesh_path.hpp"
#include "smesh_sideset.hpp"

#ifdef SFEM_ENABLE_MPI
#include <mpi.h>
#endif

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

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif

namespace smesh {
    SharedBuffer<idx_t> create_nodeset_from_sidesets(const std::shared_ptr<Mesh>                 &mesh,
                                                     const std::vector<std::shared_ptr<Sideset>> &sidesets);
}

namespace sfem {

    static bool mesh_is_mpi_distributed(const std::shared_ptr<Mesh> &mesh) {
        return mesh && mesh->is_distributed() && mesh->comm() && mesh->comm()->size() > 1 && mesh->distributed() &&
               mesh->distributed()->node_mapping();
    }

    static SharedBuffer<idx_t> coarse_nodeset_from_fine_nodeset(const std::shared_ptr<FunctionSpace> &fine_space,
                                                                const std::shared_ptr<FunctionSpace> &coarse_space,
                                                                const SharedBuffer<idx_t>            &fine_nodeset,
                                                                const idx_t                           max_coarse_idx) {
        if (!fine_nodeset || fine_nodeset->size() == 0) {
            return create_host_buffer<idx_t>(0);
        }

        auto fm = fine_space ? fine_space->mesh_ptr() : nullptr;
        auto cm = coarse_space ? coarse_space->mesh_ptr() : nullptr;
        if (!mesh_is_mpi_distributed(fm) || !mesh_is_mpi_distributed(cm)) {
            ptrdiff_t n   = 0;
            idx_t    *idx = nullptr;
            smesh::hierarchical_create_coarse_indices<idx_t>(
                    max_coarse_idx, fine_nodeset->size(), fine_nodeset->data(), &n, &idx);
            return manage_host_buffer<idx_t>(n, idx);
        }

        const ptrdiff_t n_fine_local   = fm->n_nodes();
        const ptrdiff_t n_coarse_local = cm->n_nodes();
        auto            fmap           = fm->distributed()->node_mapping()->data();
        auto            cmap           = cm->distributed()->node_mapping()->data();

        std::unordered_map<smesh::large_idx_t, idx_t> gid_to_coarse;
        gid_to_coarse.reserve(static_cast<size_t>(n_coarse_local));
        for (ptrdiff_t j = 0; j < n_coarse_local; ++j) {
            gid_to_coarse.emplace(cmap[j], static_cast<idx_t>(j));
        }

        std::vector<idx_t> out;
        out.reserve(static_cast<size_t>(fine_nodeset->size()));
        auto fns = fine_nodeset->data();
        for (ptrdiff_t k = 0; k < static_cast<ptrdiff_t>(fine_nodeset->size()); ++k) {
            const idx_t fl = fns[k];
            if (fl < 0 || static_cast<ptrdiff_t>(fl) >= n_fine_local) {
                continue;
            }
            auto it = gid_to_coarse.find(fmap[fl]);
            if (it != gid_to_coarse.end()) {
                out.push_back(it->second);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());

        auto buf = create_host_buffer<idx_t>(static_cast<ptrdiff_t>(out.size()));
        if (!out.empty()) {
            std::memcpy(buf->data(), out.data(), out.size() * sizeof(idx_t));
        }
        return buf;
    }

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
                auto mesh_for_sidesets = space->mesh_ptr();
                c.nodeset              = smesh::create_nodeset_from_sidesets(mesh_for_sidesets, c.sidesets);
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

        auto coarse = std::make_shared<DirichletConditions>(coarse_space);
        auto &conds = impl_->conditions;

        // Hierarchical SS numbering puts coarse nodes at ids 0 .. n_coarse-1.
        // Do not scan the fine SoA with the coarse element type: those local slots are not the
        // coarse subset, so max_node_id can exceed the coarse vector length.
        const int       coarse_bs      = std::max(coarse_space->block_size(), 1);
        const ptrdiff_t n_coarse_nodes = coarse_space->n_dofs() / coarse_bs;
        const idx_t     max_coarse_idx = n_coarse_nodes > 0 ? static_cast<idx_t>(n_coarse_nodes - 1) : static_cast<idx_t>(0);

        for (size_t i = 0; i < conds.size(); i++) {
            if (conds[i].nodeset->size() == 0) {
                continue;
            }

            struct Condition cdc;
            cdc.sidesets  = conds[i].sidesets;
            cdc.component = conds[i].component;
            cdc.value     = as_zero ? 0 : conds[i].value;

            const bool mpi = mesh_is_mpi_distributed(impl_->space->mesh_ptr()) &&
                             mesh_is_mpi_distributed(coarse_space->mesh_ptr());

            // Intermediate SS levels (e.g. PROTEUS_HEX125) are still semistructured. Recreating
            // the nodeset from sidesets calls LocalSideTable, which has no HEX-SS face tables.
            // Hierarchical / GID filtering of the fine nodeset is valid on serial and MPI.
            if (n_coarse_nodes <= 0) {
                cdc.nodeset = create_host_buffer<idx_t>(0);
            } else {
                cdc.nodeset = coarse_nodeset_from_fine_nodeset(
                        impl_->space, coarse_space, conds[i].nodeset, max_coarse_idx);

                if (!as_zero && conds[i].values && !mpi) {
                    cdc.values = create_host_buffer<real_t>(static_cast<ptrdiff_t>(cdc.nodeset->size()));
                    smesh::hierarchical_collect_coarse_values<idx_t>(max_coarse_idx,
                                                                     conds[i].nodeset->size(),
                                                                     conds[i].nodeset->data(),
                                                                     conds[i].values->data(),
                                                                     cdc.values->data());
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
        int  rank = comm->rank();

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
                    cdc.nodeset = Buffer<idx_t>::from_file(smesh::Path(pch));
                    if (!cdc.nodeset) {
                        SFEM_ERROR("Failed to read file %s\n", pch);
                        break;
                    }
                } else {
                    cdc.sidesets.push_back(Sideset::create_from_file(comm, smesh::Path(pch)));
                    auto mesh_for_sidesets = space->mesh_ptr();
                    cdc.nodeset            = smesh::create_nodeset_from_sidesets(mesh_for_sidesets, cdc.sidesets);
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

                    auto values = Buffer<real_t>::from_file(smesh::Path(pch + path_key_len));
                    if (!values) {
                        SFEM_ERROR("Failed to read file %s\n", pch + path_key_len);
                    }
                    const ptrdiff_t lsize = values ? (ptrdiff_t)values->size() : 0;

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
#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<DirichletConditions> DirichletConditions::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                               const ryml::NodeRef                  &node) {
        SFEM_TRACE_SCOPE("DirichletConditions::create_from_yaml");

        auto dc = std::make_unique<DirichletConditions>(space);

        for (auto c : node.children()) {
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
                    sideset = Sideset::create_from_file(space->mesh_ptr()->comm(), smesh::Path(path));
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

                nodeset = smesh::create_nodeset_from_sideset(space->mesh_ptr(), sideset);
            } else {
                if (is_file) {
                    std::string path;
                    c["path"] >> path;
                    nodeset = Buffer<idx_t>::from_file(smesh::Path(path));
                    if (!nodeset) {
                        SFEM_ERROR("Unable to read file %s!\n", path.c_str());
                    }
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
                cdc.sidesets.push_back(sideset);
                cdc.nodeset = nodeset;
                dc->impl_->conditions.push_back(cdc);
            }
        }

        return dc;
    }
#endif

    std::shared_ptr<DirichletConditions> DirichletConditions::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                               std::string                           yaml) {
        SFEM_TRACE_SCOPE("DirichletConditions::create_from_yaml");

#ifdef SFEM_ENABLE_RYAML

        ryml::Tree tree  = ryml::parse_in_place(ryml::to_substr(yaml));
        auto       conds = tree["dirichlet_conditions"];
        return create_from_yaml(space, conds);

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
            if (c.nodeset->size() == 0) continue;
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

    int DirichletConditions::value(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("DirichletConditions::value");

        // This is pure algebraic energy (may need to scale with boundary mass matrix for proper energy)
        for (auto &c : impl_->conditions) {
            if (c.values) {
                constraint_objective_nodes_to_values_vec(
                        c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, c.values->data(), x, out);
            } else {
                constraint_objective_nodes_to_value_vec(
                        c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, c.value, x, out);
            }
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::value_steps(const real_t       *x,
                                         const real_t       *h,
                                         const int           nsteps,
                                         const real_t *const steps,
                                         real_t *const       out) {
        SFEM_TRACE_SCOPE("DirichletConditions::value_steps");
        for (auto &c : impl_->conditions) {
            if (c.values) {
                constraint_objective_nodes_to_values_vec_steps(c.nodeset->size(),
                                                               c.nodeset->data(),
                                                               impl_->space->block_size(),
                                                               c.component,
                                                               c.values->data(),
                                                               x,
                                                               h,
                                                               nsteps,
                                                               steps,
                                                               out);
            } else {
                constraint_objective_nodes_to_value_vec_steps(c.nodeset->size(),
                                                              c.nodeset->data(),
                                                              impl_->space->block_size(),
                                                              c.component,
                                                              c.value,
                                                              x,
                                                              h,
                                                              nsteps,
                                                              steps,
                                                              out);
            }
        }
        return SFEM_SUCCESS;
    }

    int DirichletConditions::gradient(const real_t *const x, real_t *const g) {
        SFEM_TRACE_SCOPE("DirichletConditions::gradient");

        for (auto &c : impl_->conditions) {
            if (c.nodeset->size() == 0) continue;
            if (c.values) {
                constraint_gradient_nodes_to_values_vec(
                        c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, c.values->data(), x, g);

            } else {
                constraint_gradient_nodes_to_value_vec(
                        c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, c.value, x, g);
            }
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::apply_value(const real_t value, real_t *const x) {
        SFEM_TRACE_SCOPE("DirichletConditions::apply_value");

        for (auto &c : impl_->conditions) {
            if (c.nodeset->size() == 0) continue;
            constraint_nodes_to_value_vec(
                    c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, value, x);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        SFEM_TRACE_SCOPE("DirichletConditions::copy_constrained_dofs");

        for (auto &c : impl_->conditions) {
            if (c.nodeset->size() == 0) continue;
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
            if (c.nodeset->size() == 0) continue;
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
            if (c.nodeset->size() == 0) continue;

            bsr_constraint_nodes_to_identity_vec(
                    c.nodeset->size(), c.nodeset->data(), impl_->space->block_size(), c.component, 1, rowptr, colidx, values);
        }

        return SFEM_SUCCESS;
    }

    int DirichletConditions::mask(mask_t *mask) {
        SFEM_TRACE_SCOPE("DirichletConditions::mask");

        const int block_size = impl_->space->block_size();
        for (auto &c : impl_->conditions) {
            if (c.nodeset->size() == 0) continue;

            auto nodeset = c.nodeset->data();
            for (ptrdiff_t node = 0; node < c.nodeset->size(); node++) {
                const ptrdiff_t idx = nodeset[node] * block_size + c.component;
                mask_set(idx, mask);
            }
        }

        return SFEM_SUCCESS;
    }

#ifndef SFEM_ENABLE_CUDA
    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc) {
        return dc;
    }
#endif

}  // namespace sfem
