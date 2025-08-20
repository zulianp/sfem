#include "sfem_NeumannConditions.hpp"

#include <stddef.h>

#include "matrixio_array.h"
#include "utils.h"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "integrate_values.h"
#include "neumann.h"

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

namespace sfem {

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

        auto comm = space->mesh_ptr()->comm();
        int      rank = comm->rank();

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
                            if (array_create_from_file(comm->get(), pch, SFEM_MPI_IDX_T, (void **)&ii, &lsize, &gsize)) {
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
                    auto sideset                = Sideset::create_from_file(space->mesh_ptr()->comm(), pch);
                    cneumann_conditions.sidesets.push_back(sideset);

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
                    if (array_create_from_file(comm->get(), pch + path_key_len, SFEM_MPI_REAL_T, (void **)&values, &lsize, &gsize)) {
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
                auto it = sideset_to_surface.find(c.sidesets[0]);
                if (it == sideset_to_surface.end()) {
                    auto surface                  = create_surface_from_sidesets(space, c.sidesets);
                    c.element_type                = surface.first;
                    c.surface                     = surface.second;
                    sideset_to_surface[c.sidesets[0]] = surface;
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

}  // namespace sfem 