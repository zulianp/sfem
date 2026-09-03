#include "sfem_NeumannConditions.hpp"

#include "sfem_config.h"

#include <stddef.h>

#include "utils.h"

#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"

#include "integrate_values.hpp"
#include "neumann.hpp"

#include <sys/stat.h>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "smesh_glob.hpp"

// Mesh

#include "hex8_fff.hpp"
#include "hex8_jacobian.hpp"
//
#include "sshex8.hpp"

// C++ includes
//
// #include "smesh_semistructured.hpp"

#include "smesh_glob.hpp"

#ifdef SFEM_ENABLE_RYAML

#if defined(RYML_SINGLE_HEADER)
#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>
#elif defined(RYML_SINGLE_HEADER_LIB)
#include <ryml_all.hpp>
#else
#include <c4/format.hpp>
#include <ryml.hpp>
#include <ryml_std.hpp>
#endif

#include <sstream>
#endif

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif

namespace sfem {

    static SFEM_INLINE count_t neumann_find_col(const idx_t target, const idx_t *const SFEM_RESTRICT row, const count_t lenrow) {
        if (lenrow <= 32) {
            for (count_t k = 0; k < lenrow; ++k) {
                if (row[k] == target) return k;
            }

            return -1;
        }

        count_t left  = 0;
        count_t right = lenrow;
        while (left < right) {
            const count_t mid = left + ((right - left) >> 1);
            if (row[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left < lenrow && row[left] == target ? left : -1;
    }

    static bool is_supported_pressure_surface(const smesh::ElemType element_type) {
        return element_type == smesh::EDGE2 || element_type == smesh::EDGESHELL2 || element_type == smesh::TRI3 ||
               element_type == smesh::QUAD4;
    }

    static SFEM_INLINE void cross3(const real_t a[3], const real_t b[3], real_t out[3]) {
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    }

    static SFEM_INLINE void current_point3(const geom_t *const *const points,
                                           const real_t *const        displacement,
                                           const idx_t                node,
                                           real_t                     out[3]) {
        out[0] = points[0][node] + (displacement ? displacement[3 * node] : 0);
        out[1] = points[1][node] + (displacement ? displacement[3 * node + 1] : 0);
        out[2] = points[2][node] + (displacement ? displacement[3 * node + 2] : 0);
    }

    static SFEM_INLINE void atomic_add3(real_t *const out, const idx_t node, const real_t value[3], const real_t scale) {
        for (int d = 0; d < 3; ++d) {
#pragma omp atomic update
            out[3 * node + d] += scale * value[d];
        }
    }

    static SFEM_INLINE void pressure_triangle_gradient(const idx_t                a,
                                                       const idx_t                b,
                                                       const idx_t                c,
                                                       const geom_t *const *const points,
                                                       const real_t *const        displacement,
                                                       const real_t               q,
                                                       real_t *const              out) {
        real_t xa[3], xb[3], xc[3], cross[3];
        current_point3(points, displacement, a, xa);
        current_point3(points, displacement, b, xb);
        current_point3(points, displacement, c, xc);
        cross3(xb, xc, cross);
        atomic_add3(out, a, cross, q);
        cross3(xc, xa, cross);
        atomic_add3(out, b, cross, q);
        cross3(xa, xb, cross);
        atomic_add3(out, c, cross, q);
    }

    static SFEM_INLINE void pressure_triangle_apply(const idx_t                a,
                                                    const idx_t                b,
                                                    const idx_t                c,
                                                    const geom_t *const *const points,
                                                    const real_t *const        displacement,
                                                    const real_t *const        h,
                                                    const real_t               q,
                                                    real_t *const              out) {
        real_t xa[3], xb[3], xc[3], ha[3], hb[3], hc[3], t0[3], t1[3];
        current_point3(points, displacement, a, xa);
        current_point3(points, displacement, b, xb);
        current_point3(points, displacement, c, xc);
        for (int d = 0; d < 3; ++d) {
            ha[d] = h[3 * a + d];
            hb[d] = h[3 * b + d];
            hc[d] = h[3 * c + d];
        }

        cross3(hb, xc, t0);
        cross3(xb, hc, t1);
        for (int d = 0; d < 3; ++d) t0[d] += t1[d];
        atomic_add3(out, a, t0, q);

        cross3(hc, xa, t0);
        cross3(xc, ha, t1);
        for (int d = 0; d < 3; ++d) t0[d] += t1[d];
        atomic_add3(out, b, t0, q);

        cross3(ha, xb, t0);
        cross3(xa, hb, t1);
        for (int d = 0; d < 3; ++d) t0[d] += t1[d];
        atomic_add3(out, c, t0, q);
    }

    static SFEM_INLINE real_t pressure_triangle_volume(const idx_t                a,
                                                       const idx_t                b,
                                                       const idx_t                c,
                                                       const geom_t *const *const points,
                                                       const real_t *const        displacement) {
        real_t xa[3], xb[3], xc[3], cross[3];
        current_point3(points, displacement, a, xa);
        current_point3(points, displacement, b, xb);
        current_point3(points, displacement, c, xc);
        cross3(xb, xc, cross);
        return (xa[0] * cross[0] + xa[1] * cross[1] + xa[2] * cross[2]) / real_t(6);
    }

    static SFEM_INLINE real_t pressure_triangle_volume_step(const idx_t                a,
                                                            const idx_t                b,
                                                            const idx_t                c,
                                                            const geom_t *const *const points,
                                                            const real_t *const        displacement,
                                                            const real_t *const        increment,
                                                            const real_t               step) {
        real_t xa[3], xb[3], xc[3], cross[3];
        current_point3(points, displacement, a, xa);
        current_point3(points, displacement, b, xb);
        current_point3(points, displacement, c, xc);
        for (int d = 0; d < 3; ++d) {
            xa[d] += step * increment[3 * a + d];
            xb[d] += step * increment[3 * b + d];
            xc[d] += step * increment[3 * c + d];
        }
        cross3(xb, xc, cross);
        return (xa[0] * cross[0] + xa[1] * cross[1] + xa[2] * cross[2]) / real_t(6);
    }

    static SFEM_INLINE void atomic_add_skew(real_t *const block, const real_t scale, const real_t x[3]) {
        const real_t values[9] = {0, -scale * x[2], scale * x[1], scale * x[2], 0, -scale * x[0], -scale * x[1], scale * x[0], 0};
        for (int k = 0; k < 9; ++k) {
#pragma omp atomic update
            block[k] += values[k];
        }
    }

    static SFEM_INLINE int pressure_triangle_hessian(const idx_t                a,
                                                     const idx_t                b,
                                                     const idx_t                c,
                                                     const geom_t *const *const points,
                                                     const real_t *const        displacement,
                                                     const count_t *const       rowptr,
                                                     const idx_t *const         colidx,
                                                     const real_t               q,
                                                     real_t *const              values) {
        const idx_t nodes[3] = {a, b, c};
        real_t      x[3][3];
        current_point3(points, displacement, a, x[0]);
        current_point3(points, displacement, b, x[1]);
        current_point3(points, displacement, c, x[2]);

        real_t *blocks[3][3] = {{nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}};
        for (int i = 0; i < 3; ++i) {
            const count_t begin = rowptr[nodes[i]];
            const count_t len   = rowptr[nodes[i] + 1] - begin;
            for (int j = 0; j < 3; ++j) {
                if (i == j) continue;
                const count_t entry = neumann_find_col(nodes[j], &colidx[begin], len);
                if (entry < 0) return SFEM_FAILURE;
                blocks[i][j] = &values[(begin + entry) * 9];
            }
        }

        atomic_add_skew(blocks[0][1], -q, x[2]);
        atomic_add_skew(blocks[0][2], q, x[1]);
        atomic_add_skew(blocks[1][0], q, x[2]);
        atomic_add_skew(blocks[1][2], -q, x[0]);
        atomic_add_skew(blocks[2][0], -q, x[1]);
        atomic_add_skew(blocks[2][1], q, x[0]);
        return SFEM_SUCCESS;
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

    bool NeumannConditions::is_linear() const {
        for (const auto &c : impl_->conditions) {
            if (c.follower_pressure) return false;
        }
        return true;
    }

    int NeumannConditions::set_time(const real_t time, const real_t global_scale) {
        for (auto &c : impl_->conditions) {
            if (!c.profile_initialized) {
                c.base_value          = c.value;
                c.profile_initialized = true;
            }

            const real_t scale = global_scale * c.profile.value(time);
            c.value            = scale * c.base_value;
        }

        return SFEM_SUCCESS;
    }

    ptrdiff_t NeumannConditions::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t NeumannConditions::n_dofs_image() const { return impl_->space->n_dofs(); }

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
        int  rank = comm->rank();

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
                    std::vector<std::string> paths = smesh::find_files(pattern);

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
                            ptrdiff_t lsize{0};
                            if (smesh::array_read_convert_from_extension<idx_t>(smesh::Path(pch), &ii, &lsize) != SMESH_SUCCESS) {
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
                    auto sideset = Sideset::create_from_file(space->mesh_ptr()->comm(), smesh::Path(pch));
                    cneumann_conditions.sidesets.push_back(sideset);

                    auto mesh_for_surface            = space->mesh_ptr();
                    auto surface                     = smesh::create_surface_from_sideset(mesh_for_surface, sideset);
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

                    auto values = Buffer<real_t>::from_file(smesh::Path(pch + path_key_len));
                    if (!values) {
                        SFEM_ERROR("Failed to read file %s\n", pch + path_key_len);
                    }
                    const ptrdiff_t lsize = values ? (ptrdiff_t)values->size() : 0;

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

                    conds[i].value  = 1;
                    conds[i].values = values;

                } else {
                    conds[i].value = atof(pch);
                }
                i++;

                pch = strtok(NULL, splitter);
            }
        }

        return neumann_conditions;
    }

    std::shared_ptr<NeumannConditions> NeumannConditions::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                           std::string                           yaml) {
        SFEM_TRACE_SCOPE("NeumannConditions::create_from_yaml");

#ifdef SFEM_ENABLE_RYAML
        std::vector<struct Condition> conditions;

        ryml::Tree tree  = ryml::parse_in_place(ryml::to_substr(yaml));
        auto       conds = tree["neumann_conditions"];

        for (auto c : conds.children()) {
            std::shared_ptr<Sideset> sideset;

            const bool is_sideset  = c["type"].readable() && c["type"].val() == "sideset";
            const bool is_pressure = c["type"].readable() && c["type"].val() == "pressure";
            const bool is_file     = c["format"].readable() && c["format"].val() == "file";
            const bool is_expr     = c["format"].readable() && c["format"].val() == "expr";

            assert(is_sideset || is_pressure);
            assert(is_file || is_expr);

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

            std::vector<int>     component;
            std::vector<real_t>  value;
            SharedBuffer<real_t> file_values;
            auto                 node_value     = c["value"];
            auto                 node_component = c["component"];

            assert(node_value.readable());

            if (is_pressure) {
                assert(!node_value.is_seq());
                struct Condition nc;
                nc.sidesets.push_back(sideset);
                node_value >> nc.value;
                nc.follower_pressure = true;
                if (c.has_child("profile") && LoadProfile::from_yaml(c["profile"], nc.profile) != SFEM_SUCCESS) {
                    return nullptr;
                }
                conditions.push_back(nc);
                continue;
            }

            assert(node_component.readable());

            if (node_value.is_map()) {
                if (!node_value.has_child("path")) {
                    SFEM_ERROR("File-backed Neumann value requires path\n");
                    return nullptr;
                }
                std::string value_path;
                node_value["path"] >> value_path;
                file_values = Buffer<real_t>::from_file(smesh::Path(value_path));
                if (!file_values) {
                    SFEM_ERROR("Unable to read Neumann value file %s\n", value_path.c_str());
                    return nullptr;
                }
            } else if (node_value.is_seq()) {
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

            if (file_values && component.size() != 1) {
                SFEM_ERROR("A file-backed Neumann value requires exactly one component\n");
                return nullptr;
            }

            if (!file_values && component.size() != value.size()) {
                SFEM_ERROR("Inconsistent sizes for component (%d) and value (%d)\n", (int)component.size(), (int)value.size());
                return nullptr;
            }

            LoadProfile profile;
            if (c.has_child("profile") && LoadProfile::from_yaml(c["profile"], profile) != SFEM_SUCCESS) return nullptr;

            for (size_t i = 0; i < component.size(); i++) {
                struct Condition nc;
                nc.sidesets.push_back(sideset);
                nc.value     = file_values ? 1 : value[i];
                nc.values    = file_values;
                nc.component = component[i];
                nc.profile   = profile;
                conditions.push_back(nc);
            }
        }

        return create(space, conditions);
#else
        SFEM_ERROR("This functionaly requires -DSFEM_ENABLE_RYAML=ON\n");
        return nullptr;
#endif
    }

    std::shared_ptr<NeumannConditions> NeumannConditions::create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                           const smesh::Path                    &path) {
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

    NeumannConditions::~NeumannConditions() = default;

    int NeumannConditions::hessian_crs(const real_t *const  x,
                                       const count_t *const rowptr,
                                       const idx_t *const   colidx,
                                       real_t *const        values) {
        // Vector-valued CRS assembly uses node blocks before conversion to scalar CRS.
        return hessian_bsr(x, rowptr, colidx, values);
    }

    int NeumannConditions::hessian_bsr(const real_t *const  x,
                                       const count_t *const rowptr,
                                       const idx_t *const   colidx,
                                       real_t *const        values) {
        SFEM_TRACE_SCOPE("NeumannConditions::hessian_bsr");

        auto      space = impl_->space;
        auto      mesh  = space->mesh_ptr();
        const int dim   = mesh->spatial_dimension();
        if (space->block_size() != dim || (dim != 2 && dim != 3)) return SFEM_FAILURE;

        auto points = mesh->points();
        if (space->has_semi_structured_mesh()) points = space->mesh().points();

        for (const auto &c : impl_->conditions) {
            if (!c.follower_pressure) continue;
            if (!is_supported_pressure_surface(c.element_type)) {
                SFEM_ERROR("Follower pressure does not support surface element type %s\n", smesh::type_to_string(c.element_type));
                return SFEM_FAILURE;
            }

            const ptrdiff_t n = c.surface->extent(1);
            int             missing_entry{0};

            if (dim == 3) {
                if (!x || (c.element_type != smesh::TRI3 && c.element_type != smesh::QUAD4)) {
                    SFEM_ERROR("Three-dimensional follower pressure requires a current state and TRI3 or QUAD4 surfaces\n");
                    return SFEM_FAILURE;
                }

                const idx_t *const node0 = c.surface->data()[0];
                const idx_t *const node1 = c.surface->data()[1];
                const idx_t *const node2 = c.surface->data()[2];
                const idx_t *const node3 = c.element_type == smesh::QUAD4 ? c.surface->data()[3] : nullptr;
                const real_t       q     = c.value / real_t(6);
#pragma omp parallel for reduction(| : missing_entry)
                for (ptrdiff_t e = 0; e < n; ++e) {
                    missing_entry |=
                            pressure_triangle_hessian(
                                    node0[e], node1[e], node2[e], points->data(), x, rowptr, colidx, q, values) != SFEM_SUCCESS;
                    if (node3) {
                        missing_entry |= pressure_triangle_hessian(
                                                 node0[e], node2[e], node3[e], points->data(), x, rowptr, colidx, q, values) !=
                                         SFEM_SUCCESS;
                    }
                }
            } else {
                if (c.element_type != smesh::EDGE2 && c.element_type != smesh::EDGESHELL2) {
                    SFEM_ERROR("Two-dimensional follower pressure requires EDGE2 surfaces\n");
                    return SFEM_FAILURE;
                }

                const idx_t *const node0 = c.surface->data()[0];
                const idx_t *const node1 = c.surface->data()[1];
                const real_t       q     = c.value * real_t(0.5);

#pragma omp parallel for reduction(| : missing_entry)
                for (ptrdiff_t e = 0; e < n; ++e) {
                    const idx_t i = node0[e];
                    const idx_t j = node1[e];

                    const count_t i_begin = rowptr[i];
                    const count_t j_begin = rowptr[j];
                    const count_t ij      = neumann_find_col(j, &colidx[i_begin], rowptr[i + 1] - i_begin);
                    const count_t ji      = neumann_find_col(i, &colidx[j_begin], rowptr[j + 1] - j_begin);
                    if (ij < 0 || ji < 0) {
                        missing_entry = 1;
                        continue;
                    }

                    real_t *const block_ij = &values[(i_begin + ij) * 4];
                    real_t *const block_ji = &values[(j_begin + ji) * 4];
#pragma omp atomic update
                    block_ij[1] += q;
#pragma omp atomic update
                    block_ij[2] -= q;
#pragma omp atomic update
                    block_ji[1] -= q;
#pragma omp atomic update
                    block_ji[2] += q;
                }
            }

            if (missing_entry) {
                SFEM_ERROR("Follower pressure surface is not represented in the matrix graph\n");
                return SFEM_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    int NeumannConditions::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeumannConditions::gradient");

        auto space = impl_->space;
        auto mesh  = space->mesh_ptr();

        auto points = mesh->points();
        if (space->has_semi_structured_mesh()) {
            points = space->mesh().points();
        }

        int err = 0;
        for (auto &c : impl_->conditions) {
            if (c.follower_pressure) {
                const int dim = mesh->spatial_dimension();
                if (!x || space->block_size() != dim || !is_supported_pressure_surface(c.element_type)) {
                    SFEM_ERROR("Follower pressure requires a current state and a supported oriented surface\n");
                    return SFEM_FAILURE;
                }

                const idx_t *const node0 = c.surface->data()[0];
                const idx_t *const node1 = c.surface->data()[1];
                const ptrdiff_t    n     = c.surface->extent(1);

                if (dim == 3) {
                    if (c.element_type != smesh::TRI3 && c.element_type != smesh::QUAD4) {
                        SFEM_ERROR("Three-dimensional follower pressure requires TRI3 or QUAD4 surfaces\n");
                        return SFEM_FAILURE;
                    }

                    const idx_t *const node2 = c.surface->data()[2];
                    const idx_t *const node3 = c.element_type == smesh::QUAD4 ? c.surface->data()[3] : nullptr;
                    const real_t       q     = c.value / real_t(6);
#pragma omp parallel for
                    for (ptrdiff_t e = 0; e < n; ++e) {
                        pressure_triangle_gradient(node0[e], node1[e], node2[e], points->data(), x, q, out);
                        if (node3) pressure_triangle_gradient(node0[e], node2[e], node3[e], points->data(), x, q, out);
                    }
                    continue;
                }

                if (dim != 2 || (c.element_type != smesh::EDGE2 && c.element_type != smesh::EDGESHELL2)) {
                    SFEM_ERROR("Two-dimensional follower pressure requires EDGE2 surfaces\n");
                    return SFEM_FAILURE;
                }

                const geom_t *const px = points->data()[0];
                const geom_t *const py = points->data()[1];
                const real_t        q  = c.value * real_t(0.5);

#pragma omp parallel for
                for (ptrdiff_t e = 0; e < n; ++e) {
                    const idx_t  i  = node0[e];
                    const idx_t  j  = node1[e];
                    const real_t xi = px[i] + x[2 * i];
                    const real_t yi = py[i] + x[2 * i + 1];
                    const real_t xj = px[j] + x[2 * j];
                    const real_t yj = py[j] + x[2 * j + 1];

#pragma omp atomic update
                    out[2 * i] += q * yj;
#pragma omp atomic update
                    out[2 * i + 1] -= q * xj;
#pragma omp atomic update
                    out[2 * j] -= q * yi;
#pragma omp atomic update
                    out[2 * j + 1] += q * xi;
                }
            } else if (c.values) {
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

    int NeumannConditions::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("NeumannConditions::apply");

        auto space  = impl_->space;
        auto mesh   = space->mesh_ptr();
        auto points = mesh->points();
        if (space->has_semi_structured_mesh()) points = space->mesh().points();
        for (const auto &c : impl_->conditions) {
            if (!c.follower_pressure) continue;
            const int dim = mesh->spatial_dimension();
            if (!x || space->block_size() != dim || !is_supported_pressure_surface(c.element_type)) {
                SFEM_ERROR("Follower pressure apply requires the current state and a supported oriented surface\n");
                return SFEM_FAILURE;
            }

            const idx_t *const node0 = c.surface->data()[0];
            const idx_t *const node1 = c.surface->data()[1];
            const ptrdiff_t    n     = c.surface->extent(1);

            if (dim == 3) {
                if (c.element_type != smesh::TRI3 && c.element_type != smesh::QUAD4) {
                    SFEM_ERROR("Three-dimensional follower pressure requires TRI3 or QUAD4 surfaces\n");
                    return SFEM_FAILURE;
                }

                const idx_t *const node2 = c.surface->data()[2];
                const idx_t *const node3 = c.element_type == smesh::QUAD4 ? c.surface->data()[3] : nullptr;
                const real_t       q     = c.value / real_t(6);
#pragma omp parallel for
                for (ptrdiff_t e = 0; e < n; ++e) {
                    pressure_triangle_apply(node0[e], node1[e], node2[e], points->data(), x, h, q, out);
                    if (node3) pressure_triangle_apply(node0[e], node2[e], node3[e], points->data(), x, h, q, out);
                }
                continue;
            }

            if (dim != 2 || (c.element_type != smesh::EDGE2 && c.element_type != smesh::EDGESHELL2)) {
                SFEM_ERROR("Two-dimensional follower pressure requires EDGE2 surfaces\n");
                return SFEM_FAILURE;
            }

            const real_t q = c.value * real_t(0.5);

#pragma omp parallel for
            for (ptrdiff_t e = 0; e < n; ++e) {
                const idx_t i = node0[e];
                const idx_t j = node1[e];

#pragma omp atomic update
                out[2 * i] += q * h[2 * j + 1];
#pragma omp atomic update
                out[2 * i + 1] -= q * h[2 * j];
#pragma omp atomic update
                out[2 * j] -= q * h[2 * i + 1];
#pragma omp atomic update
                out[2 * j + 1] += q * h[2 * i];
            }
        }

        return SFEM_SUCCESS;
    }

    int NeumannConditions::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeumannConditions::value");

        auto space  = impl_->space;
        auto mesh   = space->mesh_ptr();
        auto points = mesh->points();
        if (space->has_semi_structured_mesh()) {
            points = space->mesh().points();
        }

        SharedBuffer<real_t> temp;
        real_t               acc = 0;
        int                  err = SFEM_SUCCESS;
        for (const auto &c : impl_->conditions) {
            if (c.follower_pressure) {
                const int dim = mesh->spatial_dimension();
                if (!x || space->block_size() != dim || !is_supported_pressure_surface(c.element_type)) {
                    SFEM_ERROR("Follower pressure value requires a current state and a supported oriented surface\n");
                    return SFEM_FAILURE;
                }

                if (dim == 3) {
                    if (c.element_type != smesh::TRI3 && c.element_type != smesh::QUAD4) {
                        SFEM_ERROR("Three-dimensional follower pressure requires TRI3 or QUAD4 surfaces\n");
                        return SFEM_FAILURE;
                    }

                    const idx_t *const node0         = c.surface->data()[0];
                    const idx_t *const node1         = c.surface->data()[1];
                    const idx_t *const node2         = c.surface->data()[2];
                    const idx_t *const node3         = c.element_type == smesh::QUAD4 ? c.surface->data()[3] : nullptr;
                    const ptrdiff_t    n             = c.surface->extent(1);
                    real_t             volume_change = 0;
#pragma omp parallel for reduction(+ : volume_change)
                    for (ptrdiff_t e = 0; e < n; ++e) {
                        volume_change += pressure_triangle_volume(node0[e], node1[e], node2[e], points->data(), x);
                        volume_change -= pressure_triangle_volume(node0[e], node1[e], node2[e], points->data(), nullptr);
                        if (node3) {
                            volume_change += pressure_triangle_volume(node0[e], node2[e], node3[e], points->data(), x);
                            volume_change -= pressure_triangle_volume(node0[e], node2[e], node3[e], points->data(), nullptr);
                        }
                    }
                    acc += c.value * volume_change;
                    continue;
                }

                if (dim != 2 || (c.element_type != smesh::EDGE2 && c.element_type != smesh::EDGESHELL2)) {
                    SFEM_ERROR("Two-dimensional follower pressure requires EDGE2 surfaces\n");
                    return SFEM_FAILURE;
                }

                const idx_t *const  node0           = c.surface->data()[0];
                const idx_t *const  node1           = c.surface->data()[1];
                const geom_t *const px              = points->data()[0];
                const geom_t *const py              = points->data()[1];
                const ptrdiff_t     n               = c.surface->extent(1);
                real_t              pressure_energy = 0;

#pragma omp parallel for reduction(+ : pressure_energy)
                for (ptrdiff_t e = 0; e < n; ++e) {
                    const idx_t  i  = node0[e];
                    const idx_t  j  = node1[e];
                    const real_t xi = px[i] + x[2 * i];
                    const real_t yi = py[i] + x[2 * i + 1];
                    const real_t xj = px[j] + x[2 * j];
                    const real_t yj = py[j] + x[2 * j + 1];
                    pressure_energy += xi * yj - xj * yi - (px[i] * py[j] - px[j] * py[i]);
                }

                acc += c.value * real_t(0.5) * pressure_energy;
            } else if (c.values) {
                if (!temp) {
                    temp = create_host_buffer<real_t>(space->n_dofs());
                    if (!temp) return SFEM_FAILURE;
                    std::memset(temp->data(), 0, sizeof(real_t) * static_cast<size_t>(space->n_dofs()));
                }

                err |= integrate_values(c.element_type,
                                        c.surface->extent(1),
                                        mesh->n_nodes(),
                                        c.surface->data(),
                                        points->data(),
                                        -c.value,
                                        c.values->data(),
                                        space->block_size(),
                                        c.component,
                                        temp->data());
            } else {
                if (!temp) {
                    temp = create_host_buffer<real_t>(space->n_dofs());
                    if (!temp) return SFEM_FAILURE;
                    std::memset(temp->data(), 0, sizeof(real_t) * static_cast<size_t>(space->n_dofs()));
                }

                err |= integrate_value(c.element_type,
                                       c.surface->extent(1),
                                       mesh->n_nodes(),
                                       c.surface->data(),
                                       points->data(),
                                       -c.value,
                                       space->block_size(),
                                       c.component,
                                       temp->data());
            }
        }

        if (err != SFEM_SUCCESS) return err;

        if (temp) {
            const ptrdiff_t     ndofs = space->n_dofs();
            const real_t *const g     = temp->data();

#pragma omp parallel for reduction(+ : acc)
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                acc += g[i] * x[i];
            }
        }

        *out += acc;
        return SFEM_SUCCESS;
    }

    int NeumannConditions::value_steps(const real_t *const x,
                                       const real_t *const h,
                                       const int           nsteps,
                                       const real_t *const steps,
                                       real_t *const       out) {
        SFEM_TRACE_SCOPE("NeumannConditions::value_steps");

        auto space  = impl_->space;
        auto mesh   = space->mesh_ptr();
        auto points = mesh->points();
        if (space->has_semi_structured_mesh()) {
            points = space->mesh().points();
        }

        const ptrdiff_t      ndofs = space->n_dofs();
        SharedBuffer<real_t> dead_load;
        int                  err = SFEM_SUCCESS;
        for (const auto &c : impl_->conditions) {
            if (c.follower_pressure) {
                const int dim = mesh->spatial_dimension();
                if (!x || space->block_size() != dim || !is_supported_pressure_surface(c.element_type) ||
                    (dim == 2 && c.element_type != smesh::EDGE2 && c.element_type != smesh::EDGESHELL2) ||
                    (dim == 3 && c.element_type != smesh::TRI3 && c.element_type != smesh::QUAD4)) {
                    SFEM_ERROR("Follower pressure value_steps requires a supported oriented surface\n");
                    return SFEM_FAILURE;
                }

                continue;
            }

            if (!dead_load) {
                dead_load = create_host_buffer<real_t>(ndofs);
                if (!dead_load) return SFEM_FAILURE;
                std::memset(dead_load->data(), 0, sizeof(real_t) * static_cast<size_t>(ndofs));
            }

            if (c.values) {
                err |= integrate_values(c.element_type,
                                        c.surface->extent(1),
                                        mesh->n_nodes(),
                                        c.surface->data(),
                                        points->data(),
                                        -c.value,
                                        c.values->data(),
                                        space->block_size(),
                                        c.component,
                                        dead_load->data());
            } else {
                err |= integrate_value(c.element_type,
                                       c.surface->extent(1),
                                       mesh->n_nodes(),
                                       c.surface->data(),
                                       points->data(),
                                       -c.value,
                                       space->block_size(),
                                       c.component,
                                       dead_load->data());
            }
        }

        if (err != SFEM_SUCCESS) return err;

        for (int s = 0; s < nsteps; ++s) {
            const real_t step = steps[s];
            real_t       acc  = 0;
            if (dead_load) {
                const real_t *const g = dead_load->data();
#pragma omp parallel for reduction(+ : acc)
                for (ptrdiff_t i = 0; i < ndofs; ++i) {
                    acc += g[i] * (x[i] + step * h[i]);
                }
            }

            for (const auto &c : impl_->conditions) {
                if (!c.follower_pressure) continue;

                const idx_t *const node0 = c.surface->data()[0];
                const idx_t *const node1 = c.surface->data()[1];
                const ptrdiff_t    n     = c.surface->extent(1);

                if (mesh->spatial_dimension() == 3) {
                    const idx_t *const node2         = c.surface->data()[2];
                    const idx_t *const node3         = c.element_type == smesh::QUAD4 ? c.surface->data()[3] : nullptr;
                    real_t             volume_change = 0;
#pragma omp parallel for reduction(+ : volume_change)
                    for (ptrdiff_t e = 0; e < n; ++e) {
                        volume_change += pressure_triangle_volume_step(node0[e], node1[e], node2[e], points->data(), x, h, step);
                        volume_change -= pressure_triangle_volume(node0[e], node1[e], node2[e], points->data(), nullptr);
                        if (node3) {
                            volume_change +=
                                    pressure_triangle_volume_step(node0[e], node2[e], node3[e], points->data(), x, h, step);
                            volume_change -= pressure_triangle_volume(node0[e], node2[e], node3[e], points->data(), nullptr);
                        }
                    }
                    acc += c.value * volume_change;
                    continue;
                }

                const geom_t *const px              = points->data()[0];
                const geom_t *const py              = points->data()[1];
                real_t              pressure_energy = 0;

#pragma omp parallel for reduction(+ : pressure_energy)
                for (ptrdiff_t e = 0; e < n; ++e) {
                    const idx_t  i  = node0[e];
                    const idx_t  j  = node1[e];
                    const real_t xi = px[i] + x[2 * i] + step * h[2 * i];
                    const real_t yi = py[i] + x[2 * i + 1] + step * h[2 * i + 1];
                    const real_t xj = px[j] + x[2 * j] + step * h[2 * j];
                    const real_t yj = py[j] + x[2 * j + 1] + step * h[2 * j + 1];
                    pressure_energy += xi * yj - xj * yi - (px[i] * py[j] - px[j] * py[i]);
                }

                acc += c.value * real_t(0.5) * pressure_energy;
            }

            out[s] += acc;
        }

        return SFEM_SUCCESS;
    }

    std::shared_ptr<NeumannConditions> NeumannConditions::create(const std::shared_ptr<FunctionSpace> &space,
                                                                 const std::vector<struct Condition>  &conditions) {
        auto nc               = std::make_unique<NeumannConditions>(space);
        nc->impl_->conditions = conditions;

        std::map<  //
                std::shared_ptr<Sideset>,
                std::pair<smesh::ElemType, std::shared_ptr<Buffer<idx_t *>>>>
                sideset_to_surface;

        for (auto &c : nc->impl_->conditions) {
            if (!c.surface) {
                auto it = sideset_to_surface.find(c.sidesets[0]);
                if (it == sideset_to_surface.end()) {
                    auto mesh_for_surface             = space->mesh_ptr();
                    auto surface                      = smesh::create_surface_from_sidesets(mesh_for_surface, c.sidesets);
                    c.element_type                    = surface.first;
                    c.surface                         = surface.second;
                    sideset_to_surface[c.sidesets[0]] = surface;
                } else {
                    c.element_type = it->second.first;
                    c.surface      = it->second.second;
                }
            }
        }

        return nc;
    }

    int NeumannConditions::hessian_diag(const real_t *const /*x*/, real_t *const /*values*/) {
        // Neumann conditions only affect RHS, not the system matrix diagonal
        return SFEM_SUCCESS;
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

#ifndef SFEM_ENABLE_CUDA
    std::shared_ptr<Op> to_device(const std::shared_ptr<NeumannConditions> &nc) { return nc; }
#endif

}  // namespace sfem
