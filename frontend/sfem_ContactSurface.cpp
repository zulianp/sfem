#include "sfem_ContactSurface.hpp"

// C

// 
// #include "sfem_sshex8_skin.hpp"
#include "smesh_ssquad4_mesh.hpp"

#include "smesh_device_buffer.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "cu_contact_surface.hpp"
#include "sfem_Function_incore_cuda.hpp"
#endif

// C++
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"
#include "smesh_semistructured.hpp"

#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"
#include "smesh_sshex8_graph.hpp"
#include "smesh_ssquad4.hpp"
#include "smesh_ssquad4_graph.hpp"
#include "smesh_ssquad4_mesh.hpp"
#include "smesh_sstet4.hpp"
#include "smesh_sstet4_graph.hpp"
#include "smesh_sstet4_mesh.hpp"
#include "sshex8.hpp"

#include <cstring>
#include <unordered_map>
#include <vector>

namespace sfem {

    void remap_elements_to_contiguous_index(const ptrdiff_t  n_elements,
                                            const int        nxe,
                                            idx_t **const    elements,
                                            ptrdiff_t *const out_n_contiguous,
                                            idx_t **const    out_node_mapping) {
        idx_t n = 0;
        for (int d = 0; d < nxe; d++) {
            for (ptrdiff_t i = 0; i < n_elements; i++) {
                n = MAX(elements[d][i], n);
            }
        }

        n += 1;

        idx_t *remap = (idx_t *)malloc(n * sizeof(idx_t));
        for (ptrdiff_t i = 0; i < n; ++i) {
            remap[i] = SFEM_IDX_INVALID;
        }

        ptrdiff_t n_contiguous = 0;
        for (ptrdiff_t i = 0; i < n_elements; ++i) {
            for (int d = 0; d < nxe; ++d) {
                idx_t idx = elements[d][i];
                if (remap[idx] < 0) {
                    remap[idx] = n_contiguous++;
                }
            }
        }

        for (int d = 0; d < nxe; d++) {
            for (ptrdiff_t i = 0; i < n_elements; i++) {
                elements[d][i] = remap[elements[d][i]];
            }
        }

        idx_t *node_mapping = (idx_t *)malloc(n_contiguous * sizeof(idx_t));
        for (ptrdiff_t i = 0; i < n; ++i) {
            if (remap[i] != SFEM_IDX_INVALID) {
                node_mapping[remap[i]] = i;
            }
        }

        free(remap);
        *out_n_contiguous = n_contiguous;
        *out_node_mapping = node_mapping;
    }

    // Pack owned volume nodes as a contiguous prefix and send ghosts to a geometry-only tail.
    // Constraint node_mapping is owned-only; SS sides keep ghost vertices for SDF/normals.
    static void pack_owned_contact_nodes(const std::shared_ptr<Mesh> &mesh,
                                         SharedBuffer<idx_t>        &node_mapping,
                                         SharedBuffer<idx_t *>      &ss_sides,
                                         SharedBuffer<idx_t>        &geometry_node_mapping) {
        geometry_node_mapping = node_mapping;
        if (!mesh || !mesh->is_distributed() || !mesh->comm() || mesh->comm()->size() <= 1 || !mesh->distributed()) {
            return;
        }
        if (!node_mapping || node_mapping->size() == 0 || !ss_sides) {
            return;
        }

        const ptrdiff_t n_owned_vol = mesh->distributed()->n_nodes_owned();
        const ptrdiff_t n_old       = node_mapping->size();
        auto            old_map     = node_mapping->data();

        std::vector<idx_t> owned;
        std::vector<idx_t> ghosts;
        owned.reserve(static_cast<size_t>(n_old));
        ghosts.reserve(static_cast<size_t>(n_old));
        std::vector<idx_t> old_to_new(static_cast<size_t>(n_old));

        for (ptrdiff_t i = 0; i < n_old; ++i) {
            if (old_map[i] < n_owned_vol) {
                old_to_new[static_cast<size_t>(i)] = static_cast<idx_t>(owned.size());
                owned.push_back(old_map[i]);
            } else {
                old_to_new[static_cast<size_t>(i)] = SFEM_IDX_INVALID;
            }
        }
        const ptrdiff_t n_constrained = static_cast<ptrdiff_t>(owned.size());
        for (ptrdiff_t i = 0; i < n_old; ++i) {
            if (old_to_new[static_cast<size_t>(i)] == SFEM_IDX_INVALID) {
                old_to_new[static_cast<size_t>(i)] = static_cast<idx_t>(n_constrained + static_cast<ptrdiff_t>(ghosts.size()));
                ghosts.push_back(old_map[i]);
            }
        }

        const int       nxe = static_cast<int>(ss_sides->extent(0));
        const ptrdiff_t ne  = ss_sides->extent(1);
        for (int r = 0; r < nxe; ++r) {
            auto row = ss_sides->data()[r];
            for (ptrdiff_t e = 0; e < ne; ++e) {
                const idx_t old_id = row[e];
                if (old_id < 0 || static_cast<ptrdiff_t>(old_id) >= n_old) {
                    SFEM_ERROR("SSMeshContactSurface: surface id %ld is out of range (n=%ld)\n",
                               (long)old_id,
                               (long)n_old);
                }
                row[e] = old_to_new[static_cast<size_t>(old_id)];
            }
        }

        geometry_node_mapping = create_host_buffer<idx_t>(n_old);
        auto geom             = geometry_node_mapping->data();
        for (ptrdiff_t i = 0; i < n_constrained; ++i) {
            geom[i] = owned[static_cast<size_t>(i)];
        }
        for (size_t i = 0; i < ghosts.size(); ++i) {
            geom[n_constrained + static_cast<ptrdiff_t>(i)] = ghosts[i];
        }

        node_mapping = create_host_buffer<idx_t>(n_constrained);
        auto nm      = node_mapping->data();
        for (ptrdiff_t i = 0; i < n_constrained; ++i) {
            nm[i] = owned[static_cast<size_t>(i)];
        }
    }

    template <typename T>
    using ptr = std::shared_ptr<T>;

    // std::tuple<smesh::ElemType, ptr<Buffer<idx_t *>>, ptr<Buffer<geom_t *>>, ptr<Buffer<idx_t>>> read_surface(
    //         const std::shared_ptr<FunctionSpace> &space,
    //         const std::string                    &path) {
    //     // Read mesh surface information
    //     const smesh::ElemType element_type      = space->element_type();
    //     const smesh::ElemType side_element_type = shell_type(side_type(element_type));
    //     const int           nxe               = elem_num_nodes(side_element_type);

    //     // space->has_semi_structured_mesh() ? elem_num_nodes(type_from_string(surface_elem_type.c_str()))

    //     idx_t   **raw_sides = (idx_t **)malloc(nxe * sizeof(idx_t *));
    //     ptrdiff_t _nope_ = SFEM_PTRDIFF_INVALID, len = SFEM_PTRDIFF_INVALID;

    //     char pattern[SFEM_MAX_PATH_LENGTH];
    //     sprintf(pattern, "%s/i*.*raw", path.c_str());

    //     auto paths = sfem::find_files(pattern);

    //     assert((int)paths.size() == nxe);

    //     auto mesh = space->mesh_ptr();

    //     for (int d = 0; d < nxe; d++) {
    //         idx_t    *idx   = nullptr;
    //         ptrdiff_t len_d = SFEM_PTRDIFF_INVALID;
    //         if (array_create_from_file(mesh->comm(), paths[d].c_str(), SFEM_MPI_IDX_T, (void **)&idx, &_nope_, &len_d)) {
    //             SFEM_ERROR("Unable to read path %s\n", paths[d].c_str());
    //         }

    //         raw_sides[d] = idx;

    //         assert(len == SFEM_PTRDIFF_INVALID || len_d == len);
    //         len = len_d;
    //     }

    //     auto               sides = sfem::manage_host_buffer(nxe, len, raw_sides);
    //     ptr<Buffer<idx_t>> node_mapping;

    //     // bool has_parent_indexing = points == "parent";
    //     // if (has_parent_indexing) {
    //     //     idx_t    *idx          = nullptr;
    //     //     ptrdiff_t n_contiguous = SFEM_PTRDIFF_INVALID;
    //     //     remap_elements_to_contiguous_index(
    //     //             sides->extent(1), sides->extent(0), sides->data(), &n_contiguous, &idx);
    //     //     node_mapping = sfem::manage_host_buffer(n_contiguous, idx);

    //     // } else {
    //     std::string path_node_mapping = path + "/node_mapping.raw";

    //     idx_t *idx = nullptr;
    //     if (array_create_from_file(mesh->comm(), path_node_mapping.c_str(), SFEM_MPI_IDX_T, (void **)&idx, &_nope_, &len)) {
    //         SFEM_ERROR("Unable to read path %s\n", path_node_mapping.c_str());
    //     }

    //     node_mapping = sfem::manage_host_buffer(len, idx);
    //     // }

    //     // Allocate buffer for point information
    //     auto surface_points = create_host_buffer<geom_t>(mesh->spatial_dimension(), node_mapping->size());
    //     return {side_element_type, sides, surface_points, node_mapping};
    // }

    class MeshContactSurface::Impl {
    public:
        std::shared_ptr<FunctionSpace>        space;
        std::vector<std::shared_ptr<Sideset>> sidesets;

        std::shared_ptr<Buffer<idx_t *>>  sides;
        std::shared_ptr<Buffer<idx_t>>    node_mapping;
        std::shared_ptr<Buffer<geom_t *>> surface_points;
        smesh::ElemType                   element_type{smesh::INVALID};
        enum ExecutionSpace               execution_space { EXECUTION_SPACE_HOST };

#ifdef SFEM_ENABLE_CUDA
        std::shared_ptr<Buffer<idx_t *>>  sides_device;
        std::shared_ptr<Buffer<idx_t>>    node_mapping_device;
        std::shared_ptr<Buffer<geom_t *>> surface_points_rest_device;
        std::shared_ptr<Buffer<geom_t *>> surface_points_device;
#endif

        void collect_points(std::shared_ptr<Buffer<geom_t *>> &surface_points) {
            SFEM_TRACE_SCOPE("MeshContactSurface::collect_points");

            auto               mesh = space->mesh_ptr();
            const ptrdiff_t    n    = node_mapping->size();
            const idx_t *const idx  = node_mapping->data();
            const int          dim  = mesh->spatial_dimension();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = mesh->points()->data()[d];
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = x[idx[i]];
                }
            }
        }

        void reset_points() {
            SFEM_TRACE_SCOPE("MeshContactSurface::reset_points");
            collect_points(surface_points);

#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == this->execution_space) {
                // FIXME: maybe this could be optimized by avoiding deallocating and allocating the buffer
                surface_points_rest_device = smesh::to_device(surface_points);
                surface_points_device      = smesh::to_device(surface_points);
            }
#endif
        }

        void displace_points(const real_t *disp) {
            SFEM_TRACE_SCOPE("ContactConditions::displace_points");

            auto            mesh = space->mesh_ptr();
            const ptrdiff_t n    = node_mapping->size();
            const int       dim  = mesh->spatial_dimension();

#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == this->execution_space) {
                if (!surface_points_device) {
                    // Lazy initialization of the device buffer
                    surface_points_device = smesh::create_device_buffer<geom_t>(dim, n);
                }

                cu_displace_surface_points(dim,
                                           n,
                                           node_mapping_device->data(),
                                           surface_points_rest_device->data(),
                                           disp,
                                           surface_points_device->data());
                return;
            }
#endif

            const idx_t *const idx = node_mapping->data();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = mesh->points()->data()[d];
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);
                    x_s[i]            = x[b] + disp[b * dim + d];
                }
            }
        }
    };

    MeshContactSurface::MeshContactSurface() : impl_(std::make_unique<Impl>()) {}
    MeshContactSurface::~MeshContactSurface() = default;
    SharedBuffer<geom_t *> MeshContactSurface::points() { return impl_->surface_points; }
    SharedBuffer<idx_t *>  MeshContactSurface::elements() { return impl_->sides; }
    SharedBuffer<idx_t>    MeshContactSurface::node_mapping() { return impl_->node_mapping; }

#ifdef SFEM_ENABLE_CUDA
    SharedBuffer<geom_t *> MeshContactSurface::points_device() { return impl_->surface_points_device; }
    SharedBuffer<idx_t *>  MeshContactSurface::elements_device() { return impl_->sides_device; }
    SharedBuffer<idx_t>    MeshContactSurface::node_mapping_device() { return impl_->node_mapping_device; }
#endif

    smesh::ElemType MeshContactSurface::element_type() const { return impl_->element_type; }

    void MeshContactSurface::displace_points(const real_t *disp) { impl_->displace_points(disp); }
    void MeshContactSurface::reset_points() { impl_->reset_points(); }

    std::unique_ptr<MeshContactSurface> MeshContactSurface::create(const std::shared_ptr<FunctionSpace>        &space,
                                                                   const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                                                   const enum ExecutionSpace                    es) {
        auto            mesh = space->mesh_ptr();
        smesh::ElemType st   = side_type(space->element_type());
        if (st == smesh::INVALID) {
            SFEM_ERROR("Invalid element type: %d\n", space->element_type());
        }
        const int       nnxs = elem_num_nodes(st);

        auto mesh_for_surface = space->mesh_ptr();
        auto sides = smesh::create_surface_from_sidesets(mesh_for_surface, sidesets).second;

        idx_t    *idx          = nullptr;
        ptrdiff_t n_contiguous = SFEM_PTRDIFF_INVALID;
        remap_elements_to_contiguous_index(sides->extent(1), sides->extent(0), sides->data(), &n_contiguous, &idx);
        auto node_mapping = sfem::manage_host_buffer(n_contiguous, idx);

        // Create object
        auto ret                    = std::make_unique<MeshContactSurface>();
        ret->impl_->space           = space;
        ret->impl_->sidesets        = sidesets;
        ret->impl_->sides           = sides;
        ret->impl_->node_mapping    = node_mapping;
        ret->impl_->surface_points  = create_host_buffer<geom_t>(mesh->spatial_dimension(), node_mapping->size());
        ret->impl_->element_type    = shell_type(side_type(space->element_type()));
        if (ret->impl_->element_type == smesh::INVALID) {
            SFEM_ERROR("Invalid element type: %d\n", space->element_type());
        }
        ret->impl_->execution_space = es;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            ret->impl_->sides_device        = to_device(ret->impl_->sides);
            ret->impl_->node_mapping_device = to_device(ret->impl_->node_mapping);
        }
#endif

        return ret;
    }

    std::unique_ptr<MeshContactSurface> MeshContactSurface::create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                             const std::string                    &path,
                                                                             const enum ExecutionSpace             es) {
        SFEM_TRACE_SCOPE("MeshContactSurface::create_from_file");
        auto sideset = Sideset::create_from_file(space->mesh_ptr()->comm(), smesh::Path(path));
        return create(space, {sideset}, es);
    }

    class SSMeshContactSurface::Impl {
    public:
        std::shared_ptr<FunctionSpace>        space;
        std::vector<std::shared_ptr<Sideset>> sidesets;

        std::shared_ptr<Buffer<idx_t *>>  sides;
        std::shared_ptr<Buffer<idx_t *>>  semi_structured_sides;
        std::shared_ptr<Buffer<idx_t>>    node_mapping;
        std::shared_ptr<Buffer<idx_t>>    geometry_node_mapping;
        std::shared_ptr<Buffer<geom_t *>> surface_points;
        smesh::ElemType                   element_type{smesh::INVALID};
        enum ExecutionSpace               execution_space { EXECUTION_SPACE_HOST };

#ifdef SFEM_ENABLE_CUDA
        std::shared_ptr<Buffer<idx_t *>>  sides_device;
        std::shared_ptr<Buffer<idx_t>>    node_mapping_device;
        std::shared_ptr<Buffer<geom_t *>> surface_points_rest_device;
        std::shared_ptr<Buffer<geom_t *>> surface_points_device;
#endif

        void collect_points(std::shared_ptr<Buffer<geom_t *>> &surface_points) {
            SFEM_TRACE_SCOPE("SSMeshContactSurface::collect_points");

            auto &ssmesh   = space->mesh();
            auto  sspoints = ssmesh.points();

            auto               mesh = space->mesh_ptr();
            auto               map  = geometry_node_mapping ? geometry_node_mapping : node_mapping;
            const ptrdiff_t    n    = map->size();
            const idx_t *const idx  = map->data();
            const int          dim  = mesh->spatial_dimension();
            const int          pdim = static_cast<int>(surface_points->extent(0));

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = sspoints->data()[d];
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = x[idx[i]];
                }
            }

            for (int d = dim; d < pdim; ++d) {
                geom_t *const x_s = surface_points->data()[d];
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = geom_t(0);
                }
            }
        }

        void reset_points() {
            SFEM_TRACE_SCOPE("SSMeshContactSurface::reset_points");
            collect_points(surface_points);

#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == this->execution_space) {
                // FIXME: maybe this could be optimized by avoiding deallocating and allocating the buffer
                surface_points_rest_device = to_device(surface_points);
                surface_points_device      = to_device(surface_points);
            }
#endif
        }

        void displace_points(const real_t *disp) {
            SFEM_TRACE_SCOPE("ContactConditions::displace_points_semistructured");

            auto            mesh = space->mesh_ptr();
            auto            map  = geometry_node_mapping ? geometry_node_mapping : node_mapping;
            const ptrdiff_t n    = map->size();
            const int       dim  = mesh->spatial_dimension();

#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == this->execution_space) {
                if (!surface_points_device) {
                    // Lazy initialization of the device buffer
                    surface_points_device = smesh::create_device_buffer<geom_t>(dim, n);
                }

                cu_displace_surface_points(dim,
                                           n,
                                           node_mapping_device->data(),
                                           surface_points_rest_device->data(),
                                           disp,
                                           surface_points_device->data());
                return;
            }
#endif
            auto &ssmesh   = space->mesh();
            auto  sspoints = ssmesh.points();

            const idx_t *const idx = map->data();
            const int          pdim = static_cast<int>(surface_points->extent(0));

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = sspoints->data()[d];
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);
                    x_s[i]            = x[b] + disp[b * dim + d];
                }
            }

            for (int d = dim; d < pdim; ++d) {
                geom_t *const x_s = surface_points->data()[d];
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = geom_t(0);
                }
            }
        }
    };

    SSMeshContactSurface::SSMeshContactSurface() : impl_(std::make_unique<Impl>()) {}
    SSMeshContactSurface::~SSMeshContactSurface() = default;

    SharedBuffer<geom_t *> SSMeshContactSurface::points() { return impl_->surface_points; }
    SharedBuffer<idx_t *>  SSMeshContactSurface::elements() { return impl_->sides; }
    SharedBuffer<idx_t>    SSMeshContactSurface::node_mapping() { return impl_->node_mapping; }
    SharedBuffer<idx_t>    SSMeshContactSurface::geometry_node_mapping() {
        return impl_->geometry_node_mapping ? impl_->geometry_node_mapping : impl_->node_mapping;
    }

#ifdef SFEM_ENABLE_CUDA
    SharedBuffer<geom_t *> SSMeshContactSurface::points_device() { return impl_->surface_points_device; }
    SharedBuffer<idx_t *>  SSMeshContactSurface::elements_device() { return impl_->sides_device; }
    SharedBuffer<idx_t>    SSMeshContactSurface::node_mapping_device() { return impl_->node_mapping_device; }
#endif

    smesh::ElemType SSMeshContactSurface::element_type() const { return impl_->element_type; }

    void SSMeshContactSurface::displace_points(const real_t *disp) { impl_->displace_points(disp); }
    void SSMeshContactSurface::reset_points() { impl_->reset_points(); }

    std::unique_ptr<SSMeshContactSurface> SSMeshContactSurface::create(const std::shared_ptr<FunctionSpace>        &space,
                                                                       const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                                                       const enum ExecutionSpace                    es) {
        auto     &ssmesh = space->mesh();
        const int level  = smesh::semistructured_level(ssmesh);

        std::vector<std::shared_ptr<Buffer<idx_t *>>> parts;
        smesh::ElemType                               family  = smesh::INVALID;
        smesh::ElemType                               surf_st = smesh::INVALID;

        for (const auto &ss : sidesets) {
            if (!ss) {
                continue;
            }
            auto block = ssmesh.block(ss->block_id());
            if (!block) {
                SFEM_ERROR("Unable to find block %d for contact sideset!\n", static_cast<int>(ss->block_id()));
            }
            const auto blk_family = smesh::ss_source_family(block->element_type());
            if (family == smesh::INVALID) {
                family = blk_family;
            } else if (family != blk_family) {
                SFEM_ERROR("SSMeshContactSurface: mixed SS families in one contact surface are not supported\n");
            }
            if (ss->parent()->size() == 0) {
                continue;
            }
            auto extracted = smesh::create_surface_from_sideset(space->mesh_ptr(), ss);
            if (!extracted.second) {
                SFEM_ERROR("Unable to extract surface from sideset!\n");
            }
            surf_st = extracted.first;
            parts.push_back(extracted.second);
        }

        if (family == smesh::INVALID) {
            if (ssmesh.n_blocks() == 0) {
                SFEM_ERROR("SSMeshContactSurface: mesh has no blocks\n");
            }
            family = smesh::ss_source_family(ssmesh.element_type(0));
        }

        if (family == smesh::TRI3) {
            SFEM_ERROR("SSMeshContactSurface: no SS TRI volume family; unstructured TRI3 is MeshContactSurface, not SSMGC\n");
        }

        if (family != smesh::HEX8 && family != smesh::TET4 && family != smesh::QUAD4) {
            SFEM_ERROR("SSMeshContactSurface: SS family %s is not implemented for contact\n",
                       smesh::type_to_string(family));
        }

        SharedBuffer<idx_t *> semi_structured_sides;
        if (parts.empty()) {
            int nxe = smesh::ssquad4_nxe(level);
            if (family == smesh::TET4) {
                nxe = smesh::sstri_nxe(level);
            } else if (family == smesh::QUAD4) {
                nxe = smesh::ssedge_nxe(level);
            } else {
                nxe = (level + 1) * (level + 1);
            }
            semi_structured_sides = sfem::create_host_buffer<idx_t>(nxe, 0);
        } else if (parts.size() == 1) {
            semi_structured_sides = parts[0];
        } else {
            const int       nxe = static_cast<int>(parts[0]->extent(0));
            ptrdiff_t       ne  = 0;
            for (const auto &p : parts) {
                if (static_cast<int>(p->extent(0)) != nxe) {
                    SFEM_ERROR("SSMeshContactSurface: sideset SoA row mismatch\n");
                }
                ne += p->extent(1);
            }
            semi_structured_sides = sfem::create_host_buffer<idx_t>(nxe, ne);
            ptrdiff_t off         = 0;
            for (const auto &p : parts) {
                for (int r = 0; r < nxe; ++r) {
                    std::memcpy(semi_structured_sides->data()[r] + off, p->data()[r], p->extent(1) * sizeof(idx_t));
                }
                off += p->extent(1);
            }
        }

        auto node_mapping = sfem::create_host_buffer<idx_t>(0);
        if (semi_structured_sides->extent(1) > 0) {
            idx_t           *idx          = nullptr;
            ptrdiff_t        n_contiguous = SFEM_PTRDIFF_INVALID;
            std::vector<int> levels(smesh::sshex8_hierarchical_n_levels(level));
            smesh::sshex8_hierarchical_mesh_levels(level, levels.size(), levels.data());

            if (family == smesh::HEX8) {
                smesh::ssquad4_hierarchical_remapping(level,
                                                      levels.size(),
                                                      levels.data(),
                                                      semi_structured_sides->extent(1),
                                                      ssmesh.n_nodes(),
                                                      semi_structured_sides->data(),
                                                      &idx,
                                                      &n_contiguous);
            } else if (family == smesh::TET4) {
                smesh::sstri_hierarchical_remapping(level,
                                                    levels.size(),
                                                    levels.data(),
                                                    semi_structured_sides->extent(1),
                                                    ssmesh.n_nodes(),
                                                    semi_structured_sides->data(),
                                                    &idx,
                                                    &n_contiguous);
            } else {
                smesh::ssedge_hierarchical_remapping(level,
                                                     levels.size(),
                                                     levels.data(),
                                                     semi_structured_sides->extent(1),
                                                     ssmesh.n_nodes(),
                                                     semi_structured_sides->data(),
                                                     &idx,
                                                     &n_contiguous);
            }

            node_mapping = sfem::manage_host_buffer(n_contiguous, idx);
        }

        SharedBuffer<idx_t> geometry_node_mapping = node_mapping;
        pack_owned_contact_nodes(space->mesh_ptr(), node_mapping, semi_structured_sides, geometry_node_mapping);

        int  nnxs = 4;
        int  nexs = level * level;
        auto elem_type = smesh::QUADSHELL4;
        if (family == smesh::TET4) {
            nnxs      = 3;
            nexs      = smesh::sstri_txe(level);
            elem_type = smesh::TRISHELL3;
        } else if (family == smesh::QUAD4) {
            nnxs      = 2;
            nexs      = smesh::ssedge_txe(level);
            elem_type = smesh::EDGESHELL2;
        }

        auto sides = sfem::create_host_buffer<idx_t>(nnxs, semi_structured_sides->extent(1) * nexs);
        if (semi_structured_sides->extent(1) > 0) {
            if (family == smesh::HEX8) {
                smesh::ssquad4_to_standard_quad4_mesh(
                        level, semi_structured_sides->extent(1), semi_structured_sides->data(), sides->data());
            } else if (family == smesh::TET4) {
                smesh::sstri_to_standard_tri3_mesh(
                        level, semi_structured_sides->extent(1), semi_structured_sides->data(), sides->data());
            } else {
                smesh::ssedge_to_standard_edge2_mesh(
                        level, semi_structured_sides->extent(1), semi_structured_sides->data(), sides->data());
            }
        }

        (void)surf_st;

        auto ret                          = std::make_unique<SSMeshContactSurface>();
        ret->impl_->space                 = space;
        ret->impl_->sidesets              = sidesets;
        ret->impl_->sides                 = sides;
        ret->impl_->semi_structured_sides = semi_structured_sides;
        ret->impl_->node_mapping          = node_mapping;
        ret->impl_->geometry_node_mapping = geometry_node_mapping;
        ret->impl_->surface_points        = create_host_buffer<geom_t>(3, geometry_node_mapping->size());
        ret->impl_->element_type          = elem_type;
        ret->impl_->execution_space       = es;
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            if (family != smesh::HEX8) {
                SFEM_ERROR("SSMeshContactSurface: device contact is implemented for HEX SSQUAD faces only\n");
            }
            ret->impl_->sides_device        = to_device(ret->impl_->sides);
            ret->impl_->node_mapping_device = to_device(ret->impl_->node_mapping);
        }
#endif

        return ret;
    }

    std::shared_ptr<Buffer<idx_t *>> SSMeshContactSurface::semi_structured_elements() { return impl_->semi_structured_sides; }

    std::shared_ptr<ContactSurface> create_contact_surface(const std::shared_ptr<FunctionSpace>        &space,
                                                           const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                                           const enum ExecutionSpace                    es) {
        assert(es == sfem::EXECUTION_SPACE_HOST);

        if (space->has_semi_structured_mesh()) {
            return SSMeshContactSurface::create(space, sidesets, es);
        } else {
            return MeshContactSurface::create(space, sidesets, es);
        }
    }

}  // namespace sfem

