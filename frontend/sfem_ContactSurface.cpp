#include "sfem_ContactSurface.hpp"

// C
#include "adj_table.hpp"
#include "sfem_hex8_mesh_graph.hpp"
#include "sfem_sshex8_skin.hpp"
#include "sshex8_mesh.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "cu_contact_surface.hpp"
#include "sfem_Function_incore_cuda.hpp"
#endif

// C++
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"
#include "smesh_sshex8_graph.hpp"
#include "sshex8.hpp"

#include <unordered_map>

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
                surface_points_rest_device = to_device(surface_points);
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
                    surface_points_device = create_device_buffer<geom_t>(dim, n);
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
            const ptrdiff_t    n    = node_mapping->size();
            const idx_t *const idx  = node_mapping->data();
            const int          dim  = mesh->spatial_dimension();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = sspoints->data()[d];
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = x[idx[i]];
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
            const ptrdiff_t n    = node_mapping->size();
            const int       dim  = mesh->spatial_dimension();

#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == this->execution_space) {
                if (!surface_points_device) {
                    // Lazy initialization of the device buffer
                    surface_points_device = create_device_buffer<geom_t>(dim, n);
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

            const idx_t *const idx = node_mapping->data();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = sspoints->data()[d];
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);
                    x_s[i]            = x[b] + disp[b * dim + d];
                }
            }
        }
    };

    SSMeshContactSurface::SSMeshContactSurface() : impl_(std::make_unique<Impl>()) {}
    SSMeshContactSurface::~SSMeshContactSurface() = default;

    SharedBuffer<geom_t *> SSMeshContactSurface::points() { return impl_->surface_points; }
    SharedBuffer<idx_t *>  SSMeshContactSurface::elements() { return impl_->sides; }
    SharedBuffer<idx_t>    SSMeshContactSurface::node_mapping() { return impl_->node_mapping; }

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
        auto &ssmesh = space->mesh();
        const int level = sfem::semi_structured_level(ssmesh);

        if (sidesets.size() > 1) {
            SFEM_ERROR("Not implemented!\n");
        }

        auto semi_structured_sides =
                sfem::create_host_buffer<idx_t>((level + 1) * (level + 1), sidesets[0]->parent()->size());

        if (smesh::sshex8_extract_surface_from_sideset(level,
                                                       sfem::semi_structured_element_data(ssmesh),
                                                       sidesets[0]->parent()->size(),
                                                       sidesets[0]->parent()->data(),
                                                       sidesets[0]->lfi()->data(),
                                                       semi_structured_sides->data()) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to extract surface from sideset!\n");
        }

        idx_t           *idx          = nullptr;
        ptrdiff_t        n_contiguous = SFEM_PTRDIFF_INVALID;
        std::vector<int> levels(smesh::sshex8_hierarchical_n_levels(level));

        smesh::sshex8_hierarchical_mesh_levels(level, levels.size(), levels.data());

        // auto semi_structured_sides = sfem::create_surface_from_sidesets(space, sidesets).second;

        smesh::ssquad4_hierarchical_remapping(level,
                                              levels.size(),
                                              levels.data(),
                                              semi_structured_sides->extent(1),
                                              ssmesh.n_nodes(),
                                              semi_structured_sides->data(),
                                              &idx,
                                              &n_contiguous);
        auto node_mapping = sfem::manage_host_buffer(n_contiguous, idx);

        const int nnxs  = 4;
        const int nexs  = level * level;
        auto      sides = sfem::create_host_buffer<idx_t>(nnxs, semi_structured_sides->extent(1) * nexs);

        ssquad4_to_standard_quad4_mesh(
                level, semi_structured_sides->extent(1), semi_structured_sides->data(), sides->data());

        // Create object
        auto ret                          = std::make_unique<SSMeshContactSurface>();
        ret->impl_->space                 = space;
        ret->impl_->sidesets              = sidesets;
        ret->impl_->sides                 = sides;
        ret->impl_->semi_structured_sides = semi_structured_sides;
        ret->impl_->node_mapping          = node_mapping;
        ret->impl_->surface_points  = create_host_buffer<geom_t>(space->mesh_ptr()->spatial_dimension(), node_mapping->size());
        // ret->impl_->element_type    = shell_type(side_type(macro_base_elem(space->element_type()))); //FIXME
        ret->impl_->element_type    = smesh::QUADSHELL4;
        // if (ret->impl_->element_type == smesh::INVALID) {
        //     SFEM_ERROR("Invalid element type: %d\n", space->element_type());
        // }
        ret->impl_->execution_space = es;
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
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
