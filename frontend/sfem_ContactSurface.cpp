#include "sfem_ContactSurface.hpp"

// C
#include "adj_table.h"
#include "sfem_hex8_mesh_graph.h"
#include "sfem_sshex8_skin.h"
#include "sshex8_mesh.h"

// C++
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

namespace sfem {
    template <typename T>
    using ptr = std::shared_ptr<T>;

    // std::tuple<enum ElemType, ptr<Buffer<idx_t *>>, ptr<Buffer<geom_t *>>, ptr<Buffer<idx_t>>> read_surface(
    //         const std::shared_ptr<FunctionSpace> &space,
    //         const std::string                    &path) {
    //     // Read mesh surface information
    //     const enum ElemType element_type      = space->element_type();
    //     const enum ElemType side_element_type = shell_type(side_type(element_type));
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
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Sideset>       sideset;

        std::shared_ptr<Buffer<idx_t *>>  sides;
        std::shared_ptr<Buffer<idx_t>>    node_mapping;
        std::shared_ptr<Buffer<geom_t *>> surface_points;
        enum ElemType                     element_type { INVALID };

        void collect_points() {
            SFEM_TRACE_SCOPE("MeshContactSurface::collect_points");

            auto               mesh = space->mesh_ptr();
            const ptrdiff_t    n    = node_mapping->size();
            const idx_t *const idx  = node_mapping->data();
            const int          dim  = mesh->spatial_dimension();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = mesh->points(d);
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = x[idx[i]];
                }
            }
        }

        void displace_points(const real_t *disp) {
            SFEM_TRACE_SCOPE("ContactConditions::displace_points");

            auto               mesh = space->mesh_ptr();
            const ptrdiff_t    n    = node_mapping->size();
            const idx_t *const idx  = node_mapping->data();
            const int          dim  = mesh->spatial_dimension();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = mesh->points(d);
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);
                    x_s[i] = x[b] + disp[b * dim + d];
                }
            }
        }
    };
    
    MeshContactSurface::MeshContactSurface() : impl_(std::make_unique<Impl>()) {}
    MeshContactSurface::~MeshContactSurface() = default;
    std::shared_ptr<Buffer<geom_t *>> MeshContactSurface::points() { return impl_->surface_points; }
    std::shared_ptr<Buffer<idx_t *>>  MeshContactSurface::elements() { return impl_->sides; }
    std::shared_ptr<Buffer<idx_t>>    MeshContactSurface::node_mapping() { return impl_->node_mapping; }
    enum ElemType                     MeshContactSurface::element_type() const { return impl_->element_type; }

    void MeshContactSurface::displace_points(const real_t *disp) { impl_->displace_points(disp); }
    void MeshContactSurface::collect_points() { impl_->collect_points(); }

    std::unique_ptr<MeshContactSurface> MeshContactSurface::create(const std::shared_ptr<FunctionSpace> &space,
                                                                   const std::shared_ptr<Sideset>       &sideset,
                                                                   const enum ExecutionSpace             es) {
        auto          mesh = space->mesh_ptr();
        enum ElemType st   = side_type(space->element_type());
        const int     nnxs = elem_num_nodes(st);

        auto sides = sfem::create_host_buffer<idx_t>(nnxs, sideset->parent()->size());
        if (extract_surface_from_sideset(space->element_type(),
                                         mesh->elements()->data(),
                                         sideset->parent()->size(),
                                         sideset->parent()->data(),
                                         sideset->lfi()->data(),
                                         sides->data()) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to extract surface from sideset!\n");
        }

        idx_t    *idx          = nullptr;
        ptrdiff_t n_contiguous = SFEM_PTRDIFF_INVALID;
        remap_elements_to_contiguous_index(sides->extent(1), sides->extent(0), sides->data(), &n_contiguous, &idx);
        auto node_mapping = sfem::manage_host_buffer(n_contiguous, idx);

        // Create object
        auto ret                   = std::make_unique<MeshContactSurface>();
        ret->impl_->space          = space;
        ret->impl_->sideset        = sideset;
        ret->impl_->sides          = sides;
        ret->impl_->node_mapping   = node_mapping;
        ret->impl_->surface_points = create_host_buffer<geom_t>(mesh->spatial_dimension(), node_mapping->size());
        ret->impl_->element_type   = shell_type(side_type(space->element_type()));

        if(es == EXECUTION_SPACE_DEVICE) {
            SFEM_ERROR("IMEPLEMENT ME!\n");
        }

        return ret;
    }

    std::unique_ptr<MeshContactSurface> MeshContactSurface::create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                             const std::string                    &path,
                                                                             const enum ExecutionSpace             es) {
        SFEM_TRACE_SCOPE("MeshContactSurface::create_from_file");
        auto sideset = Sideset::create_from_file(space->mesh_ptr()->comm(), path.c_str());
        return create(space, sideset, es);
    }

    class SSMeshContactSurface::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Sideset>       sideset;

        std::shared_ptr<Buffer<idx_t *>>  sides;
        std::shared_ptr<Buffer<idx_t *>>  semi_structured_sides;
        std::shared_ptr<Buffer<idx_t>>    node_mapping;
        std::shared_ptr<Buffer<geom_t *>> surface_points;
        enum ElemType                     element_type { INVALID };

        void collect_points() {
            SFEM_TRACE_SCOPE("SSMeshContactSurface::collect_points");

            auto &ssmesh   = space->semi_structured_mesh();
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

        void displace_points(const real_t *disp) {
            SFEM_TRACE_SCOPE("ContactConditions::displace_points_semistructured");

            auto &ssmesh   = space->semi_structured_mesh();
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
                    const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);
                    x_s[i] = x[b] + disp[b * dim + d];
                }
            }
        }
    };

    SSMeshContactSurface::SSMeshContactSurface() : impl_(std::make_unique<Impl>()) {}
    SSMeshContactSurface::~SSMeshContactSurface() = default;

    std::shared_ptr<Buffer<geom_t *>> SSMeshContactSurface::points() { return impl_->surface_points; }
    std::shared_ptr<Buffer<idx_t *>>  SSMeshContactSurface::elements() { return impl_->sides; }
    std::shared_ptr<Buffer<idx_t>>    SSMeshContactSurface::node_mapping() { return impl_->node_mapping; }
    enum ElemType                     SSMeshContactSurface::element_type() const { return impl_->element_type; }

    void SSMeshContactSurface::displace_points(const real_t *disp) { impl_->displace_points(disp); }
    void SSMeshContactSurface::collect_points() { impl_->collect_points(); }

    std::unique_ptr<SSMeshContactSurface> SSMeshContactSurface::create(const std::shared_ptr<FunctionSpace> &space,
                                                                       const std::shared_ptr<Sideset>       &sideset,
                                                                       const enum ExecutionSpace             es) {
        assert(es == sfem::EXECUTION_SPACE_HOST);

        auto &ssmesh = space->semi_structured_mesh();
        auto  semi_structured_sides =
                sfem::create_host_buffer<idx_t>((ssmesh.level() + 1) * (ssmesh.level() + 1), sideset->parent()->size());

        if (sshex8_extract_surface_from_sideset(ssmesh.level(),
                                                ssmesh.element_data(),
                                                sideset->parent()->size(),
                                                sideset->parent()->data(),
                                                sideset->lfi()->data(),
                                                semi_structured_sides->data()) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to extract surface from sideset!\n");
        }

        idx_t           *idx          = nullptr;
        ptrdiff_t        n_contiguous = SFEM_PTRDIFF_INVALID;
        std::vector<int> levels(sshex8_hierarchical_n_levels(ssmesh.level()));

        sshex8_hierarchical_mesh_levels(ssmesh.level(), levels.size(), levels.data());

        ssquad4_hierarchical_remapping(ssmesh.level(),
                                       levels.size(),
                                       levels.data(),
                                       semi_structured_sides->extent(1),
                                       ssmesh.n_nodes(),
                                       semi_structured_sides->data(),
                                       &idx,
                                       &n_contiguous);

        auto node_mapping = sfem::manage_host_buffer(n_contiguous, idx);

        const int nnxs  = 4;
        const int nexs  = ssmesh.level() * ssmesh.level();
        auto      sides = sfem::create_host_buffer<idx_t>(nnxs, sideset->parent()->size() * nexs);

        ssquad4_to_standard_quad4_mesh(ssmesh.level(), sideset->parent()->size(), semi_structured_sides->data(), sides->data());

        // Create object
        auto ret                          = std::make_unique<SSMeshContactSurface>();
        ret->impl_->space                 = space;
        ret->impl_->sideset               = sideset;
        ret->impl_->sides                 = sides;
        ret->impl_->semi_structured_sides = semi_structured_sides;
        ret->impl_->node_mapping          = node_mapping;
        ret->impl_->surface_points = create_host_buffer<geom_t>(space->mesh_ptr()->spatial_dimension(), node_mapping->size());
        ret->impl_->element_type   = shell_type(side_type(macro_base_elem(space->element_type())));

        if(es == EXECUTION_SPACE_DEVICE) {
            SFEM_ERROR("IMEPLEMENT ME!\n");
        }

        return ret;
    }

    std::shared_ptr<Buffer<idx_t *>> SSMeshContactSurface::semi_structured_elements() { return impl_->semi_structured_sides; }

    std::shared_ptr<ContactSurface> create_contact_surface(const std::shared_ptr<FunctionSpace> &space,
                                                           const std::shared_ptr<Sideset>       &sideset,
                                                           const enum ExecutionSpace             es) {
        assert(es == sfem::EXECUTION_SPACE_HOST);

        if (space->has_semi_structured_mesh()) {
            return SSMeshContactSurface::create(space, sideset, es);
        } else {
            return MeshContactSurface::create(space, sideset, es);
        }
    }

}  // namespace sfem
