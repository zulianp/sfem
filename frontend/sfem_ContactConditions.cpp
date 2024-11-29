#include "sfem_ContactConditions.hpp"

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"
#include "neumann.h"
#include "sfem_prolongation_restriction.h"

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "sfem_Grid.hpp"
#include "sfem_Input.hpp"

#include "sfem_resample_gap.h"

namespace sfem {
    class AxisAlignedContactConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;

        ~Impl() {
            if (conditions) {
                for (int i = 0; i < n_conditions; i++) {
                    free(conditions[i].idx);
                }

                free(conditions);
            }
        }

        int n_conditions{0};
        boundary_condition_t *conditions{nullptr};
    };

    std::shared_ptr<FunctionSpace> AxisAlignedContactConditions::space() { return impl_->space; }

    int AxisAlignedContactConditions::n_conditions() const { return impl_->n_conditions; }
    void *AxisAlignedContactConditions::impl_conditions() { return (void *)impl_->conditions; }

    AxisAlignedContactConditions::AxisAlignedContactConditions(
            const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<Constraint> AxisAlignedContactConditions::derefine(
            const std::shared_ptr<FunctionSpace> &coarse_space, const bool as_zero) const {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        auto et = (enum ElemType)impl_->space->element_type();

        const ptrdiff_t max_coarse_idx =
                max_node_id(coarse_space->element_type(), mesh->nelements, mesh->elements);

        auto coarse = std::make_shared<AxisAlignedContactConditions>(coarse_space);

        coarse->impl_->conditions =
                (boundary_condition_t *)malloc(impl_->n_conditions * sizeof(boundary_condition_t));
        coarse->impl_->n_conditions = 0;

        for (int i = 0; i < impl_->n_conditions; i++) {
            ptrdiff_t coarse_local_size = 0;
            idx_t *coarse_indices = nullptr;
            real_t *coarse_values = nullptr;
            hierarchical_create_coarse_indices(max_coarse_idx,
                                               impl_->conditions[i].local_size,
                                               impl_->conditions[i].idx,
                                               &coarse_local_size,
                                               &coarse_indices);

            if (!as_zero && impl_->conditions[i].values) {
                coarse_values = (real_t *)malloc(coarse_local_size * sizeof(real_t));

                hierarchical_collect_coarse_values(max_coarse_idx,
                                                   impl_->conditions[i].local_size,
                                                   impl_->conditions[i].idx,
                                                   impl_->conditions[i].values,
                                                   coarse_values);
            }

            long coarse_global_size = coarse_local_size;

            // MPI_CATCH_ERROR(
            // MPI_Allreduce(MPI_IN_PLACE, &coarse_global_size, 1, MPI_LONG, MPI_SUM, mesh->comm));

            if (as_zero) {
                boundary_condition_create(&coarse->impl_->conditions[coarse->impl_->n_conditions++],
                                          coarse_local_size,
                                          coarse_global_size,
                                          coarse_indices,
                                          impl_->conditions[i].component,
                                          0,
                                          nullptr);

            } else {
                boundary_condition_create(&coarse->impl_->conditions[coarse->impl_->n_conditions++],
                                          coarse_local_size,
                                          coarse_global_size,
                                          coarse_indices,
                                          impl_->conditions[i].component,
                                          impl_->conditions[i].value,
                                          coarse_values);
            }
        }

        return coarse;
    }

    std::shared_ptr<Constraint> AxisAlignedContactConditions::lor() const {
        assert(false);
        return nullptr;
    }

    AxisAlignedContactConditions::~AxisAlignedContactConditions() = default;

    void AxisAlignedContactConditions::add_condition(const ptrdiff_t local_size,
                                                     const ptrdiff_t global_size, idx_t *const idx,
                                                     const int component, const real_t value) {
        impl_->conditions = (boundary_condition_t *)realloc(
                impl_->conditions, (impl_->n_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(&impl_->conditions[impl_->n_conditions],
                                  local_size,
                                  global_size,
                                  idx,
                                  component,
                                  value,
                                  nullptr);

        impl_->n_conditions++;
    }

    void AxisAlignedContactConditions::add_condition(const ptrdiff_t local_size,
                                                     const ptrdiff_t global_size, idx_t *const idx,
                                                     const int component, real_t *const values) {
        impl_->conditions = (boundary_condition_t *)realloc(
                impl_->conditions, (impl_->n_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(&impl_->conditions[impl_->n_conditions],
                                  local_size,
                                  global_size,
                                  idx,
                                  component,
                                  0,
                                  values);

        impl_->n_conditions++;
    }

    std::shared_ptr<AxisAlignedContactConditions> AxisAlignedContactConditions::create_from_env(
            const std::shared_ptr<FunctionSpace> &space) {
        //
        auto dc = std::make_unique<AxisAlignedContactConditions>(space);

        char *SFEM_CONTACT_NODESET = 0;
        char *SFEM_CONTACT_VALUE = 0;
        char *SFEM_CONTACT_COMPONENT = 0;
        SFEM_READ_ENV(SFEM_CONTACT_NODESET, );
        SFEM_READ_ENV(SFEM_CONTACT_VALUE, );
        SFEM_READ_ENV(SFEM_CONTACT_COMPONENT, );

        auto mesh = (mesh_t *)space->mesh().impl_mesh();
        read_dirichlet_conditions(mesh,
                                  SFEM_CONTACT_NODESET,
                                  SFEM_CONTACT_VALUE,
                                  SFEM_CONTACT_COMPONENT,
                                  &dc->impl_->conditions,
                                  &dc->impl_->n_conditions);

        return dc;
    }

    int AxisAlignedContactConditions::apply(real_t *const x) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            if (impl_->conditions[i].values) {
                constraint_nodes_to_values_vec(impl_->conditions[i].local_size,
                                               impl_->conditions[i].idx,
                                               impl_->space->block_size(),
                                               impl_->conditions[i].component,
                                               impl_->conditions[i].values,
                                               x);

            } else {
                constraint_nodes_to_value_vec(impl_->conditions[i].local_size,
                                              impl_->conditions[i].idx,
                                              impl_->space->block_size(),
                                              impl_->conditions[i].component,
                                              impl_->conditions[i].value,
                                              x);
            }
        }

        return SFEM_SUCCESS;
    }

    int AxisAlignedContactConditions::gradient(const real_t *const x, real_t *const g) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            constraint_gradient_nodes_to_value_vec(impl_->conditions[i].local_size,
                                                   impl_->conditions[i].idx,
                                                   impl_->space->block_size(),
                                                   impl_->conditions[i].component,
                                                   impl_->conditions[i].value,
                                                   x,
                                                   g);
        }

        return SFEM_SUCCESS;
    }

    int AxisAlignedContactConditions::apply_value(const real_t value, real_t *const x) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            constraint_nodes_to_value_vec(impl_->conditions[i].local_size,
                                          impl_->conditions[i].idx,
                                          impl_->space->block_size(),
                                          impl_->conditions[i].component,
                                          value,
                                          x);
        }

        return SFEM_SUCCESS;
    }

    int AxisAlignedContactConditions::copy_constrained_dofs(const real_t *const src,
                                                            real_t *const dest) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            constraint_nodes_copy_vec(impl_->conditions[i].local_size,
                                      impl_->conditions[i].idx,
                                      impl_->space->block_size(),
                                      impl_->conditions[i].component,
                                      src,
                                      dest);
        }

        return SFEM_SUCCESS;
    }

    int AxisAlignedContactConditions::hessian_crs(const real_t *const x,
                                                  const count_t *const rowptr,
                                                  const idx_t *const colidx, real_t *const values) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            crs_constraint_nodes_to_identity_vec(impl_->conditions[i].local_size,
                                                 impl_->conditions[i].idx,
                                                 impl_->space->block_size(),
                                                 impl_->conditions[i].component,
                                                 1,
                                                 rowptr,
                                                 colidx,
                                                 values);
        }

        return SFEM_SUCCESS;
    }

    int AxisAlignedContactConditions::mask(mask_t *mask) {
        for (int i = 0; i < impl_->n_conditions; i++) {
            for (ptrdiff_t node = 0; node < impl_->conditions[i].local_size; node++) {
                const ptrdiff_t idx = impl_->conditions[i].idx[node] * impl_->space->block_size() +
                                      impl_->conditions[i].component;
                mask_set(idx, mask);
            }
        }
        return SFEM_SUCCESS;
    }

    // ===

    class ContactConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Grid<geom_t>> sdf;

        std::shared_ptr<Buffer<idx_t *>> sides;
        std::shared_ptr<Buffer<idx_t>> node_mapping;
        std::shared_ptr<Buffer<geom_t *>> surface_points;

        std::shared_ptr<Buffer<real_t>> gap_xnormal;
        std::shared_ptr<Buffer<real_t>> gap_ynormal;
        std::shared_ptr<Buffer<real_t>> gap_znormal;

        ~Impl() {}

        void update_displaced_points(const real_t *disp) {
            auto mesh = space->mesh_ptr();
            const ptrdiff_t n = node_mapping->size();
            const idx_t *const idx = node_mapping->data();
            const int dim = mesh->spatial_dimension();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x = mesh->points(d);
                geom_t *const x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = x[idx[i]] + disp[idx[i] * dim + d];
                }
            }
        }
    };

    ptrdiff_t ContactConditions::n_constrained_dofs() const { return impl_->node_mapping->size(); }

    std::shared_ptr<FunctionSpace> ContactConditions::space() { return impl_->space; }

    ContactConditions::ContactConditions(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<Constraint> ContactConditions::derefine(
            const std::shared_ptr<FunctionSpace> &coarse_space, const bool as_zero) const {
        assert(false);
        return nullptr;
    }

    std::shared_ptr<Constraint> ContactConditions::lor() const {
        assert(false);
        return nullptr;
    }

    ContactConditions::~ContactConditions() = default;

    std::shared_ptr<ContactConditions> ContactConditions::create_from_file(
            const std::shared_ptr<FunctionSpace> &space, const std::string &path) {
        auto in = YAMLNoIndent::create_from_file(path + "/meta.yaml");

        auto cc = std::make_unique<ContactConditions>(space);
        auto mesh = space->mesh_ptr();
        auto c_mesh = (mesh_t *)mesh->impl_mesh();

        std::string path_surface;
        in->require("surface", path_surface);

        std::string path_node_mapping = path_surface + "/node_mapping.raw";
        in->get("node_mapping", path_node_mapping);

        std::string surface_elem_type;
        in->require("element_type", surface_elem_type);

        std::string path_sdf;
        in->require("sdf", path_sdf);

        {
            // Read mesh surface information
            const enum ElemType element_type = space->element_type();
            const enum ElemType side_element_type = shell_type(side_type(element_type));
            const int nxe = elem_num_nodes(side_element_type);

            assert(type_from_string(surface_elem_type.c_str()) == side_element_type);

            idx_t **sides = (idx_t **)malloc(nxe * sizeof(idx_t *));
            ptrdiff_t _nope_ = -1, len = -1;
            char path[SFEM_MAX_PATH_LENGTH];
            for (int d = 0; d < nxe; d++) {
                sprintf(path, "%s.%d.raw", path_surface.c_str(), d);

                idx_t *idx = nullptr;
                if (!array_create_from_file(
                            mesh->comm(), path, SFEM_MPI_IDX_T, (void **)&idx, &_nope_, &len)) {
                    SFEM_ERROR("Unable to read path %s\n", path);
                }
            }

            cc->impl_->sides = Buffer<idx_t *>::own(
                    nxe,
                    len,
                    sides,
                    [=](int n, void **x) {
                        for (int i = 0; i < n; ++i) {
                            free(x[i]);
                        }
                        free(x);
                    },
                    sfem::MEMORY_SPACE_HOST);

            idx_t *idx = nullptr;
            if (!array_create_from_file(mesh->comm(),
                                        path_node_mapping.c_str(),
                                        SFEM_MPI_IDX_T,
                                        (void **)&idx,
                                        &_nope_,
                                        &len)) {
                SFEM_ERROR("Unable to read path %s\n", path_node_mapping.c_str());
            }

            cc->impl_->node_mapping = Buffer<idx_t>::own(len, idx, &free, sfem::MEMORY_SPACE_HOST);

            // Allocate buffer for point information
            cc->impl_->surface_points = h_buffer<geom_t>(mesh->spatial_dimension(), len);
        }

        cc->impl_->sdf = Grid<geom_t>::create_from_file(mesh->comm(), path_sdf.c_str());
        return cc;
    }

    std::shared_ptr<ContactConditions> ContactConditions::create_from_env(
            const std::shared_ptr<FunctionSpace> &space) {
        char *SFEM_CONTACT = nullptr;
        SFEM_REQUIRE_ENV(SFEM_CONTACT, );
        return create_from_file(space, SFEM_CONTACT);
    }

    int ContactConditions::apply(real_t *const x) { return apply_value(0, x); }

    int ContactConditions::gradient(const real_t *const x, real_t *const g) {
        impl_->update_displaced_points(x);

        auto sdf = impl_->sdf;

        interpolate_gap(
                // Mesh
                impl_->surface_points->extent(1),
                impl_->surface_points->data(),
                // SDF
                sdf->nlocal(),
                sdf->stride(),
                sdf->origin(),
                sdf->delta(),
                sdf->data(),
                // Output
                g,
                impl_->gap_xnormal->data(),
                impl_->gap_ynormal->data(),
                impl_->gap_znormal->data());

        return SFEM_SUCCESS;
    }

    int ContactConditions::apply_value(const real_t value, real_t *const x) {
        const ptrdiff_t n = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            x[idx[i]] = value;
        }

        return SFEM_FAILURE;
    }

    int ContactConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        const ptrdiff_t n = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            dest[idx[i]] = src[idx[i]];
        }

        return SFEM_FAILURE;
    }

    int ContactConditions::hessian_crs(const real_t *const x, const count_t *const rowptr,
                                       const idx_t *const colidx, real_t *const values) {
        // TODO Householder matrix
        assert(false);
        return SFEM_FAILURE;
    }

    int ContactConditions::mask(mask_t *mask) {
        const ptrdiff_t n = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            mask_set(idx[i], mask);
        }

        return SFEM_SUCCESS;
    }

}  // namespace sfem
