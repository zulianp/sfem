#include "sfem_ContactConditions.hpp"

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"
#include "neumann.h"
#include "sfem_prolongation_restriction.h"

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

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
        std::shared_ptr<Buffer<geom_t>> sdf;
        std::shared_ptr<Buffer<idx_t *>> sides;
        std::shared_ptr<Buffer<idx_t>> node_mapping;
        std::shared_ptr<Buffer<geom_t *>> surface_points;
        
        std::shared_ptr<Buffer<real_t>> gap_xnormal;
        std::shared_ptr<Buffer<real_t>> gap_ynormal;
        std::shared_ptr<Buffer<real_t>> gap_znormal;

        ptrdiff_t nlocal[3];
        ptrdiff_t nglobal[3];
        ptrdiff_t stride[3];
        geom_t origin[3];
        geom_t delta[3];

        std::shared_ptr<FunctionSpace> space;
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

    ptrdiff_t ContactConditions::n_constrained_dofs() const {}

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

    std::shared_ptr<ContactConditions> ContactConditions::create_from_env(
            const std::shared_ptr<FunctionSpace> &space) {
        auto cc = std::make_unique<ContactConditions>(space);
        auto mesh = space->mesh_ptr();
        auto c_mesh = (mesh_t *)mesh->impl_mesh();

        char *SFEM_CONTACT_SURFACE = nullptr;
        char *SFEM_CONTACT_SURFACE_NODE_MAPPING = nullptr;
        char *SFEM_CONTACT_SDF = nullptr;
        SFEM_REQUIRE_ENV(SFEM_CONTACT_SURFACE, );
        SFEM_REQUIRE_ENV(SFEM_CONTACT_SURFACE_NODE_MAPPING, );

        {
            // Read mesh surface information

            const enum ElemType element_type = space->element_type();
            const enum ElemType side_element_type = shell_type(side_type(element_type));
            const int nxe = elem_num_nodes(side_element_type);

            idx_t **sides = (idx_t **)malloc(nxe * sizeof(idx_t *));
            ptrdiff_t _nope_ = -1, len = -1;
            char path[SFEM_MAX_PATH_LENGTH];
            for (int d = 0; d < nxe; d++) {
                sprintf(path, "%s.%d.raw", SFEM_CONTACT_SURFACE, d);

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

            sprintf(path, "%s.node_mapping.raw", SFEM_CONTACT_SURFACE);

            idx_t *idx = nullptr;
            if (!array_create_from_file(
                        mesh->comm(), path, SFEM_MPI_IDX_T, (void **)&idx, &_nope_, &len)) {
                SFEM_ERROR("Unable to read path %s\n", path);
            }

            cc->impl_->node_mapping = Buffer<idx_t>::own(len, idx, &free, sfem::MEMORY_SPACE_HOST);

            // Allocate buffer for point information
            cc->impl_->surface_points = h_buffer<geom_t>(mesh->spatial_dimension(), len);
        }

        {
            // Read SDF
            SFEM_REQUIRE_ENV(SFEM_CONTACT_SDF, );

            ptrdiff_t SFEM_CONTACT_SDF_XDIM = 0;
            ptrdiff_t SFEM_CONTACT_SDF_YDIM = 0;
            ptrdiff_t SFEM_CONTACT_SDF_ZDIM = 0;

            SFEM_REQUIRE_ENV(SFEM_CONTACT_SDF_XDIM, atol);
            SFEM_REQUIRE_ENV(SFEM_CONTACT_SDF_YDIM, atol);
            SFEM_REQUIRE_ENV(SFEM_CONTACT_SDF_ZDIM, atol);

            cc->impl_->nglobal[0] = SFEM_CONTACT_SDF_XDIM;
            cc->impl_->nglobal[1] = SFEM_CONTACT_SDF_YDIM;
            cc->impl_->nglobal[2] = SFEM_CONTACT_SDF_ZDIM;

            ptrdiff_t n = cc->impl_->nglobal[0] * cc->impl_->nglobal[1] * cc->impl_->nglobal[2];

            geom_t *sdf = 0;
            ptrdiff_t nlocal[3];
            if (ndarray_create_from_file(mesh->comm(),
                                         SFEM_CONTACT_SDF,
                                         SFEM_MPI_GEOM_T,
                                         3,
                                         (void **)&sdf,
                                         nlocal,
                                         cc->impl_->nglobal) != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to read SFEM_CONTACT_SDF=%s\n", SFEM_CONTACT_SDF);
                return nullptr;
            }

            cc->impl_->stride[0] = 1;
            cc->impl_->stride[1] = nlocal[0];
            cc->impl_->stride[2] = nlocal[0] * nlocal[1];

            cc->impl_->nlocal[0] = nlocal[0];
            cc->impl_->nlocal[1] = nlocal[1];
            cc->impl_->nlocal[2] = nlocal[2];

            cc->impl_->sdf = Buffer<geom_t>::own(
                    nlocal[0] * nlocal[1] * nlocal[2], sdf, &free, sfem::MEMORY_SPACE_HOST);
        }

        return cc;
    }

    int ContactConditions::apply(real_t *const x) { return apply_value(0, x); }

    int ContactConditions::gradient(const real_t *const x, real_t *const g) {
        impl_->update_displaced_points(x);

        // FIXME for MPI

        // interpolate_gap(
        //         // Mesh
        //         impl_->surface_points->extent(1),
        //         impl_->surface_points->data(),
        //         // SDF
        //         impl_->nlocal,
        //         impl_->stride,
        //         impl_->origin,
        //         impl_->delta,
        //         impl_->sdf->data(),
        //         // Output
        //         g,
        //         impl_->gap_xnormal->data(),
        //         impl_->gap_ynormal->data(),
        //         impl_->gap_znormal->data());

        return SFEM_SUCCESS;
    }

    int ContactConditions::apply_value(const real_t value, real_t *const x) {
        // TODO Project to value
        assert(false);
        return SFEM_FAILURE;
    }

    int ContactConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        // TODO Project to src values
        assert(false);
        return SFEM_FAILURE;
    }

    int ContactConditions::hessian_crs(const real_t *const x, const count_t *const rowptr,
                                       const idx_t *const colidx, real_t *const values) {
        // TODO House-holder matrix
        assert(false);
        return SFEM_FAILURE;
    }

    int ContactConditions::mask(mask_t *mask) {
        assert(false);
        return SFEM_FAILURE;
    }

}  // namespace sfem
