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

#include "node_interpolate.h"
#include "sfem_resample_gap.h"

#include "sfem_API.hpp"

#include <glob.h>
#include <vector>

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

        int                   n_conditions{0};
        boundary_condition_t *conditions{nullptr};
    };

    std::shared_ptr<FunctionSpace> AxisAlignedContactConditions::space() { return impl_->space; }

    int   AxisAlignedContactConditions::n_conditions() const { return impl_->n_conditions; }
    void *AxisAlignedContactConditions::impl_conditions() { return (void *)impl_->conditions; }

    AxisAlignedContactConditions::AxisAlignedContactConditions(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<Constraint> AxisAlignedContactConditions::derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                                                       const bool                            as_zero) const {
        auto mesh = (mesh_t *)impl_->space->mesh().impl_mesh();
        auto et   = (enum ElemType)impl_->space->element_type();

        const ptrdiff_t max_coarse_idx = max_node_id(coarse_space->element_type(), mesh->nelements, mesh->elements);

        auto coarse = std::make_shared<AxisAlignedContactConditions>(coarse_space);

        coarse->impl_->conditions   = (boundary_condition_t *)malloc(impl_->n_conditions * sizeof(boundary_condition_t));
        coarse->impl_->n_conditions = 0;

        for (int i = 0; i < impl_->n_conditions; i++) {
            ptrdiff_t coarse_local_size = 0;
            idx_t    *coarse_indices    = nullptr;
            real_t   *coarse_values     = nullptr;
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
                                                     const ptrdiff_t global_size,
                                                     idx_t *const    idx,
                                                     const int       component,
                                                     const real_t    value) {
        impl_->conditions =
                (boundary_condition_t *)realloc(impl_->conditions, (impl_->n_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(
                &impl_->conditions[impl_->n_conditions], local_size, global_size, idx, component, value, nullptr);

        impl_->n_conditions++;
    }

    void AxisAlignedContactConditions::add_condition(const ptrdiff_t local_size,
                                                     const ptrdiff_t global_size,
                                                     idx_t *const    idx,
                                                     const int       component,
                                                     real_t *const   values) {
        impl_->conditions =
                (boundary_condition_t *)realloc(impl_->conditions, (impl_->n_conditions + 1) * sizeof(boundary_condition_t));

        boundary_condition_create(&impl_->conditions[impl_->n_conditions], local_size, global_size, idx, component, 0, values);

        impl_->n_conditions++;
    }

    std::shared_ptr<AxisAlignedContactConditions> AxisAlignedContactConditions::create_from_env(
            const std::shared_ptr<FunctionSpace> &space) {
        //
        auto dc = std::make_unique<AxisAlignedContactConditions>(space);

        char *SFEM_CONTACT_NODESET   = 0;
        char *SFEM_CONTACT_VALUE     = 0;
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

    int AxisAlignedContactConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
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

    int AxisAlignedContactConditions::hessian_crs(const real_t *const  x,
                                                  const count_t *const rowptr,
                                                  const idx_t *const   colidx,
                                                  real_t *const        values) {
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
                const ptrdiff_t idx =
                        impl_->conditions[i].idx[node] * impl_->space->block_size() + impl_->conditions[i].component;
                mask_set(idx, mask);
            }
        }
        return SFEM_SUCCESS;
    }

    // ===

    class ContactConditions::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Grid<geom_t>>  sdf;

        std::shared_ptr<Buffer<idx_t *>>  sides;
        std::shared_ptr<Buffer<idx_t>>    node_mapping;
        std::shared_ptr<Buffer<geom_t *>> surface_points;

        std::shared_ptr<Buffer<real_t>> gap_xnormal;
        std::shared_ptr<Buffer<real_t>> gap_ynormal;
        std::shared_ptr<Buffer<real_t>> gap_znormal;

        std::shared_ptr<Buffer<real_t>> mass_vector;
        bool                            variational{true};
        bool                            debug{false};

        ~Impl() {}

        void displace_points(const real_t *disp) {
            auto               mesh = space->mesh_ptr();
            const ptrdiff_t    n    = node_mapping->size();
            const idx_t *const idx  = node_mapping->data();
            const int          dim  = mesh->spatial_dimension();

            for (int d = 0; d < dim; d++) {
                const geom_t *const x   = mesh->points(d);
                geom_t *const       x_s = surface_points->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    x_s[i] = x[idx[i]] + disp[idx[i] * dim + d];
                }
            }
        }

        void collect_points() {
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

        void assemble_mass_vector() {
            collect_points();

            auto st           = shell_type(side_type(space->element_type()));
            auto surface_mesh = std::make_shared<Mesh>(space->mesh_ptr()->spatial_dimension(),
                                                       st,
                                                       sides->extent(1),
                                                       sides->data(),
                                                       surface_points->extent(1),
                                                       surface_points->data(),
                                                       [](const void *) {});

            auto trace_space = std::make_shared<FunctionSpace>(surface_mesh, 1);
            auto bop         = sfem::Factory::create_op(trace_space, "Mass");

            mass_vector = h_buffer<real_t>(trace_space->n_dofs());

            if (variational) {
                resample_weight_local(
                        // Mesh
                        st,
                        sides->extent(1),
                        node_mapping->size(),
                        sides->data(),
                        surface_points->data(),
                        // Output
                        mass_vector->data());

            } else {
                auto ones = h_buffer<real_t>(trace_space->n_dofs());
                sfem::blas<real_t>(EXECUTION_SPACE_HOST)->values(trace_space->n_dofs(), 1, ones->data());
                bop->apply(nullptr, ones->data(), mass_vector->data());
            }

            auto m = mass_vector->data();

            real_t area = 0;
            for (ptrdiff_t i = 0; i < mass_vector->size(); i++) {
                area += m[i];
            }

            printf("AREA: %g\n", (double)area);
            assert(area > 0);
        }
    };

    ptrdiff_t ContactConditions::n_constrained_dofs() const { return impl_->node_mapping->size(); }

    const std::shared_ptr<Buffer<idx_t>> &ContactConditions::node_mapping() { return impl_->node_mapping; }

    std::shared_ptr<FunctionSpace> ContactConditions::space() { return impl_->space; }

    ContactConditions::ContactConditions(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    std::shared_ptr<Constraint> ContactConditions::derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                                            const bool                            as_zero) const {
        assert(false);
        return nullptr;
    }

    std::shared_ptr<Constraint> ContactConditions::lor() const {
        assert(false);
        return nullptr;
    }

    ContactConditions::~ContactConditions() = default;

    static void find_files(const char *pattern, std::vector<std::string> &paths) {
        paths.clear();

        glob_t gl;
        glob(pattern, GLOB_MARK, NULL, &gl);

        int n_files = gl.gl_pathc;

        for (int np = 0; np < n_files; np++) {
            paths.push_back(gl.gl_pathv[np]);
        }
    }

    std::shared_ptr<ContactConditions> ContactConditions::create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                           const std::string                    &path) {
        auto in = YAMLNoIndent::create_from_file(path + "/meta.yaml");

        auto cc     = std::make_unique<ContactConditions>(space);
        auto mesh   = space->mesh_ptr();
        auto c_mesh = (mesh_t *)mesh->impl_mesh();

        bool rpath = false;
        in->get("rpath", rpath);

        std::string path_surface;
        in->require("surface", path_surface);

        in->require("variational", cc->impl_->variational);

        if (rpath) {
            path_surface = path + "/" + path_surface;
        }

        auto in_surface = YAMLNoIndent::create_from_file(path_surface + "/meta.yaml");

        std::string points;
        in_surface->require("points", points);

        std::string surface_elem_type;
        in_surface->require("element_type", surface_elem_type);

        {
            // Read mesh surface information
            const enum ElemType element_type      = space->element_type();
            const enum ElemType side_element_type = shell_type(side_type(element_type));
            const int nxe = space->has_semi_structured_mesh() ? elem_num_nodes(type_from_string(surface_elem_type.c_str()))
                                                              : elem_num_nodes(side_element_type);

            assert(space->has_semi_structured_mesh() || type_from_string(surface_elem_type.c_str()) == side_element_type);

            idx_t   **sides  = (idx_t **)malloc(nxe * sizeof(idx_t *));
            ptrdiff_t _nope_ = -1, len = -1;

            char pattern[SFEM_MAX_PATH_LENGTH];
            sprintf(pattern, "%s/i*.*raw", path_surface.c_str());

            std::vector<std::string> paths;
            find_files(pattern, paths);

            assert((int)paths.size() == nxe);

            for (int d = 0; d < nxe; d++) {
                idx_t    *idx   = nullptr;
                ptrdiff_t len_d = -1;
                if (array_create_from_file(mesh->comm(), paths[d].c_str(), SFEM_MPI_IDX_T, (void **)&idx, &_nope_, &len_d)) {
                    SFEM_ERROR("Unable to read path %s\n", paths[d].c_str());
                }

                sides[d] = idx;

                assert(len == -1 || len_d == len);
                len = len_d;
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

            bool has_parent_indexing = points == "parent";
            if (has_parent_indexing) {
                idx_t    *idx          = nullptr;
                ptrdiff_t n_contiguous = -1;
                remap_elements_to_contiguous_index(
                        cc->impl_->sides->extent(1), cc->impl_->sides->extent(0), cc->impl_->sides->data(), &n_contiguous, &idx);
                cc->impl_->node_mapping = Buffer<idx_t>::own(n_contiguous, idx, &free, sfem::MEMORY_SPACE_HOST);

            } else {
                std::string path_node_mapping = path_surface + "/node_mapping.raw";
                in_surface->get("node_mapping", path_node_mapping);

                idx_t *idx = nullptr;
                if (array_create_from_file(
                            mesh->comm(), path_node_mapping.c_str(), SFEM_MPI_IDX_T, (void **)&idx, &_nope_, &len)) {
                    SFEM_ERROR("Unable to read path %s\n", path_node_mapping.c_str());
                }

                cc->impl_->node_mapping = Buffer<idx_t>::own(len, idx, &free, sfem::MEMORY_SPACE_HOST);
            }

            // Allocate buffer for point information
            cc->impl_->surface_points = h_buffer<geom_t>(mesh->spatial_dimension(), cc->impl_->node_mapping->size());
        }

        {  // SDF

            std::string path_sdf;
            in->require("sdf", path_sdf);

            if (rpath) {
                path_sdf = path + "/" + path_sdf;
            }

            cc->impl_->sdf = Grid<geom_t>::create_from_file(mesh->comm(), path_sdf.c_str());
        }

        cc->impl_->gap_xnormal = h_buffer<real_t>(cc->n_constrained_dofs());
        cc->impl_->gap_ynormal = h_buffer<real_t>(cc->n_constrained_dofs());
        cc->impl_->gap_znormal = h_buffer<real_t>(cc->n_constrained_dofs());

        cc->impl_->assemble_mass_vector();
        return cc;
    }

    std::shared_ptr<ContactConditions> ContactConditions::create_from_env(const std::shared_ptr<FunctionSpace> &space) {
        char *SFEM_CONTACT = nullptr;
        SFEM_REQUIRE_ENV(SFEM_CONTACT, );
        return create_from_file(space, SFEM_CONTACT);
    }

    int ContactConditions::apply(real_t *const x) { return apply_value(0, x); }

    int ContactConditions::signed_distance_for_mesh_viz(const real_t *const x, real_t *const g) const {
        impl_->displace_points(x);

        auto sdf = impl_->sdf;

        auto temp = h_buffer<real_t>(n_constrained_dofs());

        auto tt = temp->data();

        int err = 0;
        if (impl_->variational) {
            auto st = shell_type(side_type(impl_->space->element_type()));
            err     = resample_gap(
                    // Mesh
                    st,
                    impl_->sides->extent(1),
                    impl_->node_mapping->size(),
                    impl_->sides->data(),
                    impl_->surface_points->data(),
                    // SDF
                    sdf->nlocal(),
                    sdf->stride(),
                    sdf->origin(),
                    sdf->delta(),
                    sdf->data(),
                    // Output
                    tt,
                    impl_->gap_xnormal->data(),
                    impl_->gap_ynormal->data(),
                    impl_->gap_znormal->data());
        } else {
            err = interpolate_gap(
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
                    tt,
                    impl_->gap_xnormal->data(),
                    impl_->gap_ynormal->data(),
                    impl_->gap_znormal->data());
        }

        assert(err == SFEM_SUCCESS);

        const ptrdiff_t    n   = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();
        int                dim = impl_->space->mesh_ptr()->spatial_dimension();

        const real_t *const normals[3] = {impl_->gap_xnormal->data(), impl_->gap_ynormal->data(), impl_->gap_znormal->data()};

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            for (int d = 0; d < dim; d++) {
                g[idx[i] * dim + d] = tt[i] * normals[d][i];
            }
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::normal_project(const real_t *const h, real_t *const out) {
        const ptrdiff_t    n   = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        assert(dim == 3);  // FIXME 2D not supported

        const real_t *const normals[3] = {impl_->gap_xnormal->data(), impl_->gap_ynormal->data(), impl_->gap_znormal->data()};

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            for (int d = 0; d < dim; d++) {
                out[i] += h[idx[i] * dim + d] * normals[d][i];
            }
        }

        if (impl_->debug) {
            for (ptrdiff_t i = 0; i < n; ++i) {
                printf("CC: %g = %g * %g\n", out[i], normals[0][i], h[idx[i] * dim + 0]);
            }
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::distribute_contact_forces(const real_t *const f, real_t *const out) {
        const ptrdiff_t    n   = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        assert(dim == 3);  // FIXME 2D not supported

        const real_t *const normals[3] = {impl_->gap_xnormal->data(), impl_->gap_ynormal->data(), impl_->gap_znormal->data()};

        auto m = impl_->mass_vector->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t fi = f[i] * m[i];
            for (int d = 0; d < dim; d++) {
                out[idx[i] * dim + d] += normals[d][i] * fi;
            }
        }

        if (impl_->debug) {
            for (ptrdiff_t i = 0; i < n; ++i) {
                printf("CC_t: %g = %g  * %g * %g\n", out[idx[i] * dim + 0], normals[0][i], f[i], m[i]);
            }
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        return update(x) || normal_project(h, out);
    }

    int ContactConditions::init() {
        auto sdf = impl_->sdf;

        if (impl_->variational) {
            auto st = shell_type(side_type(impl_->space->element_type()));
            return resample_gap_normals(
                    // Mesh
                    st,
                    impl_->sides->extent(1),
                    impl_->node_mapping->size(),
                    impl_->sides->data(),
                    impl_->surface_points->data(),
                    // SDF
                    sdf->nlocal(),
                    sdf->stride(),
                    sdf->origin(),
                    sdf->delta(),
                    sdf->data(),
                    // Output
                    impl_->gap_xnormal->data(),
                    impl_->gap_ynormal->data(),
                    impl_->gap_znormal->data());
        } else {
            return interpolate_gap_normals(
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
                    impl_->gap_xnormal->data(),
                    impl_->gap_ynormal->data(),
                    impl_->gap_znormal->data());
        }
    }

    int ContactConditions::update(const real_t *const x) {
        impl_->displace_points(x);
        return init();
    }

    std::shared_ptr<Operator<real_t>> ContactConditions::linear_constraints_op() {
        auto space = impl_->space;
        return make_op<real_t>(
                this->n_constrained_dofs(),
                space->n_dofs(),
                [=](const real_t *const h, real_t *const out) { normal_project(h, out); },
                EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<Operator<real_t>> ContactConditions::linear_constraints_op_transpose() {
        auto space = impl_->space;
        return make_op<real_t>(
                space->n_dofs(),
                this->n_constrained_dofs(),
                [=](const real_t *const f, real_t *const out) { distribute_contact_forces(f, out); },
                EXECUTION_SPACE_HOST);
    }

    int ContactConditions::signed_distance(real_t *const g) {
        auto sdf = impl_->sdf;

        int err = 0;
        if (impl_->variational) {
            auto st = shell_type(side_type(impl_->space->element_type()));
            err     = resample_gap_value(
                    // Mesh
                    st,
                    impl_->sides->extent(1),
                    impl_->node_mapping->size(),
                    impl_->sides->data(),
                    impl_->surface_points->data(),
                    // SDF
                    sdf->nlocal(),
                    sdf->stride(),
                    sdf->origin(),
                    sdf->delta(),
                    sdf->data(),
                    g);

        } else {
            err = interpolate_gap_value(
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
                    g);
        }

        assert(err == SFEM_SUCCESS);
        return err;
    }

    int ContactConditions::signed_distance(const real_t *const disp, real_t *const g) {
        impl_->displace_points(disp);
        return signed_distance(g);
    }

    int ContactConditions::gradient(const real_t *const x, real_t *const g) {
        int err = SFEM_SUCCESS;
        if (impl_->variational) {
            auto sdf = impl_->sdf;
            auto st  = shell_type(side_type(impl_->space->element_type()));
            err      = resample_gap_value_local(
                    // Mesh
                    st,
                    impl_->sides->extent(1),
                    impl_->node_mapping->size(),
                    impl_->sides->data(),
                    impl_->surface_points->data(),
                    // SDF
                    sdf->nlocal(),
                    sdf->stride(),
                    sdf->origin(),
                    sdf->delta(),
                    sdf->data(),
                    g);
        } else {
            err = signed_distance(x, g);
            assert(err == SFEM_SUCCESS);

            ptrdiff_t n = impl_->mass_vector->size();
            auto      m = impl_->mass_vector->data();
            for (ptrdiff_t i = 0; i < n; i++) {
                g[i] *= m[i];
            }
        }

        return err;
    }

    int ContactConditions::apply_value(const real_t value, real_t *const x) {
        const ptrdiff_t    n   = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            x[idx[i]] = value;
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::copy_constrained_dofs(const real_t *const src, real_t *const dest) {
        const ptrdiff_t    n   = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            dest[idx[i]] = src[idx[i]];
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::hessian_crs(const real_t *const  x,
                                       const count_t *const rowptr,
                                       const idx_t *const   colidx,
                                       real_t *const        values) {
        // TODO Householder matrix?
        assert(false);
        return SFEM_FAILURE;
    }

    int ContactConditions::mask(mask_t *mask) {
        const ptrdiff_t    n   = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            mask_set(idx[i], mask);
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::hessian_block_diag_sym(const real_t *const x, real_t *const values) {

        // if(x)
            // impl_->displace_points(x);

        const ptrdiff_t    n   = impl_->node_mapping->size();
        const idx_t *const idx = impl_->node_mapping->data();

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        assert(dim == 3);  // FIXME 2D not supported

        const real_t *const normals[3] = {impl_->gap_xnormal->data(), impl_->gap_ynormal->data(), impl_->gap_znormal->data()};

        auto m = impl_->mass_vector->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            real_t *const v = &values[i * 6];

            int d_idx = 0;
            for (int d1 = 0; d1 < dim; d1++) {
                for (int d2 = d1; d2 < dim; d2++) {
                    v[d_idx++] += m[i] * normals[d1][i] * normals[d2][i];
                }
            }
        }

        if (false) {
            const idx_t *const idx = impl_->node_mapping->data();

            for (ptrdiff_t i = 0; i < n; ++i) {
                printf("%d) ", idx[i]);
                real_t *const v = &values[i * 6];
                for (int d = 0; d < 6; d++) {
                    printf("%g\t", (double)v[d]);
                }
                printf("\n");
            }
        }

        return SFEM_SUCCESS;
    }

}  // namespace sfem
