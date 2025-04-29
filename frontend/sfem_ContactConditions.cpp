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

#include "adj_table.h"
#include "sfem_hex8_mesh_graph.h"
#include "sfem_sshex8_skin.h"
#include "sshex8_mesh.h"

#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#include "sfem_ContactSurface.hpp"
#include "sfem_Function.hpp"
#include "sfem_SDFObstacle.hpp"

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
        std::shared_ptr<FunctionSpace>         space;
        std::vector<std::shared_ptr<Obstacle>> obstacles;
        std::shared_ptr<ContactSurface>        contact_surface;
        std::shared_ptr<Buffer<real_t *>>      normals;
        std::shared_ptr<Buffer<real_t>>        mass_vector;
        bool                                   debug{false};
        bool                                   variational{true};

        ~Impl() {}

        void assemble_mass_vector() {
            SFEM_TRACE_SCOPE("ContactConditions::assemble_mass_vector");

            contact_surface->collect_points();

            auto st           = contact_surface->element_type();
            auto surface_mesh = std::make_shared<Mesh>(space->mesh_ptr()->spatial_dimension(),
                                                       st,
                                                       contact_surface->elements()->extent(1),
                                                       contact_surface->elements()->data(),
                                                       contact_surface->points()->extent(1),
                                                       contact_surface->points()->data(),
                                                       [](const void *) {});

            auto trace_space = std::make_shared<FunctionSpace>(surface_mesh, 1);
            auto bop         = sfem::Factory::create_op(trace_space, "Mass");

            mass_vector = create_host_buffer<real_t>(trace_space->n_dofs());

            if (variational) {
                resample_weight_local(
                        // Mesh
                        st,
                        contact_surface->elements()->extent(1),
                        contact_surface->node_mapping()->size(),
                        contact_surface->elements()->data(),
                        contact_surface->points()->data(),
                        // Output
                        mass_vector->data());

            } else {
                auto ones = create_host_buffer<real_t>(trace_space->n_dofs());
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

    std::shared_ptr<Buffer<idx_t *>> ContactConditions::ss_sides() { return impl_->contact_surface->semi_structured_elements(); }

    ptrdiff_t ContactConditions::n_constrained_dofs() const { return impl_->contact_surface->node_mapping()->size(); }

    const std::shared_ptr<Buffer<idx_t>> ContactConditions::node_mapping() { return impl_->contact_surface->node_mapping(); }

    std::shared_ptr<FunctionSpace> ContactConditions::space() { return impl_->space; }

    ContactConditions::ContactConditions(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    ContactConditions::~ContactConditions() = default;

    std::shared_ptr<ContactConditions> ContactConditions::create(const std::shared_ptr<FunctionSpace> &space,
                                                                 const std::shared_ptr<Grid<geom_t>>  &sdf,
                                                                 const std::shared_ptr<Sideset>       &sideset,
                                                                 const enum ExecutionSpace             es) {
        auto cc = std::make_unique<ContactConditions>(space);
        cc->impl_->obstacles.push_back(SDFObstacle::create(sdf, es));

        if (space->has_semi_structured_mesh()) {
            cc->impl_->contact_surface = SSMeshContactSurface::create(space, sideset, es);
        } else {
            cc->impl_->contact_surface = MeshContactSurface::create(space, sideset, es);
        }

        cc->impl_->normals = create_host_buffer<real_t>(space->mesh_ptr()->spatial_dimension(), cc->n_constrained_dofs());
        cc->impl_->assemble_mass_vector();
        return cc;
    }

    std::shared_ptr<ContactConditions> ContactConditions::create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                           const std::string                    &path,
                                                                           const enum ExecutionSpace             es) {
        SFEM_TRACE_SCOPE("ContactConditions::create_from_file");

        auto in   = YAMLNoIndent::create_from_file(path + "/meta.yaml");
        auto mesh = space->mesh_ptr();

        bool rpath = false;
        in->get("rpath", rpath);

        std::string path_surface;
        in->require("surface", path_surface);

        if (rpath) {
            path_surface = path + "/" + path_surface;
        }

        auto in_surface = YAMLNoIndent::create_from_file(path_surface + "/meta.yaml");

        auto sideset = Sideset::create_from_file(space->mesh_ptr()->comm(), path_surface.c_str());

        std::string path_sdf;
        in->require("sdf", path_sdf);

        if (rpath) {
            path_sdf = path + "/" + path_sdf;
        }

        auto sdf = Grid<geom_t>::create_from_file(mesh->comm(), path_sdf.c_str());

        return create(space, std::move(sdf), sideset, es);
    }

    std::shared_ptr<ContactConditions> ContactConditions::create_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                          const enum ExecutionSpace             es) {
        char *SFEM_CONTACT = nullptr;
        SFEM_REQUIRE_ENV(SFEM_CONTACT, );
        return create_from_file(space, SFEM_CONTACT, es);
    }

    int ContactConditions::signed_distance_for_mesh_viz(const real_t *const x, real_t *const g) const {
        auto cs = impl_->contact_surface;
        cs->displace_points(x);

        auto temp = create_host_buffer<real_t>(n_constrained_dofs());
        auto tt   = temp->data();

        for (auto &obs : impl_->obstacles) {
            int err = obs->sample(cs->element_type(),
                                  cs->elements()->extent(1),
                                  cs->node_mapping()->size(),
                                  cs->elements()->data(),
                                  cs->points()->data(),
                                  impl_->normals->data(),
                                  tt);

            if (SFEM_SUCCESS != err) {
                SFEM_ERROR("Unable to sample obstacle");
            }
        }

        const ptrdiff_t    n   = impl_->contact_surface->node_mapping()->size();
        const idx_t *const idx = impl_->contact_surface->node_mapping()->data();
        int                dim = impl_->space->mesh_ptr()->spatial_dimension();

        auto normals = impl_->normals->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            for (int d = 0; d < dim; d++) {
                g[idx[i] * dim + d] = tt[i] * normals[d][i];
            }
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::normal_project(const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("ContactConditions::normal_project");

        const ptrdiff_t    n   = impl_->contact_surface->node_mapping()->size();
        const idx_t *const idx = impl_->contact_surface->node_mapping()->data();

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        auto normals = impl_->normals->data();

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
        SFEM_TRACE_SCOPE("ContactConditions::distribute_contact_forces");
        const ptrdiff_t    n   = impl_->contact_surface->node_mapping()->size();
        const idx_t *const idx = impl_->contact_surface->node_mapping()->data();

        const int dim     = impl_->space->mesh_ptr()->spatial_dimension();
        auto      normals = impl_->normals->data();

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

    int ContactConditions::full_apply_boundary_mass_inverse(const real_t *const r, real_t *const s) {
        const ptrdiff_t    n   = impl_->contact_surface->node_mapping()->size();
        const idx_t *const idx = impl_->contact_surface->node_mapping()->data();
        auto               m   = impl_->mass_vector->data();
        const int          dim = impl_->space->mesh_ptr()->spatial_dimension();

        auto normals = impl_->normals->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            for (int d = 0; d < dim; d++) {
                const real_t ri = r[idx[i] * dim + d] / m[i];
                s[idx[i] * dim] += normals[d][i] * ri;
            }
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::init() {
        SFEM_TRACE_SCOPE("ContactConditions::init");

        auto cs = impl_->contact_surface;

        int err = 0;
        for (auto &obs : impl_->obstacles) {
            // FIXME always sample gap and normals together
            err += obs->sample_normals(cs->element_type(),
                                       cs->elements()->extent(1),
                                       cs->node_mapping()->size(),
                                       cs->elements()->data(),
                                       cs->points()->data(),
                                       impl_->normals->data());
        }

        return err;
    }

    int ContactConditions::update(const real_t *const x) {
        impl_->contact_surface->displace_points(x);
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
        SFEM_TRACE_SCOPE("ContactConditions::signed_distance");

        auto cs = impl_->contact_surface;

        int err = 0;
        for (auto &obs : impl_->obstacles) {
            // FIXME always sample gap and normals together
            err += obs->sample_value(cs->element_type(),
                                     cs->elements()->extent(1),
                                     cs->node_mapping()->size(),
                                     cs->elements()->data(),
                                     cs->points()->data(),
                                     g);
        }

        return err;
    }

    int ContactConditions::signed_distance(const real_t *const disp, real_t *const g) {
        impl_->contact_surface->displace_points(disp);
        return signed_distance(g);
    }

    int ContactConditions::mask(mask_t *mask) {
        const ptrdiff_t    n   = impl_->contact_surface->node_mapping()->size();
        const idx_t *const idx = impl_->contact_surface->node_mapping()->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            mask_set(idx[i], mask);
        }

        return SFEM_SUCCESS;
    }

    int ContactConditions::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("ContactConditions::hessian_block_diag_sym");

        const ptrdiff_t    n   = impl_->contact_surface->node_mapping()->size();
        const idx_t *const idx = impl_->contact_surface->node_mapping()->data();

        const int dim     = impl_->space->mesh_ptr()->spatial_dimension();
        auto      normals = impl_->normals->data();

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

        return SFEM_SUCCESS;
    }

}  // namespace sfem
