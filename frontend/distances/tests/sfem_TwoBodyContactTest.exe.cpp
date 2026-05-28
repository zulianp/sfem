#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"
// #include "sfem_SelfCollisions.hpp"

#include "integrations/smesh/sccd_smesh_CCD.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include "sfem_API.hpp"

#include "bvh/bvh.hpp"

#include <algorithm>
#include <utility>
#include <vector>

using namespace sfem;

static SFEM_INLINE void tri3_find_cols(const idx_t* const SFEM_RESTRICT targets,
                                       const idx_t* const SFEM_RESTRICT row,
                                       const int                        lenrow,
                                       idx_t* const SFEM_RESTRICT       ks) {
#pragma unroll(3)
    for (int d = 0; d < 3; ++d) {
        ks[d] = 0;
    }

    for (int i = 0; i < lenrow; ++i) {
#pragma unroll(3)
        for (int d = 0; d < 3; ++d) {
            ks[d] += row[i] < targets[d];
        }
    }
}

std::shared_ptr<Function> create_function(const ptrdiff_t nx, const ExecutionSpace es) {
    auto mesh1 = smesh::Mesh::create_tet4_cube(Communicator::self(), nx, nx, nx, 0, 0, 0, 1, 1, 1);
    auto mesh2 = smesh::Mesh::create_tet4_cube(Communicator::self(), nx, nx, nx, 0.1, 1.1, 0.1, 0.9, 1.9, 0.9);

    auto mesh = smesh::concatenate(mesh1, mesh2);

    printf("Bulk: #nodes %zu #elements %zu\n", mesh->n_nodes(), mesh->n_elements());

    mesh->write(smesh::Path("contact_mesh"));

    auto top_ss = sfem::Sideset::create_from_selector(mesh, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool {
        return y > (1.9 - 1e-4) && y < (1.9 + 1e-4);
    });

    auto bottom_ss = sfem::Sideset::create_from_selector(
            mesh, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > (-1e-4) && y < (1e-4); });

    const int dim   = mesh->spatial_dimension();
    auto      space = FunctionSpace::create(mesh, dim);

    auto op = create_op(space, "LinearElasticity", es);
    op->initialize();

    auto f = Function::create(space);
    f->add_operator(op);

    auto top_ns    = smesh::create_nodeset_from_sidesets(mesh, top_ss);
    auto bottom_ns = smesh::create_nodeset_from_sidesets(mesh, bottom_ss);

    assert(top_ns != nullptr);
    assert(bottom_ns != nullptr);
    assert(top_ns->size() > 0);
    assert(bottom_ns->size() > 0);

    DirichletConditions::Condition xtop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 0};
    DirichletConditions::Condition ytop{.sidesets = top_ss, .nodeset = top_ns, .value = -0.2, .component = 1};
    DirichletConditions::Condition ztop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 2};

    DirichletConditions::Condition xbottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 0};
    DirichletConditions::Condition ybottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 1};
    DirichletConditions::Condition zbottom{.sidesets = bottom_ss, .nodeset = bottom_ns, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(space, {xtop, ytop, ztop, xbottom, ybottom, zbottom}, es);
    f->add_constraint(conds);

    return f;
}

struct ContactData {
    std::shared_ptr<smesh::Mesh>                     surface;
    std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> graph;
    smesh::SharedBuffer<real_t>&                     values;
    smesh::SharedBuffer<real_t>&                     mass_vector;
    smesh::SharedBuffer<real_t*>&                    normals;
    smesh::SharedBuffer<real_t>&                     distances;
    smesh::SharedBuffer<real_t>                      agumentation;
};

std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> create_contact_graph(const smesh::SharedBuffer<idx_t*>& elements,
                                                                      const smesh::SharedBuffer<idx_t>&  element_idx) {
    const ptrdiff_t npoints = element_idx->size();
    const int       nxe     = elements->extent(0);
    auto            rowptr  = sfem::create_host_buffer<count_t>(npoints + 1);
    auto            colidx  = sfem::create_host_buffer<idx_t>(npoints * nxe);

    auto elements_data    = elements->data();
    auto rowptr_data      = rowptr->data();
    auto colidx_data      = colidx->data();
    auto element_idx_data = element_idx->data();

    for (ptrdiff_t i = 1; i <= npoints; i++) {
        ptrdiff_t offset = element_idx_data[i] == -1 ? 0 : nxe;
        rowptr_data[i]   = offset + rowptr_data[i - 1];
    }

    for (ptrdiff_t i = 0; i < npoints; i++) {
        const idx_t e = element_idx_data[i];
        if (e == -1) continue;
        for (int j = 0; j < nxe; j++) {
            colidx_data[rowptr_data[i] + j] = elements_data[j][e];
        }
    }

    return std::make_shared<smesh::CRSGraph<count_t, idx_t>>(rowptr, colidx);
}

void local_coordinates(const smesh::SharedBuffer<idx_t*>&     elements,
                       const smesh::SharedBuffer<real_t*>&    points,
                       const smesh::SharedBuffer<idx_t>&      element_idx,
                       const smesh::SharedBuffer<real_t*>&    c,
                       const smesh::CRSGraph<count_t, idx_t>& graph,
                       const smesh::SharedBuffer<real_t>&     values) {
    auto p   = c->data();
    auto pts = points->data();

    const ptrdiff_t           n   = c->extent(1);
    const idx_t* const* const idx = elements->data();

    SMESH_ASSERT(n == element_idx->size());

    auto rowptr = graph.rowptr()->data();
    auto colidx = graph.colidx()->data();
    auto vals   = values->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        const ptrdiff_t e = element_idx->data()[i];
        if (e == -1) continue;

        const idx_t e0 = idx[0][e];
        const idx_t e1 = idx[1][e];
        const idx_t e2 = idx[2][e];

        const real_t x0 = pts[0][e0];
        const real_t x1 = pts[0][e1];
        const real_t x2 = pts[0][e2];
        const real_t y0 = pts[1][e0];
        const real_t y1 = pts[1][e1];
        const real_t y2 = pts[1][e2];
        const real_t z0 = pts[2][e0];
        const real_t z1 = pts[2][e1];
        const real_t z2 = pts[2][e2];

        const real_t p0 = p[0][i];
        const real_t p1 = p[1][i];
        const real_t p2 = p[2][i];

        const real_t v0x = x1 - x0;
        const real_t v0y = y1 - y0;
        const real_t v0z = z1 - z0;
        const real_t v1x = x2 - x0;
        const real_t v1y = y2 - y0;
        const real_t v1z = z2 - z0;
        const real_t v2x = p0 - x0;
        const real_t v2y = p1 - y0;
        const real_t v2z = p2 - z0;

        const real_t d00 = v0x * v0x + v0y * v0y + v0z * v0z;
        const real_t d01 = v0x * v1x + v0y * v1y + v0z * v1z;
        const real_t d11 = v1x * v1x + v1y * v1y + v1z * v1z;
        const real_t d20 = v2x * v0x + v2y * v0y + v2z * v0z;
        const real_t d21 = v2x * v1x + v2y * v1y + v2z * v1z;

        const real_t inv_det = 1 / (d00 * d11 - d01 * d01);
        const real_t w1      = (d11 * d20 - d01 * d21) * inv_det;
        const real_t w2      = (d00 * d21 - d01 * d20) * inv_det;
        const real_t w0      = 1 - w1 - w2;

        SMESH_ASSERT(std::abs(w0 + w1 + w2 - 1) < 1e-6);

        real_t ws[3]   = {w0, w1, w2};
        idx_t  keys[3] = {e0, e1, e2};
        idx_t  ks[3];

        idx_t*    row    = &colidx[rowptr[e0]];
        const int lenrow = rowptr[e0 + 1] - rowptr[e0];

        tri3_find_cols(keys, row, lenrow, ks);
        for (int j = 0; j < 3; j++) {
            vals[ks[j]] = ws[j];
        }
    }
}

void displace_points(const std::shared_ptr<smesh::Mesh>&     surface,
                     const std::shared_ptr<Buffer<real_t>>&  displacement,
                     const std::shared_ptr<Buffer<real_t*>>& inout) {
    auto p = inout->data();
    auto u = displacement->data();
    auto m = surface->node_mapping()->data();

    const ptrdiff_t n   = surface->node_mapping()->size();
    const int       dim = surface->spatial_dimension();

    for (int d = 0; d < dim; d++) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            p[d][i] += u[m[i] * dim + d];
        }
    }
}

void assemble_contact_gradient(ContactData& cm, const real_t penalty, const real_t* const disp, real_t* const grad) {
    const int dim    = cm.surface->spatial_dimension();
    auto      graph  = cm.graph;
    auto      values = cm.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    auto d       = cm.distances->data();
    auto aug     = cm.agumentation->data();
    auto normals = cm.normals->data();
    auto mass    = cm.mass_vector->data();

    auto nm = cm.surface->node_mapping()->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        auto lenrow = rowptr[i + 1] - rowptr[i];
        if (lenrow == 0) continue;

        auto row = &colidx[rowptr[i]];

        auto weights = &vals[rowptr[i]];

        real_t u1[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            u1[d] = disp[nm[i] * dim + d];
        }

        real_t u2[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            for (count_t j = 0; j < lenrow; j++) {
                u2[d] += weights[j] * disp[nm[row[j]] * dim + d];
            }
        }

        const real_t g         = d[i];
        real_t       normal[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            normal[d] = normals[d][i];
        }

        real_t diff[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            diff[d] = u1[d] - u2[d];
        }

        real_t normal_diff = 0;
        for (int d = 0; d < dim; d++) {
            normal_diff += normal[d] * diff[d];
        }

        real_t pen       = normal_diff - g;
        real_t lagr_mult = aug[i];
        real_t macaulay  = std::max(pen + lagr_mult / penalty, real_t(0));

        if (macaulay == 0) continue;

        real_t force[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            force[d] = mass[i] * penalty * macaulay * normal[d];
        }

        for (int d = 0; d < dim; d++) {
#pragma omp atomic update
            grad[nm[i] * dim + d] += force[d];
        }

        for (int d = 0; d < dim; d++) {
            for (count_t j = 0; j < lenrow; j++) {
#pragma omp atomic update
                grad[nm[row[j]] * dim + d] -= force[d] * weights[j];
            }
        }
    }
}

void apply_contact_hessian(ContactData& cm) {
    auto      graph  = cm.graph;
    auto      values = cm.values;
    auto      rowptr = graph->rowptr()->data();
    auto      colidx = graph->colidx()->data();
    auto      vals   = values->data();
    ptrdiff_t n      = graph->rowptr()->size() - 1;

    // TODO
}

int test_two_body_contact() {
    ptrdiff_t nx = 14;

    auto es   = ExecutionSpace::EXECUTION_SPACE_HOST;
    auto blas = sfem::blas<real_t>(es);

    auto      f     = create_function(nx, es);
    auto      space = f->space();
    auto      mesh  = space->mesh_ptr();
    const int dim   = mesh->spatial_dimension();

    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto solver    = sfem::create_cg<real_t>(linear_op, es);
    solver->set_op(linear_op);
    solver->set_max_it(10000);
    solver->set_rtol(1e-4);
    solver->set_verbose(false);

    auto displacement = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto rhs          = sfem::create_buffer<real_t>(space->n_dofs(), es);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), displacement->data());

    auto surface = skin(mesh);

    auto ccd = sccd::CCD<real_t>::create(surface);

    auto p0 = smesh::astype<real_t>(surface->points());
    auto p1 = smesh::astype<real_t>(surface->points());

    displace_points(surface, displacement, p1);

    real_t toi = 1;
    ccd->find_earliest_impact_time(p0, p1, toi, 69, 1e-12);

    printf("TOI: %g\n", toi);
    SFEM_TEST_APPROXEQ(toi, 0.5, 1e-2);

    // toi *= 1.1;
    blas->scal(space->n_dofs(), toi, displacement->data());

    const real_t search_radius     = 0.05;
    const real_t search_radius_sqr = search_radius * search_radius;

    p1 = smesh::astype<real_t>(surface->points());
    displace_points(surface, displacement, p1);

    auto surface_elements = surface->block(0)->elements();
    auto npoints          = surface->n_nodes();
    auto nselements       = surface->n_elements();

    auto closest_points    = sfem::create_buffer<real_t>(dim, npoints, es);
    auto distances         = sfem::create_buffer<real_t>(npoints, es);
    auto closest_triangles = sfem::create_buffer<idx_t>(npoints, es);

    ssdf::closest_within_radius_bvh(npoints,
                                    p1->data()[0],
                                    p1->data()[1],
                                    p1->data()[2],
                                    nselements,
                                    surface_elements->data()[0],
                                    surface_elements->data()[1],
                                    surface_elements->data()[2],
                                    npoints,
                                    p1->data()[0],
                                    p1->data()[1],
                                    p1->data()[2],
                                    0,
                                    &search_radius_sqr,
                                    closest_triangles->data(),
                                    distances->data(),
                                    closest_points->data()[0],
                                    closest_points->data()[1],
                                    closest_points->data()[2],
                                    true);

    auto distances_whole = sfem::create_buffer<real_t>(space->n_dofs(), es);
    {
        auto node_mapping         = surface->node_mapping()->data();
        auto distances_whole_data = distances_whole->data();
        auto distances_data       = distances->data();

        for (ptrdiff_t i = 0; i < npoints; i++) {
            distances_data[i]                           = std::sqrt(distances_data[i]);
            distances_whole_data[node_mapping[i] * dim] = distances_data[i];
        }
    }

    auto directors = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto normals   = sfem::create_buffer<real_t>(dim, npoints, es);
    {
        auto node_mapping           = surface->node_mapping()->data();
        auto directors_data         = directors->data();
        auto closest_points_data    = closest_points->data();
        auto p1_data                = p1->data();
        auto closest_triangles_data = closest_triangles->data();
        auto normals_data           = normals->data();

        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < npoints; i++) {
                if (closest_triangles_data[i] == -1) continue;
                real_t dx                                 = closest_points_data[d][i] - p1_data[d][i];
                directors_data[node_mapping[i] * dim + d] = dx;
                normals_data[d][i]                        = dx;
            }
        }

        auto distances_data = distances->data();

        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < npoints; i++) {
                if (closest_triangles_data[i] == -1) continue;
                normals_data[d][i] /= distances_data[i];
            }
        }
    }

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path("contact_output"));
    out->write("disp", displacement->data());
    out->write("distance", distances_whole->data());
    out->write("directors", directors->data());

    // TODO: Compute
    // 1. local coordinates of the closest points on the triangles and P matrix
    // 2. normalized directors
    // 3. distances (sqrt?)

    auto graph  = create_contact_graph(surface_elements, closest_triangles);
    auto values = sfem::create_buffer<real_t>(graph->nnz(), es);
    local_coordinates(surface_elements, p1, closest_triangles, closest_points, *graph, values);
    auto trace_space = std::make_shared<FunctionSpace>(surface, 1);
    auto mass_vector = create_host_buffer<real_t>(trace_space->n_dofs());

    {
        auto bop = sfem::Factory::create_op(trace_space, "Mass");
        bop->initialize();

        auto ones = create_host_buffer<real_t>(trace_space->n_dofs());
        sfem::blas<real_t>(EXECUTION_SPACE_HOST)->values(trace_space->n_dofs(), 1, ones->data());
        bop->apply(nullptr, ones->data(), mass_vector->data());
    }

    auto agumentation = sfem::create_buffer<real_t>(trace_space->n_dofs(), es);

    ContactData cm = {.surface      = surface,
                      .graph        = graph,
                      .values       = values,
                      .mass_vector  = mass_vector,
                      .normals      = normals,
                      .distances    = distances,
                      .agumentation = agumentation};

    real_t penalty = 10;
    auto   grad    = sfem::create_buffer<real_t>(space->n_dofs(), es);
    assemble_contact_gradient(cm, penalty, displacement->data(), grad->data());

    // pen->update(displacement_old->data(), displacement->data());

    // auto alpha = pen->max_step_size();
    // blas->scal(space->n_dofs(), alpha, displacement->data());

    // auto g = sfem::create_buffer<real_t>(space->n_dofs(), es);
    // pen->gradient(displacement->data(), g->data());

    // auto out = f->output();
    // out->enable_AoS_to_SoA(true);
    // out->set_output_dir(smesh::Path("contact_output"));
    // out->write("g", g->data());
    // out->write("disp", displacement->data());

    // // blas->norm(space->n_dofs(), g->data());
    // printf("Gradient norm: %g\n", blas->norm2(space->n_dofs(), g->data()));

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_two_body_contact);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
