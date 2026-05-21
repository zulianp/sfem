#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"
// #include "sfem_SelfCollisions.hpp"

#include "integrations/smesh/sccd_smesh_CCD.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"

#include "sfem_API.hpp"

#include <algorithm>
#include <utility>
#include <vector>

using namespace sfem;

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

// int test_toi() {
//     ptrdiff_t nx = 20;

//     auto f       = create_function(nx, ExecutionSpace::EXECUTION_SPACE_HOST);
//     auto space   = f->space();
//     auto es      = f->execution_space();
//     auto mesh    = space->mesh_ptr();
//     auto surface = skin(mesh);

//     const int dim = mesh->spatial_dimension();

//     surface->write(smesh::Path("contact_surface"));

//     printf("Surf: #nodes %zu #elements %zu\n", surface->n_nodes(), surface->n_elements());

//     auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
//     auto solver    = sfem::create_cg<real_t>(linear_op, es);
//     solver->set_op(linear_op);
//     solver->set_max_it(1000);
//     solver->set_rtol(1e-6);
//     solver->set_verbose(false);

//     auto previous_displacement = sfem::create_buffer<real_t>(space->n_dofs(), es);
//     auto displacement          = sfem::create_buffer<real_t>(space->n_dofs(), es);
//     auto rhs                   = sfem::create_buffer<real_t>(space->n_dofs(), es);

//     f->apply_constraints(displacement->data());
//     f->apply_constraints(rhs->data());

//     solver->apply(rhs->data(), displacement->data());

//     auto prev_disp3 = convert_host_buffer_to_fake_SoA(dim, previous_displacement);
//     auto disp3      = convert_host_buffer_to_fake_SoA(dim, displacement);

//     auto collisions = SelfCollisions::create(surface);
//     collisions->find(dim, prev_disp3->data(), disp3->data());
//     real_t toi = collisions->time_of_impact();

//     auto blas = sfem::blas<real_t>(es);
//     blas->scal(space->n_dofs(), toi, displacement->data());

//     auto d       = sfem::create_host_buffer<real_t>(surface->n_nodes());
//     auto normals = sfem::create_host_buffer<real_t>(dim, surface->n_nodes());
//     collisions->distance_and_normal(toi, d->data(), dim, normals->data());

//     {
//         Output out(FunctionSpace::create(surface, 1));
//         out.enable_AoS_to_SoA(true);
//         out.set_output_dir(smesh::Path("contact_surface_output"));
//         out.write("d", d->data());
//         out.write("nx", normals->data()[0]);
//         out.write("ny", normals->data()[1]);
//         out.write("nz", normals->data()[2]);
//     }

//     // TODO find actual collisions using scaled displacement (compute penalizer)
//     // 0) Envelope or penetration ?
//     // 1) VF distances and penalization
//     // 2) EE distances and penalization

//     auto out = f->output();
//     out->enable_AoS_to_SoA(true);
//     out->set_output_dir(smesh::Path("contact_output"));
//     out->write("disp", displacement->data());

//     printf("TOI: %g\n", toi);
//     SFEM_TEST_APPROXEQ(toi, 0.5, 1e-2);

//     return SFEM_TEST_SUCCESS;
// }

void displace_points(const std::shared_ptr<smesh::Mesh>&     surface,
                     const std::shared_ptr<Buffer<real_t>>&  displacement,
                     const std::shared_ptr<Buffer<real_t*>>& inout) {
    auto p = inout->data();
    auto u = displacement->data();
    auto m = surface->node_mapping()->data();

    const ptrdiff_t n   = surface->node_mapping()->size();
    const int       dim = surface->spatial_dimension();

    for (int d = 0; d < dim; d++) {
        for (ptrdiff_t i = 0; i < n; i++) {
            p[d][i] += u[m[i] * dim + d];
        }
    }
}

int test_two_body_contact() {
    ptrdiff_t nx = 1;

    auto es   = ExecutionSpace::EXECUTION_SPACE_HOST;
    auto blas = sfem::blas<real_t>(es);

    auto      f     = create_function(nx, es);
    auto      space = f->space();
    auto      mesh  = space->mesh_ptr();
    const int dim   = mesh->spatial_dimension();

    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto solver    = sfem::create_cg<real_t>(linear_op, es);
    solver->set_op(linear_op);
    solver->set_max_it(1000);
    solver->set_rtol(1e-6);
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

    // printf("n_points: %zu\n", surface->n_nodes());

    // surface->node_mapping()->print();
    // printf("--------\n");
    // displacement->print();
    // printf("--------\n");

    // p0->print();

    // printf("--------\n");
    // p1->print();
    // printf("--------\n");

    real_t toi = 1;
    ccd->find_earliest_impact_time(p0, p1, toi, 69, 3e-8);

    printf("TOI: %g\n", toi);

    blas->scal(space->n_dofs(), toi, displacement->data());

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path("contact_output"));
    out->write("disp", displacement->data());

    {
        smesh::SharedBuffer<smesh::idx_t> v_overlap;
        smesh::SharedBuffer<smesh::idx_t> f_overlap;
        smesh::SharedBuffer<real_t>       vf_tois;
        smesh::SharedBuffer<smesh::idx_t> e0_overlap;
        smesh::SharedBuffer<smesh::idx_t> e1_overlap;
        smesh::SharedBuffer<real_t>       ee_tois;
        ccd->find_impact_times(p0, p1, v_overlap, f_overlap, vf_tois, e0_overlap, e1_overlap, ee_tois, 69, 3e-8);

        // auto print_pairs = [surface](const idx_t v, const idx_t f, const SharedBuffer<real_t*>& points) {
        //     auto p = points->data();

        //     auto x = p[0][v];
        //     auto y = p[1][v];
        //     auto z = p[2][v];

        //     auto i0 = surface->elements(0)->data()[0][f];
        //     auto i1 = surface->elements(0)->data()[1][f];
        //     auto i2 = surface->elements(0)->data()[2][f];

        //     auto x0 = p[0][i0];
        //     auto y0 = p[1][i0];
        //     auto z0 = p[2][i0];

        //     auto x1 = p[0][i1];
        //     auto y1 = p[1][i1];
        //     auto z1 = p[2][i1];

        //     auto x2 = p[0][i2];
        //     auto y2 = p[1][i2];
        //     auto z2 = p[2][i2];

        //     printf("V(%d): %g %g %g\n", v, x, y, z);
        //     printf("V(%d): %g %g %g\n", i0, x0, y0, z0);
        //     printf("V(%d): %g %g %g\n", i1, x1, y1, z1);
        //     printf("V(%d): %g %g %g\n", i2, x2, y2, z2);
        // };

        auto print_pairs = [surface, edges = ccd->edges()](const idx_t e0, const idx_t e1, const SharedBuffer<real_t*>& points) {
            auto p = points->data();
            auto e = edges->data();

            auto a0 = e[0][e0];
            auto a1 = e[1][e0];
            auto b0 = e[0][e1];
            auto b1 = e[1][e1];

            auto x0 = p[0][a0];
            auto y0 = p[1][a0];
            auto z0 = p[2][a0];

            auto x1 = p[0][a1];
            auto y1 = p[1][a1];
            auto z1 = p[2][a1];

            //

            auto x2 = p[0][b0];
            auto y2 = p[1][b0];
            auto z2 = p[2][b0];

            auto x3 = p[0][b1];
            auto y3 = p[1][b1];
            auto z3 = p[2][b1];

            printf("A0(%d): %g %g %g\n", a0, x0, y0, z0);
            printf("A1(%d): %g %g %g\n", a1, x1, y1, z1);
            printf("B0(%d): %g %g %g\n", b0, x2, y2, z2);
            printf("B1(%d): %g %g %g\n", b1, x3, y3, z3);
        };

        int weird = 36;

        printf("--------\n");
        print_pairs(e0_overlap->data()[weird], e1_overlap->data()[weird], p0);
        printf("--------\n");
        print_pairs(e0_overlap->data()[weird], e1_overlap->data()[weird], p1);
        printf("--------\n");

        printf("EE TOI: %g\n", ee_tois->data()[weird]);
        printf("--------\n");

        real_t tol = smesh::Env::read("SCCD_TOL", 1e-6);

        auto edges  = ccd->edges()->data();
        auto points = p0->data();
        auto moved  = p1->data();

        const auto edge0 = e0_overlap->data()[weird];
        const auto edge1 = e1_overlap->data()[weird];

        const auto v0 = edges[0][edge0];
        const auto v1 = edges[1][edge0];
        const auto v2 = edges[0][edge1];
        const auto v3 = edges[1][edge1];

        const real_t s1[3] = {points[0][v0], points[1][v0], points[2][v0]};
        const real_t s2[3] = {points[0][v1], points[1][v1], points[2][v1]};
        const real_t s3[3] = {points[0][v2], points[1][v2], points[2][v2]};
        const real_t s4[3] = {points[0][v3], points[1][v3], points[2][v3]};

        const real_t e1[3] = {moved[0][v0], moved[1][v0], moved[2][v0]};
        const real_t e2[3] = {moved[0][v1], moved[1][v1], moved[2][v1]};
        const real_t e3[3] = {moved[0][v2], moved[1][v2], moved[2][v2]};
        const real_t e4[3] = {moved[0][v3], moved[1][v3], moved[2][v3]};

        real_t tols[3];
        compute_edge_edge_tolerance<real_t>(tol, s1, s2, s3, s4, e1, e2, e3, e4, tols);

        printf("EE tolerance: %g %g %g\n", tols[0], tols[1], tols[2]);
        printf("--------\n");

        const real_t* const bb_points[8] = {s1, s2, s3, s4, e1, e2, e3, e4};
        real_t              bb_min[3]    = {s1[0], s1[1], s1[2]};
        real_t              bb_max[3]    = {s1[0], s1[1], s1[2]};
        for (int i = 1; i < 8; i++) {
            for (int d = 0; d < 3; d++) {
                bb_min[d] = std::min(bb_min[d], bb_points[i][d]);
                bb_max[d] = std::max(bb_max[d], bb_points[i][d]);
            }
        }

        printf("EE bounding box: %g %g %g - %g %g %g\n", bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2]);

        real_t diff[3];
        sccd::diff_ee<real_t>(s1, s2, s3, s4, e1, e2, e3, e4, 0, 0, 0, diff);

        real_t diff_bb_min[3] = {diff[0], diff[1], diff[2]};
        real_t diff_bb_max[3] = {diff[0], diff[1], diff[2]};

        for (int corner = 1; corner < 8; corner++) {
            const real_t t = corner & 1;
            const real_t u = (corner >> 1) & 1;
            const real_t v = (corner >> 2) & 1;

            sccd::diff_ee<real_t>(s1, s2, s3, s4, e1, e2, e3, e4, t, u, v, diff);
            for (int d = 0; d < 3; d++) {
                diff_bb_min[d] = std::min(diff_bb_min[d], diff[d]);
                diff_bb_max[d] = std::max(diff_bb_max[d], diff[d]);
            }
        }

        printf("EE diff bounding box: %g %g %g - %g %g %g\n",
               diff_bb_min[0],
               diff_bb_min[1],
               diff_bb_min[2],
               diff_bb_max[0],
               diff_bb_max[1],
               diff_bb_max[2]);

        // printf("V overlap: %zu\n", v_overlap->size());
        // v_overlap->print();

        // printf("F overlap: %zu\n", f_overlap->size());
        // f_overlap->print();

        // printf("VF tois: %zu\n", vf_tois->size());
        // vf_tois->print();

        // printf("E0 overlap: %zu\n", e0_overlap->size());
        // e0_overlap->print();
        // printf("E1 overlap: %zu\n", e1_overlap->size());
        // e1_overlap->print();

        // printf("EE tois: %zu\n", ee_tois->size());
        // ee_tois->print();
    }

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
    // SFEM_RUN_TEST(test_toi);
    SFEM_RUN_TEST(test_two_body_contact);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
