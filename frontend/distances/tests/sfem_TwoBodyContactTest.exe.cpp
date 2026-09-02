#include "sfem_test.hpp"

#include "sfem_ContactSkin.hpp"
#include "sfem_FunctionSpace.hpp"

#include "integrations/smesh/sccd_smesh_CCD.hpp"

#include "sfem_aliases.hpp"
#include "sfem_context.hpp"
#include "smesh_adjacency.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sort.hpp"
#include "smesh_sshex8_graph.hpp"
#include "smesh_ssquad4_mesh.hpp"

#include "sfem_API.hpp"
#include "sfem_SelfContact.hpp"
#include "sfem_mask.hpp"
#include "sfem_ssgmg.hpp"

#include "bvh/bvh.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "sfem_ContactSolveKernels.hpp"

using namespace sfem;

struct EnvOptions {
    int             demo;
    real_t          margin;
    int             outer_loops;
    int             inner_loops;
    int             nx;
    real_t          ytop;
    real_t          penalty;
    real_t          solver_tol;
    bool            enable_ccd;
    smesh::ElemType element_type;
    bool            enable_augmentation;
    real_t          toi_scale;
    real_t          search_radius;
    real_t          damping;
    int             output_frequency;
    int             linear_max_it;
    real_t          linear_rtol;
    bool            linear_verbose;

    static EnvOptions read() {
        return {
                smesh::Env::read("SFEM_DEMO", int(1)),
                smesh::Env::read("SFEM_MARGIN", real_t(0)),
                smesh::Env::read("SFEM_OUTER_LOOPS", int(1)),
                smesh::Env::read("SFEM_INNER_LOOPS", int(1000)),
                smesh::Env::read("SFEM_NX", int(10)),
                smesh::Env::read("SFEM_YTOP", real_t(-0.4)),
                smesh::Env::read("SFEM_PENALTY", real_t(10)),
                smesh::Env::read("SFEM_SOLVER_TOL", real_t(1e-6)),
                smesh::Env::read("SFEM_ENABLE_CCD", false),
                smesh::Env::read("SFEM_ELEM_TYPE", smesh::HEX8),
                smesh::Env::read("SFEM_ENABLE_AUGMENTATION", false),
                smesh::Env::read("SFEM_TOI_SCALE", real_t(1)),
                smesh::Env::read("SFEM_SEARCH_RADIUS", real_t(0.1)),
                smesh::Env::read("SFEM_DAMPING", real_t(1)),
                smesh::Env::read("SFEM_OUTPUT_FREQUENCY", int(10)),
                smesh::Env::read("SFEM_LINEAR_MAX_IT", int(10000)),
                smesh::Env::read("SFEM_LINEAR_RTOL", real_t(1e-4)),
                smesh::Env::read("SFEM_LINEAR_VERBOSE", false),
        };
    }

    void print(std::ostream& os) const {
        os << "EnvOptions:" << std::endl;
        os << "  demo: " << demo << std::endl;
        os << "  margin: " << margin << std::endl;
        os << "  outer_loops: " << outer_loops << std::endl;
        os << "  inner_loops: " << inner_loops << std::endl;
        os << "  nx: " << nx << std::endl;
        os << "  ytop: " << ytop << std::endl;
        os << "  penalty: " << penalty << std::endl;
        os << "  solver_tol: " << solver_tol << std::endl;
        os << "  enable_ccd: " << enable_ccd << std::endl;
        os << "  element_type: " << element_type << std::endl;
        os << "  enable_augmentation: " << enable_augmentation << std::endl;
        os << "  toi_scale: " << toi_scale << std::endl;
        os << "  search_radius: " << search_radius << std::endl;
    }
};

std::shared_ptr<Function> create_function(const EnvOptions& opts, const ExecutionSpace es, const EnvOptions& env) {
    if (env.demo) {
        const ptrdiff_t nx = opts.nx;

        geom_t y_bottom = 0.9;
        auto   mesh1    = smesh::Mesh::create_cube(
                Communicator::self(), env.element_type, nx, std::max<ptrdiff_t>(1, nx / 5), nx, 0, y_bottom, 0, 1, 1, 1);
        auto mesh2 = smesh::Mesh::create_cube(Communicator::self(),
                                              env.element_type,
                                              std::max<ptrdiff_t>(1, nx / 2),
                                              nx,
                                              std::max<ptrdiff_t>(1, nx / 2),
                                              0.25,
                                              1.1,
                                              0.25,
                                              0.75,
                                              1.9,
                                              0.75);

        auto mesh = smesh::concatenate(mesh1, mesh2);

        printf("Bulk: #nodes %zu #elements %zu\n", mesh->n_nodes(), mesh->n_elements());

        if (smesh::is_semistructured_type(mesh->element_type(0))) {
            smesh::semistructured_export_as_standard(mesh, smesh::Path("contact_mesh"));
        } else {
            mesh->write(smesh::Path("contact_mesh"));
        }

        auto top_ss =
                sfem::Sideset::create_from_selector(mesh, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool {
                    return y > (1.9 - 1e-4) && y < (1.9 + 1e-4);
                });

        auto left_ss = sfem::Sideset::create_from_selector(mesh, [=](const geom_t x, const geom_t /*y*/, const geom_t z) -> bool {
            return x > (-1e-4) && x < (1e-4) && z > 0.45 && z < 0.55;
        });

        auto right_ss =
                sfem::Sideset::create_from_selector(mesh, [=](const geom_t x, const geom_t /*y*/, const geom_t z) -> bool {
                    return x > (1 - 1e-4) && x < (1 + 1e-4) && z > 0.45 && z < 0.55;
                });

        const int dim   = mesh->spatial_dimension();
        auto      space = FunctionSpace::create(mesh, dim);

        auto op = create_op(space, "LinearElasticity", es);
        op->initialize();

        auto f = Function::create(space);
        f->add_operator(op);

        auto top_ns   = smesh::create_nodeset_from_sidesets(mesh, top_ss);
        auto left_ns  = smesh::create_nodeset_from_sidesets(mesh, left_ss);
        auto right_ns = smesh::create_nodeset_from_sidesets(mesh, right_ss);

        assert(top_ns != nullptr);
        assert(top_ns->size() > 0);

        assert(left_ns != nullptr);
        assert(left_ns->size() >= 0);

        assert(right_ns != nullptr);
        assert(right_ns->size() >= 0);

        DirichletConditions::Condition xtop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 0};
        DirichletConditions::Condition ytop{.sidesets = top_ss, .nodeset = top_ns, .value = opts.ytop, .component = 1};
        DirichletConditions::Condition ztop{.sidesets = top_ss, .nodeset = top_ns, .value = 0, .component = 2};

        DirichletConditions::Condition xleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 0};
        DirichletConditions::Condition yleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 1};
        DirichletConditions::Condition zleft{.sidesets = left_ss, .nodeset = left_ns, .value = 0, .component = 2};

        DirichletConditions::Condition xright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 0};
        DirichletConditions::Condition yright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 1};
        DirichletConditions::Condition zright{.sidesets = right_ss, .nodeset = right_ns, .value = 0, .component = 2};

        auto conds =
                sfem::create_dirichlet_conditions(space, {xtop, ytop, ztop, xleft, yleft, zleft, xright, yright, zright}, es);
        f->add_constraint(conds);

        return f;
    } else {
        smesh::Path mesh_path{"./mesh"};
        smesh::Path dirichlet_path{"./case.yaml"};
        auto        mesh = smesh::Mesh::create_from_file(Communicator::self(), mesh_path);

        auto space = sfem::FunctionSpace::create(mesh, mesh->spatial_dimension());
        auto f     = sfem::Function::create(space);

        auto op = create_op(space, "LinearElasticity", es);
        op->initialize();
        f->add_operator(op);

        auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(space, dirichlet_path);
        f->add_constraint(dirichlet_conditions);

        return f;
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

int test_two_body_contact() {
    const EnvOptions env = EnvOptions::read();
    env.print(std::cout);
    ptrdiff_t nx = env.nx;

    auto es   = ExecutionSpace::EXECUTION_SPACE_HOST;
    auto blas = sfem::blas<real_t>(es);

    auto      f     = create_function(env, es, env);
    auto      space = f->space();
    auto      mesh  = space->mesh_ptr();
    const int dim   = mesh->spatial_dimension();

    auto displacement     = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto rhs              = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto constraints_mask = sfem::create_buffer<mask_t>(mask_count(space->n_dofs()), es);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < constraints_mask->size(); ++i) {
        constraints_mask->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(f->constraints_mask(constraints_mask->data()) == SFEM_SUCCESS);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    if (space->has_semi_structured_mesh()) {
        auto solver = create_ssgmg(f, f->execution_space());
        solver->set_max_it(1);
        solver->apply(rhs->data(), displacement->data());
    } else {
        auto linear_op = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, nullptr, es);
        auto solver    = sfem::create_cg<real_t>(linear_op, es);
        solver->set_op(linear_op);
        solver->set_max_it(env.linear_max_it);
        solver->set_rtol(env.linear_rtol);
        solver->set_verbose(env.linear_verbose);
        solver->apply(rhs->data(), displacement->data());
    }

    auto surface = create_contact_skin(mesh, constraints_mask);
    surface->write(smesh::Path("contact_surface"));

    std::shared_ptr<sccd::CCD<real_t>> ccd;
    ccd = sccd::CCD<real_t>::create(surface);

    auto p0 = smesh::astype<real_t>(surface->points(), /*duplicate=*/true);
    auto p1 = smesh::astype<real_t>(surface->points(), /*duplicate=*/true);

    displace_points(surface, displacement, p1);

    real_t toi = 1;
    if (ccd) {
        ccd->find_earliest_impact_time(p0, p1, toi, 69, 1e-12);
    }

    printf("TOI: %g\n", toi);
    // SFEM_TEST_APPROXEQ(toi, 0.25, 1e-2);

    // toi *= 1.1;
    blas->scal(space->n_dofs(), env.toi_scale * toi, displacement->data());

    const real_t search_radius     = env.search_radius;
    const real_t search_radius_sqr = search_radius * search_radius;
    const real_t margin            = env.margin;

    auto contact_conditions = create_contact(space, surface, margin, search_radius_sqr, es);

    auto agumentation = sfem::create_buffer<real_t>(contact_conditions->mass_vector()->size(), es);

    real_t penalty          = env.penalty;
    auto   lagr_mult_normal = sfem::create_buffer<real_t>(space->n_dofs(), es);
    auto   previous_displacement = sfem::create_buffer<real_t>(space->n_dofs(), es);

    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path("contact_output"));

    const int outer_loops = env.outer_loops;
    const int inner_loops = env.inner_loops;

    contact_conditions->recompute(displacement);
    blas->copy(space->n_dofs(), displacement->data(), previous_displacement->data());

    out->write_time_step("disp", 0, displacement->data());
    out->write_time_step("distance", 0, contact_conditions->distances_whole()->data());
    out->write_time_step("directors", 0, contact_conditions->directors()->data());
    out->write_time_step("lagr_mult_normal", 0, lagr_mult_normal->data());
    out->log_time(0);

    f->apply_constraints(displacement->data());
    f->apply_constraints(rhs->data());

    for (int outer = 0; outer < outer_loops; ++outer) {
        auto coupling_matrix = sfem::h_crs_spmv(contact_conditions->graph()->n_nodes(),
                                                contact_conditions->graph()->n_nodes(),
                                                contact_conditions->graph()->rowptr(),
                                                contact_conditions->graph()->colidx(),
                                                contact_conditions->values(),
                                                real_t(0));

        auto cd = std::make_shared<sfem::ContactData>(sfem::ContactData{.f                = f,
                                                                         .surface          = surface,
                                                                         .coupling_matrix  = coupling_matrix,
                                                                         .values           = contact_conditions->values(),
                                                                         .mass_vector      = contact_conditions->mass_vector(),
                                                                         .normals          = contact_conditions->normals(),
                                                                         .distances        = contact_conditions->distances(),
                                                                         .constraints_mask = constraints_mask,
                                                                         .agumentation     = agumentation});

        sfem::ContactJacobi jacobi(cd);
        jacobi.set_penalty(penalty);
        jacobi.set_n_loops(inner_loops);
        jacobi.set_enable_augmentation(env.enable_augmentation);
        jacobi.smooth(displacement);

        {
            const real_t* const u0 = previous_displacement->data();
            real_t* const       u1 = displacement->data();
            const ptrdiff_t     n  = space->n_dofs();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                u1[i] = u0[i] + env.damping * (u1[i] - u0[i]);
            }
        }

        if (env.enable_ccd && ccd) {
            p0 = smesh::astype<real_t>(surface->points(), /*duplicate=*/true);
            displace_points(surface, previous_displacement, p0);

            p1 = smesh::astype<real_t>(surface->points(), /*duplicate=*/true);
            displace_points(surface, displacement, p1);

            real_t ccd_toi = 1;
            ccd->find_earliest_impact_time(p0, p1, ccd_toi, 69, 1e-12);
            printf("CCD TOI: %g\n", ccd_toi);

            if (ccd_toi < 1) {
                const real_t* const u0 = previous_displacement->data();
                real_t* const       u1 = displacement->data();
                const ptrdiff_t     n  = space->n_dofs();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    u1[i] = u0[i] + ccd_toi * (u1[i] - u0[i]);
                }
            }
        }

        contact_conditions->recompute(displacement);
        blas->copy(space->n_dofs(), displacement->data(), previous_displacement->data());

        blas->values(space->n_dofs(), 0, lagr_mult_normal->data());

        {
            const idx_t* const  node_mapping          = surface->node_mapping()->data();
            const real_t* const lagr_mult             = agumentation->data();
            const real_t* const normal_x              = contact_conditions->normals()->data()[0];
            const real_t* const normal_y              = contact_conditions->normals()->data()[1];
            const real_t* const normal_z              = contact_conditions->normals()->data()[2];
            real_t* const       lagr_mult_normal_data = lagr_mult_normal->data();
            const ptrdiff_t     n                     = surface->node_mapping()->size();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                const ptrdiff_t dof            = node_mapping[i] * 3;
                const real_t    lm             = lagr_mult[i];
                lagr_mult_normal_data[dof + 0] = lm * normal_x[i];
                lagr_mult_normal_data[dof + 1] = lm * normal_y[i];
                lagr_mult_normal_data[dof + 2] = lm * normal_z[i];
            }
        }

        if ((outer + 1) % env.output_frequency == 0) {
            out->write_time_step("disp", outer + 1, displacement->data());
            out->write_time_step("distance", outer + 1, contact_conditions->distances_whole()->data());
            out->write_time_step("directors", outer + 1, contact_conditions->directors()->data());
            out->write_time_step("lagr_mult_normal", outer + 1, lagr_mult_normal->data());
            out->log_time(outer + 1);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_two_body_contact);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
