#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "sfem_test.hpp"

#include "sfem_Function.hpp"

#include "sfem_CRS.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

#include "matrixio_array.h"

#include "sfem_API.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

struct EnvOptions {
    sfem::ExecutionSpace execution_space;
    int                  base_resolution;
    real_t               disp_y;
    int                  enable_output;
    int                  element_refine_level;
    geom_t               y_top;
    std::string          operator_name;
    std::string          contact_case;
    int                  use_spmg;
    int                  enable_top_bc;
    std::string          ssmgc_yaml;
    int                  n_spheres;
    int                  resolution_ratio;

    static EnvOptions read() {
        EnvOptions ret{.execution_space      = sfem::EXECUTION_SPACE_HOST,
                       .base_resolution      = smesh::Env::read("SFEM_BASE_RESOLUTION", int(1)),
                       .disp_y               = smesh::Env::read("SFEM_DISP_Y", real_t(-0.05)),
                       .enable_output        = smesh::Env::read("SFEM_ENABLE_OUTPUT", int(1)),
                       .element_refine_level = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", int(2)),
                       .y_top                = smesh::Env::read("SFEM_Y_TOP", geom_t(0.05)),
                       .operator_name        = smesh::Env::read_string("SFEM_OPERATOR", "LinearElasticity"),
                       .contact_case         = smesh::Env::read_string("SFEM_CONTACT_CASE", "sphere"),
                       .use_spmg             = smesh::Env::read("SFEM_USE_SPMG", int(1)),
                       .enable_top_bc        = smesh::Env::read("SFEM_ENABLE_TOP_BC", int(0)),
                       .ssmgc_yaml           = smesh::Env::read_string("SFEM_SSMGC_YAML", ""),
                       .n_spheres            = smesh::Env::read("SFEM_N_SPHERES", int(2)),
                       .resolution_ratio     = smesh::Env::read("SFEM_RESOLUTION_RATIO", int(20))};

        const std::string execution_space = smesh::Env::read_string("SFEM_EXECUTION_SPACE", "");
        if (!execution_space.empty()) {
            ret.execution_space = sfem::execution_space_from_string(execution_space.c_str());
        }

        return ret;
    }
};

std::shared_ptr<sfem::ContactConditions> build_cuboid_sphere_contact(const std::shared_ptr<sfem::Function> &f,
                                                                     const EnvOptions                      &opts) {
    auto fs   = f->space();
    auto m    = fs->mesh_ptr();
    auto comm = m->comm();
    auto es   = f->execution_space();
    const real_t disp_y           = opts.disp_y;
    const geom_t y_top            = opts.y_top;
    const int    resolution_ratio = opts.resolution_ratio;

    if (opts.enable_top_bc) {
        auto top_ss = sfem::Sideset::create_from_selector(m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool {
            return y > (y_top - 1e-5) && y < (y_top + 1e-5);
        });

        sfem::DirichletConditions::Condition xtop{.sidesets = top_ss, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition ytop{.sidesets = top_ss, .value = disp_y, .component = 1};
        sfem::DirichletConditions::Condition ztop{.sidesets = top_ss, .value = 0, .component = 2};

        auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);
        f->add_constraint(conds);

    } else {
        auto left_right = sfem::Sideset::create_from_selector(
                m, [=](const geom_t x, const geom_t y, const geom_t z) -> bool { return fabs(x) < 1e-8 || fabs(x - 1) < 1e-8; });

        sfem::DirichletConditions::Condition x_bc{.sidesets = left_right, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition y_bc{.sidesets = left_right, .value = disp_y, .component = 1};
        sfem::DirichletConditions::Condition z_bc{.sidesets = left_right, .value = 0, .component = 2};

        auto conds = sfem::create_dirichlet_conditions(fs, {x_bc, y_bc, z_bc}, es);
        f->add_constraint(conds);
    }

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    assert(bottom_ss[0]->size() > 0);

    const int n   = opts.base_resolution * smesh::semistructured_level(fs->mesh());
    auto      sdf = smesh::create_sdf(comm,
                                 n * resolution_ratio * 2,
                                 n * 1 * 2,
                                 n * resolution_ratio * 2,
                                 -0.1,
                                 -0.2,
                                 -0.1,
                                 1.1,
                                 y_top * 0.5,
                                 1.1,
                                 [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                     // Half-sphere
                                     geom_t cx = 0.5, cy = -0.5, cz = 0.5;
                                     geom_t radius = 0.5;

                                     geom_t dx = cx - x;
                                     geom_t dy = cy - y;
                                     geom_t dz = cz - z;

                                     geom_t dd = radius - sqrt(dx * dx + dy * dy + dz * dz);
                                     return dd;
                                 });

    if (opts.enable_output) sdf->to_file(smesh::Path("test_contact/sdf"));

    auto contact_conds = sfem::ContactConditions::create(fs, sdf, bottom_ss, es);
    return contact_conds;
}

std::shared_ptr<sfem::ContactConditions> build_cuboid_highfreq_contact(const std::shared_ptr<sfem::Function> &f,
                                                                       const EnvOptions                      &opts) {
    auto fs   = f->space();
    auto m    = fs->mesh_ptr();
    auto comm = m->comm();
    auto es   = f->execution_space();
    const real_t disp_y           = opts.disp_y;
    const geom_t y_top            = opts.y_top;
    const int    resolution_ratio = opts.resolution_ratio;

    if (opts.enable_top_bc) {
        auto top_ss = sfem::Sideset::create_from_selector(m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool {
            return y > (y_top - 1e-5) && y < (y_top + 1e-5);
        });

        sfem::DirichletConditions::Condition xtop{.sidesets = top_ss, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition ytop{.sidesets = top_ss, .value = disp_y, .component = 1};
        sfem::DirichletConditions::Condition ztop{.sidesets = top_ss, .value = 0, .component = 2};

        auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);
        f->add_constraint(conds);

    } else {
        auto ss = sfem::Sideset::create_from_selector(
                m, [=](const geom_t x, const geom_t y, const geom_t z) -> bool { return fabs(x) < 1e-8 || fabs(x - 1) < 1e-8; });

        sfem::DirichletConditions::Condition x_bc{.sidesets = ss, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition y_bc{.sidesets = ss, .value = disp_y, .component = 1};
        sfem::DirichletConditions::Condition z_bc{.sidesets = ss, .value = 0, .component = 2};

        auto conds = sfem::create_dirichlet_conditions(fs, {x_bc, y_bc, z_bc}, es);
        f->add_constraint(conds);
    }

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    const int n   = opts.base_resolution * smesh::semistructured_level(fs->mesh());
    auto      sdf = smesh::create_sdf(comm,
                                 n * resolution_ratio * 2,
                                 n * 1 * 2,
                                 n * resolution_ratio * 2,
                                 0.1,
                                 -0.2,
                                 0.1,
                                 0.9,
                                 y_top * 0.5,
                                 0.9,
                                 [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                     // High-freq surface
                                     const geom_t cx = 0.6 * (1 - (x - .5) * (x - .5));
                                     const geom_t cz = 0.6 * (1 - (z - .5) * (z - .5));

                                     geom_t fx = 0.1 * cos(cx * 3.14 * 8) * cx * cx + 0.02 * cos(cx * 3.14 * 16);
                                     geom_t fz = 0.1 * cos(cz * 3.14 * 8) * cz * cz + 0.02 * cos(cx * 3.14 * 16);
                                     fx += 0.005 * cos(cx * 3.14 * 32);
                                     fz += 0.005 * cos(cz * 3.14 * 32);
                                     fx += 0.0025 * cos(cx * 3.14 * 64);
                                     fz += 0.0025 * cos(cz * 3.14 * 64);

                                     fx += 0.001 * cos(3.14 + cx * 3.14 * 128);
                                     fz += 0.001 * cos(3.14 + cz * 3.14 * 128);
                                     fx += 0.001 * cos(cx * 3.14 * 256);
                                     fz += 0.001 * cos(cz * 3.14 * 256);

                                     fx += 0.001 * cos(cx * 3.14 * 512);
                                     fz += 0.001 * cos(cz * 3.14 * 512);

                                     const geom_t obstacle = -0.1 - fx - fz;
                                     return obstacle - y;
                                 });

    if (opts.enable_output) sdf->to_file(smesh::Path("test_contact/sdf"));

    auto contact_conds = sfem::ContactConditions::create(fs, sdf, {bottom_ss}, es);
    return contact_conds;
}

std::shared_ptr<sfem::ContactConditions> build_cuboid_multisphere_contact(const std::shared_ptr<sfem::Function> &f,
                                                                          const EnvOptions                      &opts) {
    auto fs   = f->space();
    auto m    = fs->mesh_ptr();
    auto comm = m->comm();
    auto es   = f->execution_space();
    const real_t disp_y = opts.disp_y;
    const geom_t y_top  = opts.y_top;

    if (opts.enable_top_bc) {
        auto top_ss = sfem::Sideset::create_from_selector(m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool {
            return y > (y_top - 1e-5) && y < (y_top + 1e-5);
        });

        sfem::DirichletConditions::Condition xtop{.sidesets = top_ss, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition ytop{.sidesets = top_ss, .value = disp_y, .component = 1};
        sfem::DirichletConditions::Condition ztop{.sidesets = top_ss, .value = 0, .component = 2};

        auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);
        f->add_constraint(conds);

    } else {
        auto ss = sfem::Sideset::create_from_selector(
                m, [=](const geom_t x, const geom_t y, const geom_t z) -> bool { return fabs(x) < 1e-8 || fabs(x - 1) < 1e-8; });

        sfem::DirichletConditions::Condition x_bc{.sidesets = ss, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition y_bc{.sidesets = ss, .value = disp_y, .component = 1};
        sfem::DirichletConditions::Condition z_bc{.sidesets = ss, .value = 0, .component = 2};

        auto conds = sfem::create_dirichlet_conditions(fs, {x_bc, y_bc, z_bc}, es);
        f->add_constraint(conds);
    }

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    const int n         = opts.base_resolution * smesh::semistructured_level(fs->mesh());
    const int n_spheres = opts.n_spheres;
    auto      sdf       = smesh::create_sdf(comm,
                                 n * 5 * 2,
                                 n * 1 * 2,
                                 n * 5 * 2,
                                 -0.1,
                                 -0.2,
                                 -0.1,
                                 1.1,
                                 y_top * 0.5,
                                 1.1,
                                 [n_spheres](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                     geom_t       dd = 1000000;
                                     const geom_t hx = 1. / (n_spheres + 1);
                                     const geom_t hz = 1. / (n_spheres + 1);
                                     const geom_t hy = 1. / (n_spheres + 1);

                                     for (int i = 0; i < n_spheres; i++) {
                                         for (int j = 0; j < n_spheres; j++) {
                                             geom_t cx = hx + i * hx, cy = -0.1, cz = hz + j * hz;
                                             geom_t radius = 1. / (8 + n_spheres);

                                             const geom_t dx = cx - x;
                                             const geom_t dy = cy - y;
                                             const geom_t dz = cz - z;

                                             const geom_t ddij = radius - sqrt(dx * dx + dy * dy + dz * dz);
                                             dd                = fabs(ddij) < fabs(dd) ? ddij : dd;
                                         }
                                     }

                                     return dd;
                                 });

    if (opts.enable_output) sdf->to_file(smesh::Path("test_contact/sdf"));

    auto contact_conds = sfem::ContactConditions::create(fs, sdf, {bottom_ss}, es);
    return contact_conds;
}

int test_contact() {
    auto             comm = sfem::Communicator::world();
    const EnvOptions opts = EnvOptions::read();

    if (comm->size() > 1) {
        SFEM_ERROR("test_contact() can only be run in serial!\n");
    }

    const sfem::ExecutionSpace es = opts.execution_space;
    const geom_t               y_top            = opts.y_top;
    const int                  resolution_ratio = opts.resolution_ratio;

    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::world(),
                                             opts.base_resolution * resolution_ratio,
                                             opts.base_resolution * 1,
                                             opts.base_resolution * resolution_ratio,
                                             0,
                                             0,
                                             0,
                                             1,
                                             y_top,
                                             1);

    SFEM_TEST_ASSERT(opts.element_refine_level > 1);

    mesh                 = smesh::to_semistructured(opts.element_refine_level, mesh, true, false);
    const int block_size = mesh->spatial_dimension();
    auto      fs         = sfem::FunctionSpace::create(mesh, block_size);

#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, opts.operator_name, es);
    op->initialize();

    f->add_operator(op);

    if (opts.enable_output) smesh::create_directory("test_contact");

    std::shared_ptr<sfem::ContactConditions> contact_conds;

    if (opts.contact_case == "hifreq") {
        contact_conds = build_cuboid_highfreq_contact(f, opts);
    } else if (opts.contact_case == "sphere") {
        contact_conds = build_cuboid_sphere_contact(f, opts);
    } else if (opts.contact_case == "multisphere") {
        contact_conds = build_cuboid_multisphere_contact(f, opts);
    } else {
        SFEM_ERROR("SFEM_CONTACT_CASE=%s not valid!\n", opts.contact_case.c_str());
    }

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs   = sfem::create_buffer<real_t>(ndofs, es);
    auto            gap   = sfem::create_buffer<real_t>(ndofs, es);

    f->apply_constraints(rhs->data());
    // contact_conds->update(x->data()); // FIXME
    contact_conds->init();

    f->apply_constraints(x->data());

    if (opts.use_spmg) {
        std::shared_ptr<sfem::Input> in;

        if (!opts.ssmgc_yaml.empty()) {
            in = sfem::YAMLNoIndent::create_from_file(opts.ssmgc_yaml.c_str());
        }

        auto solver = sfem::create_ssmgc(f, contact_conds, in);
        solver->apply(rhs->data(), x->data());
    } else {
        auto solver = sfem::create_shifted_penalty(f, contact_conds, nullptr);
        solver->apply(rhs->data(), x->data());
    }

    if (opts.enable_output) {
        smesh::semistructured_export_as_standard(fs->mesh_ptr(), smesh::Path("test_contact/mesh"));

        auto out = f->output();
        out->set_output_dir(smesh::Path("test_contact/out"));
        out->enable_AoS_to_SoA(true);

        if (es != sfem::EXECUTION_SPACE_DEVICE) {
            contact_conds->signed_distance_for_mesh_viz(x->data(), gap->data());
            out->write("gap", gap->data());
        }

        out->write("rhs", smesh::to_host(rhs)->data());
        x = smesh::to_host(x);

        out->write("disp", x->data());

        // FIXME
        if (es != sfem::EXECUTION_SPACE_DEVICE) {
            auto blas = sfem::blas<real_t>(es);
            blas->zeros(rhs->size(), rhs->data());
            f->gradient(x->data(), rhs->data());

            blas->zeros(x->size(), x->data());
            contact_conds->full_apply_boundary_mass_inverse(rhs->data(), x->data());
            out->write("contact_stress", x->data());
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int run_ssmgc_on_ss_mesh(const std::shared_ptr<sfem::Mesh> &ss,
                                const int                          block_size,
                                const std::vector<std::shared_ptr<sfem::Sideset>> &contact_sides,
                                const std::shared_ptr<smesh::Grid<geom_t>>        &sdf,
                                const geom_t                                      disp_y) {
    SFEM_TEST_ASSERT(ss != nullptr);
    SFEM_TEST_ASSERT(!contact_sides.empty());
    ptrdiff_t n_faces = 0;
    for (const auto &s : contact_sides) {
        SFEM_TEST_ASSERT(s != nullptr);
        n_faces += s->size();
    }
    SFEM_TEST_ASSERT(n_faces > 0);

    if (!smesh::is_hex_ss_family(ss->element_type(0))) {
        setenv("SFEM_COARSE_OP_TYPE", sfem::op_type::MATRIX_FREE, 1);
    }

    auto fs = sfem::FunctionSpace::create(ss, block_size);
    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "LinearElasticity", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
    f->add_operator(op);

    const int dim = ss->spatial_dimension();
    auto      wall = sfem::Sideset::create_from_selector(ss, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool {
        return fabs(x) < 1e-8 || fabs(x - 1) < 1e-8;
    });
    SFEM_TEST_ASSERT(!wall.empty());

    std::vector<sfem::DirichletConditions::Condition> dcs;
    for (int c = 0; c < dim; ++c) {
        dcs.push_back(sfem::DirichletConditions::Condition{
                .sidesets = wall, .value = (c == 1 ? disp_y : real_t(0)), .component = c});
    }
    f->add_constraint(sfem::create_dirichlet_conditions(fs, dcs, sfem::EXECUTION_SPACE_HOST));

    auto contact_conds = sfem::ContactConditions::create(fs, sdf, contact_sides, sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(contact_conds != nullptr);
    SFEM_TEST_ASSERT(contact_conds->n_constrained_dofs() > 0);
    SFEM_TEST_ASSERT(contact_conds->ss_sides() != nullptr);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = sfem::create_host_buffer<real_t>(ndofs);
    auto            rhs   = sfem::create_host_buffer<real_t>(ndofs);
    std::fill(x->data(), x->data() + ndofs, real_t(0));
    std::fill(rhs->data(), rhs->data() + ndofs, real_t(0));
    f->apply_constraints(rhs->data());
    contact_conds->init();
    f->apply_constraints(x->data());

    auto solver = sfem::create_ssmgc(f, contact_conds, nullptr);
    SFEM_TEST_ASSERT(solver != nullptr);
    SFEM_TEST_ASSERT(solver->apply(rhs->data(), x->data()) == SFEM_SUCCESS);

    real_t nrm = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_ASSERT(std::isfinite(x->data()[i]));
        nrm += x->data()[i] * x->data()[i];
    }
    SFEM_TEST_ASSERT(std::isfinite(nrm));
    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_hex_ssmgc() {
    auto hex = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 2, 2, 2);
    auto ss  = smesh::to_semistructured(2, hex, true, false);
    SFEM_TEST_ASSERT(ss != nullptr);

    auto contact_ss = sfem::Sideset::create_from_selector(
            ss, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });
    SFEM_TEST_ASSERT(contact_ss.size() >= 2);

    auto comm = ss->comm();
    auto sdf  = smesh::create_sdf(comm,
                                 16,
                                 8,
                                 16,
                                 -0.1,
                                 -0.2,
                                 -0.1,
                                 1.1,
                                 0.2,
                                 1.1,
                                 [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                     const geom_t cx = 0.5, cy = -0.5, cz = 0.5, radius = 0.5;
                                     const geom_t dx = cx - x, dy = cy - y, dz = cz - z;
                                     return radius - sqrt(dx * dx + dy * dy + dz * dz);
                                 });
    return run_ssmgc_on_ss_mesh(ss, 3, contact_ss, sdf, real_t(-0.05));
}

int test_tet_ssmgc() {
    auto tet = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 2, 2, 2);
    auto ss  = smesh::to_semistructured(2, tet, true, false);
    SFEM_TEST_ASSERT(ss != nullptr);

    auto contact_ss = sfem::Sideset::create_from_selector(
            ss, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });
    SFEM_TEST_ASSERT(!contact_ss.empty());

    auto comm = ss->comm();
    auto sdf  = smesh::create_sdf(comm,
                                 16,
                                 8,
                                 16,
                                 -0.1,
                                 -0.2,
                                 -0.1,
                                 1.1,
                                 0.2,
                                 1.1,
                                 [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                     const geom_t cx = 0.5, cy = -0.5, cz = 0.5, radius = 0.5;
                                     const geom_t dx = cx - x, dy = cy - y, dz = cz - z;
                                     return radius - sqrt(dx * dx + dy * dy + dz * dz);
                                 });
    return run_ssmgc_on_ss_mesh(ss, 3, contact_ss, sdf, real_t(-0.05));
}

int test_quad2d_ssmgc() {
    auto quad = sfem::Mesh::create_quad4_square(sfem::Communicator::self(), 2, 2, 0, 0, 1, 1);
    auto ss   = smesh::to_semistructured(2, quad, true, false);
    SFEM_TEST_ASSERT(ss != nullptr);
    SFEM_TEST_EQ(ss->spatial_dimension(), 2);

    auto contact_ss = sfem::Sideset::create_from_selector(
            ss, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });
    SFEM_TEST_ASSERT(!contact_ss.empty());

    auto comm = ss->comm();
    auto sdf  = smesh::create_sdf(comm,
                                 16,
                                 8,
                                 4,
                                 -0.1,
                                 -0.2,
                                 -0.1,
                                 1.1,
                                 0.2,
                                 0.1,
                                 [](const geom_t x, const geom_t y, const geom_t /*z*/) -> geom_t {
                                     const geom_t cx = 0.5, cy = -0.5, radius = 0.5;
                                     const geom_t dx = cx - x, dy = cy - y;
                                     return radius - sqrt(dx * dx + dy * dy);
                                 });
    return run_ssmgc_on_ss_mesh(ss, 2, contact_ss, sdf, real_t(-0.05));
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_contact);
    SFEM_RUN_TEST(test_checkerboard_hex_ssmgc);
    SFEM_RUN_TEST(test_tet_ssmgc);
    SFEM_RUN_TEST(test_quad2d_ssmgc);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

