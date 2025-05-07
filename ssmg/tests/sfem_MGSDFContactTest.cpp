#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

static const geom_t               y_top           = 0.2;
static const sfem::ExecutionSpace es_to_be_ported = sfem::EXECUTION_SPACE_HOST;

std::shared_ptr<sfem::ContactConditions> build_cuboid_sphere_contact(const std::shared_ptr<sfem::Function> &f,
                                                                     const int                              base_resolution) {
    auto fs   = f->space();
    auto m    = fs->mesh_ptr();
    auto comm = m->comm();
    auto es   = f->execution_space();

    auto top_ss = sfem::Sideset::create_from_selector(m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool {
        return y > (y_top - 1e-5) && y < (y_top + 1e-5);
    });

    const int n = base_resolution * fs->semi_structured_mesh().level();

    sfem::DirichletConditions::Condition xtop{.sideset = top_ss, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition ytop{.sideset = top_ss, .value = -0.05, .component = 1};
    sfem::DirichletConditions::Condition ztop{.sideset = top_ss, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);
    f->add_constraint(conds);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    auto sdf = sfem::create_sdf(comm,
                                n * 5 * 2,
                                n * 1 * 2,
                                n * 5 * 2,
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

    sdf->to_file("test_contact/sdf");
    auto contact_conds = sfem::ContactConditions::create(fs, sdf, bottom_ss, es);
    return contact_conds;
}

std::shared_ptr<sfem::ContactConditions> build_cuboid_highfreq_contact(const std::shared_ptr<sfem::Function> &f,
                                                                       const int                              base_resolution) {
    auto fs   = f->space();
    auto m    = fs->mesh_ptr();
    auto comm = m->comm();
    auto es   = f->execution_space();

    auto top_ss = sfem::Sideset::create_from_selector(m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool {
        return y > (y_top - 1e-5) && y < (y_top + 1e-5);
    });

    const int n = base_resolution * fs->semi_structured_mesh().level();

    sfem::DirichletConditions::Condition xtop{.sideset = top_ss, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition ytop{.sideset = top_ss, .value = -0.05, .component = 1};
    sfem::DirichletConditions::Condition ztop{.sideset = top_ss, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);
    f->add_constraint(conds);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    auto sdf = sfem::create_sdf(comm,
                                n * 5 * 2,
                                n * 1 * 2,
                                n * 5 * 2,
                                -0.1,
                                -0.2,
                                -0.1,
                                1.1,
                                y_top * 0.5,
                                1.1,
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

    sdf->to_file("test_contact/sdf");
    auto contact_conds = sfem::ContactConditions::create(fs, sdf, bottom_ss, es);
    return contact_conds;
}

std::shared_ptr<sfem::ContactConditions> build_cuboid_multisphere_contact(const std::shared_ptr<sfem::Function> &f,
                                                                          const int base_resolution) {
    auto fs   = f->space();
    auto m    = fs->mesh_ptr();
    auto comm = m->comm();
    auto es   = f->execution_space();

    auto top_ss = sfem::Sideset::create_from_selector(m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool {
        return y > (y_top - 1e-5) && y < (y_top + 1e-5);
    });

    const int n = base_resolution * fs->semi_structured_mesh().level();

    sfem::DirichletConditions::Condition xtop{.sideset = top_ss, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition ytop{.sideset = top_ss, .value = -0.1, .component = 1};
    sfem::DirichletConditions::Condition ztop{.sideset = top_ss, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);
    f->add_constraint(conds);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    int SFEM_N_SPHERES = 2;
    SFEM_READ_ENV(SFEM_N_SPHERES, atoi);
    auto sdf = sfem::create_sdf(comm,
                                n * 5 * 2,
                                n * 1 * 2,
                                n * 5 * 2,
                                -0.1,
                                -0.2,
                                -0.1,
                                1.1,
                                y_top * 0.5,
                                1.1,
                                [SFEM_N_SPHERES](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                    geom_t       dd = 1000000;
                                    const geom_t hx = 1. / (SFEM_N_SPHERES + 1);
                                    const geom_t hz = 1. / (SFEM_N_SPHERES + 1);
                                    const geom_t hy = 1. / (SFEM_N_SPHERES + 1);

                                    for (int i = 0; i < SFEM_N_SPHERES; i++) {
                                        for (int j = 0; j < SFEM_N_SPHERES; j++) {
                                            geom_t cx = hx + i * hx, cy = -0.1, cz = hz + j * hz;
                                            geom_t radius = 1. / (8 + SFEM_N_SPHERES);

                                            const geom_t dx = cx - x;
                                            const geom_t dy = cy - y;
                                            const geom_t dz = cz - z;

                                            const geom_t ddij = radius - sqrt(dx * dx + dy * dy + dz * dz);
                                            dd                = fabs(ddij) < fabs(dd) ? ddij : dd;
                                        }
                                    }

                                    return dd;
                                });

    sdf->to_file("test_contact/sdf");
    auto contact_conds = sfem::ContactConditions::create(fs, sdf, bottom_ss, es);
    return contact_conds;
}

int test_contact() {
    MPI_Comm comm = MPI_COMM_WORLD;

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 5, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 5, 0, 0, 0, 1, y_top, 1);

    const int block_size = m->spatial_dimension();

    auto fs = sfem::FunctionSpace::create(m, block_size);

    int SFEM_ELEMENT_REFINE_LEVEL = 2;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    SFEM_TEST_ASSERT(SFEM_ELEMENT_REFINE_LEVEL > 1);

    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

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
    auto op = sfem::create_op(fs, "LinearElasticity", es);
    op->initialize();

    f->add_operator(op);

    sfem::create_directory("test_contact");

    const char *SFEM_CONTACT_CASE = "sphere";
    SFEM_READ_ENV(SFEM_CONTACT_CASE, );

    std::shared_ptr<sfem::ContactConditions> contact_conds;

    if (strcmp(SFEM_CONTACT_CASE, "hifreq") == 0) {
        contact_conds = build_cuboid_highfreq_contact(f, SFEM_BASE_RESOLUTION);
    } else if (strcmp(SFEM_CONTACT_CASE, "sphere") == 0) {
        contact_conds = build_cuboid_sphere_contact(f, SFEM_BASE_RESOLUTION);
    } else if (strcmp(SFEM_CONTACT_CASE, "multisphere") == 0) {
        contact_conds = build_cuboid_multisphere_contact(f, SFEM_BASE_RESOLUTION);
    } else {
        SFEM_ERROR("SFEM_CONTACT_CASE=%s not valid!\n", SFEM_CONTACT_CASE);
    }

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs   = sfem::create_buffer<real_t>(ndofs, es);
    auto            gap   = sfem::create_buffer<real_t>(ndofs, es);

    f->apply_constraints(rhs->data());
    // contact_conds->update(x->data()); // FIXME
    contact_conds->init();

    fs->mesh_ptr()->write("test_contact/coarse_mesh");
    fs->semi_structured_mesh().export_as_standard("test_contact/mesh");
    auto out = f->output();
    out->set_output_dir("test_contact/out");
    out->enable_AoS_to_SoA(true);

    if (es != sfem::EXECUTION_SPACE_DEVICE) {
        contact_conds->signed_distance_for_mesh_viz(x->data(), gap->data());
        out->write("gap", gap->data());
    }

#ifdef SFEM_ENABLE_CUDA
    out->write("rhs", sfem::to_host(rhs)->data());
#else
    out->write("rhs", rhs->data());
#endif

    f->apply_constraints(x->data());

    int SFEM_USE_SPMG = 1;
    SFEM_READ_ENV(SFEM_USE_SPMG, atoi);

    if (SFEM_USE_SPMG) {
        std::shared_ptr<sfem::Input> in;
        const char                  *SFEM_SSMGC_YAML{nullptr};
        SFEM_READ_ENV(SFEM_SSMGC_YAML, );

        if (SFEM_SSMGC_YAML) {
            in = sfem::YAMLNoIndent::create_from_file(SFEM_SSMGC_YAML);
        }

        auto solver = sfem::create_ssmgc(f, contact_conds, in);
        solver->apply(rhs->data(), x->data());
    } else {
        auto solver = sfem::create_shifted_penalty(f, contact_conds, nullptr);
        solver->apply(rhs->data(), x->data());
    }

#ifdef SFEM_ENABLE_CUDA
    x = sfem::to_host(x);
#endif

    out->write("disp", x->data());

    if (es != sfem::EXECUTION_SPACE_DEVICE) {
        auto blas = sfem::blas<real_t>(es);
        blas->zeros(rhs->size(), rhs->data());
        f->gradient(x->data(), rhs->data());

        blas->zeros(x->size(), x->data());
        contact_conds->full_apply_boundary_mass_inverse(rhs->data(), x->data());
        out->write("contact_stress", x->data());
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_contact);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
