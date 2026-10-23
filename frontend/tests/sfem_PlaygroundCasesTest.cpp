// Small cases through the whole playground, one per regime.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Function.hpp"
#include "sfem_Playground.hpp"
#include "sfem_playground_test_history.hpp"
#include "sfem_test.hpp"

namespace {
    std::unique_ptr<sfem::Playground> assemble(const char *yaml, const std::vector<std::string> &overrides = {}) {
        auto config = sfem::ElasticityCase::parse(yaml, "", overrides);
        if (!config) return nullptr;
        return sfem::Playground::create(sfem::Communicator::world(), *config);
    }

}  // namespace

// Linear elasticity with the affine displacement u = A X on the whole boundary: trilinear elements
// represent it exactly, so the interior nodes take it to the solver's tolerance.
int test_steady_linear_patch() {
    const char *const yaml = R"(
mesh:
  box: {n: [3, 3, 3]}
  sidesets:
    - {name: x0, plane: {axis: x, at: min}}
    - {name: x1, plane: {axis: x, at: max}}
    - {name: y0, plane: {axis: y, at: min}}
    - {name: y1, plane: {axis: y, at: max}}
    - {name: z0, plane: {axis: z, at: min}}
    - {name: z1, plane: {axis: z, at: max}}
material: {type: GeneratedLinearElasticity, mu: 1, lmbda: 2}
regime: steady
solver: {newton: {verbose: false, rtol: 1e-12}}
)";
    // The boundary values are set below, node by node, from u = A X.
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);

    const real_t A[3][3] = {{0.01, 0.02, -0.01}, {0.0, -0.015, 0.005}, {0.02, 0.0, 0.01}};
    auto         f       = pg->function();
    auto         mesh    = f->space()->mesh_ptr();
    auto         xyz     = mesh->points()->data();
    const auto   n       = mesh->n_nodes();

    std::vector<idx_t> boundary;
    for (ptrdiff_t i = 0; i < n; ++i) {
        bool on = false;
        for (int d = 0; d < 3; ++d) on = on || std::abs(xyz[d][i]) < 1e-8 || std::abs(xyz[d][i] - 1) < 1e-8;
        if (on) boundary.push_back((idx_t)i);
    }
    auto dc = std::make_shared<sfem::DirichletConditions>(f->space());
    for (int d = 0; d < 3; ++d) {
        auto values = (real_t *)malloc(boundary.size() * sizeof(real_t));
        auto nodes  = (idx_t *)malloc(boundary.size() * sizeof(idx_t));
        for (size_t k = 0; k < boundary.size(); ++k) {
            const idx_t i = boundary[k];
            nodes[k]      = i;
            values[k]     = A[d][0] * xyz[0][i] + A[d][1] * xyz[1][i] + A[d][2] * xyz[2][i];
        }
        dc->add_condition(boundary.size(), boundary.size(), nodes, d, values);
    }
    f->add_constraint(dc);

    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    const real_t *const u = pg->solution()->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            const real_t exact = A[d][0] * xyz[0][i] + A[d][1] * xyz[1][i] + A[d][2] * xyz[2][i];
            SFEM_TEST_ASSERT(std::abs(u[i * 3 + d] - exact) < 1e-10);
        }
    }
    return SFEM_TEST_SUCCESS;
}

const char *const STRETCH = R"(
mesh:
  box: {n: [3, 3, 3]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
  - type: sideset
    format: name
    name: right
    component: 0
    value: 0.4
    profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}
regime: quasi_static
time: {dt: 0.25, t_end: 1}
solver: {newton: {verbose: false}}
)";

// The load ramps with time and every step converges; the stretched face ends where it was put.
int test_quasi_static_stretch() {
    auto pg = assemble(STRETCH);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto       mesh = pg->function()->space()->mesh_ptr();
    auto       xyz  = mesh->points()->data();
    const auto u    = pg->solution()->data();
    real_t worst = 0;
    int    on_face = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - 1) < 1e-6) {
            worst = std::max(worst, std::abs(u[i * 3] - real_t(0.4)));
            on_face++;
        }
    }
    printf("  %d nodes on the pulled face, largest deviation %g\n", on_face, (double)worst);
    SFEM_TEST_ASSERT(on_face == 16);
    SFEM_TEST_ASSERT(worst < 1e-14);
    return SFEM_TEST_SUCCESS;
}

// mesh.blocks: of a bar split at x = 0.5 into two blocks only "near" is read and stretched. The
// space holds only its nodes, and through packing each keeps its id in the file.
int test_mesh_blocks() {
    auto bar = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 6, 3, 3, 0, 0, 0, 1, 0.5, 0.5);
    auto picked = bar->select_elements([](const geom_t x, const geom_t, const geom_t) { return x < 0.5; });
    SFEM_TEST_ASSERT(picked.size() == 1);
    SFEM_TEST_ASSERT(bar->split_block(picked[0].second, "near") == SFEM_SUCCESS);
    const auto dir = (std::filesystem::temp_directory_path() /
                      ("sfem_mesh_blocks_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                             .string();
    std::filesystem::create_directories(dir);
    SFEM_TEST_ASSERT(bar->write(smesh::Path(dir) / "bar") == SFEM_SUCCESS);

    std::string yaml = std::string("mesh: {path: ") + dir + R"(/bar, blocks: [near],
  sidesets: [{name: left, plane: {axis: x, at: min}}, {name: right, plane: {axis: x, at: max}}]}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
  - {type: sideset, format: name, name: right, component: 0, value: 0.1}
regime: steady
solver: {newton: {verbose: false}}
)";
    auto pg = assemble(yaml.c_str());
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto mesh = pg->function()->space()->mesh_ptr();
    SFEM_TEST_EQ(mesh->n_nodes(), (ptrdiff_t)(4 * 4 * 4));
    auto map = mesh->node_mapping();
    SFEM_TEST_ASSERT(map != nullptr);
    const auto u        = pg->solution()->data();
    int        on_right = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        for (int d = 0; d < 3; ++d) {
            SFEM_TEST_ASSERT(mesh->points()->data()[d][i] == bar->points()->data()[d][map->data()[i]]);
        }
        if (std::abs(mesh->points()->data()[0][i] - 0.5) < 1e-6) {
            SFEM_TEST_ASSERT(std::abs(u[i * 3] - real_t(0.1)) < 1e-14);
            on_right++;
        }
    }
    SFEM_TEST_EQ(on_right, 16);

    // Read whole, the packed mesh still maps its nodes to the file's.
    auto whole = assemble((std::string("mesh: {path: ") + dir + "/bar}\nmaterial: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}\n").c_str());
    SFEM_TEST_ASSERT(whole != nullptr);
    auto wm   = whole->function()->space()->mesh_ptr();
    auto wmap = wm->node_mapping();
    SFEM_TEST_ASSERT(wmap != nullptr);
    SFEM_TEST_EQ(wm->n_nodes(), bar->n_nodes());
    for (ptrdiff_t i = 0; i < wm->n_nodes(); ++i) {
        for (int d = 0; d < 3; ++d) {
            SFEM_TEST_ASSERT(wm->points()->data()[d][i] == bar->points()->data()[d][wmap->data()[i]]);
        }
    }

    // Initial conditions in the file's numbering (component d of file node k is 3k + d) reach each
    // node through node_mapping, read by blocks and read whole and packed alike.
    for (int d = 0; d < 3; ++d) {
        std::ofstream file(dir + "/u" + std::to_string(d) + ".float64", std::ios::binary);
        for (ptrdiff_t k = 0; k < bar->n_nodes(); ++k) {
            const double v = 3 * k + d;
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
    }
    const std::string ic = "initial_conditions:\n  displacement: {components: [" + dir + "/u0.float64, " + dir + "/u1.float64, " + dir +
                           "/u2.float64]}\n";
    for (const std::string blocks : {", blocks: [near]", ""}) {
        auto read = assemble(
                (std::string("mesh: {path: ") + dir + "/bar" + blocks + "}\nmaterial: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}\n" + ic)
                        .c_str());
        SFEM_TEST_ASSERT(read != nullptr);
        auto       rm  = read->function()->space()->mesh_ptr();
        const auto ids = rm->node_mapping()->data();
        const auto x   = read->solution()->data();
        for (ptrdiff_t i = 0; i < rm->n_nodes(); ++i) {
            for (int d = 0; d < 3; ++d) SFEM_TEST_ASSERT(x[i * 3 + d] == real_t(3 * ids[i] + d));
        }
    }
    std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

// A plane restricted to a block: the bottom face of a cube split at x = 0.5 is cut by both blocks,
// and a Dirichlet value on the plane within block "near" holds only that half. A block the mesh
// lacks is refused.
int test_plane_within_block() {
    auto cube   = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 4, 4, 4, 0, 0, 0, 1, 1, 1);
    auto picked = cube->select_elements([](const geom_t x, const geom_t, const geom_t) { return x < 0.5; });
    SFEM_TEST_ASSERT(picked.size() == 1);
    SFEM_TEST_ASSERT(cube->split_block(picked[0].second, "near") == SFEM_SUCCESS);
    const auto dir = (std::filesystem::temp_directory_path() /
                      ("sfem_plane_block_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                             .string();
    std::filesystem::create_directories(dir);
    SFEM_TEST_ASSERT(cube->write(smesh::Path(dir) / "cube") == SFEM_SUCCESS);

    auto case_for = [&](const std::string &block) {
        return std::string("mesh: {path: ") + dir + "/cube, packed: false,\n" +
               "  sidesets: [{name: top, plane: {axis: y, at: max}}, {name: bottom, plane: {axis: y, at: min, block: " + block + "}}]}\n" +
               R"(material: {type: GeneratedLinearElasticity, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: top, component: [0, 1, 2], value: [0, 0, 0]}
  - {type: sideset, format: name, name: bottom, component: 1, value: 0.1}
regime: steady
solver: {newton: {verbose: false}}
)";
    };
    SFEM_TEST_ASSERT(assemble(case_for("far").c_str()) == nullptr);

    auto pg = assemble(case_for("near").c_str());
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    auto       mesh = pg->function()->space()->mesh_ptr();
    auto       xyz  = mesh->points()->data();
    const auto u    = pg->solution()->data();
    int        held = 0, free = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[1][i]) > 1e-8) continue;
        if (xyz[0][i] < 0.5 - 1e-8) {
            SFEM_TEST_ASSERT(u[i * 3 + 1] == real_t(0.1));
            held++;
        } else if (xyz[0][i] > 0.5 + 1e-8) {
            SFEM_TEST_ASSERT(u[i * 3 + 1] != real_t(0.1));
            free++;
        }
    }
    SFEM_TEST_ASSERT(held == 10 && free == 10);
    std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

// A tabulated profile names its table relative to the case, as every other path in it: a
// condition's own profile and the loads' profile both read p.csv beside the case, from anywhere.
int test_tabulated_beside_case() {
    const auto dir = (std::filesystem::temp_directory_path() /
                      ("sfem_tabulated_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                             .string();
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/p.csv") << "0, 0\n1, 1\n";

    const char *const yaml = R"(
mesh:
  box: {n: [2, 2, 2]}
  sidesets:
    - {name: x0, plane: {axis: x, at: min}}
    - {name: x1, plane: {axis: x, at: max}}
material: {type: GeneratedLinearElasticity, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: x0, component: [0, 1, 2], value: [0, 0, 0]}
loads:
  neumann:
    - {type: sideset, format: name, name: x1, component: 0, value: 0.1, profile: {type: tabulated, path: p.csv}}
  profile: {type: tabulated, path: p.csv}
regime: quasi_static
time: {dt: 1, t_end: 1}
solver: {newton: {verbose: false}}
)";
    auto config = sfem::ElasticityCase::parse(yaml, dir, {});
    SFEM_TEST_ASSERT(config != nullptr);
    auto pg = sfem::Playground::create(sfem::Communicator::world(), *config);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

// Two control points in one history, told apart by their names: a bar pulled at its free end
// moves there and not at its held end.
int test_named_probes() {
    const std::string csv  = "test_named_probes.csv";
    const char *const yaml = R"(
mesh:
  box: {n: [4, 1, 1], max: [4, 1, 1]}
  sidesets:
    - {name: x0, plane: {axis: x, at: min}}
    - {name: x1, plane: {axis: x, at: max}}
material: {type: GeneratedLinearElasticity, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: x0, component: [0, 1, 2], value: [0, 0, 0]}
loads:
  neumann:
    - {type: sideset, format: name, name: x1, component: 0, value: 0.1}
regime: quasi_static
time: {dt: 1, t_end: 1}
solver: {newton: {verbose: false}}
output:
  probes: [{type: control_point, at: [4, 0, 0], name: tip}, {type: control_point, at: [0, 0, 0], name: root}]
)";
    auto pg = assemble(yaml, {"output.history_csv=" + csv});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    auto rows = sfem::test::read_history(csv);
    SFEM_TEST_ASSERT(rows.size() == 2);
    SFEM_TEST_ASSERT(rows[1].count("tip_ux") && rows[1].count("root_ux") && !rows[1].count("ux"));
    SFEM_TEST_ASSERT(rows[1]["tip_ux"] > 0 && rows[1]["root_ux"] == 0);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_CUDA
// A ramped traction on a TET4 block's face (TRI3) on the device: each step's displacement the
// host's to rounding, so the device takes the traction's current value at every step.
int test_device_traction_matches_host() {
    auto       cube = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 3, 3, 3, 0, 0, 0, 1, 1, 1);
    const auto dir  = (std::filesystem::temp_directory_path() /
                      ("sfem_device_traction_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                             .string();
    std::filesystem::create_directories(dir);
    SFEM_TEST_ASSERT(cube->write(smesh::Path(dir) / "cube") == SFEM_SUCCESS);
    const std::string yaml = std::string("mesh: {path: ") + dir + R"(/cube, packed: false,
  sidesets: [{name: x0, plane: {axis: x, at: min}}, {name: x1, plane: {axis: x, at: max}}]}
material: {type: GeneratedLinearElasticity, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: x0, component: [0, 1, 2], value: [0, 0, 0]}
loads:
  neumann:
    - {type: sideset, format: name, name: x1, component: 1, value: 0.05,
       profile: {type: linear_ramp, start_time: 0, end_time: 3, start_value: 0, end_value: 1}}
regime: quasi_static
time: {dt: 1, t_end: 3}
solver: {newton: {verbose: false, rtol: 1e-12}}
output:
  probes: [{type: control_point, at: [1, 1, 1]}]
)";
    std::vector<std::map<std::string, double>> rows[2];
    for (int device = 0; device < 2; ++device) {
        const std::string csv = dir + (device ? "/device.csv" : "/host.csv");
        auto pg = assemble(yaml.c_str(), {"output.history_csv=" + csv, std::string("execution_space=") + (device ? "device" : "host")});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        rows[device] = sfem::test::read_history(csv);
    }
    SFEM_TEST_ASSERT(rows[0].size() == 4 && rows[1].size() == 4);
    for (int k = 1; k <= 3; ++k) {
        printf("  t = %g: uy host %.12g, device %.12g\n", rows[0][k]["t"], rows[0][k]["uy"], rows[1][k]["uy"]);
        SFEM_TEST_ASSERT(std::abs(rows[1][k]["uy"] - rows[0][k]["uy"]) <= 1e-10 * std::abs(rows[0][k]["uy"]));
    }
    SFEM_TEST_ASSERT(rows[0][3]["uy"] > rows[0][1]["uy"] && rows[0][1]["uy"] > 0);
    std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}
#endif

// A TET4 mesh with a hyperelastic material: the packed layout (the default) solves the stretch as
// the plain one does, node for node through node_mapping. TET4 has only an affine kernel, whatever
// the operator's affine preference. Packing reorders this mesh's nodes, so packed with
// renumber: false is refused rather than computed in the wrong numbering. Whole, and split at
// x = 0.5 into two blocks, each packed on its own.
int tet4_packed_matches_plain(const bool split) {
    auto cube = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 3, 3, 3, 0, 0, 0, 1, 1, 1);
    if (split) {
        auto picked = cube->select_elements([](const geom_t x, const geom_t, const geom_t) { return x < 0.5; });
        SFEM_TEST_ASSERT(picked.size() == 1);
        SFEM_TEST_ASSERT(cube->split_block(picked[0].second, "near") == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(cube->n_blocks() == 2);
    }
    const auto dir  = (std::filesystem::temp_directory_path() /
                      ("sfem_tet4_packed_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                             .string();
    std::filesystem::create_directories(dir);
    SFEM_TEST_ASSERT(cube->write(smesh::Path(dir) / "cube") == SFEM_SUCCESS);

    const std::string yaml = std::string("mesh: {path: ") + dir + R"(/cube,
  sidesets: [{name: left, plane: {axis: x, at: min}}, {name: right, plane: {axis: x, at: max}}]}
material: {type: GeneratedNeoHookeanOgden, young_modulus: 2.4, poisson_ratio: 0.2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
  - {type: sideset, format: name, name: right, component: 0, value: 0.2}
regime: steady
solver: {newton: {verbose: false}}
)";
    // Each run's displacement in the file's numbering.
    std::vector<real_t> u[2];
    for (const bool packed : {true, false}) {
        auto pg = assemble(yaml.c_str(), {std::string("mesh.packed=") + (packed ? "true" : "false")});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        const auto x    = pg->solution();
        auto       mesh = pg->function()->space()->mesh_ptr();
        auto       map  = mesh->node_mapping();
        u[packed].assign(x->size(), 0);
        for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
            const ptrdiff_t file = map ? map->data()[i] : i;
            for (int d = 0; d < 3; ++d) u[packed][file * 3 + d] = x->data()[i * 3 + d];
        }
    }
    SFEM_TEST_ASSERT(assemble(yaml.c_str(), {"mesh.renumber=false"}) == nullptr);
    real_t worst = 0, scale = 0;
    for (size_t i = 0; i < u[0].size(); ++i) {
        worst = std::max(worst, std::abs(u[0][i] - u[1][i]));
        scale = std::max(scale, std::abs(u[0][i]));
    }
    printf("  packed against plain (%s): %.3g of %.3g\n", split ? "two blocks" : "one block", (double)worst, (double)scale);
    SFEM_TEST_ASSERT(scale > 0.1 && worst <= 1e-10 * scale);
    std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

int test_tet4_packed_matches_plain() {
    SFEM_TEST_ASSERT(tet4_packed_matches_plain(false) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(tet4_packed_matches_plain(true) == SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

// A cantilever dropped under gravity, by each scheme with inertia: every step converges, and the
// tip falls.
int test_dynamic_cantilever() {
    const char *const yaml = R"(
mesh:
  box: {n: [6, 2, 2], min: [0, 0, 0], max: [3, 1, 1]}
  sidesets:
    - {name: clamp, plane: {axis: x, at: min}}
material: {type: GeneratedNeoHookeanOgden, mu: 50, lmbda: 100}
loads:
  body_force: {density: 1, g: [0, -1, 0]}
dirichlet_conditions:
  - {type: sideset, format: name, name: clamp, component: [0, 1, 2], value: [0, 0, 0]}
regime: dynamic
time: {dt: 0.05, steps: 5}
dynamics: {scheme: newmark, density: 1}
solver: {newton: {verbose: false}}
)";
    for (const char *scheme : {"newmark", "bdf2"}) {
        auto pg = assemble(yaml, {std::string("dynamics.scheme=") + scheme});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

        auto   mesh = pg->function()->space()->mesh_ptr();
        auto   xyz  = mesh->points()->data();
        real_t tip  = 0;
        for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
            if (std::abs(xyz[0][i] - 3) < 1e-8) tip = std::min(tip, pg->solution()->data()[i * 3 + 1]);
        }
        printf("  %s: tip deflection %g\n", scheme, (double)tip);
        SFEM_TEST_ASSERT(tip < 0);
    }
    return SFEM_TEST_SUCCESS;
}

// What is written is the case's to say: the displacement always, the velocity and the
// acceleration only when output.fields names them, and only in a dynamic regime.
int test_output_fields() {
    const char *const yaml = R"(
mesh:
  box: {n: [3, 1, 1], min: [0, 0, 0], max: [3, 1, 1]}
  packed: false
  sidesets:
    - {name: clamp, plane: {axis: x, at: min}}
material: {type: GeneratedNeoHookeanOgden, mu: 50, lmbda: 100}
loads:
  body_force: {density: 1, g: [0, -1, 0]}
dirichlet_conditions:
  - {type: sideset, format: name, name: clamp, component: [0, 1, 2], value: [0, 0, 0]}
regime: dynamic
time: {dt: 0.05, steps: 2}
dynamics: {scheme: newmark, density: 1}
solver: {newton: {verbose: false}}
)";
    const auto dir = (std::filesystem::temp_directory_path() /
                      ("sfem_output_fields_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                             .string();
    std::filesystem::create_directories(dir);
    auto written = [&](const std::string &out, const char *field) {
        int n = 0;
        for (auto &entry : std::filesystem::directory_iterator(out + "/out")) {
            n += entry.path().filename().string().rfind(std::string(field) + ".", 0) == 0;
        }
        return n;
    };
    const struct {
        const char *fields;
        int         velocity;
    } cases[] = {{nullptr, 0}, {"[disp, velocity, acceleration]", 9}};
    for (const auto &c : cases) {
        const std::string out       = dir + (c.fields ? "/named" : "/default");
        std::vector<std::string> overrides = {"output.path=" + out};
        if (c.fields) overrides.push_back(std::string("output.fields=") + c.fields);
        auto pg = assemble(yaml, overrides);
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        printf("  fields %s: disp %d, velocity %d, acceleration %d files\n", c.fields ? c.fields : "(none)", written(out, "disp"),
               written(out, "velocity"), written(out, "acceleration"));
        // Three exports (the initial state and two steps) of three components.
        SFEM_TEST_ASSERT(written(out, "disp") == 9);
        SFEM_TEST_ASSERT(written(out, "velocity") == c.velocity && written(out, "acceleration") == c.velocity);
    }
    SFEM_TEST_ASSERT(assemble(yaml, {"regime=quasi_static", "output.path=" + dir + "/static", "output.fields=[velocity]"}) == nullptr);
    std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

// A bar twisted at one end by a rotation about its axis, ramped over two quasi-static steps: the
// twisted face sits at the rigid rotation of its reference positions.
int test_rotation_constraint() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 2, 2], min: [0, 0, 0], max: [2, 1, 1]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
constraints:
  - type: rotation
    sideset: right
    axis: x
    center: [0, 0.5, 0.5]
    angle: 0.3
    profile: {type: linear_ramp, start_time: 0, end_time: 2, start_value: 0, end_value: 1}
regime: quasi_static
time: {dt: 1, steps: 2}
solver: {newton: {verbose: false}}
)";
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto         mesh = pg->function()->space()->mesh_ptr();
    auto         xyz  = mesh->points()->data();
    const auto   u    = pg->solution()->data();
    const real_t c = std::cos(real_t(0.3)), s = std::sin(real_t(0.3));
    int          twisted = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - 2) > 1e-6) continue;
        const real_t q1 = xyz[1][i] - real_t(0.5), q2 = xyz[2][i] - real_t(0.5);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 1] - (c * q1 - s * q2 - q1)) < 1e-12);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 2] - (s * q1 + c * q2 - q2)) < 1e-12);
        twisted++;
    }
    SFEM_TEST_ASSERT(twisted == 9);
    return SFEM_TEST_SUCCESS;
}

// A sine profile: the twisted face follows angle * sin(2 pi t / period).
int test_sine_rotation() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 2, 2], min: [0, 0, 0], max: [2, 1, 1]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
constraints:
  - {type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.3, profile: {type: sine, period: 4}}
regime: quasi_static
time: {dt: 1, steps: 3}
solver: {newton: {verbose: false}}
)";
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    // At t = 3 the angle is 0.3 sin(3 pi / 2) = -0.3.
    auto         mesh = pg->function()->space()->mesh_ptr();
    auto         xyz  = mesh->points()->data();
    const auto   u    = pg->solution()->data();
    const real_t c = std::cos(real_t(-0.3)), s = std::sin(real_t(-0.3));
    int          twisted = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - 2) > 1e-6) continue;
        const real_t q1 = xyz[1][i] - real_t(0.5), q2 = xyz[2][i] - real_t(0.5);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 1] - (c * q1 - s * q2 - q1)) < 1e-12);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 2] - (s * q1 + c * q2 - q2)) < 1e-12);
        twisted++;
    }
    SFEM_TEST_ASSERT(twisted == 9);
    return SFEM_TEST_SUCCESS;
}

const char *const PRONY_TORSION = R"(
mesh:
  box: {n: [6, 2, 2], min: [0, 0, 0], max: [3, 1, 1]}
  packed: false
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material:
  type: GeneratedMooneyRivlinProny
  C10: 0.3
  C01: 0.05
  K: 5
  relaxation: {iso: {prony: [{g: 0.5, tau: 1}]}}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
regime: quasi_static
time: {dt: 0.5, t_end: 4}
solver: {newton: {verbose: false, atol: 1e-11, rtol: 0}}
output:
  probes: [{type: torque, constraint: 0}]
)";

// A Prony solid twisted, held and released: the torque grows over the ramp, relaxes at constant
// twist, and vanishes with the grip.
int test_prony_torsion_release() {
    const std::string csv = "test_prony_torsion_release.csv";
    auto              pg  = assemble(PRONY_TORSION,
                                     {"output.history_csv=" + csv,
                                      "constraints=[{type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.4, "
                                      "profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}, "
                                      "release: 2.5}]"});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto rows = sfem::test::read_history(csv);
    SFEM_TEST_ASSERT(rows.size() == 9);
    // Rows 1..8 end at t = 0.5, 1, ..., 4: ramp to t = 1, hold to 2, released from 2.5.
    for (int k = 1; k <= 8; ++k) printf("  t = %g: angle %g, released %g, torque %g\n", rows[k]["t"], rows[k]["angle"], rows[k]["released"], rows[k]["torque"]);
    SFEM_TEST_ASSERT(rows[2]["torque"] > rows[1]["torque"] && rows[1]["torque"] > 0);
    SFEM_TEST_ASSERT(rows[3]["torque"] < rows[2]["torque"] && rows[4]["torque"] < rows[3]["torque"]);
    SFEM_TEST_ASSERT(rows[4]["angle"] == 0.4 && rows[4]["released"] == 0);
    SFEM_TEST_ASSERT(rows[5]["released"] == 1 && rows[5]["angle"] == 0);
    for (int k = 5; k <= 8; ++k) SFEM_TEST_ASSERT(std::abs(rows[k]["torque"]) < 1e-6 * rows[2]["torque"]);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

// The Prony probe on the twisted bar: its one term (tau = 1) is loaded while the twist ramps, then
// relaxes under the held twist at its own rate, and gamma is the step's factor, between g_inf = 0.5
// and 1.
int test_prony_probe() {
    const std::string csv = "test_prony_probe.csv";
    auto              pg  = assemble(PRONY_TORSION,
                                     {"output.history_csv=" + csv,
                                      "output.probes=[{type: torque, constraint: 0}, {type: prony}]",
                                      "constraints=[{type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.4, "
                                      "profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}}]"});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    auto rows = sfem::test::read_history(csv);
    SFEM_TEST_ASSERT(rows.size() == 9);
    SFEM_TEST_ASSERT(rows[1].count("iso_gamma") && rows[1].count("iso_H0") && !rows[1].count("iso_H1"));
    for (int k = 1; k <= 8; ++k) printf("  t = %g: gamma %g, H0 %g\n", rows[k]["t"], rows[k]["iso_gamma"], rows[k]["iso_H0"]);
    SFEM_TEST_ASSERT(rows[2]["iso_H0"] > rows[1]["iso_H0"] && rows[1]["iso_H0"] > 0);
    // Under the held twist the relaxed stress hardly changes (the interior creeps a little), so
    // each step decays the history by exp(-dt / tau) = exp(-0.5).
    for (int k = 3; k <= 8; ++k) SFEM_TEST_ASSERT(std::abs(rows[k]["iso_H0"] / rows[k - 1]["iso_H0"] - std::exp(-0.5)) < 1e-4);
    SFEM_TEST_ASSERT(rows[1]["iso_gamma"] > 0.5 && rows[1]["iso_gamma"] < 1);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

// The rotation driven by its torque: every step ends with the probe's torque on the ramped
// target, and under the held torque the Prony solid keeps turning.
int test_torque_control() {
    const std::string csv = "test_torque_control.csv";
    const double      T   = 2e-3;
    auto              pg  = assemble(PRONY_TORSION,
                                     {"output.history_csv=" + csv,
                                      "constraints=[{type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], "
                                      "control: {torque: 2e-3, tol: 1e-8}, "
                                      "profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}}]"});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto rows = sfem::test::read_history(csv);
    SFEM_TEST_ASSERT(rows.size() == 9);
    for (int k = 1; k <= 8; ++k) {
        const double target = T * std::min(1.0, rows[k]["t"]);
        printf("  t = %g: angle %g, torque %g (target %g), %g Newton\n", rows[k]["t"], rows[k]["angle"], rows[k]["torque"], target, rows[k]["newton"]);
        SFEM_TEST_ASSERT(std::abs(rows[k]["torque"] - target) <= 1e-8 * target);
    }
    // Creep: the angle grows at constant torque.
    for (int k = 3; k <= 8; ++k) SFEM_TEST_ASSERT(rows[k]["angle"] > rows[k - 1]["angle"]);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

// The Prony solid twisted and held with steps far longer than its relaxation time ends relaxed;
// started from that twist in a second run, it is at rest there: under the same held twist and
// short steps, the torque stays the relaxed one. Strained into the twist within the first step
// instead, it would carry the instantaneous torque, twice the relaxed one at g = 0.5.
int test_prony_starts_relaxed() {
    const auto dir = (std::filesystem::temp_directory_path() /
                      ("sfem_prony_relaxed_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                             .string();
    std::filesystem::create_directories(dir);
    const std::string csv[2] = {dir + "/relax.csv", dir + "/restart.csv"};

    auto relax = assemble(PRONY_TORSION,
                          {"output.history_csv=" + csv[0],
                           "output.path=" + dir + "/relax",
                           "time={dt: 50, t_end: 100}",
                           "constraints=[{type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.4, "
                           "profile: {type: linear_ramp, start_time: 0, end_time: 50, start_value: 0, end_value: 1}}]"});
    SFEM_TEST_ASSERT(relax != nullptr);
    SFEM_TEST_ASSERT(relax->run() == SFEM_SUCCESS);
    auto         relaxed = sfem::test::read_history(csv[0]);
    const double T       = relaxed.back()["torque"];
    SFEM_TEST_ASSERT(relaxed.size() == 3 && T > 0);

    std::string components;
    for (int d = 0; d < 3; ++d) {
        components += (d ? ", " : "") + dir + "/relax/out/disp." + std::to_string(d) + ".000000002.float64";
    }
    auto restart = assemble(PRONY_TORSION,
                            {"output.history_csv=" + csv[1],
                             "time={dt: 0.01, t_end: 0.05}",
                             "initial_conditions={displacement: {components: [" + components + "]}}",
                             "constraints=[{type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.4}]"});
    SFEM_TEST_ASSERT(restart != nullptr);
    SFEM_TEST_ASSERT(restart->run() == SFEM_SUCCESS);
    auto rows = sfem::test::read_history(csv[1]);
    SFEM_TEST_ASSERT(rows.size() == 6);
    for (int k = 1; k <= 5; ++k) {
        printf("  t = %g: torque %g, relaxed %g\n", rows[k]["t"], rows[k]["torque"], T);
        SFEM_TEST_ASSERT(std::abs(rows[k]["torque"] - T) <= 1e-6 * T);
    }
    std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

// A twist of 0.24 rad in one step moves the corners of the end face by about 2.7 of these
// elements. Set on the face alone, the increment inverts the layer behind it and the step fails;
// carried by Newton's first iteration, the interior follows and it converges. The case also has
// no solver section, which reads the defaults.
int test_boundary_increment_is_carried() {
    const char *const yaml = R"(
mesh:
  box: {n: [8, 16, 16], min: [0, 0, 0], max: [0.05, 0.1, 0.1]}
  packed: false
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedMooneyRivlinProny, C10: 0.3, C01: 0.05, K: 50}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
constraints:
  - {type: rotation, sideset: right, axis: x, center: [0, 0.05, 0.05], angle: 0.24}
regime: quasi_static
time: {dt: 1, steps: 1}
)";
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto         mesh = pg->function()->space()->mesh_ptr();
    auto         xyz  = mesh->points()->data();
    const auto   u    = pg->solution()->data();
    const real_t c = std::cos(real_t(0.24)), s = std::sin(real_t(0.24));
    int          twisted = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - real_t(0.05)) > 1e-8) continue;
        const real_t q1 = xyz[1][i] - real_t(0.05), q2 = xyz[2][i] - real_t(0.05);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 1] - (c * q1 - s * q2 - q1)) < 1e-12);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 2] - (s * q1 + c * q2 - q2)) < 1e-12);
        twisted++;
    }
    SFEM_TEST_ASSERT(twisted == 17 * 17);
    return SFEM_TEST_SUCCESS;
}

// A twist of 0.3 rad in one step of a nearly incompressible bar, with no profile to cut back:
// carried whole, the lifted state lands where the tangent is indefinite and Newton stalls;
// carried in fractions, each from the last converged state, it converges.
int test_large_increment_in_fractions() {
    const char *const yaml = R"(
mesh:
  box: {n: [40, 8, 8], min: [0, 0, 0], max: [1, 0.2, 0.2]}
  packed: false
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 0.6, lmbda: 50}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
constraints:
  - {type: rotation, sideset: right, axis: x, center: [0, 0.1, 0.1], angle: 0.3}
regime: quasi_static
time: {dt: 1, steps: 1}
solver: {newton: {verbose: false}}
)";
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto         mesh = pg->function()->space()->mesh_ptr();
    auto         xyz  = mesh->points()->data();
    const auto   u    = pg->solution()->data();
    const real_t c = std::cos(real_t(0.3)), s = std::sin(real_t(0.3));
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - 1) > 1e-8) continue;
        const real_t q1 = xyz[1][i] - real_t(0.1), q2 = xyz[2][i] - real_t(0.1);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 1] - (c * q1 - s * q2 - q1)) < 1e-12);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 2] - (s * q1 + c * q2 - q2)) < 1e-12);
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_steady_linear_patch);
    SFEM_RUN_TEST(test_quasi_static_stretch);
    SFEM_RUN_TEST(test_mesh_blocks);
    SFEM_RUN_TEST(test_tet4_packed_matches_plain);
    SFEM_RUN_TEST(test_plane_within_block);
    SFEM_RUN_TEST(test_tabulated_beside_case);
    SFEM_RUN_TEST(test_named_probes);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_device_traction_matches_host);
#endif
    SFEM_RUN_TEST(test_dynamic_cantilever);
    SFEM_RUN_TEST(test_output_fields);
    SFEM_RUN_TEST(test_rotation_constraint);
    SFEM_RUN_TEST(test_sine_rotation);
    SFEM_RUN_TEST(test_prony_torsion_release);
    SFEM_RUN_TEST(test_torque_control);
    SFEM_RUN_TEST(test_prony_starts_relaxed);
    SFEM_RUN_TEST(test_prony_probe);
    SFEM_RUN_TEST(test_boundary_increment_is_carried);
    SFEM_RUN_TEST(test_large_increment_in_fractions);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
