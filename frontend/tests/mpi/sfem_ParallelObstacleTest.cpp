// The playground's node_sdf obstacle on four ranks against the same case on one: a box held at its
// top and started inside a plane, pushed out by utopia's scheme (four steps damped by 0.8) and
// released over four quasi-static steps to convergence. Every rank solves the case alone too, and
// each owned node is compared with the node of the serial solve at its coordinates. Then a block
// of a mesh read from disk, written by both: rank 0 writes the distributed run's output as the
// serial run does, each node by its id in the file. Then a few dynamic steps from a displacement
// on disk, as the pump strokes start. With CUDA, the cases again with the distributed solve on the
// device.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <array>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Function.hpp"
#include "sfem_Playground.hpp"
#include "sfem_test.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "smesh_device_buffer.hpp"
#endif

namespace {
    const char *const BOX = R"(
mesh:
  box: {n: [6, 6, 6]}
  sidesets: [{name: top, plane: {axis: z, at: max}}]
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: top, component: [0, 1, 2], value: [0, 0, 0]}
obstacle:
  backend: node_sdf
  direction: z
  shape: {type: plane, plane: 0.1}
  curvature: false
regime: steady
solver: {newton: {max_it: 4, fixed_iterations: true, verbose: false, line_search: {rule: damped, damping: 0.8}}}
output: {probes: [{type: contact}]}
)";

    const std::vector<std::string> RELEASE = {
            "obstacle.curvature=true",
            "obstacle.release={type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}",
            "regime=quasi_static",
            "time={dt: 0.25, t_end: 1, max_cutbacks: 4}",
            "solver.newton={verbose: false}"};

    std::unique_ptr<sfem::Playground> assemble(const std::shared_ptr<sfem::Communicator> &comm,
                                               const std::vector<std::string>            &overrides) {
        auto config = sfem::ElasticityCase::parse(BOX, "", overrides);
        if (!config) return nullptr;
        return sfem::Playground::create(comm, *config);
    }

    using Key = std::tuple<long, long, long>;
    Key key(const geom_t x, const geom_t y, const geom_t z) {
        return {std::lround(x * 1e6), std::lround(y * 1e6), std::lround(z * 1e6)};
    }

    // max over owned nodes of |u_distributed - u_serial| against max |u_serial|, over all ranks;
    // the Newton iterations of both solves, all steps, in `iterations`. `distributed_only` applies
    // to the distributed solve alone (e.g. its execution space).
    int compare(const std::vector<std::string> &overrides,
                real_t *const                   relative,
                int                             iterations[2],
                const std::vector<std::string> &distributed_only = {}) {
        auto world  = sfem::Communicator::world();
        auto serial = assemble(sfem::Communicator::self(), overrides);
        auto both   = overrides;
        both.insert(both.end(), distributed_only.begin(), distributed_only.end());
        auto dist = assemble(world, both);
        if (!serial || !dist) return SFEM_FAILURE;
        iterations[0] = iterations[1] = 0;
        serial->newton().add_monitor([&](const sfem::NewtonIteration &it) { iterations[0] += it.iteration > 0; });
        dist->newton().add_monitor([&](const sfem::NewtonIteration &it) { iterations[1] += it.iteration > 0; });
        if (serial->run() != SFEM_SUCCESS || dist->run() != SFEM_SUCCESS) return SFEM_FAILURE;

        auto                 sm = serial->function()->space()->mesh_ptr();
        const geom_t *const *sp = sm->points()->data();
        std::map<Key, ptrdiff_t> at;
        for (ptrdiff_t i = 0; i < sm->n_nodes(); ++i) at[key(sp[0][i], sp[1][i], sp[2][i])] = i;

        auto                 dm    = dist->function()->space()->mesh_ptr();
        const geom_t *const *dp    = dm->points()->data();
        const ptrdiff_t      owned = dist->function()->space()->n_owned_dofs() / 3;
        const real_t        *us    = serial->solution()->data();
        auto                 dsol  = dist->solution();
#ifdef SFEM_ENABLE_CUDA
        if (dsol->mem_space() == sfem::MEMORY_SPACE_DEVICE) dsol = smesh::to_host(dsol);
#endif
        const real_t *ud = dsol->data();
        real_t               worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < owned; ++i) {
            auto found = at.find(key(dp[0][i], dp[1][i], dp[2][i]));
            if (found == at.end()) return SFEM_FAILURE;
            for (int d = 0; d < 3; ++d) {
                worst = std::max(worst, std::abs(ud[i * 3 + d] - us[found->second * 3 + d]));
                scale = std::max(scale, std::abs(us[found->second * 3 + d]));
            }
        }
        *relative = world->max(worst) / world->max(scale);
        return SFEM_SUCCESS;
    }
}  // namespace

int test_utopia_scheme_matches_serial() {
    real_t relative   = 1;
    int    its[2]     = {0, 0};
    SFEM_TEST_ASSERT(compare({}, &relative, its) == SFEM_SUCCESS);
    if (sfem::Communicator::world()->rank() == 0) {
        printf("  utopia scheme: %d ranks against one, %.3g relative, %d and %d Newton\n",
               sfem::Communicator::world()->size(),
               (double)relative,
               its[1],
               its[0]);
    }
    SFEM_TEST_ASSERT(relative < 1e-8);
    SFEM_TEST_ASSERT(its[0] == its[1]);
    return SFEM_TEST_SUCCESS;
}

int test_release_matches_serial() {
    real_t relative = 1;
    int    its[2]   = {0, 0};
    SFEM_TEST_ASSERT(compare(RELEASE, &relative, its) == SFEM_SUCCESS);
    if (sfem::Communicator::world()->rank() == 0) {
        printf("  release: %d ranks against one, %.3g relative, %d and %d Newton\n",
               sfem::Communicator::world()->size(),
               (double)relative,
               its[1],
               its[0]);
    }
    SFEM_TEST_ASSERT(relative < 1e-8);
    SFEM_TEST_ASSERT(its[0] == its[1]);
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_CUDA
namespace {
    // Both cases with the distributed solve on the device, a GPU per rank, against the serial
    // solve on the host.
    int device_matches_serial(const std::vector<std::string> &device, const char *const how) {
        for (const auto &overrides : {std::vector<std::string>{}, RELEASE}) {
            real_t relative = 1;
            int    its[2]   = {0, 0};
            SFEM_TEST_ASSERT(compare(overrides, &relative, its, device) == SFEM_SUCCESS);
            if (sfem::Communicator::world()->rank() == 0) {
                printf("  %s on the device%s: %d ranks against one on the host, %.3g relative, %d and %d Newton\n",
                       overrides.empty() ? "utopia scheme" : "release",
                       how,
                       sfem::Communicator::world()->size(),
                       (double)relative,
                       its[1],
                       its[0]);
            }
            SFEM_TEST_ASSERT(relative < 1e-8);
            SFEM_TEST_ASSERT(its[0] == its[1]);
        }
        return SFEM_TEST_SUCCESS;
    }
}  // namespace

// The gathers by GPU-aware MPI, the QP's sums completed on the host.
int test_device_matches_serial() { return device_matches_serial({"execution_space=device"}, ""); }

#ifdef SMESH_ENABLE_NCCL
// With NCCL within the node (gathers, and the QP's sums where one node holds every rank). Last:
// the communicator's device communication stays enabled for the process.
int test_device_nccl_matches_serial() {
    return device_matches_serial({"execution_space=device", "nccl=true"}, " with NCCL");
}
#endif
#endif

namespace {
    // out/<name>.<d>.<step> of a playground output, each node's values by its id in the file (its
    // index there without a mesh/node_mapping, a mesh read whole and not renumbered).
    std::map<long long, std::array<double, 3>> read_by_file_id(const std::string &out, const char *const name, const int step = 1) {
        std::map<long long, std::array<double, 3>> ret;
        std::vector<int> ids;
        {
            std::ifstream is(out + "/mesh/node_mapping.int32", std::ios::binary);
            int           v;
            while (is.read(reinterpret_cast<char *>(&v), sizeof(v))) ids.push_back(v);
        }
        std::vector<double> comp[3];
        for (int d = 0; d < 3; ++d) {
            char file[64];
            snprintf(file, sizeof(file), "%s.%d.%09d.float64", name, d, step);
            std::ifstream is(out + "/out/" + file, std::ios::binary);
            double        v;
            while (is.read(reinterpret_cast<char *>(&v), sizeof(v))) comp[d].push_back(v);
        }
        if (ids.empty()) {
            for (size_t i = 0; i < comp[0].size(); ++i) ids.push_back((int)i);
        }
        if (ids.size() != comp[0].size()) return {};
        for (size_t i = 0; i < ids.size(); ++i) ret[ids[i]] = {comp[0][i], comp[1][i], comp[2][i]};
        return ret;
    }
}  // namespace

// A block of a TET4 mesh on disk (mesh.blocks), and the whole two-block mesh, packed, the output
// written by rank 0. The gap is written at every contact node: one lost by the distributed skin
// reads zero there. Whole, a rank can hold none of a block, and the skin's collectives must still
// meet.
int test_block_read_and_output_match_serial() {
    auto       world = sfem::Communicator::world();
    const auto dir   = std::string("sfem_parallel_obstacle_case");
    if (world->rank() == 0) {
        auto cube   = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 6, 6, 6, 0, 0, 0, 1, 1, 1);
        auto picked = cube->select_elements([](const geom_t x, const geom_t, const geom_t) { return x < 0.5; });
        SFEM_TEST_ASSERT(picked.size() == 1);
        SFEM_TEST_ASSERT(cube->split_block(picked[0].second, "near") == SFEM_SUCCESS);
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        SFEM_TEST_ASSERT(cube->write(smesh::Path(dir) / "cube") == SFEM_SUCCESS);
    }
    world->barrier();

    for (const std::string blocks : {", blocks: [near]", ""}) {
        const std::vector<std::string> base = {
                "mesh={path: " + dir + "/cube" + blocks + ", sidesets: [{name: top, plane: {axis: z, at: max}}]}",
                "output.fields=[gap]"};
        std::string out[2];
        for (int distributed = 0; distributed < 2; ++distributed) {
            out[distributed] = dir + (distributed ? "/out_mpi" : "/out_serial") + (blocks.empty() ? "_whole" : "_block");
            auto overrides   = base;
            overrides.push_back("output.path=" + out[distributed]);
            if (!distributed && world->rank() != 0) continue;
            auto pg = assemble(distributed ? world : sfem::Communicator::self(), overrides);
            SFEM_TEST_ASSERT(pg != nullptr);
            SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        }
        world->barrier();

        // Compared on rank 0, the verdict on every rank: a rank leaving alone strands the others.
        int ok = 1;
        if (world->rank() == 0) {
            for (const char *name : {"disp", "gap"}) {
                auto a = read_by_file_id(out[0], name), b = read_by_file_id(out[1], name);
                if (a.empty() || a.size() != b.size()) {
                    printf("  %s: %zu nodes written by one rank, %zu by four\n", name, a.size(), b.size());
                    ok = 0;
                    continue;
                }
                double worst = 0, scale = 0;
                for (auto &[id, v] : a) {
                    auto found = b.find(id);
                    if (found == b.end()) {
                        ok = 0;
                        break;
                    }
                    for (int d = 0; d < 3; ++d) {
                        worst = std::max(worst, std::abs(v[d] - found->second[d]));
                        scale = std::max(scale, std::abs(v[d]));
                    }
                }
                printf("  %s, written by rank 0: %s over %zu nodes, %.3g of %.3g\n",
                       blocks.empty() ? "two blocks" : "one block",
                       name,
                       a.size(),
                       worst,
                       scale);
                ok = ok && scale > 0 && worst <= 1e-8 * scale;
            }
        }
        world->broadcast(&ok, 1, 0);
        SFEM_TEST_ASSERT(ok);

        // The obstacle read from the grid the serial run wrote, which the ranks split: the output
        // set up by the distributed playground holds it whole.
        {
            auto overrides = base;
            overrides.push_back("output.path=" + out[1] + "_grid");
            overrides.push_back("obstacle.sdf={file: " + out[0] + "/sdf}");
            SFEM_TEST_ASSERT(assemble(world, overrides) != nullptr);
            world->barrier();
        }
        if (world->rank() == 0) {
            auto slurp = [](const std::string &path) {
                std::ifstream is(path, std::ios::binary);
                return std::string(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>());
            };
            const auto grid_serial = slurp(out[0] + "/sdf/sdf.float32"), grid_mpi = slurp(out[1] + "_grid/sdf/sdf.float32");
            printf("  the obstacle's grid read by %d ranks, written by rank 0: %zu bytes, serial %zu, %s\n",
                   world->size(), grid_mpi.size(), grid_serial.size(), grid_mpi == grid_serial ? "the same" : "different");
            ok = !grid_serial.empty() && grid_mpi == grid_serial;
        }
        world->broadcast(&ok, 1, 0);
        SFEM_TEST_ASSERT(ok);
    }
    if (world->rank() == 0) std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

namespace {
    // The rows of a history.csv, each column by name.
    std::vector<std::map<std::string, double>> read_history(const std::string &path) {
        std::vector<std::map<std::string, double>> rows;
        std::ifstream                             is(path);
        std::string                               line;
        std::vector<std::string>                  names;
        auto split = [](const std::string &l) {
            std::vector<std::string> out;
            size_t                   at = 0;
            while (true) {
                const size_t n = l.find(',', at);
                out.push_back(l.substr(at, n == std::string::npos ? std::string::npos : n - at));
                if (n == std::string::npos) return out;
                at = n + 1;
            }
        };
        if (!std::getline(is, line)) return rows;
        names = split(line);
        while (std::getline(is, line)) {
            const auto cells = split(line);
            if (cells.size() != names.size()) return {};
            std::map<std::string, double> row;
            for (size_t i = 0; i < names.size(); ++i) {
                if (names[i] != "status") row[names[i]] = std::atof(cells[i].c_str());
            }
            rows.push_back(row);
        }
        return rows;
    }

    // The block of the TET4 mesh above, held at its top, through five BDF2 steps of the Prony
    // Mooney-Rivlin material from a displacement on disk, read in the file's numbering on every
    // rank; a traction ramped onto its bottom face (a plane sideset restricted to the block) pushes
    // it onto the plane obstacle, and a lateral one acts on a disc of that face, which some ranks
    // hold none of. Then the whole two-block mesh, the material in both, where a rank can hold
    // none of a block. A control point's probe is compared too: rank 0 writes it, wherever it is. Rank 0 writes both runs; the last step is compared by file id,
    // with the Newton iterations. `distributed` applies to the four-rank run alone.
    int strokes_match_serial(const std::vector<std::string> &distributed, const char *const how) {
        auto       world = sfem::Communicator::world();
        const auto dir   = std::string("sfem_parallel_strokes_case");
        if (world->rank() == 0) {
            auto cube   = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 6, 6, 6, 0, 0, 0, 1, 1, 1);
            auto picked = cube->select_elements([](const geom_t x, const geom_t, const geom_t) { return x < 0.5; });
            SFEM_TEST_ASSERT(picked.size() == 1);
            SFEM_TEST_ASSERT(cube->split_block(picked[0].second, "near") == SFEM_SUCCESS);
            std::filesystem::remove_all(dir);
            std::filesystem::create_directories(dir);
            SFEM_TEST_ASSERT(cube->write(smesh::Path(dir) / "cube") == SFEM_SUCCESS);
            // Zero at the held top, different in every component and node.
            const geom_t *const *p = cube->points()->data();
            for (int d = 0; d < 3; ++d) {
                std::ofstream os(dir + "/u0." + std::to_string(d) + ".float64", std::ios::binary);
                for (ptrdiff_t i = 0; i < cube->n_nodes(); ++i) {
                    const double x = p[0][i], y = p[1][i], z = p[2][i];
                    const double u = (1 - z) * (d == 0 ? 0.01 * y : d == 1 ? -0.008 * x : 0.005 * (x + y));
                    os.write(reinterpret_cast<const char *>(&u), sizeof(u));
                }
            }
        }
        world->barrier();

        for (const std::string blocks : {", blocks: [near]", ""}) {
            const std::vector<std::string> base = {
                    "mesh={path: " + dir + "/cube" + blocks + ", sidesets: [{name: top, plane: {axis: z, at: max}}, "
                    "{name: bottom, plane: {axis: z, at: min, block: near}}, "
                    "{name: corner, plane: {axis: z, at: min, block: near, center: [0.1, 0.1, 0], radius: 0.2}}]}",
                    "material={type: GeneratedMooneyRivlinProny, C10: 0.4, C01: 0.1, K: 5, relaxation: {iso: {prony: "
                    "[{g: 0.3, tau: 0.1}, {g: 0.2, tau: 1}]}}}",
                    "regime=dynamic",
                    "dynamics={scheme: bdf2, density: 1, mass: lumped}",
                    "time={dt: 0.05, t_end: 0.25, max_cutbacks: 4}",
                    "obstacle.shape={type: plane, plane: -0.02}",
                    "obstacle.curvature=true",
                    // The corner: a disc of the bottom that some ranks hold none of.
                    "loads={neumann: [{type: sideset, format: name, name: bottom, component: 2, value: -1, "
                    "profile: {type: linear_ramp, start_time: 0, end_time: 0.1, start_value: 0, end_value: 1}}, "
                    "{type: sideset, format: name, name: corner, component: 0, value: 0.5}]}",
                    "initial_conditions={displacement: {components: [" + dir + "/u0.0.float64, " + dir + "/u0.1.float64, " + dir +
                            "/u0.2.float64]}}",
                    "solver={newton: {max_it: 30, verbose: false}}",
                    // A point of the far corner of the block, which rank 0 need not hold.
                    "output.probes=[{type: contact}, {type: control_point, name: far, at: [0.45, 0.95, 0.05]}]"};
            std::string out[2];
            int         its[2] = {0, 0};
            for (int d = 0; d < 2; ++d) {
                out[d]         = dir + (d ? "/out_mpi" : "/out_serial") + (blocks.empty() ? "_whole" : "_block");
                auto overrides = base;
                overrides.push_back("output.path=" + out[d]);
                overrides.push_back("output.history_csv=" + out[d] + "/history.csv");
                if (d) overrides.insert(overrides.end(), distributed.begin(), distributed.end());
                if (!d && world->rank() != 0) continue;
                auto pg = assemble(d ? world : sfem::Communicator::self(), overrides);
                SFEM_TEST_ASSERT(pg != nullptr);
                pg->newton().add_monitor([&its, d](const sfem::NewtonIteration &it) { its[d] += it.iteration > 0; });
                SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
            }
            world->barrier();

            int ok = 1;
            if (world->rank() == 0) {
                auto a = read_by_file_id(out[0], "disp", 5), b = read_by_file_id(out[1], "disp", 5);
                auto start = read_by_file_id(out[0], "disp", 0), start_mpi = read_by_file_id(out[1], "disp", 0);
                double worst = 0, scale = 0, moved = 0, worst_start = 0;
                for (auto &[id, v] : a) {
                    auto found = b.find(id);
                    if (found == b.end() || !start_mpi.count(id)) {
                        ok = 0;
                        break;
                    }
                    for (int d = 0; d < 3; ++d) {
                        worst       = std::max(worst, std::abs(v[d] - found->second[d]));
                        worst_start = std::max(worst_start, std::abs(start[id][d] - start_mpi[id][d]));
                        scale       = std::max(scale, std::abs(v[d]));
                        moved       = std::max(moved, std::abs(v[d] - start[id][d]));
                    }
                }
                printf("  strokes%s, %s: %zu nodes, start %.3g, last step %.3g of %.3g (%.3g from the start), %d and %d Newton\n",
                       how, blocks.empty() ? "two blocks" : "one block", a.size(), worst_start, worst, scale, moved, its[0], its[1]);
                // The start is the file's displacement, and the steps move it.
                double start_scale = 0;
                for (auto &[id, v] : start) start_scale = std::max(start_scale, std::abs(v[0]));
                // The probe's columns in both histories, every step.
                const auto   ha = read_history(out[0] + "/history.csv"), hb = read_history(out[1] + "/history.csv");
                double       probe = ha.empty() || ha.size() != hb.size() ? 1 : 0, probe_scale = 0;
                for (size_t k = 0; k < ha.size() && k < hb.size(); ++k) {
                    for (auto &[name, v] : ha[k]) {
                        if (name.rfind("far_", 0) != 0) continue;
                        probe       = std::max(probe, std::abs(v - hb[k].at(name)));
                        probe_scale = std::max(probe_scale, std::abs(v));
                    }
                }
                printf("    the probe at the far corner: %.3g of %.3g over %zu rows\n", probe, probe_scale, ha.size());
                ok = ok && !a.empty() && a.size() == b.size() && start_scale > 0 && moved > 0.1 * scale && worst_start == 0 && worst <= 1e-8 * scale &&
                     its[0] == its[1] && probe_scale > 0 && probe <= 1e-8 * probe_scale;
            }
            world->broadcast(&ok, 1, 0);
            SFEM_TEST_ASSERT(ok);
        }
        if (world->rank() == 0) std::filesystem::remove_all(dir);
        return SFEM_TEST_SUCCESS;
    }
}  // namespace

int test_strokes_match_serial() { return strokes_match_serial({}, ""); }

#ifdef SFEM_ENABLE_CUDA
int test_device_strokes_match_serial() { return strokes_match_serial({"execution_space=device"}, " on the device"); }
#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_utopia_scheme_matches_serial);
    SFEM_RUN_TEST(test_release_matches_serial);
    SFEM_RUN_TEST(test_block_read_and_output_match_serial);
    SFEM_RUN_TEST(test_strokes_match_serial);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_device_matches_serial);
    SFEM_RUN_TEST(test_device_strokes_match_serial);
#ifdef SMESH_ENABLE_NCCL
    SFEM_RUN_TEST(test_device_nccl_matches_serial);
#endif
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
