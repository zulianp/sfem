// The playground's node_sdf obstacle on four ranks against the same case on one: a box held at its
// top and started inside a plane, pushed out by utopia's scheme (four steps damped by 0.8) and
// released over four quasi-static steps to convergence. Every rank solves the case alone too, and
// each owned node is compared with the node of the serial solve at its coordinates. Then a block
// of a mesh read from disk, written by both: rank 0 writes the distributed run's output as the
// serial run does, each node by its id in the file.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
    // the Newton iterations of both solves, all steps, in `iterations`.
    int compare(const std::vector<std::string> &overrides, real_t *const relative, int iterations[2]) {
        auto world  = sfem::Communicator::world();
        auto serial = assemble(sfem::Communicator::self(), overrides);
        auto dist   = assemble(world, overrides);
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
        const real_t        *ud    = dist->solution()->data();
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

namespace {
    // out/<name>.<d>.<step> of a playground output, each node's values by its id in the file (its
    // index there without a mesh/node_mapping, a mesh read whole and not renumbered).
    std::map<long long, std::array<double, 3>> read_by_file_id(const std::string &out, const char *const name) {
        std::map<long long, std::array<double, 3>> ret;
        std::vector<int> ids;
        {
            std::ifstream is(out + "/mesh/node_mapping.int32", std::ios::binary);
            int           v;
            while (is.read(reinterpret_cast<char *>(&v), sizeof(v))) ids.push_back(v);
        }
        std::vector<double> comp[3];
        for (int d = 0; d < 3; ++d) {
            std::ifstream is(out + "/out/" + name + "." + std::to_string(d) + ".000000001.float64", std::ios::binary);
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
    }
    if (world->rank() == 0) std::filesystem::remove_all(dir);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_utopia_scheme_matches_serial);
    SFEM_RUN_TEST(test_release_matches_serial);
    SFEM_RUN_TEST(test_block_read_and_output_match_serial);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
