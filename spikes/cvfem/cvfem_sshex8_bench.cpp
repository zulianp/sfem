// T3: does the macro-local gather pay?
//
// Two implementations of the same operator on the same semi-structured mesh, differing
// only in how they reach the node data -- flat gather through the global id, versus one
// gather per macro-element followed by constant-offset reads from local buffers. They
// must agree to round-off, and that agreement is checked before any timing is reported,
// because two kernels that disagree cannot be compared on speed.
//
// Sizes and levels are both swept. T1 and T2 each produced a wrong conclusion from
// numbers taken below saturation, so nothing here is measured at a single point.

#include "cvfem_sshex8_ns.hpp"
#include "cvfem_ns_channel_case.hpp"

#include "sfem_context.hpp"
#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    }

    std::vector<int> parse_list(const std::string &spec) {
        std::vector<int> out;
        std::string      rest = spec;
        while (!rest.empty()) {
            const auto pos = rest.find(',');
            const auto tok = rest.substr(0, pos);
            if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
            if (pos == std::string::npos) break;
            rest = rest.substr(pos + 1);
        }
        return out;
    }

}  // namespace

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);

    const int    reps   = smesh::Env::read<int>("SFEM_BENCH_REPS", 10);
    const int    warmup = smesh::Env::read<int>("SFEM_BENCH_WARMUP", 3);
    const real_t Lx = 4, Ly = 1, Lz = 1;
    const real_t rho = 1, mu = 0.01, U = 1;
    const double tol = 1e-11;

    // Macro-element counts in y, and the internal level of each.
    const auto macros = parse_list(smesh::Env::read_string("SFEM_BENCH_MACROS", "2,4,6,8"));
    const auto levels = parse_list(smesh::Env::read_string("SFEM_BENCH_LEVELS", "2,4,8"));

    std::printf("%-4s %-10s %-12s %-12s %-12s %-9s %-9s %-12s %s\n",
                "L", "ndof", "naive_ns/d", "macro_ns/d", "affine_ns/d", "sp_macro", "sp_affine", "best_MDOF/s", "agree_rel");

    int failures = 0;

    for (const int L : levels) {
        for (const int m : macros) {
            const int ny = m;
            const int nx = std::max(1, (int)std::lround((double)ny * (double)Lx / (double)Ly));
            const int nz = std::max(1, (int)std::lround((double)ny * (double)Lz / (double)Ly));

            auto coarse = smesh::Mesh::create_hex8_cube(ctx->communicator(), nx, ny, nz, 0, 0, 0, Lx, Ly, Lz);
            auto mesh   = smesh::to_semistructured(L, coarse, true, false);
            if (!mesh) {
                std::fprintf(stderr, "to_semistructured failed for L=%d\n", L);
                return EXIT_FAILURE;
            }

            SSMeshData d;
            sscvfem_init(d, mesh, L);
            d.rhie_chow_scale = 1;

            const ptrdiff_t ndof = d.nnodes * N_FIELDS;

            std::vector<scalar_t> x((size_t)ndof), dir((size_t)ndof);
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                const scalar_t X = (scalar_t)d.points[0][i], Y = (scalar_t)d.points[1][i], Z = (scalar_t)d.points[2][i];
                scalar_t       ux, uy, uz, p;
                cvfem_case::exact_state(cvfem_case::FlowCase::Poiseuille, mu, U, Lx, Ly, X, Y, Z, ux, uy, uz, p);
                // Developed and asymmetric, so the upwind switch is actually exercised.
                x[(size_t)i * 4 + 0] = ux + scalar_t(0.1) * std::sin(scalar_t(3) * X);
                x[(size_t)i * 4 + 1] = uy + scalar_t(0.05) * std::cos(scalar_t(2) * Y);
                x[(size_t)i * 4 + 2] = uz + scalar_t(0.05) * std::sin(scalar_t(2) * Z);
                x[(size_t)i * 4 + 3] = p;
                for (int c = 0; c < 4; ++c)
                    dir[(size_t)i * 4 + c] = std::sin(scalar_t(0.7) * (scalar_t)(i * 4 + c) + scalar_t(0.3));
            }

            sscvfem_unpack(d, x.data());
            sscvfem_nodal_p_grad(d);

            std::vector<scalar_t> y_naive((size_t)ndof, 0), y_macro((size_t)ndof, 0), y_aff((size_t)ndof, 0);
            sscvfem_apply_naive(d, rho, mu, dir.data(), y_naive.data());
            sscvfem_apply_macro_local(d, rho, mu, dir.data(), y_macro.data());
            sscvfem_apply_macro_local_affine(d, rho, mu, dir.data(), y_aff.data());

            double dmax = 0, amax = 0;
            for (ptrdiff_t i = 0; i < ndof; ++i) {
                dmax = std::max(dmax, std::fabs(y_naive[(size_t)i] - y_macro[(size_t)i]));
                dmax = std::max(dmax, std::fabs(y_naive[(size_t)i] - y_aff[(size_t)i]));
                amax = std::max(amax, std::fabs(y_naive[(size_t)i]));
            }
            const double rel = (amax > 0) ? dmax / amax : dmax;

            auto time_it = [&](auto &&fn) {
                for (int r = 0; r < warmup; ++r) fn();
                std::vector<double> t;
                for (int r = 0; r < reps; ++r) {
                    const double t0 = smesh::time_seconds();
                    fn();
                    t.push_back(smesh::time_seconds() - t0);
                }
                return median(t);
            };

            const double t_naive = time_it([&] {
                std::fill(y_naive.begin(), y_naive.end(), scalar_t(0));
                sscvfem_apply_naive(d, rho, mu, dir.data(), y_naive.data());
            });
            const double t_macro = time_it([&] {
                std::fill(y_macro.begin(), y_macro.end(), scalar_t(0));
                sscvfem_apply_macro_local(d, rho, mu, dir.data(), y_macro.data());
            });

            const double t_aff = time_it([&] {
                std::fill(y_aff.begin(), y_aff.end(), scalar_t(0));
                sscvfem_apply_macro_local_affine(d, rho, mu, dir.data(), y_aff.data());
            });

            std::printf("%-4d %-10td %-12.3f %-12.3f %-12.3f %-9.2f %-9.2f %-12.1f %.3e%s\n",
                        L,
                        ndof,
                        1e9 * t_naive / (double)ndof,
                        1e9 * t_macro / (double)ndof,
                        1e9 * t_aff / (double)ndof,
                        t_naive / t_macro,
                        t_naive / t_aff,
                        1e-6 * (double)ndof / std::min(t_macro, t_aff),
                        rel,
                        (rel < tol) ? "" : "  <-- MISMATCH");
            if (rel >= tol) ++failures;
        }
    }

    if (failures) {
        std::printf("cvfem_sshex8_bench: FAILED, %d configuration(s) disagree\n", failures);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
