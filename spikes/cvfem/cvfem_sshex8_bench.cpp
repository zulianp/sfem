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

#ifdef CVFEM_ENABLE_SUBPAR
#include "cvfem_sshex8_em.hpp"
#endif
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
    // ndof applies is only affordable on a small mesh, so the probe check is opt-in and
    // self-limiting.
    const int  probe_diag = smesh::Env::read<int>("SFEM_BENCH_PROBE_DIAG", 0);
    const auto macros = parse_list(smesh::Env::read_string("SFEM_BENCH_MACROS", "2,4,6,8"));
    const auto levels = parse_list(smesh::Env::read_string("SFEM_BENCH_LEVELS", "2,4,8"));

    std::printf("%-4s %-10s %-11s %-11s %-11s %-11s %-11s %-11s %-9s %-9s %-9s %-10s %-10s %s\n",
                "L", "ndof", "naive_ns/d", "macro_ns/d", "affine_ns/d", "hoist_ns/d", "em24_ns/d", "em32_ns/d", "bd_nv", "bd_mac", "pgrad", "hoist+pg", "agree", "bd_agree");

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
            std::vector<scalar_t> y_hoi((size_t)ndof, 0), y_em((size_t)ndof, 0);
            sscvfem_apply_macro_local_hoisted(d, rho, mu, dir.data(), y_hoi.data());
#ifdef CVFEM_ENABLE_SUBPAR
            sscvfem_apply_macro_local_em(d, rho, mu, dir.data(), y_em.data());
#endif
            std::vector<scalar_t> y_emf((size_t)ndof, 0);
#ifdef CVFEM_ENABLE_SUBPAR
            sscvfem_apply_macro_local_emfull(d, rho, mu, dir.data(), y_emf.data());
#endif

            double dmax = 0, amax = 0;
            for (ptrdiff_t i = 0; i < ndof; ++i) {
                dmax = std::max(dmax, std::fabs(y_naive[(size_t)i] - y_macro[(size_t)i]));
                dmax = std::max(dmax, std::fabs(y_naive[(size_t)i] - y_aff[(size_t)i]));
                dmax = std::max(dmax, std::fabs(y_naive[(size_t)i] - y_hoi[(size_t)i]));
#ifdef CVFEM_ENABLE_SUBPAR
                dmax = std::max(dmax, std::fabs(y_naive[(size_t)i] - y_em[(size_t)i]));
                dmax = std::max(dmax, std::fabs(y_naive[(size_t)i] - y_emf[(size_t)i]));
#endif
                amax = std::max(amax, std::fabs(y_naive[(size_t)i]));
            }
            const double rel = (amax > 0) ? dmax / amax : dmax;

            // Block diagonal, both layouts. Checked twice: against each other, and -- on
            // the smallest configuration, where ndof applies is affordable -- against the
            // operator itself, by probing sscvfem_apply with unit vectors and picking out
            // the diagonal blocks. The second is what makes this a statement about the
            // Jacobian rather than about two functions agreeing with each other.
            std::vector<scalar_t> bd_naive, bd_macro;
            sscvfem_block_diag_naive(d, rho, mu, bd_naive);
            sscvfem_block_diag(d, rho, mu, bd_macro);

            double bdmax = 0, bdref = 0;
            for (size_t i = 0; i < bd_naive.size(); ++i) {
                bdmax = std::max(bdmax, std::fabs(bd_naive[i] - bd_macro[i]));
                bdref = std::max(bdref, std::fabs(bd_naive[i]));
            }
            const double bd_rel = (bdref > 0) ? bdmax / bdref : bdmax;

            double bd_probe_rel = 0;
            if (probe_diag && ndof <= 20000) {
                std::vector<scalar_t> ecol((size_t)ndof), ycol((size_t)ndof);
                double                pmax = 0, pref = 0;
                for (ptrdiff_t c = 0; c < ndof; ++c) {
                    std::fill(ecol.begin(), ecol.end(), scalar_t(0));
                    ecol[(size_t)c] = scalar_t(1);
                    std::fill(ycol.begin(), ycol.end(), scalar_t(0));
                    sscvfem_apply(d, rho, mu, ecol.data(), ycol.data());
                    // Column c of J touches the diagonal block of node c/4 in rows of the
                    // same node.
                    const ptrdiff_t node = c / 4, fld = c % 4;
                    for (int r = 0; r < 4; ++r) {
                        const double a = (double)ycol[(size_t)node * 4 + r];
                        const double b = (double)bd_macro[(size_t)node * 16 + r * 4 + fld];
                        pmax = std::max(pmax, std::fabs(a - b));
                        pref = std::max(pref, std::fabs(a));
                    }
                }
                bd_probe_rel = (pref > 0) ? pmax / pref : pmax;
            }

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

            const double t_hoi = time_it([&] {
                std::fill(y_hoi.begin(), y_hoi.end(), scalar_t(0));
                sscvfem_apply_macro_local_hoisted(d, rho, mu, dir.data(), y_hoi.data());
            });

#ifdef CVFEM_ENABLE_SUBPAR
            const double t_em = time_it([&] {
                std::fill(y_em.begin(), y_em.end(), scalar_t(0));
                sscvfem_apply_macro_local_em(d, rho, mu, dir.data(), y_em.data());
            });
#else
            const double t_em = 0;  // subpar; rebuild with -DCVFEM_ENABLE_SUBPAR=ON
#endif
#ifdef CVFEM_ENABLE_SUBPAR
            const double t_emf = time_it([&] {
                std::fill(y_emf.begin(), y_emf.end(), scalar_t(0));
                sscvfem_apply_macro_local_emfull(d, rho, mu, dir.data(), y_emf.data());
            });
            const double t_best = std::min(std::min(t_macro, t_aff), std::min(std::min(t_hoi, t_em), t_emf));
            const double t_bdn = time_it([&] { sscvfem_block_diag_naive(d, rho, mu, bd_naive); });
            const double t_bdm = time_it([&] { sscvfem_block_diag(d, rho, mu, bd_macro); });

#else
            const double t_emf  = 0;
            const double t_best = std::min(std::min(t_macro, t_aff), t_hoi);
#endif
            const double t_bdn = time_it([&] { sscvfem_block_diag_naive(d, rho, mu, bd_naive); });
            const double t_bdm = time_it([&] { sscvfem_block_diag(d, rho, mu, bd_macro); });

            // The nodal pressure gradient, timed separately because the flat operator
            // recomputes it inside every apply while this benchmark hoists it out of the
            // timed region. Any comparison against the flat kernel has to add it back, or
            // it is not measuring the same work.
            const double t_pg = time_it([&] { sscvfem_nodal_p_grad(d); });

            std::printf("%-4d %-10td %-11.3f %-11.3f %-11.3f %-11.3f %-11.3f %-11.3f %-9.3f %-9.3f %-9.3f %-10.3f %-10.2e %.2e%s\n",
                        L,
                        ndof,
                        1e9 * t_naive / (double)ndof,
                        1e9 * t_macro / (double)ndof,
                        1e9 * t_aff / (double)ndof,
                        1e9 * t_hoi / (double)ndof,
                        1e9 * t_em / (double)ndof,
                        1e9 * t_emf / (double)ndof,
                        1e9 * t_bdn / (double)ndof,
                        1e9 * t_bdm / (double)ndof,
                        1e9 * t_pg / (double)ndof,
                        1e9 * (t_hoi + t_pg) / (double)ndof,
                        rel,
                        std::max(bd_rel, bd_probe_rel),
                        (rel < tol) ? "" : "  <-- MISMATCH");
            if (rel >= tol || bd_rel >= tol || bd_probe_rel >= tol) ++failures;
        }
    }

    if (failures) {
        std::printf("cvfem_sshex8_bench: FAILED, %d configuration(s) disagree\n", failures);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
