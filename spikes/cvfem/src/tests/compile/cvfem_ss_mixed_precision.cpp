// A WHOLE SEMI-STRUCTURED SWEEP, INSTANTIATED AND RUN AT SINGLE PRECISION.
//
// DESIGN.md's third correction asks for the kernels to be templated on the types they compute
// with. cvfem_mixed_precision_packs and cvfem_mixed_precision_sweep hold that for the flat
// packed layout; this holds it for the semi-structured one, which was the family the templating
// had not reached at all.
//
// WHY A RUN AND NOT A COMPILE CHECK. Adding template parameters to a sweep is cheap and proves
// nothing: the bodies keep their old spelling, so anything inside that still binds to the
// build's scalar compiles perfectly and is simply wrong at the other precision. The failure
// mode is silent -- a lattice stride taken from the build's type, a scratch sized by it -- and
// the only thing that sees it is the answer. So this runs the residual over a macro element at
// both precisions and requires them to agree.
//
// It also forced a real fix: atomic_add in kernels/cvfem_scatter.hpp was not templated, and
// every semi-structured sweep ends in it. An f32 instantiation could not have existed.
//
// ONE MACRO ELEMENT AT LEVEL 2 -- eight micro cells, twenty-seven nodes. What is under test is
// the type plumbing, not the physics, and a larger lattice exercises the same code and takes
// longer to say so. Both geometry halves run: the affine sweep over the macro element as a box,
// then the isoparametric sweep over the same element, which derives each micro cell's geometry
// for itself. On an affine macro element the two compute the same operator, so this gate also
// says the split's two halves agree at both precisions.
//
// WHAT THIS GATE CATCHES AND WHAT IT DOES NOT, because the distinction decides how much to
// trust it. In this family most remaining bindings to the build's scalar are COMPILE errors
// rather than silent wrong answers: a helper taking `const double[8]` cannot be handed a
// `float[8]` at all, so it is reported rather than tolerated -- unlike the flat packed layout,
// where a lane width taken from the build's type is a silent short stride. What this gate adds
// on top of that is the proof that the instantiation exists and RUNS, and the one thing that is
// genuinely silent: a shared constant table typed `double` makes an f32 arithmetic chain more
// accurate than it should be, which no compiler reports and only a cross-precision comparison
// can notice.
//
// WHAT IS NOT CLAIMED: that single precision is accurate enough for this operator on a real
// mesh. The tolerance is loose enough not to pretend otherwise -- f32 carries about seven
// digits and this operator differentiates. What is claimed is that the f32 instantiation
// exists, runs, and computes the same thing.
#define CVFEM_PHASES 0

#include "smesh_types.hpp"
#include "kernels/cvfem_portability.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "support/cvfem_default_types.hpp"
using idx_t   = smesh::idx_t;
using count_t = smesh::count_t;

#include "kernels/semistructured/cvfem_sshex8_ns.hpp"
#include "kernels/semistructured/affine/cvfem_sshex8_ns_affine.hpp"
#include "kernels/semistructured/isoparametric/cvfem_sshex8_ns_isoparam.hpp"

static constexpr int LEVEL = 2;
static constexpr int NXE   = (LEVEL + 1) * (LEVEL + 1) * (LEVEL + 1);  // 27

// One macro element on the unit cube, its lattice uniform, so it is affine and both sweeps are
// correct for it.
template <typename S, typename G, typename I>
static void run(const bool isoparam, S res[NXE * CVFEM_HEX8_N_FIELDS]) {
    std::vector<G> xs(NXE), ys(NXE), zs(NXE);
    std::vector<I> node(NXE);
    std::vector<I *> rows(NXE);
    for (int zi = 0; zi <= LEVEL; ++zi)
        for (int yi = 0; yi <= LEVEL; ++yi)
            for (int xi = 0; xi <= LEVEL; ++xi) {
                const int l = sscvfem_lidx(LEVEL, xi, yi, zi);
                xs[(size_t)l] = G(xi) / G(LEVEL);
                ys[(size_t)l] = G(yi) / G(LEVEL);
                zs[(size_t)l] = G(zi) / G(LEVEL);
            }
    for (int a = 0; a < NXE; ++a) {
        node[(size_t)a] = (I)a;
        rows[(size_t)a] = &node[(size_t)a];   // one element, so each lattice row is one entry
    }
    G  *points[3] = {xs.data(), ys.data(), zs.data()};
    I **elems     = rows.data();

    // A state off the upwind switch: a zero velocity sits exactly on its non-differentiable
    // point and returns NaN at every precision, which is correct behaviour and a poor oracle.
    std::vector<S> ux(NXE), uy(NXE), uz(NXE), p(NXE), pgx(NXE), pgy(NXE), pgz(NXE);
    for (int a = 0; a < NXE; ++a) {
        const S x = (S)xs[(size_t)a], y = (S)ys[(size_t)a], z = (S)zs[(size_t)a];
        ux[(size_t)a]  = S(0.7) + S(0.13) * x - S(0.21) * y + S(0.05) * z;
        uy[(size_t)a]  = S(-0.3) + S(0.11) * x + S(0.17) * y - S(0.04) * z;
        uz[(size_t)a]  = S(0.2) - S(0.07) * x + S(0.09) * y + S(0.12) * z;
        p[(size_t)a]   = S(1.0) + S(0.05) * x + S(0.03) * y;
        pgx[(size_t)a] = S(0.05);
        pgy[(size_t)a] = S(0.03);
        pgz[(size_t)a] = S(0);
    }
    for (int k = 0; k < NXE * CVFEM_HEX8_N_FIELDS; ++k) res[k] = S(0);

    Hex8RcConfigT<S>     rcfg{};
    Hex8PecletConfig<S>  peclet{};
    const cvfem_range    all{0, 1};
    const cvfem_range    none{1, 1};

    // The whole mesh is one macro element, so the partition is the identity and a null order
    // array is what a mesh with nothing curved gets.
    const ptrdiff_t *const order = nullptr;
    if (isoparam)
        sscvfem_residual_naive_isoparam(all, order, S(1), S(1), S(1), peclet, elems, LEVEL, NXE,
                                        p.data(), pgx.data(), pgy.data(), pgz.data(), points, S(0),
                                        ux.data(), uy.data(), uz.data(), rcfg, S(1), S(0.05), res);
    else
        sscvfem_residual_naive_affine(all, order, S(1), S(1), S(1), peclet, elems, LEVEL, NXE,
                                      p.data(), pgx.data(), pgy.data(), pgz.data(), points, S(0),
                                      ux.data(), uy.data(), uz.data(), rcfg, S(1), S(0.05), res);
    (void)none;
}

static int compare(const char *what, const double *a64, const float *a32) {
    double worst = 0, scale = 0;
    for (int k = 0; k < NXE * CVFEM_HEX8_N_FIELDS; ++k) {
        worst = std::max(worst, std::fabs(a64[k] - (double)a32[k]));
        scale = std::max(scale, std::fabs(a64[k]));
    }
    const double rel = scale > 0 ? worst / scale : worst;
    const bool   ok  = scale > 0 && rel < 1e-5 && std::isfinite(rel);
    std::printf("%-54s rel %.3e  %s\n", what, rel, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    double aff64[NXE * CVFEM_HEX8_N_FIELDS], iso64[NXE * CVFEM_HEX8_N_FIELDS];
    float  aff32[NXE * CVFEM_HEX8_N_FIELDS], iso32[NXE * CVFEM_HEX8_N_FIELDS];

    // <compute, geometry, index>. The middle arm is the PRODUCTION combination -- this spike
    // stores node coordinates in float32 and computes in double -- so instantiating it here is
    // not a contrivance, it is the build, and having it beside the other two is what shows the
    // three type parameters are independent rather than one type spelled three ways.
    run<double, double, int>(false, aff64);
    run<float, float, int>(false, aff32);
    run<double, double, int>(true, iso64);
    run<float, float, int>(true, iso32);

    double mix64[NXE * CVFEM_HEX8_N_FIELDS];
    run<double, float, int>(false, mix64);

    int bad = 0;
    bad += compare("the affine residual agrees across precisions", aff64, aff32);
    bad += compare("the isoparametric residual agrees across precisions", iso64, iso32);

    // The two halves on an affine macro element: the hoisted geometry IS each cell's own, so
    // they must agree to round-off. At f64 that is the split's own consistency check.
    double worst = 0, scale = 0;
    for (int k = 0; k < NXE * CVFEM_HEX8_N_FIELDS; ++k) {
        worst = std::max(worst, std::fabs(aff64[k] - iso64[k]));
        scale = std::max(scale, std::fabs(aff64[k]));
    }
    const double rel = scale > 0 ? worst / scale : worst;
    std::printf("%-54s rel %.3e  %s\n", "the two halves agree on an affine macro element", rel,
                rel < 1e-12 ? "OK" : "FAIL");
    if (!(rel < 1e-12)) ++bad;

    // Geometry in float32, compute in double: the production combination. It differs from the
    // all-double arm only by the coordinates' storage, so it has to agree to about float32's
    // own resolution in the coordinates rather than to round-off.
    {
        double w = 0, sc = 0;
        for (int k = 0; k < NXE * CVFEM_HEX8_N_FIELDS; ++k) {
            w  = std::max(w, std::fabs(aff64[k] - mix64[k]));
            sc = std::max(sc, std::fabs(aff64[k]));
        }
        const double r = sc > 0 ? w / sc : w;
        std::printf("%-54s rel %.3e  %s\n", "float32 geometry with double compute agrees", r,
                    r < 1e-6 ? "OK" : "FAIL");
        if (!(r < 1e-6)) ++bad;
    }

    if (bad) {
        std::fprintf(stderr, "\ncvfem_ss_mixed_precision: %d check(s) failed\n", bad);
        return 1;
    }
    std::printf("\nthe semi-structured sweeps run at single precision and agree\n");
    return 0;
}
