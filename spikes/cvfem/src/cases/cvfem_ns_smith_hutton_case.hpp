#ifndef CVFEM_NS_SMITH_HUTTON_CASE_HPP
#define CVFEM_NS_SMITH_HUTTON_CASE_HPP

#include <cmath>

// The Smith-Hutton problem (1982), which is the standard benchmark for whether a convection
// scheme is BOUNDED -- and the case this spike did not have.
//
// Every case in the matrix so far is smooth enough that a limiter is a refinement: the
// backward-facing step, the cavity, the manufactured solution. The boundedness counter says
// half to two thirds of reconstructions leave their two-node interval on the cavity, so the
// limiter is exercised, but nothing there distinguishes a scheme that bounds from one that
// overshoots a little, because there is no sharp feature and no reference to be wrong about.
//
// Smith and Hutton devised this precisely to separate them. A prescribed rotating field
// carries a near-discontinuity from the inlet, round the half-turn, to the outlet:
//
//     u =  2 y (1 - x^2)          on  -1 <= x <= 1,  0 <= y <= 1
//     v = -2 x (1 - y^2)
//
//     inlet   y = 0, x < 0 :  phi = 1 + tanh(alpha (2x + 1))      alpha = 10
//     outlet  y = 0, x > 0 :  d phi / d y = 0
//     elsewhere            :  phi = 1 - tanh(alpha)
//
// First-order upwinding smears the front into the diagonal; unlimited higher-order schemes
// overshoot and oscillate. That contrast is the entire point of the problem and is what the
// literature uses it for.
//
// HOW IT RUNS HERE WITHOUT A SCALAR TRANSPORT EQUATION. This solver carries (u, v, w, p) and
// no passive scalar, and adding one would mean a second convective path -- which is the thing
// this spike refuses to have. It is not needed. Take the domain one cell thick in z with no
// z-variation, constrain u and v at EVERY node to the field above, and leave w free. The
// z-momentum equation is then
//
//     rho (u dw/dx + v dw/dy) = mu (d2w/dx2 + d2w/dy2),
//
// which is Smith-Hutton's scalar equation exactly, with phi = w and rho/Gamma = rho/mu. The
// prescribed field is divergence-free (du/dx = -4xy, dv/dy = +4xy), so continuity holds at
// constant pressure and the momentum equations for u and v are satisfied by construction.
//
// The value of doing it this way rather than with a new scalar: phi is transported by the
// PRODUCTION convective kernel, so the limiter under test is the one the solver runs, not a
// reimplementation of it that could differ.
//
// THE MESH IS [0, 2] x [0, 1], not [-1, 1] x [0, 1]. Everything here shifts x by one, because
// the mesh generator builds boxes from the origin and a case is not a reason to change it.
// STATUS, 2026-09-25. The case is correct and converges QUADRATICALLY at rho/Gamma = 10 and
// 100 -- 2.4e-1 -> 5.6e-3 -> 5.1e-4 -> 6.4e-6 -> 1.1e-9 -- and the relative trajectories at
// those two are identical to the digit with the absolute residuals scaling exactly ten to
// one, which is the signature of a problem linear in phi and confirms the formulation.
//
// It does NOT converge at rho/Gamma = 1000, and the interesting regimes for a limiter are
// 1000 and 1e6. What is known about that failure:
//
//   * It is not the linear solve. Tightening it from 18 to 1007 iterations changed nothing.
//   * It is not the Peclet number as such. rho/Gamma = 1 converges at mu = 1e-2 and FAILS at
//     mu = 1e-3 -- the same ratio and the same physics at a different absolute scale -- so it
//     looks like conditioning or scaling rather than the scheme.
//   * Any comparison run there is void. The first attempt reported first-order upwind
//     violating the bounds by 3.9, which cannot happen: donor-cell is unconditionally bounded
//     on a problem with a maximum principle. That an unconditionally-bounded arm "failed" is
//     what exposed the runs as unconverged, before newton_converged confirmed it.
//
// So this delivers the case, not yet the measurement it was built to make.

namespace cvfem_smith_hutton {

    // Smith and Hutton's alpha. Steeper than it looks: tanh(10) = 1 - 4.1e-9, so the inlet
    // profile is a step to within single precision and the "near-discontinuity" is, on any
    // mesh this spike will run, a discontinuity.
    template <typename T>
    constexpr T alpha() {
        return T(10);
    }

    // phi on the outer boundary, and the lower of the two bounds the solution must respect.
    template <typename T>
    inline T phi_far() {
        return T(1) - std::tanh(alpha<T>());
    }

    // The inlet profile, in the paper's coordinates (X in [-1, 0]).
    template <typename T>
    inline T phi_inlet(const T X) {
        return T(1) + std::tanh(alpha<T>() * (T(2) * X + T(1)));
    }

    // THE ANALYTIC REFERENCE, in the pure-convection limit rho/Gamma -> infinity.
    //
    // The streamlines of this field are closed curves symmetric about x = 0, so with no
    // diffusion every streamline carries its inlet value to the outlet station mirrored in x.
    // The outlet profile is therefore the inlet profile reflected:
    //
    //     phi(X, 0)|outlet = phi_inlet(-X) = 1 + tanh(alpha (1 - 2X)).
    //
    // That is an exact statement about the limit, not a tabulated approximation, which is what
    // makes it usable as a gate. At finite rho/Gamma the true profile is smeared relative to
    // it, so the comparison is one-sided: a scheme may fall short of the reference front and
    // must never exceed the bounds below.
    template <typename T>
    inline T phi_outlet_pure_convection(const T X) {
        return T(1) + std::tanh(alpha<T>() * (T(1) - T(2) * X));
    }

    // The bounds phi must respect ANYWHERE, for any rho/Gamma. Both boundary data and the
    // transported values lie in [1 - tanh(alpha), 1 + tanh(alpha)]; a maximum principle holds
    // for this equation, so a value outside is a scheme overshoot and nothing else. This is
    // the crisp boundedness statement the smooth cases cannot supply.
    template <typename T>
    inline void phi_bounds(T &lo, T &hi) {
        lo = T(1) - std::tanh(alpha<T>());
        hi = T(1) + std::tanh(alpha<T>());
    }

    // The prescribed velocity, given mesh coordinates on [0, 2] x [0, 1].
    template <typename T>
    inline void velocity(const T x_mesh, const T y, T &u, T &v) {
        const T X = x_mesh - T(1);
        u         = T(2) * y * (T(1) - X * X);
        v         = T(-2) * X * (T(1) - y * y);
    }

    // phi's Dirichlet value at a boundary node, and whether it has one at all. The outlet --
    // y = 0 with x > 0 -- is the one place it does not: that boundary is zero-gradient, which
    // is this discretisation's natural condition and is imposed by constraining nothing.
    // Lx and Ly are the MESH extents, and both are needed: the first version took only Ly and
    // therefore returned the far-field value for every node with y > 0, interior included. It
    // constrained 288 of 306 nodes and left 18 unknowns, which the constraint count made
    // obvious the moment it was printed -- a case that prescribes almost its whole solution
    // cannot test a scheme.
    template <typename T>
    inline bool phi_dirichlet(const T x_mesh, const T y, const T Lx, const T Ly, T &value) {
        const T tol = T(1e-12);
        const T X   = x_mesh - T(1);
        if (y <= tol * Ly) {                   // y = 0: inlet for X < 0, outlet for X > 0
            if (X < T(0)) {
                value = phi_inlet(X);
                return true;
            }
            return false;                      // outlet, left free = zero gradient
        }
        // The rest of the OUTER boundary carries the far-field value. Interior nodes carry
        // nothing: they are what the solve is for.
        const bool on_side = x_mesh <= tol * Lx || x_mesh >= Lx - tol * Lx;
        const bool on_top  = y >= Ly - tol * Ly;
        if (on_side || on_top) {
            value = phi_far<T>();
            return true;
        }
        return false;
    }

}  // namespace cvfem_smith_hutton

#endif  // CVFEM_NS_SMITH_HUTTON_CASE_HPP
