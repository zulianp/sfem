#pragma once

// The analytic channel case: which flow, its exact solution, and the plane test that
// decides where the boundary conditions go.
//
// This is case setup, not operator internals, and both the standalone driver and any
// frontend driver need it -- to impose the boundary values and to verify against. It is
// kept deliberately free of everything else in this directory: no MeshData, no kernels,
// no file-scope scalar_t, and none of the sixteen names the solver core and the
// benchmark layouts fight over. So it is safe to include next to either family, or next
// to cvfem_hex8_ns_op.hpp alone.

#include <cmath>
#include <string>

namespace cvfem_case {

    enum class FlowCase { Poiseuille, Couette };

    inline bool parse_case(const std::string &name, FlowCase &out) {
        if (name == "poiseuille") {
            out = FlowCase::Poiseuille;
            return true;
        }
        if (name == "couette" || name == "coutte") {
            out = FlowCase::Couette;
            return true;
        }
        return false;
    }

    // Relative tolerance, so it does not become meaningless on a large domain and does
    // not admit interior nodes on a small one. Must match the boundary sub-control-
    // surface test in cvfem_hex8_boundary_scs.hpp: the two decide the same thing, and if
    // they disagree a node gets a closed control volume without a boundary condition,
    // or the reverse.
    template <typename T>
    inline bool on_plane(const T c, const T value, const T L) {
        const T tol = T(1e-8) * std::max(L, T(1));
        return std::fabs(c - value) <= tol;
    }

    // Fully developed flow between plates at y = 0 and y = Ly. Couette is driven by the
    // lid, Poiseuille by the pressure gradient G = 8 mu U / Ly^2 that produces peak
    // velocity U.
    template <typename T>
    inline void exact_state(const FlowCase flow,
                            const T        mu,
                            const T        U,
                            const T        Lx,
                            const T        Ly,
                            const T        x,
                            const T        y,
                            const T /*z*/,
                            T &ux,
                            T &uy,
                            T &uz,
                            T &p) {
        uy = T(0);
        uz = T(0);
        if (flow == FlowCase::Couette) {
            ux = U * (y / Ly);
            p  = T(0);
            return;
        }
        const T G = T(8) * mu * U / (Ly * Ly);
        ux        = T(4) * U * y * (Ly - y) / (Ly * Ly);
        p         = G * (T(0.5) * Lx - x);
    }

}  // namespace cvfem_case
