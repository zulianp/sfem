#pragma once

// Declarative description of a Prony-series viscoelastic test case.
//
// The whole scenario -- mesh, material, time integration, solver, boundary conditions and
// the torsion schedule including its release -- is read from a single YAML document so that
// a new experiment is a new file rather than a new build.

#include <iosfwd>
#include <string>
#include <vector>

#include "sfem_base.hpp"

namespace prony {

    /// One term of the relaxation function G(t) = g_inf + sum_i g_i exp(-t / tau_i).
    struct PronyTerm {
        real_t g{0};
        real_t tau{0};
    };

    struct Material {
        real_t                 C10{1.0};
        real_t                 C01{0.5};
        real_t                 K{100.0};
        std::vector<PronyTerm> prony;

        // Williams-Landel-Ferry time-temperature superposition. Off by default: with it on,
        // tau_eff = tau / a_T(T), so the same series describes a different rate of relaxation.
        bool   wlf_enabled{false};
        real_t wlf_C1{16.6253};
        real_t wlf_C2{47.4781};
        real_t wlf_T_ref{-54.29};
        real_t temperature{20.0};
    };

    struct Time {
        real_t dt{0.05};
        real_t t_end{20.0};
    };

    enum class Integrator {
        /// Equilibrium at each step, with no inertia at all. The regime a relaxation
        /// experiment lives in, and the default.
        QuasiStatic,
        /// sfem::BDF2InertiaPotential, as in drivers/mech/hyperelasticity_bdf2. Second order
        /// and L-stable, so it damps high frequencies numerically -- convenient for getting
        /// past the step change at release, misleading if the ring-down is what you came for.
        BDF2,
        /// sfem::NewmarkInertiaPotential, the scheme the Mooney-Rivlin tests in
        /// frontend/tests use. At the default beta = 1/4, gamma = 1/2 it is the trapezoidal
        /// rule: second order and non-dissipative, so the ring-down after the release is the
        /// material's own damping and not the integrator's.
        Newmark
    };

    struct Dynamics {
        Integrator integrator{Integrator::QuasiStatic};
        real_t     density{1.0};
        /// Newmark only. (1/4, 1/2) is constant average acceleration: unconditionally stable
        /// and energy conserving. gamma > 1/2 introduces numerical damping.
        real_t     beta{0.25};
        real_t     gamma{0.5};
    };

    enum class ReleaseMode {
        /// The torsion constraint is removed at release_time: the twisted face becomes
        /// traction free and the body recovers.
        Free,
        /// The constraint is never removed. The control case: pure stress relaxation at
        /// fixed twist, against which a release run is read.
        Hold
    };

    enum class TwistProfile {
        /// Linear ramp to `angle` over `ramp_time`, then held. A relaxation test.
        Ramp,
        /// angle(t) = angle * sin(2 pi t / period). A dynamic-mechanical test: the lag between
        /// the twist and the torque it produces is the thing being measured, and it shows up
        /// directly as a hysteresis loop in the torque-angle plane. A ramp shows the same
        /// physics but only as a softening against a matched elastic run; a cycle shows it as
        /// a phase angle that can be compared with the Prony series analytically.
        Cyclic
    };

    enum class TorsionControl {
        /// The twist is prescribed and the torque is whatever it takes. A relaxation test: the
        /// torque decays while the shape stays put, so the *structure* does not visibly adapt.
        Angle,
        /// The torque is prescribed and the twist is solved for. A creep test, and the mode to
        /// use to watch the structure itself keep turning after the load is applied: for a
        /// linear viscoelastic solid the twist grows from its instantaneous value towards
        /// 1/g_inf times it.
        Torque
    };

    struct Torsion {
        bool           enabled{false};
        TorsionControl control{TorsionControl::Angle};
        /// Torque control only: the target moment about the rotation axis, reached over
        /// `ramp_time` and then held.
        real_t         torque{0};
        /// Torque control only: the outer iteration that solves torque(angle) = target.
        int            torque_max_it{20};
        /// Relative tolerance on the torque, against the target.
        real_t         torque_tol{1e-6};
        TwistProfile profile{TwistProfile::Ramp};
        /// Cyclic only: the period of one cycle, in seconds.
        real_t       period{0};
        std::string sideset;
        /// 0/1/2 select rotation in the (y,z), (x,z) and (x,y) planes respectively.
        int         axis{0};
        real_t      center[3]{0, 0, 0};
        /// Total twist reached at the end of the ramp, in radians.
        real_t      angle{0};
        /// Linear ramp 0 -> angle over this many seconds. Zero applies the twist at once.
        real_t      ramp_time{0};
        bool        has_release{false};
        real_t      release_time{0};
        ReleaseMode release_mode{ReleaseMode::Free};
        bool        verbose{false};
    };

    struct Solver {
        int    nl_max_it{30};
        real_t nl_tol{1e-8};
        real_t nl_alpha{1.0};
        /// Backtracking on ||residual||. The viscoelastic operator provides no energy, so an
        /// energy line search of the kind hyperelasticity_bdf2 uses is not available here.
        bool   line_search{true};
        /// Newton is abandoned when the residual grows by this factor over its initial value.
        real_t divergence_factor{1e4};

        std::string linear_type{"cg"};  // cg | bcgs
        real_t      lin_rtol{1e-8};
        real_t      lin_atol{1e-14};
        int         lin_max_it{2000};
        bool        preconditioner{true};
        bool        lin_verbose{false};
    };

    struct Output {
        std::string path{"output"};
        int         export_freq{1};
        bool        has_control_point{false};
        real_t      control_point[3]{0, 0, 0};
        /// Per-step scalars: twist angle, reaction torque, control-point displacement.
        std::string history_csv{"history.csv"};
    };

    struct Case {
        std::string mesh;
        /// Optional path to a stand-alone SFEM dirichlet YAML.
        std::string dirichlet_file;
        /// The verbatim case document, kept when the case carries its own top-level
        /// `dirichlet_conditions:` block. SFEM's parser looks that key up at the root and
        /// ignores everything around it, so the whole file can be handed over as-is and the
        /// dirichlet schema stays SFEM's rather than being re-implemented here.
        std::string dirichlet_yaml;

        Material material;
        Time     time;
        Dynamics dynamics;
        Torsion  torsion;
        Solver   solver;
        Output   output;

        bool verbose{false};

        void print(std::ostream &os) const;
    };

    /// Reads `path` and fills `c`. Paths inside the document are resolved relative to the
    /// document's own directory unless they are absolute, so a case file travels with the
    /// data it refers to. Returns SFEM_SUCCESS or SFEM_FAILURE.
    int read_case(const std::string &path, Case &c);

}  // namespace prony
