#include "prony_case.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <sstream>

#include "sfem_defs.hpp"
#include "sfem_macros.hpp"

#include <ryml.hpp>
#include <ryml_std.hpp>

namespace prony {

    namespace {

        std::string parent_directory(const std::string &path) {
            const auto pos = path.find_last_of('/');
            if (pos == std::string::npos) {
                return ".";
            }
            return path.substr(0, pos);
        }

        std::string resolve(const std::string &base_dir, const std::string &path) {
            if (path.empty() || path[0] == '/') {
                return path;
            }
            return base_dir + "/" + path;
        }

        bool has(const ryml::ConstNodeRef &node, const char *key) {
            if (node.invalid() || !node.readable() || !node.is_map()) {
                return false;
            }
            const auto child = node.find_child(ryml::to_csubstr(key));
            return !child.invalid() && child.readable();
        }

        template <typename T>
        void read(const ryml::ConstNodeRef &node, const char *key, T &value) {
            if (has(node, key)) {
                node[ryml::to_csubstr(key)] >> value;
            }
        }

        void read_bool(const ryml::ConstNodeRef &node, const char *key, bool &value) {
            if (has(node, key)) {
                std::string s;
                node[ryml::to_csubstr(key)] >> s;
                value = (s == "true" || s == "True" || s == "yes" || s == "on" || s == "1");
            }
        }

        int read_vec3(const ryml::ConstNodeRef &node, const char *key, real_t v[3]) {
            if (!has(node, key)) {
                return SFEM_SUCCESS;
            }

            auto seq = node[ryml::to_csubstr(key)];
            if (!seq.is_seq() || seq.num_children() != 3) {
                fprintf(stderr, "[prony] '%s' must be a sequence of three numbers\n", key);
                return SFEM_FAILURE;
            }

            for (int d = 0; d < 3; ++d) {
                seq[d] >> v[d];
            }
            return SFEM_SUCCESS;
        }

        int read_axis(const ryml::ConstNodeRef &node, int &axis) {
            if (!has(node, "axis")) {
                return SFEM_SUCCESS;
            }

            std::string s;
            node["axis"] >> s;
            if (s == "x" || s == "X" || s == "0") {
                axis = 0;
            } else if (s == "y" || s == "Y" || s == "1") {
                axis = 1;
            } else if (s == "z" || s == "Z" || s == "2") {
                axis = 2;
            } else {
                fprintf(stderr, "[prony] torsion.axis must be one of x, y, z (got '%s')\n", s.c_str());
                return SFEM_FAILURE;
            }
            return SFEM_SUCCESS;
        }

        int read_material(const ryml::ConstNodeRef &node, Material &m) {
            read(node, "C10", m.C10);
            read(node, "C01", m.C01);
            read(node, "K", m.K);

            if (has(node, "prony")) {
                auto terms = node["prony"];
                if (!terms.is_seq()) {
                    fprintf(stderr, "[prony] material.prony must be a sequence of {g, tau} maps\n");
                    return SFEM_FAILURE;
                }

                for (auto term : terms.children()) {
                    PronyTerm t;
                    if (!has(term, "g") || !has(term, "tau")) {
                        fprintf(stderr, "[prony] every material.prony entry needs both 'g' and 'tau'\n");
                        return SFEM_FAILURE;
                    }
                    term["g"] >> t.g;
                    term["tau"] >> t.tau;
                    if (t.tau <= 0) {
                        fprintf(stderr, "[prony] relaxation time tau must be positive (got %g)\n", (double)t.tau);
                        return SFEM_FAILURE;
                    }
                    m.prony.push_back(t);
                }
            }

            real_t sum_g = 0;
            for (const auto &t : m.prony) {
                sum_g += t.g;
            }

            // g_inf = 1 - sum(g_i) is the long-term modulus ratio. A non-positive g_inf is a
            // solid that relaxes to nothing, which the incremental form cannot represent.
            if (sum_g >= 1) {
                fprintf(stderr,
                        "[prony] sum(g_i) = %g must be < 1 so that the long-term modulus ratio "
                        "g_inf = 1 - sum(g_i) stays positive\n",
                        (double)sum_g);
                return SFEM_FAILURE;
            }

            if (has(node, "wlf")) {
                auto wlf = node["wlf"];
                read_bool(wlf, "enabled", m.wlf_enabled);
                read(wlf, "C1", m.wlf_C1);
                read(wlf, "C2", m.wlf_C2);
                read(wlf, "T_ref", m.wlf_T_ref);
                read(wlf, "temperature", m.temperature);
            }

            return SFEM_SUCCESS;
        }

        int read_torsion(const ryml::ConstNodeRef &node, const std::string &base_dir, Torsion &t) {
            t.enabled = true;

            if (!has(node, "sideset")) {
                fprintf(stderr, "[prony] torsion needs a 'sideset' path\n");
                return SFEM_FAILURE;
            }

            std::string sideset;
            node["sideset"] >> sideset;
            t.sideset = resolve(base_dir, sideset);

            if (read_axis(node, t.axis) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            if (read_vec3(node, "rotation_center", t.center) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            read(node, "angle", t.angle);
            read(node, "ramp_time", t.ramp_time);
            read(node, "period", t.period);
            read_bool(node, "verbose", t.verbose);

            if (has(node, "profile")) {
                std::string profile;
                node["profile"] >> profile;
                if (profile == "ramp") {
                    t.profile = TwistProfile::Ramp;
                } else if (profile == "cyclic") {
                    t.profile = TwistProfile::Cyclic;
                } else {
                    fprintf(stderr, "[prony] torsion.profile must be 'ramp' or 'cyclic' (got '%s')\n", profile.c_str());
                    return SFEM_FAILURE;
                }
            }

            read(node, "torque", t.torque);
            read(node, "torque_max_it", t.torque_max_it);
            read(node, "torque_tol", t.torque_tol);

            if (has(node, "control")) {
                std::string control;
                node["control"] >> control;
                if (control == "angle") {
                    t.control = TorsionControl::Angle;
                } else if (control == "torque") {
                    t.control = TorsionControl::Torque;
                } else {
                    fprintf(stderr, "[prony] torsion.control must be 'angle' or 'torque' (got '%s')\n", control.c_str());
                    return SFEM_FAILURE;
                }
            }

            if (t.control == TorsionControl::Torque) {
                if (t.torque == 0) {
                    fprintf(stderr, "[prony] torsion.control 'torque' needs a non-zero torque\n");
                    return SFEM_FAILURE;
                }
                if (t.profile == TwistProfile::Cyclic) {
                    fprintf(stderr, "[prony] torsion.control 'torque' does not support the cyclic profile yet\n");
                    return SFEM_FAILURE;
                }
            }

            if (t.profile == TwistProfile::Cyclic && !(t.period > 0)) {
                fprintf(stderr, "[prony] torsion.profile 'cyclic' needs a positive period\n");
                return SFEM_FAILURE;
            }

            if (has(node, "release")) {
                auto rel = node["release"];

                std::string mode = "free";
                read(rel, "mode", mode);
                if (mode == "free") {
                    t.release_mode = ReleaseMode::Free;
                } else if (mode == "hold") {
                    t.release_mode = ReleaseMode::Hold;
                } else {
                    fprintf(stderr, "[prony] torsion.release.mode must be 'free' or 'hold' (got '%s')\n", mode.c_str());
                    return SFEM_FAILURE;
                }

                if (has(rel, "time")) {
                    rel["time"] >> t.release_time;
                    t.has_release = (t.release_mode == ReleaseMode::Free);
                } else if (t.release_mode == ReleaseMode::Free) {
                    fprintf(stderr, "[prony] torsion.release.mode 'free' needs a release time\n");
                    return SFEM_FAILURE;
                }
            }

            return SFEM_SUCCESS;
        }

        int read_solver(const ryml::ConstNodeRef &node, Solver &s) {
            if (has(node, "newton")) {
                auto nl = node["newton"];
                read(nl, "max_it", s.nl_max_it);
                read(nl, "tol", s.nl_tol);
                read(nl, "alpha", s.nl_alpha);
                read(nl, "divergence_factor", s.divergence_factor);
                read_bool(nl, "line_search", s.line_search);
            }

            if (has(node, "linear")) {
                auto lin = node["linear"];
                read(lin, "type", s.linear_type);
                read(lin, "rtol", s.lin_rtol);
                read(lin, "atol", s.lin_atol);
                read(lin, "max_it", s.lin_max_it);
                read_bool(lin, "preconditioner", s.preconditioner);
                read_bool(lin, "verbose", s.lin_verbose);

                if (s.linear_type != "cg" && s.linear_type != "bcgs") {
                    fprintf(stderr, "[prony] solver.linear.type must be 'cg' or 'bcgs' (got '%s')\n", s.linear_type.c_str());
                    return SFEM_FAILURE;
                }
            }

            return SFEM_SUCCESS;
        }

        int read_output(const ryml::ConstNodeRef &node, const std::string &base_dir, Output &o) {
            if (has(node, "path")) {
                std::string path;
                node["path"] >> path;
                o.path = resolve(base_dir, path);
            }

            read(node, "export_freq", o.export_freq);
            o.export_freq = o.export_freq > 0 ? o.export_freq : 1;

            if (has(node, "control_point")) {
                if (read_vec3(node, "control_point", o.control_point) != SFEM_SUCCESS) {
                    return SFEM_FAILURE;
                }
                o.has_control_point = true;
            }

            read(node, "history_csv", o.history_csv);
            return SFEM_SUCCESS;
        }

    }  // namespace

    int read_case(const std::string &path, Case &c) {
        std::ifstream is(path);
        if (!is.good()) {
            fprintf(stderr, "[prony] unable to read case file %s\n", path.c_str());
            return SFEM_FAILURE;
        }

        std::ostringstream contents;
        contents << is.rdbuf();
        std::string text = contents.str();
        is.close();

        const std::string base_dir = parent_directory(path);

        // parse_in_place aliases `text`, and the tree is read below, so `text` must outlive it.
        ryml::Tree tree = ryml::parse_in_place(ryml::to_substr(text));
        const auto root = tree.crootref();

        if (!has(root, "mesh")) {
            fprintf(stderr, "[prony] case file %s has no 'mesh' entry\n", path.c_str());
            return SFEM_FAILURE;
        }

        std::string mesh;
        root["mesh"] >> mesh;
        c.mesh = resolve(base_dir, mesh);

        read_bool(root, "verbose", c.verbose);

        if (has(root, "material") && read_material(root["material"], c.material) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        if (has(root, "time")) {
            auto t = root["time"];
            read(t, "dt", c.time.dt);
            read(t, "t_end", c.time.t_end);
        }

        if (c.time.dt <= 0 || c.time.t_end <= 0) {
            fprintf(stderr, "[prony] time.dt and time.t_end must both be positive\n");
            return SFEM_FAILURE;
        }

        if (has(root, "dynamics")) {
            auto d = root["dynamics"];
            if (has(d, "type")) {
                std::string type;
                d["type"] >> type;
                if (type == "bdf2") {
                    c.dynamics.integrator = Integrator::BDF2;
                } else if (type == "newmark") {
                    c.dynamics.integrator = Integrator::Newmark;
                } else if (type == "quasi_static") {
                    c.dynamics.integrator = Integrator::QuasiStatic;
                } else {
                    fprintf(stderr,
                            "[prony] dynamics.type must be 'quasi_static', 'bdf2' or 'newmark' (got '%s')\n",
                            type.c_str());
                    return SFEM_FAILURE;
                }
            }
            read(d, "density", c.dynamics.density);
            read(d, "beta", c.dynamics.beta);
            read(d, "gamma", c.dynamics.gamma);

            if (c.dynamics.integrator == Integrator::Newmark) {
                // Newmark is unconditionally stable only for beta >= gamma/2 >= 1/4. Outside
                // that region it is merely conditionally stable, and on a problem this stiff it
                // does not survive one step -- which is a confusing way to find out, because the
                // usual instinct on seeing a ringing solution is to raise gamma to add damping,
                // and raising gamma alone is exactly what leaves the region. Measured here:
                // (0.25, 0.5) runs, (0.25, 0.6) and (0.25, 0.7) both diverge immediately, and
                // the standard damped pairing beta = (1+gamma)^2/4 runs cleanly at both.
                const real_t beta  = c.dynamics.beta;
                const real_t gamma = c.dynamics.gamma;

                if (!(beta > 0)) {
                    fprintf(stderr, "[prony] dynamics.beta must be positive (got %g)\n", (double)beta);
                    return SFEM_FAILURE;
                }

                if (gamma < real_t(0.5)) {
                    fprintf(stderr,
                            "[prony] dynamics.gamma must be at least 0.5 (got %g); below that Newmark "
                            "injects energy instead of dissipating it\n",
                            (double)gamma);
                    return SFEM_FAILURE;
                }

                if (beta < gamma / 2) {
                    fprintf(stderr,
                            "[prony] Newmark is unconditionally stable only for beta >= gamma/2 >= 1/4, "
                            "but beta = %g and gamma = %g. Use beta = %g -- that is (1+gamma)^2/4, the "
                            "standard damped pairing -- or leave gamma at 0.5.\n",
                            (double)beta,
                            (double)gamma,
                            (double)((1 + gamma) * (1 + gamma) / 4));
                    return SFEM_FAILURE;
                }
            }
        }

        if (has(root, "torsion") && read_torsion(root["torsion"], base_dir, c.torsion) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        if (has(root, "solver") && read_solver(root["solver"], c.solver) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        if (has(root, "output") && read_output(root["output"], base_dir, c.output) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        c.output.history_csv = resolve(c.output.path, c.output.history_csv);

        if (has(root, "dirichlet_file")) {
            std::string dirichlet;
            root["dirichlet_file"] >> dirichlet;
            c.dirichlet_file = resolve(base_dir, dirichlet);
        }

        if (has(root, "dirichlet_conditions")) {
            if (!c.dirichlet_file.empty()) {
                fprintf(stderr, "[prony] give either 'dirichlet_file' or an inline 'dirichlet_conditions' block, not both\n");
                return SFEM_FAILURE;
            }
            c.dirichlet_yaml = text;
        }

        return SFEM_SUCCESS;
    }

    void Case::print(std::ostream &os) const {
        os << "prony::Case\n";
        os << "  mesh: " << mesh << "\n";
        os << "  material: C10=" << material.C10 << " C01=" << material.C01 << " K=" << material.K << "\n";

        real_t sum_g = 0;
        for (size_t i = 0; i < material.prony.size(); ++i) {
            os << "    prony[" << i << "]: g=" << material.prony[i].g << " tau=" << material.prony[i].tau << "\n";
            sum_g += material.prony[i].g;
        }
        os << "    g_inf: " << (1 - sum_g) << "\n";

        if (material.wlf_enabled) {
            os << "    wlf: C1=" << material.wlf_C1 << " C2=" << material.wlf_C2 << " T_ref=" << material.wlf_T_ref
               << " T=" << material.temperature << "\n";
        }

        os << "  time: dt=" << time.dt << " t_end=" << time.t_end << "\n";
        const char *integrator_name = dynamics.integrator == Integrator::BDF2      ? "bdf2"
                                      : dynamics.integrator == Integrator::Newmark ? "newmark"
                                                                                   : "quasi_static";
        os << "  dynamics: " << integrator_name << " density=" << dynamics.density;
        if (dynamics.integrator == Integrator::Newmark) {
            os << " beta=" << dynamics.beta << " gamma=" << dynamics.gamma;
        }
        os << "\n";

        if (torsion.enabled) {
            os << "  torsion: sideset=" << torsion.sideset << " axis=" << torsion.axis;
            if (torsion.control == TorsionControl::Torque) {
                os << " control=torque target_torque=" << torsion.torque;
            } else {
                os << " angle=" << torsion.angle;
            }
            if (torsion.profile == TwistProfile::Cyclic) {
                os << " profile=cyclic period=" << torsion.period << "\n";
            } else {
                os << " ramp_time=" << torsion.ramp_time << "\n";
            }
            os << "    center: [" << torsion.center[0] << ", " << torsion.center[1] << ", " << torsion.center[2] << "]\n";
            if (torsion.has_release) {
                os << "    release: free at t=" << torsion.release_time << "\n";
            } else {
                os << "    release: none (held for the whole run)\n";
            }
        }

        os << "  solver: newton(max_it=" << solver.nl_max_it << ", tol=" << solver.nl_tol
           << ", line_search=" << (solver.line_search ? "residual" : "none") << ")"
           << " linear(" << solver.linear_type << ", rtol=" << solver.lin_rtol << ", max_it=" << solver.lin_max_it << ")\n";
        os << "  output: " << output.path << " export_freq=" << output.export_freq << "\n";
        os << "  history_csv: " << output.history_csv << "\n";
    }

}  // namespace prony
