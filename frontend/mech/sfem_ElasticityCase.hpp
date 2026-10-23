#ifndef SFEM_ELASTICITY_CASE_HPP
#define SFEM_ELASTICITY_CASE_HPP

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "sfem_base.hpp"

#ifdef SFEM_ENABLE_RYAML
#include <ryml.hpp>
#include <ryml_std.hpp>

namespace sfem {

    /// An elasticity problem as one YAML document: mesh, material, loads, boundary conditions,
    /// regime, time, solver and output. Every key is checked against the schema, so a misspelt
    /// one is an error rather than a default silently taken.
    ///
    /// ```yaml
    /// execution_space: host                  # host | device
    /// mesh:
    ///   path: mesh                           # or box: {n: [8, 8, 8], min: [0, 0, 0], max: [1, 1, 1]}
    ///   reorder: none                        # none | hilbert3 | morton3, before the sidesets are read
    ///   promote_to_p2: false                 # TET4 to TET10
    ///   refine: 0                            # semi-structured refinement level
    ///   basis: micro                         # refined: trilinear micro elements, or high_order (one
    ///                                        # Lagrange element of the level per macro element)
    ///   packed: true                         # the packed element layout
    ///   renumber: true                       # packing renumbers the nodes; false keeps the numbering on
    ///                                        # disk, for exports compared node by node with it
    ///   sidesets:
    ///     - {name: left, plane: {axis: x, at: min}}   # at: min | max | a coordinate; tol relative
    ///     - {name: right, file: mesh/sidesets/right}
    /// material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}   # create_op_from_yaml
    /// loads:
    ///   body_force: {density: 1, g: [0, 0, -9.81]}
    ///   neumann: [...]                       # NeumannConditions YAML; format: name, name: right
    ///   profile: {type: linear_ramp, ...}    # scales every Neumann condition, on its own profile
    /// dirichlet_conditions: [...]            # DirichletConditions YAML; format: name, name: left
    ///                                        # either list may be {file: dirichlet.yaml} instead
    /// constraints:                           # host only
    ///   - {type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.5,
    ///      profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1},
    ///      release: 2}                       # angle * profile(t); removed at the first step ending at release
    ///   - {type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5],
    ///      control: {torque: 1e-3, tol: 1e-6, max_it: 20}}   # the angle at which the torque is
    ///                                        # torque * profile(t), instead of an angle
    /// fields:                                # Op::set_field on the material
    ///   - {name: active_strain, file: Fa.raw, values_per_element: 9}
    ///   - {name: active_strain, inclusion: {a11: 0.5, radius: 0.25, center: [0.5, 0.5, 0.5]}, profile: {...}}
    /// initial_conditions:
    ///   displacement: {components: [u.0.raw, u.1.raw, u.2.raw]}   # or file: interleaved
    ///   velocity: {file: v.raw}              # dynamic, Newmark or BDF2
    ///   acceleration: {file: a.raw}          # dynamic, Newmark; otherwise it starts from a = 0
    /// obstacle:                              # regime dynamic, scheme newmark; host or device
    ///   backend: node_sdf                    # nodes against the SDF's B-spline, MPRGP in frames;
    ///                                        # ssmgc: the shifted-penalty multigrid (refined mesh,
    ///                                        # micro basis, linear material; host or device)
    ///   nodes: skin                          # or a named sideset
    ///   direction: z                         # the body approaches along -direction
    ///   gap: nodal                           # ssmgc: nodal (the SDF at each node) or dual_weighted
    ///                                        # (over each node's surface patch, dual basis)
    ///   curvature: true                      # ssmgc: mu_i C_i in the tangent (nodal gap; default
    ///                                        # with it); node_sdf always adds it
    ///   shape: {type: hemisphere, plane: 0, radius: 0.25, fillet: 0.125, grid_cells: 128}
    ///                                        # plane | slope (slope, slope_axis) | staircase
    ///                                        # (step_width, step_drop) | hemisphere (radius, center)
    ///   sdf: {file: obstacle}                # an SDF grid instead of a shape
    ///   qp: {max_it: 20000, rtol: 1e-10}
    ///   ssmgc: {penalty_param: 10, coarse_op_type: BSR}    # SSMGC's own keys, over the step's:
    ///                                        # atol gap/10, rtol newton.rtol, max_it 300, full precision,
    ///                                        # enable_nl_obstacle 0 (Newton re-linearises the obstacle)
    /// regime: quasi_static                   # steady | quasi_static | dynamic
    /// time: {dt: 0.1, t_end: 1, max_cutbacks: 4}      # or steps instead of t_end; cutbacks
    ///                                        # halve a failed step, 0 and the only choice with bdf2
    /// dynamics: {scheme: newmark, density: 1, beta: 0.25, gamma: 0.5}  # newmark | bdf2 | backward_euler
    ///   mass: lumped                         # or consistent (GeneratedVectorMass), Newmark and BDF2
    /// solver:
    ///   newton: {max_it: 30, atol: 1e-9, rtol: 1e-8, divergence_factor: 1e4, merit: auto,
    ///            line_search: {enabled: true, steps: [1, 0.5, 0.25], c1: 1e-4, batched: true}}
    ///   linear: {type: cg, format: BSR, preconditioner: jacobi, rtol: 1e-3, atol: 1e-14, max_it: 2000}
    /// output:
    ///   path: output
    ///   export_freq: 1
    ///   history_csv: history.csv
    ///   fields: [material_reaction]          # exported beside disp (and velocity, acceleration);
    ///                                        # gap and contact_force with an obstacle
    ///   quantities: true                     # path/quantities.yaml: objective and reactions
    ///   probes: [{type: control_point, at: [1, 0.5, 0.5], rates: true},
    ///            {type: torque, constraint: 0},       # angle, released, torque of constraints[0]
    ///            {type: contact}]                     # the obstacle's force and whether it touched
    /// ```
    ///
    /// Relative paths are read against the case file's directory.
    class ElasticityCase {
    public:
        /// Reads the file, applies `overrides` ("solver.newton.max_it=50", "mesh.box.n=[4,4,4]"),
        /// and checks the keys. Null, with the reason on stderr, when any of that fails.
        static std::unique_ptr<ElasticityCase> read(const std::string &path, const std::vector<std::string> &overrides = {});

        /// The same for a document in memory, its relative paths read against `base_dir`.
        static std::unique_ptr<ElasticityCase> parse(const std::string              &yaml,
                                                     const std::string              &base_dir,
                                                     const std::vector<std::string> &overrides = {});

        ryml::ConstNodeRef root() const;
        const std::string &base_dir() const;

        /// The case as resolved: the file with its overrides applied.
        std::string emit() const;

        /// `path` relative to the case's directory, or as given when absolute.
        std::string resolve(const std::string &path) const;

    private:
        ElasticityCase() = default;

        ryml::Tree            tree_;
        std::list<ryml::Tree> overrides_;
        std::string           base_dir_;
    };

}  // namespace sfem

#endif  // SFEM_ENABLE_RYAML
#endif  // SFEM_ELASTICITY_CASE_HPP
