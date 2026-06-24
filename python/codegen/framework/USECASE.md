# Two-phase flow

Reference: [`twophaseflow/main.tex`](twophaseflow/main.tex)

## Scope

Generate and solve the fully coupled pressure formulation for immiscible water and
carbon-dioxide flow without gravity. The primary unknowns are `p_w` and `p_c`.
Time integration uses implicit Euler. The generated implementation must provide
the nonlinear residual and Jacobian action required by a damped Newton solver.

## M1: Symbolic implicit-Euler model

### M1.1 Constitutive model API

- Implement SymPy expressions for capillary pressure, water and carbon-dioxide
  saturation, phase densities, relative permeabilities, and phase mobilities.
- Represent physical parameters with named symbols and group them in a stable,
  documented material-parameter API.
- Keep constitutive expressions differentiable with respect to `p_w` and `p_c`.
- Define the valid pressure and saturation domain and reject invalid parameter
  combinations before code generation.

Acceptance:

- Unit tests compare every constitutive expression and its pressure derivatives
  against finite differences at representative admissible states.
- Saturations sum to one and remain in `[0, 1]` for all test states.

### M1.2 Implicit-Euler residual equations

- Define current pressures `p_w`, `p_c` and previous-step pressures
  `p_w_old`, `p_c_old`.
- Express the phase accumulation terms as
  `(m_alpha(p_w, p_c) - m_alpha(p_w_old, p_c_old)) / dt`.
- Express Darcy fluxes using the intrinsic-permeability tensor and current-step
  pressure gradients.
- Construct the weak residual for both phases, including test values and test
  gradients, with natural no-flow boundaries omitted from the volume form.
- Keep the two residual components as one coupled symbolic residual system.

Acceptance:

- Zero pressure increment with spatially constant pressure produces zero volume
  residual.
- Symbolic tests verify the accumulation, diffusion, and coupling terms against
  a direct Python reference implementation.

### M1.3 Linearization and merit function

- Differentiate the residual system in an arbitrary coupled trial direction to
  obtain a matrix-free Jacobian action.
- Generate the block derivatives `ww`, `wc`, `cw`, and `cc`.
- Define a residual-norm merit function for Newton line search because the
  pressure formulation is residual-based rather than energy-based.
- Track FLOPs, temporaries, and expensive operations for residual and Jacobian
  action expressions.

Acceptance:

- Jacobian actions match centered finite differences of the residual.
- Cross-phase blocks are nonzero for coupled constitutive states.

## M2: Residual-driven code-generation API

### M2.1 Coupled residual front end

- Add a high-level API for registering multiple residual equations, unknown
  fields, previous-step fields, test functions, and material parameters.
- Support scalar value and gradient dependencies for every field.
- Preserve field/block identity through expression graphs and generated outputs.
- Keep the existing energy-based API unchanged.

Acceptance:

- A minimal two-field diffusion problem generates residual and Jacobian-action
  kernels without material-specific generator code.
- Invalid field, test, or block dimensions fail with clear diagnostics.

### M2.2 Residual and Jacobian-action lowering

- Lower coupled weak residuals into the existing local micro-kernel structure:
  evaluate trial quantities, evaluate constitutive terms, then contract with
  specialized test quantities.
- Place the quadrature loop inside the local micro-kernel.
- Hoist previous-step and parameter invariants to the widest valid scope.
- Generate branch-free SoA code suitable for SIMD execution.
- Retain tensor-product sum factorization and simplex specialization.

Acceptance:

- Generated local kernels contain no model-specific hard-coded expressions.
- Compiler reports confirm vectorization of the intended lane loops.
- TRI3, TET4, QUAD4, and HEX8 residual and Jacobian-action kernels compile for
  `float` and `double`.

### M2.3 Kernel diagnostics

- Generate `KernelDiagnostics` for residual and every Jacobian block/action.
- Include accumulation and flux FLOPs, transcendental operations, bytes moved,
  vector size, and arithmetic intensity.
- Expose diagnostics through the generated C ABI.

Acceptance:

- Diagnostic totals are stable under regeneration and agree with expression
  graph counts.

## M3: SFEM mesh kernels and operator integration

### M3.1 Mesh-level residual and Jacobian action

- Generate gather/local/scatter mesh kernels for the two pressure fields.
- Support affine geometry with precomputed SoA adjugate/determinant streams.
- Support isoparametric geometry with quadrature-point geometry computed from
  coordinates before the local micro-kernel call.
- Keep element, quadrature, shape, and vector sizes compile-time generated;
  mesh implementations retain only `scalar_t` as a template parameter.
- Expose explicit `double` and `float` C ABI entry points.

Acceptance:

- Mesh kernels match a direct Python element assembly on reference and deformed
  meshes.
- Conservation tests show equal assembled and element-integrated phase mass
  residuals.

### M3.2 SFEM nonlinear operator

- Add an SFEM operator wrapper exposing residual evaluation and Jacobian action.
- Store current and previous pressure states without copies in local kernels.
- Apply time-dependent Dirichlet conditions for the injection and reservoir
  boundaries and preserve natural no-flow boundaries.
- Integrate the generated operator through `make_op`.

Acceptance:

- Operator-level residual and Jacobian action match direct generated-kernel
  calls.
- Boundary values remain exact after residual and linear-solve updates.

## M4: Transient damped-Newton driver

### M4.1 Problem setup and time integration

- Create an SFEM/smesh driver that builds or reads a porous-domain mesh.
- Initialize `p_w = 15 MPa` and `p_c = 15.1 MPa`.
- Implement the left-boundary carbon-dioxide pressure ramp to `20 MPa`, fixed
  right-boundary pressures, and no-flow natural boundaries.
- Advance with implicit Euler while preserving the accepted state as
  `p_old`.

Acceptance:

- A zero-ramp run preserves the uniform initial state to solver tolerance.
- Restarting from a saved time step reproduces the uninterrupted solution.

### M4.2 Damped Newton and linear solve

- Solve each time step with damped Newton using the generated residual and
  Jacobian action.
- Use CG only when the generated Jacobian is demonstrably symmetric positive
  definite; otherwise use an SFEM nonsymmetric Krylov solver.
- Implement residual-based backtracking, admissibility checks, and configurable
  nonlinear/linear tolerances.
- Reject failed steps and retry with a reduced time step.

Acceptance:

- Every accepted Newton update reduces the merit function.
- Residual, iteration counts, damping factor, and rejected-step count are
  reported for every time step.

### M4.3 Verification and performance benchmark

- Compare coarse-mesh results against a direct Python implementation.
- Verify temporal convergence by halving `dt`.
- Track phase mass balance and boundary fluxes at every accepted step.
- Report residual/Jacobian throughput, DOF/s, FLOP/s, arithmetic intensity, and
  total solve time using the generated diagnostics API.
- Add a script that generates kernels, compiles the driver, runs verification,
  and executes the benchmark.

Acceptance:

- Python and generated solutions agree within the selected discretization and
  nonlinear tolerances.
- Mass-balance error and temporal-convergence results are included in the run
  summary.
