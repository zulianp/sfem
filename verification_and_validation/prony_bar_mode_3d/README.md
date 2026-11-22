# Three-Dimensional Prony-Viscoelastic Bar Mode

The bar of `linear_kv_damped_mode_3d` (lateral displacements held, both ends fixed axially)
with `GeneratedMooneyRivlinProny`, started at rest with the velocity `v0 sin(pi x/L)` and left
to ring under Newmark (`beta = 1/4`, `gamma = 1/2`, consistent mass). The strains stay near
`1e-5`, so the linearized law applies:
`sigma = K eps + M0 (g_inf eps + sum_i h_i)`, `h_i' = -h_i/tau_i + g_i eps'`, `M0 = 8/3 (C10 + C01)`.

The mode stays a sine and its amplitude obeys `rho q'' + k2 [K q + M0 (g_inf q + sum_i h_i)] = 0`:
a linear system of `2 + n` states (Wagner and Adhikari, AIAA J. 41, 2003) whose eigenvalues are
one damped pair `-delta +- i omega_d` and `n` real relaxation rates. On a uniform mesh of linear
elements the sampled sine is an exact eigenvector of the stiffness and the consistent mass, with
`k2_h = 12/h^2 (1 - cos(pi h/L))/(4 + 2 cos(pi h/L))`, so the oracle has no spatial error.

Checks (`oracle.py`, `verify.py`):

- the modal amplitude against Newmark with the material's recursion replayed on the mode, at
  every step size (the fully discrete scheme);
- against the mesh's mode exact in time (`expm`): the history, the fitted frequency and decay at
  the finest step, and the temporal rate;
- the fitted frequency and decay against the continuum's mode (`k = pi/L`), the mesh's `O(h^2)`;
- that the field stays the sine.

The spectrum (`g = 0.3, 0.2`, `tau = 0.05, 0.3` against a period of about 0.9) damps the mode
by a log decrement of 0.47 per period, about `pi tan(delta)` at its frequency.
