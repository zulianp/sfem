# Three-Dimensional Linear Kelvin-Voigt Damped Mode

A laterally constrained unit bar is fixed axially at both ends and initialized
from file-backed `sin(pi x / L)` displacement, velocity, and consistent
acceleration fields. The axial modulus is `M = K + 2 k / 3` and the viscous
modulus is `c = 2 eta / 3`.

For the first mode, `omega_0^2 = M (pi/L)^2 / rho`,
`delta = c (pi/L)^2 / (2 rho)`, and the underdamped amplitude is the closed-form
expression implemented in `oracle.py`. The verifier projects every output state
onto the mode and records all orthogonal content as an off-mode diagnostic.
