# Two-Dimensional Finite-Strain Kelvin-Voigt Creep

The unit plane-strain strip is constrained laterally and loaded by a cosine-smoothed
axial nominal traction. For `F = diag(lambda, 1)`, the generated material reduces to

`(6 mu + lambda_Lame) (lambda - 1) + (eta_s + eta_b) dot(lambda) / lambda = p(t)`.

`oracle.py` integrates this scalar equation independently with an adaptive DOP853
solver. The three nested time steps check the second-order Newmark/trapezoidal
history, while exported internal reactions are compared with the applied traction.
