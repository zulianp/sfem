# Three-Dimensional Finite-Strain Kelvin-Voigt Creep

The axial mode uses `F = diag(lambda, 1, 1)` and the scalar balance

`(6 mu + lambda_Lame)(lambda - 1) + (4 eta_s / 3 + eta_b) dot(lambda) / lambda = p(t)`.

The additional simple-shear mode satisfies `4 mu gamma + eta_s dot(gamma) = p(t)`.
Both are integrated independently with DOP853 and compared against complete
histories from `TET4` and `HEX8` runs at three nested time steps.
