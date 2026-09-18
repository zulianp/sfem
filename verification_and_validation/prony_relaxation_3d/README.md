# Three-Dimensional Prony Relaxation

A homogeneous uniaxial deformation is prescribed from YAML-referenced nodal
values and held while `MooneyRivlinVisco` advances its quadrature history. With
zero bulk modulus, the finite-strain isochoric reaction is multiplied by

`G(t)/G0 = 1 - sum(g_i) + sum(g_i exp(-t/tau_i))`.

The WLF variant uses the operator convention
`log10(a_T) = C1 (T - T_ref) / (C2 + T - T_ref)` and
`tau_eff = tau_ref / a_T`. A deliberately discarded material trial is run before
the first accepted state; its complete reaction history must match the baseline.
