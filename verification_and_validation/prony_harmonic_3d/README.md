# Three-Dimensional Harmonic Prony Response (Virtual DMA)

A homogeneous constrained uniaxial strain `eps(t) = eps0 sin(omega t)` is prescribed from rest
while `GeneratedMooneyRivlinProny` advances its history, and the right face's reaction is
recorded every step: the force-displacement record of a dynamic mechanical analysis. Five
frequencies cover the spectrum (`omega tau_i` from 0.001 to 100,000), each at 64, 128 and 256
steps per period over four cycles.

Three oracles (`oracle.py`), from exact to continuum:

- **The recursion replayed.** The law is Simo's form on the second Piola-Kirchhoff stress,
  `S = g_inf S_iso + sum_i H_i`, with `H_i <- alpha_i H_i + b_i (S_iso^{n+1} - S_iso^n)`.
  For a homogeneous deformation the oracle replays it on the prescribed stretches, at any
  amplitude, so the simulated reaction must match to the solver's tolerance.
- **The transfer function.** At small strain the recursion is a linear filter; its response at
  `omega` and `dt` is the algorithm's own complex modulus,
  `G_dt(omega)/G0 = g_inf + sum_i b_i (1 - z^-1)/(1 - alpha_i z^-1)`, `z = exp(i omega dt)`.
  `G' + i G''` is fitted to the whole history with the transient's exponentials in the basis,
  so no steady state is needed.
- **The continuum.** The recursion at 8192 steps per period for the reaction and the loop area
  (energy dissipated per cycle), and the generalized Maxwell modulus
  `G*(omega)/G0 = g_inf + sum_i g_i i omega tau_i/(1 + i omega tau_i)` for the fitted moduli.

The continuum checks hold where the coarsest step resolves the fastest relaxation time
(`dt <= tau_min`, here `omega >= 10`): with longer steps the linear-interpolation scheme
converges at a rate below two and the midpoint rule at first order (its weight
`g exp(-dt/2tau)` vanishes for `dt >> tau`). `harmonic_levels.csv` and the report's diagnostics
record every frequency. The large-amplitude variant (`eps0 = 0.3`) keeps the replayed and
continuum checks; the moduli are defined at small strain only.
