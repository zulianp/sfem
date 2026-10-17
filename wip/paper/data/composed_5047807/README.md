# Boundary and transient derivative checks

User-run Grace job 5047807 on nid006553, 13 October 2026 UTC, GCC 14.3.0.
The twelve files in `raw/` are unchanged copies of the returned public result.
`RAW-SHA256.txt` binds every file. Replay the result with:

```sh
python3 wip/paper/data/composed_5047807/verify.py
```

The packed residual previously gathered the steady Rhie–Chow coefficient even
when the action used the optional time-dependent coefficient. The correction
passes the resolved `Hex8RcConfig`, including its time scale, through the packed
residual paths. It also passes the same time scale through the action's gather.
No new stabilization formula is introduced.

For fixed histories, the BDF contribution to momentum is
`rho * V * (a0*u + a1*u_previous + a2*u_previous2) / dt`; its derivative is
`rho * V * a0 / dt` on the velocity diagonal. The optional Rhie–Chow time scale
uses that same `a0/dt`. The test differentiates the full four-field residual,
including reconstructed pressure/velocity gradients, boundary closure and the
time term, while keeping the histories fixed.

The composed gate covers 90 configurations and 180 required finite-difference
steps at 1 and 72 threads: five convection schemes, boundary-only/BDF1/BDF2 and
their combinations, and both transient-RC settings where applicable. The fixture
is an affine box with `n=8`, pack size 64, perturbation 0.017, density 1.3 and
dynamic viscosity 0.027. Maximum whole-vector relative error is
`3.518066038e-9`; all four fields pass separately. Deliberately omitted terms
remain detectable. The archive also retains all 16 related CTest invocations.

This checks affine boundary closure before strong row elimination and fixed
BDF histories. It does not establish arbitrary boundary conditions, a time
trajectory, temporal or spatial order, nonlinear solver convergence, or speedup.
Separate generator and temporal-order gates were skipped in the Grace build
because its Python environment lacked SymPy and NumPy.

## Source provenance

`raw/SOURCE-SHA256.txt` and `raw/BUILD-SHA256.txt` identify the original cluster
bundle and binaries. The recorded revisions file contains historical pending
statuses; the completed logs supersede those statuses for this job only.
The returned patch also contains preceding cache/fixed-input changes. It is
retained as evidence, not presented as this commit's diff.

`source-comparison.json` records the current fix's twelve paths: ten match the
Grace manifest exactly. The two others also contain the subsequent irregular-mesh
implementation from commit `076aac1b`; that implementation was independently
checked by job 5052301 and local source-only builds. The present combined source
is therefore not claimed to be the exact binary source of job 5047807.

To repeat the composed check with a freshly built benchmark:

```sh
COUPLED_TEST_THREADS='1 72' bash \
  spikes/cvfem/src/tests/verification/cvfem_jacobian_composed.sh \
  /absolute/path/to/cvfem_hex8_ns_upwind_bench
```

The full commands for the additional CTests are in the retained CTest logs.
