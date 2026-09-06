# Throughput: exact vs projected apply

Measured with `bench.cpp`, linear elasticity on TET4, single-threaded, `-O3
-march=native`, Apple M-series laptop, best of 7 runs per size.

```
  elements        nodes         ndof   exact MDOF/s   proj. MDOF/s     ratio    rel. diff
      3072          729         2187          23.32          22.29      0.96x     1.65e-16
     24576         4913        14739          19.42          17.44      0.90x     1.54e-16
     82944        15625        46875          18.65          16.61      0.89x     1.51e-16
    196608        35937       107811          17.97          16.41      0.91x     1.56e-16
    384000        68921       206763          17.79          15.24      0.86x     1.66e-16
```

Throughput flattens from about 47k degrees of freedom upward, so the last three
rows are the saturated regime and the honest comparison.

**The projected apply is slower here, by ten to fifteen percent, and that is
expected rather than a defect.** Linear elasticity on TET4 is the case where
the technique has nothing to remove:

* the material is linear, so the tangent does not depend on the state and the
  exact kernel already carries a constant one;
* TET4 has a single quadrature point, so there is no quadrature loop to
  collapse.

The projected kernel therefore does strictly more work than the exact one --
it rebuilds the tangent per element, packs forty-five numbers, and runs the
staged contraction -- to arrive at the same answer, which it does to 1.6e-16.

This combination was chosen for the *correctness* gate, precisely because the
tangent is state independent and the projection is exact, so agreement is a
clean pass/fail. It is the wrong combination for a speed claim.

## Where a gain could come from, and why it is not measured yet

The technique pays where the exact kernel re-evaluates the tangent at every
quadrature point and the projected one evaluates it once: a nonlinear material
on a curved or higher-order element. On HEX8 that is eight tangent evaluations
against one.

The emitter cannot reach that case today. `emittable_inexact_apply_plan`
requires the element's reference gradients to be constant, and only the linear
simplices satisfy it:

```
TET4   n_shape=4  n_qp=1  reference gradients: 12   constant (4*3)     emittable
HEX8   n_shape=8  n_qp=8  reference gradients: 192  per point (8*8*3)  not emittable
```

So the wiring currently covers exactly the elements on which the projection is
exact -- which is the right place to have started, because it is the only place
the result can be checked against an exact answer -- and none of the elements
on which it could be faster.

Closing that gap needs the tangent evaluated at the element average rather than
at a single point, which is the L2 projection actually being taken rather than
assumed away, and the reference tensor already handles the varying basis. Until
that exists, no speed-up should be quoted for this technique.
