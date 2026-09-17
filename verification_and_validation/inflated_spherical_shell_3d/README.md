# Inflated Spherical-Octant Verification

A thick spherical octant is inflated by 3D follower pressure on its oriented
inner surface. The exterior is free and coordinate-plane normal displacement
is zero. All meshes and sidesets are generated from `case.yaml`; the pressure
load is ramped by the driver from zero to its declared final value.

`oracle.py` solves the independent radial boundary-value equations for
`r(R)` and `dr/dR` from nominal stress equilibrium. Its inner boundary uses
`P_r(A) = -p [r(A)/A]^2` and its outer boundary uses `P_r(B)=0`.
`verify.py` compares inner displacement, pressure-volume histories, radial
and hoop Cauchy stress profiles, positive Jacobian, and spatial reduction.
The cavity-volume history is integrated from the oriented deformed inner
faces, with the polygonal reference volume subtracted at each level.
Stress profiles are volume-weighted angular averages within each declared
radial layer; the oracle is averaged over the same quadrature locations and
weights. Raw local quadrature stress errors are recorded separately in the
diagnostics because first-order displacement elements can show large angular
hydrostatic oscillations at high bulk modulus. The profile comparison does
not certify pointwise stress accuracy.

Length is in metres and pressure/stress in megapascals. Moderate and nearly
incompressible variants use the same geometry and pressure; only constitutive
parameters change. The radial oracle is separately tested against Lame's
small-strain limit and the incompressible mapping `r^3 = R^3 + c`.
