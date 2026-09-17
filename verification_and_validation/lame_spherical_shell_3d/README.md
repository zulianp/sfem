# Lame Spherical-Octant Verification

An octant of a thick linear-elastic sphere with internal pressure and a free
exterior. The coordinate planes prescribe zero normal displacement. The
inner faces receive facewise dead traction equal to pressure times the inward
surface normal; no analytical displacement is prescribed on either sphere.

`case.yaml` declares radii, pressure, Lame parameters, mesh levels, and
tolerances. Length is in metres; pressure and stress are in megapascals.
`oracle.py` evaluates `u_r = A r + B/r^2` and Lame's radial/hoop stress from
the declared values. The verifier compares full displacement and stress
fields, inner-face resultant, and spatial convergence. Polygonal surface
area and side orientation are checked before the driver runs.
