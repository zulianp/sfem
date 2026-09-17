# Kirsch Quarter-Annulus Verification

Plane-strain infinite-plate solution for a circular hole of radius `a` under
remote uniaxial tension `S` in the x direction. The finite quarter annulus
uses the exact displacement on the outer arc, zero normal displacement on the
coordinate axes, and a traction-free polygonal inner boundary. No boundary
stress or displacement is prescribed on the hole.

All geometry and loading values are in `case.yaml`; length is in metres and
stress in megapascals. `oracle.py` evaluates the classical Kirsch displacement
and stress equations independently of SFEM. `verify.py` samples stress at
element quadrature points and records a hole-adjacent hoop-stress profile.

The polygonal hole differs from the analytical circle. The reported spatial
rate therefore includes geometry approximation, not only element interpolation.
