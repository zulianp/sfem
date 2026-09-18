# Two-Material Linear Patch

Two bonded HEX8 blocks occupy equal halves of the unit cube and use different
Lamé parameters. The exact axial strains are chosen so both blocks carry unit
normal stress, giving continuous traction at the interface. Zero transverse
displacement imposes a uniaxial-strain state.

The oracle checks the piecewise-affine displacement, free-node residual,
strain energy, and positive-x face reaction for matrix-free and BSR solves.
