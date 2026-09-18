# Homogeneous Active Strain

This case verifies the file-backed active deformation gradient used by SFEM's
packed HEX8 active-strain operators. A constant diagonal `Fa` is generated with
a determinant of one. The undeformed state exercises a nontrivial elastic
response, while `F = Fa` checks the compatible stress-free state.

The oracle evaluates `Fe = F Fa^-1`, the operator's strain-energy law, and
`P = det(Fa) Pe Fa^-T` independently. Displacement, interior residual, total
energy, and the resultant on the positive-x face are checked for matrix-free
and assembled BSR paths.

The legacy active kernels expose their generated arguments in a historical
order: Neo-Hookean receives physical `lambda` through `SFEM_MU` and physical
`mu` through `SFEM_LAMBDA`; Mooney-Rivlin additionally receives the physical
bulk parameter through `SFEM_LMDA`. The manifest records that case-layer
mapping explicitly while the oracle continues to use physical `mu` and
`lambda` names.
