# TODOs

## P2 workflow support

The following driver require adaptation for the the pressure projection

- surface_outflux + surface_projection_p0_to_p1 -> (combined driver and p2)
- cgrad + projection_p0_to_p1 -> (combined driver and p2)
- divergence (p2)
- lumped_mass_inv (p2)
- volumes (check that p2 works by piggy-backing)
- integrate_divergence (this is used for checks, maybe p2)

## P2 solver support

- LOR operator for preconditioner 
	- CRS graph from implicit P1 graph from P2 mesh
	- FE assembly with macro element
- Matrix free P2 application (first version with matrix)?