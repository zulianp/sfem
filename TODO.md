# TODOs

## P2 workflow support

The following driver require adaptation for the the pressure projection

- surface_outflux + surface_projection_p0_to_p1 -> (combined driver and p2) [UNTESTED]
- cgrad + projection_p0_to_p1 -> (combined driver and p2) 					[UNTESTED]
- divergence (p2), implement dispatch 										[UNTESTED]
- lumped_mass_inv (p2)   													[UNTESTED]
- volumes (check that p2 works by piggy-backing) 							[LOOKS good]
- integrate_divergence (this is used for checks, maybe p2)  				[UNTESTED]
- surface_projection_p0_to_p1 -> surface_projection_p1_to_p2 				[UNTESTED]
- Incorporate new components into workflow 									[DONE]
- Full p2 workflow test 													[TODO]

## P2 solver performance

- POC: first version with matrix  (2x speed up, reduced mem footprint]) 	[DONE]
- LOR operator for preconditioner 
	- CRS graph from implicit P1 graph from P2 mesh 						[TODO]
	- FE assembly with macro element 										[TODO]
- Matrix free P2 application 												[TODO]

