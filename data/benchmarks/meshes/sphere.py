#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np
import sys

def main(argv):
	nrefs = 1

	if(len(argv) < 2):
		print(f'usage: {argv[0]} <output_db.vtk> [nrefs=1]')
		exit(1)

	output = argv[1]
	radius=0.5

	if(len(argv) > 2):
		nrefs = int(sys.argv[2])
		print(f'nrefs = {nrefs}')

	gmsh.initialize(argv=["","-bin"])
	gmsh.option.setNumber("General.Terminal", 0)
	gmsh.option.setNumber("Mesh.SaveAll", 1)

	model = gmsh.model()
	model.add("Sphere")
	model.setCurrent("Sphere")
	model.occ.addSphere(0, 0, 0, radius)

	# Generate mesh
	model.occ.synchronize()
	model.mesh.generate(3)

	for r in range(0, nrefs):
		model.mesh.refine()

	gmsh.write(output)
	gmsh.finalize()

if __name__ == "__main__":
	main(sys.argv)


