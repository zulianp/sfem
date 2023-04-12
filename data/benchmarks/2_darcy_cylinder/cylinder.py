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
	height=1

	if(len(argv) > 2):
		nrefs = int(sys.argv[2])
		print(f'nrefs = {nrefs}')

	gmsh.initialize(argv=["","-bin"])
	gmsh.option.setNumber("General.Terminal", 0)
	gmsh.option.setNumber("Mesh.SaveAll", 1)

	model = gmsh.model()
	model.add("Cylinder")
	model.setCurrent("Cylinder")
	model.occ.addCylinder(0, 0, 0, 0, height, 0, radius, tag=1)

	inlet=1
	outlet=2
	wall=3

	walls = []

	for s in gmsh.model.occ.getEntities(dim=2):
		com = gmsh.model.occ.getCenterOfMass(s[0], s[1])
		
		if(com[1] < 1e-9):
			# print('inlet')
			gmsh.model.addPhysicalGroup(s[0], [s[1]], inlet)
			gmsh.model.setPhysicalName(s[1], inlet, "sinlet")
		elif(com[1] >= 2-1e-9):
			# print('outlet')
			gmsh.model.addPhysicalGroup(s[0], [s[1]], outlet)
			gmsh.model.setPhysicalName(s[1], outlet, "soutlet")
		else:
			# print('wall')
			walls.append(s[1])

		# print(gmsh.model.getEntityName(s[0], s[1]))


	gmsh.model.addPhysicalGroup(2, walls, wall)

	volumes = gmsh.model.occ.getEntities(dim=3)
	model.occ.rotate(volumes, 0, 0, 0, 0, 0, 1, -np.pi/2)
	model.occ.translate(volumes, -height/2, 0, 0)

	# Generate mesh
	model.occ.synchronize()
	model.mesh.generate(3)

	for r in range(0, nrefs):
		model.mesh.refine()

	gmsh.write(output)
	gmsh.finalize()

if __name__ == "__main__":
	main(sys.argv)


