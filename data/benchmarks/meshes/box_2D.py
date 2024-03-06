#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np
import sys

def main(argv):
	nrefs = 1

	if(len(argv) < 2):
		print(f'usage: {argv[0]} <output_db.vtk> [nrefs=1] [width] [height]')
		exit(1)

	output = argv[1]
	nrefs = int(argv[2])
	width=2
	height=1

	if len(argv) > 3:
		width = float(argv[3])

	if len(argv) > 4:
		height = float(argv[4])

	gmsh.initialize(argv=["","-bin"])
	gmsh.option.setNumber("General.Terminal", 0)
	gmsh.option.setNumber("Mesh.SaveAll", 1)
	gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
	gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
	gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


	model = gmsh.model()
	model.add("Rectangle")
	model.setCurrent("Rectangle")
	model.occ.addRectangle(.0, 0, 0, width, height)

	# Generate mesh
	model.occ.synchronize()
	# model.mesh.generate(3)
	model.mesh.generate(2)

	for r in range(0, nrefs):
		model.mesh.refine()

	gmsh.write(output)
	gmsh.finalize()


if __name__ == "__main__":
	main(sys.argv)

