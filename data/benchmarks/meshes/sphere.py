#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np
import sys, getopt

def main(argv):
	usage = f'usage: {argv[0]} <output_db.vtk>'

	if(len(argv) < 2):
		print(usage)
		exit(1)

	output = argv[1]
	radius=0.5
	nrefs = 1

	try:
	    opts, args = getopt.getopt(
	        argv[3:], "e:h",
	        ["refinements=", "radius=", "help"])

	except getopt.GetoptError as err:
	    print(err)
	    print(usage)
	    sys.exit(1)

	for opt, arg in opts:
	    if opt in ('-h', '--help'):
	        print(usage)
	        sys.exit()
	    elif opt in ("--refinemnts"):
	    	nrefs = int(arg)
	    elif opt in ("--radius"):
	    	radius = float(arg)

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


