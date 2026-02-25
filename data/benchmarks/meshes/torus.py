#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np
import sys, getopt

def main(argv):
    usage = f'usage: {argv[0]} <output_db.vtk> [--major-radius=R] [--minor-radius=r] [--refinements=N]'

    if len(argv) < 2:
        print(usage)
        exit(1)

    output = argv[1]
    major_radius = 5.0  # Major radius (distance from center to tube center)
    minor_radius = 1.0  # Minor radius (tube radius)
    nrefs = 0           # Number of refinements

    try:
        opts, args = getopt.getopt(
            argv[2:], "h",
            ["refinements=", "major-radius=", "minor-radius=", "help"])

    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ("--refinements"):
            nrefs = int(arg)
        elif opt in ("--major-radius"):
            major_radius = float(arg)
        elif opt in ("--minor-radius"):
            minor_radius = float(arg)

    gmsh.initialize(argv=["", "-bin"])
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    
    model = gmsh.model()
    model.add("Torus")
    model.setCurrent("Torus")
    
    # Create torus using OpenCASCADE kernel
    # addTorus(x, y, z, r1, r2, angle, tag=-1)
    # r1 = minor radius, r2 = major radius
    torus = model.occ.addTorus(0, 0, 0, major_radius, minor_radius)
    
    # Synchronize before meshing
    model.occ.synchronize()
    
    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", minor_radius * 0.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", minor_radius * 0.5)
    
    # Generate 3D mesh
    model.mesh.generate(3)

    for r in range(0, nrefs):
        print(f"refinement {r} ...")
        model.mesh.refine()

    # Write mesh
    gmsh.write(output)
    
    # Print mesh statistics
    print(f"\nMesh statistics:")
    print(f"  Major radius: {major_radius}")
    print(f"  Minor radius: {minor_radius}")
    print(f"  Refinements: {nrefs}")
    
    gmsh.finalize()

if __name__ == "__main__":
    main(sys.argv)

