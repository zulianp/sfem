#!/usr/bin/env python3

"""
Generates a single cyclic sector.

Requirements
------------

.. code::

   pip install gmsh mapdl-archive pyvista numpy

"""
import math
import time
import sys

import gmsh
# import mapdl_archive
import numpy as np
# import pyvista as pv




# Define geometric parameters
r1 = 0.057  # internal radious
r2 = 0.067  # from bottom of rotor slot to the center
r3 = 0.08  # from top of rotor slot to the center
r4 = 0.0814  # from top of stator slot to the center
r5 = 0.0944  # from bottom of rotor slot to the center
r6 = 0.1044  # external radious (half of the diameter)

N_ROTOR = 400
N_STATOR = 440


PT_ROTOR = 0.5  # fill factor for rotor teeth from 0 to 1
PT_STATOR = 0.5  # fill factor for stator teeth from 0 to 1

# state
AR = 2  # rotor position in mechanical degrees

N_RAD_DENSITY = 4
N_TAN_DENSITY = 10
Z_SUBDIVIDE = 3

TIMES=1

if len(sys.argv) == 2:
    TIMES = int(sys.argv[1])

N_RAD_DENSITY = 10 * TIMES
N_TAN_DENSITY = 20 * TIMES
Z_SUBDIVIDE = 6 * TIMES


def polar_to_cartesian(radius, angle, center):
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    return (x, y)


# Start Gmsh API
gmsh.initialize()
gmsh.model.add("Model")

# Add points to gmsh model
center_xy = (0.0, 0.0)
p_center = gmsh.model.geo.addPoint(*center_xy, 0.0)


def gen_arcs(n_arc, radius, ang_start=0, fill_factor=0.5):
    """Generate all inner arcs."""
    arc_len = math.pi / (n_arc / 2)
    points = []
    angle = ang_start
    for ii in range(n_arc):
        x, y = polar_to_cartesian(radius, angle, center_xy)
        if ii % 2:
            angle += 2 * arc_len * fill_factor
        else:
            angle += 2 * arc_len * (1 - fill_factor)
        points.append(gmsh.model.geo.addPoint(x, y, 0.0))

    # Create arcs
    arcs = []
    for ii in range(n_arc - 1):
        arcs.append(gmsh.model.geo.addCircleArc(points[ii], p_center, points[ii + 1]))

    arcs.append(gmsh.model.geo.addCircleArc(points[-1], p_center, points[0]))

    # Set transfinite lines
    for ii in range(n_arc):
        gmsh.model.geo.mesh.setTransfiniteCurve(arcs[ii], N_TAN_DENSITY)

    return points, arcs


def gen_cyc(n_arc, r_inner, r_middle, ang_start=0.0, fill_factor=0.5):
    ang_start_rad = ang_start * math.pi / 180
    points_inner, arcs_inner = gen_arcs(n_arc, r_inner, ang_start_rad, fill_factor)
    loop_inner = gmsh.model.geo.addCurveLoop(arcs_inner)

    points_middle, arcs_middle = gen_arcs(n_arc, r_middle, ang_start_rad, fill_factor)
    loop_middle = gmsh.model.geo.addCurveLoop(arcs_middle)

    # generate inner to mid connecting lines
    in_to_mid_con_lines = []
    for ii in range(n_arc):
        in_to_mid_con_lines.append(
            gmsh.model.geo.addLine(points_inner[ii], points_middle[ii])
        )
        gmsh.model.geo.mesh.setTransfiniteCurve(in_to_mid_con_lines[-1], N_RAD_DENSITY)

    # generate individual "loops" representing the magnets and iron cores
    plane_inner = []
    # plane_outer = []
    # for ii in range(n_arc):
    for ii in range(1):  # single arc override
        # generate "inner core", these are all iron
        line0 = in_to_mid_con_lines[ii]
        # wrap around last line
        if ii == n_arc - 1:
            line1 = in_to_mid_con_lines[0]
        else:
            line1 = in_to_mid_con_lines[ii + 1]
        loop = gmsh.model.geo.addCurveLoop(
            [arcs_inner[ii], line0, arcs_middle[ii], line1], reorient=True
        )
        plane = gmsh.model.geo.addPlaneSurface([loop])
        plane_inner.append(plane)
        gmsh.model.geo.mesh.setTransfiniteSurface(plane)

    return plane_inner


###############################################################################
# Generate 2D geometry
inner_plane = gen_cyc(2, 1, 2)
gmsh.model.geo.synchronize()

# Set option to generate quadrilateral elements
gmsh.option.setNumber("Mesh.RecombineAll", True)

# Generate 2D mesh
# gmsh.model.mesh.generate(2)


###############################################################################
# Generate 3D geometry
# Define extrusion vector [dx, dy, dz]
extrude_vector = [0, 0, 0.5]  # Replace 0.1 with your extrusion length

pgroup = 2
base_surf_pg = gmsh.model.addPhysicalGroup(
    pgroup, inner_plane, tag=100, name="lower_surface"
)

# Extrude the mesh
subdivision = [Z_SUBDIVIDE]
dimTags = gmsh.model.getEntities(2)
extrusion = gmsh.model.geo.extrude(
    [(pgroup, inner_plane[0])], 0, 0, 1, subdivision, recombine=True
)


###############################################################################
# Mesh in 3D
gmsh.model.geo.synchronize()
volume = gmsh.model.addPhysicalGroup(3, [extrusion[1][1]], name="volume")

gmsh.model.mesh.generate(3)
gmsh.model.mesh.refine()


###############################################################################
# Write out to a file
filename = "./model.vtk"
gmsh.write(filename)


###############################################################################
# read in using pyvista and convert to an archive file using mapdl-archive
# grid = pv.read(filename)
# grid.plot(color='w', show_edges=True, line_width=1.5)

# mapdl_archive.save_as_archive("./archive.cdb", grid)
