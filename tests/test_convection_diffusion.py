#!/usr/bin/env python3


import numpy as np
from numpy import linalg
import sfem.mesh.rectangle_mesh as rectangle_mesh
import sfem.mesh.box_mesh as box_mesh
import pysfem as pysfem
import sys, getopt
import pdb
import os

idx_t = np.int32
real_t = np.float64

def convection_diffusion(options):
	path = options.input_mesh

	if not os.path.exists(options.output_dir):
		os.mkdir(f'{options.output_dir}')

	n = 100
	h = 1./(n - 1)

	if path == "gen:rectangle":
		idx, points = rectangle_mesh.create(2, 1, 2*n, n, "triangle")

		select_inlet  = np.abs(points[0]) 	< 1e-8
		select_outlet = np.abs(points[0] - 2) < 1e-8
		select_walls  = np.logical_or(np.abs(points[1]) < 1e-8, np.abs(points[1] - 1) < 1e-8)

		sinlet  = np.array(np.where(select_inlet), dtype=idx_t)
		soutlet = np.array(np.where(select_outlet), dtype=idx_t)
		swalls  = np.array(np.where(select_walls), dtype=idx_t)

		m = pysfem.create_mesh("TRI3", np.array(idx), np.array(points))
		m.write(f"{options.output_dir}/rect_mesh")
	elif path == "gen:box":
		idx, points = box_mesh.create(2, 1, 1, n * 2, n * 1, n * 1, "tet4")
		
		select_inlet  = np.abs(points[0]) 	< 1e-8
		select_outlet = np.abs(points[0] - 2) < 1e-8
		select_walls  = np.logical_or(
			np.logical_or(
				np.abs(points[1]) < 1e-8, 
				np.abs(points[1] - 1) < 1e-8),
			np.logical_or(
				np.abs(points[2]) < 1e-8,
				np.abs(points[2] - 1) < 1e-8))

		sinlet  = np.array(np.where(select_inlet), dtype=idx_t)
		soutlet = np.array(np.where(select_outlet), dtype=idx_t)
		swalls  = np.array(np.where(select_walls), dtype=idx_t)

		m = pysfem.create_mesh("TET4", np.array(idx), np.array(points))
		m.write(f"{options.output_dir}/rect_mesh")
	else:
		m = pysfem.Mesh()		
		m.read(path)
		m.convert_to_macro_element_mesh()
		sinlet = np.unique(np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t))
		soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	fs  = pysfem.FunctionSpace(m, 1)
	fun = pysfem.Function(fs)

	fun.set_output_dir(options.output_dir)
	out = fun.output()

	velx = np.ones(fs.n_dofs(),dtype=real_t);
	vely = np.zeros(fs.n_dofs(),dtype=real_t);
	acc = velx * velx + vely * vely

	CFL = np.max(np.abs(velx))/h
	CFL += np.max(np.abs(vely))/h

	mass_op = pysfem.create_op(fs, "CVFEMMass")
	convection_op = pysfem.create_op(fs, "CVFEMUpwindConvection")

	pysfem.set_field(convection_op, "velocity", 0, velx)
	pysfem.set_field(convection_op, "velocity", 1, vely)

	if m.spatial_dimension() == 3:
		velz = np.zeros(fs.n_dofs());
		pysfem.set_field(convection_op, "velocity", 2, velz)
		CFL += np.max(np.abs(velz))/h

	speed = np.max(np.sqrt(acc))

	fun.add_operator(convection_op)

	# bc = pysfem.DirichletConditions(fs)
	# pysfem.add_condition(bc, swalls,  0, 0);
	# fun.add_dirichlet_conditions(bc)
	
	x = np.zeros(fs.n_dofs(),dtype=real_t)
	x[sinlet] = 1

	mass = np.zeros(fs.n_dofs(), dtype=real_t)
	pysfem.hessian_diag(mass_op, x, mass)
	pysfem.apply_constraints(fun, x)

	dt = 0.1/(CFL * m.spatial_dimension())
	export_freq = 0.001
	T = 1
	t = 0

	next_check_point = T * export_freq

	pysfem.write_time_step(out, "c", t, x)
	buff = np.zeros(fs.n_dofs())

	domain_integral0 = np.dot(x, mass)
	
	while(t < T):
		t += dt

		buff.fill(0)
		pysfem.gradient(fun, x, buff)

		x += dt * buff / mass

		if t >= (next_check_point - dt / 2):
			pysfem.write_time_step(out, "c", t, x)
			next_check_point += export_freq

			domain_integral = np.dot(x, mass)
			print(f't={round(t, 5)}/{T}, integr(x) = {round(domain_integral, 5)}, diff  = {round((domain_integral - domain_integral0)/domain_integral0)}')

class Opts:
	def __init__(self):
		self.input_mesh = ''
		self.output_dir = '.'

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <input_mesh> <output.raw>')
		exit(1)

	options = Opts()
	options.input_mesh = sys.argv[1]
	options.output_dir = sys.argv[2]
	
	try:
	    opts, args = getopt.getopt(
	        sys.argv[3:], "h",
	        ["help"])

	except getopt.GetoptError as err:
	    print(err)
	    print(usage)
	    sys.exit(1)

	for opt, arg in opts:
	    if opt in ('-h', '--help'):
	        print(usage)
	        sys.exit()

	pysfem.init()
	convection_diffusion(options)
	pysfem.finalize()
