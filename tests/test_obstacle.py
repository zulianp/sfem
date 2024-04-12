#!/usr/bin/env python3

import pysfem
import numpy as np
from numpy import linalg
import sfem.mesh.rectangle_mesh as rectangle_mesh
import sfem.mesh.box_mesh as box_mesh

import sys, getopt, os

idx_t = np.int32
real_t = np.float64

def solve_obstacle(options):
	path = options.input_mesh

	if not os.path.exists(options.output_dir):
		os.mkdir(f'{options.output_dir}')

	n = 20
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

	fs = pysfem.FunctionSpace(m, m.spatial_dimension())
	fun = pysfem.Function(fs)
	
	fun.set_output_dir(options.output_dir)
	out = fun.output()

	linear_elasticity_op = pysfem.create_op(fs, "LinearElasticity")
	fun.add_operator(linear_elasticity_op)

	mass_op = pysfem.create_op(fs, "LumpedMass")
	
	bc = pysfem.DirichletConditions(fs)

	pysfem.add_condition(bc, sinlet, 0, 0.3);
	pysfem.add_condition(bc, sinlet, 1, 0.);
	pysfem.add_condition(bc, sinlet, 2, 0.);

	# pysfem.add_condition(bc, soutlet, 0, 0);
	# pysfem.add_condition(bc, soutlet, 1, 0.);
	# pysfem.add_condition(bc, soutlet, 2, 0.);

	fun.add_dirichlet_conditions(bc)

	# nc = pysfem.NeumannConditions(fs)
	# pysfem.add_condition(nc, soutlet, 0, 0.01)
	# fun.add_operator(nc)

	selector = np.ones(fs.n_dofs())
	selector[0:m.spatial_dimension():fs.n_dofs()] = 1

	dt = 0.0001
	export_freq = 1
	T = 20
	t = 0

	x = np.zeros(fs.n_dofs())

	next_check_point = export_freq

	pysfem.write_time_step(out, "disp", t, x)
	buff = np.zeros(fs.n_dofs(), dtype=real_t)
	mass = np.zeros(fs.n_dofs(), dtype=real_t)
	
	pysfem.hessian_diag(mass_op, x, mass)
	pysfem.apply_value(bc, 1, mass)

	obs = np.zeros(fs.n_dofs())
	obs[0::m.spatial_dimension()] = 2.0 - pysfem.points(m, 0)
	penalty_param = (1/(dt * 1000))
	penalty_param = 0 # Deactivate penalty

	for d in range(1, m.spatial_dimension()):
		obs[d::m.spatial_dimension()] = 1000

	while(t < T):
		t += dt
		buff.fill(0)
		pysfem.gradient(fun, x, buff)
		
		penalty = np.minimum(obs - x, 0)
		x -= dt * (buff - penalty_param * penalty) / mass

		if t >= (next_check_point - dt / 2):
			pysfem.write_time_step(out, "disp", t, x)
			next_check_point += export_freq

			norm_disp = linalg.norm(x)
			print(f't={round(t, 5)}/{T} norm_disp={norm_disp}, max penet={round(-np.min(penalty), 5)}')

class Opts:
	def __init__(self):
		self.input_mesh = ''
		self.output_dir = './output'

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <input_mesh> <output>')
		exit(1)

	pysfem.init()

	options = Opts()
	options.input_mesh = sys.argv[1]
	options.output = sys.argv[2]
	
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

	solve_obstacle(options)
	pysfem.finalize()
