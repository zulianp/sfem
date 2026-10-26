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

	n = 10
	# h = 1./(n - 1)

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
		if options.elem_type == "INVALID":
			options.elem_type = "TET4"

		xlen = 1
		idx, points = box_mesh.create(xlen, 1, 1, n * xlen, n * 1, n * 1, options.elem_type)
		
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

	
		m = pysfem.create_mesh(options.elem_type, np.array(idx), np.array(points))
		m.write(f"{options.output_dir}/rect_mesh")
	else:
		m = pysfem.Mesh.read(path)
		m.convert_to_macro_element_mesh()
		sinlet = np.unique(np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t))
		soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)


	fs = pysfem.FunctionSpace(m, m.spatial_dimension())
	fun = pysfem.Function(fs)
	
	fun.set_output_dir(options.output_dir)
	out = fun.output()
	# The displacement is written one component per file.
	out.enable_AoS_to_SoA(True)

	linear_elasticity_op = pysfem.create_op(fs, "LinearElasticity")
	linear_elasticity_op.initialize()
	fun.add_operator(linear_elasticity_op)

	mass_op = pysfem.create_op(fs, "LumpedMass")
	mass_op.initialize()
	
	bc = pysfem.DirichletConditions(fs)

	bc.add_condition(sinlet, 0, 0.3);
	bc.add_condition(sinlet, 1, 0.);
	bc.add_condition(sinlet, 2, 0.);

	# bc.add_condition(soutlet, 0, 0);
	# bc.add_condition(soutlet, 1, 0.);
	# bc.add_condition(soutlet, 2, 0.);

	fun.add_dirichlet_conditions(bc)

	# nc = pysfem.NeumannConditions(fs)
	# nc.add_condition(soutlet, 0, 0.01)
	# fun.add_operator(nc)

	selector = np.ones(fs.n_dofs())
	selector[0:m.spatial_dimension():fs.n_dofs()] = 1

	dt = 0.0001
	export_freq = 1
	T = 20
	t = 0

	x = np.zeros(fs.n_dofs())
	# Buffers over the arrays: the library reads and writes them in place.
	xb = pysfem.view(x)

	next_check_point = export_freq

	pysfem.write_time_step(out, "disp", t, xb)
	buff = np.zeros(fs.n_dofs(), dtype=real_t)
	buffb = pysfem.view(buff)
	mass = np.zeros(fs.n_dofs(), dtype=real_t)
	
	pysfem.hessian_diag(mass_op, xb, pysfem.view(mass))
	pysfem.apply_value(bc, 1, pysfem.view(mass))

	obs = np.zeros(fs.n_dofs())
	obs[0::m.spatial_dimension()] = 1.1 - pysfem.points(m, 0)
	penalty_param = 1
	# penalty_param = 0 # Deactivate penalty

	for d in range(1, m.spatial_dimension()):
		obs[d::m.spatial_dimension()] = 1000

	while(t < T):
		t += dt
		buff.fill(0)
		pysfem.gradient(fun, xb, buffb)
		
		penalty = np.minimum(obs - x, 0)
		x -= dt * (buff - penalty_param * penalty) / mass

		if t >= (next_check_point - dt / 2):
			pysfem.write_time_step(out, "disp", t, xb)
			next_check_point += export_freq

			norm_disp = linalg.norm(x)
			print(f't={round(t, 5)}/{T} norm_disp={norm_disp}, max penet={round(-np.min(penalty), 5)}')

class Opts:
	def __init__(self):
		self.input_mesh = ''
		self.output_dir = './output'
		self.elem_type = "INVALID"

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) < 3:
		print(f'usage: {sys.argv[0]} <input_mesh> <output>')
		exit(1)

	pysfem.init()

	options = Opts()
	options.input_mesh = sys.argv[1]
	options.output_dir = sys.argv[2]
	
	try:
	    opts, args = getopt.getopt(
	        sys.argv[3:], "h",
	        ["help", "elem_type=", "cell_type="])

	except getopt.GetoptError as err:
	    print(err)
	    print(usage)
	    sys.exit(1)

	for opt, arg in opts:
	    if opt in ('-h', '--help'):
	        print(usage)
	        sys.exit()
	    elif opt in ('--elem_type', '--cell_type'):
	       	options.elem_type = arg

	solve_obstacle(options)
	pysfem.finalize()
