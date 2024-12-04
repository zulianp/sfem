#!/usr/bin/env python3

import pysfem as sfem
import numpy as np
from numpy import linalg
import sys, getopt, os
from sfem.sfem_config import *

import yaml
try:
    from yaml import SafeLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

execution_space = sfem.ExecutionSpace.EXECUTION_SPACE_HOST
# -----------------------

# TODO move to pysfem
def create_dirichlet_conditions(fs, config):
	dirichlet_conditions = sfem.DirichletConditions(fs)
	for c in config:
		name = c['name']
		nodeset = c['nodeset']
		component = c['component']
		value = c['value']

		print(f"reading \"{name}\"")

		idx = np.unique(np.fromfile(nodeset, dtype=idx_t))
		if isinstance(component, list):
			for k in range(0, len(component)):
				print(f'{name}: {component[k]}, {value[k]}')
				sfem.add_condition(dirichlet_conditions, idx, component[k], value[k])
		else:
			print(f'{name}: {component}, {value}')
			sfem.add_condition(dirichlet_conditions, idx, component, value)

	return dirichlet_conditions

def run(config):
	m = sfem.Mesh()
	m.read(config['mesh'])


	dim = m.spatial_dimension()
	block_size = config['block_size']

	fs = sfem.FunctionSpace(m, block_size)

	fun = sfem.Function(fs)
	op = sfem.create_op(fs, config["operator"])
	fun.add_operator(op)

	dirichlet_conditions = create_dirichlet_conditions(fs, config['dirichlet_conditions'])

	fun.add_dirichlet_conditions(dirichlet_conditions)

	out = fun.output()
	out.set_output_dir(config['output'])
	out.enable_AoS_to_SoA(True)

	x = np.zeros(fs.n_dofs(), dtype=real_t)
	rhs = np.zeros(fs.n_dofs(), dtype=real_t)
	c = np.zeros(fs.n_dofs(), dtype=real_t)

	cc = sfem.contact_conditions_from_file(fs, str(config['obstacle']))
	
	# Update problem with current solution and linearize
	sfem.update(cc, x)
	
	op = sfem.make_op(fun, x)
	g = np.zeros(fs.n_dofs(), dtype=real_t)
	sfem.gradient_for_mesh_viz(cc, x, g)
	sfem.write(out, "gap", g)
	
	if config['solver'] == 'MPRGP':
		upper_bound = np.ones(fs.n_dofs(), dtype=real_t) * 1000
		sfem.gradient_for_mesh_viz(cc, x, upper_bound)
		solver = sfem.MPRGP()
		solver.default_init()
		solver.set_atol(1e-12)
		solver.set_rtol(1e-6);
		solver.set_max_it(2000)
		solver.set_op(op)
		sfem.set_upper_bound(solver, upper_bound)
	elif config['solver'] == "SPMG":
		spmg = sfem.create_spmg(fun, execution_space)

		upper_bound = np.zeros(cc.n_constrained_dofs(), dtype=real_t)
		sfem.gradient(cc, x, upper_bound)
		cc_op = cc.linear_constraints_op()
		cc_op_t = cc.linear_constraints_op_transpose()
		sp.set_constraints_op(cc_op, cc_op_t)
		sfem.set_upper_bound(sp, upper_bound)

	else:
		sp = sfem.ShiftedPenalty()
		sp.set_op(op)
		sp.default_init()

		linear_solver = sfem.ConjugateGradient()
		linear_solver.default_init()
		linear_solver.set_rtol(1e-3)
		linear_solver.set_verbose(False)
		linear_solver.set_max_it(100)
		sp.set_linear_solver(linear_solver)

		upper_bound = np.zeros(cc.n_constrained_dofs(), dtype=real_t)
		sfem.gradient(cc, x, upper_bound)
		cc_op = cc.linear_constraints_op()
		cc_op_t = cc.linear_constraints_op_transpose()
		print(f'Constrained dofs {cc.n_constrained_dofs()}/{fs.n_dofs()}')
		sp.set_constraints_op(cc_op, cc_op_t)
		sfem.set_upper_bound(sp, upper_bound)

		sp.set_max_it(40)
		sp.set_max_inner_it(10)
		sp.set_penalty_param(2)
		sp.set_atol(1e-8)
		sp.set_damping(0.01)
		# sp.enable_steepest_descent(True)
		solver = sp

	sfem.apply_constraints(fun, x)
	sfem.apply_constraints(fun, rhs)

	# if not use_MPRGP:
	# 	linear_solver.set_op(op)
	# 	sfem.apply(linear_solver, rhs, x)

	sfem.apply(solver, rhs, x)

	sfem.write(out, "disp", x)
	sfem.write(out, "rhs", rhs)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f'usage: {sys.argv[0]} <case>')
		exit(1)

	sfem.init()

	case_file = sys.argv[1]

	try:
	    opts, args = getopt.getopt(
	        sys.argv[2:], "ho:",
	        ["help","output="])

	except getopt.GetoptError as err:
	    print(err)
	    print(usage)
	    sys.exit(1)

	for opt, arg in opts:
	    if opt in ('-h', '--help'):
	        print(usage)
	        sys.exit()

	with open(case_file, 'r') as f:
	    config = list(yaml.load_all(f, Loader=Loader))[0]

	run(config)
	sfem.finalize()
