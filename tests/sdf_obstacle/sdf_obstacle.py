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

def run(case):
	m = sfem.Mesh()
	m.read(case['mesh'])

	dim = m.spatial_dimension()
	block_size = case['block_size']

	fs = sfem.FunctionSpace(m, block_size)

	fun = sfem.Function(fs)
	op = sfem.create_op(fs, case["operator"])
	fun.add_operator(op)

	dirichlet_conditions = create_dirichlet_conditions(fs, case['dirichlet_conditions'])

	fun.add_dirichlet_conditions(dirichlet_conditions)

	out = fun.output()
	out.set_output_dir(config['output'])
	out.enable_AoS_to_SoA(True)

	x = np.zeros(fs.n_dofs(), dtype=real_t)
	rhs = np.zeros(fs.n_dofs(), dtype=real_t)
	c = np.zeros(fs.n_dofs(), dtype=real_t)
	
	cc = sfem.contact_conditions_from_file(fs, str(config['obstacle']))
	upper_bound = np.zeros(cc.n_constrained_dofs(), dtype=real_t)

	
	# Update problem with current solution and linearize
	sfem.update(cc, x)
	sfem.gradient(cc, x, upper_bound)
	# upper_bound *= -1
	cc_op = cc.linear_constraints_op()
	cc_op_t = cc.linear_constraints_op_transpose()

	print(f'Constrained dofs {cc.n_constrained_dofs()}/{fs.n_dofs()}')
	


	op = sfem.make_op(fun, x)
	linear_solver = sfem.ConjugateGradient()
	linear_solver.default_init()
	linear_solver.set_verbose(False)
	linear_solver.set_max_it(10000)

	g = np.zeros(fs.n_dofs(), dtype=real_t)
	sfem.gradient_for_mesh_viz(cc, x, g)
	sfem.write(out, "gap", g)

	sp = sfem.ShiftedPenalty()
	sp.set_op(op)
	sp.default_init()
	sp.set_linear_solver(linear_solver)
	sp.set_upper_bound(sfem.view(upper_bound))
	sp.set_constraints_op(cc_op, cc_op_t)
	sp.set_max_it(20)
	sp.set_max_inner_it(10)
	sp.set_penalty_param(1.1)
	sp.set_damping(0.3)
	# sp.enable_steepest_descent(True)

	sfem.apply_constraints(fun, x)
	sfem.apply_constraints(fun, rhs)

	# print(g)
	# print(rhs)

	linear_solver.set_op(op)
	sfem.apply(linear_solver, rhs, x)

	sfem.apply(sp, rhs, x)	


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
