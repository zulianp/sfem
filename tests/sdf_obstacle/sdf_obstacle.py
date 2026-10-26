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
				dirichlet_conditions.add_condition(idx, component[k], value[k])
		else:
			print(f'{name}: {component}, {value}')
			dirichlet_conditions.add_condition(idx, component, value)

	return dirichlet_conditions

def run(config):
	m = sfem.Mesh.read(config['mesh'])


	dim = m.spatial_dimension()
	block_size = config['block_size']

	if 'refine_level' in config and config['refine_level'] > 1:
		m = sfem.to_semistructured(config['refine_level'], m)

	fs = sfem.FunctionSpace(m, block_size)

	fun = sfem.Function(fs)
	op = sfem.create_op(fs, config["operator"])
	op.initialize()
	fun.add_operator(op)

	dirichlet_conditions = create_dirichlet_conditions(fs, config['dirichlet_conditions'])

	fun.add_dirichlet_conditions(dirichlet_conditions)

	out = fun.output()
	out.set_output_dir(config['output'])
	out.enable_AoS_to_SoA(True)

	x = np.zeros(fs.n_dofs(), dtype=real_t)
	rhs = np.zeros(fs.n_dofs(), dtype=real_t)
	c = np.zeros(fs.n_dofs(), dtype=real_t)
	# Buffers over the arrays: the library reads and writes them in place.
	xb, rhsb = sfem.view(x), sfem.view(rhs)

	cc = sfem.contact_conditions_from_file(fs, str(config['obstacle']))
	
	# Update problem with current solution and linearize
	cc.update(xb)
	
	op = sfem.make_op(fun, xb)
	g = np.zeros(fs.n_dofs(), dtype=real_t)
	cc.signed_distance_for_mesh_viz(xb, sfem.view(g))
	sfem.write(out, "gap", sfem.view(g))
	
	if config['solver'] == 'MPRGP':
		upper_bound = np.ones(fs.n_dofs(), dtype=real_t) * 1000
		cc.signed_distance_for_mesh_viz(xb, sfem.view(upper_bound))

		solver = sfem.MPRGP()
		solver.default_init()
		solver.set_atol(1e-12)
		solver.set_rtol(1e-6);
		solver.set_max_it(2000)
		solver.set_op(op)
		sfem.set_upper_bound(solver, sfem.view(upper_bound))
		sfem.apply_constraints(fun, xb)
		sfem.apply_constraints(fun, rhsb)
		sfem.apply(solver, rhsb, xb)
		
	elif config['solver'] == "SPMG":
		spmg = sfem.create_spmg(fun, execution_space)
		spmg.set_max_it(1)
		
		# upper_bound = np.zeros(cc.n_constrained_dofs(), dtype=real_t)
		# sfem.gradient(cc, x, upper_bound)
		# cc_op = cc.linear_constraints_op()
		# cc_op_t = cc.linear_constraints_op_transpose()
		# spmg.set_constraints_op(cc_op, cc_op_t)

		upper_bound = np.ones(fs.n_dofs(), dtype=real_t) * 1000
		cc.signed_distance_for_mesh_viz(xb, sfem.view(upper_bound))
		# print(np.min(upper_bound))

		sfem.set_upper_bound(spmg, sfem.view(upper_bound))
		
		sfem.apply_constraints(fun, xb)
		sfem.apply_constraints(fun, rhsb)
		# sfem.apply(spmg, rhs, x)

	elif config['solver'] == "SP":
		# The constraints, their bound at the current state and the normal products, as the
		# library builds them; the solver settings below are this case's.
		sp = sfem.create_shifted_penalty(fun, cc)

		linear_solver = sfem.ConjugateGradient()
		linear_solver.default_init()
		linear_solver.set_rtol(1e-3)
		linear_solver.set_verbose(False)
		linear_solver.set_max_it(100)
		sp.set_linear_solver(linear_solver)

		print(f'Constrained dofs {cc.n_constrained_dofs()}/{fs.n_dofs()}')

		sp.set_max_it(40)
		sp.set_max_inner_it(10)
		sp.set_penalty_param(2)
		sp.set_atol(1e-8)
		sp.set_damping(0.01)
		# sp.enable_steepest_descent(True)
		sfem.apply_constraints(fun, xb)
		sfem.apply_constraints(fun, rhsb)

		linear_solver.set_op(op)
		sfem.apply(linear_solver, rhsb, xb)
		sfem.apply(sp, rhsb, xb)

	sfem.write(out, "disp", xb)
	sfem.write(out, "rhs", rhsb)


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
