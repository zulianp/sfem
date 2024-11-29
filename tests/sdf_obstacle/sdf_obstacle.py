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

		idx = np.unique(np.fromfile(nodeset, dtype=idx_t))
		if isinstance(component, list):
			for k in range(0, len(component)):
				sfem.add_condition(dirichlet_conditions, idx, component[k], value[k])
		else:
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
	fun.set_output_dir(config['output'])

	x = np.zeros(fs.n_dofs())
	g = np.zeros(fs.n_dofs())
	c = np.zeros(fs.n_dofs())
	
	cg = sfem.ConjugateGradient()
	cg.default_init()
	cg.set_max_it(400)

	lop = sfem.make_op(fun, x)
	cg.set_op(lop)

	sfem.apply_constraints(fun, x)
	sfem.gradient(fun, x, g)

	print(f'Solving system with {fs.n_dofs()} dofs')
	sfem.apply(cg, g, c)

	x -= c

	sfem.report_solution(fun, x)

	# TODO
	# cc = sfem.contact_conditions_from_file(fs, str(config['obstacle']))

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
