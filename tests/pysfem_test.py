import pysfem as s
import numpy as np

idx_t = np.int32

def main():
	s.init()
	m = s.Mesh()

	path = '/Users/patrickzulian/Desktop/code/sfem/examples/mesh'
	m.read(path)

	sinlet = np.fromfile(f'{path}/sidesets_aos/sinlet.raw', dtype=idx_t)
	soutlet = np.fromfile(f'{path}/sidesets_aos/soutlet.raw', dtype=idx_t)

	fs = s.FunctionSpace(m, 1)
	fun = s.Function(fs)

	op = s.create_op(fs, "Laplacian")
	fun.add_operator(op)

	bc = s.DirichletConditions(fs)
	s.add_condition(bc, sinlet, 0, -1);
	s.add_condition(bc, soutlet, 0, 1);
	# fun.add_constraint(bc)

	x = np.linspace(0, 1, fs.n_dofs())
	y = np.zeros(fs.n_dofs())
	
	s.gradient(fun, x, y)
	s.apply_constraints(fun, x)

	s.report_solution(fun, x)

	print(y)

if __name__ == '__main__':
	main()
