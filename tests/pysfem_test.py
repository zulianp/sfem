import pysfem as s
import numpy as np

def main():
	s.init()
	m = s.Mesh()
	m.read('/Users/patrickzulian/Desktop/code/sfem/examples/mesh')

	fs = s.FunctionSpace(m, 1)
	fun = s.Function(fs)

	op = s.create_op(fs, "Laplacian")
	fun.add_operator(op)

	bc = s.DirichletConditions(fs)
	# fun.add_constraint(bc)

	x = np.linspace(0, 1, fs.n_dofs())
	y = np.zeros(fs.n_dofs())
	
	s.gradient(fun, x, y)
	s.apply_constraints(fun, x)

	s.report_solution(fun, x)

	print(y)

if __name__ == '__main__':
	main()
