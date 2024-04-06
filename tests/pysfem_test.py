import pysfem as s
import numpy as np

def main():
	s.init()
	m = s.Mesh()
	m.read('/Users/patrickzulian/Desktop/code/sfem/examples/mesh')
	
	fs = s.FunctionSpace(m, 1)
	op = s.create_op(fs, "Laplacian")
	fun = s.Function(fs)
	fun.add_operator(op)

	x = np.linspace(0, 1, fs.n_dofs())
	y = np.zeros(fs.n_dofs())
	
	s.gradient(fun, x, y)
	print(y)

if __name__ == '__main__':
	main()
