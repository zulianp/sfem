#!/usr/bin/env python3

from sfem_codegen import *
from tet10 import *
from fields import *

class CGradOp:
	def __init__(self, field, q, q_evals):
		self.field = field
		self.q = q
		self.q_evals = q_evals

	def apply(self):
		field = self.field 
		q = self.q
		q_evals = self.q_evals 
		fe = field.fe

		expr = []

		xnames = ['x', 'y', 'z']

		J_inv = fe.jacobian_inverse(q)
		g = J_inv.T * field.grad(q)

		qidx = 0
		for qeval in q_evals:
			geval = [0]*fe.spatial_dim()

			# Substitute symbolic with nodal evaluations of interpolated function
			for d1 in range(0, fe.spatial_dim()):
				geval[d1] = g[d1]

				for d2 in range(0, fe.manifold_dim()):
					geval[d1] = geval[d1].subs(q[d2], qeval[d2])
			
			for d in range(0, fe.spatial_dim()):
				dfdx = sp.symbols(f'dfd{xnames[d]}[{qidx}]')
				expr.append(ast.Assignment(dfdx, geval[d]))

			qidx += 1

		return expr


def main():
	field = Field(Tet10(), coeffs('f', 10))
	q = vec3(qx, qy, qz)
	# P1 nodes
	q_evals = [ vec3(0, 0, 0),  vec3(1, 0, 0),  vec3(0, 1, 0),  vec3(0, 0, 1)]
	op = CGradOp(field, q, q_evals)

	c_code(op.apply())

if __name__ == '__main__':
	main()