#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from symbolic_fe import *
from phase_field_for_fracture_op_base import PhaseFieldForFractureOpBase

from time import perf_counter

# import pdb

# AT2
def g(c):
	return (1 - c)**2

def omega(c):
	return c**2

def comega(c):
 	return 2

class PhaseFieldForFractureOp(PhaseFieldForFractureOpBase):
	def __init__(self, fe):
		super().__init__(fe)

	def energy(self, c, gradc, gradu):
		Gc, ls = sp.symbols('Gc ls', real=True)
		mu, lmbda = sp.symbols('mu lambda', real=True)
		epsu = (gradu + gradu.T) / 2
		
		# strain energy function
		eu = mu * inner(epsu, epsu) + (lmbda/2) * (tr(epsu) * tr(epsu))
		ec = (Gc / comega(c)) * (omega(c)/ls + ls * inner(gradc, gradc))
		e = g(c) * eu + ec

		self.fracture_energy = ec
		self.elastic_energy  = g(c) * eu

		self.mu = mu
		self.lmbda = lmbda
		self.Gc = Gc
		self.ls = ls

		self.params = [mu, lmbda, Gc, ls]
		return e

	def hessian_check(self):
		H = self.integr_hessian
		rows, cols = H.shape

		A = sp.Matrix(rows, cols, [0] * (rows * cols))
		for i in range(0, rows):
			for j in range(0, cols):
				integr = H[i, j]

				coord = 1
				integr = integr.subs(self.mu, 0.5)
				integr = integr.subs(self.lmbda, 1)
				integr = integr.subs(self.ls, 1)
				integr = integr.subs(self.Gc, 1)

				for c in self.c:
					integr = integr.subs(c, 0)

				for u in self.disp:
					integr = integr.subs(u, 0)

				# coord = 1.
				# integr = integr.subs(self.mu, sp.Rational(1, 2))
				# integr = integr.subs(self.lmbda, 1)
				# integr = integr.subs(self.mu, 1)
				# integr = integr.subs(self.lmbda, 0)

				integr = integr.subs(x0, 0)
				integr = integr.subs(y0, 0)
				
				integr = integr.subs(x1, coord)
				integr = integr.subs(y1, 0)

				if rows == 8:
					integr = integr.subs(x2, coord)
					integr = integr.subs(y2, coord)

					integr = integr.subs(x3, 0)
					integr = integr.subs(y3, coord)
				else:
					integr = integr.subs(x2, 0)
					integr = integr.subs(y2, coord)
				
				A[i, j] = integr

		S = A.T - A
		row_sum = sp.Matrix(rows, 1, [0] * rows)
		for i in range(0, rows):
			for j in range(0, cols):
				row_sum[i] += A[i, j]

		for i in range(0, rows):
			line = ""
			for j in range(0, rows):
				line += "%.5g " % A[i,j]
			c_log(line)

		c_log(S)
		c_log(row_sum)

def main():
	start = perf_counter()
	# fe = AxisAlignedQuad4()
	# fe = SymbolicFE2D()
	# fe = SymbolicFE2D()

	# fe = Tri3()
	# fe = Tri6()
	# fe = Tet4()
	# fe = Tet10()
	
	# op = PhaseFieldForFractureOp(fe)
	# op.hessian_check()

	# op.generate_c_code()


	PhaseFieldForFractureOp(SymbolicFE2D()).generate_c_code()
	PhaseFieldForFractureOp(SymbolicFE3D()).generate_c_code()

	# if False:
	# if True:
		# c_log("--------------------------")
		# c_log("value")
		# c_log("--------------------------")
		# c_code(op.value())

		# c_log("--------------------------")
		# c_log("gradient")
		# c_log("--------------------------")
		# c_code(op.gradient())

		# c_log("--------------------------")
		# c_log("hessian")	
		# c_log("--------------------------")
		# c_code(op.hessian())

		# c_log("--------------------------")
		# c_log("gradient_u")	
		# c_log("--------------------------")
		# c_code(op.gradient_u())

		# c_log("--------------------------")
		# c_log("gradient_c")	
		# c_log("--------------------------")
		# c_code(op.gradient_c())

		# c_log("--------------------------")
		# c_log("hessian_uu")	
		# c_log("--------------------------")
		# c_code(op.hessian_uu())

		# c_log("--------------------------")
		# c_log("hessian_cc")	
		# c_log("--------------------------")
		# c_code(op.hessian_cc())

		# c_log("--------------------------")
		# c_log("hessian_cu")	
		# c_log("--------------------------")
		# c_code(op.hessian_cu())

		# c_log("--------------------------")
		# c_log("hessian_uc")	
		# c_log("--------------------------")
		# c_code(op.hessian_uc())


		# c_log("--------------------------")
		# c_log("apply")	
		# c_log("--------------------------")
		# c_code(op.apply())

	stop = perf_counter()
	console.print(f'Overall: {stop - start} seconds')


if __name__ == '__main__':
	main()