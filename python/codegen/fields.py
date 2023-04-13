#!/usr/bin/env python3

class VectorField:
	def __init__(self, fe, coeff_SoA):
		self.fe = fe
		self.coeff = coeff_SoA

	def eval(self, q):
		fe = self.fe
		coeff = self.coeff

		rf = fe.fun(q)
		ncoeffs = len(coeff)

		vfx = sp.Matrix(ncoeffs, 1, [0]*ncoeffs)

		for i in range(0, fe.n_nodes()):
			for d in range(0, ncoeffs):
				vfx[d] += rf[i] * coeff[d][i]

		return vfx
