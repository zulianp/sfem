#!/usr/bin/env python3

from sfem_codegen import *

class Field:
	def __init__(self, fe, coeff):
		self.fe = fe
		self.coeff = coeff

	def eval(self, q):
		fe = self.fe
		coeff = self.coeff

		rf = fe.fun(q)
		ncoeffs = len(coeff)

		fx = 0

		for i in range(0, fe.n_nodes()):
			fx += rf[i] * coeff[i]

		return fx

	def grad(self, q):
		fe = self.fe
		coeff = self.coeff

		rgrad = fe.grad(q)
		ncoeffs = len(coeff)

		grad_u = sp.Matrix(fe.spatial_dim(), 1, [0]*fe.spatial_dim())

		for i in range(0, fe.n_nodes()):
			for d in range(0, fe.spatial_dim()):
				grad_u[d] += rgrad[i][d] * coeff[i]

		return grad_u

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
