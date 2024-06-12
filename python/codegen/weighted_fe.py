from fe import FE
from sfem_codegen import *

class WeightedFE(FE):
	def __init__(self, fe, weights, prefix_ = "WeightedFE"):
		self.fe = fe
		self.weights = weights
		self.prefix_  = prefix_

	def fun(self, p):
		fe = self.fe
		weights = self.weights

		nn = fe.n_nodes()
		f = fe.fun(p)
		ret = [0] * nn

		for i in range(0, nn):
			for j in range(0, nn):
				ret[i] += f[j] * weights[i, j]
		return ret

	def coords_sub_parametric(self):
		return self.fe.coords_sub_parametric()

	def name(self):
		return f'{self.prefix_}({self.fe.name()})'
				
	def n_nodes(self):
		return self.fe.n_nodes()

	def manifold_dim(self):
		return self.fe.manifold_dim()

	def spatial_dim(self):
		return self.fe.spatial_dim()

	def integrate(self, q, expr):
		return self.fe.integrate(q, expr)
	
	def jacobian(self, q):
		return self.fe.jacobian(q)

	def jacobian_inverse(self, q):
		return self.fe.jacobian_inverse(q)

	def jacobian_determinant(self, q):
		return self.fe.jacobian_determinant(q)

	def transform(self, q):
		return self.fe.transform(q)

	def inverse_transform(self, p):
		return self.fe.inverse_transform(p)

	def measure(self, q):
		return self.fe.measure(q)

	def reference_measure(self):
		return self.fe.reference_measure()
