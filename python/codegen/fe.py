import sympy as sp

class FE:
	def grad(self, p):
		fx = self.fun(p)
		dims, __ = p.shape
		nn = len(fx)

		g = [0] * nn

		for i in range(0, nn):
			gi = []

			for d in range(0, dims):
				gi.append(sp.diff(fx[i], p[d]))

			g[i] = sp.Matrix(dims, 1, gi)
		return g

	def tgrad(self, p):
		ret = []
		g = self.grad(p)

		for gi in g:
			for d1 in range(0, self.manifold_dim()):
				G = sp.Matrix(self.manifold_dim(), self.spatial_dim(), [0] * self.manifold_dim() * self.spatial_dim() )

				for d2 in range(0, self.spatial_dim()):
					G[d1, d2] = gi[d2]

				ret.append(G)
		return ret

	def physical_tgrad(self, p):
		ret = []
		g = self.tgrad(p)
		J_inv = self.symbol_jacobian_inverse()

		for gi in g:
			ret.append(J_inv.T * gi)

		return ret


	def physical_grad(self, p):
		fx = self.fun(p)
		dims, __ = p.shape
		nn = len(fx)

		g = [0] * nn

		J_inv = self.symbol_jacobian_inverse()

		for i in range(0, nn):
			gi = []

			for d in range(0, dims):
				gi.append(sp.diff(fx[i], p[d]))

			gi = sp.Matrix(dims, 1, gi)
			# gi = J_inv * gi
			gi = J_inv.T * gi # ATTENTION!
			g[i] = gi

		return g

	def symbol_jacobian_inverse(self):
		rows = self.manifold_dim()
		cols = self.spatial_dim()

		sls = []

		for i in range(0, rows):
			for j in range(0, cols):
				var = sp.symbols(f'jac_inv[{i*cols + j}*stride]')
				sls.append(var)
		return sp.Matrix(rows, cols, sls)
