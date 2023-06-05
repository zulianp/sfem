import sympy as sp

class FE:
	SoA = True
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
		g = self.grad(p)
		dim = self.manifold_dim() 
		tensor_size = self.manifold_dim() * self.spatial_dim() 
		ndofs = len(g) * dim
		nnodes = len(g)

		ret = [0] * ndofs
		for i in range(0, nnodes):
			gi = g[i]
			for d1 in range(0, self.manifold_dim()):
				G = sp.Matrix(self.manifold_dim(), self.spatial_dim(), [0] * tensor_size)
				for d2 in range(0, self.spatial_dim()):
					G[d1, d2] = gi[d2]
				
				if self.SoA:
					ret[d1 * nnodes + i] = G
				else:
					ret[i*dim + d1] = G
					
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
