import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def poisson_2d_matrix(n):
    """Create 2D Poisson matrix with 5-point stencil on n x n grid."""
    I = sp.eye(n)
    e = np.ones(n)
    T = sp.diags([e, -4*e, e], [-1, 0, 1], shape=(n, n))
    A = sp.kron(I, T) + sp.kron(sp.diags([e, e], [-1, 1], shape=(n, n)), I)

    A[0, :] = 0
    A[0, 0] = 1

    A[-1, :] = 0
    A[-1, -1] = 1
    
    return A.tocsr()

def block_jacobi_preconditioner(A, n, block_size):
    """Create block Jacobi preconditioner as a LinearOperator."""
    N = A.shape[0]
    indices = np.arange(N).reshape(n, n)
    blocks = []

    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            row_idx = indices[i:i+block_size, j:j+block_size].flatten()
            row_idx = row_idx[row_idx < N]
            block = A[row_idx, :][:, row_idx].tocsc()
            try:
                block_inv = spla.inv(block).tocsc()
            except:
                block_inv = sp.eye(len(row_idx)).tocsc()
            blocks.append((row_idx, block_inv))

    def apply(v):
        result = np.zeros_like(v)
        for idx, invB in blocks:
            result[idx] = invB @ v[idx]
        return result

    return spla.LinearOperator(matvec=apply, dtype=A.dtype, shape=A.shape)

# Parameters
n = 128                 # grid size
A = poisson_2d_matrix(n)
N = A.shape[0]
b = np.ones(N)
b[0] = 0
b[-1] = 0
x0 = np.zeros_like(b)

def intercept(x, residuals):
    r = A * x - b
    norm_r = np.linalg.norm(r)
    residuals.append(norm_r)


# Run CG with no preconditioner
print("Running CG with no preconditioner...")
residuals = []
x, info = spla.cg(A, b, x0=x0, atol=1e-8, maxiter=1000, callback=lambda r: intercept(r, residuals))

residuals = []
x, info = spla.cg(A, b, x0=x0, atol=1e-8, callback=lambda r: intercept(r, residuals))
plt.semilogy(residuals, label='No Preconditioner', marker='o')

# Scalar Jacobi
print("Running CG with scalar Jacobi preconditioner...")
M_diag = A.diagonal()
M_inv = spla.LinearOperator((N, N), matvec=lambda x: x / M_diag)

residuals = []
x, info = spla.cg(A, b, x0=x0, atol=1e-8, M=M_inv, callback=lambda r: intercept(r, residuals))
plt.semilogy(residuals, label='Scalar Jacobi')


# Block Jacobi with 2x2 blocks
print("Running CG with 2x2 block Jacobi preconditioner...")
block_size = 2
M_block = block_jacobi_preconditioner(A, n, block_size)
residuals = []
x, info = spla.cg(A, b, x0=x0, atol=1e-8, M=M_block, callback=lambda r: intercept(r, residuals))
plt.semilogy(residuals, label='Block Jacobi 2x2')


# Block Jacobi with 4x4 blocks
print("Running CG with 4x4 block Jacobi preconditioner...")
block_size = 4
M_block = block_jacobi_preconditioner(A, n, block_size)
residuals = []
x, info = spla.cg(A, b, x0=x0, atol=1e-8, M=M_block, callback=lambda r: intercept(r, residuals))
plt.semilogy(residuals, label='Block Jacobi 4x4')

# Block Jacobi with 8x8 blocks
print("Running CG with 8x8 block Jacobi preconditioner...")
block_size = 8
M_block = block_jacobi_preconditioner(A, n, block_size)
residuals = []
x, info = spla.cg(A, b, x0=x0, atol=1e-8, M=M_block, callback=lambda r: intercept(r, residuals))
plt.semilogy(residuals, label='Block Jacobi 8x8')


# Block Jacobi with 16x16 blocks
print("Running CG with 16x16 block Jacobi preconditioner...")
block_size = 16
M_block = block_jacobi_preconditioner(A, n, block_size)
residuals = []
x, info = spla.cg(A, b, x0=x0, atol=1e-8, M=M_block, callback=lambda r: intercept(r, residuals))
plt.semilogy(residuals, label='Block Jacobi 16x16')


# Block Jacobi with 32x32 blocks
# print("Running CG with 32x32 block Jacobi preconditioner...")
# block_size = 32
# M_block = block_jacobi_preconditioner(A, n, block_size)
# residuals = []
# x, info = spla.cg(A, b, x0=x0, atol=1e-8, M=M_block, callback=lambda r: intercept(r, residuals))
# plt.semilogy(residuals, label='Block Jacobi 32x32')


# Plotting
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Convergence of CG with Various Preconditioners')
plt.legend()
plt.grid(True)
plt.show()