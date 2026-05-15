#!/usr/bin/env python3

import numpy as np
from numba import njit


# Nonlinear Gauss-Seidel (penalty method) converges to Projected Gauss-Seidel
# for penalty -> inf


@njit(fastmath=True, boundscheck=False, nogil=True)
def penalty_gs_step(A, I, x, b, ub, penalty, shift):
    rows = A.shape[0]

    temp = -ub + shift / penalty

    for i in range(0, rows):
        ri = b[i]
        for j in range(0, rows):
            ri -= A[i, j] * x[j]

        x[i] += ri / A[i, i]

        Dmu = 0.0 if (x[i] + temp[i]) < 0 else I[i, i] * penalty

        # Nonlinear Gauss-Seidel (penalty method)
        ri = I[i, i] * penalty * max(0.0, x[i] + temp[i])
        x[i] -= ri / (A[i, i] + Dmu)

        # Projected Gauss-Seidel
        # x[i] = min(ub[i], x[i])


def penalty_gradient(A, I, x, b, ub, penalty, shift):
    rows = A.shape[0]

    grad = np.zeros(rows)

    for i in range(0, rows):
        for j in range(0, rows):
            grad[i] -= A[i, j] * x[j]

        grad[i] += b[i] - I[i, i] * penalty * max(
            0.0, x[i] - ub[i] + shift[i] / penalty
        )

    return grad


n = int(100)
h = 1.0 / (n - 1)
A = np.zeros((n, n))
A[0, 0] = 1
A[n - 1, n - 1] = 1


for i in range(1, n - 1):
    A[i, i - 1] = -1
    A[i, i + 1] = -1
    A[i, i] = 2

A *= 1 / (h)
I = np.eye(n) * h

b = np.ones(n) * 2 * h
b[0] = 0
b[n - 1] = 0

x = np.zeros(n)

# Solve A x = b
x = np.linalg.solve(A, b)
ub = np.ones(n) * 0.2

penalty = 1 / h

xc = x.copy()

shift = np.zeros(n)

# TODO: collect all solutions then show a movie instead of the plot

for i in range(1000):
    penalty_gs_step(A, I, xc, b, ub, penalty, shift)
    shift = penalty * np.maximum(0.0, xc - ub + shift / penalty)

    g = np.linalg.norm(penalty_gradient(A, I, xc, b, ub, penalty, shift))
    norm_g = np.linalg.norm(g)
    norm_pen = np.linalg.norm(np.maximum(0.0, xc - ub))
    norm_shift = np.linalg.norm(shift)
    print(f"norm_g[{i}]: {norm_g}, norm_pen: {norm_pen}, norm_shift: {norm_shift}")

    if norm_g < 1e-6:
        break


# Plot x
import matplotlib.pyplot as plt

plt.plot(x)
plt.plot(xc)
plt.plot(ub)
plt.legend(["x", "xc", "ub"])
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
