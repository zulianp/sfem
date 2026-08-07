#!/usr/bin/env python3

import numpy as np


def zero_mean_solve(A, P, b):
    n = A.shape[0]
    m = P.shape[0]
    K = np.empty((n + m, n + m), dtype=np.result_type(A, P, b))
    K[:n, :n] = A
    K[:n, n:] = P.T
    K[n:, :n] = P
    K[n:, n:] = 0

    rhs = np.empty((n + m, b.shape[1]), dtype=np.result_type(A, P, b))
    rhs[:n] = b
    rhs[n:] = 0

    return np.linalg.solve(K, rhs)[:n]


def jacobi_zero_mean_solve(A, P, b, x, max_iter=50000, tol=1e-10):
    dinv = 1.0 / np.diag(A)
    dinv_col = dinv[:, None]
    constraint_mass = (P @ (dinv_col * P.T))[0]
    x -= P.T @ (P @ x) / (P @ P.T)[0]
    solutions = [x.copy()]

    for i in range(max_iter):
        r = b - A @ x
        lagrange_multiplier = (P @ (dinv_col * r)) / constraint_mass
        dx = dinv_col * (r - P.T @ lagrange_multiplier)
        x += dx

        if i % 500 == 0:
            solutions.append(x.copy())

        if np.linalg.norm(dx) <= tol * max(1.0, np.linalg.norm(x)):
            solutions.append(x.copy())
            return x, solutions, i + 1

    solutions.append(x.copy())
    return x, solutions, max_iter


n = int(100)
h = 1.0 / (n - 1)
A = np.zeros((n, n))

A[0, 0] = 1
A[0, 1] = -1
A[n - 1, n - 1] = 1
A[n - 1, n - 2] = -1


for i in range(1, n - 1):
    A[i, i - 1] = -1
    A[i, i + 1] = -1
    A[i, i] = 2

A *= 1 / (h)
P = np.ones((1, n)) * h

# b = np.ones(n) * 2 * h
b = np.zeros((n, 1))
b[0] = 1 * h
# b[n - 1] = -1
b[n - 1] = -1 * h
# b[0] = 0
# b[n - 1] = 0

x = np.zeros((n, 1))

xc = x.copy()

reference = zero_mean_solve(A, P, b)
xc, solutions, iterations = jacobi_zero_mean_solve(A, P, b, xc)

print((P @ xc)[0, 0])
print(iterations)
print(np.linalg.norm(xc - reference))


# Plot x
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plt.plot(np.linspace(0, 1, x.shape[0]), reference)
xc_line = plt.plot(np.linspace(0, 1, x.shape[0]), solutions[0])[0]
plt.legend(["reference", "jacobi"])
plt.xlabel("Index")
plt.ylabel("Value")


def update(frame):
    xc_line.set_ydata(solutions[frame])
    return (xc_line,)


ani = FuncAnimation(
    plt.gcf(),
    update,
    frames=len(solutions),
    interval=30,
    blit=True,
    repeat=False,
)
plt.show()
