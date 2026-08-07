#!/usr/bin/env python3

import numpy as np
from numba import njit
from scipy.linalg import lu_factor, lu_solve


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


# @njit(fastmath=True, boundscheck=False, nogil=True)
# def penalty_jacobi_step(A, I, x, b, ub, penalty, shift):
#     rows = A.shape[0]
#     x_old = x.copy()

#     temp = -ub + shift / penalty

#     for i in range(0, rows):
#         ri = b[i]
#         for j in range(0, rows):
#             ri -= A[i, j] * x_old[j]

#         x[i] += ri / A[i, i]

#         Dmu = 0.0 if (x[i] + temp[i]) < 0 else I[i, i] * penalty

#         # Nonlinear Jacobi (penalty method)
#         ri = I[i, i] * penalty * max(0.0, x[i] + temp[i])
#         x[i] -= ri / (A[i, i] + Dmu)


# @njit(fastmath=True, boundscheck=False, nogil=True)
# def penalty_jacobi_step(A, I, x, b, ub, penalty, shift):
#     rows = A.shape[0]
#     x_old = x.copy()

#     temp = -ub + shift / penalty

#     for i in range(0, rows):
#         ri = b[i]
#         for j in range(0, rows):
#             ri -= A[i, j] * x_old[j]

#         x[i] += ri / A[i, i]

#         # Projected Jacobi
#         x[i] = min(ub[i], x[i])

omega = 1
# omega = 0.5

@njit(fastmath=True, boundscheck=False, nogil=True)
def penalty_jacobi_step(A, I, x, b, ub, penalty, shift):
    rows = A.shape[0]
    x_old = x.copy()

    temp = -ub + shift / penalty

    for i in range(0, rows):
        ri = b[i] - I[i, i] * penalty * max(0.0, x_old[i] + temp[i])
        Dmu = 0.0 if (x_old[i] + temp[i]) < 0 else I[i, i] * penalty

        for j in range(0, rows):
            ri -= A[i, j] * x_old[j]

        x[i] += omega * ri / (A[i, i] + Dmu)


@njit(fastmath=True, boundscheck=False, nogil=True)
def jacobi_step(A, x, b):
    rows = A.shape[0]
    x_old = x.copy()

    r = b

    for i in range(0, rows):
        ri = r[i]

        for j in range(0, rows):
            ri -= A[i, j] * x_old[j]

        x[i] += omega * ri / A[i, i]


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


n = int(50)
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
# b = np.ones(n) * h
b[0] = 0
b[n - 1] = 0

x = np.zeros(n)

# Solve A x = b
x = np.linalg.solve(A, b)
ub = np.ones(n) * 0.075

penalty = 1e12 / h

xc = x.copy()

shift = np.zeros(n)

solutions = [xc.copy()]
multipliers = [np.zeros(xc.shape)]

print(f'h={h}')


I_diag = np.diag(I)
A_factor = lu_factor(A, check_finite=False)
delta_old = shift.copy()
lambda_hat_old = shift.copy()

aug_damping = h 
njacs = 10
for i in range(0, 4000):

    if False:
    # if True:
    # penalty_gs_step(A, I, xc, b, ub, penalty, shift)

        for j in range(njacs):
            penalty_jacobi_step(A, I, xc, b, ub, penalty, shift)
            # penalty_gs_step(A, I, xc, b, ub, penalty, shift)
            shift = penalty * np.maximum(0.0, aug_damping * (xc - ub) + shift / penalty)

        norm_g = np.linalg.norm(penalty_gradient(A, I, xc, b, ub, penalty, shift))

    # if (i + 1) % 10 == 0:
    # shift = penalty * np.maximum(0.0, (xc - ub) + shift / penalty)

    else: # Crossed secant method
        xc[:] = lu_solve(A_factor, b - I_diag * shift, check_finite=False)

        # for j in range(njacs):
        #     jacobi_step(A, xc, b - I_diag * shift)

        # Algorithm 5 from the paper, applied to the unprojected G.
        lambda_hat = penalty * (xc - ub)
        delta = lambda_hat - shift

        if i >= 1:
            diff_delta = delta - delta_old
            denom = np.dot(diff_delta, diff_delta)
            if denom > 1e-30:
                beta = np.dot(lambda_hat - lambda_hat_old, diff_delta) / denom
                shift = lambda_hat - beta * delta
            else:
                shift = lambda_hat
        else:
            shift = lambda_hat

        shift = np.maximum(0.0, shift)

        delta_old = delta.copy()
        lambda_hat_old = lambda_hat.copy()

        norm_g = np.linalg.norm(A @ xc - b + I_diag * shift)

    solutions.append(xc.copy())
    multipliers.append(shift.copy())

    
    norm_pen = np.linalg.norm(np.maximum(0.0, xc - ub))
    norm_shift = np.linalg.norm(shift)
    print(
        f"norm_g[{i*njacs}]: {norm_g}, norm_pen: {norm_pen}, norm_shift: {norm_shift}"
    )

    if norm_g < 1e-12:
        break


# Plot x
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.plot(x)
plt.plot(ub)
xc_line = plt.plot(solutions[0])[0]
multipliers_line = plt.plot(multipliers[0])[0]
plt.legend(["x",  "ub", "xc", "multipliers (rescaled)"])
plt.xlabel("Index")
plt.ylabel("Value")


def update(frame):
    xc_line.set_ydata(solutions[frame])
    multipliers_line.set_ydata(0.01 * multipliers[frame])
    return (xc_line, multipliers_line)


ani = FuncAnimation(
    plt.gcf(),
    update,
    frames=len(solutions),
    interval=30,
    blit=True,
    repeat=False,
)
plt.show()
