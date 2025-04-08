#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt


def cross(a, b):
    s = sp.Matrix(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )
    return s


def subspoints(ss):
    scaling = 1
    ss = ss.subs(x0, 0)
    ss = ss.subs(y0, 0)
    ss = ss.subs(z0, 0)

    ss = ss.subs(x1, scaling)
    ss = ss.subs(y1, 0)
    ss = ss.subs(z1, 0)

    ss = ss.subs(x2, 0)
    ss = ss.subs(y2, scaling)
    ss = ss.subs(z2, 0)

    ss = ss.subs(x3, 0)
    ss = ss.subs(y3, 0)
    ss = ss.subs(z3, scaling)
    return ss


def plot_normals(b, dS):
    import matplotlib.pyplot as plt
    import numpy as np

    # Define data points
    X = np.zeros(len(dS))
    Y = np.zeros(len(dS))
    Z = np.zeros(len(dS))

    # Define vector components (modify these for different directions)
    U = np.zeros(len(dS))
    V = np.zeros(len(dS))
    W = np.zeros(len(dS))

    for i in range(0, len(dS)):
        d = subspoints(dS[i])
        U[i] = d[0] * 10
        V[i] = d[1] * 10
        W[i] = d[2] * 10

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot the quiver with options
    ax.quiver(
        X, Y, Z, U, V, W, normalize=True, color="red"
    )  # Normalize for consistent size

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Quiver Plot")

    plt.show()


A = sp.Matrix(3, 3, [0] * 9)

K = sp.symbols("K")

Jac = sp.Matrix(
    3,
    3,
    [
        x1 - x0,
        x2 - x0,
        x3 - x0,
        y1 - y0,
        y2 - y0,
        y3 - y0,
        z1 - z0,
        z2 - z0,
        z3 - z0,
    ],
)

JacInv = inverse(Jac)

a = sp.Matrix([x0, y0, z0])
b = sp.Matrix([x1, y1, z1])
c = sp.Matrix([x2, y2, z2])
d = sp.Matrix([x3, y3, z3])

rf = [1 - qx - qy - qz, qx, qy, qz]


def p(x, y, z):
    f = [1 - x - y - z, x, y, z]
    return f[0] * a + f[1] * b + f[2] * c + f[3] * d


half = sp.Rational(1, 2)
centroid = p(half, half, half)

# print(f'{centroid[0]}\n{centroid[1]}\n{centroid[2]}')

dS = [0] * 6

# A -> B
dS[0] = sp.Rational(1, 24) * (
    cross(a, c) - cross(a, d) + cross(b, c) - cross(b, d) + 2 * cross(c, d)
)

# A -> C
dS[1] = sp.Rational(1, 24) * (
    -cross(a, b) + cross(a, d) + cross(b, c) - 2 * cross(b, d) + cross(c, d)
)

# A -> D
dS[2] = sp.Rational(1, 24) * (
    cross(a, b) - cross(a, c) + 2 * cross(b, c) - cross(b, d) + cross(c, d)
)

# B -> C
dS[3] = sp.Rational(1, 24) * (
    -cross(a, b) - cross(a, c) + 2 * cross(a, d) - cross(b, d) - cross(c, d)
)

# B -> D
dS[4] = sp.Rational(1, 24) * (
    cross(a, b) - 2 * cross(a, c) + cross(a, d) + cross(b, c) - cross(c, d)
)

# C -> D
dS[5] = sp.Rational(1, 24) * (
    2 * cross(a, b) - cross(a, c) - cross(a, d) + cross(b, c) + cross(b, d)
)

# plot_normals(centroid, dS)

# for k in range(0, len(dS)):
# 	dd = dS[k]
# 	print(f'{k}) {subspoints(dd)}')

subs = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]

signs = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [1, 1, 1]]

# V_ABCD = sp.Rational(1, 6) * dot3((d - a), (cross(b - a, c - a)))
# print(V_ABCD)

V_ABCD = dot3(sp.Rational(1, 6) * (d - a), cross(b - a, c - a))
print(subspoints(V_ABCD))
# V_ABCD = det3(Jac)
# Divide by reference volume 1/6
# and number of sub-volumes (4)
# dV = V_ABCD / (6 * 4)

g = [0, 0, 0, 0]

for i in range(0, 4):
    gi = sp.Matrix(3, 1, [sp.diff(rf[i], qx), sp.diff(rf[i], qy), sp.diff(rf[i], qz)])
    gi = JacInv.T * gi
    g[i] = sp.simplify(gi)

# Following the paper
# g[0] = sp.Rational(1, 6) / V_ABCD * cross(d - b, c - b)
# g[1] = sp.Rational(1, 6) / V_ABCD * cross(c - a, d - a)
# g[2] = sp.Rational(1, 6) / V_ABCD * cross(d - a, b - a)
# g[3] = sp.Rational(1, 6) / V_ABCD * cross(b - a, c - a)

expr = []
for i in range(0, 4):
    for j in range(0, 4):

        # Integrate over CV surface integration points
        dA = sp.zeros(3, 1)
        for l in range(0, 3):
            k = subs[i][l]
            s = signs[i][l]
            dA = dA + dS[k] * s

        integr = dot3(dA, g[j])

        if True:
            # if False:
            ss = subspoints(integr)
            ssdA = subspoints(dA)

            if j == 0:
                print("\n")
                # print(ssdA)
            print(ss, end=" ")

        # integr = sp.simplify(integr)
        var = sp.symbols(f"element_matrix[{i*4+j}]")
        expr.append(ast.Assignment(var, integr))
print("\n")
c_code(expr)
