#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt

A = sp.Matrix(2, 2, [0] * 4)

K = sp.symbols("K")

A2 = sp.Matrix(2, 2, [x1 - x0, x2 - x0, y1 - y0, y2 - y0])

A2inv = inv2(A2)

p0 = sp.Matrix([x0, y0])
p1 = sp.Matrix([x1, y1])
p2 = sp.Matrix([x2, y2])


def perp(e):
    return sp.Matrix(2, 1, [-e[1], e[0]])
    # return sp.Matrix(2, 1, [e[1], -e[0]])


rf = [1 - qx - qy, qx, qy]


def p(x, y):
    f = [1 - x - y, x, y]
    return f[0] * p0 + f[1] * p1 + f[2] * p2


third = sp.Rational(1, 3)
b = p(third, third)

half = sp.Rational(1, 2)
e = [p(half, 0), p(half, half), p(0, half)]

g = [0, 0, 0]
k = sp.symbols("k")

dV = det2(A2) / (2 * 3)

for i in range(0, 3):
    gi = sp.Matrix(2, 1, [sp.diff(rf[i], qx), sp.diff(rf[i], qy)])
    gi = A2inv.T * gi
    g[i] = sp.simplify(gi)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)

px = [0, 1, 0]
py = [0, 0, 1]
pbx = np.sum(px) / 3.0
pby = np.sum(py) / 3.0

ax.scatter(px, py)
ax.scatter(pbx, pby)
ax.plot([0, 1, 0, 0], [0, 0, 1, 0])


def subs_point(expr):
    expr = expr.subs(x0, px[0])
    expr = expr.subs(y0, py[0])
    expr = expr.subs(x1, px[1])
    expr = expr.subs(y1, py[1])
    expr = expr.subs(x2, px[2])
    expr = expr.subs(y2, py[2])
    return expr


expr = []
for i in range(0, 3):
    # Normals scaled by the element area
    s1 = b - e[i]
    s2 = e[(i - 1 + 3) % 3] - b

    midp = [e[i], e[(i - 1 + 3) % 3]]
    segk = [(b + e[i]) / 2, (e[(i - 1 + 3) % 3] + b) / 2]
    dS = [perp(s1), perp(s2)]

    # if False:
    if True:
        # if i == 2:
        for k in range(0, 2):
            ss = subs_point(dS[k])
            es = subs_point(segk[k])
            ms = subs_point(midp[k])

            ax.scatter(es[0], es[1])
            ax.scatter(ms[0], ms[1])
            ax.plot([ms[0], es[0], pbx], [ms[1], es[1], pby])
            ax.quiver(es[0], es[1], float(ss[0]), float(ss[1]))

    for j in range(0, 3):
        integr = 0

        # Integrate over CV surface integration points
        for k in range(0, 2):
            dGdS = 0
            for d in range(0, 2):
                dGdS += dS[k][d] * g[j][d]
            integr += dGdS
            # * dV

        if True:
            ss = integr
            ss = ss.subs(x0, 0)
            ss = ss.subs(y0, 0)
            ss = ss.subs(x1, 1)
            ss = ss.subs(y1, 0)
            ss = ss.subs(x2, 0)
            ss = ss.subs(y2, 1)
            print(f"{i}, {j}) {ss}")

        var = sp.symbols(f"element_matrix[{i*3+j}]")
        expr.append(ast.Assignment(var, integr))

c_code(expr)

plt.show()

# test = []

# for e in expr:
# 	ss = e
# 	ss = ss.subs(x0, 0)
# 	ss = ss.subs(y0, 0)
# 	ss = ss.subs(x1, 1)
# 	ss = ss.subs(y1, 0)
# 	ss = ss.subs(x2, 0)
# 	ss = ss.subs(y2, 1)

# 	test.append(ss)

# c_code(test)

# -1 -1
#  1  0
#  0  1
