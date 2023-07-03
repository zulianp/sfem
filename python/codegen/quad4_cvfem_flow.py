#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt

A = sp.Matrix(2, 2, [0] * 4)

K = sp.symbols('K')



p0 = sp.Matrix([x0, y0])
p1 = sp.Matrix([x1, y1])
p2 = sp.Matrix([x2, y2])
p3 = sp.Matrix([x3, y3])

def perp(e):
 	return sp.Matrix(2, 1, [-e[1], e[0]])
 	# return sp.Matrix(2, 1, [e[1], -e[0]])

def shape_fun(x, y):
	return [(1 - x) * (1 - y), x * (1 - y) , x  * y, (1 - x)  * y ]

rf = shape_fun(qx, qy)

def p(x, y):
	f = shape_fun(x, y)
	return f[0] * p0 + f[1] * p1 + f[2] * p2 + f[3] * p3

# Sub-parametric
# A2 = sp.Matrix(2, 2, [
# 	 x1 - x0, x2 - x0,
# 	 y1 - y0, y2 - y0
# ])

# Axis-aligend
# A2 = sp.Matrix(2, 2, [
# 	 x1 - x0, 0,
# 	 0 , y2 - y0
# ])

# Iso-parametric
Tp = p(qx, qy)
A2 = sp.Matrix(2, 2, 
	[ sp.diff(Tp[0], qx), sp.diff(Tp[0], qy),
	  sp.diff(Tp[1], qx), sp.diff(Tp[1], qy) ])

A2inv = inv2(A2)

half = sp.Rational(1, 2)
e = [p(half, 0), p(1, half), p(half, 1), p(0, half)]
b = p(half, half)

g = [0, 0, 0, 0]
k = sp.symbols('k')

dV = det2(A2) / (2 * 3)

for i in range(0, 4):
	gi = sp.Matrix(2, 1, [sp.diff(rf[i], qx), sp.diff(rf[i], qy)])
	# gi = A2inv.T * gi
	gi = A2inv.T * gi
	g[i] = sp.simplify(gi)

px = [0, 1, 1.4, 0]
py = [0, 0, 1.2, 1]

pbx = np.sum(px) / 4.
pby = np.sum(py) / 4.

def subs_point(expr):
	expr = expr.subs(x0, px[0])
	expr = expr.subs(y0, py[0])
	expr = expr.subs(x1, px[1])
	expr = expr.subs(y1, py[1])
	expr = expr.subs(x2, px[2])
	expr = expr.subs(y2, py[2])
	expr = expr.subs(x3, px[3])
	expr = expr.subs(y3, py[3])
	return expr

fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 0.8, 0.8], frameon=False)
ax.scatter(px, py)
ax.scatter(pbx, pby)
ax.plot([px[0], px[1], px[2], px[3], px[0]], [py[0], py[1], py[2], py[3], py[0]])

expr = []
for i in range(0, 4):
	# Normals scaled by the element area
	s1 = b - e[i]
	s2 = e[(i - 1 + 4) % 4] - b

	midp = [ e[i] ,e[(i - 1 + 4) % 4] ]
	segk = [(b + e[i])/2, (e[(i - 1 + 4) % 4] + b)/2]
	dS = [ perp(s1), perp(s2) ]

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

	for j in range(0, 4):
		integr = 0

		# Integrate over CV surface integration points
		for k in range(0, 2):
			dGdS = 0
			for d in range(0, 2): 
				gik = g[j][d].subs(qx, midp[k][0]).subs(qy, midp[k][1])
				dGdS += dS[k][d] * gik

			integr += dGdS

		if True:
			ss = subs_point(integr)
			print(f'{i}, {j}) {ss}')

		var = sp.symbols(f'element_matrix[{i*4+j}]')
		expr.append(ast.Assignment(var, integr))

c_code(expr)

# plt.show()

