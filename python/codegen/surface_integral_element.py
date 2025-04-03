#!/usr/bin/env python3

from sfem_codegen import *
from hex8 import *
from quad4 import *
from functools import reduce

# Hex8 coordinates
x = coeffs("x", 8)
y = coeffs("y", 8)
z = coeffs("z", 8)

# Volumetric element
hex8 = Hex8()

# Surface element
quad4 = QuadShell4()

dS = [0]*6

q3 = hex8.quadrature_point()
q2 = quad4.quadrature_point()

# Faces
fidx = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [3, 2, 1, 0], [4, 5, 6, 7]]

tpl = """
static SFEM_INLINE void hex8_surface_element_{FACE}(
	const scalar_t *const x,
	const scalar_t *const y, 
	const scalar_t *const z, 
	scalar_t * const SFEM_RESTRICT surface_element)
{{
{CODE}
}}
"""

for i in range(0, len(fidx)):
	f = fidx[i]

	px = [ x[j] for j in f ]
	py = [ y[j] for j in f ]
	pz = [ z[j] for j in f ]

	s = quad4.fun(q2)

	p = [	
		reduce(lambda x, y,: x + y, [ px[j] * s[j] for j in range(0, len(px)) ]),
		reduce(lambda x, y,: x + y, [ py[j] * s[j] for j in range(0, len(py)) ]),
	 	reduce(lambda x, y,: x + y, [ pz[j] * s[j] for j in range(0, len(pz)) ])
	]

	g = sp.zeros(3, 2)

	for k in range(0, len(p)):
		for j in range(0, len(q2)):
			g[k, j] = sp.simplify(sp.diff(p[k], q2[j]))

	n = cross(g[:,0], g[:, 1])
	expr = assign_matrix("surface_element", n)

	code = tpl.format(
		FACE=i,
		CODE = c_gen(expr)
	)
		
	print(f"{i}) {code}\n")
