#!/usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt

import sympy as sp
from sympy import init_printing
init_printing() 

def dot3(u, v):
	return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def dot2(u, v):
	return u[0] * v[0] + u[1] * v[1]

def sdf_point_tri(p, B, p0, p1):
	E0 = p0 - B
	E1 = p1 - B

	a00 = dot3(E0, E0)
	a01 = dot3(E0, E1)
	a11 = dot3(E1, E1)

	bmp = B - p
	b0 = dot3(E0, bmp)
	b1 = dot3(E1, bmp) 
	c = dot3(bmp, bmp)

	det = a00 * a11 - a01 * a01

	s = (a01 * b1 - a11 * b0) #/ det
	t = (a01 * b0 - a00 * b1) #/ det

	print(f'{b0}, {b1}, {c}')
	print(f'det={det}, s={s}, t={t}')
	
	region = 0
	if(s + t <= det):
		if(s < 0): 
			if(t < 0):
				region = 4
			else:
				region = 3
		elif t < 0:
			region = 5
		else:
			region = 0
	else:
		if s<0:
			region = 2
		elif t < 0:
			region = 6
		else:
			region = 1

	if region == 0:
		s /= det
		t /= det
	elif region == 1:	
		numer = a11 - a01 + b1 - b0
		if numer <= 0:
			s = 0
		else:
			denom = a00 - 2 * a01 + a11
			if (numer >= denom):
				s = 1
			else:
				s = numer/denom
		t = 1 - s
	elif region == 3:
		s = 0
		if b1 >= 0:
			t = 0
		else:
			if -b1 >= a11:
				t = 1
			else:
				t = -b1/a11
	elif region == 5:
		t = 0
		if b0 >= 0:
			s = 0
		else:
			if -b0 >= a00:
				s = 1
			else:
				s = -b0/a11
	elif region == 2:
		tmp0 = a01 + b0
		tmp1 = a11 + b1
		# F_s = -temp1
		# F_t = temp0 - temp1

		if(tmp1 > tmp0):
			numer = tmp1 - tmp0
			denom = a00 - 2 * a01 + a11
			if numer >= denom:
				s = 1
			else:
				s = numer / denom
			t = 1 - s
		else:
			s = 0
			if tmp1 <= 0:
				t = 1
			else:
				if b1 >= 0:
					t = 0
				else:
					t = -b1 / a11
	elif region == 4:
		tmp0 = b0
		tmp1 = b1
		# F_s = -temp1
		# F_t = temp0 - temp1
		if(tmp1 > tmp0):
			numer = tmp1 - tmp0
			denom = a00 - 2 * a01 + a11
			if numer >= denom:
				s = 1
			else:
				s = numer / denom
			t = 1 - s
		else:
			s = 0
			if tmp1 <= 0:
				t = 1
			else:
				if b1 >= 0:
					t = 0
				else:
					t = -b1 / a11
	elif region == 6:
		tmp0 = a01 + b1
		tmp1 = a00 + b0
		# F_s = -temp1
		# F_t = temp0 - temp1

		if(tmp1 > tmp0):
			numer = tmp1 - tmp0
			denom = a00 - 2 * a01 + a11
			if numer >= denom:
				s = 1
			else:
				s = numer / denom
			t = 1 - s
		else:
			s = 0
			if tmp1 <= 0:
				t = 1
			else:
				if b1 >= 0:
					t = 0
				else:
					t = -b1 / a11


	print(f'r={region}, s={s}, t={t}')
	q = B + s*E0 + t*E1
	print(f'{q}')

	# pdb.set_trace()
	return q




p = np.array([0.5, -0.6, 1])
t0 = np.array([0, 0, 0])
t1 = np.array([1, 0, 0])
t2 = np.array([0, 1, 0])

# q = sdf_point_tri(p, t0, t1, t2)


def vec3(name):
	x, y, z = sp.symbols(f'{name}[0] {name}[1] {name}[2]')
	return sp.Matrix(3, 1, [x, y, z])

s, t = sp.symbols('s t')
B = vec3('B')
E0 = vec3('E0')
E1 = vec3('E1')
P = vec3('P')

a00, a01, a11 = sp.symbols('a00 a01 a11')
b0, b1 = sp.symbols('b0 b1')
det = sp.symbols('det')


def T(s, t):
	return B + s * E0 + t * E1 


def Q(s, t):
	d = T(s, t) - P
	return dot3(d, d)

def grad_Q(s, t):
	# q = Q(s, t)
	# gq = sp.zeros(2, 1)
	# gq[0] = sp.diff(q, s)
	# gq[1] = sp.diff(q, t)

	gq = sp.zeros(2, 1)
	gq[0] = a00 * s + a01 * t + b0
	gq[1] = a01 * s + a11 * t + b1
	return gq

def vec2_sbus(vec_expr, syms, vals):
	vec_ret = sp.zeros(2, 1)
	for d in range(0, 2):
		e = vec_expr[d]

		for i in range(0, len(syms)):
			e = e.subs(syms[i], vals[i])

		vec_ret[d] = sp.sympify(e)
	return vec_ret


# sp.pprint(grad_Q(s, t))

c2 = [0, 1]
c4 = [0, 0]
c6 = [1, 0]

corners = [ c2, c4, c6 ]
c2_dirs = [[0, -1], [1, -1]]
c4_dirs = [[0, 1],  [1, 0]]
c6_dirs = [[-1, 0], [-1, 1]]
dirs 	= [c2_dirs, c4_dirs, c6_dirs]
tag = [2, 4, 6]

for i in range(0, 3):
	expr  = grad_Q(s, t)
	vec2 = vec2_sbus(expr, [s, t], corners[i])

	print(f'corner {tag[i]})')

	directiona_deriv = [0, 0]
	for d in range(0, 2):
		d2 = dot2(dirs[i][d], vec2)
		directiona_deriv[d] = d2

	sp.pprint(f'[\n{directiona_deriv[0]},\n {directiona_deriv[1]}\n]')


