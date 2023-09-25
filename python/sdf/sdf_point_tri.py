#!/usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt

def dot3(u, v):
	return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

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
		tmp0 = a01 + b0
		tmp1 = a11 + b1
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
	# elif region == 6:
	# 	tmp0 = a01 + b0
	# 	tmp1 = a11 + b1
	# 	if(tmp1 > tmp0):
	# 		numer = tmp1 - tmp0
	# 		denom = a00 - 2 * a01 + a11
	# 		if numer >= denom:
	# 			s = 1
	# 		else:
	# 			s = numer / denom
	# 		t = 1 - s

	print(f'r={region}, s={s}, t={t}')
	q = B + s*E0 + t*E1
	print(f'{q}')

	# pdb.set_trace()
	return q




p = np.array([0.5, -0.6, 1])
t0 = np.array([0, 0, 0])
t1 = np.array([1, 0, 0])
t2 = np.array([0, 1, 0])

q = sdf_point_tri(p, t0, t1, t2)


