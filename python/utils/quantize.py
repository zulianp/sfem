#!/usr/bin/env python3

import numpy as np

real_t = np.float64

prec = 64
prec = 32

if prec == 64:
	quant_t = np.int64
	max_quant_t = real_t(2)**(56-1e-9)
	M = 1
elif prec == 32:
	quant_t = np.int32
	max_quant_t = real_t(2)**(31-1e-9)
	M = 2
else:
	quant_t = np.int8
	max_quant_t = real_t(2)**(7-1e-9)
	M = 8

n = 1000
seg_size = n
nfactors = n // seg_size

a = np.linspace(0.1, (n-1)*10000, n, dtype=real_t)

def quantize(real_t, a, quant_t, max_quant_t, M, nfactors = len(a)):
	n = len(a)
	factors = np.ones(nfactors, dtype=real_t)
	q = []

	for i in range(0, M):
		q.append( np.ones(n, dtype=quant_t) )

	for fact in range(0, nfactors):
		s = (fact*seg_size)
		e = ((fact+1)*seg_size)
		
		a_seg = np.copy(a[s:e])
		a_range = np.max(np.abs(a_seg))

		factors[fact] = a_range

		# Normalize between -1 and 1
		a_seg /= a_range

		# Cast to int
		q[0][s:e] = (max_quant_t * a_seg).astype(quant_t)

		a_scale = real_t(max_quant_t)
		for i in range(1, M):
			a_diff = max_quant_t**i * a_seg 

			for j in range(0, i):
				a_diff -= q[j][s:e] * real_t(max_quant_t ** (i-j-1))

			q[i][s:e] = (max_quant_t * a_diff).astype(quant_t)

	return q, factors

def dequantize(quant_t, max_quant_t, q, factors, real_t):
	a_reconstructed = np.zeros(n, dtype=real_t)

	for fact in range(0, nfactors):	
		s = (fact*seg_size)
		e = ((fact+1)*seg_size)

		for i in range(0, M):
			qi = q[i][s:e].astype(real_t)
			qi /= real_t(max_quant_t)**(i+1)
			a_reconstructed[s:e] += qi

		a_reconstructed[s:e] *= factors[fact]  
	return a_reconstructed

q, factors = quantize(real_t, a, quant_t, max_quant_t, M, nfactors)
a_reconstructed = dequantize(quant_t, max_quant_t, q, factors, real_t)

diff = a_reconstructed - a
print('1 norm ', np.sum(np.abs(diff)))
print('2 norm ', np.sum(np.sqrt(np.sum(diff * diff))))

print('1 rel norm ', np.sum(np.abs(diff/a)))
