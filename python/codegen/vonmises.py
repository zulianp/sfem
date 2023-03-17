#!/usr/bin/env python3

from sfem_codegen import *
import sympy as sp

def vonmises(stress):
	s = stress

	a = (s[0, 0] - s[1, 1]) * (s[0, 0] - s[1, 1])
	b = (s[1, 1] - s[2, 2]) * (s[1, 1] - s[2, 2])
	c = (s[2, 2] - s[0, 0]) * (s[2, 2] - s[0, 0])

	d = s[0, 1] * s[0, 1]
	e = s[1, 2] * s[1, 2]
	f = s[2, 0] * s[2, 0]

	ret = (a + b + c)/2 + 3 * (d + e + f)
	ret = sp.sqrt(ret)
	return ret
