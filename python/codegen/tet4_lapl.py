#!/usr/bin/env python3

from sfem_codegen import *
from tet4 import Tet4
from laplace_op import LaplaceOp

simplify_expr = False

fe = Tet4()
op = LaplaceOp(fe, [qx, qy, qz])

print_hessian = False
print_gradient = False
print_value = False

print_hessian = True
print_gradient = True
print_value = True

if print_hessian:
	print('Hessian')
	print('---------------------------------------------------')
	c_code(op.hessian())
	print('---------------------------------------------------')

if print_gradient:
	print('Gradient')
	print('---------------------------------------------------')
	c_code(op.gradient())
	print('---------------------------------------------------')

if print_value:
	print('Value')
	print('---------------------------------------------------')
	c_code(op.value())
	print('---------------------------------------------------')
	
