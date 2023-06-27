#!/usr/bin/env python3

import sympy as sp
from sfem_codegen import c_gen
from sfem_codegen import c_log
from sfem_codegen import q as quadrature_point
from sfem_codegen import real_t
from sfem_codegen import coeffs
from sfem_codegen import det2
from sfem_codegen import det3
import sympy.codegen.ast as ast
import sympy as sp


def read_file(path):
	with open(path, 'r') as f:
	    tpl = f.read()
	    return tpl
	assert False
	return ""

def str_to_file(path, mystr):
	with open(path, 'w') as f:
		f.write(mystr)
		f.close()

tpl = read_file('tpl/inverse_tpl.c')

real_t = 'real_t'

input_stream_prefix  = f'const {real_t}'
output_stream_prefix = f'{real_t} *const SFEM_RESTRICT'

def create_array(name, size):
	ret = [0] * size
	for i in range(0, size):
		ret[i] = sp.symbols(f'{name}_{i}', real=True)
	return ret

if __name__ == '__main__':

	code_all = '//Auto-generated\n'
	code_all += '#include "sfem_base.h"\n'
	code_all += '#include "sfem_vec.h"\n'
	code_all += '#include <assert.h>\n'
	code_all += '#include <stddef.h>\n'
	code_all += '\n'

	max_size = 4

	for size in range(1, max_size+1):
		n = size*size
		mat = create_array('mat', n)
		mat_inv = create_array('mat_inv', n)

		input_args=''
		output_args=''

		A = sp.Matrix(size, size, mat)
		s_A_inv = sp.Matrix(size, size, mat_inv)

		A_inv = A.inv()
		expr = []

		for i in range(0, size):
			for j in range(0, size):
				var = sp.symbols(f'*{s_A_inv[i, j]}')
				expr.append(ast.Assignment(var, A_inv[i, j]))

		body = c_gen(expr)

		for i in range(0, n):
			input_args += f'{input_stream_prefix} {mat[i]},\n'

		for i in range(0, n):
			output_args += f'{output_stream_prefix} {mat_inv[i]}'
			if i < n-1:
				output_args +=',\n'

		row_idx='i'
		diag_idx='diag_idx'
		dinvert_pass_args=''

		for i in range(0, n):
			dinvert_pass_args += f'values[{i}][{diag_idx}],\n'

		for i in range(0, n):
			dinvert_pass_args += f'&inv_diag[{i}][{row_idx}]'

			if i < n-1:
				dinvert_pass_args +=',\n'

		code_string = tpl.format(
			SIZE=size,
			INPUT_ARGS=input_args,
			OUTPUT_ARGS=output_args,
			BODY=body,
			DINVERT_PASS_ARGS=dinvert_pass_args)

		code_all += code_string
		code_all += '\n'


	c_log(code_all)

	str_to_file('inverse.c', code_all)

