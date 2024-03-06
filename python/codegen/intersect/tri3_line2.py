#!/usr/bin/env python3

import numpy as np
import sympy as sp

import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen
from sympy import Piecewise

import sympy.codegen.ast as ast
import rich
from time import perf_counter

from rich.syntax import Syntax
console = rich.get_console()

real_t = "real_t"

def c_log(expr):
    console.print(expr)

def c_gen(expr, dump=False):
    console.print("--------------------------")
    console.print(f'Running cse')
    start = perf_counter()

    sub_expr, simpl_expr = sp.cse(expr)

    # sub_ops = sp.count_ops(sub_expr, visual=True)
    # result_ops = sp.count_ops(simpl_expr, visual=True)
    # cost = f'FLOATING POINT OPS!\n//\t- Result: {result_ops}\n//\t- Subexpressions: {sub_ops}'
    
    printer = sp.printing.c.C99CodePrinter()
    lines = []

    for var,expr in sub_expr:
        lines.append(f'const {real_t} {var} = {printer.doprint(expr)};')

    for v in simpl_expr:
        lines.append(printer.doprint(v))

    code_string=f'\n'.join(lines)

    stop = perf_counter()
    console.print(f'Elapsed  {stop - start} seconds')
    console.print("--------------------------")
    console.print(f'generated code')

    # code_string = f'//{cost}\n' + code_string
    # code_string = f'//TODO COST\n' + code_string

    if dump:
        console.print(code_string)

    return code_string

def c_code(expr):
    code_string = c_gen(expr)
    console.print(code_string)

def triangle():
    x0, x1, x2 = sp.symbols("tx[0] tx[1] tx[2]")
    y0, y1, y2 = sp.symbols("ty[0] ty[1] ty[2]")

    x = [x0, x1, x2]
    y = [y0, y1, y2]
    return x, y

x, y = triangle()

x3, x4 = sp.symbols("lx[0] lx[1]")
y3, y4 = sp.symbols("ly[0] ly[1]")

t = sp.zeros(len(x), 1)
u = sp.zeros(len(x), 1)


c_log("Intersection")

for i in range(0, len(x)):
    ip1 = (i + 1) % len(x)

    p1 = [x[i], y[i] ]
    p2 = [x[ip1], y[ip1] ]

    num = ((p1[0] - x3) * (y3 - y4) - (p1[1] - y3) * (x3 - x4))
    denom = ((p1[0] - p2[0]) * (y3 - y4) - (p1[1] - p2[1]) * (x3 - x4))

    t[i] = num / denom

    num = ((p1[0] - x3) * (p1[1] - p2[1]) - (p1[1] - y3) * (p1[0] - p2[0]))
    denom = ((p1[0] - p2[0]) * (y3 - y4) - (p1[1] - p2[1]) * (x3 - x4))
    
    u[i] =  num / denom

expr = []
for i in range(0, len(x)):
    expr.append(ast.Assignment(sp.symbols(f't[{i}]'), t[i]))

for i in range(0, len(x)):
    expr.append(ast.Assignment(sp.symbols(f'u[{i}]'), u[i]))


c_code(expr)

# c_log("Segment or point")

# expr = []
# for i in range(0, len(x)):
#      pred = Piecewise((0, u[i] < 0), (0, u[i] > 1), (0, u[i] != u[i]), (0, t[i] < 0), (0, t[i] > 1), (0, t[i] != t[i]),(1, True))
#      expr.append(ast.Assignment(sp.symbols(f'p[{i}]'), pred))
# c_code(expr)
