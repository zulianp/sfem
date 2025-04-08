#!/usr/bin/env python3

from sfem_codegen import *

dV = det3(A)

var = sp.symbols(f"element_value[0]")
expr = [ast.Assignment(var, dV)]

c_code(expr)
