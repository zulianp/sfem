
from fields import *
from hex8 import *
from mass_op import *

fe = Hex8()
symbolic_integration = False
rows = fe.n_nodes() * fe.manifold_dim()

coeff_SoA = coeffs('u', rows)

vector_field = VectorField(fe, coeff_SoA)
op = MassOp(vector_field, fe, symbolic_integration)

print("----------------------------")
print("matrix")
print("----------------------------")
c_code(op.matrix())
print("----------------------------")	

print("----------------------------")
print("apply")
print("----------------------------")
c_code(op.apply())
print("----------------------------")