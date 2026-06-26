import sympy as sp

from .symbolic import directional_derivative, matrix_inner
from .symbolic_fields import SymbolicArgument, SymbolicField, previous_function


def value(expr):
    if isinstance(expr, (SymbolicField, SymbolicArgument)):
        return expr.value
    if hasattr(expr, "value"):
        return expr.value
    return sp.sympify(expr)


def grad(expr, dim=None, name=None):
    if isinstance(expr, (SymbolicField, SymbolicArgument)):
        return _symbolic_gradient(expr, dim, name)
    if dim is None:
        raise ValueError("gradient dimension is required for raw SymPy expressions")
    name = "grad" if name is None else str(name)
    return sp.Matrix(int(dim), 1, _component_symbols("%s" % name, int(dim)))


def Identity(dim):
    return sp.eye(int(dim))


def old(expr):
    if isinstance(expr, SymbolicField):
        return previous_function(expr)
    raise TypeError("old(...) requires a symbolic field")


def div(expr, dim=None):
    if isinstance(expr, (SymbolicField, SymbolicArgument)):
        expr = grad(expr, dim)
    else:
        expr = value(expr)
    if isinstance(expr, sp.MatrixBase):
        if expr.cols == 1:
            raise ValueError("divergence requires a vector field or gradient matrix")
        n = min(expr.rows, expr.cols)
        return sum(expr[i, i] for i in range(n))
    raise ValueError("divergence requires a vector field or matrix")


def deformation_gradient(displacement, dim=None):
    G = grad(displacement, dim)
    if not isinstance(G, sp.MatrixBase) or G.rows != G.cols:
        raise ValueError("deformation gradient requires a square displacement gradient")
    return sp.eye(G.rows) + G


def inner(left, right):
    return matrix_inner(value(left), value(right))


def det(expr):
    expr = value(expr)
    if not isinstance(expr, sp.MatrixBase):
        raise ValueError("determinant requires a matrix expression")
    return expr.det()


def inv(expr):
    expr = value(expr)
    if not isinstance(expr, sp.MatrixBase):
        raise ValueError("inverse requires a matrix expression")
    return expr.inv()


def adjugate(expr):
    expr = value(expr)
    if not isinstance(expr, sp.MatrixBase):
        raise ValueError("adjugate requires a matrix expression")
    return expr.adjugate()


def log(expr):
    return sp.log(value(expr))


def exp(expr):
    return sp.exp(value(expr))


def sqrt(expr):
    return sp.sqrt(value(expr))


def derivative(form, coefficient, argument=None):
    variables = tuple(coefficient.symbols)
    if argument is None:
        directions = tuple(sp.Symbol("%s_trial" % coefficient.name) for _ in variables)
    else:
        directions = tuple(argument.symbols)
    return directional_derivative(value(form), variables, directions)


def _symbolic_gradient(expr, dim, name):
    if dim is None:
        if expr.is_vector:
            dim = expr.shape[0]
        elif expr.is_tensor and expr.rank == 2 and expr.shape[0] == expr.shape[1]:
            dim = expr.shape[0]
        elif "dim" in getattr(expr, "metadata", ()):
            dim = expr.metadata["dim"]
        elif hasattr(expr, "field") and "dim" in expr.field.metadata:
            dim = expr.field.metadata["dim"]
        else:
            raise ValueError("gradient dimension is required for scalar fields")
    dim = int(dim)
    if dim <= 0:
        raise ValueError("gradient dimension must be positive")
    name = "%s_grad" % expr.name if name is None else str(name)
    if not name or not name.isidentifier():
        raise ValueError("gradient name must be a valid identifier")
    if expr.is_scalar:
        return sp.Matrix(dim, 1, _component_symbols(name, dim))
    if expr.is_vector:
        return sp.Matrix(expr.shape[0], dim, _component_symbols(name, expr.shape[0] * dim))
    return sp.ImmutableDenseNDimArray(
        _component_symbols(name, expr.size * dim),
        expr.shape + (dim,),
    )


def _component_symbols(name, count):
    return tuple(sp.Symbol("%s[%d]" % (name, i)) for i in range(int(count)))


__all__ = [
    "adjugate",
    "det",
    "deformation_gradient",
    "derivative",
    "div",
    "grad",
    "Identity",
    "inner",
    "inv",
    "exp",
    "log",
    "old",
    "sqrt",
    "value",
]
