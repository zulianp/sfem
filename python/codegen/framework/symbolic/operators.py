import sympy as sp

from codegen.framework.symbolic.core import directional_derivative, matrix_inner
from codegen.framework.symbolic.fields import (
    SymbolicArgument,
    SymbolicField,
    TimeRate,
    _sum_time_rate_terms,
    previous_function,
    time_rate,
)


def value(expr):
    if isinstance(expr, (SymbolicField, SymbolicArgument)):
        return expr.value
    if hasattr(expr, "value"):
        return expr.value
    return sp.sympify(expr)


def grad(expr, dim=None, name=None):
    if isinstance(expr, TimeRate):
        return _time_rate_gradient(expr, dim)
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


def dt(expr):
    """The time derivative of a field, left for a scheme to weight.

    The material writes this and names no scheme; `symbolic.fields.TimeRate`
    says what it carries and why.
    """
    if isinstance(expr, SymbolicField):
        return time_rate(expr)
    raise TypeError("dt(...) requires a symbolic field")


def _time_rate_gradient(rate, dim):
    """The gradient of a rate is the same sum, one gradient per term.

    No `name` here: each term names its gradient after its own argument, and one
    name shared across the terms would collapse them onto the same symbols.
    """
    return _sum_time_rate_terms(rate.terms, lambda argument: grad(argument, dim))


def div(expr, dim=None):
    # A rate goes through `grad` like a field does: `grad` knows how to sum a
    # `TimeRate`'s terms, and the divergence is the trace of what comes back.
    # Without this a rate reached `value` instead and arrived here as a column,
    # which reads as "not a gradient" -- the error a poroelastic material writing
    # `div(dt(u))` got.
    if isinstance(expr, (SymbolicField, SymbolicArgument, TimeRate)):
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
    "dt",
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
