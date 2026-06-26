from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class CodegenQualifier:
    name: str
    attributes: tuple = ()

    def __post_init__(self):
        name = str(self.name)
        if not name or not name.isidentifier():
            raise ValueError("qualifier name must be a valid identifier")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "attributes", tuple(self.attributes))


class HyperelasticQualifier(CodegenQualifier):
    pass


class MaterialParameterQualifier(CodegenQualifier):
    pass


class FieldQualifier(CodegenQualifier):
    pass


@dataclass(frozen=True)
class QualifiedExpression:
    expression: object
    qualifiers: tuple

    def __post_init__(self):
        expression = _sympy_value(self.expression)
        qualifiers = tuple(_as_qualifier(qualifier) for qualifier in self.qualifiers)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "qualifiers", qualifiers)

    @property
    def value(self):
        return self.expression

    @property
    def free_symbols(self):
        return self.expression.free_symbols

    def _sympy_(self):
        if isinstance(self.expression, sp.MatrixBase):
            raise TypeError("matrix qualified expressions do not coerce to scalar SymPy")
        return self.expression


class MaterialParameter:
    __slots__ = ("name", "symbol", "default", "qualifiers")

    def __init__(self, name, default=None, qualifiers=()):
        name = str(name)
        if not name or not name.isidentifier():
            raise ValueError("material parameter name must be a valid identifier")
        self.name = name
        self.symbol = sp.Symbol(name)
        self.default = default
        self.qualifiers = tuple(_as_qualifier(qualifier) for qualifier in qualifiers)

    @property
    def value(self):
        return self.symbol

    @property
    def free_symbols(self):
        return {self.symbol}

    def _sympy_(self):
        return self.symbol

    def __neg__(self):
        return -self.symbol

    def __add__(self, other):
        return self.symbol + _sympy_value(other)

    def __radd__(self, other):
        return _sympy_value(other) + self.symbol

    def __sub__(self, other):
        return self.symbol - _sympy_value(other)

    def __rsub__(self, other):
        return _sympy_value(other) - self.symbol

    def __mul__(self, other):
        return self.symbol * _sympy_value(other)

    def __rmul__(self, other):
        return _sympy_value(other) * self.symbol

    def __truediv__(self, other):
        return self.symbol / _sympy_value(other)

    def __rtruediv__(self, other):
        return _sympy_value(other) / self.symbol

    def __pow__(self, other):
        return self.symbol ** _sympy_value(other)

    def __rpow__(self, other):
        return _sympy_value(other) ** self.symbol

    def __repr__(self):
        return "MaterialParameter(%r)" % self.name


DEFORMATION_GRADIENT = HyperelasticQualifier("deformation_gradient")
MATERIAL_PARAMETER = MaterialParameterQualifier("material_parameter")
DISPLACEMENT = FieldQualifier("displacement")
PRESSURE = FieldQualifier("pressure")


def material_parameter(name, default=None, qualifiers=()):
    qualifier_tuple = (MATERIAL_PARAMETER,) + tuple(qualifiers)
    return MaterialParameter(name, default, qualifier_tuple)


def qualify(expression, *qualifiers):
    return QualifiedExpression(expression, qualifiers)


def variable(expression, name="F", qualifier=DEFORMATION_GRADIENT):
    expression = _sympy_value(expression)
    if isinstance(expression, sp.MatrixBase):
        symbols = sp.Matrix(
            expression.rows,
            expression.cols,
            sp.symbols("%s[0:%d]" % (name, expression.rows * expression.cols)),
        )
        return QualifiedExpression(symbols, (qualifier,))
    return QualifiedExpression(sp.Symbol(str(name)), (qualifier,))


def qualifiers(expression):
    return tuple(getattr(expression, "qualifiers", ()))


def _as_qualifier(qualifier):
    if isinstance(qualifier, CodegenQualifier):
        return qualifier
    return CodegenQualifier(str(qualifier))


def _sympy_value(value):
    if hasattr(value, "value"):
        return value.value
    return sp.sympify(value)


__all__ = [
    "CodegenQualifier",
    "DEFORMATION_GRADIENT",
    "DISPLACEMENT",
    "FieldQualifier",
    "HyperelasticQualifier",
    "MATERIAL_PARAMETER",
    "MaterialParameter",
    "MaterialParameterQualifier",
    "PRESSURE",
    "QualifiedExpression",
    "material_parameter",
    "qualifiers",
    "qualify",
    "variable",
]
