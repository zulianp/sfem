import math
from contextlib import contextmanager
from itertools import product

import sympy as sp
from dataclasses import dataclass


TEST_ARGUMENT = "test"
TRIAL_ARGUMENT = "trial"
PREVIOUS_ARGUMENT = "old"
_GEOMETRIC_DIM_STACK = []


@contextmanager
def geometric_dimension_context(dim):
    dim = int(dim)
    if dim <= 0:
        raise ValueError("geometric dimension must be positive")
    _GEOMETRIC_DIM_STACK.append(dim)
    try:
        yield
    finally:
        _GEOMETRIC_DIM_STACK.pop()


def current_geometric_dimension():
    if not _GEOMETRIC_DIM_STACK:
        return None
    return _GEOMETRIC_DIM_STACK[-1]


@dataclass(frozen=True)
class FiniteElement:
    family: str
    cell: str = ""
    degree: int = 1
    value_shape: tuple = ()

    def __post_init__(self):
        family = str(self.family)
        cell = str(self.cell)
        degree = int(self.degree)
        value_shape = tuple(
            "geometric" if extent == "geometric" else int(extent)
            for extent in self.value_shape
        )
        if not family:
            raise ValueError("finite element family must be non-empty")
        if degree < 0:
            raise ValueError("finite element degree must be non-negative")
        if any(extent != "geometric" and extent <= 0 for extent in value_shape):
            raise ValueError("finite element value shape extents must be positive")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "value_shape", value_shape)


@dataclass(frozen=True)
class FunctionSpace:
    element: FiniteElement
    dim: int = None
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.element, FiniteElement):
            raise TypeError("FunctionSpace requires a FiniteElement")
        if self.dim is not None and int(self.dim) <= 0:
            raise ValueError("function-space dimension must be positive")
        object.__setattr__(self, "dim", None if self.dim is None else int(self.dim))
        object.__setattr__(self, "name", str(self.name))

    @property
    def value_shape(self):
        return self.element.value_shape


@dataclass(frozen=True)
class MixedFunctionSpace:
    spaces: tuple
    name: str = ""

    def __init__(self, *spaces, name=""):
        if len(spaces) == 1 and isinstance(spaces[0], (tuple, list)):
            spaces = tuple(spaces[0])
        spaces = tuple(spaces)
        if not spaces:
            raise ValueError("MixedFunctionSpace requires at least one space")
        if not all(isinstance(space, FunctionSpace) for space in spaces):
            raise TypeError("MixedFunctionSpace entries must be FunctionSpace instances")
        object.__setattr__(self, "spaces", spaces)
        object.__setattr__(self, "name", str(name))

    def __iter__(self):
        return iter(self.spaces)

    def __len__(self):
        return len(self.spaces)

    def __getitem__(self, index):
        return self.spaces[index]


class SymbolicField:
    __slots__ = ("name", "shape", "family", "metadata", "_symbols")

    def __init__(self, name, shape=(), family="", metadata=None):
        name = str(name)
        family = str(family)
        shape = tuple(int(extent) for extent in shape)
        if not name or not name.isidentifier():
            raise ValueError("field name must be a valid identifier")
        if family and not family.isidentifier():
            raise ValueError("field family must be a valid identifier")
        if any(extent <= 0 for extent in shape):
            raise ValueError("field shape extents must be positive")
        self.name = name
        self.shape = shape
        self.family = family
        self.metadata = dict(metadata or ())
        self._symbols = _component_symbols(name, shape)

    @property
    def rank(self):
        return len(self.shape)

    @property
    def size(self):
        return len(self._symbols)

    @property
    def symbols(self):
        return self._symbols

    @property
    def free_symbols(self):
        return set(self._symbols)

    @property
    def is_scalar(self):
        return self.rank == 0

    @property
    def is_vector(self):
        return self.rank == 1

    @property
    def is_tensor(self):
        return self.rank >= 2

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self._symbols)

    def __getitem__(self, index):
        return self._symbols[_flat_index(self.shape, index)]

    def as_array(self):
        return _symbols_as_array(self._symbols, self.shape)

    @property
    def value(self):
        return self.as_array()

    def __repr__(self):
        return "%s(%r, shape=%r)" % (
            self.__class__.__name__,
            self.name,
            self.shape,
        )


class ScalarField(SymbolicField):
    def __init__(self, name, family="", metadata=None):
        super().__init__(name, (), family, metadata)

    @property
    def symbol(self):
        return self._symbols[0]

    @property
    def value(self):
        return self.symbol

    def _sympy_(self):
        return self.symbol

    def __neg__(self):
        return -self.symbol

    def __add__(self, other):
        return self.symbol + _sympify_field(other)

    def __radd__(self, other):
        return _sympify_field(other) + self.symbol

    def __sub__(self, other):
        return self.symbol - _sympify_field(other)

    def __rsub__(self, other):
        return _sympify_field(other) - self.symbol

    def __mul__(self, other):
        return self.symbol * _sympify_field(other)

    def __rmul__(self, other):
        return _sympify_field(other) * self.symbol

    def __truediv__(self, other):
        return self.symbol / _sympify_field(other)

    def __rtruediv__(self, other):
        return _sympify_field(other) / self.symbol

    def __pow__(self, other):
        return self.symbol ** _sympify_field(other)

    def __rpow__(self, other):
        return _sympify_field(other) ** self.symbol


class VectorField(SymbolicField):
    def __init__(self, name, dim, family="", metadata=None):
        super().__init__(name, (int(dim),), family, metadata)

    @property
    def dim(self):
        return self.shape[0]

    def as_matrix(self):
        return sp.Matrix(self.dim, 1, self._symbols)

    @property
    def value(self):
        return self.as_matrix()


class TensorField(SymbolicField):
    def __init__(self, name, shape, family="", metadata=None):
        shape = tuple(int(extent) for extent in shape)
        if len(shape) < 2:
            raise ValueError("tensor fields require rank at least 2")
        super().__init__(name, shape, family, metadata)

    def as_matrix(self):
        if self.rank != 2:
            raise ValueError("only rank-2 tensor fields can be converted to Matrix")
        return sp.Matrix(self.shape[0], self.shape[1], self._symbols)

    @property
    def value(self):
        if self.rank == 2:
            return self.as_matrix()
        return self.as_array()


class SymbolicArgument:
    __slots__ = ("field", "name", "role", "_symbols")

    def __init__(self, field, role, name=None):
        if not isinstance(field, SymbolicField):
            raise TypeError("symbolic arguments require a SymbolicField")
        role = str(role)
        if role not in (TEST_ARGUMENT, TRIAL_ARGUMENT, PREVIOUS_ARGUMENT):
            raise ValueError("argument role must be 'test', 'trial', or 'old'")
        name = "%s_%s" % (field.name, role) if name is None else str(name)
        if not name or not name.isidentifier():
            raise ValueError("argument name must be a valid identifier")
        self.field = field
        self.name = name
        self.role = role
        self._symbols = _component_symbols(name, field.shape)

    @property
    def shape(self):
        return self.field.shape

    @property
    def family(self):
        return self.field.family

    @property
    def rank(self):
        return self.field.rank

    @property
    def size(self):
        return len(self._symbols)

    @property
    def symbols(self):
        return self._symbols

    @property
    def free_symbols(self):
        return set(self._symbols)

    @property
    def is_scalar(self):
        return self.field.is_scalar

    @property
    def is_vector(self):
        return self.field.is_vector

    @property
    def is_tensor(self):
        return self.field.is_tensor

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self._symbols)

    def __getitem__(self, index):
        return self._symbols[_flat_index(self.shape, index)]

    def as_array(self):
        return _symbols_as_array(self._symbols, self.shape)

    def as_matrix(self):
        if self.rank == 1:
            return sp.Matrix(self.shape[0], 1, self._symbols)
        if self.rank == 2:
            return sp.Matrix(self.shape[0], self.shape[1], self._symbols)
        raise ValueError("only vector and rank-2 tensor arguments can be converted to Matrix")

    @property
    def value(self):
        if self.is_scalar:
            return self._symbols[0]
        if self.rank <= 2:
            return self.as_matrix()
        return self.as_array()

    def _sympy_(self):
        return self._scalar_expr()

    def _scalar_expr(self):
        if not self.is_scalar:
            raise TypeError("only scalar arguments coerce to SymPy expressions")
        return self._symbols[0]

    def __neg__(self):
        return -self._scalar_expr()

    def __add__(self, other):
        return self._scalar_expr() + _sympify_field(other)

    def __radd__(self, other):
        return _sympify_field(other) + self._scalar_expr()

    def __sub__(self, other):
        return self._scalar_expr() - _sympify_field(other)

    def __rsub__(self, other):
        return _sympify_field(other) - self._scalar_expr()

    def __mul__(self, other):
        return self._scalar_expr() * _sympify_field(other)

    def __rmul__(self, other):
        return _sympify_field(other) * self._scalar_expr()

    def __truediv__(self, other):
        return self._scalar_expr() / _sympify_field(other)

    def __rtruediv__(self, other):
        return _sympify_field(other) / self._scalar_expr()

    def __pow__(self, other):
        return self._scalar_expr() ** _sympify_field(other)

    def __rpow__(self, other):
        return _sympify_field(other) ** self._scalar_expr()

    def __repr__(self):
        return "%s(%r, %s)" % (
            self.__class__.__name__,
            self.field.name,
            self.name,
        )


class TestFunction(SymbolicArgument):
    def __init__(self, field_or_space, name=None, dim=None):
        if isinstance(field_or_space, FunctionSpace):
            name = name or "v"
            field_or_space = _field_from_space(field_or_space, name, dim=dim)
            super().__init__(field_or_space, TEST_ARGUMENT, name)
        else:
            super().__init__(field_or_space, TEST_ARGUMENT, name)


class TrialFunction(SymbolicArgument):
    def __init__(self, field_or_space, name=None, dim=None):
        if isinstance(field_or_space, FunctionSpace):
            name = name or "du"
            field_or_space = _field_from_space(field_or_space, name, dim=dim)
            super().__init__(field_or_space, TRIAL_ARGUMENT, name)
        else:
            super().__init__(field_or_space, TRIAL_ARGUMENT, name)


class PreviousFunction(SymbolicArgument):
    def __init__(self, field, name=None):
        super().__init__(field, PREVIOUS_ARGUMENT, name or "%s_old" % field.name)


def scalar_field(name, family="", metadata=None):
    return ScalarField(name, family, metadata)


def Function(space_or_name, name=None, metadata=None, dim=None, qualifier=None):
    if isinstance(space_or_name, FunctionSpace):
        return _field_from_space(space_or_name, name, metadata, dim, qualifier)
    return scalar_field(space_or_name, _family_from_qualifier(qualifier) or name or "", metadata)


def vector_field(name, dim, family="", metadata=None):
    return VectorField(name, dim, family, metadata)


def VectorElement(family, cell="", degree=1, components=None):
    value_shape = ("geometric",) if components is None else (int(components),)
    return FiniteElement(family, cell, degree, value_shape)


def VectorFunctionSpace(element, dim=None, name=""):
    if not isinstance(element, FiniteElement):
        element = VectorElement(str(element), components=dim)
    return FunctionSpace(element, dim, name)


def VectorFunction(space_or_name, name=None, dim=None, family="", metadata=None, qualifier=None):
    if isinstance(space_or_name, FunctionSpace):
        return _field_from_space(space_or_name, name, metadata, dim, qualifier)
    return vector_field(space_or_name, dim, _family_from_qualifier(qualifier) or family, metadata)


def tensor_field(name, shape, family="", metadata=None):
    return TensorField(name, shape, family, metadata)


def TensorFunction(space_or_name, name=None, shape=None, family="", metadata=None, qualifier=None):
    if isinstance(space_or_name, FunctionSpace):
        return _field_from_space(space_or_name, name, metadata, qualifier=qualifier)
    return tensor_field(space_or_name, shape, _family_from_qualifier(qualifier) or family, metadata)


def _field_from_space(space, name, metadata=None, dim=None, qualifier=None):
    if not name:
        raise ValueError("function name is required when using FunctionSpace")
    shape = space.value_shape
    dim = space.dim if dim is None else int(dim)
    if dim is None:
        dim = current_geometric_dimension()
    if shape == ("geometric",):
        if dim is None:
            raise ValueError("geometric vector space requires a concrete dimension")
        shape = (dim,)
    metadata_dict = dict(metadata or ())
    metadata_dict["space"] = space
    if qualifier is not None:
        metadata_dict["qualifiers"] = _qualifier_tuple(qualifier)
    if dim is not None:
        metadata_dict["dim"] = dim
    family = _family_from_qualifier(qualifier)
    if not shape:
        return ScalarField(name, family, metadata_dict)
    if len(shape) == 1:
        return VectorField(name, shape[0], family, metadata_dict)
    return TensorField(name, shape, family, metadata_dict)


def _qualifier_tuple(qualifier):
    if qualifier is None:
        return ()
    if isinstance(qualifier, (tuple, list)):
        return tuple(qualifier)
    return (qualifier,)


def _family_from_qualifier(qualifier):
    qualifiers = _qualifier_tuple(qualifier)
    if not qualifiers:
        return ""
    name = str(getattr(qualifiers[0], "name", qualifiers[0]))
    if not name.isidentifier():
        raise ValueError("field qualifier name must be a valid identifier")
    return name


def test_function(field, name=None):
    return TestFunction(field, name)


def trial_function(field, name=None):
    return TrialFunction(field, name)


def previous_function(field, name=None):
    return PreviousFunction(field, name)


def _sympify_field(value):
    if isinstance(value, ScalarField):
        return value.symbol
    if isinstance(value, SymbolicArgument):
        return value._scalar_expr()
    return sp.sympify(value)


def _component_symbols(name, shape):
    return tuple(
        sp.Symbol(name if not shape else "%s[%d]" % (name, ordinal))
        for ordinal, _ in enumerate(_field_component_indices(shape))
    )


def _symbols_as_array(symbols, shape):
    if not shape:
        return symbols[0]
    return sp.ImmutableDenseNDimArray(symbols, shape)


def _field_component_indices(shape):
    if not shape:
        return ((),)
    return tuple(product(*(range(extent) for extent in shape)))


def _flat_index(shape, index):
    if not shape:
        if index not in (0, (), None):
            raise IndexError("scalar field has one component")
        return 0
    if isinstance(index, tuple):
        if len(index) != len(shape):
            raise IndexError("field index rank mismatch")
        return _flat_index_from_multi(index, shape)
    if len(shape) == 1:
        index = int(index)
        if index < 0 or index >= shape[0]:
            raise IndexError("field component index out of range")
        return index
    index = int(index)
    if index < 0 or index >= math.prod(shape):
        raise IndexError("field component index out of range")
    return index


def _flat_index_from_multi(index, shape):
    flat = 0
    stride = 1
    for component, extent in zip(reversed(index), reversed(shape)):
        component = int(component)
        if component < 0 or component >= extent:
            raise IndexError("field component index out of range")
        flat += component * stride
        stride *= extent
    return flat


__all__ = [
    "ScalarField",
    "SymbolicField",
    "SymbolicArgument",
    "FiniteElement",
    "FunctionSpace",
    "current_geometric_dimension",
    "geometric_dimension_context",
    "MixedFunctionSpace",
    "PREVIOUS_ARGUMENT",
    "TEST_ARGUMENT",
    "TensorField",
    "PreviousFunction",
    "TestFunction",
    "TRIAL_ARGUMENT",
    "TrialFunction",
    "VectorField",
    "Function",
    "scalar_field",
    "tensor_field",
    "TensorFunction",
    "previous_function",
    "test_function",
    "trial_function",
    "VectorElement",
    "VectorFunctionSpace",
    "VectorFunction",
    "vector_field",
]
