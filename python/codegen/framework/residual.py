from dataclasses import dataclass
from typing import Optional, Tuple

import sympy as sp

from .symbolic import (
    ExpressionRole,
    GeneratedKernelCode,
    KernelExpressions,
    directional_derivative,
    generate_cpp_kernel,
)


@dataclass(frozen=True)
class ResidualField:
    name: str
    dim: int
    value: sp.Symbol
    gradient: Tuple[sp.Symbol, ...]
    previous_value: Optional[sp.Symbol]
    previous_gradient: Tuple[sp.Symbol, ...]
    test_value: sp.Symbol
    test_gradient: Tuple[sp.Symbol, ...]
    direction_value: sp.Symbol
    direction_gradient: Tuple[sp.Symbol, ...]

    @property
    def variables(self):
        return (self.value,) + self.gradient

    @property
    def directions(self):
        return (self.direction_value,) + self.direction_gradient

    @property
    def current_symbols(self):
        return self.variables

    @property
    def previous_symbols(self):
        if self.previous_value is None:
            return ()
        return (self.previous_value,) + self.previous_gradient

    @property
    def test_symbols(self):
        return (self.test_value,) + self.test_gradient

    @property
    def direction_symbols(self):
        return self.directions


@dataclass(frozen=True)
class ResidualJacobianBlock:
    row_field: str
    column_field: str
    expression: sp.Expr

    @property
    def name(self):
        return "jacobian_%s_%s" % (self.row_field, self.column_field)


@dataclass(frozen=True)
class CoupledResidualKernels:
    residual: GeneratedKernelCode
    jacobian_action: GeneratedKernelCode


@dataclass(frozen=True)
class ResidualDependencies:
    current: bool
    previous: bool
    direction: bool
    parameters: Tuple[sp.Symbol, ...]


class CoupledResidualSystem:
    def __init__(self, dim):
        self.dim = int(dim)
        if self.dim not in (1, 2, 3):
            raise ValueError("coupled residual dimension must be 1, 2, or 3")
        self._fields = []
        self._field_by_name = {}
        self._parameters = []
        self._residuals = {}

    @property
    def fields(self):
        return tuple(self._fields)

    @property
    def parameters(self):
        return tuple(self._parameters)

    def add_field(self, name, previous=True):
        name = str(name)
        if not name or not name.isidentifier():
            raise ValueError("field name must be a valid identifier")
        if name in self._field_by_name:
            raise ValueError("field '%s' is already registered" % name)
        field = ResidualField(
            name=name,
            dim=self.dim,
            value=sp.Symbol(name),
            gradient=_symbols("%s_grad" % name, self.dim),
            previous_value=sp.Symbol("%s_old" % name) if previous else None,
            previous_gradient=(
                _symbols("%s_old_grad" % name, self.dim) if previous else ()
            ),
            test_value=sp.Symbol("%s_test" % name),
            test_gradient=_symbols("%s_test_grad" % name, self.dim),
            direction_value=sp.Symbol("%s_direction" % name),
            direction_gradient=_symbols("%s_direction_grad" % name, self.dim),
        )
        self._fields.append(field)
        self._field_by_name[name] = field
        return field

    def field(self, name):
        try:
            return self._field_by_name[str(name)]
        except KeyError:
            raise ValueError("unknown residual field '%s'" % name)

    def add_parameters(self, *parameters):
        for parameter in parameters:
            parameter = sp.sympify(parameter)
            if not isinstance(parameter, sp.Symbol):
                raise ValueError("residual parameters must be SymPy symbols")
            if parameter in self._parameters:
                raise ValueError("parameter '%s' is already registered" % parameter)
            self._parameters.append(parameter)
        return self

    def add_residual(self, field, expression):
        field = self.field(field.name if isinstance(field, ResidualField) else field)
        expression = sp.sympify(expression)
        if isinstance(expression, sp.MatrixBase):
            raise ValueError("residual equation for field '%s' must be scalar" % field.name)
        unknown = expression.free_symbols.difference(self.registered_symbols())
        if unknown:
            raise ValueError(
                "residual equation for field '%s' contains unregistered symbols: %s"
                % (field.name, ", ".join(sorted(map(str, unknown))))
            )
        self._residuals[field.name] = self._residuals.get(field.name, sp.S.Zero) + expression
        return self

    def set_residual(self, field, expression):
        field = self.field(field.name if isinstance(field, ResidualField) else field)
        self._residuals.pop(field.name, None)
        return self.add_residual(field, expression)

    def residual_expression(self, field):
        field = self.field(field.name if isinstance(field, ResidualField) else field)
        if field.name not in self._residuals:
            raise ValueError("residual equation for field '%s' is not registered" % field.name)
        return self._residuals[field.name]

    def registered_symbols(self):
        symbols = set(self._parameters)
        for field in self._fields:
            symbols.update(field.current_symbols)
            symbols.update(field.previous_symbols)
            symbols.update(field.test_symbols)
            symbols.update(field.direction_symbols)
        return symbols

    def residual_expressions(self):
        self._validate_complete()
        expressions = KernelExpressions()
        for field in self._fields:
            expressions.residual(
                self._residuals[field.name],
                "residual_%s" % field.name,
            )
        return expressions

    def jacobian_blocks(self):
        self._validate_complete()
        blocks = []
        for row in self._fields:
            residual = self._residuals[row.name]
            for column in self._fields:
                blocks.append(
                    ResidualJacobianBlock(
                        row_field=row.name,
                        column_field=column.name,
                        expression=directional_derivative(
                            residual,
                            column.variables,
                            column.directions,
                        ),
                    )
                )
        return tuple(blocks)

    def jacobian_action_expressions(self, include_blocks=False):
        self._validate_complete()
        blocks = {
            (block.row_field, block.column_field): block
            for block in self.jacobian_blocks()
        }
        expressions = KernelExpressions()
        if include_blocks:
            for row in self._fields:
                for column in self._fields:
                    block = blocks[(row.name, column.name)]
                    expressions.jacobian_action(block.expression, block.name)
        for row in self._fields:
            action = sum(
                blocks[(row.name, column.name)].expression
                for column in self._fields
            )
            expressions.jacobian_action(
                action,
                "jacobian_action_%s" % row.name,
            )
        return expressions

    def build_residual_graph(self, temporary_prefix="residual_tmp"):
        return self.residual_expressions().build_graph(
            data_symbols=self.residual_data_symbols(),
            temporary_prefix=temporary_prefix,
        )

    def build_jacobian_action_graph(
        self,
        include_blocks=False,
        temporary_prefix="jacobian_action_tmp",
    ):
        return self.jacobian_action_expressions(include_blocks).build_graph(
            data_symbols=self.jacobian_action_data_symbols(),
            temporary_prefix=temporary_prefix,
        )

    def generate_cpp_kernels(self, prefix, scalar_type="double"):
        prefix = str(prefix)
        return CoupledResidualKernels(
            residual=generate_cpp_kernel(
                self.build_residual_graph(),
                function_name="%s_residual" % prefix,
                scalar_type=scalar_type,
            ),
            jacobian_action=generate_cpp_kernel(
                self.build_jacobian_action_graph(),
                function_name="%s_jacobian_action" % prefix,
                scalar_type=scalar_type,
            ),
        )

    def residual_data_symbols(self):
        candidates = []
        for field in self._fields:
            candidates.extend(field.current_symbols)
            candidates.extend(field.previous_symbols)
            candidates.extend(field.test_symbols)
        candidates.extend(self._parameters)
        free_symbols = set().union(
            *(self._residuals[field.name].free_symbols for field in self._fields)
        )
        return tuple(symbol for symbol in candidates if symbol in free_symbols)

    def jacobian_action_data_symbols(self):
        candidates = []
        for field in self._fields:
            candidates.extend(field.current_symbols)
            candidates.extend(field.previous_symbols)
            candidates.extend(field.test_symbols)
        for field in self._fields:
            candidates.extend(field.direction_symbols)
        candidates.extend(self._parameters)
        free_symbols = set()
        for block in self.jacobian_blocks():
            free_symbols.update(block.expression.free_symbols)
        return tuple(symbol for symbol in candidates if symbol in free_symbols)

    def residual_dependencies(self):
        return self._dependencies(
            tuple(self._residuals[field.name] for field in self._fields)
        )

    def jacobian_action_dependencies(self):
        return self._dependencies(
            tuple(
                sum(
                    block.expression
                    for block in self.jacobian_blocks()
                    if block.row_field == field.name
                )
                for field in self._fields
            )
        )

    def dependencies_for_expressions(self, expressions):
        return self._dependencies(tuple(expressions))

    def _dependencies(self, expressions):
        free_symbols = set()
        for expression in expressions:
            free_symbols.update(sp.sympify(expression).free_symbols)
        return ResidualDependencies(
            current=any(
                free_symbols.intersection(field.current_symbols)
                for field in self._fields
            ),
            previous=any(
                free_symbols.intersection(field.previous_symbols)
                for field in self._fields
            ),
            direction=any(
                free_symbols.intersection(field.direction_symbols)
                for field in self._fields
            ),
            parameters=tuple(
                parameter
                for parameter in self._parameters
                if parameter in free_symbols
            ),
        )

    def _validate_complete(self):
        if not self._fields:
            raise ValueError("coupled residual system has no fields")
        missing = [
            field.name for field in self._fields if field.name not in self._residuals
        ]
        if missing:
            raise ValueError(
                "missing residual equations for fields: %s" % ", ".join(missing)
            )


def coupled_residual_system(dim):
    return CoupledResidualSystem(dim)


def _symbols(prefix, count):
    return tuple(sp.Symbol("%s_%d" % (prefix, index)) for index in range(count))
