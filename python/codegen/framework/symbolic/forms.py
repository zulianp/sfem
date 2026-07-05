from dataclasses import dataclass
from enum import Enum

import sympy as sp

from codegen.framework.symbolic.core import (
    ExpressionRole,
    KernelExpressions,
    gradient_from_energy,
    hessian_action_from_energy,
    jacobian_action_from_residual,
    residual_from_energy,
)


class FormKind(Enum):
    ENERGY = "energy"
    RESIDUAL = "residual"


class FormOrder(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2


class StandardFormName(Enum):
    ZERO = "form_0"
    ONE = "form_1"
    TWO = "form_2"

    @classmethod
    def from_order(cls, order):
        order = FormOrder(order)
        if order is FormOrder.ZERO:
            return cls.ZERO
        if order is FormOrder.ONE:
            return cls.ONE
        return cls.TWO


class PipelineStage(Enum):
    USER_INPUT = "user_input"
    FORM_EVALUATION = "form_evaluation"
    SPECIALIZED_FORM_MANIPULATION = "specialized_form_manipulation"
    CODE_GENERATION = "code_generation"


@dataclass(frozen=True)
class FormDependencies:
    current: bool = False
    previous: bool = False
    direction: bool = False
    geometry: bool = False
    parameters: tuple = ()
    current_symbols: tuple = ()
    previous_symbols: tuple = ()
    direction_symbols: tuple = ()
    geometry_symbols: tuple = ()
    symbols: tuple = ()

    def __post_init__(self):
        parameters = tuple(self.parameters)
        current_symbols = tuple(self.current_symbols)
        previous_symbols = tuple(self.previous_symbols)
        direction_symbols = tuple(self.direction_symbols)
        geometry_symbols = tuple(self.geometry_symbols)
        symbols = tuple(self.symbols)
        if not symbols:
            symbols = tuple(
                dict.fromkeys(
                    current_symbols
                    + previous_symbols
                    + direction_symbols
                    + geometry_symbols
                    + parameters
                )
            )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "current_symbols", current_symbols)
        object.__setattr__(self, "previous_symbols", previous_symbols)
        object.__setattr__(self, "direction_symbols", direction_symbols)
        object.__setattr__(self, "geometry_symbols", geometry_symbols)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "current", bool(self.current or current_symbols))
        object.__setattr__(self, "previous", bool(self.previous or previous_symbols))
        object.__setattr__(self, "direction", bool(self.direction or direction_symbols))
        object.__setattr__(self, "geometry", bool(self.geometry or geometry_symbols))


class FormCollectionMixin:
    @property
    def stage(self):
        return PipelineStage.FORM_EVALUATION

    def form(self, order):
        order = FormOrder(order)
        for form in self.forms:
            if form.order is order:
                return form
        raise ValueError("form order %s was not evaluated" % order.name)

    def standard_form(self, name):
        name = StandardFormName(name)
        for form in self.forms:
            if form.standard_form is name:
                return form
        raise ValueError("standard form %s was not evaluated" % name.value)

    def standard_forms(self):
        return {form.standard_name: form for form in self.forms}

    def expressions(self):
        expressions = KernelExpressions()
        for form in self.forms:
            form.add_to(expressions)
        return expressions


@dataclass(frozen=True)
class UnifiedForm:
    kind: FormKind
    order: FormOrder
    role: ExpressionRole
    name: str
    expression: object

    @property
    def standard_name(self):
        return StandardFormName.from_order(self.order).value

    @property
    def standard_form(self):
        return StandardFormName.from_order(self.order)

    def add_to(self, expressions):
        return expressions.add(self.role, self.expression, self.name)


@dataclass(frozen=True)
class FormEvaluation(FormCollectionMixin):
    kind: FormKind
    forms: tuple


@dataclass(frozen=True)
class FormCollection(FormCollectionMixin):
    equation_name: str
    kind: FormKind
    fields: tuple
    forms: tuple
    measure: str = "dx"
    variables: tuple = ()
    directions: tuple = ()
    coefficients: tuple = ()
    qualifiers: tuple = ()
    dependencies: object = None
    blocks: tuple = ()
    source: object = None
    metadata: tuple = ()

    def form_metadata(self, order):
        order = FormOrder(order)
        for metadata in self.metadata:
            if metadata.order is order:
                return metadata
        raise ValueError("metadata for form order %s is not available" % order.name)

    def blocks_for(self, order):
        return self.form_metadata(order).blocks

    def block(self, order, row_field, column_field=None):
        order = FormOrder(order)
        row_field = str(row_field)
        column_field = None if column_field is None else str(column_field)
        for block in self.blocks_for(order):
            if block.row_field == row_field and block.column_field == column_field:
                return block
        if column_field is None:
            raise ValueError(
                "block for form order %s and row field '%s' is not available"
                % (order.name, row_field)
            )
        raise ValueError(
            "block for form order %s, row field '%s', and column field '%s' is not available"
            % (order.name, row_field, column_field)
        )

    def block_matrix(self, order):
        fields = tuple(field.name for field in self.fields)
        blocks = {
            (block.row_field, block.column_field): block
            for block in self.blocks_for(order)
        }
        return tuple(
            tuple(blocks.get((row, column)) for column in fields)
            for row in fields
        )

    @classmethod
    def from_evaluation(
        cls,
        equation_name,
        evaluation,
        *,
        measure="dx",
        fields=(),
        variables=(),
        directions=(),
        coefficients=(),
        qualifiers=(),
        dependencies=None,
        blocks=(),
        source=None,
        metadata=(),
    ):
        return cls(
            str(equation_name),
            evaluation.kind,
            tuple(fields),
            tuple(evaluation.forms),
            str(measure),
            tuple(variables),
            tuple(directions),
            tuple(coefficients),
            tuple(qualifiers),
            dependencies,
            tuple(blocks),
            source,
            tuple(metadata),
        )


@dataclass(frozen=True)
class FormQualifier:
    target: str
    name: str
    value: object = None

    def __post_init__(self):
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "name", str(self.name))


@dataclass(frozen=True)
class FormBlock:
    order: FormOrder
    row_field: str
    column_field: str = None
    expression: object = None
    coefficients: tuple = ()
    dependencies: object = None

    def __post_init__(self):
        object.__setattr__(self, "order", FormOrder(self.order))
        object.__setattr__(self, "row_field", str(self.row_field))
        if self.column_field is not None:
            object.__setattr__(self, "column_field", str(self.column_field))
        object.__setattr__(self, "expression", sp.sympify(self.expression))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))

    @property
    def name(self):
        if self.column_field is None:
            return "%s_%s" % (
                StandardFormName.from_order(self.order).value,
                self.row_field,
            )
        return "%s_%s_%s" % (
            StandardFormName.from_order(self.order).value,
            self.row_field,
            self.column_field,
        )

    @property
    def is_diagonal(self):
        return self.column_field is None or self.row_field == self.column_field

    @property
    def is_coupling(self):
        return self.column_field is not None and self.row_field != self.column_field


@dataclass(frozen=True)
class FormMetadata:
    order: FormOrder
    coefficients: tuple = ()
    dependencies: object = None
    blocks: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "order", FormOrder(self.order))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "blocks", tuple(self.blocks))


class FormPipeline:
    def __init__(self, kind, zero_form, variables, directions=None, *, merit=None):
        self.kind = FormKind(kind)
        self.zero_form = sp.sympify(zero_form)
        self.variables = tuple(variables)
        self.directions = None if directions is None else tuple(directions)
        self._merit = None if merit is None else sp.sympify(merit)

    @classmethod
    def energy(cls, energy, variables, directions=None):
        return cls(FormKind.ENERGY, energy, variables, directions)

    @classmethod
    def residual(cls, residual, variables, directions=None, *, merit=None):
        return cls(FormKind.RESIDUAL, residual, variables, directions, merit=merit)

    def form(self, order):
        order = FormOrder(order)
        if self.kind is FormKind.ENERGY:
            return self._energy_form(order)
        return self._residual_form(order)

    def forms(self, orders=(FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO)):
        return self.evaluate(orders).forms

    def evaluate(self, orders=(FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO)):
        return FormEvaluation(
            self.kind,
            tuple(self.form(order) for order in orders),
        )

    def expressions(self, orders=(FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO)):
        return self.evaluate(orders).expressions()

    def _energy_form(self, order):
        if order is FormOrder.ZERO:
            return UnifiedForm(
                FormKind.ENERGY,
                order,
                ExpressionRole.ENERGY,
                "energy",
                self.zero_form,
            )
        if order is FormOrder.ONE:
            return UnifiedForm(
                FormKind.ENERGY,
                order,
                ExpressionRole.GRADIENT,
                "gradient",
                gradient_from_energy(self.zero_form, self.variables),
            )
        return UnifiedForm(
            FormKind.ENERGY,
            order,
            ExpressionRole.HESSIAN_ACTION,
            "hessian_action",
            hessian_action_from_energy(
                self.zero_form,
                self.variables,
                self._require_directions(),
            ),
        )

    def _residual_form(self, order):
        if order is FormOrder.ZERO:
            return UnifiedForm(
                FormKind.RESIDUAL,
                order,
                ExpressionRole.MERIT,
                "merit",
                self._merit_expression(),
            )
        if order is FormOrder.ONE:
            return UnifiedForm(
                FormKind.RESIDUAL,
                order,
                ExpressionRole.RESIDUAL,
                "residual",
                self.zero_form,
            )
        return UnifiedForm(
            FormKind.RESIDUAL,
            order,
            ExpressionRole.JACOBIAN_ACTION,
            "jacobian_action",
            jacobian_action_from_residual(
                self.zero_form,
                self.variables,
                self._require_directions(),
            ),
        )

    def _require_directions(self):
        if self.directions is None:
            raise ValueError("directions are required for second-order forms")
        return self.directions

    def _merit_expression(self):
        if self._merit is not None:
            return self._merit
        residual = sp.Matrix(self.zero_form)
        return sp.simplify(sp.Rational(1, 2) * sum(value * value for value in residual))


def energy_form_pipeline(energy, variables, directions=None):
    return FormPipeline.energy(energy, variables, directions)


def residual_form_pipeline(residual, variables, directions=None, *, merit=None):
    return FormPipeline.residual(residual, variables, directions, merit=merit)
