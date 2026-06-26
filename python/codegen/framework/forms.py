from dataclasses import dataclass
from enum import Enum

import sympy as sp

try:
    from .symbolic import (
        ExpressionRole,
        KernelExpressions,
        gradient_from_energy,
        hessian_action_from_energy,
        jacobian_action_from_residual,
        residual_from_energy,
    )
except ImportError:
    from symbolic import (
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
class FormEvaluation:
    kind: FormKind
    forms: tuple

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
