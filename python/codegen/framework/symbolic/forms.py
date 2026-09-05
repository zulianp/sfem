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

    def component_blocks_for(self, order):
        """The blocks of this form, one per lowered field component.

        ``blocks_for`` keys by assembled field: Stokes reports ``u`` and ``p``
        at 1-form order.  This keys by lowered field, so the same collection
        reports ``u0``, ``u1``, ``u2``, ``p`` -- the representation everything
        below actually consumes, and the one an energy formulation and a
        residual formulation can share.

        Nothing new is computed.  A residual lowering already builds both: the
        per-component 1-form expressions sit in ``residual_expressions``,
        aligned with ``residual_fields``, and the per-component 2-form blocks
        in ``jacobian_action_blocks`` -- sixteen of them for Stokes, ``u0`` by
        ``u0`` through ``p`` by ``p``.  They were reachable only under
        residual-specific names, which is why the parallel representation has
        49 consumers against the form accessors' 19.  This is the accessor
        those consumers can move onto.

        Returns an empty tuple where the collection carries no per-component
        expansion -- today that is every energy formulation, which declares its
        structure through ``add_energy(..., fields=..., variables=...)``
        instead.  Deriving these for energy is the other half of the work; see
        ARCHITECTURE.html OP 20.
        """
        order = FormOrder(order)
        fields = tuple(getattr(self, "residual_fields", ()) or ())
        if order is FormOrder.ONE:
            expressions = tuple(getattr(self, "residual_expressions", ()) or ())
            if not fields or len(expressions) != len(fields):
                return ()
            return tuple(
                FormBlock(
                    FormOrder.ONE,
                    row_field=field.name,
                    expression=expression,
                )
                for field, expression in zip(fields, expressions)
            )
        if order is FormOrder.TWO:
            # The lowering's own block objects, not copies of them.  They
            # already satisfy what a per-component block has to offer -- row
            # field, column field, expression, name -- and re-wrapping them as
            # ``FormBlock`` would rename them: ``FormBlock.name`` is derived as
            # ``form_2_<row>_<column>``, while these carry the name that
            # already appears in generated code.  Re-exposing is the job here;
            # renaming is not, and a phase that must be byte-identical cannot
            # afford it.
            return tuple(getattr(self, "jacobian_action_blocks", ()) or ())
        return ()

    def component_block(self, order, row_field, column_field=None):
        """One per-component block by name, or ``None`` if it does not exist.

        Unlike ``block``, a missing block is not an error: a Jacobian is sparse
        between components that do not couple, and asking is how a caller finds
        out.
        """
        row_field = str(row_field)
        column_field = None if column_field is None else str(column_field)
        for block in self.component_blocks_for(order):
            if block.row_field == row_field and block.column_field == column_field:
                return block
        return None

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
    # The lowered residual field records, carrying the value, gradient, test,
    # previous and direction symbols that downstream planning needs.  These used
    # to be reached through `source`, i.e. by holding on to the pre-lowering
    # system; carrying them explicitly is what lets the planning layer stop
    # doing that.
    residual_fields: tuple = ()
    # The per-field residual expressions and the per-component Jacobian-action
    # blocks, aligned with `residual_fields`.  These cannot be inferred from the
    # 1-/2-form blocks: for mixed formulations those are keyed by assembled field
    # name (`u`), while the lowered fields are per component (`u0`, `u1`).
    residual_expressions: tuple = ()
    jacobian_action_blocks: tuple = ()
    parameters: tuple = ()
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
        residual_fields=(),
        residual_expressions=(),
        jacobian_action_blocks=(),
        parameters=(),
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
            tuple(residual_fields),
            tuple(residual_expressions),
            tuple(jacobian_action_blocks),
            tuple(parameters),
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

    def component_blocks_for(self, order):
        """The blocks of this form, one per lowered field component.

        ``blocks_for`` keys by assembled field: Stokes reports ``u`` and ``p``
        at 1-form order.  This keys by lowered field, so the same collection
        reports ``u0``, ``u1``, ``u2``, ``p`` -- the representation everything
        below actually consumes, and the one an energy formulation and a
        residual formulation can share.

        Nothing new is computed.  A residual lowering already builds both: the
        per-component 1-form expressions sit in ``residual_expressions``,
        aligned with ``residual_fields``, and the per-component 2-form blocks
        in ``jacobian_action_blocks`` -- sixteen of them for Stokes, ``u0`` by
        ``u0`` through ``p`` by ``p``.  They were reachable only under
        residual-specific names, which is why the parallel representation has
        49 consumers against the form accessors' 19.  This is the accessor
        those consumers can move onto.

        Returns an empty tuple where the collection carries no per-component
        expansion -- today that is every energy formulation, which declares its
        structure through ``add_energy(..., fields=..., variables=...)``
        instead.  Deriving these for energy is the other half of the work; see
        ARCHITECTURE.html OP 20.
        """
        order = FormOrder(order)
        fields = tuple(getattr(self, "residual_fields", ()) or ())
        if order is FormOrder.ONE:
            expressions = tuple(getattr(self, "residual_expressions", ()) or ())
            if not fields or len(expressions) != len(fields):
                return ()
            return tuple(
                FormBlock(
                    FormOrder.ONE,
                    row_field=field.name,
                    expression=expression,
                )
                for field, expression in zip(fields, expressions)
            )
        if order is FormOrder.TWO:
            # The lowering's own block objects, not copies of them.  They
            # already satisfy what a per-component block has to offer -- row
            # field, column field, expression, name -- and re-wrapping them as
            # ``FormBlock`` would rename them: ``FormBlock.name`` is derived as
            # ``form_2_<row>_<column>``, while these carry the name that
            # already appears in generated code.  Re-exposing is the job here;
            # renaming is not, and a phase that must be byte-identical cannot
            # afford it.
            return tuple(getattr(self, "jacobian_action_blocks", ()) or ())
        return ()

    def component_block(self, order, row_field, column_field=None):
        """One per-component block by name, or ``None`` if it does not exist.

        Unlike ``block``, a missing block is not an error: a Jacobian is sparse
        between components that do not couple, and asking is how a caller finds
        out.
        """
        row_field = str(row_field)
        column_field = None if column_field is None else str(column_field)
        for block in self.component_blocks_for(order):
            if block.row_field == row_field and block.column_field == column_field:
                return block
        return None

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
    # The lowered residual field records, carrying the value, gradient, test,
    # previous and direction symbols that downstream planning needs.  These used
    # to be reached through `source`, i.e. by holding on to the pre-lowering
    # system; carrying them explicitly is what lets the planning layer stop
    # doing that.
    residual_fields: tuple = ()
    # The per-field residual expressions and the per-component Jacobian-action
    # blocks, aligned with `residual_fields`.  These cannot be inferred from the
    # 1-/2-form blocks: for mixed formulations those are keyed by assembled field
    # name (`u`), while the lowered fields are per component (`u0`, `u1`).
    residual_expressions: tuple = ()
    jacobian_action_blocks: tuple = ()
    parameters: tuple = ()
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
        residual_fields=(),
        residual_expressions=(),
        jacobian_action_blocks=(),
        parameters=(),
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
            tuple(residual_fields),
            tuple(residual_expressions),
            tuple(jacobian_action_blocks),
            tuple(parameters),
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

    def admits(self, order):
        """Whether this pipeline can produce a form of this order at all.

        Every order exists for an energy, and orders one and two exist for any
        residual.  The 0-form is the one that can genuinely be absent: it is a
        potential, and a residual whose flux Jacobian is not symmetric is not
        the gradient of anything.  Asking is how a caller avoids requesting a
        form that does not exist, rather than catching the failure to build it.
        """
        order = FormOrder(order)
        if order is not FormOrder.ZERO or self.kind is FormKind.ENERGY:
            return True
        if self._merit is not None:
            return True
        try:
            self._recovered_potential()
        except ValueError:
            return False
        return True

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
            return self._residual_zero_form()
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

    def _residual_zero_form(self):
        """A residual always has a 0-form.  There are two of them.

        When the flux Jacobian is symmetric the residual is the gradient of a
        potential, and that potential is an element integral: it is summed over
        elements exactly like an energy, and it adds across operators, so it
        can be a term in the sum `Function::value` accumulates.

        When it is not -- a saddle point, or any genuinely non-symmetric system
        -- there is still a scalar to measure, and it is the one a Newton line
        search actually wants: ``1/2 * ||R||^2`` over the *assembled* residual.
        Nothing about that is unavailable.  R is what the 1-form already
        computes, so the merit costs no new element kernel at all: assemble,
        then take one dot product.

        What separates the two is not whether the 0-form exists but where its
        reduction happens -- inside the element loop, or after the scatter.
        So the merit form carries the residual itself rather than an integrand,
        because the residual is the thing to be assembled before reducing, and
        the role says which reduction applies.

        The consequence worth being explicit about: an assembled norm is not
        additive over operators.  ``1/2*||sum_op R_op||^2`` is not the sum of
        ``1/2*||R_op||^2``, so this 0-form belongs to whoever holds the whole
        residual -- the Function -- and an operator can only report it when it
        is the only contributor.  A potential has no such restriction.
        """
        try:
            potential = self._recovered_potential()
        except ValueError:
            return UnifiedForm(
                FormKind.RESIDUAL,
                FormOrder.ZERO,
                ExpressionRole.MERIT,
                "merit",
                self.zero_form,
            )
        return UnifiedForm(
            FormKind.RESIDUAL,
            FormOrder.ZERO,
            ExpressionRole.POTENTIAL,
            "potential",
            potential,
        )

    def _require_directions(self):
        if self.directions is None:
            raise ValueError("directions are required for second-order forms")
        return self.directions

    def _merit_expression(self):
        """The residual's 0-form: a potential the residual is the gradient of.

        This used to return ``1/2 * sum(r_i**2)`` over the *element-local*
        residual entries.  That is not a merit function for the assembled
        system and it cannot be made into one, for two separate reasons.

        It is not zero at the solution.  Element residuals cancel when they are
        scattered; their squares do not.  On a 1D Laplacian with two elements
        at the exact discrete solution the assembled residual is zero while
        ``sum_e 1/2*||r_e||^2`` is 1/2.

        And it is not additive, which is what ``Op::value`` has to be:
        ``Function::value`` loops over its operators letting each accumulate
        into one scalar, so an operator's contribution must be a term of a sum.
        ``1/2*||sum_op R_op||^2`` is not the sum of ``1/2*||R_op||^2``.

        What *is* additive, and is an element integral exactly like an energy,
        is a potential -- a scalar whose gradient is the residual.  One exists
        precisely when the flux's Jacobian is symmetric, and then the Poincare
        line integral recovers it in closed form.  For a Laplacian this returns
        ``kappa/2 * ||grad u||^2``, which is the energy the residual was
        derived from in the first place; the derivation simply runs backwards.

        A residual with a non-symmetric Jacobian has no potential, and the
        merit such a system needs -- ``1/2*||R||^2`` over the *assembled*
        residual -- is a functional of the whole vector rather than an integral
        over an element, so it cannot be produced here at all.  This raises
        rather than returning something that would silently be wrong.
        """
        if self._merit is not None:
            return self._merit
        return self._recovered_potential()

    def _flux_and_fields(self):
        """The residual as a flux vector, paired with the fields it varies in.

        A residual reaches this pipeline in one of two shapes, and the
        potential is recovered from either the same way once they are put in
        these terms.

        A weak form is a scalar linear in test symbols named for the field they
        test -- ``u_test_grad_0`` beside ``u_grad_0`` -- so the flux is what
        multiplies each test symbol, and the pairing is recovered from the
        names rather than passed alongside the expression.  The field a test
        symbol pairs with need not itself appear: a Neumann traction ``-t . v``
        contains no ``u`` at all, its flux is the constant ``-t``, and its
        potential is the linear work ``-t . u``.

        A residual vector is already one entry per variable, so it *is* the
        flux and the pipeline's declared variables are the fields.
        """
        if getattr(self.zero_form, "is_Matrix", False) and len(self.zero_form) > 1:
            flux = sp.Matrix(self.zero_form)
            if len(flux) == len(self.variables):
                return flux, tuple(self.variables)

        residual = (
            sp.Matrix(self.zero_form)
            if getattr(self.zero_form, "is_Matrix", False)
            else sp.Matrix([self.zero_form])
        )
        expression = sum(residual)
        symbols = sorted(expression.free_symbols, key=lambda symbol: symbol.name)
        declared = set(self.variables)
        tests, fields = [], []
        for symbol in symbols:
            if "_test" not in symbol.name:
                continue
            field = sp.Symbol(symbol.name.replace("_test", ""))
            if field not in symbols and field not in declared:
                return None, ()
            tests.append(symbol)
            fields.append(field)
        if not tests:
            return None, ()
        flux = sp.Matrix([sp.diff(expression, test) for test in tests])
        return flux, tuple(fields)

    def _recovered_potential(self):
        flux, fields = self._flux_and_fields()
        if flux is None:
            raise ValueError(
                "cannot derive a 0-form for this residual: it is neither a "
                "vector with one entry per variable nor a weak form whose "
                "test symbols pair with field symbols, so there is nothing to "
                "integrate a potential over"
            )
        trials = fields
        jacobian = flux.jacobian(sp.Matrix(trials))
        # `expand`, not `simplify`.  This runs for every residual the framework
        # lowers, including two-phase flow, whose Jacobian is large enough that
        # `simplify` on the difference does not finish in a useful time -- and
        # it was returning None there anyway, which is `simplify` saying it
        # could not decide.  Expansion settles the symmetric cases, which are
        # the ones that go on to have a potential; anything it cannot show to
        # be zero is treated as having none, which is the safe direction.
        difference = sp.expand(jacobian - jacobian.T)
        if not all(entry == 0 for entry in difference):
            raise ValueError(
                "this residual has no potential: the flux Jacobian is not "
                "symmetric, so no scalar has it as a gradient.  A 0-form for "
                "such a system is 1/2*||R||^2 over the assembled residual, "
                "which is a functional of the global vector and cannot be "
                "emitted as an element integral"
            )
        parameter = sp.Dummy("t", positive=True)
        scaled = {trial: parameter * trial for trial in trials}
        integrand = sum(
            component.subs(scaled) * trial
            for component, trial in zip(flux, trials)
        )
        return sp.simplify(sp.integrate(sp.expand(integrand), (parameter, 0, 1)))


def energy_form_pipeline(energy, variables, directions=None):
    return FormPipeline.energy(energy, variables, directions)


def residual_form_pipeline(residual, variables, directions=None, *, merit=None):
    return FormPipeline.residual(residual, variables, directions, merit=merit)
