"""What a residual emitter needs, derived from a closed form collection.

The residual emitters used to take a ``CoupledResidualSystem`` -- an object from
the symbolic layer, produced two stages earlier and carried down intact.  That
is what kept ``FormCollection.source`` alive: the emission path had no other way
to obtain it, so the boundary the architecture calls closed had a back-pointer
through it.

Measuring the actual dependency showed it was narrow.  Across 10,102 lines the
residual emitter reads exactly eight members off that system, and every one is a
function of data the lowered ``FormCollection`` already carries: the field
records, the parameters, and the per-order block expressions.

``ResidualEmissionModel`` is that surface, and nothing more.  It is built from a
collection, so the emitter's input is now derived from the closed artifact
rather than from the stage that produced it.  The class deliberately mirrors the
system's method names, because the point of this step is to sever the
dependency, not to rewrite ten thousand lines of emitter at the same time.
"""

from dataclasses import dataclass

import sympy as sp

from codegen.framework.symbolic.core import KernelExpressions
from codegen.framework.symbolic.residual import (
    CoupledResidualSystem,
    residual_dependencies_for,
)


@dataclass(frozen=True)
class ResidualEmissionModel:
    """The eight-member surface a residual emitter consumes."""

    fields: tuple
    dim: int
    parameters: tuple
    residual_forms: tuple
    jacobian_action_blocks: tuple

    def residual_expression(self, field):
        """The residual form for one field, by record or by name."""
        name = getattr(field, "name", field)
        for candidate, residual in zip(self.fields, self.residual_forms):
            if candidate.name == name:
                return residual
        raise ValueError("no residual for field '%s'" % name)

    def field(self, name):
        name = getattr(name, "name", name)
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise ValueError("no field '%s'" % name)

    def residual_expressions(self):
        """The residual as role-tagged kernel expressions, one per field."""
        expressions = KernelExpressions()
        for field, residual in zip(self.fields, self.residual_forms):
            expressions.residual(residual, "residual_%s" % field.name)
        return expressions

    def jacobian_action_expressions(self, include_blocks=False):
        blocks = {
            (block.row_field, block.column_field): block
            for block in self.jacobian_action_blocks
        }
        expressions = KernelExpressions()
        if include_blocks:
            for row in self.fields:
                for column in self.fields:
                    block = blocks[(row.name, column.name)]
                    expressions.jacobian_action(block.expression, block.name)
        for row in self.fields:
            action = sum(
                blocks[(row.name, column.name)].expression for column in self.fields
            )
            expressions.jacobian_action(action, "jacobian_action_%s" % row.name)
        return expressions

    def residual_data_symbols(self):
        candidates = []
        for field in self.fields:
            candidates.extend(field.current_symbols)
            candidates.extend(field.previous_symbols)
            candidates.extend(field.test_symbols)
        candidates.extend(self.parameters)
        free_symbols = set().union(
            *(residual.free_symbols for residual in self.residual_forms)
        ) if self.residual_forms else set()
        return tuple(symbol for symbol in candidates if symbol in free_symbols)

    def residual_dependencies(self):
        return residual_dependencies_for(
            self.fields, self.parameters, self.residual_forms
        )

    def jacobian_action_dependencies(self):
        return residual_dependencies_for(
            self.fields, self.parameters, self._row_action_expressions()
        )

    def dependencies_for_expressions(self, expressions):
        return residual_dependencies_for(self.fields, self.parameters, tuple(expressions))

    def jacobian_blocks(self):
        return self.jacobian_action_blocks

    def jacobian_action_data_symbols(self):
        candidates = []
        for field in self.fields:
            candidates.extend(field.current_symbols)
            candidates.extend(field.previous_symbols)
            candidates.extend(field.test_symbols)
        for field in self.fields:
            candidates.extend(field.direction_symbols)
        candidates.extend(self.parameters)
        free_symbols = set()
        for block in self.jacobian_action_blocks:
            free_symbols.update(block.expression.free_symbols)
        return tuple(symbol for symbol in candidates if symbol in free_symbols)

    def _row_action_expressions(self):
        """The Jacobian action per row field, summed over columns."""
        return tuple(
            sum(
                block.expression
                for block in self.jacobian_action_blocks
                if block.row_field == field.name
            )
            for field in self.fields
        )


def residual_emission_model(collection):
    """Build the model for a lowered residual ``FormCollection``."""
    fields = tuple(collection.residual_fields)
    if not fields:
        raise ValueError(
            "form collection '%s' carries no lowered residual fields"
            % collection.equation_name
        )
    residual_expressions = tuple(collection.residual_expressions)
    if len(residual_expressions) != len(fields):
        raise ValueError(
            "form collection '%s' has %d residual expressions for %d lowered fields"
            % (collection.equation_name, len(residual_expressions), len(fields))
        )
    return ResidualEmissionModel(
        fields=fields,
        dim=fields[0].dim,
        parameters=tuple(collection.parameters),
        residual_forms=residual_expressions,
        jacobian_action_blocks=tuple(collection.jacobian_action_blocks),
    )


def diagonal_block_emission_model(collection, field, action_block):
    """The model for a single diagonal block of a coupled system.

    A block kernel computes one row/column pair, so it needs field records for
    just that field's components, with zero residuals -- only the 2-form block
    carries an expression.  Constructing those records is a planning decision
    about what the block kernel computes, so it lives here; the backend used to
    do it inline, which meant a backend was building symbolic objects.

    ``CoupledResidualSystem`` is used purely as the builder that knows how to
    mint a lowered field's value, gradient, test, previous and direction
    symbols.  Nothing of it escapes: the return value is a model.
    """
    dim = _collection_dim(collection)
    builder = CoupledResidualSystem(dim)
    if collection.parameters:
        builder.add_parameters(*collection.parameters)
    for component, component_name in enumerate(_component_field_names(field)):
        lowered = builder.add_field(
            component_name,
            field_name=field.name,
            component=component,
            components=field.components,
        )
        builder.add_residual(lowered, sp.S.Zero)
    fields = tuple(builder.fields)
    return ResidualEmissionModel(
        fields=fields,
        dim=dim,
        parameters=tuple(collection.parameters),
        residual_forms=tuple(sp.S.Zero for _ in fields),
        jacobian_action_blocks=tuple(builder.jacobian_blocks()),
    ), action_block


def _collection_dim(collection):
    for field in collection.residual_fields:
        return field.dim
    raise ValueError(
        "form collection '%s' carries no lowered residual fields" % collection.equation_name
    )


def _component_field_names(field):
    components = int(getattr(field, "components", 1))
    if components == 1:
        return (field.name,)
    return tuple("%s%d" % (field.name, component) for component in range(components))


def residual_emission_model_from_system(system):
    """Build the model directly from a ``CoupledResidualSystem``.

    The production path goes through ``residual_emission_model`` and a lowered
    ``FormCollection``; this is for callers that construct a system by hand --
    chiefly tests -- so they do not have to hand-roll the record.  It lives in
    the planning layer, which may name the symbolic layer; the emitters may not.
    """
    fields = tuple(system.fields)
    return ResidualEmissionModel(
        fields=fields,
        dim=system.dim,
        parameters=tuple(system.parameters or ()),
        residual_forms=tuple(system.residual_expression(field) for field in fields),
        jacobian_action_blocks=tuple(system.jacobian_blocks()),
    )
