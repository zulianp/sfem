from dataclasses import dataclass
from enum import Enum
import math

import sympy as sp

from .forms import (
    FormBlock,
    FormCollection,
    FormMetadata,
    FormOrder,
    FormQualifier,
    energy_form_pipeline,
    residual_form_pipeline,
)
from .residual import CoupledResidualSystem
from .residual_codegen import (
    coupled_residual_weak_coefficients,
    weak_residual_coefficients,
)
from .symbolic_fields import (
    ScalarField,
    SymbolicField,
    TensorField,
    VectorField,
    Function,
    VectorFunction,
    TensorFunction,
    _family_from_qualifier,
    scalar_field,
    geometric_dimension_context,
    tensor_field,
    test_function,
    trial_function,
    vector_field,
)


class EquationForm(Enum):
    ENERGY = "energy"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class EquationField:
    name: str
    components: int = 1
    family: str = ""
    metadata: object = None

    def __post_init__(self):
        name = str(self.name)
        family = str(self.family)
        components = int(self.components)
        metadata = dict(self.metadata or ())
        if not name or not name.isidentifier():
            raise ValueError("equation field name must be a valid identifier")
        if components <= 0:
            raise ValueError("equation field components must be positive")
        if family and not family.isidentifier():
            raise ValueError("equation field family must be a valid identifier")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "metadata", metadata)

    @property
    def is_scalar(self):
        return self.components == 1

    @property
    def is_vector(self):
        return self.components > 1


@dataclass(frozen=True)
class Equation:
    name: str
    form: EquationForm
    define: object
    fields: tuple = ()
    variables: tuple = ()
    directions: tuple = ()
    kernels: tuple = ()
    diagnostics: bool = True

    def __post_init__(self):
        name = str(self.name)
        if name and not name.isidentifier():
            raise ValueError("equation name must be empty or a valid identifier")
        form = EquationForm(self.form)
        fields = tuple(self.fields)
        if not all(isinstance(field, EquationField) for field in fields):
            raise TypeError("equation fields must be EquationField instances")
        variables = tuple(self.variables)
        define = self.define
        if form is EquationForm.ENERGY:
            if callable(define):
                raise TypeError("energy equations require an expression, not a callable")
            if not variables:
                raise ValueError("energy equations require explicit variables")
            define = _expression_value(define)
        elif not callable(define):
            raise TypeError("residual equations require a callable definition")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "form", form)
        object.__setattr__(self, "define", define)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(self, "kernels", tuple(self.kernels))
        object.__setattr__(self, "diagnostics", bool(self.diagnostics))

    @property
    def is_energy(self):
        return self.form is EquationForm.ENERGY

    @property
    def is_residual(self):
        return self.form is EquationForm.RESIDUAL


class EquationSystem:
    def __init__(self, dim):
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("equation system dimension must be positive")
        self._fields = []
        self._equations = []
        self._form_collections = {}

    @property
    def fields(self):
        return tuple(self._fields)

    @property
    def equations(self):
        return tuple(self._equations)

    def form_collection(self, equation_or_name, orders=None):
        equation = self._resolve_equation(equation_or_name)
        normalized_orders = (
            (FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO)
            if orders is None
            else tuple(FormOrder(order) for order in orders)
        )
        key = (id(equation), normalized_orders)
        collection = self._form_collections.get(key)
        if collection is None:
            collection = _build_form_collection(self, equation, normalized_orders)
            self._form_collections[key] = collection
        return collection

    def form_collections(self, orders=None):
        return tuple(
            self.form_collection(equation, orders=orders)
            for equation in self._equations
        )

    def field(self, name, components=1, family="", metadata=None):
        field = EquationField(name, components, family, metadata)
        if any(existing.name == field.name for existing in self._fields):
            raise ValueError("equation field '%s' is already registered" % field.name)
        self._fields.append(field)
        return field

    def scalar_field(self, name, family="", metadata=None):
        return self.field(name, 1, family, metadata)

    def vector_field(self, name, components=None, family="", metadata=None):
        return self.field(name, self.dim if components is None else components, family, metadata)

    def equation(
        self,
        name,
        form,
        define,
        *,
        fields=(),
        variables=(),
        directions=(),
        kernels=(),
        diagnostics=True,
    ):
        equation = Equation(
            name,
            form,
            define,
            fields=tuple(fields),
            variables=_symbols_from_variables(variables),
            directions=_symbols_from_variables(directions),
            kernels=tuple(kernels),
            diagnostics=diagnostics,
        )
        if equation.name and any(existing.name == equation.name for existing in self._equations):
            raise ValueError("equation '%s' is already registered" % equation.name)
        self._equations.append(equation)
        return equation

    def add_energy(
        self,
        name,
        define,
        *,
        fields=(),
        variables=None,
        directions=(),
        kernels=("objective", "gradient", "apply"),
        diagnostics=True,
    ):
        _validate_energy_variable_groups(fields, variables)
        return self.equation(
            name,
            EquationForm.ENERGY,
            define,
            fields=fields,
            variables=variables,
            directions=directions,
            kernels=kernels,
            diagnostics=diagnostics,
        )

    def add_residual(self, name, define, *, fields=()):
        return self.equation(
            name,
            EquationForm.RESIDUAL,
            define,
            fields=fields,
        )

    def _resolve_equation(self, equation_or_name):
        if isinstance(equation_or_name, Equation):
            if any(equation is equation_or_name for equation in self._equations):
                return equation_or_name
            raise ValueError("equation is not registered in this system")
        name = str(equation_or_name)
        matches = tuple(equation for equation in self._equations if equation.name == name)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError("unknown equation '%s'" % name)
        raise ValueError("equation name '%s' is ambiguous" % name)


class EquationSystems:
    def __init__(self, *systems):
        if len(systems) == 1 and isinstance(systems[0], (tuple, list)):
            systems = tuple(systems[0])
        self._by_dim = {}
        for system in systems:
            self.add(system)

    @property
    def systems(self):
        return tuple(self._by_dim[dim] for dim in sorted(self._by_dim))

    @property
    def dims(self):
        return tuple(sorted(self._by_dim))

    def add(self, system):
        if not isinstance(system, EquationSystem):
            raise TypeError("EquationSystems entries must be EquationSystem instances")
        if system.dim in self._by_dim:
            raise ValueError("equation system for dim %d is already registered" % system.dim)
        self._by_dim[system.dim] = system
        return system

    def for_dim(self, dim):
        dim = int(dim)
        try:
            return self._by_dim[dim]
        except KeyError:
            raise ValueError("material does not define an equation system for dim %d" % dim)

    def __len__(self):
        return len(self._by_dim)

    def __iter__(self):
        return iter(self.systems)


class EquationSystemBuilder:
    def __init__(self, dim):
        self._system = EquationSystem(dim)
        self._symbolic_fields = []
        self._equation_fields_by_name = {}

    @property
    def dim(self):
        return self._system.dim

    @property
    def system(self):
        return self._system

    @property
    def fields(self):
        return tuple(self._symbolic_fields)

    @property
    def equations(self):
        return self._system.equations

    def build(self):
        return self._system

    def field(self, name, components=1, family=""):
        components = int(components)
        if components <= 0:
            raise ValueError("equation field components must be positive")
        if components == 1:
            return self.scalar_field(name, family)
        return self.vector_field(name, components, family)

    def scalar_field(self, name, family=""):
        return self._register_symbolic_field(
            scalar_field(name, family, {"dim": self.dim}),
            1,
        )

    def Function(self, space_or_name, name=None, family="", qualifier=None):
        if name is None:
            return self.scalar_field(space_or_name, _family_from_qualifier(qualifier) or family)
        with geometric_dimension_context(self.dim):
            return self._register_external_field(Function(space_or_name, name, qualifier=qualifier))

    def vector_field(self, name, components=None, family=""):
        components = self.dim if components is None else int(components)
        return self._register_symbolic_field(
            vector_field(name, components, family, {"dim": self.dim}),
            components,
        )

    def VectorFunction(self, space_or_name, name=None, components=None, family="", qualifier=None):
        if name is None:
            return self.vector_field(
                space_or_name,
                components,
                _family_from_qualifier(qualifier) or family,
            )
        with geometric_dimension_context(self.dim):
            return self._register_external_field(VectorFunction(space_or_name, name, qualifier=qualifier))

    def tensor_field(self, name, shape, family=""):
        field = tensor_field(name, shape, family, {"dim": self.dim})
        return self._register_symbolic_field(field, math.prod(field.shape))

    def TensorFunction(self, space_or_name, name=None, shape=None, family="", qualifier=None):
        if name is None:
            return self.tensor_field(
                space_or_name,
                shape,
                _family_from_qualifier(qualifier) or family,
            )
        return self._register_external_field(TensorFunction(space_or_name, name, qualifier=qualifier))

    def equation(
        self,
        name,
        form,
        define,
        *,
        fields=(),
        variables=(),
        directions=(),
        kernels=(),
        diagnostics=True,
    ):
        return self._system.equation(
            name,
            form,
            define,
            fields=self._resolve_fields(fields),
            variables=_symbols_from_variables(variables),
            directions=_symbols_from_variables(directions),
            kernels=kernels,
            diagnostics=diagnostics,
        )

    def add_energy(
        self,
        name,
        define,
        *,
        fields=(),
        variables=None,
        directions=None,
        kernels=("objective", "gradient", "apply"),
        diagnostics=True,
    ):
        if variables is None:
            raise ValueError("energy requires explicit variables")
        _validate_energy_variable_groups(fields, variables)
        variable_symbols = _symbols_from_variables(variables)
        if not variable_symbols:
            raise ValueError("energy requires explicit variables")
        direction_symbols = () if directions is None else _symbols_from_variables(directions)
        return self._system.add_energy(
            name,
            _expression_value(define),
            fields=self._resolve_fields(fields),
            variables=variables,
            directions=direction_symbols,
            kernels=kernels,
            diagnostics=diagnostics,
        )

    def add_residual(self, name, define, *, fields=()):
        symbolic_fields = tuple(fields)
        return self._system.add_residual(
            name,
            _residual_define(define, symbolic_fields, self.dim),
            fields=self._resolve_fields(fields),
        )

    def derive_energy_forms(
        self,
        expression,
        *,
        variables=None,
        fields=None,
        directions=None,
        orders=None,
    ):
        if variables is None:
            if fields is None:
                raise ValueError("derive_energy_forms requires variables")
            variables = fields
        variables = _symbols_from_variables(variables)
        if directions is None:
            directions = _default_direction_symbols(variables)
        else:
            directions = _symbols_from_variables(directions)
        pipeline = energy_form_pipeline(
            _expression_value(expression),
            variables,
            directions,
        )
        if orders is None:
            return pipeline.evaluate()
        return pipeline.evaluate(orders)

    def derive_merit_forms(
        self,
        expression,
        *,
        variables=None,
        fields=None,
        directions=None,
        orders=None,
    ):
        return self.derive_energy_forms(
            expression,
            variables=variables,
            fields=fields,
            directions=directions,
            orders=orders,
        )

    def derive_residual_forms(self, residual, *, fields, directions=None, orders=None):
        variables = _symbols_from_fields(fields)
        if directions is None:
            directions = tuple(
                symbol
                for field in fields
                for symbol in trial_function(field).symbols
            )
        else:
            directions = _symbols_from_fields(directions)
        pipeline = residual_form_pipeline(
            _residual_value(residual),
            variables,
            directions,
        )
        if orders is None:
            return pipeline.evaluate()
        return pipeline.evaluate(orders)

    def derive_gradient_forms(self, residual, *, fields, directions=None, orders=None):
        return self.derive_residual_forms(
            residual,
            fields=fields,
            directions=directions,
            orders=orders,
        )

    def equation_field(self, field):
        if isinstance(field, EquationField):
            return field
        name = field.name if isinstance(field, SymbolicField) else str(field)
        try:
            return self._equation_fields_by_name[name]
        except KeyError:
            if isinstance(field, SymbolicField):
                self._register_external_field(field)
                return self._equation_fields_by_name[name]
            raise ValueError("unknown equation field '%s'" % name)

    def _register_symbolic_field(self, symbolic, components):
        if symbolic.name in self._equation_fields_by_name:
            raise ValueError("equation field '%s' is already registered" % symbolic.name)
        equation_field = self._system.field(
            symbolic.name,
            components,
            symbolic.family,
            symbolic.metadata,
        )
        self._symbolic_fields.append(symbolic)
        self._equation_fields_by_name[symbolic.name] = equation_field
        return symbolic

    def _register_external_field(self, symbolic):
        if not isinstance(symbolic, SymbolicField):
            raise TypeError("external equation fields must be symbolic fields")
        components = math.prod(symbolic.shape) if symbolic.shape else 1
        return self._register_symbolic_field(symbolic, components)

    def _resolve_fields(self, fields):
        return tuple(self.equation_field(field) for field in fields)


def _symbols_from_fields(fields):
    symbols = []
    for field in fields:
        if isinstance(field, (SymbolicField, ScalarField, VectorField, TensorField)):
            symbols.extend(field.symbols)
        elif hasattr(field, "symbols"):
            symbols.extend(field.symbols)
        else:
            symbols.append(sp.sympify(field))
    return tuple(symbols)


def _build_form_collection(system, equation, orders):
    if equation.is_energy:
        variables = tuple(equation.variables)
        directions = tuple(equation.directions)
        if FormOrder.TWO in orders and not directions:
            directions = _default_direction_symbols(variables)
        evaluation = energy_form_pipeline(
            equation.define,
            variables,
            directions or None,
        ).evaluate(orders)
        metadata = _evaluation_metadata(evaluation)
        return FormCollection.from_evaluation(
            equation.name,
            evaluation,
            fields=equation.fields,
            variables=variables,
            directions=directions,
            qualifiers=_equation_qualifiers(equation),
            dependencies=tuple(entry.dependencies for entry in metadata),
            metadata=metadata,
        )
    if equation.is_residual:
        residual_system = CoupledResidualSystem(system.dim)
        equation.define(residual_system)
        residual_vector = sp.Matrix(
            [
                residual_system.residual_expression(field)
                for field in residual_system.fields
            ]
        )
        variables = tuple(
            symbol
            for field in residual_system.fields
            for symbol in field.variables
        )
        directions = tuple(
            symbol
            for field in residual_system.fields
            for symbol in field.directions
        )
        evaluation = residual_form_pipeline(
            residual_vector,
            variables,
            directions,
        ).evaluate(orders)
        residual_metadata = []
        if FormOrder.ZERO in orders:
            residual_metadata.append(FormMetadata(FormOrder.ZERO))
        if FormOrder.ONE in orders:
            blocks = _residual_row_blocks(residual_system)
            residual_metadata.append(
                FormMetadata(
                    FormOrder.ONE,
                    coefficients=coupled_residual_weak_coefficients(
                        residual_system,
                        False,
                    ),
                    dependencies=residual_system.residual_dependencies(),
                    blocks=blocks,
                )
            )
        if FormOrder.TWO in orders:
            blocks = _residual_jacobian_blocks(residual_system)
            residual_metadata.append(
                FormMetadata(
                    FormOrder.TWO,
                    coefficients=coupled_residual_weak_coefficients(
                        residual_system,
                        True,
                    ),
                    dependencies=residual_system.jacobian_action_dependencies(),
                    blocks=blocks,
                )
            )
        blocks = tuple(
            block
            for metadata in residual_metadata
            for block in metadata.blocks
        )
        return FormCollection.from_evaluation(
            equation.name,
            evaluation,
            fields=equation.fields,
            variables=variables,
            directions=directions,
            coefficients=tuple(
                metadata.coefficients
                for metadata in residual_metadata
                if metadata.coefficients
            ),
            dependencies=residual_system.residual_dependencies()
            if FormOrder.ONE in orders
            else None,
            blocks=blocks,
            qualifiers=_equation_qualifiers(equation),
            source=residual_system,
            metadata=tuple(residual_metadata),
        )
    raise TypeError("unsupported equation form %s" % equation.form)


def _residual_row_blocks(residual_system):
    return tuple(
        FormBlock(
            FormOrder.ONE,
            row_field=field.name,
            expression=residual_system.residual_expression(field),
            coefficients=(
                weak_residual_coefficients(
                    residual_system,
                    residual_system.residual_expression(field),
                    field.name,
                ),
            ),
            dependencies=residual_system.dependencies_for_expressions(
                (residual_system.residual_expression(field),)
            ),
        )
        for field in residual_system.fields
    )


def _residual_jacobian_blocks(residual_system):
    return tuple(
        FormBlock(
            FormOrder.TWO,
            row_field=block.row_field,
            column_field=block.column_field,
            expression=block.expression,
            coefficients=(
                weak_residual_coefficients(
                    residual_system,
                    block.expression,
                    block.row_field,
                ),
            ),
            dependencies=residual_system.dependencies_for_expressions(
                (block.expression,)
            ),
        )
        for block in residual_system.jacobian_blocks()
    )


def _equation_qualifiers(equation):
    ret = []
    for field in equation.fields:
        if field.family:
            ret.append(FormQualifier(field.name, "field_family", field.family))
        ret.append(FormQualifier(field.name, "field_components", field.components))
    return tuple(ret)


def _evaluation_metadata(evaluation):
    return tuple(
        FormMetadata(
            form.order,
            dependencies=_free_symbols(form.expression),
        )
        for form in evaluation.forms
    )


def _free_symbols(expression):
    if isinstance(expression, sp.MatrixBase):
        symbols = set()
        for entry in expression:
            symbols.update(sp.sympify(entry).free_symbols)
        return tuple(sorted(symbols, key=str))
    return tuple(sorted(sp.sympify(expression).free_symbols, key=str))


def _validate_energy_variable_groups(fields, variables):
    if variables is None:
        raise ValueError("energy requires explicit variables")
    field_groups = tuple(fields)
    variable_groups = tuple(variables) if isinstance(variables, (tuple, list)) else None
    if variable_groups is None:
        raise TypeError("energy variables must be a tuple/list with one entry per field")
    if len(variable_groups) != len(field_groups):
        raise ValueError(
            "energy variables must provide one differentiating variable group per field"
        )


def _symbols_from_variables(variables):
    if variables is None:
        return ()
    if isinstance(variables, (SymbolicField, ScalarField, VectorField, TensorField)):
        return tuple(variables.symbols)
    if isinstance(variables, (tuple, list)):
        symbols = []
        for variable in variables:
            symbols.extend(_symbols_from_variables(variable))
        return tuple(symbols)
    value = _expression_value(variables)
    if isinstance(value, sp.MatrixBase):
        return tuple(value)
    if isinstance(value, sp.NDimArray):
        return tuple(value)
    if hasattr(variables, "symbols"):
        return tuple(variables.symbols)
    return (sp.sympify(variables),)


def _default_direction_symbols(variables):
    directions = []
    for variable in variables:
        name = str(variable)
        if name.startswith("F["):
            directions.append(sp.Symbol("d%s" % name))
        elif "[" in name:
            directions.append(sp.Symbol("d_%s" % name))
        else:
            directions.append(sp.Symbol("%s_trial" % name))
    return tuple(directions)


def _expression_value(expression):
    if hasattr(expression, "value"):
        return expression.value
    return sp.sympify(expression)


def _residual_value(residual):
    if isinstance(residual, sp.MatrixBase):
        return residual
    if isinstance(residual, (tuple, list)):
        return sp.Matrix([_expression_value(entry) for entry in residual])
    return sp.Matrix([_expression_value(residual)])


def _residual_define(define, fields, dim):
    if callable(define):
        return define
    expression = _expression_value(define)

    def evaluate(system):
        if not isinstance(system, CoupledResidualSystem):
            raise TypeError("residual expression lowering requires CoupledResidualSystem")
        residual_fields, substitutions = _lower_residual_fields(system, fields, dim)
        lowered = expression.xreplace(substitutions)
        parameters = tuple(
            symbol
            for symbol in sorted(lowered.free_symbols.difference(system.registered_symbols()), key=str)
            if isinstance(symbol, sp.Symbol)
        )
        if parameters:
            system.add_parameters(*parameters)
        for residual_field in residual_fields:
            row_expression = _extract_row_weak_form(lowered, residual_field)
            system.add_residual(residual_field, row_expression)

    return evaluate


def _lower_residual_fields(system, fields, dim):
    residual_fields = []
    substitutions = {}
    for field in fields:
        if isinstance(field, ScalarField):
            lowered = system.add_field(field.name)
            residual_fields.append(lowered)
            _map_scalar_field_symbols(substitutions, field, lowered)
        elif isinstance(field, VectorField):
            tests = test_function(field)
            trials = trial_function(field)
            previous = trial_function(field, name="%s_old" % field.name)
            for component in range(field.dim):
                lowered = system.add_field("%s%d" % (field.name, component))
                residual_fields.append(lowered)
                _map_vector_component_symbols(
                    substitutions,
                    field,
                    tests,
                    trials,
                    previous,
                    component,
                    lowered,
                    dim,
                )
        else:
            raise ValueError("residual expression fields must be scalar or vector fields")
    return tuple(residual_fields), substitutions


def _map_scalar_field_symbols(substitutions, field, lowered):
    substitutions[field.value] = lowered.value
    substitutions[trial_function(field).value] = lowered.direction_value
    substitutions[test_function(field).value] = lowered.test_value
    substitutions[trial_function(field, name="%s_old" % field.name).value] = lowered.previous_value
    for d in range(lowered.dim):
        substitutions[sp.Symbol("%s_grad[%d]" % (field.name, d))] = lowered.gradient[d]
        substitutions[sp.Symbol("%s_trial_grad[%d]" % (field.name, d))] = lowered.direction_gradient[d]
        substitutions[sp.Symbol("%s_test_grad[%d]" % (field.name, d))] = lowered.test_gradient[d]
        substitutions[sp.Symbol("%s_old_grad[%d]" % (field.name, d))] = lowered.previous_gradient[d]


def _map_vector_component_symbols(
    substitutions,
    field,
    tests,
    trials,
    previous,
    component,
    lowered,
    dim,
):
    substitutions[field[component]] = lowered.value
    substitutions[tests[component]] = lowered.test_value
    substitutions[trials[component]] = lowered.direction_value
    substitutions[previous[component]] = lowered.previous_value
    for d in range(dim):
        flat = component * dim + d
        substitutions[sp.Symbol("%s_grad[%d]" % (field.name, flat))] = lowered.gradient[d]
        substitutions[sp.Symbol("%s_grad[%d]" % (tests.name, flat))] = lowered.test_gradient[d]
        substitutions[sp.Symbol("%s_grad[%d]" % (trials.name, flat))] = lowered.direction_gradient[d]
        substitutions[sp.Symbol("%s_grad[%d]" % (previous.name, flat))] = lowered.previous_gradient[d]


def _extract_row_weak_form(expression, residual_field):
    ret = sp.diff(expression, residual_field.test_value) * residual_field.test_value
    for symbol in residual_field.test_gradient:
        ret += sp.diff(expression, symbol) * symbol
    return sp.simplify(ret)
