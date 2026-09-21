"""SoA weak-form lowering: energy density to gradient and Hessian-action forms.

This is form lowering -- differentiation of an energy against a deformation
gradient and assembly of the resulting weak coefficients.  It sits above the
symbolic objects it manipulates and below anything that knows about elements,
targets, or text.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import sympy as sp

from codegen.framework.symbolic.core import (
    _as_matrix,
    _check_scalar_expression,
    directional_derivative,
    matrix_inner,
)


@dataclass(frozen=True)
class SfemSoAKernelForm:
    name: str
    expression_graph: Optional["ExpressionGraph"] = None
    has_direction: bool = False
    output_mode: str = "accumulate"
    weak_form: Optional["SfemSoAWeakForm"] = None
    dependencies: object = None

    def __post_init__(self):
        if self.output_mode not in ("assign", "accumulate"):
            raise ValueError("output_mode must be 'assign' or 'accumulate'")
        if self.expression_graph is None and self.weak_form is None:
            raise ValueError("SfemSoAKernelForm requires expression_graph or weak_form")


def sfem_soa_kernel_form(
    name,
    expression_graph=None,
    has_direction=False,
    output_mode="accumulate",
    weak_form=None,
    dependencies=None,
):
    return SfemSoAKernelForm(name, expression_graph, has_direction, output_mode, weak_form, dependencies)


@dataclass(frozen=True)
class SfemSoAWeakForm:
    """An energy density and the field gradient it is differentiated against.

    The gradient is ``n_field_components x dim``: one row per component of the field,
    one column per spatial direction.  A displacement in `dim` dimensions makes
    that square and it is the deformation gradient, which is the only shape this
    held for a long time.  A scalar field makes it ``1 x dim`` -- the gradient of
    a potential like `kappa/2 * ||grad u||^2`, whose first Piola is the flux
    `kappa * grad u`.

    Nothing else about the lowering changes with the shape, which is why the
    restriction was worth removing rather than working around: the derivative of
    a scalar against a matrix of symbols does not care whether that matrix is
    square, and neither does the contraction against test gradients below it.
    """

    energy_density: sp.Expr
    deformation_gradient: Tuple[sp.Expr, ...]
    dim: int
    n_field_components: Optional[int] = None

    def __post_init__(self):
        dim = int(self.dim)
        deformation_gradient = tuple(self.deformation_gradient)
        n_field_components = dim if self.n_field_components is None else int(self.n_field_components)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_field_components", n_field_components)
        object.__setattr__(self, "energy_density", sp.sympify(self.energy_density))
        object.__setattr__(self, "deformation_gradient", deformation_gradient)
        if dim <= 0:
            raise ValueError("weak form dim must be positive")
        if n_field_components <= 0:
            raise ValueError("weak form n_field_components must be positive")
        if len(deformation_gradient) != n_field_components * dim:
            raise ValueError(
                "deformation_gradient must have n_field_components * dim entries"
            )

    @property
    def is_deformation_gradient(self):
        """Whether the variable is `I + grad(u)` rather than `grad(u)` itself.

        Inferred from the shape: a square gradient in this framework is always a
        displacement's deformation gradient, and the identity belongs in it.  A
        scalar field's `1 x dim` gradient has no identity to add.

        The shape is a proxy.  What actually settles it is the qualifier the
        material used -- `gen.deformation_gradient(u)` against `gen.grad(u)` --
        and that does not reach here.  Every material today is square and uses
        the former, so the proxy is exact for all of them; a vector field whose
        variable is a plain gradient would be the case that breaks it, and the
        fix then is to carry the qualifier down rather than to guess better.
        """
        return self.n_field_components == self.dim

    def deformation_gradient_matrix(self):
        return sp.Matrix(self.n_field_components, self.dim, self.deformation_gradient)

    def first_piola(self):
        variables = self.deformation_gradient
        return sp.Matrix(
            self.n_field_components,
            self.dim,
            [sp.diff(self.energy_density, variable) for variable in variables],
        )

    def linearized_first_piola(self, trial_gradient):
        trial_gradient = tuple(trial_gradient)
        if len(trial_gradient) != self.n_field_components * self.dim:
            raise ValueError(
                "trial_gradient must have n_field_components * dim entries"
            )
        P = self.first_piola()
        variables = self.deformation_gradient
        return sp.Matrix(
            self.n_field_components,
            self.dim,
            [
                directional_derivative(P[i, j], variables, trial_gradient)
                for i in range(self.n_field_components)
                for j in range(self.dim)
            ],
        )

    def diagnostic_expressions(self, has_direction=False):
        expressions = [self.energy_density]
        expressions.extend(tuple(self.first_piola()))
        if has_direction:
            trial_gradient = tuple(
                sp.symbols("trial_grad[%d]" % i)
                for i in range(self.n_field_components * self.dim)
            )
            expressions.extend(tuple(self.linearized_first_piola(trial_gradient)))
        return tuple(expressions)


@dataclass(frozen=True)
class SfemSoAFluxForm:
    """What a form contracts against a test function: a flux and a source.

    ``flux`` is ``n_field_components x dim`` -- one row per lowered field
    component, one column per spatial direction -- and is contracted against
    the test *gradient*.  ``source`` is one entry per component and is
    contracted against the test *value*.  Together they reconstruct the weak
    form exactly::

        R_i = sum_j flux[i][j] * test_grad_j  +  source[i] * test_value

    Both front ends reach this object, and that is the whole point of it
    existing.  An energy formulation differentiates its density against the
    field gradient, which is the first Piola stress.  A residual formulation
    differentiates the weak form against the test symbols, which is exact
    rather than approximate: a weak form is linear in its test function by
    construction, so the derivative is the coefficient and nothing is lost.
    Verified on every residual material in the tree -- laplace, stokes,
    navier_stokes, two_phase_flow and neumann -- including the ones whose flux
    is a nonlinear function of the state, where the *test* dependence is still
    linear and so the extraction is still exact.

    This is what lets sum factorization serve a residual formulation.  Sum
    factorization needs a strong quantity to contract against test gradients,
    and a weak form has already contracted; recovering the flux undoes that
    contraction symbolically, once, in this layer, so that nothing below has to
    know which front end the material was written in.
    """

    flux: Tuple[sp.Expr, ...]
    source: Tuple[sp.Expr, ...]
    gradient: Tuple[sp.Expr, ...]
    dim: int
    n_field_components: int
    #: Whether ``gradient`` holds `I + grad(u)` rather than `grad(u)`.  It
    #: changes what the flux is a function of, so any predicate that asks how
    #: the flux depends on the gradient has to know.  A residual formulation
    #: differentiates against true test gradients and is never this.
    is_deformation_gradient: bool = False
    #: The gradient of the *previous* state, where the form has one -- a rate
    #: dependent material such as Kelvin-Voigt reads it.  Laid out like
    #: ``gradient``, and empty when the form does not.  It is field data, one
    #: value per element per component per direction, and keeping it here is
    #: what stops it being mistaken for a material parameter: a uniform and a
    #: per-element field need different kernel arguments, and the difference is
    #: not recoverable from the symbol name.
    previous_gradient: Tuple[sp.Expr, ...] = ()

    def __post_init__(self):
        dim = int(self.dim)
        n_field_components = int(self.n_field_components)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_field_components", n_field_components)
        object.__setattr__(self, "flux", tuple(sp.sympify(e) for e in self.flux))
        object.__setattr__(self, "source", tuple(sp.sympify(e) for e in self.source))
        object.__setattr__(self, "gradient", tuple(self.gradient))
        object.__setattr__(
            self, "is_deformation_gradient", bool(self.is_deformation_gradient)
        )
        object.__setattr__(self, "previous_gradient", tuple(self.previous_gradient))
        if self.previous_gradient and len(self.previous_gradient) != n_field_components * dim:
            raise ValueError(
                "previous_gradient must be empty or have n_field_components * dim entries"
            )
        if dim <= 0:
            raise ValueError("flux form dim must be positive")
        if n_field_components <= 0:
            raise ValueError("flux form n_field_components must be positive")
        if len(self.flux) != n_field_components * dim:
            raise ValueError("flux must have n_field_components * dim entries")
        if len(self.source) != n_field_components:
            raise ValueError("source must have one entry per field component")
        if len(self.gradient) != n_field_components * dim:
            raise ValueError("gradient must have n_field_components * dim entries")

    def flux_matrix(self):
        return sp.Matrix(self.n_field_components, self.dim, self.flux)

    @property
    def parameters(self):
        """The material parameters the flux and source mention, sorted.

        The form describing itself, so that nothing downstream has to reach for
        `free_symbols` to find out what a kernel's signature needs -- which is
        a planning question and one an emitter must not be answering.
        """
        state = {str(symbol) for symbol in self.gradient}
        state |= {str(symbol) for symbol in self.previous_gradient}
        return tuple(
            sorted(
                {
                    str(symbol)
                    for entry in tuple(self.flux) + tuple(self.source)
                    for symbol in entry.free_symbols
                }
                - state
            )
        )

    @property
    def has_source(self):
        """Whether anything contracts against the test value.

        A pure diffusion has none, and a kernel for it reads no test values and
        needs no determinant to weight them -- which is the distinction
        ``plans.geometry_quantities`` already draws.
        """
        return any(entry != 0 for entry in self.source)

    def linearized_flux(self, trial_gradient):
        """The flux's directional derivative along a trial field gradient."""
        trial_gradient = tuple(trial_gradient)
        if len(trial_gradient) != self.n_field_components * self.dim:
            raise ValueError(
                "trial_gradient must have n_field_components * dim entries"
            )
        return sp.Matrix(
            self.n_field_components,
            self.dim,
            [
                directional_derivative(entry, self.gradient, trial_gradient)
                for entry in self.flux
            ],
        )

    def weak_form_expression(self, test_gradient, test_value=None):
        """The weak form this flux contracts to, as a check on the extraction."""
        test_gradient = tuple(test_gradient)
        if len(test_gradient) != self.n_field_components * self.dim:
            raise ValueError(
                "test_gradient must have n_field_components * dim entries"
            )
        expression = sum(
            (flux * test for flux, test in zip(self.flux, test_gradient)),
            sp.Integer(0),
        )
        if test_value is not None:
            expression = expression + sum(
                (source * test_value for source in self.source), sp.Integer(0)
            )
        return sp.expand(expression)


def flux_form_from_energy(weak_form):
    """The flux an energy density implies: its first Piola stress, no source.

    An energy contributes nothing to the test *value* -- differentiating a
    density against a field gradient can only produce a gradient coefficient --
    so the source is zero and the two front ends still meet in the same shape.
    """
    return SfemSoAFluxForm(
        tuple(weak_form.first_piola()),
        tuple(sp.Integer(0) for _ in range(weak_form.n_field_components)),
        tuple(weak_form.deformation_gradient),
        weak_form.dim,
        weak_form.n_field_components,
        is_deformation_gradient=bool(weak_form.is_deformation_gradient),
    )


def flux_form_from_residual(residual_expressions, fields, dim):
    """The flux a residual weak form implies, by differentiating out the test.

    ``fields`` are the lowered residual field records, in the same order as
    ``residual_expressions``; each carries the test gradient and test value
    symbols this differentiates against, and the field gradient symbols the
    linearization is taken along.

    The extraction is checked rather than assumed: a weak form must be linear
    in every test symbol for its derivative to be the coefficient, so that is
    verified here and a form that fails it is refused by name instead of
    silently losing its second-order part.
    """
    dim = int(dim)
    residual_expressions = tuple(sp.sympify(e) for e in residual_expressions)
    fields = tuple(fields)
    if len(residual_expressions) != len(fields):
        raise ValueError("one residual expression per lowered field is required")
    flux = []
    source = []
    gradient = []
    previous_gradient = []
    for expression, field in zip(residual_expressions, fields):
        test_gradient = tuple(field.test_gradient)
        if len(test_gradient) != dim:
            raise ValueError(
                "field '%s' has %d test gradient symbols; expected %d"
                % (field.name, len(test_gradient), dim)
            )
        test_value = field.test_value
        symbols = test_gradient + ((test_value,) if test_value is not None else ())
        for symbol in symbols:
            if sp.diff(expression, symbol, 2) != 0:
                raise ValueError(
                    "residual for field '%s' is not linear in test symbol '%s'; "
                    "a weak form must be, and its flux cannot be recovered "
                    "otherwise" % (field.name, symbol)
                )
        flux.extend(sp.diff(expression, symbol) for symbol in test_gradient)
        source.append(
            sp.diff(expression, test_value) if test_value is not None else sp.Integer(0)
        )
        gradient.extend(field.gradient)
        previous_gradient.extend(field.previous_gradient)
    # A field records its previous gradient whether or not the form reads it;
    # carry it only when the flux actually does, so a rate-independent material
    # is not handed an unused state stream.
    mentioned = set()
    for entry in flux + source:
        mentioned |= entry.free_symbols
    if not mentioned.intersection(previous_gradient):
        previous_gradient = []
    return SfemSoAFluxForm(
        tuple(flux), tuple(source), tuple(gradient), dim, len(fields),
        previous_gradient=tuple(previous_gradient),
    )


def sfem_soa_weak_form(energy_density, deformation_gradient):
    """Lower an energy density against a field gradient of any shape.

    Rows are field components and columns are spatial directions.  This used to
    demand a square matrix, which is the deformation gradient of a
    `dim`-component displacement and excluded every scalar field -- a Laplacian
    potential among them, whose gradient is `1 x dim`.
    """
    deformation_gradient = _as_matrix(deformation_gradient, "deformation_gradient")
    rows, cols = deformation_gradient.shape
    return SfemSoAWeakForm(
        energy_density,
        tuple(deformation_gradient),
        cols,
        n_field_components=rows,
    )
