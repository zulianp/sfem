from dataclasses import dataclass
from enum import Enum


class EquationForm(Enum):
    ENERGY = "energy"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class EquationField:
    name: str
    components: int = 1
    family: str = ""

    def __post_init__(self):
        name = str(self.name)
        family = str(self.family)
        components = int(self.components)
        if not name or not name.isidentifier():
            raise ValueError("equation field name must be a valid identifier")
        if components <= 0:
            raise ValueError("equation field components must be positive")
        if family and not family.isidentifier():
            raise ValueError("equation field family must be a valid identifier")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "family", family)

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
    kernels: tuple = ()
    diagnostics: bool = True

    def __post_init__(self):
        name = str(self.name)
        if name and not name.isidentifier():
            raise ValueError("equation name must be empty or a valid identifier")
        if not callable(self.define):
            raise TypeError("equation definition must be callable")
        fields = tuple(self.fields)
        if not all(isinstance(field, EquationField) for field in fields):
            raise TypeError("equation fields must be EquationField instances")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "form", EquationForm(self.form))
        object.__setattr__(self, "fields", fields)
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

    @property
    def fields(self):
        return tuple(self._fields)

    @property
    def equations(self):
        return tuple(self._equations)

    def field(self, name, components=1, family=""):
        field = EquationField(name, components, family)
        if any(existing.name == field.name for existing in self._fields):
            raise ValueError("equation field '%s' is already registered" % field.name)
        self._fields.append(field)
        return field

    def scalar_field(self, name, family=""):
        return self.field(name, 1, family)

    def vector_field(self, name, components=None, family=""):
        return self.field(name, self.dim if components is None else components, family)

    def equation(
        self,
        name,
        form,
        define,
        *,
        fields=(),
        kernels=(),
        diagnostics=True,
    ):
        equation = Equation(
            name,
            form,
            define,
            fields=tuple(fields),
            kernels=tuple(kernels),
            diagnostics=diagnostics,
        )
        if equation.name and any(existing.name == equation.name for existing in self._equations):
            raise ValueError("equation '%s' is already registered" % equation.name)
        self._equations.append(equation)
        return equation

    def energy(
        self,
        name,
        define,
        *,
        fields=(),
        kernels=("objective", "gradient", "apply"),
        diagnostics=True,
    ):
        return self.equation(
            name,
            EquationForm.ENERGY,
            define,
            fields=fields,
            kernels=kernels,
            diagnostics=diagnostics,
        )

    def residual(self, name, define, *, fields=()):
        return self.equation(
            name,
            EquationForm.RESIDUAL,
            define,
            fields=fields,
        )

    def hyperelastic(
        self,
        name,
        define,
        *,
        fields=(),
        kernels=("objective", "gradient", "apply"),
        diagnostics=True,
    ):
        return self.energy(
            name,
            define,
            fields=fields,
            kernels=kernels,
            diagnostics=diagnostics,
        )
