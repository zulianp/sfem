"""Prony-series relaxation as an extension of any hyperelastic energy.

A Simo-type viscoelastic material splits its strain energy into an
equilibrium part psi_e, never relaxed, and a relaxed part psi_r, whose stress
S_r = 2 dpsi_r/dC relaxes through a Prony series.  Within one time step the
algorithmic second Piola-Kirchhoff stress is

    S = S_e + gamma S_r + S_hist,

with gamma and S_hist fixed for the step: gamma is a scalar weight of the
integrator and S_hist a symmetric tensor per integration point built from the
history the step starts from.  So the step is the minimisation of

    W = psi_e + gamma psi_r + 1/2 S_hist : (C - I),

whose first Piola-Kirchhoff stress is dW/dF = F (S_e + gamma S_r + S_hist).
That is an ordinary hyperelastic energy with one scalar parameter and one
integration-point field, and it lowers through the energy pipeline like any
other -- the consistent tangent included, with nothing derived by hand.

What the step energy does not contain is the history update, which needs S_r
at the converged state.  `relaxed_stress` is that quantity, published as an
integration-point output of the same material so the generated kernel
evaluates it at the same points the energy kernels read S_hist at.

Everything else -- the spectrum, the time-temperature shift, the recursion,
the storage of H_i -- is material independent and lives in SFEM
(`frontend/ops/prony`).  The kernels never see the number of terms.
"""

from dataclasses import dataclass

import sympy as sp

from codegen.framework.symbolic.fields import QuadratureField, symmetric_tensor_entries


RELAXATION_GAMMA_SUFFIX = "_relaxation_gamma"


def relaxation_gamma_name(group):
    """The step weight's parameter name, spelled in one place."""
    return "%s%s" % (group, RELAXATION_GAMMA_SUFFIX)


def is_relaxation_gamma(name):
    """Whether a kernel's scalar argument is a relaxation group's step weight.

    It is not a material parameter: the operator always holds the group's
    relaxation state and passes its weight, which is 1 for a group with no
    spectrum.
    """
    return str(name).endswith(RELAXATION_GAMMA_SUFFIX)


def relaxation_group_of_gamma(name):
    """The group a step weight's parameter name belongs to."""
    name = str(name)
    if not is_relaxation_gamma(name):
        raise ValueError("'%s' is not a relaxation step weight" % name)
    return name[: -len(RELAXATION_GAMMA_SUFFIX)]


@dataclass(frozen=True)
class RelaxationDeclaration:
    """What packaging needs to bind one relaxation group to its runtime object."""

    group: str
    gamma: str
    history: str
    stress: str
    n_components: int


class PronyRelaxation:
    """One relaxation group: a relaxed energy and the Prony series that relaxes it.

    The declaration side of SFEM's runtime `PronyRelaxation`, which holds the
    group's spectrum and history per block.  `relax(psi_r, F)` returns the
    group's contribution to the step energy and records what the group needs;
    the group is then handed to `add_energy(..., relaxations=...)`.  A material
    may hold several groups -- an isochoric and a volumetric spectrum, say --
    each with its own weight, its own history and its own output.
    """

    def __init__(self, group):
        group = str(group)
        if not group.isidentifier():
            raise ValueError("relaxation group name must be an identifier")
        self.group = group
        self.gamma = sp.Symbol(relaxation_gamma_name(group))
        self._history = None
        self._stress = None

    @property
    def history_name(self):
        return "%s_history" % self.group

    @property
    def stress_name(self):
        return "%s_stress" % self.group

    def relax(self, psi_r, F):
        """The step-energy contribution of relaxing `psi_r`, a function of `F`.

        `F` is the deformation-gradient variable the energy is written in, so
        the relaxed stress is taken with respect to the same symbols the
        pipeline differentiates.
        """
        if self._history is not None:
            raise ValueError("relaxation group '%s' already relaxes an energy" % self.group)

        F_value = _matrix(F)
        dim = F_value.rows
        history = QuadratureField(self.history_name, dim)
        self._history = history
        self._stress = _second_piola(psi_r, F_value)

        C = F_value.T * F_value
        identity = sp.eye(dim)
        work = sum(
            history.value[i, j] * (C[i, j] - identity[i, j])
            for i in range(dim)
            for j in range(dim)
        )
        return self.gamma * psi_r + sp.Rational(1, 2) * work

    def declaration(self):
        self._require_relaxed()
        return RelaxationDeclaration(
            group=self.group,
            gamma=str(self.gamma),
            history=self.history_name,
            stress=self.stress_name,
            n_components=self._history.n_components,
        )

    def quadrature_outputs(self):
        """{name: components} of what this group's history update reads."""
        self._require_relaxed()
        return {self.stress_name: self._stress}

    def _require_relaxed(self):
        if self._history is None:
            raise ValueError("relaxation group '%s' has not relaxed an energy yet" % self.group)


def _matrix(F):
    value = getattr(F, "value", F)
    if not isinstance(value, sp.MatrixBase) or value.rows != value.cols:
        raise TypeError("relax(...) needs the square deformation-gradient variable")
    return value


def _second_piola(psi, F):
    """The symmetric entries of S = F^{-1} dpsi/dF, in the stored order.

    Through the first Piola-Kirchhoff stress rather than 2 dpsi/dC, because the
    energies are written in F.  S is symmetric for any objective energy, so the
    upper triangle is all there is to store.
    """
    dim = F.rows
    P = sp.Matrix(dim, dim, lambda i, j: sp.diff(psi, F[i, j]))
    S = F.adjugate() * P / F.det()
    return tuple(S[i, j] for i, j in symmetric_tensor_entries(dim))
