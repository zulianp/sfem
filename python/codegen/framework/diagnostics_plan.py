from dataclasses import dataclass

from .forms import FormOrder


@dataclass(frozen=True)
class KernelDiagnosticsEntryPlan:
    public_name: str
    expression_name: str
    form_order: FormOrder
    dependencies: object
    cost: object = None
    mesh_signature: object = None
    local_signature: object = None
    reference_dataset: object = None
    block_name: str = ""

    def __post_init__(self):
        public_name = str(self.public_name)
        expression_name = str(self.expression_name)
        block_name = str(self.block_name)
        if not public_name:
            raise ValueError("diagnostics entry requires a public name")
        if not expression_name:
            raise ValueError("diagnostics entry requires an expression name")
        object.__setattr__(self, "public_name", public_name)
        object.__setattr__(self, "expression_name", expression_name)
        object.__setattr__(self, "form_order", FormOrder(self.form_order))
        object.__setattr__(self, "block_name", block_name)

    @property
    def uses_current(self):
        return bool(getattr(self.dependencies, "current", False))

    @property
    def uses_previous(self):
        return bool(getattr(self.dependencies, "previous", False))

    @property
    def uses_direction(self):
        return bool(getattr(self.dependencies, "direction", False))

    @property
    def parameter_count(self):
        return len(tuple(getattr(self.dependencies, "parameters", ())))

    def to_dict(self):
        return {
            "public_name": self.public_name,
            "expression_name": self.expression_name,
            "form_order": self.form_order.value,
            "block_name": self.block_name,
            "uses_current": self.uses_current,
            "uses_previous": self.uses_previous,
            "uses_direction": self.uses_direction,
            "parameter_count": self.parameter_count,
            "mesh_signature": None
            if self.mesh_signature is None
            else self.mesh_signature.name,
            "local_signature": None
            if self.local_signature is None
            else self.local_signature.name,
            "reference_stage": None
            if self.reference_dataset is None
            else self.reference_dataset.stage,
        }


@dataclass(frozen=True)
class KernelDiagnosticsPlan:
    prefix: str
    kind: str
    entries: tuple

    def __post_init__(self):
        prefix = str(self.prefix)
        kind = str(self.kind)
        entries = tuple(self.entries)
        if not prefix:
            raise ValueError("diagnostics plan requires a prefix")
        names = set()
        for entry in entries:
            if not isinstance(entry, KernelDiagnosticsEntryPlan):
                raise TypeError("diagnostics entries must be KernelDiagnosticsEntryPlan objects")
            if entry.public_name in names:
                raise ValueError("duplicate diagnostics entry '%s'" % entry.public_name)
            names.add(entry.public_name)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "entries", entries)

    @property
    def public_names(self):
        return tuple(entry.public_name for entry in self.entries)

    def entry(self, public_name):
        public_name = str(public_name)
        for entry in self.entries:
            if entry.public_name == public_name:
                return entry
        raise ValueError("diagnostics entry '%s' is not available" % public_name)

    def to_dict(self):
        return {
            "prefix": self.prefix,
            "kind": self.kind,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def kernel_diagnostics_plan_from_plan(
    unit,
    emission_plan,
    operator_prefix,
    kind,
    reference_data_plan,
    mesh_signature,
    local_signatures,
):
    kind = str(kind)
    operator_prefix = str(operator_prefix)
    local_by_order = {signature.form_order: signature for signature in local_signatures}
    entries = []
    if kind == "energy_soa":
        entries.extend(
            _energy_diagnostics_entries(
                unit,
                emission_plan,
                operator_prefix,
                reference_data_plan,
                mesh_signature,
                local_by_order,
            )
        )
    elif kind == "residual_soa":
        entries.extend(
            _residual_diagnostics_entries(
                unit,
                operator_prefix,
                reference_data_plan,
                mesh_signature,
                local_by_order,
                include_block_entries=True,
            )
        )
    elif kind == "mixed_residual_soa":
        entries.extend(
            _residual_diagnostics_entries(
                unit,
                operator_prefix,
                reference_data_plan,
                mesh_signature,
                local_by_order,
                include_block_entries=False,
            )
        )
    elif kind == "boundary_residual_soa":
        entries.extend(
            _boundary_diagnostics_entries(
                unit,
                operator_prefix,
                mesh_signature,
                local_by_order,
            )
        )
    return KernelDiagnosticsPlan(operator_prefix, kind, tuple(entries))


def validate_diagnostics_plan_names(plan, expected_names):
    if plan is None:
        return None
    if not isinstance(plan, KernelDiagnosticsPlan):
        raise TypeError("diagnostics_plan must be a KernelDiagnosticsPlan")
    expected_names = tuple(str(name) for name in expected_names)
    missing = tuple(name for name in expected_names if name not in plan.public_names)
    if missing:
        raise ValueError(
            "diagnostics plan is missing entries: %s" % ", ".join(missing)
        )
    return plan


def _energy_diagnostics_entries(
    unit,
    emission_plan,
    operator_prefix,
    reference_data_plan,
    mesh_signature,
    local_by_order,
):
    rule = emission_plan.isoparametric_specialization.quadrature_rule
    entries = []
    for expression_plan in unit.expression_plans:
        public_name = "%s_%s_%s_soa" % (
            operator_prefix,
            rule.element_type.lower(),
            expression_plan.name,
        )
        entries.append(
            _entry_from_expression_plan(
                public_name,
                expression_plan,
                mesh_signature,
                local_by_order,
                reference_data_plan.isoparametric,
            )
        )
    return tuple(entries)


def _residual_diagnostics_entries(
    unit,
    operator_prefix,
    reference_data_plan,
    mesh_signature,
    local_by_order,
    include_block_entries,
):
    by_order = {plan.form_order: plan for plan in unit.expression_plans}
    entries = []
    residual_plan = by_order.get(FormOrder.ONE)
    if residual_plan is not None:
        entries.append(
            _entry_from_expression_plan(
                "%s_residual_element_soa" % operator_prefix,
                residual_plan,
                mesh_signature,
                local_by_order,
                reference_data_plan.isoparametric,
            )
        )
    action_plan = by_order.get(FormOrder.TWO)
    if action_plan is not None and include_block_entries:
        for block_name in _diagnostic_block_names(unit, action_plan):
            diagnostic_block_name = _diagnostic_block_name(block_name)
            entries.append(
                _entry_from_expression_plan(
                    "%s_%s" % (operator_prefix, diagnostic_block_name),
                    action_plan,
                    mesh_signature,
                    local_by_order,
                    reference_data_plan.isoparametric,
                    block_name=diagnostic_block_name,
                )
            )
    if action_plan is not None:
        entries.append(
            _entry_from_expression_plan(
                "%s_jacobian_action_element_soa" % operator_prefix,
                action_plan,
                mesh_signature,
                local_by_order,
                reference_data_plan.isoparametric,
            )
        )
    return tuple(entries)


def _boundary_diagnostics_entries(unit, operator_prefix, mesh_signature, local_by_order):
    entries = []
    for expression_plan in unit.expression_plans:
        if expression_plan.form_order is FormOrder.ONE:
            entries.append(
                _entry_from_expression_plan(
                    "%s_boundary_residual_soa" % operator_prefix,
                    expression_plan,
                    mesh_signature,
                    local_by_order,
                    None,
                )
            )
    return tuple(entries)


def _diagnostic_block_names(unit, action_plan):
    system = getattr(unit.form_collection, "source", None)
    if getattr(unit, "is_block", False) and system is not None and hasattr(system, "jacobian_blocks"):
        return tuple(block.name for block in system.jacobian_blocks())
    names = tuple(getattr(block, "name", str(block)) for block in action_plan.blocks)
    if names:
        return names
    if system is not None and hasattr(system, "jacobian_blocks"):
        return tuple(block.name for block in system.jacobian_blocks())
    return ()


def _diagnostic_block_name(block_name):
    block_name = str(block_name)
    if block_name.startswith("form_2_"):
        return "jacobian_%s" % block_name[len("form_2_"):]
    return block_name


def _entry_from_expression_plan(
    public_name,
    expression_plan,
    mesh_signature,
    local_by_order,
    reference_dataset,
    block_name="",
):
    diagnostic_graph = expression_plan.diagnostics
    cost = diagnostic_graph.cost if diagnostic_graph is not None else None
    if cost is None and hasattr(expression_plan.expression_graph, "cost"):
        cost = expression_plan.expression_graph.cost
    return KernelDiagnosticsEntryPlan(
        public_name,
        expression_plan.name,
        expression_plan.form_order,
        expression_plan.dependencies,
        cost=cost,
        mesh_signature=mesh_signature,
        local_signature=local_by_order.get(expression_plan.form_order),
        reference_dataset=reference_dataset,
        block_name=block_name,
    )
