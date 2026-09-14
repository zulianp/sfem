from dataclasses import dataclass

from codegen.framework.symbolic.forms import FormOrder


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


def reference_data_traffic(entries):
    """Split a kernel's reference arrays into basis scalars and weights.

    Two of the numbers every `KernelDiagnostics` record carries, and the split
    between them is on the `q_weight` name prefix -- which is ABI, set by
    `fem.reference.sfem_reference_data`, and so belongs beside the plan that
    reads it rather than being re-spelled wherever a record is built.
    """
    return (
        sum(len(entry.values) for entry in entries
            if not entry.name.startswith("q_weight")),
        sum(len(entry.values) for entry in entries
            if entry.name.startswith("q_weight")),
    )


def energy_reference_data_traffic(rule, reference_inputs, n_qp):
    """The same two numbers for an energy kernel, which counts differently.

    An energy kernel reads only the reference arrays its own signature takes,
    so the non-tensor-product count is over `reference_inputs` -- a kernel that
    never reads `shape` is not charged for it -- while a tensor-product one
    reads the 1D tables whole and counts them from the rule.

    Those are two different measures and both are wanted; what was wrong is
    that `emitters/energy_codegen.py` chose between them by asking the *basis
    family*, which is a question the rule answers about itself.
    `sfem_reference_data` branches on `rule.is_tensor_product` internally and
    returns the 1D tables for exactly these elements.

    That the two agree where they overlap was measured, not assumed: for HEX8,
    QUAD4, PROTEUS_HEX8 and HEX27 the rule's own split gives the same pair this
    branch computes -- (8, 2) for the first three and (18, 3) for HEX27.  The
    non-tensor-product branch is deliberately left counting the kernel's inputs,
    because it is the more accurate of the two and nothing here should quietly
    start charging a kernel for a table it does not read.

    `reproducibility --all` would not catch a mistake here -- it compares kernel
    *answers*, and a diagnostics number is not one -- but byte-identity of the
    regenerated tree does, because the record is emitted as literal integers
    into the operator source.  That is a narrower gate than it sounds: it holds
    a number steady, it does not say the number is right.
    """
    if rule.is_tensor_product:
        return (
            len(rule.tensor_product_shape_values_1d)
            + len(rule.tensor_product_shape_gradients_1d),
            len(rule.tensor_product_weights_1d),
        )
    return (
        sum(array_input.size for array_input in reference_inputs),
        n_qp,
    )


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
        public_name = _energy_public_name(
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


def _energy_public_name(operator_prefix, element, form_name):
    operator_prefix = str(operator_prefix)
    element = str(element).lower()
    if operator_prefix.lower().endswith("_%s" % element):
        return "%s_%s_soa" % (operator_prefix, form_name)
    return "%s_%s_%s_soa" % (operator_prefix, element, form_name)


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
                "%s_residual_esoa" % operator_prefix,
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
                "%s_jacobian_action_esoa" % operator_prefix,
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
    """Names of the Jacobian-action blocks, in field order.

    Read from the form collection's own 2-form block metadata.  This used to
    call `jacobian_blocks()` on the pre-lowering system through
    `FormCollection.source`; the blocks carry the same names and are already
    part of the lowered collection.
    """
    collection = unit.form_collection
    try:
        blocks = collection.blocks_for(FormOrder.TWO)
    except (AttributeError, ValueError):
        blocks = ()
    names = tuple(block.name for block in blocks if getattr(block, "name", ""))
    if names:
        return names
    names = tuple(getattr(block, "name", str(block)) for block in action_plan.blocks)
    if names:
        return names
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


# ---------------------------------------------------------------------------
# Cost of a residual kernel's diagnostic entries.
#
# Every generated kernel carries FLOP and arithmetic-intensity reporting, which
# the PRD asks for so performance analyses can be produced automatically.  The
# numbers come from the scheduling layer's cost model, and choosing which
# kernels are measured and under which temporary naming is a planning decision.
#
# Both residual emitters computed these inline, with the same three call shapes
# written out twice -- once in the coupled path and once in the mixed one.  The
# temporary prefixes matter: they name the intermediates in the scheduled graph
# and so feed into the cost, which is why they are fixed here rather than left
# to each caller.
# ---------------------------------------------------------------------------


def residual_diagnostic_cost(system):
    """Cost of evaluating the residual for one element."""
    from codegen.framework.plans.scheduling import build_residual_graph

    return build_residual_graph(system, "residual_diagnostics_tmp").cost


def jacobian_action_diagnostic_cost(system):
    """Cost of applying the Jacobian to one element's worth of data.

    This is the matrix-free apply, the kernel the framework exists to make
    fast, so its cost is the number most worth reporting accurately.
    """
    from codegen.framework.plans.scheduling import build_jacobian_action_graph

    return build_jacobian_action_graph(
        system, temporary_prefix="jacobian_action_diagnostics_tmp"
    ).cost


def jacobian_block_diagnostic_cost(system, block):
    """Cost of one row/column block of the Jacobian action."""
    from codegen.framework.symbolic.core import KernelExpressions
    from codegen.framework.plans.scheduling import build_expression_graph

    return build_expression_graph(
        KernelExpressions().jacobian_action(block.expression, block.name),
        data_symbols=system.jacobian_action_data_symbols(),
        temporary_prefix="%s_diagnostics_tmp" % block.name,
    ).cost
