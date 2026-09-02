"""Kernel planning: expression graph, evaluation schedule, and cost model.

This is the layer that decides *how* a kernel computes, as opposed to *what* it
computes (the symbolic layer above) or *how the result is spelled* (the emission
layer below).  It owns common-subexpression elimination, placement of each
statement into an execution scope, liveness and register-pressure estimates, and
the arithmetic-intensity numbers the diagnostics report.

It came out of ``symbolic/core.py``, where it had accumulated under a name that
suggested it was part of the mathematical specification.  It is not: nothing here
manipulates a form, and everything here is a scheduling decision.

The entry point is ``build_expression_graph``.  It accepts either a
``KernelExpressions`` builder or a plain iterable of ``KernelExpression``.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Optional, Tuple, Union

import networkx as nx
import sympy as sp
import sympy.codegen.ast as ast

from codegen.framework.symbolic.core import (
    SympyExpr,
    DimensionSpecialization,
    ExecutionScope,
    ExpressionPattern,
    ExpressionRole,
    KernelExpression,
    KernelTemplateParameter,
    PatternKind,
    ScopeKind,
    SymbolicObject,
    _dimension_specialization,
    _lhs,
    _normalize_kernel_expression,
    _reattach_lhs,
    _rhs,
    _template_parameters,
)


@dataclass(frozen=True)
class ExpressionCost:
    adds: int = 0
    muls: int = 0
    divs: int = 0
    sqrts: int = 0
    pows: int = 0
    exps: int = 0
    logs: int = 0
    trigs: int = 0
    loads: int = 0
    stores: int = 0
    temporaries: int = 0
    estimated_registers: int = 0

    @property
    def flops(self):
        return (
            self.adds
            + self.muls
            + 8 * self.divs
            + 12 * self.sqrts
            + self.pows
            + 20 * self.exps
            + 20 * self.logs
            + 24 * self.trigs
        )


@dataclass(frozen=True)
class EvaluationStatement:
    target: object
    expression: sp.Expr
    kind: str
    dependencies: Tuple[sp.Symbol, ...]
    cost: ExpressionCost
    role: Optional[ExpressionRole] = None
    output_index: Optional[int] = None
    augmented: bool = False
    scopes: Tuple[ScopeKind, ...] = ()
    hoist_scope: ScopeKind = ScopeKind.MESH


@dataclass(frozen=True)
class LivenessState:
    statement_index: int
    target: object
    live_temporaries_after: Tuple[sp.Symbol, ...]
    register_pressure: int


@dataclass(frozen=True)
class EvaluationMetrics:
    total_flops: int
    total_loads: int
    total_stores: int
    peak_registers: int
    peak_live_temporaries: int
    liveness: Tuple[LivenessState, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    statements: Tuple[EvaluationStatement, ...]
    intermediates: Tuple[EvaluationStatement, ...]
    outputs: Tuple[EvaluationStatement, ...]
    metrics: EvaluationMetrics

    @property
    def temporary_symbols(self):
        return tuple(statement.target for statement in self.intermediates)


@dataclass(frozen=True)
class ExpressionGraph:
    graph: nx.DiGraph
    outputs: Tuple[KernelExpression, ...]
    intermediates: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    reduced_outputs: Tuple[SympyExpr, ...]
    patterns: Tuple[ExpressionPattern, ...]
    evaluation_plan: EvaluationPlan
    cost: ExpressionCost
    scopes: Tuple[ExecutionScope, ...] = ()
    template_parameters: Tuple[KernelTemplateParameter, ...] = ()
    specialization: Optional[DimensionSpecialization] = None

    def topological_nodes(self):
        return tuple(nx.topological_sort(self.graph))

    def patterns_by_kind(self, kind):
        kind = PatternKind(kind)
        return tuple(pattern for pattern in self.patterns if pattern.kind == kind)

    def scope_symbols(self, kind):
        kind = ScopeKind(kind)
        symbols = []
        for scope in self.scopes:
            if scope.kind == kind:
                symbols.extend(scope.symbols)
        return tuple(symbols)


def build_expression_graph(
    expressions: Iterable[KernelExpression],
    *,
    data_symbols: Optional[Iterable[sp.Symbol]] = None,
    loop_symbols: Optional[Mapping[str, Iterable[sp.Symbol]]] = None,
    scopes: Optional[Iterable[ExecutionScope]] = None,
    symbolic_objects: Optional[Iterable[SymbolicObject]] = None,
    template_parameters: Optional[Iterable[KernelTemplateParameter]] = None,
    specialization: Optional[DimensionSpecialization] = None,
    temporary_prefix="t",
    temporary_symbols=None,
    optimizations="basic",
):
    expressions = getattr(expressions, "expressions", expressions)
    outputs = tuple(_normalize_kernel_expression(expr) for expr in expressions)
    reduced_inputs = [_rhs(expr.expression) for expr in outputs]
    cse_symbols = (
        temporary_symbols
        if temporary_symbols is not None
        else sp.numbered_symbols(temporary_prefix)
    )
    intermediates, reduced_rhs = sp.cse(
        reduced_inputs,
        symbols=cse_symbols,
        optimizations=optimizations,
    )
    reduced_outputs = _reattach_lhs(outputs, reduced_rhs)
    intermediates = _prune_dead_cse_intermediates(intermediates, reduced_outputs)

    graph = nx.DiGraph()
    data_symbol_set = set(data_symbols or ())
    symbolic_objects = tuple(symbolic_objects or ())
    kernel_template_parameters = _template_parameters(
        symbolic_objects,
        template_parameters,
    )
    graph.graph["template_parameters"] = kernel_template_parameters
    kernel_specialization = _dimension_specialization(
        symbolic_objects,
        specialization,
    )
    graph.graph["specialization"] = kernel_specialization
    layout_symbol_map = _layout_symbol_map(symbolic_objects)
    execution_scopes = _normalize_scopes(loop_symbols, scopes)
    scope_symbol_map = _scope_symbol_map(execution_scopes)

    for scope in execution_scopes:
        for symbol in scope.symbols:
            _add_node(
                graph,
                symbol,
                "loop_index",
                scope=scope.kind.value,
                scope_name=scope.name,
                scope_kind=scope.kind,
            )

    for symbol in data_symbol_set:
        _add_node(graph, symbol, "data")
        _annotate_data_layout(graph, symbol, layout_symbol_map)

    for var, expr in intermediates:
        _add_expression_node(
            graph,
            var,
            expr,
            data_symbol_set,
            scope_symbol_map,
            layout_symbol_map,
            "intermediate",
        )
        graph.nodes[var]["scopes"] = _expression_scopes(expr, scope_symbol_map)

    output_nodes = []
    for idx, (kernel_expr, reduced_expr) in enumerate(zip(outputs, reduced_outputs)):
        output_node = _output_node_name(kernel_expr, idx)
        output_nodes.append(output_node)
        graph.add_node(
            output_node,
            kind="output",
            role=kernel_expr.role.value,
            name=kernel_expr.name,
            expression=reduced_expr,
            scopes=_expression_scopes(_rhs(reduced_expr), scope_symbol_map),
        )
        for dep in _dependencies(_rhs(reduced_expr)):
            _ensure_dependency_node(graph, dep, data_symbol_set, scope_symbol_map, layout_symbol_map)
            graph.add_edge(dep, output_node)

    patterns = _detect_patterns(
        graph,
        intermediates,
        reduced_outputs,
        output_nodes,
        symbolic_objects,
    )
    evaluation_plan = _build_evaluation_plan(
        intermediates,
        reduced_outputs,
        output_nodes,
        outputs,
        data_symbol_set,
        scope_symbol_map,
    )
    _annotate_graph_scope_placements(graph, evaluation_plan, output_nodes)
    cost = _expression_cost(
        intermediates,
        reduced_outputs,
        data_symbol_set,
        evaluation_plan.metrics.peak_registers,
    )
    return ExpressionGraph(
        graph,
        outputs,
        tuple(intermediates),
        reduced_outputs,
        patterns,
        evaluation_plan,
        cost,
        execution_scopes,
        kernel_template_parameters,
        kernel_specialization,
    )


def _build_evaluation_plan(
    intermediates,
    reduced_outputs,
    output_nodes,
    kernel_outputs,
    data_symbols,
    scope_symbol_map,
):
    intermediate_statements = []
    output_statements = []

    for target, expr in intermediates:
        intermediate_statements.append(
            EvaluationStatement(
                target=target,
                expression=expr,
                kind="intermediate",
                dependencies=_dependencies(expr),
                cost=_statement_cost(expr, data_symbols, stores=1),
                scopes=_expression_scopes(expr, scope_symbol_map),
            )
        )

    for idx, (output, output_node, kernel_output) in enumerate(
        zip(reduced_outputs, output_nodes, kernel_outputs)
    ):
        expr = _rhs(output)
        lhs = _lhs(output)
        output_statements.append(
            EvaluationStatement(
                target=lhs if lhs is not None else "output:%d" % idx,
                expression=expr,
                kind="output",
                dependencies=_dependencies(expr),
                cost=_statement_cost(expr, data_symbols, stores=1),
                role=kernel_output.role,
                output_index=idx,
                augmented=isinstance(output, ast.AddAugmentedAssignment),
                scopes=_expression_scopes(expr, scope_symbol_map),
            )
        )

    statements = _resolve_statement_scopes(tuple(intermediate_statements + output_statements))
    intermediate_count = len(intermediate_statements)
    metrics = _evaluation_metrics(
        statements,
        tuple(stmt.target for stmt in statements[:intermediate_count]),
    )
    return EvaluationPlan(
        statements,
        tuple(statements[:intermediate_count]),
        tuple(statements[intermediate_count:]),
        metrics,
    )


def _annotate_graph_scope_placements(graph, evaluation_plan, output_nodes):
    for statement in evaluation_plan.intermediates:
        if statement.target in graph:
            graph.nodes[statement.target]["scopes"] = statement.scopes
            graph.nodes[statement.target]["hoist_scope"] = statement.hoist_scope

    for statement, output_node in zip(evaluation_plan.outputs, output_nodes):
        if output_node in graph:
            graph.nodes[output_node]["scopes"] = statement.scopes
            graph.nodes[output_node]["hoist_scope"] = statement.hoist_scope


def _resolve_statement_scopes(statements):
    scope_by_target = {}
    resolved = []

    for statement in statements:
        scopes = set(statement.scopes)
        for dependency in statement.dependencies:
            scopes.update(scope_by_target.get(dependency, ()))

        ordered_scopes = _ordered_scopes(scopes)
        resolved_statement = replace(
            statement,
            scopes=ordered_scopes,
            hoist_scope=_hoist_scope(ordered_scopes),
        )
        resolved.append(resolved_statement)
        scope_by_target[statement.target] = ordered_scopes

    return tuple(resolved)


def _evaluation_metrics(statements, temporary_symbols):
    temporary_symbol_set = set(temporary_symbols)
    last_use = {}

    for idx, statement in enumerate(statements):
        for dependency in statement.dependencies:
            last_use[dependency] = idx

    produced_temporaries = set()
    liveness = []
    peak_registers = 0
    peak_live_temporaries = 0
    total_flops = 0
    total_loads = 0
    total_stores = 0

    for idx, statement in enumerate(statements):
        dependencies = set(statement.dependencies)
        live_temporaries_before = {
            symbol
            for symbol in produced_temporaries
            if last_use.get(symbol, -1) >= idx
        }
        live_during = dependencies | live_temporaries_before

        if statement.target in temporary_symbol_set and last_use.get(statement.target, -1) > idx:
            live_during.add(statement.target)

        register_pressure = len(live_during)
        if register_pressure > peak_registers:
            peak_registers = register_pressure

        if statement.target in temporary_symbol_set:
            produced_temporaries.add(statement.target)

        live_temporaries_after = tuple(
            sorted(
                (
                    symbol
                    for symbol in produced_temporaries
                    if last_use.get(symbol, -1) > idx
                ),
                key=str,
            )
        )
        if len(live_temporaries_after) > peak_live_temporaries:
            peak_live_temporaries = len(live_temporaries_after)

        liveness.append(
            LivenessState(
                idx,
                statement.target,
                live_temporaries_after,
                register_pressure,
            )
        )

        total_flops += statement.cost.flops
        total_loads += statement.cost.loads
        total_stores += statement.cost.stores

    return EvaluationMetrics(
        total_flops,
        total_loads,
        total_stores,
        peak_registers,
        peak_live_temporaries,
        tuple(liveness),
    )


def _detect_patterns(graph, intermediates, reduced_outputs, output_nodes, symbolic_objects):
    patterns = []
    symbolic_objects = tuple(symbolic_objects)

    for symbol, expr in intermediates:
        _append_pattern(
            graph,
            patterns,
            ExpressionPattern(
                PatternKind.REPEATED_SUBEXPRESSION,
                symbol,
                expr,
                _dependencies(expr),
                "sympy_cse",
            ),
        )
        _extend_patterns(graph, patterns, _match_objects(symbol, expr, symbolic_objects))

    for output_node, output in zip(output_nodes, reduced_outputs):
        expr = _rhs(output)
        _extend_patterns(graph, patterns, _match_objects(output_node, expr, symbolic_objects))

    return tuple(patterns)


def _extend_patterns(graph, patterns, new_patterns):
    for pattern in new_patterns:
        _append_pattern(graph, patterns, pattern)


def _append_pattern(graph, patterns, pattern):
    patterns.append(pattern)
    if pattern.node in graph:
        node_attrs = graph.nodes[pattern.node]
        node_patterns = list(node_attrs.get("patterns", ()))
        node_patterns.append(pattern)
        node_attrs["patterns"] = tuple(node_patterns)


def _match_objects(node, expression, symbolic_objects):
    matches = []

    for symbolic_object in symbolic_objects:
        matched_symbols, matched_expressions = symbolic_object.match(expression)
        if matched_symbols or matched_expressions:
            matches.append(
                ExpressionPattern(
                    symbolic_object.kind,
                    node,
                    expression,
                    matched_symbols,
                    symbolic_object.name,
                    matched_expressions,
                    symbolic_object,
                )
            )

    return matches


def _expression_cost(intermediates, reduced_outputs, data_symbols, estimated_registers):
    adds = muls = divs = sqrts = pows = exps = logs = trigs = stores = 0
    loaded = set()

    for _, expr in intermediates:
        a, m, d, s, p, exp_count, log_count, trig_count = _op_counts(expr)
        adds += a
        muls += m
        divs += d
        sqrts += s
        pows += p
        exps += exp_count
        logs += log_count
        trigs += trig_count
        loaded.update(expr.free_symbols)
        stores += 1

    for output in reduced_outputs:
        expr = _rhs(output)
        a, m, d, s, p, exp_count, log_count, trig_count = _op_counts(expr)
        adds += a
        muls += m
        divs += d
        sqrts += s
        pows += p
        exps += exp_count
        logs += log_count
        trigs += trig_count
        loaded.update(expr.free_symbols)
        stores += 1

    loads = len(loaded.intersection(data_symbols)) if data_symbols else len(loaded)
    temporaries = len(intermediates)
    return ExpressionCost(
        adds=adds,
        muls=muls,
        divs=divs,
        sqrts=sqrts,
        pows=pows,
        exps=exps,
        logs=logs,
        trigs=trigs,
        loads=loads,
        stores=stores,
        temporaries=temporaries,
        estimated_registers=estimated_registers,
    )


def _statement_cost(expression, data_symbols, stores):
    adds, muls, divs, sqrts, pows, exps, logs, trigs = _op_counts(expression)
    loaded = expression.free_symbols
    loads = len(loaded.intersection(data_symbols)) if data_symbols else len(loaded)
    return ExpressionCost(
        adds=adds,
        muls=muls,
        divs=divs,
        sqrts=sqrts,
        pows=pows,
        exps=exps,
        logs=logs,
        trigs=trigs,
        loads=loads,
        stores=stores,
        temporaries=0,
        estimated_registers=loads,
    )


def _op_counts(expression):
    adds = muls = divs = sqrts = pows = exps = logs = trigs = 0
    trig_functions = {
        sp.sin,
        sp.cos,
        sp.tan,
        sp.asin,
        sp.acos,
        sp.atan,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        sp.asinh,
        sp.acosh,
        sp.atanh,
    }

    for node in sp.preorder_traversal(expression):
        if isinstance(node, sp.Add):
            adds += max(0, len(node.args) - 1)
        elif isinstance(node, sp.Mul):
            muls += max(0, len(node.args) - 1)
        elif isinstance(node, sp.Pow):
            if node.exp == -1:
                divs += 1
            elif isinstance(node.exp, sp.Number) and float(node.exp) == 0.5:
                sqrts += 1
            else:
                pows += 1
        elif getattr(node, "is_Function", False):
            if node.func == sp.log:
                logs += 1
            elif node.func == sp.exp:
                exps += 1
            elif node.func in trig_functions:
                trigs += 1
            elif node.func == sp.sqrt:
                sqrts += 1

    return adds, muls, divs, sqrts, pows, exps, logs, trigs


def _prune_dead_cse_intermediates(intermediates, outputs):
    required_symbols = set()
    for output in outputs:
        required_symbols.update(_rhs(output).free_symbols)

    retained = []
    for symbol, expression in reversed(tuple(intermediates)):
        if symbol not in required_symbols:
            continue
        required_symbols.remove(symbol)
        required_symbols.update(expression.free_symbols)
        retained.append((symbol, expression))

    return tuple(reversed(retained))


def _add_expression_node(
    graph,
    symbol,
    expression,
    data_symbols,
    scope_symbol_map,
    layout_symbol_map,
    kind,
):
    graph.add_node(symbol, kind=kind, expression=expression)
    for dep in _dependencies(expression):
        _ensure_dependency_node(graph, dep, data_symbols, scope_symbol_map, layout_symbol_map)
        graph.add_edge(dep, symbol)


def _add_node(graph, symbol, kind, **attrs):
    if symbol not in graph:
        graph.add_node(symbol, kind=kind, **attrs)
    else:
        graph.nodes[symbol].update(attrs)
        graph.nodes[symbol]["kind"] = kind


def _ensure_dependency_node(graph, symbol, data_symbols, scope_symbol_map, layout_symbol_map):
    if symbol in graph:
        _annotate_data_layout(graph, symbol, layout_symbol_map)
        return

    if symbol in scope_symbol_map:
        scope = scope_symbol_map[symbol]
        graph.add_node(
            symbol,
            kind="loop_index",
            scope=scope.kind.value,
            scope_name=scope.name,
            scope_kind=scope.kind,
        )
        return

    if symbol in data_symbols:
        graph.add_node(symbol, kind="data")
        _annotate_data_layout(graph, symbol, layout_symbol_map)
    else:
        graph.add_node(symbol, kind="symbol")
        _annotate_data_layout(graph, symbol, layout_symbol_map)


def _dependencies(expression):
    return tuple(sorted(expression.free_symbols, key=str))


def _annotate_data_layout(graph, symbol, layout_symbol_map):
    if symbol not in graph or symbol not in layout_symbol_map:
        return

    symbolic_object = layout_symbol_map[symbol]
    component = symbolic_object.component_index(symbol)
    item_index = sp.symbols("%s_idx" % symbolic_object.name, integer=True)
    graph.nodes[symbol]["layout"] = symbolic_object.layout
    graph.nodes[symbol]["layout_kind"] = symbolic_object.layout.kind
    graph.nodes[symbol]["symbolic_object"] = symbolic_object.name
    graph.nodes[symbol]["component"] = component
    graph.nodes[symbol]["layout_index"] = item_index
    graph.nodes[symbol]["layout_offset"] = symbolic_object.layout_offset(symbol, item_index)
    graph.nodes[symbol]["object_metadata"] = symbolic_object.metadata
    if "n_nodes" in symbolic_object.metadata and "dim" in symbolic_object.metadata:
        dim = symbolic_object.metadata["dim"]
        graph.nodes[symbol]["node"] = component // dim
        graph.nodes[symbol]["dim_component"] = component % dim


def _expression_scopes(expression, scope_symbol_map):
    kinds = {
        scope_symbol_map[symbol].kind
        for symbol in expression.free_symbols
        if symbol in scope_symbol_map
    }
    return _ordered_scopes(kinds)


def _scope_sort_key(kind):
    return tuple(ScopeKind).index(kind)


def _ordered_scopes(scopes):
    return tuple(sorted(scopes, key=_scope_sort_key))


def _hoist_scope(scopes):
    if not scopes:
        return ScopeKind.MESH
    return scopes[-1]


def _scope_symbol_map(scopes):
    ret = {}
    for scope in scopes:
        for symbol in scope.symbols:
            ret[symbol] = scope
    return ret


def _layout_symbol_map(symbolic_objects):
    ret = {}
    for symbolic_object in symbolic_objects:
        for symbol in symbolic_object.direct_symbols:
            ret[symbol] = symbolic_object
    return ret


def _normalize_scopes(loop_symbols, scopes):
    normalized = []

    for raw_scope in scopes or ():
        if isinstance(raw_scope, ExecutionScope):
            normalized.append(raw_scope)
        else:
            raise TypeError("scopes must contain ExecutionScope instances")

    for raw_kind, symbols in (loop_symbols or {}).items():
        normalized.append(ExecutionScope(_scope_kind(raw_kind), symbols))

    return tuple(normalized)


def _scope_kind(value):
    if isinstance(value, ScopeKind):
        return value
    normalized = str(value).replace("-", "_")
    aliases = {
        "mesh_wide": ScopeKind.MESH,
        "meshwide": ScopeKind.MESH,
        "mesh": ScopeKind.MESH,
    }
    if normalized in aliases:
        return aliases[normalized]
    return ScopeKind(normalized)


def _output_node_name(kernel_expr, idx):
    if kernel_expr.name is not None:
        return "output:%s:%s" % (kernel_expr.role.value, kernel_expr.name)
    lhs = _lhs(kernel_expr.expression)
    if lhs is not None:
        return "output:%s:%s" % (kernel_expr.role.value, lhs)
    return "output:%s:%d" % (kernel_expr.role.value, idx)


# Convenience entry points that used to be methods on the symbolic systems.
# They are scheduling, not specification: each one takes a system, asks it for
# its expressions, and hands those to the scheduler.  Keeping them here is what
# lets the symbolic layer stop importing the planning layer.


def build_residual_graph(system, temporary_prefix="residual_tmp"):
    return build_expression_graph(
        system.residual_expressions(),
        data_symbols=system.residual_data_symbols(),
        temporary_prefix=temporary_prefix,
    )


def build_jacobian_action_graph(
    system,
    include_blocks=False,
    temporary_prefix="jacobian_action_tmp",
):
    return build_expression_graph(
        system.jacobian_action_expressions(include_blocks),
        data_symbols=system.jacobian_action_data_symbols(),
        temporary_prefix=temporary_prefix,
    )


def build_two_phase_flow_graph(
    model,
    water_pressure,
    co2_pressure,
    include_derivatives=False,
    temporary_prefix="two_phase_tmp",
):
    return build_expression_graph(
        model.kernel_expressions(
            water_pressure,
            co2_pressure,
            include_derivatives,
        ),
        data_symbols=(
            water_pressure,
            co2_pressure,
        )
        + model.parameters.as_tuple(),
        temporary_prefix=temporary_prefix,
    )
