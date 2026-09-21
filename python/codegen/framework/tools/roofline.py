"""Roofline model for the generated kernels, read out of the emitted source.

A throughput figure on its own says how fast a kernel ran; it does not say
whether that was fast.  The roofline answers the second question by placing the
kernel against two ceilings the machine imposes -- its peak arithmetic rate and
its peak memory bandwidth -- at the kernel's own arithmetic intensity.

Both coordinates come from the generated tree rather than from a hand-kept
table, which is the point of putting this here instead of in a spike:

* **FLOPs per element** are counted from the printed arithmetic, reusing
  ``flops_audit.count_flops`` so that a count here and a count there are weighed
  on one scale.  Unlike that audit, this does not need a straight-line body: the
  generated kernels put exactly one element's work in a loop over ``lane`` (a
  blocked kernel) or over ``element`` (a scalar one), so the per-element scope
  is identifiable and its body is straight-line inside.

* **Bytes per element** are counted from the memory references in those same
  scopes, with each array's element width taken from the parameter that declares
  it.  Two models are produced, because for a mesh kernel the truth depends on
  the connectivity and not on the kernel:

  ``streamed``
      every reference moves its own bytes.  No reuse at all -- the pessimistic
      bound, and the honest one for a kernel run on a mesh whose numbering
      defeats the cache.
  ``compulsory``
      a value reached *through the connectivity* is moved once per node rather
      than once per element, so its cost per element is scaled by
      ``nnodes / nelements``.  Everything reached by the element index is
      already once per element and is unchanged.  The optimistic bound.

  A kernel's real intensity lies between them, so it is drawn as a band rather
  than a point.

Usage::

    python -m codegen.framework.tools.roofline frontend/ops/generated --list
    python -m codegen.framework.tools.roofline frontend/ops/generated \\
        --kernel 'neohookean_ogden_hex8_inexact_apply' --machine grace
    python -m codegen.framework.tools.roofline frontend/ops/generated \\
        --measured spikes/inexact_apply_compare/measured.json --plot roofline.svg

The machine peaks are vendor figures, not measurements taken here; each carries
its provenance in ``Machine.source`` and every report prints it.  A measured
ceiling is better than a quoted one and can be substituted with
``--peak-gflops`` / ``--peak-bandwidth`` without touching this file.
"""

import argparse
import glob
import json
import os
import re
import sys

from codegen.framework.tools.flops_audit import count_flops, function_body


class Machine(object):
    """One machine's two ceilings, and where the numbers came from.

    ``flops_per_cycle_per_core`` is for 64-bit arithmetic.  Narrower scalars are
    modelled as doubling per halving of the width, which is what a fixed-width
    SIMD pipe gives; it is an upper bound for 16-bit, where the pipe may not
    issue FMAs at all, so a half-precision ceiling drawn from this is generous.
    """

    def __init__(self, name, label, cores, clock_hz, flops_per_cycle_per_core,
                 bandwidth_bytes_per_s, source):
        self.name = name
        self.label = label
        self.cores = cores
        self.clock_hz = clock_hz
        self.flops_per_cycle_per_core = flops_per_cycle_per_core
        self.bandwidth_bytes_per_s = bandwidth_bytes_per_s
        self.source = source

    def peak_flops(self, scalar_bytes=8, cores=None):
        cores = self.cores if cores is None else cores
        width = max(1, 8 // max(1, scalar_bytes))
        return cores * self.clock_hz * self.flops_per_cycle_per_core * width

    def scalar_flops(self, scalar_bytes=8, cores=None):
        """The same issue rate with one lane per operation.

        A kernel the compiler declined to vectorise cannot pass this, and the
        distance to it says whether the arithmetic is the problem or the
        vectorisation is.  The pipes are 128-bit, so this is the vector peak
        divided by the lane count at this width.
        """
        lanes = max(1, 16 // max(1, scalar_bytes))
        return self.peak_flops(scalar_bytes, cores) / lanes

    def ridge_intensity(self, scalar_bytes=8, cores=None):
        """FLOP per byte at which the two ceilings cross."""
        return self.peak_flops(scalar_bytes, cores) / self.bandwidth_bytes_per_s


#: The two machines this repository measures on.  Both are 128-bit SIMD cores
#: with four floating-point pipes, so one core retires four 2-wide FMAs per
#: cycle: 4 * 2 * 2 = 16 double-precision FLOPs per cycle.
MACHINES = {
    "m1max": Machine(
        name="m1max",
        label="Apple M1 Max (8 performance cores)",
        cores=8,
        clock_hz=3.228e9,
        flops_per_cycle_per_core=16,
        bandwidth_bytes_per_s=400e9,
        source=(
            "Apple specification: 8 performance cores at 3.228 GHz, four 128-bit "
            "NEON pipes each; 400 GB/s unified memory. The two efficiency cores "
            "are excluded -- they do not run these benchmarks' hot threads and "
            "counting them would raise a ceiling nothing reaches."
        ),
    ),
    "grace": Machine(
        name="grace",
        label="NVIDIA Grace (GH200, one socket, 72 Neoverse V2 cores)",
        cores=72,
        clock_hz=3.1e9,
        flops_per_cycle_per_core=16,
        bandwidth_bytes_per_s=500e9,
        source=(
            "NVIDIA specification: 72 Neoverse V2 cores at 3.1 GHz all-core, four "
            "128-bit SVE2 pipes each; up to 500 GB/s LPDDR5X on the Grace side. "
            "Quoted peaks, not measured here."
        ),
    ),
}


#: The scalar typedefs the generated kernels are instantiated with, by the name
#: that appears in a kernel's parameter list.  Resolved from the build's config
#: headers where they are available, so this table is only the fallback.
DEFAULT_WIDTHS = {
    "double": 8,
    "float": 4,
    "half_t": 2,
    "real_t": 8,
    "geom_t": 4,
    "idx_t": 4,
    "element_idx_t": 4,
    "metric_tensor_t": 4,
    "compressed_t": 2,
    "scaling_t": 4,
    "ptrdiff_t": 8,
    # The template parameters, as the published entry points instantiate them.
    "s_t": 8,
    "g_t": 4,
    "tangent_t": 4,
    "scale_t": 4,
}

_TYPEDEF = re.compile(
    r"typedef\s+(?:unsigned\s+|signed\s+)?([A-Za-z_][A-Za-z_0-9]*)\s+"
    r"([A-Za-z_][A-Za-z_0-9]*)\s*;"
)
_STDINT = {
    "int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2,
    "int32_t": 4, "uint32_t": 4, "int64_t": 8, "uint64_t": 8,
    "float": 4, "double": 8, "__fp16": 2, "_Float16": 2, "int": 4, "long": 8,
}


def resolve_widths(config_paths=(), overrides=None):
    """Scalar widths, with the build's own typedefs preferred over the table.

    A width that has drifted -- ``geom_t`` moved to double, say -- would rescale
    every intensity this tool reports, so it is read from the headers the kernels
    are compiled against rather than assumed.
    """
    widths = dict(DEFAULT_WIDTHS)
    aliases = {}
    for path in config_paths:
        try:
            text = open(path).read()
        except (IOError, OSError):
            continue
        for base, name in _TYPEDEF.findall(text):
            aliases[name] = base
    for _pass in range(4):
        for name, base in aliases.items():
            if base in _STDINT:
                widths[name] = _STDINT[base]
            elif base in widths:
                widths[name] = widths[base]
    widths.update(overrides or {})
    return widths


# --------------------------------------------------------------------------
# Reading one kernel out of the emitted source
# --------------------------------------------------------------------------

#: A loop whose body is exactly one element's work.  The generated kernels have
#: one of these two shapes and no other: a blocked kernel puts the element in
#: the lane, a scalar one puts it in the element index itself.
_PER_ELEMENT_LOOP = re.compile(
    r"for\s*\(\s*(?:const\s+)?\w[\w\s]*\b(lane|element)\s*=\s*0\s*;"
)

_PARAMETER = re.compile(
    r"(?:const\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*\**\s*(?:const\s*)?"
    r"(?:RSTR|__restrict__|__restrict)?\s*\**\s*([A-Za-z_][A-Za-z_0-9]*)\s*"
    r"(\[\s*\d*\s*\])?\s*$"
)


def _brace_body(text, open_index):
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index], index
    return None, len(text)


def per_element_scopes(body):
    """Every loop body that runs once per element, outermost-first.

    A blocked kernel has several -- the gather pass, the arithmetic, the scatter
    -- and their union is the element's work.  Nested lane loops do not occur;
    if one ever does, it is counted once by its outer scope and skipped here, so
    the count does not double.
    """
    scopes = []
    index = 0
    while True:
        match = _PER_ELEMENT_LOOP.search(body, index)
        if match is None:
            break
        open_brace = body.find("{", match.end())
        if open_brace < 0:
            break
        scope, close = _brace_body(body, open_brace)
        if scope is None:
            break
        scopes.append((match.group(1), scope))
        index = close + 1
    lane = [scope for kind, scope in scopes if kind == "lane"]
    return lane if lane else [scope for _kind, scope in scopes]


def signature_parameters(source, signature_prefix):
    """``{name: type}`` for the pointer and scalar parameters of one function."""
    start = source.find(signature_prefix)
    if start < 0:
        return {}
    open_paren = source.find("(", start)
    close_paren = source.find(")", open_paren)
    if open_paren < 0 or close_paren < 0:
        return {}
    parameters = {}
    for text in source[open_paren + 1:close_paren].split(","):
        match = _PARAMETER.match(text.strip())
        if match:
            parameters[match.group(2)] = match.group(1)
    return parameters


def derived_pointers(body, parameters):
    """Block-local base pointers, with the parameter's type they point into.

    `const g_t *const RSTR bg_adj0 = g_adj0 + evb;` is the geometry, reached
    through a hoisted offset; it moves the parameter's bytes and inherits its
    width.
    """
    derived = {}
    for name, source_name in re.findall(
        r"\*\s*(?:const\s+)?(?:RSTR\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*"
        r"([A-Za-z_][A-Za-z_0-9]*)\b",
        body,
    ):
        if source_name in parameters:
            derived[name] = parameters[source_name]
    return derived


def scratch_arrays(body):
    """Arrays the kernel declares for itself: `s_t bhx_0[VS];`, `idx_t bev0[VS];`.

    These are the staging the blocking exists to do.  They are stack objects a
    handful of vectors wide, live for one block, and never reach memory the
    roofline is drawn against -- counting them would charge the kernel twice for
    every value it gathers, once at the load and once at the reload.
    """
    return frozenset(
        name
        for _type, name in re.findall(
            r"\b([A-Za-z_][A-Za-z_0-9]*)\s+([A-Za-z_][A-Za-z_0-9]*)\s*\[\s*\w+\s*\]\s*;",
            body,
        )
    )


def _connectivity_names(body):
    """Identifiers holding a node index, so a subscript can be called indirect."""
    names = set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\[[^\]]*\]\s*=\s*elements\s*\[", body))
    names.update(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*=\s*elements\s*\[", body))
    names.add("elements")
    return names


_REFERENCE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\[")

#: The compound assignments a scatter uses.  `out[node] += ...` touches the
#: line twice -- it is an accumulation, not an overwrite -- and charging it once
#: would halve the cost of every scatter in the tree.
_UPDATE = re.compile(r"^(\+=|-=|\*=|/=)")


def _access_kind(scope, close_bracket):
    """`"read"`, `"write"` or `"update"` for the reference ending here."""
    rest = scope[close_bracket + 1:close_bracket + 40].lstrip()
    if _UPDATE.match(rest):
        return "update"
    if rest.startswith("=") and not rest.startswith("=="):
        return "write"
    return "read"


def _references(scope):
    """``(name, subscript, kind)`` for every array reference in a scope."""
    found = []
    for match in _REFERENCE.finditer(scope):
        open_bracket = match.end() - 1
        depth = 0
        close = open_bracket
        for index in range(open_bracket, len(scope)):
            if scope[index] == "[":
                depth += 1
            elif scope[index] == "]":
                depth -= 1
                if depth == 0:
                    close = index
                    break
        subscript = scope[open_bracket + 1:close]
        found.append((match.group(1), subscript, _access_kind(scope, close)))
    return found


_TEMPLATE_HEAD = re.compile(r"template\s*<([^>]*)>\s*$")


def template_parameters(source, signature_prefix):
    """The `typename` names a kernel's template declares, in order."""
    start = source.find(signature_prefix)
    if start < 0:
        return ()
    head = source[:start].rstrip()
    match = _TEMPLATE_HEAD.search(head)
    if not match:
        return ()
    names = []
    for part in match.group(1).split(","):
        part = part.strip()
        if part.startswith("typename") or part.startswith("class"):
            names.append(part.split()[-1])
        else:
            names.append(None)          # a non-type parameter, `int VS`
    return tuple(names)


def instantiation_bindings(kernel, template_names, instantiation_sources):
    """`{template name: concrete type}` from how the ABI instantiates the kernel.

    A kernel's `tangent_t` is `metric_tensor_t` in one entry point and
    `compressed_t` in another, and those differ by a factor of two in width --
    which is the whole subject of the store-precision work.  Reading the width
    off the template parameter's *name* would report the two kernels as moving
    the same bytes.  The `extern "C"` wrapper says which is which, so that is
    what is read.
    """
    pattern = re.compile(r"\b%s_impl\s*<([^>]*)>\s*\(" % re.escape(kernel))
    for source in instantiation_sources:
        match = pattern.search(source)
        if not match:
            continue
        arguments = [a.strip() for a in match.group(1).split(",")]
        return {
            name: argument
            for name, argument in zip(template_names, arguments)
            if name is not None
        }
    return {}


def analyse_kernel(source, signature_prefix, widths, nodes_per_element=1.0,
                   instantiation_sources=(), overrides=None):
    """FLOPs and both byte models for one element of one kernel."""
    body = function_body(source, signature_prefix)
    if body is None:
        return None
    kernel = signature_prefix.split("(")[0].strip().split()[-1][:-len("_impl")]
    bindings = instantiation_bindings(
        kernel, template_parameters(source, signature_prefix), instantiation_sources
    )
    # An explicit override outranks the instantiation: it is how a caller asks
    # about a configuration the published ABI does not offer.
    resolved = dict(widths)
    for name, concrete in bindings.items():
        if concrete in widths and name not in (overrides or {}):
            resolved[name] = widths[concrete]
    resolved.update(overrides or {})
    widths = resolved
    scopes = per_element_scopes(body)
    if not scopes:
        return None
    parameters = signature_parameters(source, signature_prefix)
    # What actually moves bytes: the kernel's own parameters, and the base
    # pointers hoisted out of them.  Everything else a subscript can name is
    # stack scratch.
    traffic_types = dict(parameters)
    traffic_types.update(derived_pointers(body, parameters))
    scratch = scratch_arrays(body)
    indirect = _connectivity_names(body)

    # Unweighted for the roofline's vertical axis: the ceiling it is drawn
    # against is an issue rate, so a divide is one operation there even though
    # `plans/scheduling.py` prices it at eight for cost modelling.  Both are
    # carried, because the weighted figure is the one the rest of the framework
    # quotes and the two must not be silently confused.
    counted = [count_flops(scope) for scope in scopes]
    flops = sum(
        entry["add"] + entry["mul"] + entry["div"] + entry["call"] for entry in counted
    )
    weighted = sum(entry["weighted"] for entry in counted)
    # An atomic read-modify-write is an add the body does not spell.
    atomics = sum(len(re.findall(r"#pragma\s+omp\s+atomic", scope)) for scope in scopes)
    flops += atomics
    weighted += atomics

    detail = {}
    for scope in scopes:
        for name, subscript, kind in _references(scope):
            if name in scratch or name not in traffic_types:
                continue
            width = widths.get(traffic_types[name])
            if width is None:
                continue
            through_connectivity = name == "elements" or any(
                re.search(r"\b%s\b" % re.escape(key), subscript) for key in indirect
            )
            entry = detail.setdefault(
                name,
                {"type": traffic_types[name], "width": width, "reads": 0,
                 "writes": 0, "updates": 0,
                 # The connectivity is itself an element-indexed array: it is
                 # reached through no one, so it is not on the shared side of
                 # this even though every subscript that mentions it is.
                 "indirect": through_connectivity and name != "elements"},
            )
            entry[kind + "s"] += 1

    streamed = 0.0
    compulsory = 0.0
    for entry in detail.values():
        # An accumulation reads the line it writes; a plain store does not.
        references = entry["reads"] + entry["writes"] + 2 * entry["updates"]
        streamed += references * entry["width"]
        if entry["indirect"]:
            # The mesh holds `nnodes` of these, not `nelements * n_local_nodes`:
            # a node's value is fetched once however many elements reach it, so
            # its cost per element is the node-to-element ratio, once for the
            # array rather than once for each of its subscripts.
            per_node = 2 if entry["updates"] else 1
            compulsory += per_node * entry["width"] * nodes_per_element
        else:
            compulsory += references * entry["width"]

    return {
        "kernel": kernel,
        "instantiation": bindings,
        "flops_per_element": flops,
        "weighted_flops_per_element": weighted,
        "streamed_bytes_per_element": streamed,
        "compulsory_bytes_per_element": compulsory,
        "arrays": detail,
        "per_element_scopes": len(scopes),
    }


_IMPL = re.compile(r"^static SFEM_INLINE int ([A-Za-z_][A-Za-z_0-9]*)_impl\(", re.M)


def kernels_in(path):
    """``(name, signature_prefix)`` for every templated kernel in one header."""
    source = open(path).read()
    return [
        (match.group(1), "static SFEM_INLINE int %s_impl(" % match.group(1))
        for match in _IMPL.finditer(source)
    ]


def analyse_tree(generated, pattern=None, widths=None, nodes_per_element=1.0,
                 overrides=None):
    widths = widths or resolve_widths()
    records = []
    for path in sorted(glob.glob(os.path.join(generated, "**", "*.hpp"), recursive=True)):
        try:
            source = open(path).read()
        except (IOError, OSError):
            continue
        # The `extern "C"` wrappers sit beside the header, in their own
        # translation unit; they are where the template parameters acquire
        # concrete types.
        siblings = [
            open(neighbour).read()
            for neighbour in sorted(
                glob.glob(os.path.join(os.path.dirname(path), "*.cpp"))
            )
        ]
        for name, prefix in kernels_in(path):
            if pattern and not re.search(pattern, name):
                continue
            record = analyse_kernel(source, prefix, widths, nodes_per_element,
                                    siblings, overrides)
            if record is None:
                continue
            record["path"] = os.path.relpath(path, generated)
            records.append(record)
    return records


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def intensity(record, model="compulsory"):
    bytes_per_element = record["%s_bytes_per_element" % model]
    if not bytes_per_element:
        return float("inf")
    return record["flops_per_element"] / bytes_per_element


def measured_gflops(point, record):
    """A measured dof rate, as a FLOP rate, using this kernel's own model."""
    flops_per_dof = (
        record["flops_per_element"] * point["nelements"] / float(point["ndof"])
    )
    return point["mdofs"] * 1e6 * flops_per_dof / 1e9


def report_lines(records, machine, points=(), scalar_bytes=8, cores=None):
    lines = [
        "%s" % machine.label,
        "  peak %.0f GFLOP/s at %d-byte scalars (%.0f without SIMD), %.0f GB/s, "
        "ridge %.2f FLOP/byte"
        % (
            machine.peak_flops(scalar_bytes, cores) / 1e9,
            scalar_bytes,
            machine.scalar_flops(scalar_bytes, cores) / 1e9,
            machine.bandwidth_bytes_per_s / 1e9,
            machine.ridge_intensity(scalar_bytes, cores),
        ),
        "  %s" % machine.source,
        "",
        "%-52s %9s %9s %9s %7s %7s"
        % ("kernel", "FLOP/el", "B/el(str)", "B/el(cmp)", "I(str)", "I(cmp)"),
    ]
    for record in records:
        lines.append(
            "%-52s %9d %9.0f %9.0f %7.2f %7.2f"
            % (
                record["kernel"][:52],
                record["flops_per_element"],
                record["streamed_bytes_per_element"],
                record["compulsory_bytes_per_element"],
                intensity(record, "streamed"),
                intensity(record, "compulsory"),
            )
        )
    if points:
        by_name = {record["kernel"]: record for record in records}
        lines.extend(["", "%-40s %6s %10s %10s %8s %10s"
                      % ("measured", "thr", "MDOF/s", "GFLOP/s", "of peak",
                         "of no-SIMD")])
        for point in points:
            record = by_name.get(point["kernel"])
            if record is None:
                continue
            achieved = measured_gflops(point, record)
            peak = machine.peak_flops(point.get("scalar_bytes", scalar_bytes),
                                      point.get("threads", cores)) / 1e9
            scalar_peak = machine.scalar_flops(
                point.get("scalar_bytes", scalar_bytes), point.get("threads", cores)
            ) / 1e9
            lines.append(
                "%-40s %6s %10.1f %10.1f %7.1f%% %9.1f%%"
                % (point.get("label", point["kernel"])[:40], point.get("threads", "-"),
                   point["mdofs"], achieved, 100.0 * achieved / peak,
                   100.0 * achieved / scalar_peak)
            )
    return lines


def short_labels(names):
    """The part of each kernel's name that distinguishes it from the others.

    Every kernel in one report shares a material and an element, and printing
    them thirty characters deep into a plot annotation says nothing while
    costing the space the distinguishing word needed.
    """
    trimmed = [re.sub(r"_a_msoa$", "", name) for name in names]
    if len(trimmed) < 2:
        return {name: short for name, short in zip(names, trimmed)}
    parts = [short.split("_") for short in trimmed]
    common = 0
    while all(len(p) > common + 1 and p[common] == parts[0][common] for p in parts):
        common += 1
    return {
        name: "_".join(part[common:]) for name, part in zip(names, parts)
    }


def plot(records, machine, points, out_path, scalar_bytes=8, cores=None, title=None):
    """A log-log roofline: each kernel a band, each measurement a point.

    The band is the kernel's intensity between its two traffic models, because
    a mesh kernel does not have one intensity -- how much of a gather is
    compulsory is a property of the connectivity.  One colour per kernel, shared
    between its band and its measurement, so the dot can be read against the
    band it belongs to.
    """
    try:
        import matplotlib
    except ImportError:
        raise SystemExit(
            "matplotlib is needed for --plot and is not in this interpreter.\n"
            "Install it, or point SFEM_PYTHON at a venv that has it; the text\n"
            "report above needs nothing extra."
        )
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = cores or machine.cores
    peak = machine.peak_flops(scalar_bytes, cores) / 1e9
    scalar_peak = machine.scalar_flops(scalar_bytes, cores) / 1e9
    bandwidth = machine.bandwidth_bytes_per_s / 1e9
    ridge = peak / bandwidth

    by_name = {record["kernel"]: record for record in records}
    labels = short_labels([record["kernel"] for record in records])
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour_of = {
        record["kernel"]: palette[index % len(palette)]
        for index, record in enumerate(records)
    }

    lows = [min(intensity(r, "streamed"), intensity(r, "compulsory")) for r in records]
    highs = [max(intensity(r, "streamed"), intensity(r, "compulsory")) for r in records]
    left = min(lows + [1.0]) / 4.0
    right = max(highs + [1.0]) * 4.0

    figure, axes = plt.subplots(figsize=(8.5, 5.6))
    grid = [left * (right / left) ** (n / 400.0) for n in range(401)]
    axes.plot(grid, [min(peak, bandwidth * x) for x in grid], color="0.15", linewidth=2,
              label="%s, %d threads  (%.0f GFLOP/s, %.0f GB/s)"
                    % (machine.name, cores, peak, bandwidth))
    # The ceiling a kernel the compiler declined to vectorise is held under.
    # Where a measurement sits against *this* line rather than the one above it
    # is the whole of what these kernels' rooflines have to say.
    axes.plot(grid, [min(scalar_peak, bandwidth * x) for x in grid], color="0.45",
              linewidth=1.2, linestyle="--",
              label="no SIMD, one lane per operation  (%.0f GFLOP/s)" % scalar_peak)
    axes.axvline(ridge, color="0.8", linestyle=":", linewidth=1)

    for index, record in enumerate(records):
        colour = colour_of[record["kernel"]]
        low, high = lows[index], highs[index]
        axes.axvspan(low, high, color=colour, alpha=0.12, linewidth=0)
        # Staggered: neighbouring bands can be a factor of two apart and their
        # labels wider than that, so a single height makes them illegible.
        axes.annotate(
            labels[record["kernel"]],
            ((low * high) ** 0.5, peak * (1.25 if index % 2 == 0 else 1.75)),
            fontsize=8, color=colour, ha="center", va="bottom",
        )

    for point in points:
        record = by_name.get(point["kernel"])
        if record is None:
            continue
        achieved = measured_gflops(point, record)
        model = point.get("model", "compulsory")
        axes.plot([intensity(record, model)], [achieved], "o", markersize=8,
                  color=colour_of[point["kernel"]], markeredgecolor="white",
                  markeredgewidth=0.8,
                  label="%s  %.0f GFLOP/s (%.0f%% of no-SIMD)"
                        % (point.get("label", labels[point["kernel"]]), achieved,
                           100.0 * achieved / scalar_peak))

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(left, right)
    axes.set_ylim(peak / 300.0, peak * 3.0)
    axes.set_xlabel("arithmetic intensity (FLOP / byte)   "
                    "band = streamed .. compulsory traffic")
    axes.set_ylabel("GFLOP/s")
    axes.set_title(title or "%s" % machine.label, fontsize=10)
    axes.grid(True, which="both", linewidth=0.3, alpha=0.35)
    axes.legend(fontsize=7, loc="lower right", framealpha=0.92)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("generated", help="Generated tree to read kernels from.")
    parser.add_argument("--kernel", help="Regular expression selecting kernels.")
    parser.add_argument("--machine", default="grace", choices=sorted(MACHINES))
    parser.add_argument("--threads", type=int, help="Cores the ceiling assumes.")
    parser.add_argument("--scalar-bytes", type=int, default=8)
    parser.add_argument(
        "--nodes-per-element", type=float, default=1.0,
        help="nnodes/nelements of the mesh, for the compulsory traffic model.",
    )
    parser.add_argument("--config", action="append", default=[],
                        help="Config header to resolve scalar typedefs from.")
    parser.add_argument(
        "--bind", action="append", default=[], metavar="NAME=TYPE",
        help="Override a template parameter's concrete type, for modelling an "
             "instantiation the published ABI does not use (a double store, say).",
    )
    parser.add_argument("--measured", help="JSON of measured points.")
    parser.add_argument("--plot", help="Write the roofline plot here.")
    parser.add_argument("--json", action="store_true", help="Emit the model as JSON.")
    parser.add_argument("--list", action="store_true", help="List kernels and stop.")
    arguments = parser.parse_args(argv)

    machine = MACHINES[arguments.machine]
    overrides = {}
    for binding in arguments.bind:
        name, _, concrete = binding.partition("=")
        base = resolve_widths(arguments.config)
        if concrete not in base:
            parser.error("unknown type in --bind: %s" % concrete)
        overrides[name] = base[concrete]
    widths = resolve_widths(arguments.config, overrides)

    points = []
    nodes_per_element = arguments.nodes_per_element
    if arguments.measured:
        measured = json.load(open(arguments.measured))
        points = measured.get("points", [])
        if measured.get("machine") in MACHINES and "--machine" not in (argv or sys.argv):
            machine = MACHINES[measured["machine"]]
        if nodes_per_element == 1.0 and points:
            first = points[0]
            if first.get("nnodes") and first.get("nelements"):
                nodes_per_element = first["nnodes"] / float(first["nelements"])

    records = analyse_tree(arguments.generated, arguments.kernel, widths,
                           nodes_per_element, overrides)
    if arguments.list:
        for record in records:
            print("%-60s %s" % (record["kernel"], record["path"]))
        return 0
    if not records:
        print("no kernels matched", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps({"machine": machine.name, "records": records}, indent=2))
    else:
        print("\n".join(report_lines(records, machine, points,
                                     arguments.scalar_bytes, arguments.threads)))
    if arguments.plot:
        written = plot(records, machine, points, arguments.plot,
                       arguments.scalar_bytes, arguments.threads)
        print("\nwrote %s" % written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
