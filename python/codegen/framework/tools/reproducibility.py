"""Input-output reproducibility for the generated kernels.

Every step of the layering work has been gated on generated source coming back
byte-identical.  That is the strongest possible statement while it holds, and it
settles performance by construction -- identical text compiles to identical
kernels.  It also stops being available the moment a change is *meant* to alter
the text: removing the dead temporaries of OP 11, hoisting the graph validation
of OP 10, reconciling the assembly plan names of OP 13, and every loop-structure
pass the IR is supposed to hold.  For those, byte-identity is not the proof, it
is the thing that must break.

This is the gate that replaces it, and it asks the only question that survives a
deliberate output change: **given the same inputs, does the kernel still produce
the same numbers?**

It differs from ``apply_bench`` in scope rather than in kind.  That tool proves
nine matrix-free apply variants of one material agree with each other and with a
recorded answer, which is a parity check between kernels.  This one drives every
entry point it can bind, across every maintained material, and records what each
one returns -- so a change that alters one kernel's arithmetic is caught even
when there is no sibling kernel to disagree with it.

How a kernel is driven
----------------------

Nothing here knows anything about any particular material.  The generated
manifest lists every C ABI entry point with its full declaration, and the
parameter *names* in those declarations are the ABI that ``plans/streams.py``
defines: ``nelements``, ``elements``, ``points``, ``g_jacobian_adjugate<i>``,
``<role>_stride``, ``<field>``, ``<field>_old``, ``<field>_direction``,
``<field>_out``.  Binding a call is therefore a matter of reading names, not of
knowing physics, and a material added tomorrow is driven without touching this
file.

An entry point whose parameters cannot all be bound -- one that wants a sparsity
pattern, a packed partition, or a geometry metric this harness does not build --
is skipped and reported as skipped.  That list is part of the output on purpose:
it says exactly how much of the surface is covered, rather than leaving the
uncovered part invisible.

Inputs are a pure function of the parameter's name and the index, so two runs on
two machines fill the same arrays.  The mesh is a Cartesian grid of hexahedra
whose map is affine, so the affine and isoparametric kernels see the same
geometry and are expected to agree; that is inherited from ``apply_bench`` and
is a real check on geometry routing rather than a convenience.

Usage::

    python -m codegen.framework.tools.reproducibility --material laplace
    python -m codegen.framework.tools.reproducibility --all
    python -m codegen.framework.tools.reproducibility --all --record

``--record`` rewrites the committed baseline and is a deliberate act: it is what
you run *after* deciding that an output change is intended, and the diff it
produces is the reviewable statement of which kernels moved.
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile


BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reproducibility.json"
)

#: Digests are compared as a relative change.  The kernels are deterministic, so
#: a rerun on the same machine reproduces bit-for-bit; this tolerance covers
#: compiler and libm differences across machines, not genuine drift.
TOLERANCE = 1e-9

#: The materials the snapshot maintains.  Kept as a list rather than discovered
#: so that a material silently failing to generate is a visible absence.
MATERIALS = (
    "laplace",
    "linear_elasticity",
    "neohookean_ogden",
    "saint_venant_kirchhoff",
    "modified_mooney_rivlin",
    "poro_elasticity",
    "stokes",
    "two_phase_flow",
)

#: Element the grid is built from, and the elements that must be generated with
#: it because their kernels are aliases forwarding to another's.
ELEMENT = "HEX8"
ALIAS_TARGETS = ("PROTEUS_HEX8",)

#: Materials this harness cannot drive, and why.  These are facts about the
#: material rather than defects in the tool, so they are stated rather than
#: left to surface as a failure -- and they are the honest edge of the
#: coverage, which is the number that matters when this gate is used to justify
#: an intended output change.
NOT_COVERED = {
    "poro_elasticity": (
        "Taylor-Hood: enabled for TRI6_TRI3 / TET10_TET4 / HEX27_HEX8, whose "
        "topology this Cartesian HEX8 grid does not build"
    ),
    "stokes": (
        "Taylor-Hood: enabled for TRI6_TRI3 / TET10_TET4 / HEX27_HEX8, whose "
        "topology this Cartesian HEX8 grid does not build"
    ),
    "two_phase_flow": (
        "its per-form operators each define the same element-generic entry "
        "points, so the translation units cannot be linked into one driver"
    ),
}


def _repo_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(path, "base")) and os.path.isdir(
            os.path.join(path, "python")
        ):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError("could not locate the repository root")
        path = parent


def _compiler():
    for candidate in ("mpic++", "mpicxx", "c++"):
        try:
            subprocess.run(
                [candidate, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def _include_flags(root, generated_dir):
    dirs = [
        generated_dir,
        os.path.join(root, "base"),
        os.path.join(root, "build"),
        os.path.join(root, "build", "external", "smesh"),
        os.path.join(root, "external", "smesh", "src"),
    ]
    for material_dir in sorted(glob.glob(os.path.join(generated_dir, "*"))):
        if os.path.isdir(material_dir):
            dirs.append(material_dir)
            for current, subdirs, _files in os.walk(material_dir):
                subdirs[:] = [d for d in subdirs if d not in ("cuda", "CMakeFiles")]
                dirs.append(current)
    operators_root = os.path.join(root, "operators")
    if os.path.isdir(operators_root):
        for current, subdirs, _files in os.walk(operators_root):
            subdirs[:] = [d for d in subdirs if d not in ("cuda", "CMakeFiles")]
            dirs.append(current)
    smesh_src = os.path.join(root, "external", "smesh", "src")
    if os.path.isdir(smesh_src):
        for current, subdirs, _files in os.walk(smesh_src):
            subdirs[:] = [d for d in subdirs if d not in ("cuda", "CMakeFiles")]
            dirs.append(current)
    return ["-I" + d for d in dirs if os.path.isdir(d)]


# --------------------------------------------------------------------------
# Reading the manifest
# --------------------------------------------------------------------------

_PARAM = re.compile(r"^\s*(?P<type>.+?)\s*(?P<name>[A-Za-z_]\w*)\s*$")


def _parameters(declaration):
    """The (type, name) pairs of one C ABI declaration, in order."""
    inner = declaration[declaration.index("(") + 1 : declaration.rindex(")")]
    params = []
    for part in inner.split(","):
        part = " ".join(part.split())
        if not part:
            continue
        match = _PARAM.match(part)
        if match is None:
            return None
        params.append((match.group("type").strip(), match.group("name")))
    return params


def _manifest_entries(generated_dir, material):
    pattern = os.path.join(generated_dir, material, "op", "*_manifest.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return []
    with open(matches[0], encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest.get("c_abi", [])


# --------------------------------------------------------------------------
# Binding a call
# --------------------------------------------------------------------------

#: Scalar element types the harness can pass a value for.
SCALARS = ("const double", "const float", "const real_t", "const int")

#: Pointer element types that name a field the kernel reads or writes.
IN_FIELDS = (
    "const double *const SFEM_RESTRICT",
    "const float *const SFEM_RESTRICT",
    "const real_t *const SFEM_RESTRICT",
)
OUT_FIELDS = (
    "double *const SFEM_RESTRICT",
    "float *const SFEM_RESTRICT",
    "real_t *const SFEM_RESTRICT",
)


def _scalar_type(ctype):
    for name in ("double", "float", "real_t"):
        if name in ctype:
            return name
    return None


class Unbindable(Exception):
    """A parameter this harness has no deterministic value for."""


def _bind(params, element, components):
    """One C argument expression per parameter, plus the outputs to digest.

    ``components`` is how many field values each node carries in this entry
    point's layout: one for a structure-of-arrays kernel, which takes a stride
    and one pointer per field, and the field count for an array-of-structures
    kernel, which takes a single interleaved array.  Getting it wrong is not a
    wrong answer but a buffer overrun, which is how the first version of this
    crashed on the vector-valued materials.

    Raises ``Unbindable`` naming the first parameter it cannot supply, which is
    what puts an entry point on the skipped list instead of into the baseline.
    """
    args = []
    inputs = []
    outputs = []
    for ctype, name in params:
        if name == "element_type":
            args.append("smesh::ElemType::%s" % element)
        elif name == "nelements":
            args.append("mesh.nelements")
        elif name == "nnodes":
            args.append("mesh.nnodes")
        elif name == "elements" and ctype.startswith("idx_t **"):
            args.append("mesh.element_ptrs.data()")
        elif name == "points" and "*const *const" in ctype:
            args.append("mesh.point_ptrs.data()")
        elif re.fullmatch(r"g_jacobian_adjugate(\d+)", name or ""):
            index = int(re.fullmatch(r"g_jacobian_adjugate(\d+)", name).group(1))
            args.append("mesh.adjugate[%d].data()" % index)
        elif name == "g_jacobian_determinant0":
            args.append("mesh.determinant.data()")
        elif name.endswith("_stride") and ctype == "const ptrdiff_t":
            args.append("1")
        elif ctype in SCALARS:
            args.append("material_scalar(\"%s\")" % name)
        elif ctype in OUT_FIELDS:
            scalar = _scalar_type(ctype)
            buffer = "out_%s_%s" % (scalar, name)
            outputs.append((buffer, scalar, name, components))
            args.append("%s.data()" % buffer)
        elif ctype in IN_FIELDS:
            scalar = _scalar_type(ctype)
            buffer = "in_%s_%s" % (scalar, name)
            inputs.append((buffer, scalar, name, components))
            args.append("%s.data()" % buffer)
        else:
            raise Unbindable("%s %s" % (ctype, name))
    if not outputs:
        raise Unbindable("no output to digest")
    return args, inputs, outputs


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------

DRIVER_HEAD = r"""
// Generated by codegen.framework.tools.reproducibility.  Do not edit.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "sfem_base.hpp"
#include "smesh_elem_type.hpp"

// The grid, its affine geometry, and the deterministic fills are shared with
// codegen.framework.tools.apply_bench: a Cartesian grid of hexahedra whose map
// is affine, so the affine and isoparametric kernels see the same geometry.
struct Mesh {
    ptrdiff_t nelements = 0, nnodes = 0;
    double h = 0.0;
    std::vector<std::vector<idx_t>> elements;
    std::vector<std::vector<geom_t>> points;
    std::vector<idx_t *> element_ptrs;
    std::vector<const geom_t *> point_ptrs;
    std::vector<std::vector<geom_t>> adjugate;
    std::vector<geom_t> determinant;
};

static Mesh build_grid(int n) {
    Mesh m;
    const int nn = n + 1;
    m.h = 1.0 / (double)n;
    m.nnodes = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n;
    m.elements.assign(8, std::vector<idx_t>(m.nelements));
    m.points.assign(3, std::vector<geom_t>(m.nnodes));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    for (int k = 0; k < nn; ++k)
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i) {
                const idx_t id = nid(i, j, k);
                m.points[0][id] = (geom_t)(i * m.h);
                m.points[1][id] = (geom_t)(j * m.h);
                m.points[2][id] = (geom_t)(k * m.h);
            }
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i, ++e) {
                m.elements[0][e] = nid(i, j, k);
                m.elements[1][e] = nid(i + 1, j, k);
                m.elements[2][e] = nid(i + 1, j + 1, k);
                m.elements[3][e] = nid(i, j + 1, k);
                m.elements[4][e] = nid(i, j, k + 1);
                m.elements[5][e] = nid(i + 1, j, k + 1);
                m.elements[6][e] = nid(i + 1, j + 1, k + 1);
                m.elements[7][e] = nid(i, j + 1, k + 1);
            }
    for (auto &row : m.elements) m.element_ptrs.push_back(row.data());
    for (auto &row : m.points) m.point_ptrs.push_back(row.data());
    m.adjugate.assign(9, std::vector<geom_t>(m.nelements, (geom_t)0));
    m.determinant.assign(m.nelements, (geom_t)(m.h * m.h * m.h));
    for (ptrdiff_t i = 0; i < m.nelements; ++i) {
        m.adjugate[0][i] = (geom_t)(m.h * m.h);
        m.adjugate[4][i] = (geom_t)(m.h * m.h);
        m.adjugate[8][i] = (geom_t)(m.h * m.h);
    }
    return m;
}

// A stable hash of the parameter name, so every field is filled differently and
// the same way on every machine.  Two kernels reading the same named field read
// the same numbers, which is what lets their digests be compared.
static double name_seed(const char *name) {
    unsigned long h = 1469598103934665603UL;
    for (const char *p = name; *p; ++p) {
        h ^= (unsigned char)*p;
        h *= 1099511628211UL;
    }
    return (double)(h % 1000u) / 1000.0;
}

template <typename T>
static void fill_field(std::vector<T> &v, const char *name) {
    const double s = name_seed(name);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = (T)(std::sin(0.5 + s + 0.125 * (double)i) * (0.25 + s));
    }
}

// Material parameters are positive and distinct, and never 1, so a kernel that
// drops one is visible in the digest rather than silently correct.
static double material_scalar(const std::string &name) {
    return 0.5 + name_seed(name.c_str());
}

struct Digest {
    double l1 = 0.0, l2 = 0.0;
};

template <typename T>
static void accumulate(Digest &d, const std::vector<T> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const double x = (double)v[i];
        d.l1 += std::fabs(x);
        d.l2 += x * x;
    }
}

static void report(const char *name, const Digest &d) {
    std::printf("kernel %s l1=%.12e l2=%.12e\n", name, d.l1, std::sqrt(d.l2));
}
"""


def _call_block(name, args, inputs, outputs):
    lines = ["    {"]
    for buffer, scalar, param, components in inputs:
        lines.append(
            "        std::vector<%s> %s(nodes * %d); fill_field(%s, \"%s\");"
            % (scalar, buffer, components, buffer, param)
        )
    for buffer, scalar, _param, components in outputs:
        lines.append(
            "        std::vector<%s> %s(nodes * %d, (%s)0);"
            % (scalar, buffer, components, scalar)
        )
    lines.append("        %s(" % name)
    lines.append("            " + ",\n            ".join(args))
    lines.append("        );")
    lines.append("        Digest d;")
    for buffer, _scalar, _param, _components in outputs:
        lines.append("        accumulate(d, %s);" % buffer)
    lines.append('        report("%s", d);' % name)
    lines.append("    }")
    return "\n".join(lines)


def _driver_source(declarations, blocks, refine):
    return "\n".join(
        [
            DRIVER_HEAD,
            'extern "C" {',
            "\n".join(declarations),
            "}",
            "",
            "int main() {",
            "    Mesh mesh = build_grid(%d);  // not const: the kernels take idx_t **const" % refine,
            "    const size_t nodes = (size_t)mesh.nnodes;",
            "",
            "\n\n".join(blocks),
            "    return 0;",
            "}",
            "",
        ]
    )


# --------------------------------------------------------------------------
# Driving one material
# --------------------------------------------------------------------------


def _generate(root, generated, material):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        os.path.join(root, "python") + os.pathsep + env.get("PYTHONPATH", "")
    )
    elements = (ELEMENT,) + ALIAS_TARGETS
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "codegen.framework.generators.%s" % material,
            "--out-dir",
            os.path.join(generated, material),
            *sum((["--element", e] for e in elements), []),
        ],
        cwd=os.path.join(root, "python"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed


def run_material(root, generated, material, refine, compiler, verbose=False):
    """Generate, compile and drive one material.  Returns (digests, skipped)."""
    completed = _generate(root, generated, material)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout.decode("utf-8", "replace"))
        raise RuntimeError("generator failed for %s" % material)

    entries = _manifest_entries(generated, material)
    if not entries:
        raise RuntimeError("no manifest for %s" % material)

    # How many fields the system carries, read off the structure-of-arrays
    # kernels: those take one output pointer per field.  The interleaved
    # array-of-structures kernels take a single pointer holding all of them, so
    # this is what their buffers have to be sized by.
    n_fields = 1
    for entry in entries:
        params = _parameters(entry["declaration"])
        if params is None or not any(n.endswith("_stride") for _t, n in params):
            continue
        outs = sum(1 for t, _n in params if t in OUT_FIELDS)
        n_fields = max(n_fields, outs)

    declarations, blocks, skipped = [], [], []
    for entry in sorted(entries, key=lambda e: e["name"]):
        params = _parameters(entry["declaration"])
        if params is None:
            skipped.append((entry["name"], "unparsed declaration"))
            continue
        interleaved = not any(name.endswith("_stride") for _t, name in params)
        try:
            args, inputs, outputs = _bind(
                params, ELEMENT, n_fields if interleaved else 1
            )
        except Unbindable as reason:
            skipped.append((entry["name"], str(reason)))
            continue
        declarations.append(entry["declaration"].replace('extern "C" ', "").strip())
        blocks.append(_call_block(entry["name"], args, inputs, outputs))

    if not blocks:
        return {}, skipped

    workdir = os.path.join(generated, "_driver_%s" % material)
    os.makedirs(workdir, exist_ok=True)
    driver_path = os.path.join(workdir, "driver.cpp")
    with open(driver_path, "w", encoding="utf-8") as handle:
        handle.write(_driver_source(declarations, blocks, refine))

    operators = sorted(
        os.path.join(directory, name)
        for directory, _dirs, files in os.walk(os.path.join(generated, material))
        for name in files
        # The per-element operators, plus the dispatch units that define the
        # element-generic entry points.  Both are self-contained: the dispatch
        # translation unit includes only the C ABI header, so none of this
        # needs the SFEM library to link.
        if name.endswith("_operator.cpp") or name.endswith("_dispatch.cpp")
    )
    binary = os.path.join(workdir, "driver")
    command = (
        [compiler, "-O2", "-std=c++17", "-o", binary, driver_path]
        + operators
        + _include_flags(root, generated)
    )
    build = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if build.returncode != 0:
        sys.stderr.write(build.stdout.decode("utf-8", "replace")[-4000:])
        raise RuntimeError("driver failed to compile for %s" % material)

    run = subprocess.run([binary], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = run.stdout.decode("utf-8", "replace")
    if run.returncode != 0:
        sys.stderr.write(output)
        raise RuntimeError("driver failed to run for %s" % material)
    if verbose:
        print(output)

    digests = {}
    for match in re.finditer(
        r"kernel (\S+) l1=([-\d.e+]+) l2=([-\d.e+]+)", output
    ):
        digests[match.group(1)] = {
            "l1": float(match.group(2)),
            "l2": float(match.group(3)),
        }
    return digests, skipped


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


#: An affine grid means the isoparametric geometry *is* the affine one, so a
#: kernel pair differing only in geometry mode has to agree.  This is free --
#: both are driven anyway -- and it is the check that catches a bug in the
#: adjugate routing, which a digest alone would happily record as the new
#: correct answer.
PARITY_TOLERANCE = {"double": 1e-12, "float": 1e-6}


def _geometry_parity(measured):
    """Pairs that differ only in geometry mode and disagree anyway."""
    disagreements = []
    for name in sorted(measured):
        if "_affine_" not in name:
            continue
        twin = name.replace("_affine_", "_isoparametric_")
        if twin not in measured:
            continue
        tolerance = PARITY_TOLERANCE["float" if name.endswith("_float") else "double"]
        worst = 0.0
        for key in ("l1", "l2"):
            a, b = measured[name][key], measured[twin][key]
            scale = max(abs(a), abs(b), 1e-30)
            worst = max(worst, abs(a - b) / scale)
        if worst > tolerance:
            disagreements.append((name, twin, worst, tolerance))
    return disagreements


def _compare(recorded, measured):
    """Kernels that moved, that appeared, and that went missing."""
    moved, appeared, missing = [], [], []
    for name, digest in sorted(measured.items()):
        expected = recorded.get(name)
        if expected is None:
            appeared.append(name)
            continue
        worst = 0.0
        for key in ("l1", "l2"):
            scale = max(abs(expected[key]), abs(digest[key]), 1e-30)
            worst = max(worst, abs(expected[key] - digest[key]) / scale)
        if worst > TOLERANCE:
            moved.append((name, worst, expected, digest))
    for name in sorted(recorded):
        if name not in measured:
            missing.append(name)
    return moved, appeared, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--material", action="append", default=None)
    parser.add_argument("--all", action="store_true", help="every maintained material")
    parser.add_argument("--refine", type=int, default=6, help="cells per side")
    parser.add_argument("--record", action="store_true", help="rewrite the baseline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    materials = args.material or (list(MATERIALS) if args.all else ["laplace"])

    compiler = _compiler()
    if compiler is None:
        sys.stderr.write("no C++ compiler available\n")
        return 2

    root = _repo_root()
    recorded = {}
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as handle:
            recorded = json.load(handle)

    measured, skipped_all, failures = {}, {}, []
    with tempfile.TemporaryDirectory(prefix="sfem_reproducibility_") as workdir:
        generated = os.path.join(workdir, "generated")
        os.makedirs(generated)
        for material in materials:
            print("==> %s" % material)
            if material in NOT_COVERED:
                print("    not driven: %s" % NOT_COVERED[material])
                continue
            try:
                digests, skipped = run_material(
                    root, generated, material, args.refine, compiler, args.verbose
                )
            except RuntimeError as error:
                failures.append("%s: %s" % (material, error))
                continue
            measured.update(digests)
            skipped_all[material] = skipped
            print(
                "    %d kernels driven, %d skipped" % (len(digests), len(skipped))
            )

    for material, skipped in sorted(skipped_all.items()):
        reasons = {}
        for _name, reason in skipped:
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:4]:
            print("    %s skipped %d for: %s" % (material, count, reason))

    if failures:
        for failure in failures:
            sys.stderr.write("FAILED %s\n" % failure)
        return 1

    key_scope = set()
    for material in materials:
        key_scope.update(name for name in measured if name.startswith(material))

    disagreements = _geometry_parity(measured)
    if disagreements:
        sys.stderr.write(
            "%d kernel pairs differ by geometry mode alone on an affine mesh:\n"
            % len(disagreements)
        )
        for name, twin, worst, tolerance in disagreements:
            sys.stderr.write(
                "    %s vs %s  rel=%.3e  tolerance=%.0e\n"
                % (name, twin, worst, tolerance)
            )
        return 1
    parity_pairs = sum(
        1
        for name in measured
        if "_affine_" in name and name.replace("_affine_", "_isoparametric_") in measured
    )
    if parity_pairs:
        print("affine/isoparametric parity holds for %d pairs" % parity_pairs)

    if args.record:
        merged = dict(recorded)
        for name in list(merged):
            if any(name.startswith(m) for m in materials):
                del merged[name]
        merged.update(measured)
        with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("recorded %d kernel digests" % len(measured))
        return 0

    scoped = {
        name: digest
        for name, digest in recorded.items()
        if any(name.startswith(m) for m in materials)
    }
    moved, appeared, missing = _compare(scoped, measured)

    if appeared:
        print("new kernels, not yet in the baseline: %d" % len(appeared))
        for name in appeared[:10]:
            print("    %s" % name)
    if missing:
        print("kernels in the baseline that no longer generate: %d" % len(missing))
        for name in missing[:10]:
            print("    %s" % name)
    if moved:
        sys.stderr.write("%d kernel answers moved:\n" % len(moved))
        for name, worst, expected, digest in moved[:20]:
            sys.stderr.write(
                "    %s  rel=%.3e  was l1=%.12e l2=%.12e  now l1=%.12e l2=%.12e\n"
                % (name, worst, expected["l1"], expected["l2"], digest["l1"], digest["l2"])
            )
        sys.stderr.write(
            "\nIf the change was intended, re-record with --record and review the "
            "baseline diff: it is the statement of which kernels moved.\n"
        )
        return 1

    if not scoped:
        print("no baseline recorded for these materials yet; run with --record")
        return 0
    print("all %d kernel answers match the recorded baseline" % len(measured))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
