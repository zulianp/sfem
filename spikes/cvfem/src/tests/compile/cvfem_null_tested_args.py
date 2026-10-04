"""Find pointer parameters that are NULL-TESTED, and call sites that pass them unguarded.

A sweep that writes `if (mask)` or `mask ? ... : ...` is using null to mean "this input is
absent". A call site that spells it `d.mask.data()` breaks that: std::vector::data() is only
required to return null for a vector that never allocated, and SSMeshData clears several of
these -- sscvfem_classify_macros does `d.macro_curved.clear()` after assigning nmacro entries --
so the pointer is non-null and the sweep reads the stale buffer.
"""
import os
import re, glob, sys

# ctest runs with the build directory as the working directory, so the roots are resolved from
# this file's location rather than from the cwd.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

def paren_end(t, i):
    d = 0
    while i < len(t):
        if t[i] == "(": d += 1
        elif t[i] == ")":
            d -= 1
            if d == 0: return i
        i += 1
    raise AssertionError

def split_top(a):
    out, cur, depth = [], "", 0
    for ch in a:
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1
        if ch == "," and depth == 0: out.append(cur); cur = ""
        else: cur += ch
    out.append(cur)
    return out

FILES = sorted(set(glob.glob("src/**/*.hpp", recursive=True)) | set(glob.glob("src/**/*.cpp", recursive=True)))
DEFS = {}   # name -> {param index: param name} for null-tested pointer params
PARAMS, BODIES = {}, {}
for f in FILES:
    t = open(f).read()
    for m in re.finditer(r"^(?:template <[^>]*>\n)?(?:static |inline )[^;{\n]*?\b(sscvfem_\w+)\(", t, re.M):
        k = t.index("(", m.end() - 1); j = paren_end(t, k)
        if t[j + 1:].lstrip(" \n")[:1] != "{": continue
        head = re.sub(r"//[^\n]*", "", t[k + 1:j])
        body = t[j:t.index("\n}\n", j) + 3]
        body = re.sub(r"//[^\n]*", "", body)
        tested = {}
        for i, p in enumerate(split_top(head)):
            nm = re.findall(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", p.strip())
            if not nm or "*" not in p: continue
            n = nm[0]
            if re.search(r"(?:if\s*\(\s*!?" + n + r"\s*[)&|]|!\s*" + n + r"\b|\b" + n + r"\s*\?|\b" + n
                         + r"\s*(?:==|!=)\s*nullptr|\b" + n + r"\s*&&)", body):
                tested[i] = n
        if tested: DEFS.setdefault(m.group(1), {}).update(tested)
        PARAMS[m.group(1)] = [re.findall(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", p.strip())
                              for p in split_top(head)]
        BODIES[m.group(1)] = (f, body)

# TRANSITIVE. A sweep that forwards a parameter to a kernel which null-tests it is itself
# null-testing it: sscvfem_block_diag_sweep never writes `if (macro_curved)`, it hands the
# pointer to sscvfem_macro_curved, which is where `macro_curved && macro_curved[e]` lives. One
# level of inspection found nothing at all, which is how this class of bug stays invisible.
changed = True
while changed:
    changed = False
    for host, (f, body) in BODIES.items():
        names = {n[0]: i for i, n in enumerate(PARAMS.get(host, [])) if n}
        for callee, tested in list(DEFS.items()):
            for cm in re.finditer(r"(?<![\w])" + callee + r"\s*(?:<[^>()]*>)?\s*\(", body):
                try: k = paren_end(body, cm.end() - 1)
                except AssertionError: continue
                parts = [q.strip() for q in split_top(body[cm.end():k])]
                for i, _ in tested.items():
                    if i >= len(parts): continue
                    if parts[i] in names and i not in DEFS.get(host, {}):
                        idx = names[parts[i]]
                        if idx not in DEFS.get(host, {}):
                            DEFS.setdefault(host, {})[idx] = parts[i]
                            changed = True

bad = []
for f in FILES:
    t = open(f).read()
    for name, tested in DEFS.items():
        for cm in re.finditer(r"(?<![\w])" + name + r"\s*(?:<[^>()]*>)?\s*\(", t):
            if t[:cm.start()].rstrip().endswith("void"): continue
            k = paren_end(t, cm.end() - 1)
            parts = [q.strip() for q in split_top(t[cm.end():k])]
            if len(parts) < max(tested) + 1: continue
            for i, n in tested.items():
                a = parts[i]
                if re.fullmatch(r"(?:[\w.\->]+)\.data\(\)", a) and "empty()" not in a:
                    bad.append((f, t[:cm.start()].count("\n") + 1, name, n, a))
for f, ln, name, n, a in bad:
    print(f"{f}:{ln}: {name}'s `{n}` is null-tested but passed `{a}`")
print(f"\n{len(bad)} unguarded call-site argument(s); {len(DEFS)} function(s) null-test a pointer parameter")
sys.exit(1 if bad else 0)
