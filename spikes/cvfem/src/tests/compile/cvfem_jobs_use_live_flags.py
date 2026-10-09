"""Every option a job script passes to a CVFEM driver must still be one that driver parses.

This gate exists because of a specific, expensive failure. DESIGN.md's second correction removed
the micro-kernel selector, and with it the drivers' `--kernel` flag -- and, with the hand-written
scalar higher-order hosts, `--ho-scalar` and `--ho-simd`. Twenty-six job scripts passed those
flags, including thirteen that feed the paper's generated macros, and a driver rejects an unknown
option and exits: every one of them would die on its first `srun`. Nothing noticed. The breakage
was found only when a re-measurement of the paper's higher-order table came back with `-` in
every row, because the jobs scrape the driver's stdout with `sed` and an empty match reads as a
missing number rather than as a failure.

So the invariant: a `--flag` written in `jobs/` names an option the driver it reaches still
parses. A driver's accepted set comes from the `arg == "--name"` comparisons it is parsed with --
not from its `--help` text, which is prose and has drifted from the parser before.

Attributing a flag to the driver rather than to the launcher around it takes two passes, because
the jobs write their options in two places:

* **On the command line**, where the driver's own options are what follows the driver. The scan
  starts at the driver's name, or at the variable holding it, so everything to its left --
  `timeout --kill-after`, `srun --cpus-per-task`, `perf stat -e` -- belongs to a different
  program and is not read. This half knows *which* driver, so it checks the exact option set.
* **In an arm table**, the dominant idiom here: `"ho_unlim_scalar|packed|--rhie-chow --conv-ho 0"`
  is a string that reaches the driver as `$extra` many lines later, with no driver reference
  anywhere near it. Those lines are checked against the union of the drivers the job can run,
  which is weaker but is what caught `--ho-scalar`. A line naming another program is skipped
  here, since its flags are that program's.

A job that reaches its driver in a way neither pass can see -- a variable assigned in an included
file, or `$@` -- is reported as unattributed rather than passed silently, so the count is visible
if it grows.
"""
import os, re, glob, sys

# ctest runs with the build directory as the working directory.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

accepts = {}
for d in sorted(glob.glob("src/drivers/*.cpp")):
    t = open(d).read()
    accepts[os.path.basename(d)[:-4]] = \
        set(re.findall(r'==\s*"(--[a-z0-9][a-z0-9-]*)"', t)) | \
        set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"\s*==', t))
NAMES = sorted(accepts, key=len, reverse=True)
UNION = set().union(*accepts.values())

# A variable holds a driver only if its value names one exactly: `OUT=cvfem_jacho` and
# `CSV=.../cvfem_hex8_bench.csv` are not drivers and must not claim the flags on their lines.
ASSIGN = re.compile(r'(?:^|[;&|]|\bthen\b|\belse\b|\bdo\b)\s*'
                    r'(?:local\s+|export\s+|declare\s+-\w+\s+)?([A-Za-z_]\w*)="?\S*?(' +
                    "|".join(re.escape(n) for n in NAMES) + r')(?![\w.])')
FLAG = re.compile(r"(?<![\w.=-])(--[a-z0-9][a-z0-9-]*)")

# The programs the jobs run beside the driver. A line naming one of these is not read by the
# arm-table pass, because the flags on it are that program's.
PROG = re.compile(r'\b(srun|sbatch|squeue|scancel|mpirun|timeout|perf|python3?|ctest|cmake|make|'
                  r'git|uenv|module|numactl|taskset|nvidia-smi|lscpu|nproc|free|du|'
                  r'awk|sed|grep|sort|tail|head|cat|printf|echo|tee|column|jq|wc|xargs|find|'
                  r'bash|sh|env|ssh|scp|rsync|cp|mv|rm|mkdir|ls|date|'
                  r'perf_regression\.sh|cvfem_kernel_report|record_cmd|_cmd\+=|die|err|fail|warn|usage)\b'
                  r'|\$\{?(?:PY|ANALYZER|PYTHON)\}?')
# Flags of other tools that appear on a line naming none of the above. Each is listed with its
# owner, because a bare allowlist is how a real finding gets waved through.
FOREIGN = {
    "--against": "scripts/perf_regression.sh",
    "--record":  "scripts/perf_regression.sh",
    "--html":    "python/cvfem_kernel_report.py",
    "--call-graph": "perf record",
    "--j":       "perf record, on a continuation line",
}

JOBS = sorted(glob.glob("jobs/*.sbatch")) + sorted(glob.glob("tools/*.sh")) + \
       sorted(glob.glob("scripts/*.sh"))

# A hand-written script in scripts/ has a CLI of its own, and its option names appear in three
# places that are not a call to anything: the `case` branch that parses them, the usage text in
# a heredoc, and the comment block at the top. Only the first two need code -- comments are
# already skipped -- and both are skipped as LINES rather than by subtracting the names, so a
# script that declares `--kernel` for itself and then also hands `--kernel` to the driver is
# still caught on the second line. Both of them do, and both are broken.
CASE_PATTERN = re.compile(r'^\s*\(?\s*(--[a-z0-9][a-z0-9-]*(\s*\|\s*-{1,2}[a-z0-9-]+)*)\s*\)')
HEREDOC = re.compile(r"<<-?\s*'?\"?([A-Za-z_][A-Za-z0-9_]*)'?\"?")

bad, checked, unattributed = [], 0, []
for j in JOBS:
    lines = open(j).read().split("\n")
    # Heredoc bodies are text the script prints, not commands it runs.
    text, term = set(), None
    for no, line in enumerate(lines, 1):
        if term is None:
            m = HEREDOC.search(line)
            if m:
                term = m.group(1)
        else:
            text.add(no)
            if line.strip() == term:
                term = None
    var = {}
    for line in lines:
        if line.strip().startswith("#"):
            continue
        m = ASSIGN.search(line)
        if m:
            var[m.group(1)] = m.group(2)

    starts = []
    for v, drv in var.items():
        starts += [("${%s}" % v, drv), ("$%s" % v, drv)]
    starts += [(n, n) for n in NAMES]
    starts.sort(key=lambda p: -len(p[0]))
    mine = set().union(*(accepts[d] for d in set(var.values()))) if var else UNION

    # An argv built up in a bash array is attributed by where the array is EXPANDED, which is a
    # dataflow question and not a guess at its name. `xcrun "${xctrace_args[@]}" --launch --
    # "$BENCH" "${bench_args[@]}"` expands both on one line: the one to the right of the driver
    # carries the driver's options, the one to its left carries xcrun's. So the lines that
    # assign to the former are scanned against the driver, and the latter's are not.
    argv_of = {}
    for line in lines:
        if line.strip().startswith("#"):
            continue
        at = min([k for k, _ in ((line.find(t), d) for t, d in starts) if k >= 0] or [None]) \
             if any(line.find(t) >= 0 for t, _ in starts) else None
        if at is None:
            continue
        drv = next(d for t, d in starts if line.find(t) == at)
        for m in re.finditer(r'\$\{(\w+)\[[@*]\]\}', line):
            if m.start() > at:
                argv_of[m.group(1)] = drv
    ARRAY_ADD = re.compile(r'^\s*(?:local\s+-\w+\s+|declare\s+-\w+\s+)?(\w+)\+?=\(')

    # A backslash continuation is one command spread over several lines, so the lines are
    # joined before anything is attributed. Without this, the flags on the second line of an
    # `srun \` invocation look like a command of their own with no launcher in sight.
    logical, buf, start = [], "", None
    for no, line in enumerate(lines, 1):
        if line.strip().startswith("#") or no in text or CASE_PATTERN.match(line):
            if buf and not line.rstrip().endswith("\\"):
                logical.append((start, buf)); buf, start = "", None
            continue
        if start is None:
            start = no
        buf += " " + line.rstrip("\\")
        # A bash array assignment spans lines with no backslash at all, so an unbalanced `(`
        # keeps the command open: `local -a xctrace_args=(` and the six lines of its body are
        # one command, and the driver is not in it.
        if not line.rstrip().endswith("\\") and buf.count("(") <= buf.count(")"):
            logical.append((start, buf)); buf, start = "", None
    if buf:
        logical.append((start, buf))

    hit = False
    for no, line in logical:
        # Pass one: the leftmost driver reference that starts a command rather than sitting on
        # the right of an assignment (`BIN=$NEW` passes nothing to anything).
        at, drv = None, None
        for tok, d in starts:
            k = line.find(tok)
            if k > 0 and line[k - 1] == "=":
                continue
            if k >= 0 and (at is None or k < at):
                at, drv = k, d
        if at is not None:
            hit = True
            for f in sorted(set(FLAG.findall(line[at:]))):
                checked += 1
                if f not in accepts[drv] and f not in FOREIGN:
                    bad.append((j, no, f, drv, line.strip()[:110]))
            continue
        # Pass two: a line that assigns to an array known to hold the driver's argv.
        m = ARRAY_ADD.match(line)
        if m and m.group(1) in argv_of:
            drv = argv_of[m.group(1)]
            for f in sorted(set(FLAG.findall(line))):
                checked += 1
                if f not in accepts[drv] and f not in FOREIGN:
                    bad.append((j, no, f, drv, line.strip()[:110]))
            continue
        if m:
            continue   # an array that never reaches a driver
        # Pass three: an arm table, on a line that names no other program.
        if PROG.search(line):
            continue
        for f in sorted(set(FLAG.findall(line))):
            if f in FOREIGN:
                continue
            checked += 1
            if f not in mine:
                who = "/".join(sorted(set(var.values()))) or "any driver"
                bad.append((j, no, f, who, line.strip()[:110]))
    if not hit and any("--" in l for l in lines):
        unattributed.append(j)

if bad:
    bad.sort()
    print(f"\n{len(bad)} option(s) no longer parsed by the driver they reach:\n")
    for j, no, f, drv, s in bad:
        print(f"  {j}:{no}  {f}\n      {drv} has no such option\n      {s}")
    print("\nA driver rejects an unknown option and exits, so the job dies on its first run and")
    print("the scrape of its stdout silently returns nothing -- which reads as a missing number,")
    print("not as a failure. Drop the flag, or restore the option it names.")
    sys.exit(1)

print(f"{len(JOBS)} scripts, {checked} driver options passed: every one is parsed by the driver "
      f"it reaches\n({len(unattributed)} scripts reach no driver this can attribute)")
