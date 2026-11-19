#!/usr/bin/env python3
"""The documentation is true to the code.

  - Every fenced block marked ```yaml playground in README.md and docs/ is a Playground case that
    passes the schema (`playground case.yaml --dry-run`) and, with --run, runs to completion.
  - Every relative link in README.md and docs/ points at a file or folder that exists.

Usage: check_docs.py <source root> [--playground <path>] [--run]
Exit status 1 lists each finding.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

CASE = re.compile(r"^```yaml playground\n(.*?)^```", re.S | re.M)
LINK = re.compile(r"\]\(([^)\s]+)\)")


def documents(root):
    yield os.path.join(root, "README.md")
    for directory, _subdirectories, files in os.walk(os.path.join(root, "docs")):
        for name in sorted(files):
            if name.endswith(".md"):
                yield os.path.join(directory, name)


def check_links(root, path, text):
    for target in LINK.findall(text):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        target = target.split("#")[0]
        resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
        if not os.path.exists(resolved):
            yield "%s: broken link %s" % (os.path.relpath(path, root), target)


def check_cases(root, path, text, playground, run):
    for k, case in enumerate(CASE.findall(text)):
        label = "%s: case %d" % (os.path.relpath(path, root), k + 1)
        with tempfile.TemporaryDirectory() as tmp:
            case_file = os.path.join(tmp, "case.yaml")
            with open(case_file, "w") as f:
                f.write(case)
            for args in [["--dry-run"]] + ([[]] if run else []):
                result = subprocess.run([playground, case_file] + args, cwd=tmp, capture_output=True, text=True)
                if result.returncode != 0:
                    tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
                    yield "%s %s: exit %d: %s" % (label, " ".join(args) or "(run)", result.returncode, " | ".join(tail))
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--playground", help="the playground executable; without it cases are not checked")
    parser.add_argument("--run", action="store_true", help="run each case too, not only --dry-run")
    args = parser.parse_args()

    if args.playground:
        args.playground = os.path.abspath(args.playground)

    findings, cases = [], 0
    for path in documents(args.root):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            text = f.read()
        findings.extend(check_links(args.root, path, text))
        if args.playground:
            cases += len(CASE.findall(text))
            findings.extend(check_cases(args.root, path, text, args.playground, args.run))

    for finding in findings:
        print(finding)
    print("%d playground cases checked, %d findings" % (cases, len(findings)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
