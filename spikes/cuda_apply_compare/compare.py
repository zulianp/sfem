"""Difference the two runs' output vectors, entry by entry.

The scatter is contended and floating-point addition is not associative, so the
device's accumulation order differs from the host's and the answers are not
expected to be bit-identical.  The tolerance is relative to the largest entry of
the vector.

This replaced a comparison that printed each run's own L2 and L-infinity norms:
two answers can share both norms and still differ entrywise, so that version
could only see gross disagreement.  A `__global__` kernel that kept the host's
serial mesh loop -- which is what the energy family shipped for the constant-P1
simplices until it was measured -- is off by roughly the thread count, and would
have been caught either way; a wrong index would not.
"""
import sys

TOLERANCE = 1e-12


def read(path):
    vectors, name, values = {}, None, []
    for line in open(path):
        if line.startswith("#"):
            fields = line.split()
            if fields[1] == "where":
                continue
            if name is not None:
                vectors[name] = values
            name, values = fields[1], []
        else:
            values.append(float(line))
    if name is not None:
        vectors[name] = values
    return vectors


def main(left_path, right_path):
    left, right = read(left_path), read(right_path)
    if set(left) != set(right):
        print("the two runs recorded different vectors:")
        print("  only in %s: %s" % (left_path, sorted(set(left) - set(right))))
        print("  only in %s: %s" % (right_path, sorted(set(right) - set(left))))
        return 1

    failed, worst_overall = 0, 0.0
    for name in sorted(left):
        a, b = left[name], right[name]
        if len(a) != len(b):
            print("  LENGTH  %s: %d against %d" % (name, len(a), len(b)))
            failed += 1
            continue
        if not all(x == x and abs(x) != float("inf") for x in a + b):
            # fmax-style comparisons return the non-NaN operand, so a
            # non-finite entry has to be rejected before anything is compared
            print("  NON-FINITE  %s" % name)
            failed += 1
            continue
        scale = max(abs(x) for x in a) or 1.0
        nonzero = sum(1 for x in a if x != 0.0)
        worst = max(abs(x - y) for x, y in zip(a, b)) / scale
        worst_overall = max(worst_overall, worst)
        verdict = "ok" if worst < TOLERANCE else "FAIL"
        failed += verdict == "FAIL"
        print("  %-52s n=%-6d nonzero=%-6d largest=%-11.4g worst rel diff=%.3e  %s"
              % (name, len(a), nonzero, scale, worst, verdict))

    print("worst over all vectors: %.3e (tolerance %.0e)" % (worst_overall, TOLERANCE))
    print("FAILED %d vectors" % failed if failed else "all vectors agree")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
