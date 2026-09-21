#!/usr/bin/env python3
"""Write the headers that belong to a target rather than to any material.

Kernel maths, the geometry kernels, the diagnostics record, the packed thread
scratch and the sum-factorization micro-kernels.  A material generation writes
its target's set as a side effect, which covers the `.hpp` ones -- every
material in `regenerate_all.sh` builds for OpenMP.  Nothing covers the `.cuh`
ones, because generating for CUDA needs `SFEM_GENERATE_CUDA=1` and also writes
a per-material CUDA operator tree the repository does not track.

The consequence was four tracked files that no step produced, no build
consumed and `codegen_snapshot` exempted, drifting several refactors behind
their `.hpp` counterparts: `scalar_t` where the rest of the tree says `s_t`,
no `RSTR` define, `stdio.h` for `cstdio`.  Asking the backend for its shared
primitives needs no material, so it can run every time and the exemption can
go.
"""
import argparse
import os

try:
    from ._script_common import bootstrap_python_path, generated_output_dir
except ImportError:
    from _script_common import bootstrap_python_path, generated_output_dir


bootstrap_python_path(__file__, 3)

from codegen.framework.pipeline.driver import (  # noqa: E402
    BACKENDS_BY_TARGET,
    KernelTarget,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=tuple(target.value for target in KernelTarget),
        help="Target whose shared headers to write; may be repeated.",
    )
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    # CUDA by default because it is the set nothing else writes.  The OpenMP
    # set is written by every material generation, and writing it again here
    # would be a second path to the same files.
    targets = tuple(KernelTarget(name) for name in (args.targets or ("cuda",)))
    out_dir = os.path.abspath(args.out_dir or generated_output_dir(__file__, "", 4))
    written = []
    for target in targets:
        for generated in BACKENDS_BY_TARGET[target].shared_primitive_files():
            path = os.path.join(out_dir, generated.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            source = generated.source
            if not source.endswith("\n"):
                source += "\n"
            with open(path, "w") as handle:
                handle.write(source)
            written.append(path)
    print("Generated shared headers for %s:" % ", ".join(t.value for t in targets))
    for path in written:
        print("  %s" % path)


if __name__ == "__main__":
    main()
