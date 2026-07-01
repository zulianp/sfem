#!/usr/bin/env python3
import argparse
import json
import os
from types import SimpleNamespace

try:
    from ._script_common import bootstrap_python_path, print_generation_result
except ImportError:
    from _script_common import bootstrap_python_path, print_generation_result


bootstrap_python_path(__file__, 2)

from sfem import gen  # noqa: E402


def _read_manifests(paths):
    manifests = []
    for path in paths:
        with open(path, encoding="utf-8") as input_file:
            manifests.append(json.load(input_file))
    return manifests


def _write_files(out_dir, files):
    paths = []
    for name, source in sorted(files.items()):
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path) or out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as output:
            output.write(source)
        paths.append(path)
    return tuple(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate aggregate SFEM factory registration from generated-op manifests."
    )
    parser.add_argument(
        "manifest",
        nargs="+",
        help="Path to op/sfem_<Op>_manifest.json; may be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory for sfem_generated_ops_registration.{hpp,cpp}.",
    )
    parser.add_argument(
        "--function-name",
        default="register_generated_ops",
        help="Aggregate registration function name in namespace sfem.",
    )
    args = parser.parse_args(argv)

    files = gen.generate_op_registration_files(
        _read_manifests(args.manifest),
        function_name=args.function_name,
    )
    paths = _write_files(os.path.abspath(args.out_dir), files)
    print_generation_result(
        SimpleNamespace(sources=paths, objects=(), plan_dump=None),
        "Generated generated-op factory registration:",
    )


if __name__ == "__main__":
    main()
