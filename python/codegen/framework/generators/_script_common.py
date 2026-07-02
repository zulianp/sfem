import os
import sys


def bootstrap_python_path(anchor_file, levels_to_python):
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(anchor_file),
            *(("..",) * int(levels_to_python)),
        )
    )
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def generated_output_dir(anchor_file, name, levels_to_frontend_root):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(anchor_file),
            *(("..",) * int(levels_to_frontend_root)),
            "frontend",
            "ops",
            "generated",
            name,
        )
    )


def print_generation_result(result, title="Generated:"):
    print(title)
    for path in result.sources:
        print("  %s" % path)
    if result.objects:
        print("Compiled:")
        for path in result.objects:
            print("  %s" % path)
    if result.plan_dump:
        print("Plan:")
        print("  %s" % result.plan_dump)
