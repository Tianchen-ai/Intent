"""One tool request per process; Python source and SDK state end with it."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def load_definition(program: str, symbol: str):
    from intent.api import KernelDefinition

    if not symbol.isidentifier():
        raise ValueError("kernel must be a module-level Python identifier")
    path = Path(program).expanduser()
    if path.suffix == ".py" or "/" in program:
        path = path.resolve(strict=True)
        if not path.is_file() or path.suffix != ".py":
            raise ValueError("program must name an existing Python file")
        name = "_intent_user_program"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load Python program {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(program)
    definition = getattr(module, symbol)
    if not isinstance(definition, KernelDefinition):
        raise TypeError(f"{program}:{symbol} is not an @intent.kernel definition")
    return definition


def main() -> None:
    from .compilation import (_compile_request, doctor, generate_ir_request,
                              materialize_request, optimize_request)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.load(sys.stdin)
    actions = {"compile": _compile_request, "generate_from_ir": generate_ir_request,
               "materialize": materialize_request, "optimize": optimize_request,
               "environment": doctor}
    result = actions[request["action"]](**request["arguments"])
    # A separate result channel keeps arbitrary host/native stdout and stderr
    # away from the tool protocol, including writes that bypass Python streams.
    arguments.result.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
