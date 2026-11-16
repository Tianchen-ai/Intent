"""Run one installed author program; batch checks and measurements belong to Bench."""
import argparse
import json
from .registry import PROGRAMS, load_program
from intent.targets.components import COMPONENTS
from intent.targets.provider import provider

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", nargs="?", choices=PROGRAMS)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--backend", choices=tuple(COMPONENTS))
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--compiler")
    parser.add_argument("--stage", choices=("source", "native", "run"), default="run")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--option", action="append", default=[], metavar="NAME=JSON")
    args = parser.parse_args()
    if args.list:
        for name, (_, description) in PROGRAMS.items():
            print(f"{name}: {description}")
        return
    if args.program is None or args.backend is None:
        parser.error("a program and explicit --backend are required")
    from .programs.common import ExampleContext, describe_outputs
    execution = provider(args.backend).local
    device = args.device or (f"{execution.device_kind}:0" if execution.indexed else execution.device_kind)
    options = {}
    for value in args.option:
        key, separator, encoded = value.partition("=")
        if not separator or key in options:
            parser.error("--option requires distinct NAME=JSON entries")
        options[key] = json.loads(encoded)
    context = ExampleContext(args.backend, device, compiler=args.compiler, workers=args.workers,
                             prepared=args.prepared, stage=args.stage, backend_options=options)
    try:
        describe_outputs(load_program(args.program)(context))
    finally:
        context.close()

if __name__ == "__main__":
    main()
