"""Command-line entrypoints for installed IntentDSL packages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import doctor, layout, library_path, manual, setup, skills


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="intentdsl", description="IntentDSL runtime diagnostics and offline manual")
    parser.add_argument("--library", default=None, help="Explicit path to libintentdsl.so")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Probe installed library, external toolchains, and devices")
    p_doc.add_argument("--backend", default=None, help="Probe a single backend exit")

    p_set = sub.add_parser("setup", help="Verify packaged assets and print environment setup guidance")
    p_set.add_argument("--backend", default=None, help="Check setup for a single backend exit")

    sub.add_parser("layout", help="Print the compiled product layout and active configuration")
    sub.add_parser("skills", help="List or print manifest-verified product skills")

    p_man = sub.add_parser("manual", help="Read the packaged offline manual")
    p_man.add_argument("topic", nargs="?", default=None, help="Optional topic: install | language | model | components")
    p_man.add_argument("--lang", choices=["zh", "en"], default="zh", help="Manual language (default: zh)")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(backend=args.backend, library=args.library), indent=2, ensure_ascii=False))
    elif args.command == "setup":
        print(json.dumps(setup(backend=args.backend, library=args.library), indent=2, ensure_ascii=False))
    elif args.command == "layout":
        print(json.dumps(layout(library=args.library), indent=2, ensure_ascii=False))
    elif args.command == "skills":
        print(json.dumps(skills(library=args.library), indent=2, ensure_ascii=False))
    elif args.command == "manual":
        print(manual(args.topic, lang=args.lang))
    return 0


def compile_main() -> None:
    lib = library_path()
    candidates = [
        lib.parent.parent / "bin" / "intentdsl-compile",
        Path(sys.prefix) / "bin" / "intentdsl-compile",
    ]
    exe = next((c for c in candidates if c.is_file() and os.access(c, os.X_OK)), None)
    if exe is None:
        raise FileNotFoundError("bundled intentdsl-compile binary not found")
    env = dict(os.environ)
    ld = str(lib.parent)
    env["LD_LIBRARY_PATH"] = f"{ld}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ld
    os.execve(str(exe), [str(exe), *sys.argv[1:]], env)


if __name__ == "__main__":
    raise SystemExit(main())
