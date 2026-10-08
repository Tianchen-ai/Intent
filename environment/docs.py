#!/usr/bin/env python3
"""Install, build or preview the bilingual handbook without building Intent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _dependencies(root: Path) -> list[str]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError as error:
            raise RuntimeError(
                "Documentation setup on Python 3.10 requires tomli. "
                "Run 'python -m pip install tomli' or use Python 3.11+."
            ) from error
    with (root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["optional-dependencies"]["docs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "build", "serve"))
    parser.add_argument("--site-dir", type=Path, help="Static output outside the checkout")
    parser.add_argument("--address", default="127.0.0.1:8000", help="Preview bind address")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "install":
        if args.site_dir is not None:
            parser.error("--site-dir applies to build and serve")
        subprocess.run([sys.executable, "-m", "pip", "install", *_dependencies(root)], check=True)
        return
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    site = (args.site_dir or cache / "intentdsl" / "docs" / "site").expanduser().resolve()
    if site == root or root in site.parents:
        parser.error("documentation output must be outside the checkout")
    command = [sys.executable, "-m", "mkdocs", args.command, "--config-file", str(root / "mkdocs.yml")]
    if args.command == "build":
        command.append("--strict")
    else:
        command.extend(["--dev-addr", args.address])
    env = dict(os.environ, INTENT_DOCS_SITE_DIR=str(site))
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
