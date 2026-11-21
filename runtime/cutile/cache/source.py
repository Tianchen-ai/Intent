"""Load inspectable Python source from the product's immutable native cache."""

import hashlib
import importlib.util
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def load(source, entry):
    root = (
        Path(os.getenv("INTENT_CACHE_DIR", Path.home() / ".cache" / "intentdsl"))
        / "cutile"
    )
    root.mkdir(parents=True, exist_ok=True)
    name = "ct_" + hashlib.sha256(source.encode()).hexdigest()
    file = root / (name + ".py")
    if not file.exists():
        with NamedTemporaryFile(dir=root, mode="w", delete=False) as staged:
            staged.write(source)
        os.replace(staged.name, file)
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)
