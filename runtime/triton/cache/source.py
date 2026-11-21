import hashlib
import importlib.util
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def load(source, entry):
    root = (
        Path(os.environ.get("INTENT_CACHE_DIR", Path.home() / ".cache" / "intentdsl"))
        / "triton-source"
    )
    root.mkdir(parents=True, exist_ok=True)
    name = "intent_kernel_" + hashlib.sha256(source.encode()).hexdigest()
    path = root / (name + ".py")
    if not path.exists():
        with NamedTemporaryFile("w", dir=root, delete=False) as f:
            f.write(source)
        os.replace(f.name, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entry)
