"""The installed provider catalog can be read without importing a runtime."""
from importlib.resources import files
import json

COMPONENTS = json.loads(files(__package__).joinpath("components.json").read_text(encoding="utf-8"))
