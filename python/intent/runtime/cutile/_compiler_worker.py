"""Private entry for a clean, compile-only cuTile interpreter."""

import pickle
import sys

from .compilation import _compile_configuration, _initialize_compiler


def main():
    request, response = sys.argv[1:]
    with open(request, "rb") as stream:
        module_name, filename, source, batch = pickle.load(stream)
    _initialize_compiler(module_name, filename, source)
    results = []
    for ordinal, *configuration in batch:
        result, error = _compile_configuration(*configuration)
        results.append((ordinal, result, error))
    with open(response, "wb") as stream:
        pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
