from __future__ import annotations

import argparse
import json
from pathlib import Path

from intent.targets.bangc import BangCTarget
from intent.compiler.artifact import GeneratedProgram
from intent.runtime.interface import ViewParameter
from . import DeviceBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run an exported Intent BANG C artifact")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("inputs", type=Path, help="JSON array of scalar values or {file, shape, dtype} buffer descriptions")
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--neuware", default="/usr/local/neuware")
    args = parser.parse_args()
    generated = GeneratedProgram.load(args.artifact)
    target = BangCTarget.from_program(generated, device=args.device, neuware=args.neuware)
    program = generated.materialize(target=target).runtime
    buffers = []
    try:
        arguments = []
        for description in json.loads(args.inputs.read_text()):
            if isinstance(description, dict):
                buffer = DeviceBuffer.from_host((args.inputs.parent / description["file"]).read_bytes(),
                    shape=tuple(description.get("storage_shape", description["shape"])),
                    dtype=description["dtype"], device=args.device, neuware=args.neuware)
                buffers.append(buffer)
                if "strides" in description:
                    arguments.append(buffer.view(tuple(description["shape"]), tuple(description["strides"]),
                                                 offset=description.get("offset", 0)))
                else:
                    arguments.append(buffer)
            else:
                arguments.append(description)
        call = program.prepare(tuple(arguments))
        buffers.extend(call.outputs)
        call.launch()
        args.outputs.mkdir(parents=True, exist_ok=True)
        results = []
        # InOut buffers remain caller-owned, but the file interface exports
        # their post-call contents alongside explicitly produced Out values.
        for parameter, value in zip(program.interface.parameters, call.arguments):
            if not isinstance(parameter, ViewParameter) or not parameter.writable:
                continue
            filename = f"output{len(results)}.bin"
            (args.outputs / filename).write_bytes(value.to_host())
            results.append({"name": parameter.name, "file": filename,
                            "shape": value.shape, "dtype": value.dtype})
        (args.outputs / "outputs.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"status": "executed", "outputs": str(args.outputs / "outputs.json")}))
    finally:
        program.close()
        for buffer in buffers:
            buffer.close()


if __name__ == "__main__":
    main()
