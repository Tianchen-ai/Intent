"""Call the existing softmax algorithm from PyTorch using an installed Intent."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import intent
import torch

from kernels.normalization.softmax import stable_softmax_f16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("triton", "cutile", "mojo"), default="triton")
    parser.add_argument("--program", type=Path,
                        help="Load a saved generated program and bind it to the selected runtime")
    parser.add_argument("--inspect-native", action="store_true",
                        help="Show the last provider observation; unavailable fields remain explicit")
    invocation = parser.add_mutually_exclusive_group()
    invocation.add_argument("--prepared", action="store_true",
                            help="Bind an explicit output once, then launch the prepared call")
    invocation.add_argument("--torch-compile", action="store_true",
                            help="Call through the opaque PyTorch operator adapter")
    args = parser.parse_args()
    target = {"triton": intent.TritonTarget, "cutile": intent.CuTileTarget,
              "mojo": intent.MojoTarget}[args.target]()
    if args.program is None:
        softmax = intent.compile(stable_softmax_f16, target=target)
    else:
        softmax = intent.GeneratedProgram.load(args.program).materialize(target=target)
    device = torch.device("cuda", softmax.device) if softmax.device_type == "cuda" else torch.device("cpu")
    x = torch.randn((8192, 8192), device=device, dtype=torch.float16)
    if args.torch_compile:
        operation = softmax.as_torch_op("intent_examples::softmax")
        # Prepare the same real invocation before graph capture so first-use
        # provider compilation and tuning happen outside the captured graph.
        operation(x)

        def apply(value):
            return operation(value)

        y = torch.compile(apply, fullgraph=True)(x)
    elif args.prepared:
        output = torch.empty_like(x)
        call = softmax.prepare(x, outputs=(output,))
        call.launch()
        y = call.result()
    else:
        y = softmax.run(x)
    print(f"softmax: {tuple(y.shape)}, {y.dtype}, {y.device}")
    print(y[0, :8])
    print(f"Generated source and IR: {softmax.cache_directory}")
    if args.inspect_native:
        observed = softmax.observation
        print(json.dumps(asdict(observed) if observed is not None else None, indent=2))


if __name__ == "__main__":
    main()
