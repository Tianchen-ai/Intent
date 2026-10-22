"""Call the existing softmax algorithm from PyTorch using an installed Intent."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import intent
import torch

from intent.examples.kernels.normalization.softmax import stable_softmax_f16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "cutile", "mojo"), required=True)
    parser.add_argument("--device")
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
    device = args.device or ("cpu" if args.backend == "mojo" else "cuda:0")
    x = torch.randn((8192, 8192), device=device, dtype=torch.float16)
    if args.program is None:
        softmax = intent.jit(stable_softmax_f16, backend=args.backend)
    else:
        from intent.targets.local import resolve_local, select_device
        target = resolve_local(args.backend, select_device(args.backend, (x,), device))
        softmax = intent.GeneratedProgram.load(args.program).materialize(target=target)
    if args.torch_compile:
        softmax.prepare(x)
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
    artifacts = softmax.artifacts if args.program is None else (softmax,)
    for artifact in artifacts:
        print(f"Generated source and IR: {artifact.cache_directory}")
    if args.inspect_native:
        observed = artifacts[-1].observation
        print(json.dumps(asdict(observed) if observed is not None else None, indent=2))


if __name__ == "__main__":
    main()
