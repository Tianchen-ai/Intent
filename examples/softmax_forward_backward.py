"""Register an authored softmax backward with PyTorch autograd."""
import argparse

import intent
import torch

from kernels.backward.softmax import ROWS, COLUMNS, softmax_backward
from kernels.normalization.softmax import stable_softmax


def make_softmax(target):
    # Compile once; these artifacts belong to the host wrapper's lifetime.
    forward = intent.compile(stable_softmax, target=target)
    backward = intent.compile(softmax_backward, target=target)

    operation = forward.as_torch_op("intent_examples::softmax_forward")
    backward_operation = backward.as_torch_op("intent_examples::softmax_backward")

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    def backward_formula(ctx, upstream):
        probabilities, = ctx.saved_tensors
        return backward_operation(probabilities, upstream)

    operation.register_autograd(backward_formula, setup_context=setup_context)
    return operation, backward_operation, (forward, backward)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("triton", "cutile", "mojo"), default="triton")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Capture the registered forward and authored backward with PyTorch")
    args = parser.parse_args()
    target = {"triton": intent.TritonTarget, "cutile": intent.CuTileTarget,
              "mojo": intent.MojoTarget}[args.target]()
    operation, backward_operation, artifacts = make_softmax(target)
    forward, _ = artifacts
    device = torch.device("cuda", forward.device) if forward.device_type == "cuda" else torch.device("cpu")
    x = torch.randn((ROWS, COLUMNS), dtype=torch.float32, device=device, requires_grad=True)
    upstream = torch.randn_like(x)
    if args.torch_compile:
        # First-use provider compilation and tuning stay outside graph capture.
        with torch.no_grad():
            warmup_output = operation(x)
            backward_operation(warmup_output, upstream)

        def apply(value):
            return operation(value)

        probabilities = torch.compile(apply, fullgraph=True)(x)
    else:
        probabilities = operation(x)
    probabilities.backward(upstream)
    print(f"probabilities: {probabilities.shape}; input gradient: {x.grad.shape}")
    for artifact in artifacts:
        print(f"Compiler artifacts: {artifact.cache_directory}")


if __name__ == "__main__":
    main()
