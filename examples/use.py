"""Compile or execute the thirty complete public programs through their host entry."""

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from importlib import import_module
import json
from pathlib import Path
from statistics import median
import sys
import time


# This is usage navigation. Names are never sent to the compiler as policy.
EXAMPLES = {
    "relu": ("elementwise", "Pointwise ReLU"),
    "swiglu": ("elementwise", "Two-input SwiGLU"),
    "rope": ("elementwise", "In-place Q/K rotary embedding"),
    "transpose": ("elementwise", "Matrix transpose"),
    "embedding": ("irregular", "BF16 embedding lookup with i64 indices"),
    "csr_spmv": ("irregular", "CSR sparse matrix-vector product"),
    "jagged_mean": ("irregular", "Segmented mean with explicit offsets"),
    "softmax": ("collectives", "FP16 softmax"),
    "layer_norm": ("collectives", "Weighted LayerNorm using the original second moment"),
    "batch_norm": ("collectives", "Welford statistics and mutable running state"),
    "group_norm_backward": ("collectives", "Two-kernel GroupNorm backward"),
    "cumsum": ("collectives", "Ordered row prefix"),
    "causal_linear_attention": ("collectives", "Region scan with matrix state"),
    "nonzero": ("irregular", "Prefix-based compaction and counts"),
    "moe_alignment": ("irregular", "Explicit count/prefix/scatter/mark"),
    "gemm": ("matrix", "FP16 GEMM with an algorithmic activation specialization"),
    "batched_gemm": ("matrix", "Batched BF16 GEMM"),
    "ragged_gemm": ("matrix", "Unequal groups and matrix supply"),
    "int8_gemm": ("matrix", "INT8 GEMM with i32 bias and accumulation"),
    "q4_projection": ("matrix", "Packed Q4_K weights and Q8_K activation quantization"),
    "fp8_gemm": ("matrix", "Block-scaled E4M3 with E8M0 scale storage"),
    "attention": ("streaming", "Causal BF16 grouped-query attention"),
    "paged_decode": ("streaming", "Explicit paged partials and FP32-to-FP16 merge"),
    "mamba_chunk_state": ("streaming", "BF16 matrix supply and FP32 chunk state"),
    "gated_delta": ("streaming", "Ordered gated recurrence and final state"),
    "causal_convolution": ("structured", "Causal depthwise convolution"),
    "cholesky": ("structured", "In-place small batched Cholesky"),
    "dropout": ("elementwise", "Author's integer-mixing dropout"),
    "adamw": ("elementwise", "Mutable parameters and optimizer moments"),
    "histogram": ("collectives", "Histogram with out-of-range sample masking"),
}


def _option(text):
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("target options require NAME=JSON")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _describe(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(getattr(value, "device", "host"))}
    if isinstance(value, dict):
        return {name: _describe(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_describe(item) for item in value]
    return value


def _measure(context, count, device):
    """Measure complete warm author calls, preserving supplied mutable inputs."""
    gpu = str(device).startswith("cuda")
    if gpu:
        import torch
    samples = []
    with torch.cuda.device(device) if gpu else nullcontext():
        if gpu:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # CUDA events allocate their native handles on first record. Keep
            # that setup outside the measured call and reuse completed events.
            start.record()
            end.record()
            end.synchronize()
        for _ in range(count):
            for invocation in context.invocations:
                invocation.reset_inputs()
            if gpu:
                torch.cuda.synchronize(device)
                start.record()
            else:
                started = time.perf_counter()
            for invocation in context.invocations:
                invocation.prepared.launch()
            if gpu:
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": median(samples), "samples_ms": samples,
            "scope": "Complete prepared author call after first-use compilation/tuning; "
                     "supplied InOut restoration and host readback excluded; "
                     "author workspace initialization included",
            "clock": "CUDA events" if gpu else "host wall clock"}


def _run(name, args, options):
    from intent.tools.backends import make_target
    from programs.common import ExampleContext
    from programs.assets import KernelAssets
    from programs.outputs import capture_outputs
    from programs import inputs

    started = time.perf_counter()
    context = None
    try:
        inputs.seed([args.seed, tuple(EXAMPLES).index(name)])
        target = make_target(args.target, options)
        device = f"cuda:{args.device}" if args.target in ("triton", "cutile") else "cpu"
        assets = None
        if args.assets or args.export_assets:
            root = args.assets if args.assets else args.export_assets
            assets = KernelAssets(Path(root) / name, mode="load" if args.assets else "export")
        context = ExampleContext(target, device, compiler=args.compiler,
                                 prepared=args.prepared, stage=args.stage, assets=assets)
        module, _ = EXAMPLES[name]
        result = getattr(import_module(f"programs.{module}"), name)(context)
        if args.stage == "run" and args.target in ("triton", "cutile"):
            import torch
            torch.cuda.synchronize(device)
        output_manifest = capture_outputs(result, Path(args.outputs) / name) if args.outputs else None
        measurement = _measure(context, args.measure, device) if args.measure else None
        native = []
        for artifact in context.artifacts:
            observation = getattr(artifact, "observation", None)
            if observation is not None:
                native.append({"provider": observation.provider, "stage": observation.stage,
                               "configuration": observation.configuration,
                               "resources": [asdict(item) for item in observation.resources],
                               "candidate_statuses": [item.status for item in observation.candidates]})
        return {"program": name, "target": args.target, "stage": args.stage,
                "status": {"source": "generated", "native": "compiled", "run": "executed"}[args.stage],
                "elapsed_seconds": time.perf_counter() - started,
                "input_seed": args.seed, "measurement": measurement,
                "native": native,
                "outputs": str(output_manifest) if output_manifest is not None else None,
                "result": _describe(result),
                "artifacts": [str(artifact.cache_directory) for artifact in context.artifacts]}
    except Exception as error:
        # A batch reports the actual failing program and exits unsuccessfully;
        # it does not retry with another target, dtype or algorithm.
        return {"program": name, "target": args.target, "stage": args.stage,
                "status": "failed", "failed_stage": getattr(error, "stage", args.stage),
                "elapsed_seconds": time.perf_counter() - started,
                "error": str(error),
                "cache_directory": str(error.cache_directory) if getattr(error, "cache_directory", None) else None}
    finally:
        if context is not None:
            context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?", choices=tuple(EXAMPLES))
    parser.add_argument("--list", action="store_true", help="List usage examples without importing a provider")
    parser.add_argument("--all", action="store_true", help="Use the existing thirty programs in their declaration order")
    parser.add_argument("--target", choices=("triton", "cutile", "mojo", "weft", "bangc"), default="triton")
    parser.add_argument("--target-option", action="append", type=_option, default=[], metavar="NAME=JSON")
    parser.add_argument("--stage", choices=("source", "native", "run"), default="run",
                        help="Generate source, compile bound native calls, or execute the complete program")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent preparation for source/native stages; execution remains sequential")
    parser.add_argument("--json", action="store_true", help="Emit complete program outcomes as JSON without a performance reference")
    parser.add_argument("--jsonl", action="store_true", help="Stream one JSON outcome as each program completes")
    parser.add_argument("--seed", type=int, default=0, help="Reproduce the same host inputs across targets (default: 0)")
    parser.add_argument("--measure", type=int, default=0, metavar="N",
                        help="Time N complete warm calls after initial execution, restoring supplied InOut outside timing")
    parser.add_argument("--device", type=int, default=0, help="CUDA device for a GPU target")
    parser.add_argument("--compiler", help="Use an explicitly selected intent-compile")
    parser.add_argument("--prepared", action="store_true", help="Bind each kernel invocation before launch")
    assets = parser.add_mutually_exclusive_group()
    assets.add_argument("--export-assets", metavar="DIR",
                        help="Save this host program's specialized kernel assets for deployment")
    assets.add_argument("--assets", metavar="DIR",
                        help="Load explicitly saved kernel assets instead of generating kernels")
    parser.add_argument("--outputs", metavar="DIR", help="Save all typed results from actual execution")
    args = parser.parse_args()
    if args.list:
        for name, (_, description) in EXAMPLES.items():
            print(f"{name:24} {description}")
        return
    if args.all and args.example is not None:
        parser.error("select one example or --all")
    if args.json and args.jsonl:
        parser.error("select --json or --jsonl")
    if args.example is None and not args.all:
        parser.error("select an example or use --list")
    if args.jobs <= 0:
        parser.error("jobs must be positive")
    if args.seed < 0 or args.measure < 0:
        parser.error("seed and measurement count must be nonnegative")
    if args.measure and args.stage != "run":
        parser.error("measurement requires actual execution with --stage run")
    if args.outputs and args.stage != "run":
        parser.error("saving output values requires actual execution with --stage run")
    if args.stage == "run" and args.jobs != 1:
        parser.error("run uses one program at a time; use concurrent source/native preparation separately")
    options = {}
    for name, value in args.target_option:
        if name in options:
            parser.error(f"duplicate target option: {name}")
        options[name] = value
    if args.target in ("triton", "cutile"):
        options.setdefault("device", args.device)
    names = tuple(EXAMPLES) if args.all else (args.example,)
    outcomes = {}
    def record(result):
        outcomes[result["program"]] = result
        if args.jsonl:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        elif args.all:
            print(f"{result['program']}: {result['status']} ({result['stage']})", file=sys.stderr, flush=True)
    if args.jobs == 1:
        for name in names:
            record(_run(name, args, options))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            pending = [executor.submit(_run, name, args, options) for name in names]
            for future in as_completed(pending):
                record(future.result())
    results = [outcomes[name] for name in names]
    if args.json:
        print(json.dumps(results if args.all else results[0], ensure_ascii=False, indent=2))
    elif not args.jsonl:
        for result in results:
            print(f"{result['program']}: {result['status']} ({result['stage']}, {result['elapsed_seconds']:.3f}s)")
            if result["status"] == "failed":
                print(f"  {result['failed_stage']}: {result['error']}", file=sys.stderr)
            else:
                print(f"  {result['result']}")
                if result["measurement"] is not None:
                    print(f"  Warm complete call: {result['measurement']['median_ms']:.6f} ms")
                for path in result["artifacts"]:
                    print(f"  Source, IR and diagnostics: {path}")
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
