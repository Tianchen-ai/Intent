# IntentDSL User and Component Developer Manual (English)

## 1. Installation and Environment Setup (`install`)

### Installing from Wheel (No LLVM Dev SDK Required)

Release wheels statically link the required MLIR/LLVM libraries into `libintentdsl.so`; no runtime LLVM/MLIR SDK is needed on user machines:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install compilerinfra==0.4.1 intentdsl==2.0.0
# Optional CLI / MCP compiler development host tools:
python -m pip install compilerinfra-agent==0.4.1
```

Supported native wheel baselines:
- **x86_64**: `manylinux_2_31_x86_64` (`glibc >= 2.31`, covering Ubuntu 20.04 MLU hosts and newer x86_64 Linux) and native `linux_x86_64`.
- **riscv64**: `linux_riscv64` (`glibc >= 2.38`, verified natively on SG2044 and RVV V100 RVV 1.0 systems).

Verify the bundled library, toolchains, and visible hardware offline:

```sh
intentdsl doctor
intentdsl setup
intentdsl manual --lang en
```

### External Backend Dependencies and Process Isolation

Vendor compilers and device runtimes remain external user dependencies and are probed on demand:

| Backend (`backend`) | Execution Device | External Toolchains & Runtime Libraries |
|---|---|---|
| `triton` | NVIDIA GPU | PyTorch (CUDA), GPU `triton`, CUDA driver |
| `cutile` | NVIDIA GPU (`SM90`/`SM120`) | PyTorch (CUDA), `cuda-tile` (`cuda.tile`), `nvcc`, CUDA driver |
| `mojo` | x86_64 CPU (`AVX2`/`AVX-512`/`AMX`) | `mojo` compiler (or `INTENT_MOJO`), `gcc` + `libgomp.so`; optional `oneDNN >=3.13,<4` (`ONEDNN_ROOT` or `INTENT_ONEDNN_LIB`) |
| `weft` | x86_64 / riscv64 CPU | `triton-weft` + `weft_core`, C/C++ toolchain, homogeneous CPU affinity set (`taskset`) |
| `bangc` | Cambricon MLU (`MLU370`) | `cncc` (`mtp_372`), `libcnrt.so`, `torch_mlu`, Cambricon driver (`/dev/cambricon_dev*`); `libcnnl.so` recorded as external reference dependency |

**Triton Distribution Note**: GPU `triton` and CPU `triton-weft` both provide the top-level `triton` Python package and therefore cannot be imported in the same Python process; install GPU `triton`/`cutile` and CPU `weft` in separate virtual environments (`mojo` coexists in-process with either GPU `triton`/`cutile` or CPU `weft`; `bangc` runs independently on MLU hosts).

### Rebuilding a Wheel from `sdist` in a Clean Environment

Build prerequisites:
- `CompilerInfra == 0.4.1` native SDK and Python package
- `LLVM / MLIR == 20.1.8` (static archives built with `-fPIC`)
- `CMake >= 3.22`, `Ninja`, C++17 compiler (`GCC >= 11` or `Clang >= 16`), `Python >= 3.10`

```sh
unset PYTHONPATH INTENTDSL_LIBRARY
python -m venv /tmp/intentdsl-build-env
/tmp/intentdsl-build-env/bin/pip install --upgrade pip build
/tmp/intentdsl-build-env/bin/pip install compilerinfra-0.4.1-*.whl
CMAKE_PREFIX_PATH="/path/to/compilerinfra-0.4.1:/path/to/llvm-20" \
  /tmp/intentdsl-build-env/bin/pip wheel --no-build-isolation --no-deps \
  intentdsl-2.0.0.tar.gz -w /tmp/dist
```

---

## 2. Author Language and Numerical Contracts (`language`)

IntentDSL parses Python ASTs into typed `kir` IR; kernel bodies are never executed by the Python interpreter.

- **Directions and Symbolic Views**: `I.In[dtype, shape]`, `I.Out[dtype, shape]`, `I.InOut[dtype, shape]` with symbolic dimensions (`("M", "N")`) and typed scalars (`I.f32`, `I.i32`, `I.u64`).
- **Domains and Collectives**: `I.parallel(...)` declares parallel instances; `I.domain(0, extent)` declares reduction/streaming member axes; `I.static_range(K)` declares compile-time unrolled stripes; collectives include `I.reduce`, `I.sum`, `I.welford`, `I.cumsum`, `I.reduce_by_key`, `I.ordered_sum`, contractions, and ordered recurrences.
- **Strict Numerical Contracts**: IEEE-754 `f32` accumulation, NaN propagation, and round-to-nearest-even are preserved by default (including 0-ULP certified `div`/`sqrt` on MLU and exact BF16 rounding). Fast hardware paths that flush subnormals (such as AMX BF16) are admitted only when explicitly licensed by the author (e.g. `allow_subnormal_ftz=True`).

```python
import intentdsl as I

@I.kernel
def rms_norm(
    x: I.In[I.f32, ("M", "N")],
    gain: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
    inv_c: I.f32,
    eps: I.f32,
):
    for r in I.parallel(x.dim(0)):
        u = x[r, :]
        s = I.sum(u * u, axis=0, acc=I.f32)
        out[r, :] = (u * I.rsqrt(s * inv_c + eps)) * gain[:]
```

---

## 3. Model Integration and Runtime Execution (`model`)

- **JIT Compilation, Warm Execution, and Buffer Swapping**:
  `kernel = I.jit(rms_norm, backend="bangc")` (or `"triton"` / `"cutile"` / `"mojo"` / `"weft"`).
  - `y = kernel.run(x, gain, 1.0 / N, 1e-6)` allocates outputs, compiles on first call, and executes.
  - `call = kernel.prepare(x, gain, 1.0 / N, 1e-6, outputs=y)` separates preparation from warm replay (`call.execute()`, or `call.launch()` returning a `Completion` handle). Re-binding new buffers via `kernel.prepare(x2, gain, 1.0 / N, 1e-6, outputs=y2)` reuses the compiled native binary whenever shapes satisfy the artifact `guard`.
- **Static `@I.program` Sequences**: Captures `I.workspace`, `I.output`, `I.fill`, and multi-kernel topology without host Python overhead during replay (see 30 complete workloads in `intentdsl.examples`).

---

## 4. Component Guide, Candidate Derivation, and Agent Integration (`components`)

### Extending an Existing Node or Exit

1. **Locate Registered Units**: Run `layout` / `inspect` to inspect the `kir -> {block -> {triton, cutile, dsa -> bangc}, cpu -> {mojo, weft}}` tree, boundary contracts, and exact C++ source locations.
2. **Adding a Node Strategy**: Implement candidate enumeration and `verify` in the owning unit (e.g., `levels/storage/dsa/decisions/Supply.cpp` or `levels/execution/block/decisions/strategies/`), update the unit's `cost/` model, and enable the strategy in `cmake/default.json`.
3. **Adding an Exit Microkernel**: Place kernel source under `runtime/<exit>/kernels/`, register its `ImplementationContract` in `exits/<exit>/ir/` (operands, memory space, accumulator, numerical bound), lower to a terminal op in `legalize/`, and print op-by-op via `TerminalPrinter`.

### Copy-Pasteable CLI and MCP Workflow Commands

Given product checkout `$REPO`, installed library `$LIB` (`python -c "import intentdsl; print(intentdsl.library_path())"`), and out-of-tree scratch directory `/tmp/ci-dev`:

```sh
# 1. Inspect compiled layout, frontend bindings, and manifest-verified skills
compilerinfra-dev --library "$LIB" --workspace "$REPO" --output-dir /tmp/ci-dev inspect
compilerinfra-dev --library "$LIB" --workspace "$REPO" --output-dir /tmp/ci-dev \
  read-skill '{"name":"agent/skills/intentdsl-development.md"}'

# 2. Derive a candidate, inspect doctor probes, compile/explain decisions, and measure in a persistent session
compilerinfra-dev --library "$LIB" --workspace "$REPO" --output-dir /tmp/ci-dev << 'EOF'
{"action":"derive","arguments":{"configuration":{"exits":{"bangc":{"options":{"isa":372,"nram_bytes":786432,"wram_bytes":524288,"clusters":4,"cores":4,"calibration":"mlu370"}}}}}}
{"action":"doctor","arguments":{"compiler":"base"}}
EOF

# Using the returned handle <cid>, compile, explain decisions, and run external Bench measurement:
# {"action":"compile","arguments":{"compiler":"<cid>","input":"softmax.py","exit":"triton","author_input":true}}
# {"action":"explain","arguments":{"compilation_id":"<compilation_id>"}}
# {"action":"measure","arguments":{"compiler":"<cid>","output_dir":"/tmp/bench-out","workload":{"name":"program/softmax","cases":[],"implementations":["intent-v2:triton"]}}}

# 3. Generate Codex and Claude stdio MCP configurations exposing the exact same action catalog
compilerinfra-dev host-config -- \
  '{"host":"codex","library":"'"$LIB"'","workspace":"'"$REPO"'","output_dir":"/tmp/ci-dev"}'
compilerinfra-dev host-config -- \
  '{"host":"claude","library":"'"$LIB"'","workspace":"'"$REPO"'","output_dir":"/tmp/ci-dev"}'
```
