# IntentDSL Handbook

IntentDSL combines a Python DSL for structured kernel algorithms with an MLIR compiler. Authors express logical members, computation, control, state and explicit orchestration of multiple kernels. The compiler constructs a physical program for the selected execution model and passes it to target toolchains such as Triton, cuTile, Mojo, Weft or BANG C.

## Start here

- [Installation](getting-started/installation.md): install from source or an existing wheel and select one backend environment.
- [Compile and run](getting-started/usage.md): run complete programs and inspect KIR, physical IR and generated source.
- [Configuration and tuning](getting-started/configuration.md): understand complete empirical configurations, first-call tuning and warm calls.
- [Agents and MCP](getting-started/mcp.md): connect the read-only language manual and explicit compilation tools.

## Find a language or compiler rule

Start writing kernels with the [authoring quick reference](dsl/authoring.md). Consult [language constructs](dsl/core.md) and [types, numerics and effects](dsl/types-numerics-and-effects.md) for the exact contract. Compiler contributors can start with the [compiler overview](compiler/README.md), [passes and analyses](compiler/passes-and-analyses.md), and [extending passes](development/passes.md).

This directory contains usage instructions and stable design. Implementation progress, test results, performance measurements and historical decision logs belong elsewhere. The specification has three layers:

- [Programming model](programming-model/README.md): semantic boundaries between authors, kernels, hosts and the compiler.
- [Language reference](dsl/README.md): the author surface, canonical semantics, shorthand and idealized examples.
- [Compiler](compiler/README.md): executable physical programs after canonical KIR, passes, target extensions and the boundary with lower compilers.

## Semantic authority

A compilation call selects the target outside the DSL. A target is not a source value, constexpr, type or control condition. The same source and canonical Kernel IR can be compiled for different targets. Target capabilities determine lowering legality without changing logical results, control, effects or interface semantics.

Compiler IR, passes, target lowering and runtime semantics build on the programming model and DSL. Existing implementations, historical reports and target APIs do not redefine the language.

Use the language switcher for 中文 or English. Chinese specifications retain their Markdown paths; English translations use matching `.en.md` files. [Build and publish documentation](development/documentation.md) covers local preview and GitHub Pages deployment.
