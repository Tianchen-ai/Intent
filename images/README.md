# Image sources / 图片来源

These are original figures from the IntentDSL paper. The PDFs are copied without modification; PNGs are rendered at 3,200 pixels on the longest side and cropped only to remove white margins. English and Chinese READMEs share the same figures and provide their own captions. No diagrams, labels, data, or performance values have been redrawn or changed.

| Public files | Original paper file | Meaning |
|---|---|---|
| `paper-fig1-overview.pdf` / `.png` | `paper/figv2/fig1.pdf` | Algorithm reuse and the compiler workflow. Target labels describe the paper implementation, including its RISC-V C path. |
| `paper-fig5-execution-models.pdf` / `.png` | `paper/figv2/fig5.pdf` | GPU program ownership, CPU tasks and local blocks, and DSA local storage/supply for the same logical work. |
| `paper-eval-pass-mechanisms.pdf` / `.png` | `paper/fig/eval-pass-mechanisms.pdf` | Single-mechanism ablations: causal pruning, partial accumulation, fused matrix update, and coordinated supply across five implementation paths. |

The performance figure belongs to the historical paper evaluation. As described in `paper/evaluation.tex`, it compares mechanism-disabled and mechanism-enabled latency with inputs, precision, tiles, and micro-kernels fixed. The five paths are H100 Triton, H100 cuTile, x86 Mojo, RISC-V intrinsic C, and MLU BANG C. Its RISC-V measurements do not describe the current Weft deployment. The figure does not measure the current Python distribution or the 30 product programs.

这些是 IntentDSL 论文原图，PDF 原文件直接保留，PNG 仅高分辨率转换并裁去白边。中英文 README 共用原图并使用各自图注，没有重画图形、标签或数值。总览图对应论文实现路径，包括当时的 RISC-V C 路径；执行模型图展示同一逻辑工作的三种物理组织。

性能图属于论文历史评测，定义来自 `paper/evaluation.tex`：固定输入、精度、tile 与 micro-kernel，比较单个机制关闭与开启的耗时。五条路径是 H100 Triton、H100 cuTile、x86 Mojo、RISC-V intrinsic C 与 MLU BANG C。RISC-V 成绩不等同于当前 Weft 部署，整张图也不是当前 Python 安装包或 30 个产品程序的新测量。

The project [Apache-2.0 license](../LICENSE) applies to these project-owned images.
