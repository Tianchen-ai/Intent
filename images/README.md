# Image sources / 图片来源

These are original figures from the IntentDSL paper. The PDFs are copied without modification; PNGs are rendered at 3,200 pixels on the longest side and cropped only to remove white margins. English and Chinese READMEs share the same figures and provide their own captions. No diagrams, labels, data, or performance values have been redrawn or changed.

| Public files | Original paper file | Meaning |
|---|---|---|
| `paper-fig1-overview.pdf` / `.png` | `paper/figv2/fig1.pdf` | Algorithm reuse and the compiler workflow. Target labels describe the paper implementation, including its RISC-V C path. |
| `paper-fig5-execution-models.pdf` / `.png` | `paper/figv2/fig5.pdf` | GPU program ownership, CPU tasks and local blocks, and DSA local storage/supply for the same logical work. |
| `paper-eval-gpu-triton-reference.pdf` / `.png` | `paper/fig/eval-gpu-triton-reference.pdf` | Complete GPU operators: eight representatives and all 44 common workloads' geometric mean, with both generated providers on RTX 5090D and H100. |
| `paper-eval-cpu-native.pdf` / `.png` | `paper/fig/eval-cpu-mlu-native.pdf` | Seven complete x86 Mojo programs compared with five Modular/MAX and two independent Mojo references. |
| `paper-eval-mlu-libraries.pdf` / `.png` | `paper/fig/eval-cpu-mlu-mlu.pdf` | Six complete MLU BANG C programs compared with CNNL, CNNL Extra, and TMO implementations. |

The performance figures belong to the historical paper evaluation in `paper/evaluation.tex`. Speedup is reference/generated median operator latency: preparation and required layout materialization are included; compilation, JIT, and tuning are excluded. GPU timing uses CUDA Graphs or the workload's CUDA-event protocol; MLU uses CNRT notifiers. Inputs, outputs, and tolerance follow each workload. For a GPU workload, both generated providers use the same DSL algorithm, input scale, and external dtype; representative shapes/precisions appear in the figure. The GPU geometric mean covers all 44 common workloads. The CPU and MLU figures show individual ratios, without aggregation.

The CPU hardware is eight physical x86 cores within one NUMA node; its comparisons use Mojo. MLU uses MLU370 with CNCC/CNRT; ReLU, softmax, GEMM, and state passing compare with CNNL, causal attention with CNNL Extra, and paged GQA with TMO plus Torch-MLU gathering. The paper's MLU GEMM has M=4096, N=14336, K=4096. The figures do not measure the current Python distribution or the 30 product programs, and do not establish current Weft performance.

这些是 IntentDSL 论文原图，PDF 原文件直接保留，PNG 仅高分辨率转换并裁去白边。中英文 README 共用原图并使用各自图注，没有重画图形、标签或数值。总览图对应论文实现路径，包括当时的 RISC-V C 路径；执行模型图展示同一逻辑工作的三种物理组织。

性能图属于 `paper/evaluation.tex` 中的论文历史评测。比值为 reference/generated 的完整算子耗时中位数，包含准备和必要布局转换，排除编译、JIT 与调优；GPU 使用 CUDA Graphs 或 workload 的 CUDA-event 计时，MLU 使用 CNRT notifier。输入、输出和容差沿用各 workload 设置。同一 GPU workload 的两个 generated provider 使用同一 DSL 算法、输入规模与外部 dtype，代表性 shape/精度在图中标注；GPU 几何平均覆盖全部 44 个共同 workload，CPU 与 MLU 小图展示独立比值、不做平均。

CPU 图使用同一 NUMA 节点内 8 个物理 x86 核，展示 Mojo 路径；MLU 图使用 MLU370/CNCC/CNRT，ReLU、softmax、GEMM、state passing 对比 CNNL，causal attention 对比 CNNL Extra，paged GQA 对比 TMO 加 Torch-MLU gather；其中 GEMM 的 M=4096、N=14336、K=4096。以上图不是当前 Python 分发包或 30 个产品程序的新测量，也不表示当前 Weft 性能。

The project [Apache-2.0 license](../LICENSE) applies to these project-owned images.
