# V1 论文结果 · V1 paper evaluation

These are the original V1 paper figures. Speedup is reference latency divided by generated latency for a complete operator, including required layout conversion. Compilation and tuning are excluded. The V2 measurements use their own inputs and comparators in the [V2 scorecard](https://github.com/Tianchen-ai/Intent/blob/main/doc/plan/v2-6.md).

### GPU · Triton reference

![Eight representative operators and the geometric mean across 44 common workloads on RTX 5090D and H100](images/paper-eval-gpu-triton-reference.png)

Both generated GPU providers use the same input precision and shape. GPU measurements use CUDA Graphs or the workload’s CUDA-event protocol. [Original PDF](images/paper-eval-gpu-triton-reference.pdf)

### CPU · Native programs

![Seven complete Mojo programs on eight physical x86 cores in one NUMA node](images/paper-eval-cpu-native.png)

Five references come from Modular/MAX and two from independent Mojo implementations. The seven bars are individual ratios. [Original PDF](images/paper-eval-cpu-native.pdf)

### MLU · Vendor libraries

![Six BANG C programs on MLU370 compared with CNNL, CNNL Extra and TMO](images/paper-eval-mlu-libraries.png)

CNRT notifiers time the device work. Paged GQA includes the reference’s Torch-MLU gather. [Original PDF](images/paper-eval-mlu-libraries.pdf)

The three plots and the execution-model illustration are reproduced from [V1 paper assets and provenance](https://github.com/Tianchen-ai/Intent/tree/v1/images).
