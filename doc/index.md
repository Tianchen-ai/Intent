# IntentDSL 工具书

[English handbook](index.en.md)

IntentDSL 2.1.0 使用 CompilerInfra 0.6.0。五出口同口径配对 336 项，246 快、85 平、5 慢。计分板列出各项的设备、参照及离散度。[里程碑实测](plan/v0.6-semantics.md)

1. [从源码到第一次测量](first-route.md) · 先让自己的 GPU kernel 通过 Torch 对照，再用 Bench 计时。
2. [作者语言与数值合同](manual.zh.md) · 视图、并行实例、归约与精度。
3. [公式速查](manual.zh.md) · 从逐元素到 Q6_K 收缩，对照公式、类型、DSL 原语和九个可运行示例。
4. [编译架构](architecture.md) · GPU、CPU、DSA 的所有权与供数。
5. [组件开发](manual.zh.md) · 在同一会话内试换并检查候选。

离线阅读用 `intentdsl manual formulas --lang zh`。网页搜索框可按函数名或原语检索。

## 本地搜索网站

从 Intent 仓根目录构建。生成的 `site/` 可交给静态 HTTP 服务器，搜索索引随页面一起生成。

```sh
python -m pip install 'mkdocs>=1.6,<2'
python -m mkdocs build
```
