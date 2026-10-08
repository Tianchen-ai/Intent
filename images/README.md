# Image sources / 图片来源

These diagrams belong to the IntentDSL project. They are redrawn as scalable, bilingual illustrations for the product guide, using the paper's compiler and region-computation diagrams as the source of their structure. They illustrate responsibilities and execution organization, not measured performance or identical backend support.

- **`overview.svg` / `overview.zh-CN.svg`**: a new product overview based on `paper/figv2/fig1.pdf`. It shows the author's algorithm, typed MLIR and shared/execution-model passes, provider lowering, and artifact use. Provider labels follow the current implementation: Triton, cuTile, Mojo, Weft, and BANG C.
- **`execution-models.svg` / `execution-models.zh-CN.svg`**: a simplified illustration based on `paper/figv2/fig4.pdf` and `paper/figv2/fig5.pdf`. It keeps the author-defined region computation and the distinction between GPU query-tile ownership, a CPU task hierarchy, and DSA local supply. Low-level equations and notation are omitted to keep the README readable.

以上配图依据 IntentDSL 论文重新绘制，提供中英文两套 SVG，按产品手册的阅读尺度简化排版。总览图依据 `paper/figv2/fig1.pdf`，展示作者程序、typed MLIR、shared 与执行模型 passes、后端及产物使用；执行模型图依据 `paper/figv2/fig4.pdf` 和 `fig5.pdf`，保留 region 算法和三种物理组织，省略细粒度方程与记号。配图不表示性能成绩，也不表示所有后端具有相同支持范围。

The project [Apache-2.0 license](../LICENSE) applies to these project-owned images.
