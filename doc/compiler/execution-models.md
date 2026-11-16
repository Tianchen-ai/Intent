# 执行模型与源码组件

## 身份与降低责任

设备身份、执行模型与编译provider是三个独立维度。设备描述实际CPU或加速器；执行模型定义当前IR的执行含义；provider决定在哪个阶段接管，并完成原生编译和调用。GPU、CPU、DSA目录归属不决定唯一降低深度。

Canonical KIR保存作者算法。Block保存program-instance空间、ownership、shaped SSA、访问坐标、控制、累加与依赖。CPU保存任务、captures、阶段依赖、join及任务内计算。DSA显式执行层保存参与者、存储、传输与完成关系。每次降低转换当前程序，不能留下两份需要联合解释才能执行的计划。

Triton和cuTile消费GPU映射后的Block。Triton–Ascend从Block交给下层NPU编译器，不经过BANG C存储实现。BANG C消费显式DSA存储与引擎实现。Mojo可以消费较低层CPU循环与向量结构；Triton–Weft通过Canonical Weft `.weftir`进入其正式JIT。共同模型不要求各provider停在同一阶段。

## CPU执行不变量

任务不对应固定线程或hart，块大小不等于SIMD宽度。任务参数显式捕获输入，输出具有唯一owner或明确原子规则；依赖阶段完成后才能启动消费者，join定义完成边界。动态shape、stride、共享准备与ordered control保留其原语义。

SSA值与必须存储的状态分别处理。目标实现声明输入表示、packing、局部块与scratch需求，由物化pass建立实际存储和供数。private/shared存储具有owner、初始化与生命周期。固定SIMD、RVV和矩阵扩展通过能力及实现接口参与合法性判断，不以固定向量宽度或全局矩阵布尔值替代实现合同。

硬件能力、CPU affinity、线程预算和工作集预算分开。一次调用必须有明确调度所有者；不能同时由Intent与provider线程池重复派发。CPU调用完成后返回，任务内的异步实现不能改变此边界。

## DSA执行不变量

计算层保留Block值、控制和访问，不把所有值提前放入片上buffer。显式执行层只表达确实由Intent决定的参与者、执行组、存储资源、容量、对齐、可见性、生命周期，以及搬运、计算、等待和依赖。

同步由依赖与作用域定义，不能以某家barrier的拼写定义公共同步。异步传输必须有明确完成点，消费者不得提前访问，存储也不能在完成前重用。共享访问的参与者和可见范围由当前IR说明。

NRAM、WRAM、SRAM、memory core、Union和固定组规模属于BANG C目标组件；Cube、Vector及昇腾流水线属于Triton–Ascend或其目标扩展。硬件容量、SDK保留量、配置使用预算分别表达，未知能力不能以假数值填补。具体存储和微程序的限制不能变成DSL类型或算法限制。

## 组件合同

源码组件登记设备识别、本机探测、specialization、编译选项、输入执行模型、pipeline、目标dialect/pass、配置资源、原生产物、绑定与launch。声明查询不导入SDK，不探测设备。正常扩展通过登记及重新构建完成，不要求稳定独立二进制插件ABI。

公共驱动只执行阶段接口，不按provider名称重建资源或ABI。后端之间的发行包冲突由安装环境及诊断处理，不能修改`sys.path`、全局替换driver或偷偷转入子进程。

Pass使用MLIR的pass、analysis、interface与pipeline。必需lowering和可选优化分开；关闭可选优化仍必须能够形成合法程序。组件声明输入阶段、适用条件、完整改写和后置条件。坐标、effects、依赖及生命周期分析可以共享，具体改写可以目标专属。候选配置是有限且完整的经验集合，不构造无界组合。

## 本机JIT与扩充

`intent.jit(definition, backend=...)`显式选择provider，实际输入确定本机设备。目标探测产生硬件事实，compiler决定物理结构，runtime绑定资源并选择已声明候选。`prepare`不执行、不调优；`compile`完成原生编译；`launch`执行；`result`返回输出容器。产物缓存与buffer绑定、调优选择分离。热调用不重复设备探测或源码生成。

作者控制算法、多kernel顺序和workspace。GPU/NPU保留对应stream语义。Bench只负责执行环境、输入、正确性、对照与计时，不补足产品编译和绑定能力。

Agent通常在所属组件增加能力、lowering或优化，再查看当前IR和真实运行结果。核心错误须修复，但不能为单个后端悄悄改变共同语义。运行产品不依赖在线Agent修补生成代码。
