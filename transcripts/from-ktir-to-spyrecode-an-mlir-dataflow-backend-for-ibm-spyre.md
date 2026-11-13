# From KTIR to SpyreCode: Scheduling PyTorch Work for IBM Spyre

原文：[From KTIR to SpyreCode: An MLIR Dataflow Backend for IBM Spyre - PyTorch](https://pytorch.org/blog/from-ktir-to-spyrecode-an-mlir-dataflow-backend-for-ibm-spyre/)

## 摘要

本期解释 torch-spyre 如何把 Inductor 已确定的核心分工、数据布局和片上存放位置，转换成 Spyre 可执行的程序。KTIR 记录这些决策；开源 Dataflow Scheduler 根据硬件描述，把计算拆成程序，并逐步构建经过真实存储路径的流水线。随后，设备专用阶段生成带符号地址和形状的 SpyreCode，供运行时准备和启动。文章报告了算子开发时间的改善，也说明当前调度器主要依赖局部启发式规则，矩阵乘法支持和成本模型仍在推进。

## 对话

Ava: If Inductor has already decided which core owns each piece of work, what still stands between that decision and a running kernel? That's the question this Spyre backend answers, and it matters whenever data placement is part of performance.

Ava：如果 Inductor 已经决定了每一部分工作由哪个核心负责，从这个决定到内核真正运行之间还差什么？这个 Spyre 后端回答的就是这个问题；只要数据放在哪里会影响性能，这个问题就很重要。

Brian: Exactly. Inductor chooses the layout, tiling, core split, and scratchpad residency. KTIR records those choices. Then the Dataflow Scheduler builds a schedule the hardware can run, before device-specific stages produce SpyreCode for the runtime.

Brian：没错。Inductor 选择布局、分块方式、核心间的工作划分，以及哪些数据常驻暂存存储器。KTIR 记录这些选择。接着，Dataflow Scheduler 构建硬件可以执行的调度，再由设备专用阶段生成供运行时使用的 SpyreCode。

Ava: Let's make that concrete. The post shows an elementwise add over 96 × 64 fp16 tensors. It assigns 3 rows to each of 32 compute tiles. So the backend receives a work split; it doesn't need to invent one.

Ava：举个具体例子。文章展示了对 96 × 64 的 fp16 张量做逐元素加法，并把每个计算块的 3 行分配给 32 个计算块中的一个。因此，后端收到的是已经划分好的工作，不需要自己另做划分。

Brian: Right. Spyre has 32 cores on a ring, each with 2 MB of scratchpad. Device memory holds data off-chip, and transfers use sticks: fixed 128-byte chunks containing 64 fp16 values. KTIR presents a simpler view of those memories and cores.

Brian：对。Spyre 有 32 个呈环形连接的核心，每个核心都有 2 MB 暂存存储器。设备内存保存片外数据，传输使用 stick：每块固定为 128 字节，包含 64 个 fp16 值。KTIR 则以更简单的方式呈现这些存储器和核心。

Ava: That simplified view seems useful. A compute tile can refer to its own scratchpad, another core's scratchpad, or device memory. But KTIR doesn't yet say which load engine or FIFO will move the data. Is that the intended boundary?

Ava：这种简化视图看来很有用。一个计算块可以引用自己的暂存存储器、另一个核心的暂存存储器，或设备内存。但 KTIR 还不指定由哪个加载引擎或 FIFO 来搬运数据。这就是预期的职责边界吗？

Brian: Yes. KTIR describes what a tile computes and where its data lives. The scheduler decides how data reaches the compute engine. That distinction matters. Inductor's placement stays binding, while the backend still has room to build a legal pipeline.

Brian：是的。KTIR 描述一个计算块算什么，以及它的数据放在哪里。调度器决定数据怎样到达计算引擎。这个区别很重要：Inductor 确定的数据位置必须得到遵守，同时后端仍有空间构建合法的流水线。

Ava: The add example separates those concerns quite neatly. A memory view describes how to interpret storage. An access tile selects coordinates. Loads and stores move data. Then linalg does the addition. Changing the schedule shouldn't change which rows a tile owns.

Ava：加法示例把这些职责分得很清楚。内存视图描述如何解释存储的数据；访问块选定坐标；加载和存储操作搬运数据；然后由 linalg 完成加法。改变调度不应该改变一个计算块负责哪些行。

Brian: And access tiles can express more than rectangles. They can be strided, skewed, or triangular. KTIR also has indirect access tiles for cases such as paged KV-cache gathers. Those operations give the backend precise coordinates without prescribing every transfer step.

Brian：而且访问块能表达的不只是矩形区域，也可以是带步长、倾斜或三角形的区域。KTIR 还有间接访问块，可用于分页 KV-cache 的聚集读取等情况。这些操作为后端提供精确坐标，但不规定每一步如何传输。

Ava: I'm especially interested in tensors that stay on-chip across operations. If a producer leaves an intermediate in scratchpad, a consumer needs to know its exact location. Otherwise the compiler could accidentally turn a cheap handoff into a trip through device memory.

Ava：我尤其关心跨操作留在片上的张量。如果生产者把中间结果留在暂存存储器中，消费者就需要知道它的确切位置。否则，编译器可能无意间把一次低成本的交接变成往返设备内存的传输。

Brian: KTIR handles that with explicit memory spaces and distributed memory views. Several per-core scratchpad views can form one logical tensor in global coordinates. A consumer then loads the coordinates it needs, even when its access crosses from one core's scratchpad into another's.

Brian：KTIR 用显式内存空间和分布式内存视图处理这个问题。多个按核心划分的暂存存储器视图，可以在全局坐标中组成一个逻辑张量。这样，消费者就能加载所需坐标上的数据，即使一次访问会从一个核心的暂存存储器跨到另一个核心的暂存存储器。

Ava: So a different core split for the next operation becomes a question of which scratchpad locations its access tile covers. The scheduler can route those reads. That's a strong reason to keep physical placement visible in the IR.

Ava：所以，下一个操作即使采用不同的核心划分，问题也会转化为它的访问块覆盖了哪些暂存存储器位置。调度器可以为这些读取安排路径。这充分说明了为什么要在 IR 中保留物理位置的信息。

Brian: It is. Spyre doesn't have a hardware-managed cache to cover for an unspecified placement. Inductor's scratchpad allocator chooses which buffers remain resident, their addresses, and their cores. KTIR writes those choices down so later stages can respect them.

Brian：确实如此。Spyre 没有硬件管理的缓存来弥补未指定的数据位置。Inductor 的暂存存储器分配器会选择哪些缓冲区保持常驻，以及它们的地址和所属核心。KTIR 把这些选择记录下来，让后续阶段能够遵守。

Ava: What about computations that aren't named by ordinary arithmetic operations? A normalization or activation kernel might need a Spyre-specific instruction. I'd expect the scheduling interface to preserve that intent as well.

Ava：那普通算术操作无法直接表示的计算怎么办？归一化或激活内核可能需要 Spyre 专用指令。我想调度接口也得保留这种意图。

Brian: That's what the spyreop dialect does. It includes intrinsics such as gelu and operations used in normalization. They appear inside linalg.generic bodies, so they can fuse with surrounding elementwise work. Their final instruction sequences come from device patterns in the architecture description.

Brian：这正是 spyreop 方言的作用。它包含 gelu 等内在操作，以及归一化所用的操作。它们出现在 linalg.generic 的主体中，因此可以与周围的逐元素计算融合。最终的指令序列由架构描述中的设备模式生成。

Ava: Now we have a mapped kernel, with explicit data locations and computation. Where does the scheduler first cut it up? I assume a whole KTIR kernel isn't necessarily one configuration for the device.

Ava：现在我们有了一个已完成映射的内核，数据位置和计算也都明确了。调度器首先在哪里把它拆开？我想，整个 KTIR 内核未必只对应一种设备配置。

Brian: Correct. It traces backward from each store through its compute to the loads it needs. Each connected load, compute, and store group becomes a separately scheduled program. The original kernel becomes an orchestrator that keeps control flow and calls those programs in order.

Brian：对。调度器从每个存储操作出发，沿着计算过程反向追踪到它所需的加载操作。每组相互关联的加载、计算和存储操作，都成为一个单独调度的程序。原来的内核则成为编排程序，保留控制流，并依次调用这些程序。

Ava: There's a practical consequence. Loops supplied by Inductor for coarse tiling stay in that orchestrator and lead to program launches over tiles. Loops created inside a program are different: the scheduler can reshape those to match the hardware.

Ava：这有个实际影响。Inductor 为粗粒度分块提供的循环会留在编排程序里，并在各个数据块上触发程序启动。程序内部创建的循环则不同：调度器可以调整它们的结构，以适应硬件。

Brian: Yes. Within a program, elementwise operations are generalized and fused into one linalg.generic. The scheduler tiles that fused compute to the default unit's SIMD width, which it reads from the architecture description. That creates the first small pipeline.

Brian：是的。在一个程序内部，逐元素操作会被泛化并融合到一个 linalg.generic 中。调度器根据从架构描述中读取的默认单元 SIMD 宽度，对融合后的计算分块。这样就形成了最初的小型流水线。

Ava: The first version is load, compute, store, with FIFOs between stages. But if its endpoints imply device memory feeding compute directly, that isn't yet a legal hardware path. How does that change the pipeline?

Ava：最初的流水线是加载、计算、存储，各阶段之间用 FIFO 连接。但如果它的端点意味着设备内存直接向计算阶段供给数据，这还不是一条合法的硬件路径。流水线会怎样改变？

Brian: Path expansion searches the architecture's routing graph. It inserts the required hops and buffers. In the article's sample device, the 3 stages become 5: DMA into scratchpad, scratchpad load into a FIFO, compute, scratchpad store, and DMA back to device memory.

Brian：路径扩展会搜索架构的路由图，插入必要的传输步骤和缓冲区。在文章中的样例设备上，3 个阶段变成 5 个：DMA 将数据送入暂存存储器、从暂存存储器加载到 FIFO、计算、存回暂存存储器，以及通过 DMA 将数据送回设备内存。

Ava: And if Inductor placed an input in scratchpad already, that route starts there. The scheduler doesn't insert a device-memory transfer just because its initial pipeline was simple. Residency from the frontend has now changed the actual transfer stages.

Ava：如果 Inductor 已经把输入放在暂存存储器中，这条路径就从那里开始。调度器不会仅仅因为初始流水线很简单，就插入一次设备内存传输。前端决定的数据常驻位置，此时已经改变了实际的传输阶段。

Brian: Exactly. The same architecture description supplies memory sizes, datapaths, access granularities, FIFO depths, and compute widths. It also holds device-specific rewrite patterns. The scheduler runs against Spyre's description, while its tests use a separate sample device with the same passes.

Brian：正是如此。同一份架构描述还提供存储器容量、数据通路、访问粒度、FIFO 深度和计算宽度，并包含设备专用的重写模式。调度器依据 Spyre 的架构描述运行，而测试则在一台独立的样例设备上使用相同的编译过程。

Ava: Once transfers are legal, one vector at a time sounds inefficient for DMA. The scratchpad should hold a larger tile before compute drains it. Does the scheduler give those two parts different loop sizes?

Ava：传输路径合法之后，DMA 一次只处理一个向量听起来效率不高。暂存存储器应该先容纳一个更大的数据块，再由计算阶段逐步处理。调度器会给这两部分设置不同的循环尺寸吗？

Brian: It builds nested pipelines. An outer pipeline moves whole tiles between device memory and scratchpad. Inside its compute stage, another pipeline works at vector granularity through scratchpad, FIFO, and compute. Tile sizes begin as constrained placeholders and are chosen later.

Brian：它会构建嵌套流水线。外层流水线在设备内存和暂存存储器之间搬运整个数据块。在外层的计算阶段内部，另一条流水线以向量粒度，让数据经过暂存存储器、FIFO 和计算单元。数据块大小起初是受约束的占位值，稍后才会确定。

Ava: That also gives it a place to avoid repeated transfers. If an operand is broadcast and doesn't change along a loop, loading it once before the loop should be enough. What happens with reductions?

Ava：这样也有了避免重复传输的地方。如果一个操作数是广播值，沿某个循环不会改变，那么在循环前加载一次就够了。归约又怎么处理？

Brian: Invariant transfers move into a sibling pipeline before the loop. Large reductions are chunked to fit scratchpad, with an outer loop carrying the accumulator. The innermost, stick-width reduction is left for a device pattern to map onto hardware support.

Brian：不随循环变化的数据传输会移到循环前的一条同级流水线中。大型归约会被分块，以便放入暂存存储器，并由外层循环传递累加值。最内层、宽度为一个 stick 的归约，则留给设备模式映射到硬件支持的操作上。

Ava: So far, the stages have a useful shape. How do they overlap? If a DMA stage has to wait until compute finishes with the only scratchpad buffer, the nested structure alone won't hide that wait.

Ava：到这里，各阶段的结构已经很清楚了。它们怎样重叠执行？如果 DMA 阶段必须等计算阶段用完唯一的暂存存储器缓冲区，光有嵌套结构也无法隐藏这段等待时间。

Brian: The scheduler double-buffers scratchpad data shared by stages across loop iterations. One buffer can hold the tile being computed while the other receives the next tile. It also distributes suitable parallel loops across corelets, the sub-core instances inside each assigned core.

Brian：调度器会对跨循环迭代由多个阶段共享的暂存存储器数据使用双缓冲。一个缓冲区可以存放正在计算的数据块，另一个则接收下一个数据块。它还会把合适的并行循环分配到 corelet 上，也就是每个已分配核心内部的子核心实例。

Ava: That adds parallelism inside Inductor's existing core split. But are the buffer depth, corelet split, and tile size searched together? Their best values could depend on the workload.

Ava：这在 Inductor 已有的核心划分之内增加了并行度。但缓冲深度、corelet 划分和数据块大小是一起搜索的吗？它们的最佳取值可能取决于工作负载。

Brian: In version 1, local heuristics make those choices. There's no search or cost model yet. The choices remain explicit in the IR, though, including buffering factors and instance counts. That leaves a clear place for later performance work.

Brian：在版本 1 中，这些选择由局部启发式规则作出，目前还没有搜索机制或成本模型。不过，包括缓冲系数和实例数量在内的选择，仍会明确记录在 IR 中。这为以后的性能优化留出了清晰的位置。

Ava: We still need real addresses and particular engines. At what point does this schedule stop describing kinds of resources and start describing a program for each engine?

Ava：我们还需要实际地址和具体的引擎。这个调度在什么时候不再只描述资源类别，而开始描述每个引擎要运行的程序？

Brian: In the final scheduling steps. Buffers receive addresses checked against memory capacity. Unit kinds become specific units for each core or corelet, and dependencies become signals. DataflowIR then holds per-engine program units. Tensor addresses and runtime shapes can remain symbolic.

Brian：在调度的最后几个步骤。缓冲区会获得经过存储器容量校验的地址；单元类别会落实为各核心或 corelet 上的具体单元；依赖关系则会转化为信号。随后，DataflowIR 保存每个引擎的程序单元。张量地址和运行时形状仍可以保留为符号值。

Ava: That symbolic part is important for PyTorch tensors. Their addresses come from the allocator, and some shapes are known only at runtime within a bound. The compiler can still prepare a reusable artifact if it records how to calculate those values.

Ava：符号值对 PyTorch 张量很重要。它们的地址由分配器提供，有些形状也只有在运行时才能确定，但会落在给定界限内。只要编译器记录了这些值的计算方法，仍然可以预先准备一个可复用的产物。

Brian: The device-specific stages turn DataflowIR into SpyreCode. It contains a symbolic SpyreBinary, metadata for substituting symbols, and a job plan. The preparation part loads the binary once. On each call, the execution part resolves needed addresses and shapes, transfers those values, and launches.

Brian：设备专用阶段会把 DataflowIR 转换成 SpyreCode。它包含符号化的 SpyreBinary、用于替换符号的元数据，以及作业计划。准备阶段只加载一次二进制文件。每次调用时，执行阶段会确定所需的地址和形状，传输这些值，然后启动运行。

Ava: One last question. What has this delivered so far? I'd separate evidence that the compiler path works from evidence about runtime speed, because those tell an engineer different things.

Ava：最后一个问题：目前取得了哪些成果？我会把证明编译器流程可用的证据与运行速度方面的证据分开看，因为它们对工程师说明的是不同的事情。

Brian: Version 1 handles elementwise work with broadcast and reduction dimensions across multiple cores. The team validated the path through scheduling and code generation on representative transformer operations. They report that bringing up the first op was about 4x faster than on the previous stack, and later ops about 8x faster. Those figures describe developer time.

Brian：版本 1 支持跨多个核心处理带有广播维度和归约维度的逐元素计算。团队已用具有代表性的 Transformer 操作验证了从调度到代码生成的完整流程。他们报告说，与之前的技术栈相比，完成第一个算子的适配速度约快 4x，后续算子约快 8x。这些数字说的是开发耗时。

Ava: So the useful takeaways are the explicit placement contract, the step-by-step pipeline construction, and symbolic values that survive until launch. The open scheduler also has paths from both Inductor and Triton frontends. What's next for it?

Ava：所以值得关注的是明确的数据位置约定、逐步构建的流水线，以及一直保留到启动时的符号值。这个开放的调度器还能接收来自 Inductor 和 Triton 两种前端的输入。接下来会做什么？

Brian: The team is completing matrix multiplication support and adding cost-model-driven optimizations, aiming to run LLMs end to end. For now, KTIR says where work and data belong; the scheduler says how they flow; and SpyreCode lets the runtime prepare once and launch with each call's values. Thanks for listening.

Brian：团队正在完成矩阵乘法支持，并加入由成本模型驱动的优化，目标是让 LLM 端到端运行。目前，KTIR 规定工作和数据应放在哪里；调度器规定它们如何流转；SpyreCode 则让运行时准备一次，再使用每次调用的值启动运行。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| KTIR | Kernel Tile IR；记录计算分工、访存坐标和数据存放位置的 MLIR 表示。 |
| Dataflow Scheduler | 将 KTIR 转成分阶段数据流程序的开源调度器。 |
| ScheduleIR | 表示流水线、阶段、循环和资源分配等调度决策的中间表示。 |
| DataflowIR | 按执行引擎组织程序单元的中间表示，也是开源调度器与设备专用阶段的接口。 |
| SpyreCode | 包含符号化 SpyreBinary、符号替换元数据及任务计划的编译产物。 |
| scratchpad | 由编译器明确安排数据存放位置的片上存储器。 |
| distributed memory view | 把多个核心的局部存储视图组合成一个可按全局坐标访问的逻辑张量。 |
| access tile | 描述某个计算 tile 要访问哪些坐标、以及访问顺序的结构。 |
| path expansion | 依据硬件连接图，为数据传输补齐实际经过的执行单元和存储层级。 |
| double-buffering | 轮换使用两份缓冲区，让下一块数据的装载与当前块的计算重叠。 |
| corelet | 核心内部可承接并行工作的子核心实例。 |
| architecture description | 描述存储器、执行单元、连接、容量及设备专用改写规则的 MLIR 文件。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's make that concrete. | 用一个具体例子来说明。 |
| That distinction matters. | 这个区别很关键。 |
| There's a practical consequence. | 这会带来一个实际影响。 |
| How does that change the pipeline? | 这会怎样改变流水线？ |
| What happens with reductions? | 归约运算会怎样处理？ |
| At what point does this schedule stop describing kinds of resources and start describing a program for each engine? | 调度在哪一步从描述资源类型，转为描述每个执行引擎的程序？ |
| One last question. | 最后再问一个问题。 |

## 中文内容总结

文章介绍 torch-spyre 如何把 PyTorch Inductor 已确定的工作划分和片上存储位置，编译成 IBM Spyre 可执行的程序。KTIR 记录这些决定，开源 Dataflow Scheduler 构造执行流水线，设备专用阶段再生成供运行时使用的 SpyreCode。

### 编译接口与存储模型

Spyre 有 32 个环形互连的核心，每核配有 2 MB 片上 scratchpad；设备内存以 128 字节、容纳 64 个 fp16 值的 stick 为单位搬运数据。由于片上数据放置直接影响后续算子，KTIR 用计算分片、显式内存视图和访问分片记录 Inductor 选定的核心分工、数据布局及驻留位置。分布式内存视图可把多个核心 scratchpad 中的数据表示为一个逻辑张量，让后续操作按全局坐标读取，包括跨核心读取，而不必先写回设备内存。

### 数据流调度

调度器先沿每次存储操作的依赖链，把内核拆成独立程序；Inductor 原有的粗粒度循环留在负责依次调用程序的外层。每个程序从加载、计算、存储三级流水线起步，再依据设备架构描述中的存储器、执行单元、数据通路和容量，为传输寻找合法路径，插入 DMA、scratchpad 和 FIFO 阶段。随后它形成按数据块搬运的外层流水线与按向量计算的内层流水线，安排双缓冲、核内并行实例和分块大小。架构描述还提供设备指令的改写规则；V1 的这些调度选择采用局部启发式方法。

### 代码生成与运行

调度结果以 DataflowIR 表示各执行单元的程序，这是开源调度器与 Spyre 专用代码生成阶段的交界。后续阶段生成含符号值的 SpyreBinary、符号替换信息和两阶段任务计划，并打包为 SpyreCode。运行时先准备一次二进制；每次调用时，再根据实际张量地址和形状求出符号值并启动计算。因此，编译时尚未确定的地址和有上界的动态形状可以保留到启动时。

### 进展与适用范围

作者称，V1 已支持多核心 KTIR 程序中的逐元素运算、广播和归约，并在代表性的 Transformer 算子上完成从 KTIR 到代码生成的端到端验证；Inductor 和 Triton 前端均已有可用路径。作者报告，与旧软件栈相比，首个算子的开发速度约快 4 倍，后续算子约快 8 倍。这些数字衡量的是算子开发效率，文章没有给出运行速度提升数据。矩阵乘法支持和基于成本模型的优化仍在开发中；KTIR 与调度器已开源，调度器及其方言采用 Apache-2.0 许可。
