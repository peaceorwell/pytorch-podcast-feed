# Keeping Muse Spark Open Moving on TPU v7

原文：[Optimizing Muse Spark Open on TPU v7 with SGLang-TorchTPU - PyTorch](https://pytorch.org/blog/optimizing-muse-spark-open-on-tpu-v7-with-sglang-torchtpu/)

## 摘要

文章介绍如何用 SGLang-TorchTPU 在单台四芯片 TPU v7 主机上运行 816B 参数的 Muse Spark Open。核心是让一次解码或验证步骤的全部 62 层在同一个 Pallas 程序中执行，并将权重、KV 缓存和跨设备传输与计算重叠。在使用较早模型检查点、GSM8K 和 DFlash2 的测试中，单请求每输出 token 的耗时从原生路径的 3.78 ms 降至 1.95 ms。文章还介绍了长上下文注意力、KV 缓存和预填充优化，以及它们各自的测试条件与结果。

## 对话

Ava: Serving Muse Spark Open on four TPU v7 chips sounds like a memory challenge. But this article shows that pauses in the weight stream can also slow down each decode step.

Ava：在 4 块 TPU v7 芯片上运行 Muse Spark Open 推理，听起来首先是个内存容量难题。但这篇文章表明，权重流中的停顿也会拖慢每一步解码。

Brian: Their main fix is a megakernel. One Pallas program runs all 62 layers of a decode or verify step, while it overlaps weight loads, KV reads and transfers between devices with compute.

Brian：他们的主要改进是一个 megakernel。单个 Pallas 程序运行一次解码或验证步骤的全部 62 层，同时让权重加载、KV 读取和设备间传输与计算重叠进行。

Ava: The model makes that a serious scheduling job. Muse Spark Open has 816B parameters, about 42B active per token, and every layer routes each token to 8 of 256 experts.

Ava：这个模型让调度变得相当复杂。Muse Spark Open 有 816B 个参数，每个 token 激活约 42B 个参数，而且每一层都会将每个 token 路由到 256 个专家中的 8 个。

Brian: One TPU v7 host has four chips, exposed as eight devices. The model is split eight ways with tensor parallelism. Each device runs one program at a time, so overlapping transfers with compute inside that program matters.

Brian：一台 TPU v7 主机有 4 块芯片，对外呈现为 8 个设备。模型通过张量并行分成 8 份。每个设备同一时间只运行 1 个程序，因此在这个程序内部让传输与计算重叠很重要。

Ava: And SGLang-TorchTPU uses speculative decoding. How does that change the work done by the full model?

Ava：SGLang-TorchTPU 还使用推测解码。这会如何改变完整模型要做的工作？

Brian: The smaller DFlash2 model proposes a block of tokens at once. The full model checks the block in one verify step. It keeps the longest matching prefix of draft tokens, plus one token of its own.

Brian：较小的 DFlash2 模型一次提出一组 token。完整模型在一个验证步骤中检查这一组 token，保留与自身结果匹配的最长草稿 token 前缀，再加上自己生成的 1 个 token。

Ava: So the full model still has to move its weights. The older native path ran each layer through several kernels and collectives, with a boundary between them.

Ava：所以完整模型仍然得搬运权重。旧的原生路径让每一层经过多个 kernel 和集合通信操作，每次操作之间都有一道边界。

Brian: Right. Its verify step took 2.9 times the time needed just to stream the weights. In the first megakernel prototype, profiling at batch 1 found about 45 microseconds of collective cost per layer. Those interruptions were expensive.

Brian：没错。它的验证步骤耗时是单纯流式加载权重所需时间的 2.9 倍。在第一个 megakernel 原型中，batch 为 1 时的性能分析发现，每层的集合通信开销约为 45 微秒。这些中断代价很高。

Ava: The single program lets them plan further ahead. Dense weights move through a ring of tiles across layer boundaries. Once routing picks an expert, its weights can start loading.

Ava：单个程序让他们能提前规划更远的工作。稠密权重通过跨越层边界的 tile 环形缓冲区传输。一旦路由选定某个专家，就可以开始加载该专家的权重。

Brian: It also sends partial results directly to other devices with remote DMA, instead of launching a separate collective after each matmul. An o-proj tile can leave as soon as it's computed, while the next tile is still being multiplied.

Brian：它还通过远程 DMA 将部分结果直接发往其他设备，而不是每次矩阵乘法后再启动单独的集合通信。一个 o-proj tile 算完就能发出，此时下一个 tile 仍在进行乘法计算。

Ava: There's a separate KV-write kernel after the megakernel. It now overlaps page reads and writes across layers. At 16 rows, its cost fell from 810 to 155 microseconds per step, with bit-identical results.

Ava：megakernel 之后还有一个独立的 KV 写入 kernel。它现在让不同层的页面读取和写入重叠进行。在 16 行时，每一步的开销从 810 微秒降到 155 微秒，结果逐位一致。

Brian: The overall megakernel result is measured per output token. With DFlash2 on GSM8K, using an earlier Muse Spark checkpoint, one request went from 3.78 milliseconds on the native path to 1.95 milliseconds. That's 1.94 times faster.

Brian：整体 megakernel 效果按每个输出 token 的耗时衡量。在 GSM8K 上使用 DFlash2 和较早的 Muse Spark 检查点时，单个请求从原生路径的 3.78 毫秒降到 1.95 毫秒，速度提高到 1.94 倍。

Ava: DFlash2 has its own contribution too. On GSM8K with real draft acceptance, it reduced the megakernel's time per output token for one request from 3.93 to 1.95 milliseconds.

Ava：DFlash2 本身也有贡献。在 GSM8K 上按真实的草稿 token 接受情况测试时，它将 megakernel 处理单个请求的每个输出 token 耗时从 3.93 毫秒降到 1.95 毫秒。

Brian: Yes. A few verification rows fit in the same row tile as one token, so checking a block costs little more than checking one token on this TPU path.

Brian：对。几个验证行可以放进与 1 个 token 相同的行 tile，因此在这条 TPU 路径上，检查一组 token 比检查 1 个 token 多花不了多少时间。

Ava: Another delay is waiting for an all-gather before the next projection. Collective folding starts that projection as each slice arrives. Why can they move the normalization?

Ava：另一个延迟来自等待 all-gather 完成后才能开始下一次投影。集合通信折叠让投影随着各个切片到达就开始。为什么他们能移动归一化操作？

Brian: The relevant RMS norm factors are scalars per row. The projection can accumulate contributions from arriving slices, then apply those factors at the end. They fold two all-gathers per layer this way. Some rounding moves, so the results aren't bit-identical.

Brian：相关的 RMS 归一化因子对每一行都是标量。投影可以先累加已到达切片的贡献，最后再应用这些因子。他们以这种方式折叠了每层的 2 次 all-gather。部分舍入发生的位置变了，所以结果并非逐位一致。

Ava: Long-context attention has a different serial chain. Online softmax normally carries a running maximum from one KV chunk to the next. That prevents the next chunk's exponentials from starting independently.

Ava：长上下文注意力有另一条串行依赖链。在线 softmax 通常要把运行中的最大值从一个 KV 块传给下一个块。这使下一个块的指数运算无法独立开始。

Brian: For this model, QK-norm and a fixed query gain bound every attention score's magnitude at 43.8. That's below the overflow point for f32 exponentials. The kernel checks the bound when it's built, then drops the running maximum and its rescaling.

Brian：对这个模型而言，QK-norm 和固定的 query gain 将每个注意力分数的绝对值限制在 43.8 以内。这低于 f32 指数运算的溢出点。kernel 在构建时检查这个上界，然后去掉运行中的最大值及相应的重新缩放操作。

Ava: Now independent chunks can overlap their stages. The kernel scores later chunks while earlier chunks multiply probabilities by values. Global layers also use longer chunks than sliding-window layers.

Ava：这样，独立的块就可以让各处理阶段重叠运行。较早的块用概率乘以 value 时，kernel 已经在计算后续块的分数。全局注意力层使用的块也比滑动窗口层更长。

Brian: In a microbenchmark, longer chunks with independent streams cut attention time 2.4 times. Removing the running maximum brought the total reduction to 3.9 times. In the server, the pipelined loop raised per-request decode speed by 18% for two requests at 500K context.

Brian：在微基准测试中，更长的块配合独立流，使注意力耗时缩短了 2.4 倍。去掉运行中的最大值后，总耗时缩短了 3.9 倍。在服务器上，对于上下文长度为 500K 的 2 个请求，流水线循环使每个请求的解码速度提高了 18%。

Ava: The expert loop had a less obvious problem. A heavy eight-row branch made the compiler spill registers even when the common one- or two-row path ran.

Ava：专家循环还有一个不太明显的问题。处理 8 行的重分支让编译器把寄存器内容溢出到内存，即使实际运行的是常见的 1 行或 2 行路径也是如此。

Brian: They split experts into two loops: at most four rows, and more than four. The light loop carries 61 to 64% fewer spill instructions. Outputs stay bit-identical, and kernel tests with random routing found decode steps with many requests 13 to 19% faster.

Brian：他们把专家处理拆成 2 个循环：一个处理最多 4 行，另一个处理超过 4 行。轻量循环的溢出指令减少了 61% 到 64%。输出仍然逐位一致，而使用随机路由的 kernel 测试发现，在请求较多时，解码步骤快了 13% 到 19%。

Ava: Now for cache capacity. Simply changing the token-major KV cache from BF16 to FP8 didn't save space, because the layout padded the packed values.

Ava：接下来是缓存容量。仅仅把以 token 为主序的 KV 缓存从 BF16 改成 FP8 并没有节省空间，因为其布局对打包后的数值进行了填充。

Brian: The sequence-on-lane layout removes that padding. Per token, per layer, per device, KV storage falls from 512 bytes to 256 bytes with FP8. It also halves the KV bytes read by each step.

Brian：sequence-on-lane 布局消除了这种填充。采用 FP8 后，每个 token、每层、每个设备的 KV 存储量从 512 字节降到 256 字节。它还使每一步读取的 KV 字节数减半。

Ava: The draft model only attends to the last 2,048 tokens. Its KV had been allocated for the full context, so moving it into the sliding-window pool seems like a direct capacity win.

Ava：草稿模型只关注最近的 2,048 个 token。此前它的 KV 是按完整上下文分配的，所以把它移入滑动窗口池，似乎能直接增加容量。

Brian: It lets the same HBM hold 38% more tokens. They also enabled L2 HiCache on TPU, using host memory behind the HBM pools. An evicted prefix can come back from host memory on the next turn instead of being recomputed.

Brian：这样，同样大小的 HBM 可以容纳多 38% 的 token。他们还在 TPU 上启用了 L2 HiCache，将主机内存用作 HBM 池的后备存储。被驱逐的前缀可以在下一轮从主机内存取回，而不必重新计算。

Ava: On agent traffic, that L2 cache raised throughput by up to 77%. But a long pause while subagents work can still make an active conversation look cold to ordinary LRU eviction.

Ava：在智能体流量下，这个 L2 缓存让吞吐量最多提高了 77%。但子代理工作期间的一段长时间停顿，仍可能让普通的 LRU 驱逐策略把活跃对话误判为冷数据。

Brian: Session-aware eviction protects KV that a live conversation will reuse in host memory. HBM keeps ordinary LRU for running requests. On this trace, prefill had computed about 1.2 times the tokens required by the best possible cache. In the highest-throughput run, it now computes within 1 to 5% of that amount.

Brian：会话感知驱逐会保护仍在进行的对话将再次使用的 KV，使其保留在主机内存中。对于正在运行的请求，HBM 仍采用普通 LRU。在这条流量轨迹上，预填充原本计算的 token 数约为理想最优缓存所需计算量的 1.2 倍。如今在吞吐量最高的一次运行中，它的计算量只比那个数量高 1% 到 5%。

Ava: Prefill has its own bottleneck. With a 4,096-token chunk, the matrix units have plenty of rows, but the MoE layer still has to move partial results between devices.

Ava：预填充也有自己的瓶颈。使用 4,096 个 token 的块时，矩阵计算单元有足够多的行可处理，但 MoE 层仍须在设备之间传输部分结果。

Brian: Their fused MoE kernel uses a pair exchange between the two devices on each chip, reducing cross-chip traffic. The MoE kernel went from 3,247 to 2,368 microseconds per layer. With other changes, a 4,096-token prefill chunk on a cold 100K-token prompt fell from 440 to 384 milliseconds.

Brian：他们的融合 MoE kernel 让每块芯片上的 2 个设备成对交换数据，减少跨芯片流量。MoE kernel 的每层耗时从 3,247 微秒降到 2,368 微秒。加上其他改动，对于一个未缓存的 100K-token 提示词，4,096-token 预填充块的耗时从 440 毫秒降到 384 毫秒。

Ava: Sequence parallelism tackles repeated work between matmuls. Each device keeps only its own tokens between layers, so norms and residual updates no longer run over the whole chunk on every device.

Ava：序列并行解决的是矩阵乘法之间的重复工作。每个设备在层与层之间只保留属于自己的 token，因此归一化和残差更新不再需要在每台设备上处理整个块。

Brian: At 131K context, it cut a 4,096-token forward by 51 milliseconds, about 11%. It changes some reduction orders, so those results aren't bit-identical.

Brian：在 131K 上下文长度下，它使一次 4,096-token 前向计算缩短了 51 毫秒，约 11%。它改变了部分归约顺序，因此结果并非逐位一致。

Ava: A long cold prompt can also occupy many prefill chunks in a row. The scheduler now gives it at least 1,024 tokens of each batch and lets waiting turns use the rest.

Ava：一个很长且未缓存的提示词也可能连续占用许多个预填充块。调度器现在为它分配每个批次中至少 1,024 个 token 的处理量，并让等待中的后续轮次使用剩余容量。

Brian: In an AgentX test, that cut median end-to-end latency by 58% and p90 latency by 33%, with throughput within 5%.

Brian：在一次 AgentX 测试中，这使端到端延迟的中位数降低了 58%，p90 延迟降低了 33%，而吞吐量变化在 5% 以内。

Ava: AgentX matters here because it replays request lengths, shared prefixes and tool pauses from 393 opt-in coding sessions, using synthetic tokens. The reported run uses its 1M-context trace set.

Ava：AgentX 在这里很重要，因为它使用合成 token，重放来自 393 次自愿参与的编程会话中的请求长度、共享前缀和工具使用间歇。报告中的运行使用了它的 1M 上下文流量轨迹集。

Brian: Across that run's reported curve, the endpoints are 433 output tokens per second per request at 6,101 total tokens per second per chip, and 95 output tokens per second per request at 25,906 total tokens per second per chip. Total throughput counts input and output tokens. Every TPU point finishes requests within 47 seconds at p90.

Brian：在那次运行报告的性能曲线上，两个端点分别是每个请求每秒输出 433 个 token、每块芯片每秒处理总计 6,101 个 token，以及每个请求每秒输出 95 个 token、每块芯片每秒处理总计 25,906 个 token。总吞吐量计入输入和输出 token。所有 TPU 数据点的请求完成时间在 p90 上都不超过 47 秒。

Ava: So the useful pattern is clear: keep decode transfers moving, make long-context KV cheaper to hold and read, and share prefill batches so short turns can proceed.

Ava：所以实用的思路很明确：让解码时的传输持续进行，降低长上下文 KV 的存储和读取成本，并共享预填充批次，让短轮次也能向前推进。

Brian: And check each result against its workload and measured unit. That's our tour of Muse Spark Open on TPU v7. Thanks for listening.

Brian：还要结合各项结果对应的工作负载和计量单位来检验。这就是我们对 TPU v7 上的 Muse Spark Open 的介绍。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| megakernel | 将一次解码或验证步骤的全部 62 层放进同一个 Pallas 程序的内核。 |
| verify step | 完整模型一次检查草稿模型提出的一组 token 的步骤。 |
| speculative decoding | 先由较小模型提出候选 token，再由完整模型检查的解码方式。 |
| collective folding | 在其他设备的数据片段到达时就开始后续矩阵运算，减少等待完整 all-gather 的时间。 |
| online softmax | 逐块处理注意力分数、持续更新归一化统计量的方法。 |
| register spill | 寄存器容不下仍需使用的值时，编译器将其暂存到内存并在之后读回。 |
| sequence-on-lane layout | 把一个 KV 页的 token 沿向量寄存器的 lane 排列；文中以 FP8 存储时，每 token、每层、每设备的 KV 容量从 512 bytes 降至 256 bytes。 |
| L2 HiCache | 以主机内存作为 HBM 后方第二级缓存的分层 KV 缓存。 |
| session-aware eviction | 在主机内存中延后驱逐仍会被活跃会话复用的 KV 数据。 |
| sequence parallelism | 在层间按 token 分配激活值，减少每台设备重复执行的逐 token 工作。 |
| prefill | 生成输出 token 前，先处理请求中新输入的 prompt token 的阶段。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| as soon as it's computed | 一计算完成就…… |
| with bit-identical results | 同时保持逐位一致的结果。 |
| How does that change the work done by the full model? | 这会怎样改变完整模型所做的工作？ |
| The single program lets them plan further ahead. | 单个程序让他们能提前安排更远的工作。 |
| Prefill has its own bottleneck. | 预填充也有自己的瓶颈。 |
| That's our tour of Muse Spark Open on TPU v7. | 以上就是我们对 TPU v7 上 Muse Spark Open 的介绍。 |

## 中文内容总结

作者用 SGLang-TorchTPU，在一台配备四颗 TPU v7 芯片的主机上部署 Meta 的 Muse Spark Open。该模型共有 8160 亿参数，每个 token 约激活 420 亿参数。优化围绕解码时持续读取权重、扩大 KV 缓存容量，以及缩短长提示词的预填充时间展开。

### 负载与瓶颈

模型的 62 层均为混合专家层，每层从 256 个专家中选用 8 个；其中 16 层支持最长约 100 万 token 的全局注意力，其余 46 层使用 2048 token 的滑动窗口。四颗芯片对外呈现为八个设备，模型按八路张量并行切分。小批量解码主要受权重读取和跨设备通信限制，长上下文请求还会消耗大量 KV 缓存。

### 解码机制

核心改动是让一次解码或验证步骤的全部 62 层在每个设备上作为一个 Pallas 程序运行，并在计算期间预取权重、读取 KV、传输部分结果。集合通信折叠使下一次矩阵乘法能逐片处理已到达的数据。专家按所服务的行数分成两个循环，避免繁重分支让常用路径反复溢出寄存器。在单请求测试中，每个输出 token 的时间从原路径的 3.78 毫秒降至 1.95 毫秒；这项测试使用较早的模型检查点。

### 注意力与推测解码

模型的 QK 归一化和固定增益把注意力分数的绝对值限制在 43.8 以内，因此作者去掉在线 softmax 的运行最大值，使不同 KV 块能够流水处理。微基准测试中，相关注意力优化最终达到 3.9 倍加速；服务器上，两条请求在 50 万 token 上下文时的单请求解码速度提高 18%。DFlash2 草稿模型一次提出一组 token，主模型一次验证；GSM8K 实测中，单请求每个输出 token 的巨核时间从 3.93 毫秒降至 1.95 毫秒。

### 缓存与预填充

按序列排列在向量通道上的 fp8 KV 布局，把每层每设备的缓存开销从每 token 512 字节降至 256 字节。将草稿模型的 KV 放入滑动窗口池后，同等 HBM 可容纳的 token 数增加 38%。主机内存充当 HiCache 的二级缓存，在代理任务流量中使吞吐量最多提高 77%。会话感知淘汰优先保留活跃会话的前缀，使最高吞吐量运行中的预填充计算量仅比理想缓存多 1%至5%。预填充方面，融合 MoE 内核利用同芯片设备间的快速连接交换数据；配合其他改动，冷启动的 10 万 token 提示词中，一个 4096 token 块的处理时间从 440 毫秒降至 384 毫秒。序列并行在 13.1 万 token 上下文的测试中使同规格前向计算缩短约 11%；让等待中的请求共享预填充批次，则使 AgentX 测试的中位延迟下降 58%、p90 延迟下降 33%，吞吐量变化在 5% 以内。

### 结果与适用范围

在 AgentX 的 100 万上下文轨迹集上，作者报告的曲线两端分别为每请求每秒 433 个输出 token、每芯片每秒 6101 个总 token，以及每请求每秒 95 个输出 token、每芯片每秒 25906 个总 token。轨迹来自 393 个自愿提供的编码会话，以合成 token 替代原始内容；所有图中 TPU 测点的请求完成时间 p90 均在 47 秒内。TPU v7 测点使用单主机四芯片、FP8 计算及 NVFP4 专家权重；GB200 的 FP4 基线取自截至 2026-11-02 的 InferenceX 已发表结果。图中采用 InferenceX 的外部持有成本：每 TPU v7 芯片小时 1.21 美元，每 GB200 GPU 小时 1.86 美元。作者指出，高并发时预填充仍是瓶颈，后续计划改进注意力、MoE 布局和预填充，并探索预填充与解码分离部署。
