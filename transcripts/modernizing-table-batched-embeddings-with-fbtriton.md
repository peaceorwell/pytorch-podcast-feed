# FBTriton: Making Embedding Kernels Faster and Easier to Change

原文：[Modernizing Table Batched Embeddings with FBTriton](https://pytorch.org/blog/modernizing-table-batched-embeddings-with-fbtriton/)

## 摘要

本期介绍 FBTriton 如何用 Triton 重写推荐系统中的 Table-Batched Embedding 前向与反向计算，并通过调整任务划分和访存方式提升性能。前向包含通用 gather 路径与适用条件严格的小表直方图路径；反向则按同一行被访问的次数分配工作，汇总梯度后只执行一次优化器更新。在 GB200 的 307 个分片配置中，前向加速比中位数为 1.28 倍，反向的主要优势来自中等长度访问段的内存吞吐提升，但极短访问段仍落后于 CUDA。对话也区分了默认路径、可选优化与未来融合方向，说明更易维护的 Python 实现为何同样重要。

## 对话

Ava: What if your recommendation system's embedding kernels could run faster and become easier to change?

Ava：如果推荐系统的嵌入内核运行得更快，也更容易修改，会怎样？

That's the promise we're exploring today.

这正是我们今天要探讨的可能性。

I'm Ava, and this is a story about moving data efficiently.

我是 Ava，今天要讲一个高效搬运数据的故事。

Brian: And I'm Brian.

Brian：我是 Brian。

The article reports a median forward speedup of one point two eight times over the legacy CUDA implementation.

文章称，与旧版 CUDA 实现相比，前向计算的加速比中位数为 1.28 倍。

It also says the new sparse path is smaller than the original CUDA templates alone.

文章还说，新的稀疏路径代码量比原有的 CUDA 模板本身还少。

Ava: Faster kernels and less code to maintain. That's a combination I'd take.

Ava：内核更快，要维护的代码更少。这组合我很乐意接受。

What's the operation they're rewriting?

他们在重写什么操作？

Brian: TBE, or Table-Batched Embedding.

Brian：TBE，也就是 Table-Batched Embedding（表批量嵌入）。

It looks up rows from many embedding tables and pools them in one graphics processing unit, or GPU, launch.

它从多个嵌入表中查找行，并通过一次图形处理器（GPU）启动完成池化。

Combining that work reduces launch overhead and improves memory efficiency.

合并这些操作能减少启动开销，提高内存效率。

Ava: Let's unpack pooling. I've got a list of item identifiers.

Ava：我们来讲讲池化。我手里有一列物品标识符。

Each identifier selects a row of numbers. Then I add those rows together?

每个标识符选出一行数字，然后把这些行相加？

Brian: Exactly, for the sum described here. That list is an embedding bag.

Brian：没错，这里说的就是求和。这组索引叫作嵌入袋。

You can also multiply each selected row by a per-sample weight before adding it.

也可以先给每行乘上对应样本的权重，再相加。

Each bag produces one vector.

每个嵌入袋生成一个向量。

Ava: And recommendation systems have enough tables for this to become a serious engineering problem?

Ava：推荐系统里的表多到足以让这成为棘手的工程问题？

Brian: Yes.

Brian：是的。

The article places these operators in recommendation systems spread across thousands of GPUs.

文章讨论的是分布在数千个 GPU 上的推荐系统中的这些算子。

We'll use three shape terms: E is the number of table rows, D is the row width, and L is the bag size.

我们用三个形状参数：E 是表的行数，D 是每行的宽度，L 是嵌入袋的大小。

Ava: All right. How does FBTriton's forward pass collect those rows?

Ava：明白了。FBTriton 的前向计算怎么收集这些行？

Brian: Its generic gather path loads indexed rows and accumulates them.

Brian：通用的 gather 路径会加载索引指定的行，并把它们累加起来。

Each Triton program loops over the table features.

每个 Triton 程序都会遍历表的各个特征。

Depending on the workload, a program handles one, two, or four bags.

视工作负载而定，一个程序处理一、二或四个嵌入袋。

Ava: By program, you mean a unit of GPU work here.

Ava：你说的程序，是指这里的一个 GPU 工作单元。

Does the accumulation use the same precision as the stored weights?

累加时的精度和存储权重的精度一样吗？

Brian: It doesn't.

Brian：不一样。

FP16, meaning sixteen-bit floating point, and BF16, meaning bfloat sixteen, accumulate in FP32, or thirty-two-bit floating point.

FP16（16 位浮点数）和 BF16（bfloat16）都用 FP32（32 位浮点数）累加。

FP32 weights accumulate in FP64, or sixty-four-bit floating point, to preserve accuracy at large row widths.

FP32 权重用 FP64（64 位浮点数）累加，以保证行宽较大时的精度。

Ava: There's also a histogram path. Why count identifiers when the job is to add vectors?

Ava：还有一条直方图处理路径。既然要做向量求和，为什么要统计标识符？

Brian: Imagine ordering the same dish ten times.

Brian：想象一下，同一道菜点了十次。

You can write ten separate orders, or write the dish once with a count of ten.

你可以写十张订单，也可以只写一次菜名，再标上数量十。

Here, repeated identifiers become counts, which are multiplied by the table.

这里会把重复的标识符转成计数，再用计数乘以表中的对应向量。

Ava: So repeated row additions become a matrix calculation that can use tensor cores, the GPU's matrix math hardware?

Ava：也就是说，把重复的行相加转成矩阵计算，交给 GPU 的矩阵运算硬件 Tensor Core？

Brian: Right.

Brian：对。

But eligibility is narrow: at most sixty-four rows, row widths from sixty-four through one hundred twenty-eight, and at least sixty-four entries per bag.

但适用范围很窄：最多 64 行，行宽为 64 到 128，每个 bag 至少有 64 个条目。

It also requires FP16 weights, FP32 output, no per-sample weights, and no variable batching.

还要求权重为 FP16、输出为 FP32，不能有逐样本权重，也不能使用可变批处理。

Ava: Definitely read the small print. Does the histogram cover the whole bag?

Ava：这些限制得仔细看。直方图会覆盖整个 bag 吗？

Brian: It covers the first two hundred fifty-six indices. A scalar path handles the rest.

Brian：它只覆盖前 256 个索引，剩下的由标量路径处理。

One program processes sixteen bags. Other shapes use the generic kernel.

一个程序处理 16 个 bag，其他形状则使用通用内核。

The key point is that this fast path serves a specific workload.

关键是，这条快速路径只服务于特定工作负载。

Ava: Before we move to backward, what happens if an input index is invalid?

Ava：说到反向传播之前，输入索引无效时会怎样？

Brian: The standalone path runs CUDA validation first.

Brian：独立处理路径会先运行 CUDA 校验。

Optional fused bounds checking lets eligible kernels validate and repair inputs, although offset repair stays separate.

可选的融合边界检查让符合条件的内核校验并修正输入，但偏移量仍需单独修正。

It defaults off, and several configurations use standard validation.

这个选项默认关闭，而且有些配置仍使用标准校验。

Ava: Okay. Now backward sounds harder. Many bags can touch the same table row.

Ava：好。反向传播听起来更难，很多 bag 可能访问表中的同一行。

Who gets to update it?

那这一行由谁来更新？

Brian: First, sum all gradient contributions for each unique table and row pair.

Brian：先把每个唯一的表与行组合对应的梯度贡献加起来。

Then apply exactly one optimizer update to that row.

然后对该行只执行一次优化器更新。

Think of collecting every expense receipt before calculating the final account balance.

就像先收齐所有费用收据，再计算最终账户余额。

Ava: That's the part I wanted to pin down. How do they organize all those contributions?

Ava：这正是我想弄清楚的。他们怎么组织这么多次贡献？

Brian: They group accesses into runs, each associated with one unique row.

Brian：他们把访问分成若干连续段，每段对应唯一的一行。

Segment length, abbreviated SL, measures how many samples touch that row.

段长度简称 SL，表示有多少个样本访问这一行。

Within one batch, that length can range from one to millions.

在同一个批次中，这个长度可能从 1 到数百万不等。

Ava: One worker gets one receipt. Another gets two million.

Ava：一个工作单元只处理一次访问，另一个却要处理两百万次。

That's a fairly terrible division of labor.

这分工也太不均衡了。

Brian: Exactly.

Brian：没错。

For runs below two hundred fifty-six lookups, one program gathers, accumulates, updates, and stores.

对少于 256 次查找的连续段，一个程序完成数据收集、累加、更新和写回。

It owns the row exclusively, so its final store doesn't need an atomic operation to coordinate competing writers.

它独占这一行，因此最终写回时无需用原子操作协调其他写入者。

Ava: And a long run gets divided among more workers?

Ava：长的连续段会分给更多工作单元？

Brian: Yes, into chunks of two hundred fifty-six lookups.

Brian：对，每块处理 256 次查找。

By default, programs write partial results to a workspace, and another kernel applies the optimizer.

默认情况下，各程序把部分结果写入工作区，再由另一个内核应用优化器。

A two-million-lookup run becomes roughly eight thousand sub-programs.

一个包含两百万次查找的连续段，会拆成大约八千个子程序。

Ava: So the basic rule is simple: keep small jobs together and divide large jobs.

Ava：基本原则很简单：小任务放在一起，大任务拆开。

What changes on Blackwell hardware?

在 Blackwell 硬件上有什么变化？

Brian: For very large batches, a fused path can finish the update in the same launch.

Brian：对于很大的批次，融合路径可以在同一次启动中完成更新。

It uses a device-scope fence, which ensures each partial result is visible across the GPU before its completion counter decreases.

它使用设备范围的内存栅栏，确保每个部分结果在完成计数器递减前，对整个 GPU 可见。

Ava: Wait, so reaching zero isn't enough?

Ava：等等，计数器归零还不够？

The data actually has to arrive before you announce that you're done?

必须先确保数据确实写入，才能宣布完成？

Brian: Exactly. Without that ordering, the last program could read an incomplete sum.

Brian：没错。没有这个顺序保证，最后一个程序可能读到不完整的求和结果。

The article also describes Blackwell features for stealing available work and merging partial results through a bulk reduction instruction.

文章还介绍了 Blackwell 的功能：空闲工作单元可以接手任务，并通过批量归约指令合并部分结果。

Ava: What else helped? I'd expect loading more rows at once to be faster.

Ava：还有什么优化？我以为一次加载更多行会更快。

Brian: That can backfire.

Brian：这可能适得其反。

Buffered rows keep many addresses in registers, the GPU's small working storage.

缓存多行数据会让许多地址占用寄存器，而寄存器是 GPU 容量有限的工作存储。

In one unweighted short-run case, reducing gather width from eight to two cut register use from one hundred eighty-four to sixty-four.

在一个无权重的短连续段案例中，把收集宽度从 8 降到 2，寄存器用量就从 184 降到了 64。

Ava: Less luggage per worker leaves room for more workers?

Ava：每个工作单元少带点行李，就能容纳更多工作单元？

Brian: That's a useful analogy.

Brian：这个比喻很贴切。

Occupancy, the share of available GPU execution capacity occupied by active work, rose from twelve point five percent to about fifty percent.

占用率，也就是活跃任务占 GPU 可用执行容量的比例，从 12.5% 升至约 50%。

They also group short runs by row width to reduce unused lanes.

他们还按行宽将短连续段分组，减少闲置的执行通道。

Ava: And does the central processing unit, or CPU, decide how many long runs there are?

Ava：长连续段的数量由中央处理器，也就是 CPU，来决定吗？

Brian: They keep those counts on the GPU.

Brian：他们把这些计数留在 GPU 上。

Reading them back would synchronize the CUDA stream every backward pass.

如果读回 CPU，每次反向传播都会同步 CUDA 流。

Instead, GPU classification fills preallocated workspace, and the kernels consume the counts directly.

因此，他们让 GPU 完成分类并填充预分配的工作区，内核再直接使用这些计数。

Ava: Let's talk results. How broad was the evaluation?

Ava：说说结果吧。评测覆盖面有多广？

Brian: Three hundred seven shard configurations, representing two hundred eighty-three distinct shapes, on GB200.

Brian：他们在 GB200 上测试了 307 种分片配置，涵盖 283 种不同形状。

They used FP16 weights and exact row-wise Adagrad, the optimizer in this evaluation.

他们使用 FP16 权重，以及此次评测采用的精确逐行 Adagrad 优化器。

The median forward speedup was one point two eight times.

前向计算的加速比中位数是 1.28 倍。

Ava: What explains the backward gains?

Ava：反向计算的性能提升来自哪里？

Brian: CUDA switches to a cooperative thread array, or CTA, per row at segment length thirty-two.

Brian：从段长度达到 32 开始，CUDA 就为每行启用一个协作线程数组，即 CTA。

Triton keeps simple streaming below two hundred fifty-six.

Triton 则在段长度低于 256 时继续采用简单的流式处理。

Deep in that band, the article reports four point three times faster execution.

在这个区间内，文章报告的执行速度最高达到 4.3 倍。

Ava: Because it moves less data?

Ava：因为它搬运的数据更少？

Brian: It moves roughly the same amount.

Brian：数据搬运量差不多。

The highlighted comparison shows much higher memory throughput at identical occupancy.

重点对比显示，在占用率相同的情况下，它的内存吞吐量高得多。

Just above thirty-two accesses, the cooperative CUDA path doesn't have enough work to spread out its synchronization cost.

访问次数刚超过 32 时，CUDA 的协作路径没有足够的工作量来摊薄同步开销。

Ava: Where does Triton still struggle?

Ava：Triton 在哪些场景下仍然吃力？

Brian: Workloads made entirely of runs shorter than four lose to CUDA.

Brian：如果工作负载全部由长度小于 4 的连续段组成，Triton 就比 CUDA 慢。

That's eleven shards, each under about a millisecond.

这涉及 11 个分片，每个分片的耗时都不到约 1 毫秒。

Above segment length two hundred fifty-six, performance is at parity.

段长度超过 256 时，两者性能相当。

The supplied text doesn't give an overall backward median.

提供的文本没有给出反向计算加速比的总体中位数。

Ava: And what's next? Are forward and backward already one giant kernel?

Ava：接下来呢？前向和反向已经融合成一个大内核了吗？

Brian: No. Histogram reuse still uses separate launches.

Brian：还没有。直方图复用仍需分别启动内核。

Optional preprocessing moved into forward reduced combined latency by sixteen point eight percent in one large B200 configuration.

在一种大型 B200 配置中，将可选预处理移入前向计算，使总延迟降低了 16.8%。

It defaults off and isn't exposed through the current TorchRec wrapper.

该选项默认关闭，当前的 TorchRec 封装也未提供此选项。

Broader fusion is a future direction.

更大范围的融合是未来的方向。

Ava: Let's wrap up with three points. First, batching embedding work cuts launch overhead.

Ava：最后总结三点。第一，批量处理嵌入计算能减少内核启动开销。

Second, matching work to run length and row width improves execution.

第二，根据连续段长度和行宽安排计算，能提高执行效率。

Third, the results depend on workload shape, and larger fusion opportunities remain ahead.

第三，效果取决于工作负载的形态，未来还有更大的融合空间。

Brian: Thanks for listening, and we'll see you next time.

Brian：感谢收听，我们下次见。

## 术语

| Term | 释义 |
|---|---|
| Table-Batched Embedding | 表批量嵌入：在一次 GPU 启动中对多张嵌入表执行查找与池化。 |
| embedding bag | 嵌入袋：一组待查找的索引，其对应向量被聚合为一个输出向量。 |
| per-sample weights | 逐样本权重：在聚合前用于缩放各次查找结果或梯度贡献的权重。 |
| histogram | 直方图：此处用于统计各索引出现的次数。 |
| tensor cores | 张量核心：用于加速矩阵运算的 GPU 硬件单元。 |
| bounds checking | 边界检查：检查输入索引、偏移等是否有效。 |
| segment length | 段长度：同一嵌入表行被多少个样本访问，简称 SL。 |
| device-scope fence | 设备作用域内存栅栏：确保相关内存操作按要求对整个设备可见。 |
| atomic operation | 原子操作：多个执行单元并发访问时，保证特定更新不可分割地完成的操作。 |
| gather width | 聚集读取宽度：此处指同时发起或缓冲的行读取数量。 |
| occupancy | 占用率：GPU 可用执行资源中由活跃工作占据的比例。 |
| Cooperative Thread Array | 协作线程数组，简称 CTA：共同执行任务的一组 GPU 线程。 |
| exact row-wise Adagrad | 精确逐行 Adagrad：文章评测所使用的优化器，也是直方图复用路径支持的优化器。 |
| kernel fusion | 内核融合：将多个计算阶段合并到同一次内核执行中。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's unpack pooling. | 我们来具体解释一下池化。可用 Let's unpack… 引出概念解释。 |
| That's the part I wanted to pin down. | 这就是我想弄清楚的部分。 |
| Wait, so reaching zero isn't enough? | 等等，所以计数到零还不够？用于确认出乎意料的信息。 |
| That can backfire. | 那可能适得其反。 |
| That's a useful analogy. | 这个类比很有帮助。 |
| Let's talk results. | 我们来谈谈结果。 |
| Where does Triton still struggle? | Triton 在哪些方面仍然表现不佳？可替换主语来询问局限。 |
| Let's wrap up with three points. | 我们用三点来收尾。 |
