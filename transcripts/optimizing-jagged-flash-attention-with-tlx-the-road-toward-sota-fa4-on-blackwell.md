# Keeping Blackwell Busy: Jagged Flash Attention with TLX

原文：[Optimizing Jagged Flash Attention with TLX: The Road Toward SOTA FA4 on Blackwell](https://pytorch.org/blog/optimizing-jagged-flash-attention-with-tlx-the-road-toward-sota-fa4-on-blackwell/)

## 摘要

这篇文章介绍了如何用 TLX 在 NVIDIA Blackwell B200 上优化 Jagged Flash Attention，直接处理变长序列，避免填充带来的计算浪费。核心方法包括 warp 专门化、显式内存管理、持久化执行，以及负载均衡、梯度写回流水线、提前释放张量内存和循环剥离等优化。在文章测试的 jagged 形状上，相比二〇二六年五月版本的 FlashAttention-4，前向吞吐量平均提升约百分之十三，反向平均提升约百分之五十，但 dense 前向仍落后。文章还展示了低精度和块稀疏变体，强调同一套内核结构在性能优化与模型迭代中的复用价值。

## 对话

Ava: What if your attention kernel is doing work for tokens that aren't even there?

Ava：如果注意力核函数还在为根本不存在的 token 做计算呢？

Today we're looking at variable-length sequences, wasted compute, and a faster way to handle them on Blackwell.

今天我们聊聊变长序列、浪费的算力，以及 Blackwell 上更快的处理方法。

Brian: And there's a development story too.

Brian：这里还有个开发方面的故事。

The article's kernel beats FlashAttention-4, or FA4, on the jagged workloads tested, while using roughly a third as many lines of code.

在测试的锯齿状序列任务中，文章中的核函数比 FlashAttention-4（FA4）更快，代码行数却只有大约三分之一。

Ava: Faster and shorter? All right, you've got my attention. What's the actual workload?

Ava：更快，代码还更短？我来兴趣了。具体是什么任务？

Brian: It's attention in Meta's Generative Ads Model, or GEM.

Brian：是 Meta 的生成式广告模型 GEM 中的注意力计算。

Users have histories of different lengths.

用户的历史记录长短不一。

Those sequences are jagged, meaning variable-length.

这些序列是锯齿状的，也就是长度不固定。

Attention is GEM's single slowest kernel.

注意力计算是 GEM 中最慢的核函数。

Ava: Why not pad every history to the same length? That makes the shapes easier.

Ava：为什么不把每段历史记录填充到同样长度？这样形状更规整。

Brian: It does, but the article says padding can waste up to fifty percent of compute.

Brian：是更规整，但文章说填充最多会浪费 50% 的计算量。

GEM packs sequences together instead.

GEM 改用把序列紧密拼接在一起的方式。

An offsets tensor records where each sequence starts and ends.

一个偏移量张量记录每条序列的起止位置。

Ava: Like putting different-length chapters in one book, with an index telling you where to look.

Ava：就像把长短不同的章节装订成一本书，再用目录标出位置。

Brian: Exactly.

Brian：没错。

Jagged Flash Attention, or JFA, applies the FlashAttention algorithm directly to those packed query, key, and value tensors.

锯齿状 Flash Attention（JFA）直接对紧密拼接的查询、键和值张量应用 FlashAttention 算法。

We'll call them Q, K, and V. There's no need to create padded tokens.

我们把它们叫作 Q、K、V，不需要生成填充 token。

Ava: So the representation already saves work. What's still holding the kernel back?

Ava：数据表示已经省了计算量，核函数还有什么瓶颈？

Brian: Scheduling.

Brian：调度。

The original Triton kernel leaves data movement and scheduling to the compiler.

原来的 Triton 核函数把数据搬运和调度交给编译器。

One warp group handles loads, softmax, and matrix multiplications.

一个 warp 组同时负责加载数据、计算 softmax 和矩阵乘法。

A warp is a group of GPU threads.

warp 是一组 GPU 线程。

Ava: And when that group handles softmax, the matrix hardware can sit waiting?

Ava：那这组线程计算 softmax 时，矩阵计算硬件可能就在等？

Brian: Right.

Brian：对。

Blackwell's tensor cores, its matrix-computation hardware, need a steady supply of work.

Blackwell 的张量核心，也就是矩阵计算硬件，需要持续有任务可做。

TLX, short for Triton Low-level Extensions, gives programmers explicit control over memory, synchronization, and overlapping operations.

TLX 是 Triton Low-level Extensions 的缩写，让程序员能明确控制内存、同步和操作重叠。

Ava: What's the first big change they make with that control?

Ava：有了这些控制能力，他们做的第一个重大改动是什么？

Brian: Warp specialization: different warps get different jobs.

Brian：warp 专业化，让不同的 warp 各司其职。

Some load data, some issue matrix operations, some handle softmax, and others write results.

有的加载数据，有的发起矩阵运算，有的计算 softmax，还有的写入结果。

Those jobs can overlap.

这些工作可以同时进行。

Ava: A kitchen with separate stations.

Ava：就像厨房里分了不同工位。

The person cooking doesn't also stop to wash every plate.

做菜的人不用每做一步就停下来洗盘子。

Brian: That's the idea.

Brian：就是这个意思。

They also manage shared memory, or SMEM, and tensor memory, or TMEM, explicitly.

他们还明确管理共享内存 SMEM 和张量内存 TMEM。

Buffers hold data between stages.

缓冲区在各阶段之间存放数据。

Barriers coordinate when producers and consumers can use it.

同步屏障协调生产者和消费者何时可以使用数据。

Ava: So the stations have storage and clear handoffs. How does the kernel keep finding work?

Ava：工位有了存放区，交接也清楚了。核函数怎么持续找到活干？

Brian: Through persistent execution. A cooperative thread array, or CTA, is a thread block.

Brian：靠持久化执行。协作线程阵列 CTA 就是一个线程块。

Each streaming multiprocessor, or SM, gets one CTA that loops over tiles, the chunks of work.

每个流式多处理器 SM 分配一个 CTA，由它循环处理各个图块，也就是一块块任务。

Ava: Let me check I've got this. Divide the jobs, overlap them, and keep the workers running.

Ava：我确认一下：分配不同工作，让它们并行，再让工作线程持续运行。

But different sequence lengths must make the assignments uneven.

但序列长短不同，任务分配肯定不均匀。

Brian: Very uneven. Some SMs finish while others process long sequences.

Brian：非常不均匀。有些 SM 已经做完了，另一些还在处理长序列。

In the forward pass, tile cost follows key length.

在前向计算中，图块的计算量取决于键序列的长度。

The team sorts tiles from longest workload to shortest.

团队按任务量从大到小给图块排序。

Ava: Then gives each worker a mix?

Ava：然后让每个工作线程都分到长短搭配的任务？

Brian: Yes, using a zigzag assignment: left to right, then right to left.

Brian：对，采用之字形分配：先从左到右，再从右到左。

It's cheap to prepare on the CPU.

这种分配方式在 CPU 上准备起来开销很小。

The article reports roughly a twenty percent forward improvement from this balancing.

文章称，这种负载均衡让前向计算速度提升了约 20%。

Ava: Does the backward pass use the same trick?

Ava：反向计算也用同样的方法吗？

Brian: No. Its tiles have roughly equal cost, but sequences produce different numbers of tiles.

Brian：不用。反向计算中每个图块的计算量大致相同，但不同序列产生的图块数量不同。

The host lists valid tiles and distributes them round-robin.

主机端列出有效计算块，再按轮询顺序分配。

Sorting by cost wouldn't help there.

按计算成本排序在这里也没用。

Ava: What happens when the planned assignments still don't finish together?

Ava：如果预先分配的任务还是不能同时完成呢？

Brian: Cluster Launch Control, or CLC, handles runtime variation.

Brian：集群启动控制（CLC）会处理运行时的差异。

It's a Blackwell feature that lets an SM request more work when it finishes.

这是 Blackwell 的一项功能，让 SM 完成任务后能请求更多工作。

They use it in both passes.

他们在两个阶段都用了它。

Ava: Wait, so why keep the software balancing if the hardware can hand out work?

Ava：等等，既然硬件能分配任务，为什么还要用软件做负载均衡？

Brian: Because CLC can't identify empty tiles.

Brian：因为 CLC 无法识别空计算块。

Preparing valid tiles on the host removes those first.

主机端先筛出有效计算块，就能排除空块。

Software avoids useless assignments; CLC handles the remaining variation.

软件避免分配无用任务，CLC 处理剩下的负载差异。

The two approaches work together.

两种方法相互配合。

Ava: Got it. Now, the backward improvement is especially large. Where was it getting stuck?

Ava：明白了。反向计算的提升尤其大，之前卡在哪里？

Brian: The production case uses broadcast-Q: one dense Q is shared across every sequence.

Brian：生产场景采用广播式 Q：所有序列共享同一个稠密 Q。

Its gradient, dQ, must therefore accumulate contributions from the entire batch.

因此，它的梯度 dQ 必须累加整个批次的贡献。

Many SMs update the same locations.

许多 SM 都要更新相同的位置。

Ava: Everybody's trying to write their total into the same spreadsheet cell.

Ava：就像所有人都想把自己的合计写进电子表格的同一个单元格。

That sounds crowded.

听起来很拥挤。

Brian: It is.

Brian：确实如此。

Profiling identified the dQ epilogue, the final accumulation and writeback stage, as the biggest backward bottleneck.

性能分析发现，dQ 的收尾阶段，也就是最后的累加和写回，是反向计算的最大瓶颈。

The existing high-level operation also processes column slices serially.

现有的高层算子还会串行处理各个列切片。

Ava: How do they overlap that work?

Ava：他们怎么让这些操作重叠执行？

Brian: With double-buffered shared-memory staging.

Brian：用双缓冲共享内存暂存。

While one slice is being accumulated into high-bandwidth memory, or HBM, the next slice moves out of tensor memory.

一个切片正被累加到高带宽内存（HBM）时，下一个切片就从张量内存移出。

Two staging buffers keep the process moving.

两个暂存缓冲区让流程持续运转。

Ava: But doesn't the next matrix operation still need that tensor-memory buffer?

Ava：但下一次矩阵运算不是还要用那个张量内存缓冲区吗？

Brian: Yes.

Brian：是的。

They copy the final one or two slices into registers, the threads' working storage, then release the tensor-memory buffer early.

他们把最后一两个切片复制到寄存器，也就是线程的工作存储区，然后提前释放张量内存缓冲区。

The next matrix operation starts while the previous writes finish.

前一次写回尚未结束，下一次矩阵运算就能开始。

Ava: Why stop at two slices? Why not move everything into registers?

Ava：为什么只搬一两个切片？为什么不全搬进寄存器？

Brian: Registers are limited.

Brian：寄存器容量有限。

Holding more data increases register pressure and can hurt performance.

存放更多数据会增加寄存器压力，可能拖慢性能。

They autotune between one and two slices.

他们通过自动调优，在搬运一个或两个切片之间选择。

The point is to release the buffer sooner without creating another bottleneck.

目的是更早释放缓冲区，同时避免制造新瓶颈。

Ava: That brings us to loop peeling. The name sounds like a kitchen task too.

Ava：接下来是循环剥离。这个名字也像厨房里的活儿。

Brian: Here it means separating complete tiles from the partial last tile.

Brian：这里指把完整计算块和最后一个不完整的计算块分开。

Only that final tile needs a mask to exclude invalid positions.

只有最后那个计算块需要掩码来排除无效位置。

Previously, a mask branch lived inside the repeated loop.

之前，处理掩码的分支放在反复执行的循环里。

Ava: Would a rarely needed branch really make that much difference?

Ava：一个很少用到的分支，影响真有这么大吗？

Brian: The issue is register allocation.

Brian：问题出在寄存器分配。

The compiler reserves registers for mask-related variables across the loop.

编译器会在整个循环中为掩码相关变量预留寄存器。

That pressure can cause register spills, where values generate local-memory traffic.

这可能导致寄存器溢出，让数据转而产生本地内存访问。

Separating the tail removes those variables from the bulk.

把末尾部分单独处理，就能让主体循环不再带着这些变量。

Ava: So a tiny edge case was making every normal iteration carry extra baggage.

Ava：也就是说，一个小小的边界情况，让每次正常迭代都背上了额外负担。

Brian: Exactly.

Brian：正是。

One more backward technique comes from FA4: two CTAs cooperate on a matrix multiplication.

Brian：还有一项反向计算技术来自 FA4：让两个 CTA 协同完成一次矩阵乘法。

For the production broadcast-Q case with head dimension one hundred twenty-eight, that adds about twelve percent throughput.

在头维度为 128 的生产级广播式 Q 场景中，这让吞吐量提高约 12%。

Ava: Let's put the results in context. What did they actually compare?

Ava：我们来看看结果是怎么比较的。他们具体比了什么？

Brian: They tested on NVIDIA B two hundred with bfloat16, a sixteen-bit floating-point format, against the May twenty twenty-six version of FA4.

Brian：他们在 NVIDIA B200 上使用 16 位浮点格式 bfloat16，与 2026 年 5 月版 FA4 进行比较。

On jagged shapes, average throughput gains were about thirteen percent forward and fifty percent backward.

在不规则形状下，正向吞吐量平均提高约 13%，反向提高约 50%。

Ava: Are those wins across the board?

Ava：这些提升在所有情况下都有吗？

Brian: Backward wins on every tested jagged shape.

Brian：在所有测试过的变长形状上，反向计算都更快。

Forward trails on the longest sequences at high density.

在高密度的最长序列上，前向计算则落后。

For equal-length dense inputs, forward reaches about eighty-seven percent of FA4's performance; backward is about seventeen percent faster.

对于等长的稠密输入，前向性能约为 FA4 的 87%；反向则快约 17%。

Ava: And we shouldn't add all the individual optimization percentages together.

Ava：但我们不能把各项优化的百分比直接相加。

Brian: Right. Those describe different comparisons and conditions.

Brian：对，它们对应不同的比较和条件。

The overall results are the useful endpoint.

最终的整体结果更有参考价值。

The implementation is also about three thousand two hundred lines, versus roughly ten thousand for the FA4 kernels.

这个实现约有 3200 行代码，而 FA4 内核约有 10000 行。

Ava: What does that structure buy them when the model changes?

Ava：模型变化时，这种结构有什么好处？

Brian: They reuse it for MXFP8, a microscaling eight-bit floating-point variant, and block-sparse attention, which visits selected blocks.

Brian：他们将它复用于 MXFP8（一种微缩放 8 位浮点格式）和块稀疏注意力；后者只访问选定的块。

At a selection ratio of zero point five, sparse forward is roughly one point three to one point five times faster than dense.

选择比例为 0.5 时，稀疏前向计算的速度约为稠密计算的 1.3 到 1.5 倍。

Ava: Any limits we should keep in mind before trying those variants?

Ava：尝试这些变体前，有什么局限需要注意？

Brian: The low-precision variant targets training that tolerates FP8.

Brian：低精度变体适用于能够容忍 FP8 精度的训练。

The article doesn't provide detailed model-quality results for these variants or a concrete future roadmap.

文章没有提供这些变体对模型质量影响的详细结果，也没有给出具体的后续路线图。

Its performance claims are tied to the workloads and hardware tested.

其性能结论仅适用于测试过的工作负载和硬件。

Ava: Let's finish with three points. First, packed jagged sequences avoid padding work.

Ava：最后总结三点。第一，紧凑存储的变长序列避免了填充带来的额外计算。

Second, explicit scheduling and memory control keep the hardware busier.

第二，显式调度和内存控制提高了硬件利用率。

Third, the measured gains come with a kernel structure that's reusable.

第三，实测性能提升来自一种可复用的内核结构。

Brian: Thanks for listening, and we'll see you next time.

Brian：感谢收听，我们下次见。

## 术语

| Term | 释义 |
|---|---|
| Jagged Flash Attention | 直接在打包的变长序列及其偏移信息上执行 FlashAttention 的注意力内核。 |
| TLX | Triton Low-level Extensions，为 Triton 提供显式的底层硬件控制能力。 |
| offsets tensor | 偏移量张量，用于记录打包后各序列的边界。 |
| warp specialization | 让不同 warp 分别负责加载、矩阵运算、softmax 或写回等任务。 |
| SMEM | shared memory，共享内存，文中用于缓冲和分阶段传递数据。 |
| TMEM | tensor memory，张量内存，文中用于保存矩阵运算相关数据。 |
| CTA | cooperative thread array，协作线程数组，即线程块。 |
| SM | streaming multiprocessor，流式多处理器。 |
| persistent execution | 持久化执行，让线程块持续循环处理多个工作分块。 |
| Cluster Launch Control | Blackwell 的动态工作调度能力，允许完成当前工作的 SM 请求后续工作。 |
| broadcast-Q | 整个批次中的各序列共享同一个稠密 Q，因此 dQ 需要跨批次累加。 |
| epilogue | 内核主计算之后的收尾阶段，文中主要涉及梯度累加与写回。 |
| loop peeling | 循环剥离，将完整分块的主循环与需要掩码的尾部分开。 |
| register spills | 寄存器溢出，寄存器不足导致部分值产生局部内存访问。 |
| MXFP8 | 采用分块缩放因子的八位浮点数格式，文中用于低精度注意力变体。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| You've got my attention. | 你引起我的兴趣了。 |
| What's still holding the kernel back? | 还有什么在限制这个内核的性能？ |
| Let me check I've got this. | 让我确认一下自己理解得对不对。 |
| Wait, so why keep the software balancing if the hardware can hand out work? | 等等，既然硬件可以分派工作，为什么还要保留软件负载均衡？ |
| That brings us to loop peeling. | 这就引出了循环剥离这个话题。 |
| Let's put the results in context. | 我们结合具体条件来看这些结果。 |
| Are those wins across the board? | 这些提升在所有情况下都成立吗？ |
| Any limits we should keep in mind before trying those variants? | 尝试这些变体之前，有什么限制需要注意？ |
