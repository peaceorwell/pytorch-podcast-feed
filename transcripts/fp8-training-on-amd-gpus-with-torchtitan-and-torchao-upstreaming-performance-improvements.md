# FP8 Training on AMD GPUs with TorchTitan and TorchAO: Upstreaming Performance Improvements

原文：[FP8 Training on AMD GPUs with TorchTitan and TorchAO: Upstreaming Performance Improvements](https://pytorch.org/blog/fp8-training-on-amd-gpus-with-torchtitan-and-torchao-upstreaming-performance-improvements/)

## 摘要

本期讨论 TorchTitan 和 TorchAO 如何让 AMD Instinct GPU 原生支持 FP8 训练，并把 Primus-Turbo 中的优化合入 PyTorch 主线。节目解释了 AMD 的 e4m3fnuz 格式、MoE grouped GEMM，以及通过 Triton 融合减少量化开销的方法。在 Llama3-8B 上，FP8 相比 BF16 吞吐提升 13.4%；在 DeepSeek-V3 671B 上，前向融合恢复了 89% 的 FP8 量化开销。最后介绍了 MXFP8 grouped GEMM 和 MI355X 的后续工作。

## 对话

Ava: If you have AMD Instinct GPUs, FP8 training can now work out of the box in the standard PyTorch stack. And on Llama3-8B, it delivered a 13.4% throughput gain over BF16. That sounds pretty practical.

Ava: 如果你有 AMD Instinct GPU，现在就可以直接用标准 PyTorch 技术栈进行 FP8 训练。在 Llama3-8B 上，相比 BF16，吞吐量提高了 13.4%。听起来很实用。

Brian: Exactly. The key story is upstreaming. Optimizations first demonstrated with Primus-Turbo were merged into TorchAO and TorchTitan, so you don't need a separate AMD-specific training library.

Brian: 没错。关键在于将优化成果合入上游。最初通过 Primus-Turbo 展示的优化已经合入 TorchAO 和 TorchTitan，因此不需要单独的 AMD 专用训练库。

Ava: So this episode is about the path from a big optimization project to normal PyTorch support?

Ava: 所以这一期要讲的是，一个大型优化项目如何逐步成为 PyTorch 的常规支持？

Brian: Right. We'll walk through correctness first, then performance, then the limits and what's next.

Brian: 对。我们会先讲正确性，再讲性能，最后讲局限和下一步计划。

Ava: Let's start with FP8 itself. What changes when a linear layer uses it?

Ava: 先从 FP8 本身说起。线性层使用 FP8 时会发生什么变化？

Brian: A linear layer has three matrix multiplications: the forward pass, the gradient input, and the gradient weight update. FP8 training quantizes these operations from 16 bits to 8 bits, which can improve throughput.

Brian：线性层涉及三次矩阵乘法：前向传播、输入梯度计算和权重梯度更新。FP8 训练将这些运算从 16 位量化到 8 位，从而可能提高吞吐量。

Ava: But AMD doesn't use exactly the same FP8 format as NVIDIA, correct?

Ava: 但 AMD 使用的 FP8 格式与 NVIDIA 的并不完全一样，对吗？

Brian: Correct. AMD Instinct GPUs use an FNUZ variant. FNUZ means Finite, No NaN, Unsigned Zero. The specific format here is e4m3fnuz.

Brian：没错。AMD Instinct GPU 使用 FNUZ 变体。FNUZ 意为“有限值、无 NaN、无符号零”。这里的具体格式是 e4m3fnuz。

Ava: What matters about that format?

Ava: 这种格式有哪些关键之处？

Brian: Its maximum value is 240, and it has no NaN or infinity encodings. It's implemented on MI300X, MI325X, and MI350X GPUs.

Brian: 它的最大值是 240，而且没有 NaN 或无穷大的编码。MI300X、MI325X 和 MI350X GPU 都实现了这种格式。

Ava: Why is choosing that exact format a correctness issue?

Ava: 为什么选对这个具体格式会关系到正确性？

Brian: TorchAO originally calculated scales using a different maximum value, from NVIDIA's e4m3fn format. On AMD, values could be scaled beyond what e4m3fnuz can represent. Activations were clipped, gradients were corrupted, and model quality degraded.

Brian: TorchAO 最初使用另一个最大值来计算缩放系数，这个值来自 NVIDIA 的 e4m3fn 格式。在 AMD 上，数值可能被缩放到超出 e4m3fnuz 可表示范围的程度。激活值因此被截断，梯度遭到破坏，模型质量也随之下降。

Ava: And because there are no NaN or infinity encodings, the overflow didn't raise an error?

Ava: 而且因为没有 NaN 或无穷大的编码，溢出也不会报错？

Brian: Exactly. It silently produced wrong results. So the team added hardware auto-detection. TorchAO now selects the correct FP8 dtype and maximum value automatically.

Brian: 正是如此。它会悄无声息地产生错误结果。因此团队加入了硬件自动检测。现在 TorchAO 会自动选择正确的 FP8 数据类型和最大值。

Ava: They also fixed the reported peak FLOPS and loss baselines, right?

Ava: 他们还修正了报告中的峰值 FLOPS 和损失基线，对吧？

Brian: Yes. TorchTitan now reports correct MI300X peak FLOPS for MFU, or Model FLOPS Utilization, numbers. It also has platform-specific loss baselines for FNUZ numerics.

Brian: 是的。现在 TorchTitan 会报告正确的 MI300X 峰值 FLOPS，用于计算 MFU，也就是 Model FLOPS Utilization。它还针对 FNUZ 的数值特性设置了平台专属的损失基线。

Ava: Before we move to speed, give us the scaling choices. FP8 still needs scales.

Ava: 在谈速度之前，先介绍一下缩放方式。FP8 仍然需要缩放系数。

Brian: There are four granularities. Tensorwise uses one scale for the whole tensor. It's fastest but coarsest. Rowwise uses one scale per row. Blockwise uses one per fixed-size tile. MXFP8 uses scales per group packed alongside the data.

Brian：共有四种粒度。Tensorwise 对整个张量使用一个缩放因子，速度最快，但粒度最粗。Rowwise 对每一行使用一个缩放因子。Blockwise 对每个固定大小的数据块使用一个缩放因子。MXFP8 按组使用缩放因子，并将其与数据一起打包。

Ava: And TorchAO and TorchTitan support all four on AMD?

Ava：TorchAO 和 TorchTitan 在 AMD 上都支持这四种粒度吗？

Brian: Yes. The AMD work made each strategy operate correctly with AMD numerics, and added blockwise kernel support for MI300 and MI350.

Brian: 是的。AMD 方面的工作使每种策略都能按 AMD 的数值特性正确运行，还为 MI300 和 MI350 增加了 blockwise 内核支持。

Ava: So the first takeaway is simple: format detection and scaling have to be correct before any benchmark means anything.

Ava: 所以第一个要点很简单：格式检测和缩放必须正确，否则任何基准测试都没有意义。

Brian: That's the right recap. Correct dtype, correct maximum value, correct baselines, and correct scaling strategies. Now let's talk about dense models.

Brian: 总结得对。数据类型要正确，最大值要正确，基线要正确，缩放策略也要正确。现在来谈谈稠密模型。

Ava: The headline result uses Llama3-8B on eight MI300X GPUs?

Ava：主要结果是在 8 块 MI300X GPU 上使用 Llama3-8B 得到的吗？

Brian: Yes. With rowwise FP8, throughput increased 13.4% over BF16. Peak memory was nearly identical, around 39 gigabytes.

Brian: 是的。使用 rowwise FP8 时，吞吐量比 BF16 提高了 13.4%。峰值内存几乎相同，约为 39 GB。

Ava: So the gain wasn't from saving memory.

Ava: 所以性能提升并不是靠节省内存获得的。

Brian: Right. The recipe keeps the weight-update GEMM in BF16 for higher precision. The forward and gradient-input GEMMs use FP8. The speedup comes from faster FP8 matrix cores.

Brian: 对。这个训练方案为了保持更高精度，仍用 BF16 执行权重更新的 GEMM；前向传播和输入梯度计算的 GEMM 则使用 FP8。加速来自更快的 FP8 矩阵计算核心。

Ava: The setup also used torch.compile, FSDP2, and selective activation checkpointing?

Ava: 这套配置还使用了 torch.compile、FSDP2 和选择性激活检查点技术？

Brian: Correct. FSDP means Fully Sharded Data Parallel. The reported run used batch size one, sequence length eight thousand one hundred ninety-two, and one hundred steps.

Brian：没错。FSDP 是 Fully Sharded Data Parallel，即全分片数据并行。报告中的运行采用了 1 的批大小、8192 的序列长度，共执行 100 步。

Ava: Dense models sound relatively straightforward. Why are Mixture-of-Experts models harder?

Ava: 稠密模型听起来相对直接。为什么 Mixture-of-Experts 模型更难处理？

Brian: MoE models route each token to a subset of experts. That creates variable-size batches. Instead of one uniform GEMM, you need a grouped GEMM.

Brian：MoE 模型会将每个 token 路由给部分专家。这会产生大小不一的批次，因此需要分组 GEMM，而不是一次处理统一批次的 GEMM。

Ava: Grouped GEMM means several expert matrix multiplications handled together?

Ava: Grouped GEMM 是指把多个专家的矩阵乘法放在一起处理？

Brian: Exactly. It needs per-row activation scales, per-expert-column weight scales, and an offset tensor that routes rows to the right expert.

Brian: 没错。它需要逐行的激活值缩放系数、每个专家各列的权重缩放系数，以及一个将各行路由到对应专家的偏移量张量。

Ava: And the team enabled FP8 grouped GEMM on ROCm through the Composable Kernel backend?

Ava: 团队还通过 Composable Kernel 后端，在 ROCm 上启用了 FP8 grouped GEMM？

Brian: Yes. They adapted the quantization pipeline to use the correct AMD dtype and dispatch path through Composable Kernel.

Brian: 是的。他们调整了量化流程，使其使用正确的 AMD 数据类型，并通过 Composable Kernel 走正确的分派路径。

Ava: Now let's get into the expensive part: quantization overhead.

Ava: 现在来说开销较大的部分：量化开销。

Brian: The basic FP8 pipeline has several steps. First, compute the absolute maximum, or absmax, per row or column. Then derive a scale and apply it. Finally, clamp and cast to FP8.

Brian: 基本的 FP8 流程有几个步骤。首先，逐行或逐列计算绝对值最大值，也就是 absmax。然后推导出缩放系数并应用它。最后，将数值截取到允许范围内，再转换为 FP8。

Ava: Each step used to be a separate kernel launch?

Ava: 以前每一步都需要单独启动一个内核？

Brian: Yes. Each launch could materialize an intermediate tensor in HBM, or High Bandwidth Memory. With dozens of expert weight tensors, those round-trips dominated the overhead.

Brian: 是的。每次启动都可能在 HBM，也就是 High Bandwidth Memory 中生成一个中间张量。有几十个专家权重张量时，这些往返数据传输就成了主要开销。

Ava: So the problem was memory movement, not the eight-bit arithmetic itself.

Ava: 所以问题在于数据搬运，而不是 8 位运算本身。

Brian: Exactly. On these MoE shapes, the math was cheap. Kernel launches and HBM traffic were expensive. The optimizations attacked data movement at three levels.

Brian：没错。对于这些 MoE 张量形状，数学计算的开销很低，内核启动和 HBM 数据传输的开销却很高。这些优化从三个层面减少了数据移动。

Ava: Level one was launching fewer kernels. What happened in the backward pass?

Ava：第一层是减少内核启动次数。反向传播阶段发生了什么？

Brian: There were two issues. A transpose pattern, written as transpose, contiguous, transpose, forced a full tensor copy through HBM. Pull request number 3972 removed those redundant copies.

Brian：有两个问题。其中一种转置模式，写作 transpose、contiguous、transpose，会迫使整个张量通过 HBM 复制一次。第 3972 号拉取请求移除了这些冗余复制。

Ava: And the scale-and-cast chain was still split across kernels.

Ava: 而缩放和转换的数据处理链仍然分散在多个内核中。

Brian: Right. Pull request 4069 fused that chain into single Triton kernels in several places. It also added a companion dual-kernel for simultaneous gradient-output and activation quantization.

Brian: 对。拉取请求 4069 在多个位置将这条处理链融合为单个 Triton 内核。它还增加了一个配套的 dual-kernel，用于同时量化输出梯度和激活值。

Ava: What was the measured effect?

Ava: 测得的效果如何？

Brian: On eight MI300X GPUs with DeepSeek-MoE-16B, the backward fusions delivered a 4.2 times throughput improvement for the backward pass.

Brian：在使用八块 MI300X GPU 运行 DeepSeek-MoE-16B 时，反向传播阶段的融合优化使该阶段的吞吐量提升至原来的 4.2 倍。

Ava: That's a large jump. Did the forward path have the same pattern?

Ava: 提升相当大。前向路径也存在同样的情况吗？

Brian: Yes. Quantizing expert weights launched five generic kernels per call. There were 24 calls per step, adding about 90 milliseconds per step.

Brian：对。每次量化专家权重都会启动五个通用内核。每一步要调用 24 次，导致每步增加约 90 毫秒。

Ava: And the fix was one fused Triton kernel?

Ava：解决办法是用一个融合的 Triton 内核？

Brian: Exactly. Pull request 4311 replaced the chain with one fused kernel. It parallelized across both experts and output-dimension blocks, so the surrounding GEMMs could issue sooner.

Brian：没错。第 4311 号拉取请求用一个融合内核替代了原来的内核链。它同时在专家和输出维度块上实现并行，因此周围的 GEMM 运算可以更早发起。

Ava: What happened on DeepSeek-V3 671B?

Ava：DeepSeek-V3 671B 的情况怎么样？

Brian: On eight MI325X GPUs, end-to-end throughput rose 17%, from 5,996 to 7,027 tokens per second. The forward portion went from about 19 milliseconds to about 7 milliseconds.

Brian：在八块 MI325X GPU 上，端到端吞吐量提高了 17%，从每秒 5,996 个 token 增至每秒 7,027 个 token。前向部分的耗时从约 19 毫秒降至约 7 毫秒。

Ava: The BF16 baseline was 7,156 tokens per second, so FP8 got quite close.

Ava：BF16 基线是每秒 7,156 个 token，所以 FP8 已经相当接近了。

Brian: Yes. The fused forward kernel recovered 89% of the BF16-to-FP8 gap. The upstream FP8 configuration had added 127 milliseconds per step, and 92% of that was in generic quantization kernels. Fusion removed most of it.

Brian：是的。融合前向内核弥补了 BF16 与 FP8 之间性能差距的 89%。上游的 FP8 配置使每一步增加了 127 毫秒，其中 92% 花在通用量化内核上。融合消除了其中大部分开销。

Ava: Level two was making each remaining kernel move memory efficiently. What was wrong with the colwise scales kernel?

Ava：第二层优化是让剩余的每个内核都高效地搬运内存数据。colwise scales 内核出了什么问题？

Brian: Its writes were non-coalesced. Consecutive SIMD lanes wrote to addresses separated by K bytes, causing separate memory transactions.

Brian：它的写入没有合并访存。相邻的 SIMD 通道写入的地址相隔 K 字节，导致产生单独的内存事务。

Ava: How did they fix that?

Ava：他们是怎么修复的？

Brian: They transposed the output tile through LDS, or Local Data Share, before storing it. They also added a fused single-pass variant that removed a redundant HBM read.

Brian：他们在写入前通过 LDS（Local Data Share）转置输出数据块。他们还增加了一个融合的单遍版本，去掉了一次多余的 HBM 读取。

Ava: And the per-layer result?

Ava：每层的结果呢？

Brian: On MI300X with DeepSeek-V3 671B shapes, one MoE layer dropped from 7,290 microseconds to 1,170 microseconds. That's a 6.2 times speedup.

Brian：在 MI300X 上，使用 DeepSeek-V3 671B 的张量形状时，一个 MoE 层的耗时从 7,290 微秒降到了 1,170 微秒，速度提升了 6.2 倍。

Ava: Level three removed synchronization that the hardware didn't need.

Ava：第三层优化去掉了硬件并不需要的同步。

Brian: Right. Triton's atomic_add, atomic_max, and atomic_min defaulted to acquire-release ordering. On AMD, that inserted memory fences before and after every atomic operation.

Brian：对。Triton 的 atomic_add、atomic_max 和 atomic_min 默认使用 acquire-release 内存序。在 AMD 上，这会在每次原子操作前后都插入内存栅栏。

Ava: Those fences are unnecessary for commutative reductions?

Ava：对于满足交换律的归约，这些内存栅栏是多余的吗？

Brian: In this case, yes. The team switched to relaxed ordering on AMD, guarded by a torch.version.hip check. NVIDIA behavior stayed unchanged.

Brian：在这个场景下，是的。团队通过 torch.version.hip 检查，将 AMD 上的内存序改为 relaxed ordering。NVIDIA 上的行为保持不变。

Ava: Did every attempted optimization work?

Ava：尝试过的优化都奏效了吗？

Brian: No. They expanded the Triton autotune search space from one to eight or sixteen candidate configurations, hoping to find better tile sizes for AMD wavefronts.

Brian：没有。他们把 Triton 自动调优的搜索范围从一种候选配置扩大到八种或十六种，希望找到更适合 AMD wavefront 的数据块尺寸。

Ava: But Llama 4 shapes on MI300X showed no measurable improvement.

Ava：但在 MI300X 上，Llama 4 的张量形状没有显示出可测量的改进。

Brian: Correct. The extra configurations increased first-iteration compile time, so the change was reverted. The lesson was to shape autotuning around wavefront size, LDS capacity, and register pressure, rather than adding candidates by default.

Brian：没错。额外的配置增加了首次迭代的编译时间，所以这项改动被撤回了。经验是，自动调优应围绕 wavefront 大小、LDS 容量和寄存器压力来设计，而不是默认增加候选配置。

Ava: Let's connect the pieces. What are the main results across workloads?

Ava：我们把这些结果串起来看。不同工作负载的主要结果是什么？

Brian: For Llama3-8B dense training, rowwise FP8 gave 13.4% more throughput than BF16. For DeepSeek-MoE-16B, backward fusion reached 4.2 times the backward throughput. For DeepSeek-V3 671B, colwise scale optimization reached 6.2 times per MoE layer, and forward fusion improved end-to-end throughput by 17%.

Brian：对于 Llama3-8B 稠密模型训练，rowwise FP8 的吞吐量比 BF16 高 13.4%。对于 DeepSeek-MoE-16B，反向计算融合使反向计算吞吐量达到原来的 4.2 倍。对于 DeepSeek-V3 671B，colwise scale 优化使每个 MoE 层提速 6.2 倍，前向计算融合则使端到端吞吐量提高了 17%。

Ava: And all of this is now upstream?

Ava：这些改进现在都已经合入上游了吗？

Brian: Yes. The contributions are merged into mainline pytorch/ao and pytorch/torchtitan. Teams with AMD Instinct GPUs can get the gains by upgrading TorchAO and TorchTitan, with nothing AMD-specific to install.

Brian：是的。这些贡献已经合入 pytorch/ao 和 pytorch/torchtitan 的主线。使用 AMD Instinct GPU 的团队只需升级 TorchAO 和 TorchTitan，就能获得这些性能提升，无需安装任何 AMD 专用组件。

Ava: What's next?

Ava：接下来呢？

Brian: Work is continuing on MXFP8 grouped GEMM and quantization kernels for forward and backward passes on MI355X GPUs. More Triton optimizations are also coming along the fusion pipeline.

Brian：团队正在继续开发适用于 MI355X GPU 前向和反向计算的 MXFP8 grouped GEMM 与量化内核。沿着融合处理流程，也会有更多 Triton 优化。

Ava: Before we sign off, let's do the promised three-point recap.

Ava：结束前，我们来做之前答应的三点回顾。

Brian: First, AMD FP8 correctness depends on hardware-aware FNUZ support, automatic dtype selection, and accurate loss and MFU baselines.

Brian：第一，AMD FP8 的正确性依赖于考虑硬件特性的 FNUZ 支持、自动 dtype 选择，以及准确的 loss 和 MFU 基线。

Ava: Second, MoE models need grouped GEMM with per-row scales, per-expert-column scales, and offset-based routing.

Ava：第二，MoE 模型需要 grouped GEMM，并配合逐行 scale、逐专家列 scale 和基于 offset 的路由。

Brian: Third, the biggest performance wins came from reducing data movement: fewer launches, coalesced memory access, and relaxed synchronization. That's how the team recovered most of the FP8 overhead.

Brian：第三，最大的性能收益来自减少数据搬运：减少内核启动次数、合并内存访问，以及放宽同步要求。团队就是这样消除了 FP8 的大部分额外开销。

Ava: Great. If you're running AMD Instinct hardware, the improvements are already in the PyTorch stack.

Ava：很好。如果你使用 AMD Instinct 硬件，这些改进已经进入 PyTorch 技术栈了。

Brian: Upgrade TorchAO and TorchTitan, and give the upstream recipes a try. Thanks for listening.

Brian：升级 TorchAO 和 TorchTitan，试试上游提供的方案。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| FP8 | 8 位浮点格式，用于降低矩阵乘法的数据精度并提升吞吐 |
| FNUZ | Finite, No NaN, Unsigned Zero，AMD 使用的 FP8 数值格式变体 |
| e4m3fnuz | AMD 的 FP8 数据类型，最大值为 240，不能编码 NaN 或 Inf |
| TorchAO | PyTorch 的量化与低精度训练库 |
| TorchTitan | PyTorch 的大规模模型训练框架 |
| MoE | Mixture-of-Experts，将 token 路由到部分专家网络的架构 |
| grouped GEMM | 把多个专家的矩阵乘法组合处理的 GEMM 方式 |
| ROCm | AMD GPU 的软件平台和计算栈 |
| Triton | 用于编写高性能 GPU kernel 的编程框架 |
| HBM | High Bandwidth Memory，高带宽显存 |
| LDS | Local Data Share，AMD GPU 上的本地共享存储 |
| FSDP2 | Fully Sharded Data Parallel 的第二代实现 |
| MFU | Model FLOPS Utilization，模型浮点运算利用率 |
| absmax | 张量元素绝对值中的最大值，用于计算量化 scale |
| MXFP8 | 按组存储 scale 与数据的 FP8 缩放策略 |

## 口语表达

| Phrase | 释义 |
|---|---|
| That sounds pretty practical. | 这听起来很实用。 |
| Let's start with... | 我们先从……开始。 |
| What matters about that format? | 这种格式的关键点是什么？ |
| So the first takeaway is simple. | 所以第一个要点很简单。 |
| Before we move to speed... | 在进入性能之前…… |
| Let's get into the expensive part. | 我们来讲最昂贵的部分。 |
| What happened on... ? | 在……上结果怎么样？ |
| Did every attempted optimization work? | 每个尝试的优化都有效吗？ |
| Let's connect the pieces. | 我们把这些部分串起来。 |
| Before we sign off... | 在结束之前…… |
