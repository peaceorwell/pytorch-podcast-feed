# FP8 Training on AMD GPUs with TorchTitan and TorchAO: Upstreaming Performance Improvements

原文：[FP8 Training on AMD GPUs with TorchTitan and TorchAO: Upstreaming Performance Improvements](https://pytorch.org/blog/fp8-training-on-amd-gpus-with-torchtitan-and-torchao-upstreaming-performance-improvements/)

## 摘要

本期讨论 TorchTitan 和 TorchAO 如何让 AMD Instinct GPU 原生支持 FP8 训练，并把 Primus-Turbo 中的优化合入 PyTorch 主线。节目解释了 AMD 的 e4m3fnuz 格式、MoE grouped GEMM，以及通过 Triton 融合减少量化开销的方法。在 Llama3-8B 上，FP8 相比 BF16 吞吐提升 13.4%；在 DeepSeek-V3 671B 上，前向融合恢复了 89% 的 FP8 量化开销。最后介绍了 MXFP8 grouped GEMM 和 MI355X 的后续工作。

## 对话

Ava: If you have AMD Instinct GPUs, FP8 training can now work out of the box in the standard PyTorch stack.

Ava：如果你用的是 AMD Instinct GPU，现在就能在标准 PyTorch 技术栈中直接进行 FP8 训练。

And on Llama3-8B, it delivered a 13. 4% throughput gain over BF16.

在 Llama3-8B 上，吞吐量比 BF16 提高了 13.4%。

That sounds pretty practical.

这听起来很实用。

Brian: Exactly. The key story is upstreaming.

Brian：没错。关键是把优化合入上游项目。

Optimizations first demonstrated with Primus-Turbo were merged into TorchAO and TorchTitan, so you don't need a separate AMD-specific training library.

最早在 Primus-Turbo 中展示的优化已合入 TorchAO 和 TorchTitan，因此无需另装 AMD 专用训练库。

Ava: So this episode is about the path from a big optimization project to normal PyTorch support?

Ava：所以这期要讲的是，一个大型优化项目如何变成 PyTorch 的常规支持？

Brian: Right.

Brian：对。

We'll walk through correctness first, then performance, then the limits and what's next.

我们先讲正确性，再讲性能，最后谈局限和后续方向。

Ava: Let's start with FP8 itself. What changes when a linear layer uses it?

Ava：先说 FP8。线性层使用它后会发生什么变化？

Brian: A linear layer has three matrix multiplications: the forward pass, the gradient input, and the gradient weight update.

Brian：线性层涉及三次矩阵乘法：前向计算、输入梯度计算和权重梯度计算。

FP8 training quantizes these operations from 16 bits to 8 bits, which can improve throughput.

FP8 训练将这些运算从 16 位量化为 8 位，从而提高吞吐量。

Ava: But AMD doesn't use exactly the same FP8 format as NVIDIA, correct?

Ava：但 AMD 使用的 FP8 格式与 NVIDIA 的并不完全相同，对吗？

Brian: Correct. AMD Instinct GPUs use an FNUZ variant.

Brian：对。AMD Instinct GPU 使用 FNUZ 变体。

FNUZ means Finite, No NaN, Unsigned Zero. The specific format here is e4m3fnuz.

FNUZ 指有限值、无 NaN、无符号零；这里的具体格式是 e4m3fnuz。

Ava: What matters about that format?

Ava：这个格式有什么关键之处？

Brian: Its maximum value is 240, and it has no NaN or infinity encodings.

Brian：它的最大值是 240，而且没有 NaN 或无穷大的编码。

It's implemented on MI300X, MI325X, and MI350X GPUs.

MI300X、MI325X 和 MI350X GPU 都支持它。

Ava: Why is choosing that exact format a correctness issue?

Ava：为什么选对格式会影响正确性？

Brian: TorchAO originally calculated scales using a different maximum value, from NVIDIA's e4m3fn format.

Brian：TorchAO 原先按 NVIDIA 的 e4m3fn 格式计算缩放系数，用的是另一个最大值。

On AMD, values could be scaled beyond what e4m3fnuz can represent.

在 AMD 上，缩放后的数值可能超出 e4m3fnuz 的表示范围。

Activations were clipped, gradients were corrupted, and model quality degraded.

激活值被截断，梯度出错，模型质量随之下降。

Ava: And because there are no NaN or infinity encodings, the overflow didn't raise an error?

Ava：因为没有 NaN 或无穷大编码，溢出也不会报错？

Brian: Exactly. It silently produced wrong results. So the team added hardware auto-detection.

Brian：没错。它会悄悄产生错误结果，所以团队加入了硬件自动检测。

TorchAO now selects the correct FP8 dtype and maximum value automatically.

现在 TorchAO 会自动选择正确的 FP8 数据类型和最大值。

Ava: They also fixed the reported peak FLOPS and loss baselines, right?

Ava：他们还修正了报告中的峰值 FLOPS 和损失基线，对吧？

Brian: Yes.

Brian：对。

TorchTitan now reports correct MI300X peak FLOPS for MFU, or Model FLOPS Utilization, numbers.

现在 TorchTitan 会使用正确的 MI300X 峰值 FLOPS 来计算 MFU，也就是模型 FLOPS 利用率。

It also has platform-specific loss baselines for FNUZ numerics.

它还针对 FNUZ 的数值特性设置了平台专属的损失基线。

Ava: Before we move to speed, give us the scaling choices. FP8 still needs scales.

Ava：讲速度之前，先说说缩放方式。FP8 仍然需要缩放系数。

Brian: There are four granularities. Tensorwise uses one scale for the whole tensor.

Brian：一共有四种粒度。Tensorwise 为整个张量使用一个缩放系数。

It's fastest but coarsest. Rowwise uses one scale per row.

它速度最快，但粒度最粗。Rowwise 为每一行使用一个缩放系数。

Blockwise uses one per fixed-size tile.

Blockwise 为每个固定大小的数据块使用一个缩放系数。

MXFP8 uses scales per group packed alongside the data.

MXFP8 按组设置缩放系数，并与数据打包存放。

Ava: And TorchAO and TorchTitan support all four on AMD?

Ava：TorchAO 和 TorchTitan 在 AMD 上支持这四种方式吗？

Brian: Yes.

Brian：支持。

The AMD work made each strategy operate correctly with AMD numerics, and added blockwise kernel support for MI300 and MI350.

这项 AMD 工作让每种策略都能适配 AMD 的数值格式，并为 MI300 和 MI350 增加了 Blockwise 内核支持。

Ava: So the first takeaway is simple: format detection and scaling have to be correct before any benchmark means anything.

Ava：所以第一个结论很简单：格式检测和缩放必须正确，基准测试才有意义。

Brian: That's the right recap.

Brian：总结得对。

Correct dtype, correct maximum value, correct baselines, and correct scaling strategies.

数据类型、最大值、基线和缩放策略都要正确。

Now let's talk about dense models.

接下来聊聊稠密模型。

Ava: The headline result uses Llama3-8B on eight MI300X GPUs?

Ava：主要结果是在八块 MI300X GPU 上运行 Llama3-8B 得到的？

Brian: Yes. With rowwise FP8, throughput increased 13. 4% over BF16.

Brian：对。使用 Rowwise FP8 时，吞吐量比 BF16 提高了 13.4%。

Peak memory was nearly identical, around 39 gigabytes.

峰值显存几乎相同，约为 39 GB。

Ava: So the gain wasn't from saving memory.

Ava：所以性能提升并非来自节省显存。

Brian: Right. The recipe keeps the weight-update GEMM in BF16 for higher precision.

Brian：对。这套方案为保证较高精度，让权重梯度计算的 GEMM 保持 BF16。

The forward and gradient-input GEMMs use FP8.

前向计算和输入梯度计算的 GEMM 则使用 FP8。

The speedup comes from faster FP8 matrix cores.

提速来自更快的 FP8 矩阵计算核心。

Ava: The setup also used torch. compile, FSDP2, and selective activation checkpointing?

Ava：测试配置还用了 torch.compile、FSDP2 和选择性激活检查点？

Brian: Correct. FSDP means Fully Sharded Data Parallel.

Brian：对。FSDP 是完全分片数据并行。

The reported run used batch size one, sequence length eight thousand one hundred ninety-two, and one hundred steps.

报告中的测试批大小为 1，序列长度为 8192，运行了 100 步。

Ava: Dense models sound relatively straightforward. Why are Mixture-of-Experts models harder?

Ava：稠密模型听起来比较简单。混合专家模型为什么更难？

Brian: MoE models route each token to a subset of experts. That creates variable-size batches.

Brian：MoE 模型会把每个 token 分配给部分专家，因此批次大小不固定。

Instead of one uniform GEMM, you need a grouped GEMM.

这时需要分组 GEMM，而不是一次统一的 GEMM。

Ava: Grouped GEMM means several expert matrix multiplications handled together?

Ava：分组 GEMM 是把多个专家的矩阵乘法一起处理吗？

Brian: Exactly.

Brian：没错。

It needs per-row activation scales, per-expert-column weight scales, and an offset tensor that routes rows to the right expert.

它需要逐行的激活缩放因子、每个专家逐列的权重缩放因子，以及把各行分配给对应专家的偏移张量。

Ava: And the team enabled FP8 grouped GEMM on ROCm through the Composable Kernel backend?

Ava：团队还通过 Composable Kernel 后端，在 ROCm 上启用了 FP8 分组 GEMM？

Brian: Yes.

Brian：对。

They adapted the quantization pipeline to use the correct AMD dtype and dispatch path through Composable Kernel.

他们调整了量化流程，使用正确的 AMD 数据类型，并通过 Composable Kernel 走对应的分派路径。

Ava: Now let's get into the expensive part: quantization overhead.

Ava：接下来说说开销大的部分：量化开销。

Brian: The basic FP8 pipeline has several steps.

Brian：基本的 FP8 量化流程有几步。

First, compute the absolute maximum, or absmax, per row or column.

首先，计算每行或每列的绝对值最大值，也就是 absmax。

Then derive a scale and apply it. Finally, clamp and cast to FP8.

然后算出缩放因子并应用，最后截断数值并转换为 FP8。

Ava: Each step used to be a separate kernel launch?

Ava：以前每一步都要单独启动一个内核？

Brian: Yes.

Brian：对。

Each launch could materialize an intermediate tensor in HBM, or High Bandwidth Memory.

每次启动都可能在 HBM，也就是高带宽内存中，生成一个中间张量。

With dozens of expert weight tensors, those round-trips dominated the overhead.

专家权重张量有几十个，这些数据往返搬运成了主要开销。

Ava: So the problem was memory movement, not the eight-bit arithmetic itself.

Ava：所以问题出在内存搬运，而不是八位运算本身。

Brian: Exactly. On these MoE shapes, the math was cheap.

Brian：没错。对这些 MoE 张量形状来说，计算本身开销很小。

Kernel launches and HBM traffic were expensive.

内核启动和 HBM 数据传输才费时。

The optimizations attacked data movement at three levels.

优化从三个层面减少了数据搬运。

Ava: Level one was launching fewer kernels. What happened in the backward pass?

Ava：第一层是减少内核启动次数。反向传播中改了什么？

Brian: There were two issues.

Brian：有两个问题。

A transpose pattern, written as transpose, contiguous, transpose, forced a full tensor copy through HBM.

一种依次执行 transpose、contiguous、transpose 的转置操作，会迫使整个张量经过 HBM 复制一遍。

Pull request number 3972 removed those redundant copies.

3972 号拉取请求去掉了这些多余的复制。

Ava: And the scale-and-cast chain was still split across kernels.

Ava：缩放和类型转换这条处理链仍分散在多个内核里。

Brian: Right. Pull request 4069 fused that chain into single Triton kernels in several places.

Brian：对。4069 号拉取请求在多处把它融合成单个 Triton 内核。

It also added a companion dual-kernel for simultaneous gradient-output and activation quantization.

还增加了一个配套的双重内核，同时量化输出梯度和激活值。

Ava: What was the measured effect?

Ava：实测效果如何？

Brian: On eight MI300X GPUs with DeepSeek-MoE-16B, the backward fusions delivered a 4.

Brian：在八张 MI300X GPU 上运行 DeepSeek-MoE-16B 时，反向传播中的融合带来了 4.

2 times throughput improvement for the backward pass.

2 倍的反向传播吞吐量提升。

Ava: That's a large jump. Did the forward path have the same pattern?

Ava：提升很大。前向传播也有同样的问题吗？

Brian: Yes. Quantizing expert weights launched five generic kernels per call.

Brian：有。每次量化专家权重，都会启动五个通用内核。

There were 24 calls per step, adding about 90 milliseconds per step.

每一步要调用 24 次，增加约 90 毫秒。

Ava: And the fix was one fused Triton kernel?

Ava：解决办法是融合成一个 Triton 内核？

Brian: Exactly. Pull request 4311 replaced the chain with one fused kernel.

Brian：没错。4311 号拉取请求用一个融合内核替换了整条处理链。

It parallelized across both experts and output-dimension blocks, so the surrounding GEMMs could issue sooner.

它同时按专家和输出维度的数据块并行处理，让前后的 GEMM 能更早启动。

Ava: What happened on DeepSeek-V3 671B?

Ava：在 DeepSeek-V3 671B 上效果如何？

Brian: On eight MI325X GPUs, end-to-end throughput rose 17%, from 5,996 to 7,027 tokens per second.

Brian：在八张 MI325X GPU 上，端到端吞吐量提高了 17%，从每秒 5,996 个 token 增至 7,027 个。

The forward portion went from about 19 milliseconds to about 7 milliseconds.

前向传播耗时从约 19 毫秒降至约 7 毫秒。

Ava: The BF16 baseline was 7,156 tokens per second, so FP8 got quite close.

Ava：BF16 基线是每秒 7,156 个 token，FP8 已经很接近了。

Brian: Yes. The fused forward kernel recovered 89% of the BF16-to-FP8 gap.

Brian：对。融合后的前向内核弥合了 FP8 与 BF16 吞吐量差距的 89%。

The upstream FP8 configuration had added 127 milliseconds per step, and 92% of that was in generic quantization kernels.

上游 FP8 配置让每步多花 127 毫秒，其中 92% 耗在通用量化内核上。

Fusion removed most of it.

融合消除了其中大部分开销。

Ava: Level two was making each remaining kernel move memory efficiently.

Ava：第二层是让剩下的每个内核更高效地搬运数据。

What was wrong with the colwise scales kernel?

逐列缩放因子内核有什么问题？

Brian: Its writes were non-coalesced.

Brian：它的写入没有合并成连续的内存访问。

Consecutive SIMD lanes wrote to addresses separated by K bytes, causing separate memory transactions.

相邻的 SIMD 通道写入的地址相隔 K 字节，导致内存事务分散。

Ava: How did they fix that?

Ava：他们怎么解决的？

Brian: They transposed the output tile through LDS, or Local Data Share, before storing it.

Brian：存储前，先通过 LDS，也就是局部数据共享内存，转置输出数据块。

They also added a fused single-pass variant that removed a redundant HBM read.

他们还加入了融合的单遍版本，省去一次多余的 HBM 读取。

Ava: And the per-layer result?

Ava：单层效果呢？

Brian: On MI300X with DeepSeek-V3 671B shapes, one MoE layer dropped from 7,290 microseconds to 1,170 microseconds.

Brian：在 MI300X 上，针对 DeepSeek-V3 671B 的张量形状，单个 MoE 层的耗时从 7,290 微秒降至 1,170 微秒。

That's a 6. 2 times speedup.

相当于提速 6.2 倍。

Ava: Level three removed synchronization that the hardware didn't need.

Ava：第三阶段去掉了硬件并不需要的同步。

Brian: Right.

Brian：对。

Triton's atomic_add, atomic_max, and atomic_min defaulted to acquire-release ordering.

Triton 的 atomic_add、atomic_max 和 atomic_min 默认采用获取-释放内存序。

On AMD, that inserted memory fences before and after every atomic operation.

在 AMD 上，这会在每次原子操作前后插入内存栅栏。

Ava: Those fences are unnecessary for commutative reductions?

Ava：可交换归约不需要这些栅栏？

Brian: In this case, yes. The team switched to relaxed ordering on AMD, guarded by a torch.

Brian：这个场景下不需要。团队在 AMD 上改用宽松内存序，并用 torch

version. hip check. NVIDIA behavior stayed unchanged.

的 version.hip 检查限定这一改动。NVIDIA 的行为保持不变。

Ava: Did every attempted optimization work?

Ava：每项优化尝试都奏效了吗？

Brian: No.

Brian：没有。

They expanded the Triton autotune search space from one to eight or sixteen candidate configurations, hoping to find better tile sizes for AMD wavefronts.

他们把 Triton 自动调优的搜索范围从一种配置扩大到八种或十六种，希望找到更适合 AMD wavefront 的分块大小。

Ava: But Llama 4 shapes on MI300X showed no measurable improvement.

Ava：但在 MI300X 上，Llama 4 的张量形状没有出现可测量的提升。

Brian: Correct.

Brian：没错。

The extra configurations increased first-iteration compile time, so the change was reverted.

额外配置增加了首轮编译时间，所以这项改动被撤销了。

The lesson was to shape autotuning around wavefront size, LDS capacity, and register pressure, rather than adding candidates by default.

经验是按 wavefront 大小、LDS 容量和寄存器压力设计自动调优，而不是默认增加候选配置。

Ava: Let's connect the pieces. What are the main results across workloads?

Ava：把这些串起来看，各类工作负载的主要结果是什么？

Brian: For Llama3-8B dense training, rowwise FP8 gave 13. 4% more throughput than BF16.

Brian：在 Llama3-8B 稠密模型训练中，按行 FP8 的吞吐量比 BF16 高 13.4%。

For DeepSeek-MoE-16B, backward fusion reached 4. 2 times the backward throughput.

对于 DeepSeek-MoE-16B，反向融合使反向传播吞吐量达到原来的 4.2 倍。

For DeepSeek-V3 671B, colwise scale optimization reached 6.

对于 DeepSeek-V3 671B，按列缩放因子优化使单个 MoE 层提速 6.

2 times per MoE layer, and forward fusion improved end-to-end throughput by 17%.

2 倍；前向融合还使端到端吞吐量提高了 17%。

Ava: And all of this is now upstream?

Ava：这些改进都已合入上游了吗？

Brian: Yes. The contributions are merged into mainline pytorch/ao and pytorch/torchtitan.

Brian：是的。相关贡献已合入 pytorch/ao 和 pytorch/torchtitan 的主线。

Teams with AMD Instinct GPUs can get the gains by upgrading TorchAO and TorchTitan, with nothing AMD-specific to install.

使用 AMD Instinct GPU 的团队只需升级 TorchAO 和 TorchTitan，就能获得这些性能提升，无需安装 AMD 专用组件。

Ava: What's next?

Ava：接下来呢？

Brian: Work is continuing on MXFP8 grouped GEMM and quantization kernels for forward and backward passes on MI355X GPUs.

Brian：团队正继续为 MI355X GPU 开发前向和反向计算所需的 MXFP8 分组 GEMM 与量化内核。

More Triton optimizations are also coming along the fusion pipeline.

融合流程中还会加入更多 Triton 优化。

Ava: Before we sign off, let's do the promised three-point recap.

Ava：结束前，按约定用三点回顾一下。

Brian: First, AMD FP8 correctness depends on hardware-aware FNUZ support, automatic dtype selection, and accurate loss and MFU baselines.

Brian：第一，AMD 上 FP8 的正确性依赖适配硬件的 FNUZ 支持、自动选择数据类型，以及准确的损失值和 MFU 基线。

Ava: Second, MoE models need grouped GEMM with per-row scales, per-expert-column scales, and offset-based routing.

Ava：第二，MoE 模型需要支持逐行缩放因子、逐专家逐列缩放因子和基于偏移量路由的分组 GEMM。

Brian: Third, the biggest performance wins came from reducing data movement: fewer launches, coalesced memory access, and relaxed synchronization.

Brian：第三，最大的性能提升来自减少数据搬运：减少内核启动次数、合并内存访问，以及放宽同步要求。

That's how the team recovered most of the FP8 overhead.

团队就这样抵消了 FP8 的大部分额外开销。

Ava: Great.

Ava：很好。

If you're running AMD Instinct hardware, the improvements are already in the PyTorch stack.

如果你在使用 AMD Instinct 硬件，这些改进已经进入 PyTorch 技术栈。

Brian: Upgrade TorchAO and TorchTitan, and give the upstream recipes a try.

Brian：升级 TorchAO 和 TorchTitan，试试上游的配置方案。

Thanks for listening.

感谢收听。

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
