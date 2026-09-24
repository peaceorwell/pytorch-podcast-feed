# FP8 Training on AMD GPUs with TorchTitan and TorchAO: Upstreaming Performance Improvements

原文：[FP8 Training on AMD GPUs with TorchTitan and TorchAO: Upstreaming Performance Improvements](https://pytorch.org/blog/fp8-training-on-amd-gpus-with-torchtitan-and-torchao-upstreaming-performance-improvements/)

## 摘要

本期讨论 TorchTitan 和 TorchAO 如何让 AMD Instinct GPU 原生支持 FP8 训练，并把 Primus-Turbo 中的优化合入 PyTorch 主线。节目解释了 AMD 的 e4m3fnuz 格式、MoE grouped GEMM，以及通过 Triton 融合减少量化开销的方法。在 Llama3-8B 上，FP8 相比 BF16 吞吐提升 13.4%；在 DeepSeek-V3 671B 上，前向融合恢复了 89% 的 FP8 量化开销。最后介绍了 MXFP8 grouped GEMM 和 MI355X 的后续工作。

## 对话

**Ava:** If you have AMD Instinct GPUs, FP8 training can now work out of the box in the standard PyTorch stack. And on Llama3-8B, it delivered a 13.4% throughput gain over BF16. That sounds pretty practical.

**Brian:** Exactly. The key story is upstreaming. Optimizations first demonstrated with Primus-Turbo were merged into TorchAO and TorchTitan, so you don't need a separate AMD-specific training library.

**Ava:** So this episode is about the path from a big optimization project to normal PyTorch support?

**Brian:** Right. We'll walk through correctness first, then performance, then the limits and what's next.

**Ava:** Let's start with FP8 itself. What changes when a linear layer uses it?

**Brian:** A linear layer has three matrix multiplications: the forward pass, the gradient input, and the gradient weight update. FP8 training quantizes these operations from 16 bits to 8 bits, which can improve throughput.

**Ava:** But AMD doesn't use exactly the same FP8 format as NVIDIA, correct?

**Brian:** Correct. AMD Instinct GPUs use an FNUZ variant. FNUZ means Finite, No NaN, Unsigned Zero. The specific format here is e4m3fnuz.

**Ava:** What matters about that format?

**Brian:** Its maximum value is 240, and it has no NaN or infinity encodings. It's implemented on MI300X, MI325X, and MI350X GPUs.

**Ava:** Why is choosing that exact format a correctness issue?

**Brian:** TorchAO originally calculated scales using a different maximum value, from NVIDIA's e4m3fn format. On AMD, values could be scaled beyond what e4m3fnuz can represent. Activations were clipped, gradients were corrupted, and model quality degraded.

**Ava:** And because there are no NaN or infinity encodings, the overflow didn't raise an error?

**Brian:** Exactly. It silently produced wrong results. So the team added hardware auto-detection. TorchAO now selects the correct FP8 dtype and maximum value automatically.

**Ava:** They also fixed the reported peak FLOPS and loss baselines, right?

**Brian:** Yes. TorchTitan now reports correct MI300X peak FLOPS for MFU, or Model FLOPS Utilization, numbers. It also has platform-specific loss baselines for FNUZ numerics.

**Ava:** Before we move to speed, give us the scaling choices. FP8 still needs scales.

**Brian:** There are four granularities. Tensorwise uses one scale for the whole tensor. It's fastest but coarsest. Rowwise uses one scale per row. Blockwise uses one per fixed-size tile. MXFP8 uses scales per group packed alongside the data.

**Ava:** And TorchAO and TorchTitan support all four on AMD?

**Brian:** Yes. The AMD work made each strategy operate correctly with AMD numerics, and added blockwise kernel support for MI300 and MI350.

**Ava:** So the first takeaway is simple: format detection and scaling have to be correct before any benchmark means anything.

**Brian:** That's the right recap. Correct dtype, correct maximum value, correct baselines, and correct scaling strategies. Now let's talk about dense models.

**Ava:** The headline result uses Llama3-8B on eight MI300X GPUs?

**Brian:** Yes. With rowwise FP8, throughput increased 13.4% over BF16. Peak memory was nearly identical, around 39 gigabytes.

**Ava:** So the gain wasn't from saving memory.

**Brian:** Right. The recipe keeps the weight-update GEMM in BF16 for higher precision. The forward and gradient-input GEMMs use FP8. The speedup comes from faster FP8 matrix cores.

**Ava:** The setup also used torch.compile, FSDP2, and selective activation checkpointing?

**Brian:** Correct. FSDP means Fully Sharded Data Parallel. The reported run used batch size one, sequence length eight thousand one hundred ninety-two, and one hundred steps.

**Ava:** Dense models sound relatively straightforward. Why are Mixture-of-Experts models harder?

**Brian:** MoE models route each token to a subset of experts. That creates variable-size batches. Instead of one uniform GEMM, you need a grouped GEMM.

**Ava:** Grouped GEMM means several expert matrix multiplications handled together?

**Brian:** Exactly. It needs per-row activation scales, per-expert-column weight scales, and an offset tensor that routes rows to the right expert.

**Ava:** And the team enabled FP8 grouped GEMM on ROCm through the Composable Kernel backend?

**Brian:** Yes. They adapted the quantization pipeline to use the correct AMD dtype and dispatch path through Composable Kernel.

**Ava:** Now let's get into the expensive part: quantization overhead.

**Brian:** The basic FP8 pipeline has several steps. First, compute the absolute maximum, or absmax, per row or column. Then derive a scale and apply it. Finally, clamp and cast to FP8.

**Ava:** Each step used to be a separate kernel launch?

**Brian:** Yes. Each launch could materialize an intermediate tensor in HBM, or High Bandwidth Memory. With dozens of expert weight tensors, those round-trips dominated the overhead.

**Ava:** So the problem was memory movement, not the eight-bit arithmetic itself.

**Brian:** Exactly. On these MoE shapes, the math was cheap. Kernel launches and HBM traffic were expensive. The optimizations attacked data movement at three levels.

**Ava:** Level one was launching fewer kernels. What happened in the backward pass?

**Brian:** There were two issues. A transpose pattern, written as transpose, contiguous, transpose, forced a full tensor copy through HBM. Pull request number 3972 removed those redundant copies.

**Ava:** And the scale-and-cast chain was still split across kernels.

**Brian:** Right. Pull request 4069 fused that chain into single Triton kernels in several places. It also added a companion dual-kernel for simultaneous gradient-output and activation quantization.

**Ava:** What was the measured effect?

**Brian:** On eight MI300X GPUs with DeepSeek-MoE-16B, the backward fusions delivered a 4.2 times throughput improvement for the backward pass.

**Ava:** That's a large jump. Did the forward path have the same pattern?

**Brian:** Yes. Quantizing expert weights launched five generic kernels per call. There were 24 calls per step, adding about 90 milliseconds per step.

**Ava:** And the fix was one fused Triton kernel?

**Brian:** Exactly. Pull request 4311 replaced the chain with one fused kernel. It parallelized across both experts and output-dimension blocks, so the surrounding GEMMs could issue sooner.

**Ava:** What happened on DeepSeek-V3 671B?

**Brian:** On eight MI325X GPUs, end-to-end throughput rose 17%, from 5,996 to 7,027 tokens per second. The forward portion went from about 19 milliseconds to about 7 milliseconds.

**Ava:** The BF16 baseline was 7,156 tokens per second, so FP8 got quite close.

**Brian:** Yes. The fused forward kernel recovered 89% of the BF16-to-FP8 gap. The upstream FP8 configuration had added 127 milliseconds per step, and 92% of that was in generic quantization kernels. Fusion removed most of it.

**Ava:** Level two was making each remaining kernel move memory efficiently. What was wrong with the colwise scales kernel?

**Brian:** Its writes were non-coalesced. Consecutive SIMD lanes wrote to addresses separated by K bytes, causing separate memory transactions.

**Ava:** How did they fix that?

**Brian:** They transposed the output tile through LDS, or Local Data Share, before storing it. They also added a fused single-pass variant that removed a redundant HBM read.

**Ava:** And the per-layer result?

**Brian:** On MI300X with DeepSeek-V3 671B shapes, one MoE layer dropped from 7,290 microseconds to 1,170 microseconds. That's a 6.2 times speedup.

**Ava:** Level three removed synchronization that the hardware didn't need.

**Brian:** Right. Triton's atomic_add, atomic_max, and atomic_min defaulted to acquire-release ordering. On AMD, that inserted memory fences before and after every atomic operation.

**Ava:** Those fences are unnecessary for commutative reductions?

**Brian:** In this case, yes. The team switched to relaxed ordering on AMD, guarded by a torch.version.hip check. NVIDIA behavior stayed unchanged.

**Ava:** Did every attempted optimization work?

**Brian:** No. They expanded the Triton autotune search space from one to eight or sixteen candidate configurations, hoping to find better tile sizes for AMD wavefronts.

**Ava:** But Llama 4 shapes on MI300X showed no measurable improvement.

**Brian:** Correct. The extra configurations increased first-iteration compile time, so the change was reverted. The lesson was to shape autotuning around wavefront size, LDS capacity, and register pressure, rather than adding candidates by default.

**Ava:** Let's connect the pieces. What are the main results across workloads?

**Brian:** For Llama3-8B dense training, rowwise FP8 gave 13.4% more throughput than BF16. For DeepSeek-MoE-16B, backward fusion reached 4.2 times the backward throughput. For DeepSeek-V3 671B, colwise scale optimization reached 6.2 times per MoE layer, and forward fusion improved end-to-end throughput by 17%.

**Ava:** And all of this is now upstream?

**Brian:** Yes. The contributions are merged into mainline pytorch/ao and pytorch/torchtitan. Teams with AMD Instinct GPUs can get the gains by upgrading TorchAO and TorchTitan, with nothing AMD-specific to install.

**Ava:** What's next?

**Brian:** Work is continuing on MXFP8 grouped GEMM and quantization kernels for forward and backward passes on MI355X GPUs. More Triton optimizations are also coming along the fusion pipeline.

**Ava:** Before we sign off, let's do the promised three-point recap.

**Brian:** First, AMD FP8 correctness depends on hardware-aware FNUZ support, automatic dtype selection, and accurate loss and MFU baselines.

**Ava:** Second, MoE models need grouped GEMM with per-row scales, per-expert-column scales, and offset-based routing.

**Brian:** Third, the biggest performance wins came from reducing data movement: fewer launches, coalesced memory access, and relaxed synchronization. That's how the team recovered most of the FP8 overhead.

**Ava:** Great. If you're running AMD Instinct hardware, the improvements are already in the PyTorch stack.

**Brian:** Upgrade TorchAO and TorchTitan, and give the upstream recipes a try. Thanks for listening.

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
