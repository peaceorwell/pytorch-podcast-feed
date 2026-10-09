# PyTorch 2.13: Faster Kernels, Leaner Training, and Better Distributed Systems

原文：[PyTorch 2.13 Release Blog](https://pytorch.org/blog/pytorch-2-13-release-blog/)

## 摘要

PyTorch 2.13 继续把框架推向面向生产、跨硬件、可大规模训练和推理的平台。它为 Apple Silicon 带来 FlexAttention，为 CUDA 提供确定性反向路径，并通过 CuTeDSL 和原生 Metal kernel 改善性能。大词表训练可使用融合的 nn.LinearCrossEntropyLoss，将峰值显存最多降低约四倍；分布式方面新增 torchcomms，并让 FSDP2 的 all-gather 与 reduce-scatter 重叠。该版本也支持 Linux 上的 Python 3.15 wheel，但仍有 beta、平台覆盖和 API unstable 等限制。

## 对话

Ava: If you're training large language models or running PyTorch on Apple Silicon, this release could change your daily work.

Ava：如果你在训练大语言模型，或在 Apple Silicon 上运行 PyTorch，这次发布可能会改变你的日常工作。

PyTorch 2.

PyTorch 2 点

13 is out, and it targets speed, memory, and cluster reliability at the same time.

13 已发布，同时着力提升速度、内存效率和集群可靠性。

Brian: Right.

Brian：没错。

The headline is that PyTorch keeps moving from a research-first framework toward a unified, hardware-agnostic platform for production training and inference at scale.

重点是，PyTorch 正从以研究为主的框架，走向支持大规模生产训练和推理的统一跨硬件平台。

Ava: So let's start with the biggest practical wins. What would an engineer notice first?

Ava：先说最实用的改进。工程师会最先注意到什么？

Brian: There are four big areas.

Brian：主要有四个方面。

FlexAttention expands to Apple Silicon and gets deterministic backward on CUDA.

FlexAttention 扩展到 Apple Silicon，并在 CUDA 上支持确定性反向传播。

CuTeDSL gives Inductor another high-performance GPU path.

CuTeDSL 为 Inductor 提供了另一条高性能 GPU 路径。

A fused linear-plus-loss operation cuts memory for huge vocabularies.

融合的线性层与损失计算可降低超大词表的内存占用。

And distributed training gets torchcomms plus better communication overlap in FSDP2.

分布式训练则加入了 torchcomms，并改进了 FSDP2 中通信与计算的重叠执行。

Ava: Let's unpack FlexAttention first.

Ava：先来看看 FlexAttention。

People often hear “custom attention” and think they need to write a CUDA kernel.

一听到“自定义注意力”，很多人就以为得自己写 CUDA 内核。

Brian: That is exactly what PyTorch is trying to avoid.

Brian：PyTorch 正是想省去这一步。

FlexAttention is a unified API where you describe an attention pattern as a plain Python function.

FlexAttention 提供统一的 API，让你用普通 Python 函数描述注意力模式。

The compiler turns that description into a fused kernel. In 2.

编译器会据此生成融合内核。在 2 点

13, it works on Metal, also called MPS, Apple's backend.

13 版中，它支持 Metal，也就是苹果的 MPS 后端。

Ava: So a Mac user can express sparse attention without hand-writing Metal code?

Ava：也就是说，Mac 用户不用手写 Metal 代码，也能实现稀疏注意力？

Brian: Yes.

Brian：对。

The MPS implementation includes hand-written Metal kernels for sparse prefill and decode paths, including grouped-query attention, or GQA, and captured buffers.

MPS 实现包含手写的 Metal 内核，用于稀疏预填充和解码，支持分组查询注意力（GQA）及捕获缓冲区。

The article describes the user experience as writing a two-line Python function while the compiler builds the fast kernel.

文章说，用户只需写两行 Python 函数，编译器就会生成高速内核。

Ava: What kind of speed are we talking about?

Ava：速度能快多少？

Brian: For sparse masks, the numbers are substantial.

Brian：对于稀疏掩码，提升相当明显。

On a one-by-eight-by-thirty-two-thousand-seven-hundred-sixty-eight-by-sixty-four shape, with a two-hundred-fifty-six-element sliding window and zero point eight percent density, FlexAttention runs in about thirty-five milliseconds.

在 1×8×32768×64 的形状下，使用 256 元素的滑动窗口、密度为 0.8% 时，FlexAttention 耗时约 35 毫秒。

SDPA, scaled dot-product attention, takes about four hundred thirty-one milliseconds.

SDPA，也就是缩放点积注意力，耗时约 431 毫秒。

That's roughly twelve point three times faster.

速度大约快 12.3 倍。

Ava: And a shorter sequence?

Ava：较短的序列呢？

Brian: An eight-thousand-one-hundred-ninety-two length with a sixty-four-window case reaches about four point one five times faster.

Brian：序列长度为 8192、窗口大小为 64 时，速度约快 4.15 倍。

But dense patterns still favor SDPA, so the gain depends heavily on sparsity.

但密集模式下 SDPA 仍更快，所以收益很大程度上取决于稀疏程度。

Ava: That's a useful limitation. What changed on CUDA?

Ava：这个限制很有参考价值。CUDA 上有什么变化？

Brian: The existing FlexAttention flash backend now has a deterministic backward path.

Brian：现有的 FlexAttention Flash 后端现在有了确定性反向传播路径。

By default, it used atomic operations to accumulate dQ, which could make repeated runs produce slightly different gradients.

此前默认用原子操作累加 dQ，重复运行可能产生略有差异的梯度。

Ava: That sounds painful for debugging and regression tests.

Ava：这会让调试和回归测试很头疼。

Brian: Exactly.

Brian：正是。

The new compute_dq_write_order path precomputes a write order instead of using atomics.

新的 compute_dq_write_order 路径会预先计算写入顺序，不再使用原子操作。

It gives bit-for-bit reproducible gradients without a meaningful performance penalty.

它能让梯度逐位一致，且几乎没有性能损失。

The measured end-to-end overhead on create_block_mask is well under one percent at longer sequence lengths, with zero point two percent at a sequence length of thirty-two-thousand-seven-hundred-sixty-eight.

在较长序列下，create_block_mask 的实测端到端开销远低于 1%；序列长度为 32768 时仅为 0.2%。

Ava: How do users enable it?

Ava：用户怎么开启？

Brian: They opt in with the existing torch. use_deterministic_algorithms setting.

Brian：通过现有的 torch.use_deterministic_algorithms 设置选择开启。

No extra code changes are required. The API is marked unstable, though.

无需修改其他代码。不过，这个 API 仍标记为不稳定。

Ava: Before we leave Apple Silicon, there is also a larger native Metal migration.

Ava：说完 Apple Silicon 之前，还得提到更大范围的原生 Metal 迁移。

Brian: Yes.

Brian：没错。

Many operations previously went through MPSGraph, which adds compilation and scheduling overhead on each dispatch.

许多操作以前要经过 MPSGraph，每次分发都会增加编译和调度开销。

PyTorch now moves a broad set to hand-written Metal compute kernels.

PyTorch 现在把大量操作迁移到了手写的 Metal 计算内核。

Ava: Which operations?

Ava：包括哪些操作？

Brian: Copy and cast, random-number generation, comparisons, reductions like sum and mean, cumulative operations, sorting, embedding backward, and scatter or gather with bounds checking.

Brian：复制和类型转换、随机数生成、比较、求和与均值等归约、累积运算、排序、嵌入层反向传播，以及带边界检查的 scatter 和 gather。

Direct control over thread dispatch and memory access should reduce kernel launch latency for common training and inference workloads.

直接控制线程分发和内存访问，应能降低常见训练和推理任务的内核启动延迟。

Ava: Now let's talk about memory, because large-vocabulary models can burn through it before the real computation begins.

Ava：接着谈谈内存。大词表模型还没开始主要计算，内存就可能耗尽。

Brian: The new nn. LinearCrossEntropyLoss addresses that.

Brian：新的 nn.LinearCrossEntropyLoss 就是为此设计的。

In the standard path, the final linear projection creates the full logits matrix over the vocabulary.

在标准计算路径中，最后的线性投影会生成覆盖整个词表的 logits 矩阵。

With vocabularies above one hundred thousand tokens, that matrix can consume tens of gigabytes of GPU memory.

词表超过 10 万个 token 时，这个矩阵可能占用数十 GB 的 GPU 显存。

Ava: And the fused operation avoids materializing that matrix?

Ava：融合运算就不用实际生成这个矩阵了？

Brian: Right.

Brian：对。

It combines the final linear projection and cross-entropy computation, processing the vocabulary dimension in chunks.

它把最后的线性投影和交叉熵计算合并，分块处理词表维度。

Peak memory drops by up to about four times for large-vocabulary workloads, while keeping numerical equivalence with the unfused path.

对于大词表任务，峰值显存最多可降至约四分之一，同时保持与非融合路径在数值上等价。

Ava: Does it support the features people already use?

Ava：它支持大家已经在用的功能吗？

Brian: It supports label smoothing, weight tying, and z-loss regularization out of the box.

Brian：开箱即用地支持标签平滑、权重绑定和 z-loss 正则化。

It also works with torch. compile. Since it replaces separate nn. Linear and nn.

它也兼容 torch.compile。由于它取代了独立的 nn.Linear 和 nn.

CrossEntropyLoss modules, adoption requires no other code changes.

CrossEntropyLoss 模块，接入时无需改动其他代码。

Ava: That sounds unusually convenient for a low-level optimization.

Ava：作为底层优化，这用起来还真方便。

Brian: Yes, although the API is still unstable. PyTorch 2. 13 also makes torch.

Brian：是的，不过 API 仍不稳定。PyTorch 2.13 还让 torch.

load detect safetensors files natively.

load 能原生识别 safetensors 文件。

Safetensors is popular because it supports memory-mapped loading and avoids arbitrary code execution risk.

Safetensors 很受欢迎，因为它支持内存映射加载，还能避免执行任意代码的风险。

You no longer need a separate library just to load those weights.

现在加载这类权重，不必再单独安装一个库。

Ava: Let's move to distributed training. What problem is torchcomms solving?

Ava：聊聊分布式训练吧。torchcomms 要解决什么问题？

Brian: PyTorch Distributed historically relied on c10d's ProcessGroup abstraction for collectives such as all-reduce and all-gather.

Brian：PyTorch Distributed 过去依赖 c10d 的 ProcessGroup 抽象来执行 all-reduce、all-gather 等集合通信。

It was designed mainly around NCCL.

它主要是围绕 NCCL 设计的。

As training uses more dimensions, elastic scaling, and mixed interconnects, error handling and observability become bottlenecks.

随着训练采用更多并行维度、弹性扩缩容和混合互连，错误处理与可观测性成了瓶颈。

Ava: So torchcomms is a new backend?

Ava：所以 torchcomms 是一个新后端？

Brian: Yes. It is integrated into PyTorch Distributed's CI and device-mesh paths.

Brian：对。它已集成到 PyTorch Distributed 的 CI 和设备网格路径中。

It aims to improve fault tolerance through graceful timeouts and partial-group recovery, scalability across large clusters, and debugging through structured logging and collective tracing.

它希望通过平稳处理超时和恢复部分通信组来提高容错能力，同时支持大规模集群扩展，并用结构化日志和集合通信追踪帮助调试。

It keeps API compatibility with the existing c10d backends.

它与现有 c10d 后端保持 API 兼容。

Ava: And FSDP2 gets communication overlap?

Ava：FSDP2 也能让通信重叠执行了？

Brian: FSDP means Fully Sharded Data Parallel.

Brian：FSDP 是“完全分片数据并行”的缩写。

By default, all-gather and reduce-scatter share one NCCL communicator.

默认情况下，all-gather 和 reduce-scatter 共用一个 NCCL 通信器。

NCCL serializes operations on the same communicator, so the two collectives cannot overlap.

NCCL 会串行执行同一通信器上的操作，因此这两种集合通信无法重叠。

Ava: Which leaves bandwidth unused.

Ava：这样就浪费了带宽。

Brian: Exactly. FSDPModule.

Brian：没错。FSDPModule.

set_separate_reduce_scatter_group enables a dedicated communicator for reduce-scatter.

set_separate_reduce_scatter_group 可以为 reduce-scatter 启用专用通信器。

Then all-gather and reduce-scatter can progress concurrently.

这样 all-gather 和 reduce-scatter 就能同时进行。

That creates AG/RS overlap and improves throughput for fully sharded workloads without model-code changes.

这实现了 AG/RS 重叠，无需修改模型代码就能提高完全分片任务的吞吐量。

It is opt-in, and the API is unstable.

这个功能需要主动启用，而且 API 仍不稳定。

Ava: What does PyTorch 2. 13 add on the compiler side?

Ava：PyTorch 2.13 在编译器方面增加了什么？

Brian: There is torch. compiler. set_default_backend.

Brian：增加了 torch.compiler.set_default_backend。

Before this, a custom or out-of-tree backend had to be passed explicitly to every torch.

以前，自定义或树外后端必须显式传给每次 torch.

compile call.

compile 调用。

Now infrastructure teams can set a process-wide default, while an explicit backend argument still overrides it.

现在，基础设施团队可以设置进程级默认后端；显式指定的 backend 参数仍会覆盖它。

Ava: And on NVIDIA GPUs, CuTeDSL is the other major compiler story.

Ava：在 NVIDIA GPU 上，CuTeDSL 也是编译器方面的一大进展。

Brian: CuTeDSL is a Python-native domain-specific language built on NVIDIA's CuTe, or CUDA Templates, library.

Brian：CuTeDSL 是基于 NVIDIA 的 CuTe（CUDA Templates）库构建的 Python 原生领域专用语言。

It gives direct control over tensor layouts, tiling, and memory access.

它能直接控制张量布局、分块和内存访问。

Inductor can use it alongside Triton for matrix multiplication, also called GEMM, and RMSNorm.

Inductor 可以让它与 Triton 一起用于矩阵乘法（也称 GEMM）和 RMSNorm。

Ava: So Inductor now has a second code-generation route for key transformer kernels.

Ava：所以，对于关键的 Transformer 算子，Inductor 现在多了一条代码生成路径。

Brian: Right. The Quack-derived overrides aim for higher-quality matrix-multiply code.

Brian：对。源自 Quack 的覆盖实现旨在生成质量更高的矩阵乘法代码。

Compilation also moved from a thread pool to a subprocess pool, reducing the Python GIL, or Global Interpreter Lock, bottleneck and improving compile-time parallelism.

编译任务也从线程池移到了子进程池，减轻了 Python GIL（全局解释器锁）造成的瓶颈，提高了编译时的并行度。

Ava: What about other hardware?

Ava：其他硬件呢？

Brian: ROCm gets AOTriton zero point twelve beta, analytical Origami GEMM selection, and native HIP support in CMake.

Brian：ROCm 增加了 AOTriton 0.12 测试版、基于分析模型的 Origami GEMM 选择机制，以及 CMake 中的原生 HIP 支持。

Arm adds Armv9-A targeting for torch.

Arm 增加了面向 Armv9-A 的 torch.

compile, including the correct feature set for one-hundred-twenty-eight-bit and two-hundred-fifty-six-bit SVE.

compile 支持，包括针对 128 位和 256 位 SVE 的正确特性集。

Intel XPU adds telemetry APIs for memory, utilization, power, clocks, temperature, cache size, synchronization, and integrated-GPU detection.

Intel XPU 增加了用于获取内存、利用率、功耗、时钟频率、温度、缓存大小和同步信息，以及检测集成 GPU 的遥测 API。

Ava: There is also Python 3. 15 support, but the details matter.

Ava：它还支持 Python 3.15，不过具体情况需要细看。

Brian: Very much. Linux torch wheels support Python 3. 15 and the free-threaded 3. 15t build.

Brian：没错。Linux 版 torch wheel 支持 Python 3.15 和自由线程版 3.15t。

Python 3. 15 is still beta, and stable release is scheduled for October 2026.

Python 3.15 仍处于测试阶段，稳定版计划于 2026 年 10 月发布。

Support covers x86_64 and aarch64 CPU, CUDA, ROCm, and XPU builds.

支持 x86_64 和 aarch64 CPU，以及 CUDA、ROCm 和 XPU 构建版本。

Ava: What is missing?

Ava：还有哪些不支持？

Brian: Torchvision has no 3. 15 build yet. torch. compile is not supported on Python 3. 15.

Brian：Torchvision 还没有适配 3.15 的构建版本。Python 3.15 也不支持 torch.compile。

There are no Windows or macOS wheels.

目前没有 Windows 或 macOS 版 wheel。

The wheels are also available only through the PyTorch download index, not PyPI.

这些 wheel 只能从 PyTorch 下载索引获取，PyPI 上没有。

Ava: Any changes that could break older projects?

Ava：有哪些变化可能影响旧项目？

Brian: Named tensors have been removed completely.

Brian：命名张量已被彻底移除。

Distributed collectives are moving to a single naming scheme: all_gather_single and reduce_scatter_single.

分布式集合通信正在统一命名为 all_gather_single 和 reduce_scatter_single。

The old names remain as deprecated wrappers. The Bazel build is gone too. Python 3.

旧名称仍作为已弃用的封装函数保留。Bazel 构建也已移除。Python 3.

13t was dropped from the Linux binary matrix, and CUDA 13 remains the default build.

13t 已从 Linux 二进制构建矩阵中移除，CUDA 13 仍是默认构建版本。

Ava: And ExecuTorch?

Ava：ExecuTorch 呢？

Brian: This release marks ExecuTorch's integration into PyTorch Core, making on-device inference a first-class capability of the framework.

Brian：此版本将 ExecuTorch 集成进 PyTorch Core，让端侧推理成为框架的原生能力。

Ava: Let's finish with the three things listeners should remember.

Ava：最后总结一下听众需要记住的三点。

Brian: First, performance follows the hardware and workload: FlexAttention is especially strong for sparse attention, native Metal reduces dispatch overhead, and CuTeDSL adds another optimized GPU path.

Brian：第一，性能取决于硬件和负载：FlexAttention 尤其擅长稀疏注意力，原生 Metal 减少调度开销，CuTeDSL 则提供另一条优化的 GPU 路径。

Ava: Second, memory and scale improve together: fused linear cross-entropy cuts peak memory, torchcomms improves cluster fault handling and visibility, and FSDP2 can overlap all-gather with reduce-scatter.

Ava：第二，内存效率和扩展能力同步提升：融合线性交叉熵降低峰值内存占用，torchcomms 改善集群故障处理和可观测性，FSDP2 可让 all-gather 与 reduce-scatter 重叠执行。

Brian: Third, check the boundaries before upgrading. Many APIs are unstable, Python 3.

Brian：第三，升级前先看清支持范围。许多 API 尚不稳定，Python 3.

15 support is limited and still beta, dense attention may favor SDPA, and named tensors plus Bazel are removed.

15 支持有限且仍处于测试阶段，密集注意力可能更适合 SDPA，命名张量和 Bazel 则已移除。

Ava: That is PyTorch 2.

Ava：以上就是 PyTorch 2.

13: faster kernels, less memory pressure, and more tools for large-scale systems.

13：内核更快、内存压力更小，也为大规模系统提供了更多工具。

Brian: Try the features, read the release notes, and report issues as the two-series keeps evolving.

Brian：试试这些功能，阅读发布说明，并在 2.x 系列持续演进的过程中反馈问题。

Ava: Thanks for listening. See you next time.

Ava：感谢收听，下次见。

## 术语

| Term | 释义 |
|---|---|
| FlexAttention | 用于用 Python 表达自定义注意力模式并编译成融合 kernel 的统一 API |
| MPS | Apple Silicon 的 Metal Performance Shaders 后端 |
| SDPA | Scaled Dot-Product Attention，缩放点积注意力 |
| Deterministic backward | 确定性反向传播，使相同输入得到逐 bit 可复现的梯度 |
| nn.LinearCrossEntropyLoss | 融合最终线性投影和交叉熵计算、降低峰值显存的模块 |
| torchcomms | PyTorch Distributed 的新通信后端，增强容错、扩展性和可调试性 |
| FSDP2 | Fully Sharded Data Parallel 2，全分片数据并行第二代实现 |
| all-gather | 集合通信操作，把各进程数据收集到所有进程 |
| reduce-scatter | 先归约再分发结果的集合通信操作 |
| CuTeDSL | 基于 NVIDIA CuTe 的 Python 原生领域特定语言 |
| GEMM | General Matrix-Matrix Multiplication，通用矩阵乘法 |
| Inductor | PyTorch 编译器中的代码生成与优化组件 |
| NCCL | NVIDIA Collective Communications Library，NVIDIA 集合通信库 |
| Safetensors | 支持内存映射且避免任意代码执行风险的模型权重格式 |
| CUPTI | CUDA Profiling Tools Interface，用于收集 GPU 活动和指标的接口 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What would an engineer notice first? | 工程师首先会注意到什么？ |
| Let's unpack that. | 我们把这个拆开讲讲。 |
| That sounds painful for debugging. | 这对调试来说听起来很麻烦。 |
| What kind of speed are we talking about? | 我们说的速度提升大概是什么量级？ |
| That is exactly what PyTorch is trying to avoid. | 这正是 PyTorch 想要避免的情况。 |
| What is missing? | 还有哪些东西没有支持？ |
| Let's finish with the three things listeners should remember. | 最后总结听众应该记住的三件事。 |
| Check the boundaries before upgrading. | 升级前先确认限制条件。 |
