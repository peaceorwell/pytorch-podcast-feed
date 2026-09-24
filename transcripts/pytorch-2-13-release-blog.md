# PyTorch 2.13: Faster Kernels, Leaner Training, and Better Distributed Systems

原文：[PyTorch 2.13 Release Blog](https://pytorch.org/blog/pytorch-2-13-release-blog/)

## 摘要

PyTorch 2.13 继续把框架推向面向生产、跨硬件、可大规模训练和推理的平台。它为 Apple Silicon 带来 FlexAttention，为 CUDA 提供确定性反向路径，并通过 CuTeDSL 和原生 Metal kernel 改善性能。大词表训练可使用融合的 nn.LinearCrossEntropyLoss，将峰值显存最多降低约四倍；分布式方面新增 torchcomms，并让 FSDP2 的 all-gather 与 reduce-scatter 重叠。该版本也支持 Linux 上的 Python 3.15 wheel，但仍有 beta、平台覆盖和 API unstable 等限制。

## 对话

**Ava:** If you're training large language models or running PyTorch on Apple Silicon, this release could change your daily work. PyTorch 2.13 is out, and it targets speed, memory, and cluster reliability at the same time.

**Brian:** Right. The headline is that PyTorch keeps moving from a research-first framework toward a unified, hardware-agnostic platform for production training and inference at scale.

**Ava:** So let's start with the biggest practical wins. What would an engineer notice first?

**Brian:** There are four big areas. FlexAttention expands to Apple Silicon and gets deterministic backward on CUDA. CuTeDSL gives Inductor another high-performance GPU path. A fused linear-plus-loss operation cuts memory for huge vocabularies. And distributed training gets torchcomms plus better communication overlap in FSDP2.

**Ava:** Let's unpack FlexAttention first. People often hear “custom attention” and think they need to write a CUDA kernel.

**Brian:** That is exactly what PyTorch is trying to avoid. FlexAttention is a unified API where you describe an attention pattern as a plain Python function. The compiler turns that description into a fused kernel. In 2.13, it works on Metal, also called MPS, Apple's backend.

**Ava:** So a Mac user can express sparse attention without hand-writing Metal code?

**Brian:** Yes. The MPS implementation includes hand-written Metal kernels for sparse prefill and decode paths, including grouped-query attention, or GQA, and captured buffers. The article describes the user experience as writing a two-line Python function while the compiler builds the fast kernel.

**Ava:** What kind of speed are we talking about?

**Brian:** For sparse masks, the numbers are substantial. On a one-by-eight-by-thirty-two-thousand-seven-hundred-sixty-eight-by-sixty-four shape, with a two-hundred-fifty-six-element sliding window and zero point eight percent density, FlexAttention runs in about thirty-five milliseconds. SDPA, scaled dot-product attention, takes about four hundred thirty-one milliseconds. That's roughly twelve point three times faster.

**Ava:** And a shorter sequence?

**Brian:** An eight-thousand-one-hundred-ninety-two length with a sixty-four-window case reaches about four point one five times faster. But dense patterns still favor SDPA, so the gain depends heavily on sparsity.

**Ava:** That's a useful limitation. What changed on CUDA?

**Brian:** The existing FlexAttention flash backend now has a deterministic backward path. By default, it used atomic operations to accumulate dQ, which could make repeated runs produce slightly different gradients.

**Ava:** That sounds painful for debugging and regression tests.

**Brian:** Exactly. The new compute_dq_write_order path precomputes a write order instead of using atomics. It gives bit-for-bit reproducible gradients without a meaningful performance penalty. The measured end-to-end overhead on create_block_mask is well under one percent at longer sequence lengths, with zero point two percent at a sequence length of thirty-two-thousand-seven-hundred-sixty-eight.

**Ava:** How do users enable it?

**Brian:** They opt in with the existing torch.use_deterministic_algorithms setting. No extra code changes are required. The API is marked unstable, though.

**Ava:** Before we leave Apple Silicon, there is also a larger native Metal migration.

**Brian:** Yes. Many operations previously went through MPSGraph, which adds compilation and scheduling overhead on each dispatch. PyTorch now moves a broad set to hand-written Metal compute kernels.

**Ava:** Which operations?

**Brian:** Copy and cast, random-number generation, comparisons, reductions like sum and mean, cumulative operations, sorting, embedding backward, and scatter or gather with bounds checking. Direct control over thread dispatch and memory access should reduce kernel launch latency for common training and inference workloads.

**Ava:** Now let's talk about memory, because large-vocabulary models can burn through it before the real computation begins.

**Brian:** The new nn.LinearCrossEntropyLoss addresses that. In the standard path, the final linear projection creates the full logits matrix over the vocabulary. With vocabularies above one hundred thousand tokens, that matrix can consume tens of gigabytes of GPU memory.

**Ava:** And the fused operation avoids materializing that matrix?

**Brian:** Right. It combines the final linear projection and cross-entropy computation, processing the vocabulary dimension in chunks. Peak memory drops by up to about four times for large-vocabulary workloads, while keeping numerical equivalence with the unfused path.

**Ava:** Does it support the features people already use?

**Brian:** It supports label smoothing, weight tying, and z-loss regularization out of the box. It also works with torch.compile. Since it replaces separate nn.Linear and nn.CrossEntropyLoss modules, adoption requires no other code changes.

**Ava:** That sounds unusually convenient for a low-level optimization.

**Brian:** Yes, although the API is still unstable. PyTorch 2.13 also makes torch.load detect safetensors files natively. Safetensors is popular because it supports memory-mapped loading and avoids arbitrary code execution risk. You no longer need a separate library just to load those weights.

**Ava:** Let's move to distributed training. What problem is torchcomms solving?

**Brian:** PyTorch Distributed historically relied on c10d's ProcessGroup abstraction for collectives such as all-reduce and all-gather. It was designed mainly around NCCL. As training uses more dimensions, elastic scaling, and mixed interconnects, error handling and observability become bottlenecks.

**Ava:** So torchcomms is a new backend?

**Brian:** Yes. It is integrated into PyTorch Distributed's CI and device-mesh paths. It aims to improve fault tolerance through graceful timeouts and partial-group recovery, scalability across large clusters, and debugging through structured logging and collective tracing. It keeps API compatibility with the existing c10d backends.

**Ava:** And FSDP2 gets communication overlap?

**Brian:** FSDP means Fully Sharded Data Parallel. By default, all-gather and reduce-scatter share one NCCL communicator. NCCL serializes operations on the same communicator, so the two collectives cannot overlap.

**Ava:** Which leaves bandwidth unused.

**Brian:** Exactly. FSDPModule.set_separate_reduce_scatter_group enables a dedicated communicator for reduce-scatter. Then all-gather and reduce-scatter can progress concurrently. That creates AG/RS overlap and improves throughput for fully sharded workloads without model-code changes. It is opt-in, and the API is unstable.

**Ava:** What does PyTorch 2.13 add on the compiler side?

**Brian:** There is torch.compiler.set_default_backend. Before this, a custom or out-of-tree backend had to be passed explicitly to every torch.compile call. Now infrastructure teams can set a process-wide default, while an explicit backend argument still overrides it.

**Ava:** And on NVIDIA GPUs, CuTeDSL is the other major compiler story.

**Brian:** CuTeDSL is a Python-native domain-specific language built on NVIDIA's CuTe, or CUDA Templates, library. It gives direct control over tensor layouts, tiling, and memory access. Inductor can use it alongside Triton for matrix multiplication, also called GEMM, and RMSNorm.

**Ava:** So Inductor now has a second code-generation route for key transformer kernels.

**Brian:** Right. The Quack-derived overrides aim for higher-quality matrix-multiply code. Compilation also moved from a thread pool to a subprocess pool, reducing the Python GIL, or Global Interpreter Lock, bottleneck and improving compile-time parallelism.

**Ava:** What about other hardware?

**Brian:** ROCm gets AOTriton zero point twelve beta, analytical Origami GEMM selection, and native HIP support in CMake. Arm adds Armv9-A targeting for torch.compile, including the correct feature set for one-hundred-twenty-eight-bit and two-hundred-fifty-six-bit SVE. Intel XPU adds telemetry APIs for memory, utilization, power, clocks, temperature, cache size, synchronization, and integrated-GPU detection.

**Ava:** There is also Python 3.15 support, but the details matter.

**Brian:** Very much. Linux torch wheels support Python 3.15 and the free-threaded 3.15t build. Python 3.15 is still beta, and stable release is scheduled for October 2026. Support covers x86_64 and aarch64 CPU, CUDA, ROCm, and XPU builds.

**Ava:** What is missing?

**Brian:** Torchvision has no 3.15 build yet. torch.compile is not supported on Python 3.15. There are no Windows or macOS wheels. The wheels are also available only through the PyTorch download index, not PyPI.

**Ava:** Any changes that could break older projects?

**Brian:** Named tensors have been removed completely. Distributed collectives are moving to a single naming scheme: all_gather_single and reduce_scatter_single. The old names remain as deprecated wrappers. The Bazel build is gone too. Python 3.13t was dropped from the Linux binary matrix, and CUDA 13 remains the default build.

**Ava:** And ExecuTorch?

**Brian:** This release marks ExecuTorch's integration into PyTorch Core, making on-device inference a first-class capability of the framework.

**Ava:** Let's finish with the three things listeners should remember.

**Brian:** First, performance follows the hardware and workload: FlexAttention is especially strong for sparse attention, native Metal reduces dispatch overhead, and CuTeDSL adds another optimized GPU path.

**Ava:** Second, memory and scale improve together: fused linear cross-entropy cuts peak memory, torchcomms improves cluster fault handling and visibility, and FSDP2 can overlap all-gather with reduce-scatter.

**Brian:** Third, check the boundaries before upgrading. Many APIs are unstable, Python 3.15 support is limited and still beta, dense attention may favor SDPA, and named tensors plus Bazel are removed.

**Ava:** That is PyTorch 2.13: faster kernels, less memory pressure, and more tools for large-scale systems.

**Brian:** Try the features, read the release notes, and report issues as the two-series keeps evolving.

**Ava:** Thanks for listening. See you next time.

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
