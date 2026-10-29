# PyTorch 2.12: Faster Math, Unified Graphs, and Wider Hardware Support

原文：[PyTorch 2.12 Release Blog](https://pytorch.org/blog/pytorch-2-12-release-blog/)

## 摘要

本期介绍 PyTorch 2.12 的主要更新：CUDA 上批量特征值分解最高提速一百倍，新增跨硬件的 torch.accelerator.Graph，并让 torch.export 支持 Microscaling 量化。我们还讨论了融合版 Adagrad、torch.cond 在 CUDA Graph 中的捕获、分布式训练分析工具，以及 ROCm、Apple MPS 等平台更新。最后，两位主持人解释了 torchcomms 和 TorchScript 的后续变化，以及 CUDA wheel 的弃用安排。

## 对话

Ava: If your training job spends minutes solving lots of small eigenvalue problems, PyTorch 2.12 may change that to seconds. And the headline number is up to one hundred times faster on CUDA.

Ava：如果你的训练任务要花几分钟求解大量小型特征值问题，PyTorch 2.12 可能把这个时间缩短到几秒。最引人注目的数字是：在 CUDA 上最高可提速一百倍。

Brian: Right. This release is also about making PyTorch feel more consistent across hardware. We get a device-agnostic graph API, wider export support, and several distributed and ROCm improvements.

Brian：没错。这次发布也致力于让 PyTorch 在不同硬件上的使用体验更加一致。我们有了与设备无关的图 API、更广泛的导出支持，以及多项分布式和 ROCm 方面的改进。

Ava: So let's start with the speed story. What exactly changed for batched linalg.eigh?

Ava：那就先从速度说起。批量 linalg.eigh 到底发生了什么变化？

Brian: linalg.eigh computes eigenvalues and eigenvectors for symmetric or Hermitian matrices. In earlier releases, the CUDA path could dispatch each matrix solve separately. That created a big performance gap, especially when you had many small or medium matrices.

Brian：linalg.eigh 用于计算对称矩阵或 Hermitian 矩阵的特征值和特征向量。在此前的版本中，CUDA 路径可能会分别调度每个矩阵的求解。这造成了很大的性能差距，尤其是在处理大量小型或中型矩阵时。

Ava: So the GPU was doing lots of tiny jobs instead of one big batch?

Ava：所以 GPU 当时是在执行许多很小的任务，而不是一次处理一整个批次？

Brian: Exactly. PyTorch has overhauled backend selection. The old MAGMA backend was deprecated in favor of cuSolver, and the dispatch heuristics now use cuSolver's syevj_batched kernel unconditionally for this case.

Brian：正是这样。PyTorch 全面调整了后端选择机制。旧的 MAGMA 后端已被弃用，转而采用 cuSolver；对于这种情况，调度启发式规则现在无条件使用 cuSolver 的 syevj_batched 内核。

Ava: And that kernel is designed for many matrices at once.

Ava：而这个内核就是为一次处理多个矩阵设计的。

Brian: Yes. It processes the batch as a single GPU operation. The article says workloads that used to take minutes can now run in seconds, with speedups of up to one hundred times over the previous release.

Brian：对。它把整个批次作为一次 GPU 操作来处理。文章称，过去需要几分钟的工作负载现在几秒就能运行完，相比上一版本，提速最高可达一百倍。

Ava: That sounds huge, but the wording matters: up to one hundred times, not every workload.

Ava：听起来提升巨大，但措辞很关键：是最高可达一百倍，并不是每种工作负载都能达到。

Brian: Correct. The gain depends on the workload. The release calls out scientific computing and machine learning jobs that rely on batched eigendecompositions. It also says this helps close longstanding performance gaps with CuPy.

Brian：没错。提升幅度取决于工作负载。这次发布特别提到了依赖批量特征分解的科学计算和机器学习任务，还表示这有助于缩小与 CuPy 之间长期存在的性能差距。

Ava: Let's move to optimizers. Adagrad now has fused equals true. What does fused mean here?

Ava：接着说优化器。Adagrad 现在支持将 fused 设为 true。这里的 fused 是什么意思？

Brian: A fused optimizer performs the whole optimizer step in one CUDA kernel. Without fusion, separate operations launch separate kernels. Fusion reduces kernel launch overhead and memory traffic.

Brian：融合优化器会在一个 CUDA 内核中完成整个优化器更新步骤。如果不融合，各项操作就会分别启动内核。融合可以减少内核启动开销和内存数据传输。

Ava: So Adagrad joins Adam, AdamW, and SGD in having a fused variant.

Ava：所以 Adagrad 和 Adam、AdamW、SGD 一样，也有了融合版本。

Brian: Exactly. The CUDA kernel was contributed during the 2.11 cycle, and the Python frontend is now exposed in 2.12. The practical idea is simple: fewer launches and less movement of intermediate data.

Brian：正是如此。CUDA 内核是在 2.11 周期贡献的，Python 前端现在于 2.12 中开放使用。实际思路很简单：减少内核启动次数，减少中间数据的搬运。

Ava: Now, the bigger architectural change seems to be torch.accelerator.Graph.

Ava：现在看来，更大的架构变化是 torch.accelerator.Graph。

Brian: Yes. torch.accelerator.Graph is a new device-agnostic API for graph capture and replay. Instead of users learning a separate graph interface for every backend, they get one common abstraction.

Brian：是的。torch.accelerator.Graph 是一个新的、与设备无关的图捕获和重放 API。用户不必为每种后端分别学习一套图接口，而是可以使用同一个通用抽象。

Ava: Can you give a plain-language analogy?

Ava：能用通俗的话打个比方吗？

Brian: Think of recording a fixed sequence of GPU work and replaying that recording many times. The API is like a universal remote. CUDA, XPU, and out-of-tree backends can keep their own internal implementations, but users interact with a consistent control surface.

Brian：可以把它想成录下一套固定的 GPU 工作流程，然后反复播放这段录像。这个 API 就像一个通用遥控器。CUDA、XPU 和树外后端可以保留各自的内部实现，而用户面对的是一致的操作接口。

Ava: How do backends plug into it?

Ava：后端要怎样接入它？

Brian: Each backend can register an implementation through GraphImplInterface. That keeps backend autonomy while giving applications a shared API. The release has initial XPU support and an extension path for out-of-tree backends through PrivateUse1.

Brian：每个后端都可以通过 GraphImplInterface 注册自己的实现。这样既保留了后端的自主性，也为应用提供了统一的 API。这次发布提供了初步的 XPU 支持，并通过 PrivateUse1 为树外后端提供了扩展路径。

Ava: There is also a stream change, right?

Ava：还有一项与流有关的变化，对吧？

Brian: Right. c10d::Stream and torch.Stream now expose is_capturing(), which replaces the device-specific is_current_stream_capturing. Stream context manager reentrance was fixed too. Together, these changes improve parity across backends.

Brian：对。c10d::Stream 和 torch.Stream 现在都公开了 is_capturing()，用它取代针对特定设备的 is_current_stream_capturing。流上下文管理器的重入问题也得到了修复。这些变化共同改善了不同后端之间的一致性。

Ava: What about export and quantization?

Ava：导出和量化方面呢？

Brian: torch.export.save and torch.export.load now support Microscaling, or MX, quantization formats. Before 2.12, export could not handle the float8_e8m0fnu dtype used as the shared block-scale exponent in MXFP4, MXFP6, and MXFP8.

Brian：torch.export.save 和 torch.export.load 现在支持 Microscaling，也就是 MX 量化格式。在 2.12 之前，导出功能无法处理 float8_e8m0fnu 数据类型；这种类型在 MXFP4、MXFP6 和 MXFP8 中用作共享的块缩放指数。

Ava: So a model could use MX quantization, but the export-to-deployment path was blocked.

Ava：也就是说，模型虽然能使用 MX 量化，但从导出到部署的流程会被卡住。

Brian: Exactly. Now tensors with that dtype can be serialized and deserialized correctly. That unblocks the full workflow for aggressively compressed models, which matters for large language models deployed in cost-constrained or edge environments.

Brian：正是这样。现在，使用这种数据类型的张量可以正确序列化和反序列化。这打通了高度压缩模型的完整工作流程；对于部署在成本受限或边缘环境中的大语言模型，这一点很重要。

Ava: That is a nice example of a small missing dtype support causing a much larger production problem.

Ava：这很好地说明了，缺少对一种小小的数据类型的支持，也可能造成大得多的生产问题。

Brian: Yes. The quantization method existed, but the deployment handoff was incomplete. This release closes that gap.

Brian：是的。量化方法本身已经存在，但交付部署的环节还不完整。这次发布补上了这个缺口。

Ava: The release also mentions torch.cond inside CUDA Graphs. Why was that difficult?

Ava：这次发布还提到 CUDA Graphs 中的 torch.cond。此前难点在哪里？

Brian: torch.cond represents data-dependent control flow. Previously, branches were evaluated on the CPU, so CUDA graph capture had to fall back to CUDA graph trees. With CUDA 12.4 conditional IF nodes, both branches can now be evaluated on the GPU inside one graph capture.

Brian：torch.cond 表示依赖数据的控制流。此前，分支是在 CPU 上求值的，所以 CUDA 图捕获必须回退到 CUDA graph trees。借助 CUDA 12.4 的条件 IF 节点，现在两个分支都可以在一次图捕获中于 GPU 上求值。

Ava: So the graph can contain the decision itself.

Ava：所以图本身就能包含这个判断。

Brian: Exactly. It can capture and replay the control-flow region. The current support works with the eager and cudagraphs backends. Inductor support is planned for a future release, so that is one limitation to remember.

Brian：正是如此。它可以捕获并重放这段控制流。目前的支持适用于 eager 和 cudagraphs 后端。Inductor 支持计划在未来的版本中加入，所以这是一个需要记住的限制。

Ava: There is also a numerical correctness update for XPU involving addcdiv.

Ava：XPU 还有一项涉及 addcdiv 的数值正确性更新。

Brian: Yes. addcdiv is a fused arithmetic operation: input plus value times tensor one divided by tensor two. It appears in optimizer update rules such as Adam, AdamW, and RMSprop.

Brian：是的。addcdiv 是一种融合算术运算：输入加上 value 乘以张量一除以张量二的结果。它出现在 Adam、AdamW 和 RMSprop 等优化器的更新规则中。

Ava: What was going wrong?

Ava：之前出了什么问题？

Brian: Inductor previously lowered the operation with separate multiply and divide instructions. That could create small floating-point rounding differences compared with eager CUDA execution. Over thousands of steps, those differences can accumulate.

Brian：此前 Inductor 会把这个运算转换成独立的乘法和除法指令。与 eager CUDA 执行相比，这可能产生细微的浮点舍入差异。经过数千步之后，这些差异可能逐渐累积。

Ava: And now it uses fused multiply-add instructions?

Ava：现在它会使用融合乘加指令了？

Brian: For CUDA first, and now for XPU as well. The goal is bitwise numerical parity with eager CUDA execution while preserving Triton kernel fusion benefits. That helps teams validate compiled optimizer-heavy training loops on NVIDIA and Intel hardware.

Brian：最先是 CUDA，现在 XPU 也一样。目标是在保留 Triton 内核融合优势的同时，与 eager CUDA 执行实现逐位一致的数值结果。这有助于团队在 NVIDIA 和 Intel 硬件上验证大量使用优化器的编译式训练循环。

Ava: Let's talk distributed training. What changed for custom operators?

Ava：来说说分布式训练。自定义算子有什么变化？

Brian: Custom operators can now accept ProcessGroup objects directly. Callers no longer have to turn a group into a string name and look it up in a global registry. The c10d functional collective operations, including all_reduce and reduce_scatter, accept both objects and string names.

Brian：自定义算子现在可以直接接收 ProcessGroup 对象。调用方不再需要把组转换为字符串名称，再到全局注册表中查找。c10d 函数式集合通信操作，包括 all_reduce 和 reduce_scatter，现在同时接受对象和字符串名称。

Ava: That sounds like a cleaner interface.

Ava：听起来接口更简洁了。

Brian: It is, and profiling got richer too. The PyTorch Profiler Events API now exposes flow IDs, flow types, activity types, unfinished events, and Python function events. Its events() output is closer to the Chrome trace JSON output.

Brian：确实如此，而且性能分析能提供的信息也更丰富了。PyTorch Profiler Events API 现在公开流 ID、流类型、活动类型、尚未结束的事件和 Python 函数事件。它的 events() 输出也更接近 Chrome trace JSON 输出。

Ava: How does that help with multi-node debugging?

Ava：这对多节点调试有什么帮助？

Brian: NCCL collective traces can now be correlated across ranks with a seq_num field. All ranks participating in the same collective share the same sequence number within a process group. That makes post-hoc analysis much more practical.

Brian：现在可以通过 seq_num 字段关联不同 rank 上的 NCCL 集合通信跟踪记录。在同一个进程组内，参与同一次集合通信的所有 rank 都共享同一个序列号。这让事后分析实用得多。

Ava: And FlightRecorder covers more communication backends.

Ava：FlightRecorder 也覆盖了更多通信后端。

Brian: Yes. Its trace analyzer now supports ncclx and gloo alongside nccl and xccl. It also recognizes torchcomms operations such as all_gather_single, reduce_scatter_v, and barrier. A race condition involving multiple process groups was fixed as well.

Brian：是的。它的跟踪分析器现在除了支持 nccl 和 xccl，还支持 ncclx 和 gloo。它也能识别 all_gather_single、reduce_scatter_v 和 barrier 等 torchcomms 操作。此外，还修复了涉及多个进程组的一个竞态条件。

Ava: What platform updates should ROCm users notice?

Ava：ROCm 用户应该留意哪些平台更新？

Brian: Several. On AMD GPUs with ROCm 7.02 or newer, the caching allocator supports expandable memory segments. That dynamically grows allocations through virtual memory APIs and reduces fragmentation, matching the CUDA feature.

Brian：有好几项。在使用 ROCm 7.02 或更新版本的 AMD GPU 上，缓存分配器支持可扩展内存段。这项功能通过虚拟内存 API 动态扩大内存分配，减少内存碎片，与 CUDA 的对应功能保持一致。

Ava: There is also rocSHMEM.

Ava：还有 rocSHMEM。

Brian: Right. rocSHMEM brings symmetric memory collective operations to AMD GPUs through torch.ops.symm_mem. It ports on-GPU primitives such as point-to-point, broadcast, all-to-all, and MoE-oriented two-dimensional AllToAllv.

Brian：对。rocSHMEM 通过 torch.ops.symm_mem 为 AMD GPU 带来对称内存集合通信操作。它移植了在 GPU 上运行的基础操作，包括点对点通信、广播、全对全通信，以及面向 MoE 的二维 AllToAllv。

Ava: And sparse computation?

Ava：稀疏计算呢？

Brian: hipSPARSELt is enabled by default in ROCm builds at 7.12 or newer. It adds semi-structured two-to-four sparsity support, and FP8 inputs are supported on MI350X with FP32 output. That enables the torch._cslt_sparse_mm acceleration path on AMD.

Brian：在 ROCm 7.12 或更新版本的构建中，hipSPARSELt 默认启用。它增加了对 2:4 半结构化稀疏性的支持，并且在 MI350X 上支持 FP8 输入和 FP32 输出。这使 AMD 平台能够使用 torch._cslt_sparse_mm 加速路径。

Ava: FlexAttention also gets a pipeline change.

Ava：FlexAttention 的流水线也有变化。

Brian: Yes. On AMD GPUs, the Triton backend now uses two-stage pipelining for FlexAttention. The article reports five to twenty-six percent speedups across causal, alibi, and sliding-window attention patterns on MI350X. The change was just a one-line configuration adjustment from one stage to two.

Brian：是的。在 AMD GPU 上，Triton 后端现在为 FlexAttention 使用两阶段流水线。文章报告称，在 MI350X 上，因果注意力、ALiBi 注意力和滑动窗口注意力模式的速度提升了 5% 至 26%。这项改动只是把一行配置中的阶段数从 1 改为 2。

Ava: What about Apple users?

Ava：Apple 用户呢？

Brian: Apple Silicon binary wheels now ship with ahead-of-time-compiled Metal-4 shaders. They were built on macOS 26 with the Metal-4 standard, so MPS workloads avoid runtime shader compilation on first run and get lower startup latency.

Brian：Apple Silicon 二进制 wheel 现在随附预先编译好的 Metal-4 着色器。这些着色器是在 macOS 26 上按 Metal-4 标准构建的，因此 MPS 工作负载首次运行时无需在运行时编译着色器，启动延迟也更低。

Ava: Before we wrap up, we need the migration warnings.

Ava：结束之前，我们还得说说迁移方面的注意事项。

Brian: The biggest future change is torchcomms. In an upcoming release, 2.13 or later, PyTorch plans to use torchcomms by default in Distributed. That brings breaking changes to ProcessGroup behavior, even though the team aims to make most migrations automatic.

Brian：未来最大的变化是 torchcomms。PyTorch 计划在即将发布的 2.13 或更高版本中，让 Distributed 默认使用 torchcomms。这会给 ProcessGroup 的行为带来破坏性变更，尽管团队希望让大多数迁移自动完成。

Ava: What should engineers watch for?

Ava：工程师应该留意什么？

Brian: ProcessGroups and communicators will need eager initialization during dist.init_process_group, with one backend device. P2P operations on the same group and stream will not be guaranteed to run concurrently; concurrent operations will need batch APIs or a separate group. PyTorch also plans to make torchcomms a required package and deprecate the existing c10d backends.

Brian：ProcessGroups 和通信器需要在调用 dist.init_process_group 时，使用单个后端设备提前初始化。同一组和同一流上的 P2P 操作将不再保证并发执行；需要并发执行时，必须使用批量 API 或单独的组。PyTorch 还计划将 torchcomms 设为必需的软件包，并弃用现有的 c10d 后端。

Ava: And TorchScript?

Ava：TorchScript 呢？

Brian: TorchScript is now deprecated. It was deprecated in 2.10. The recommended direction is torch.export instead of the jit trace and script APIs, and Executorch instead of the embedded runtime.

Brian：TorchScript 现在已经弃用。它是在 2.10 中被弃用的。建议用 torch.export 取代 jit 的 trace 和 script API，用 Executorch 取代嵌入式运行时。

Ava: There is one more packaging detail: CUDA 12.8 wheels.

Ava：还有一个打包细节：CUDA 12.8 wheel。

Brian: Starting with 2.12, the CUDA 12.8 binary wheel is deprecated and will no longer be in the standard release matrix. The default wheel is CUDA 13.0, CUDA 13.2 is experimental, and CUDA 12.6 remains supported for older architectures. Newer GPUs need CUDA 13.0 or newer and an NVIDIA driver upgrade to the versions listed in the release notes.

Brian：从 2.12 开始，CUDA 12.8 二进制 wheel 被弃用，也将不再列入标准发布矩阵。默认 wheel 是 CUDA 13.0，CUDA 13.2 仍处于实验阶段，CUDA 12.6 则继续支持较旧的架构。较新的 GPU 需要 CUDA 13.0 或更新版本，还需要将 NVIDIA 驱动升级到发行说明中列出的版本。

Ava: Let's do the three-point recap. First?

Ava：我们来做个三点回顾。第一点是什么？

Brian: First, performance: batched CUDA linalg.eigh can be up to one hundred times faster, and Adagrad gets a fused optimizer step.

Brian：第一点是性能：批量 CUDA linalg.eigh 的速度最高可提升至 100 倍，Adagrad 也获得了融合优化器步骤。

Ava: Second: portability and deployment.

Ava：第二，跨平台适用性和部署。

Brian: Right. torch.accelerator.Graph unifies graph capture and replay, torch.export supports MX quantization, and torch.cond can run inside CUDA Graph capture.

Brian：对。torch.accelerator.Graph 统一了图捕获与重放，torch.export 支持 MX 量化，torch.cond 也可以在 CUDA Graph 捕获过程中运行。

Ava: Third: production tooling and platform reach.

Ava：第三，生产环境工具和平台覆盖范围。

Brian: That includes richer distributed profiling, broader ROCm support, Metal-4 offline shaders, and migration work toward torchcomms. The release has 2,926 commits from 457 contributors, so there is a lot packed in.

Brian：这包括更丰富的分布式性能分析、更广泛的 ROCm 支持、Metal-4 离线着色器，以及向 torchcomms 迁移的工作。这个版本包含来自 457 位贡献者的 2,926 次提交，内容确实很多。

Ava: PyTorch 2.12 is faster, more hardware-agnostic, and more ready for deployment, while some migration work is still ahead.

Ava：PyTorch 2.12 速度更快、对硬件的依赖更少，也更适合部署，不过后续仍有一些迁移工作要做。

Brian: Exactly. Try the features, check the limits for your backend, and report issues as the 2.x series keeps moving.

Brian：没错。试试这些功能，确认你所用后端的限制，并在 2.x 系列持续演进的过程中反馈问题。

Ava: Thanks for listening. We'll be back with another PyTorch release explained.

Ava：感谢收听。下次我们会继续解读另一个 PyTorch 版本。

## 术语

| Term | 释义 |
|---|---|
| batched eigendecomposition | 批量特征值分解，同时处理多个矩阵的特征值和特征向量 |
| cuSolver | NVIDIA CUDA 数值线性代数库 |
| syevj_batched | cuSolver 中用于批量对称/Hermitian 特征值问题的内核 |
| fused optimizer | 把优化器步骤合并到单个 GPU 内核中的实现 |
| torch.accelerator.Graph | 跨设备统一的图捕获与重放 API |
| GraphImplInterface | 后端注册自有图实现的接口 |
| Microscaling (MX) quantization | 使用共享块尺度进行激进压缩的量化格式 |
| torch.export | 用于序列化和部署 PyTorch 模型的导出路径 |
| CUDA Graph | 捕获一组 CUDA 操作并重复重放的机制 |
| torch.cond | 表达数据相关条件控制流的 PyTorch API |
| ProcessGroup | 分布式通信中管理进程和集体操作的对象 |
| NCCL | NVIDIA 集体通信库 |
| rocSHMEM | ROCm 平台上的对称内存通信库 |
| FlexAttention | 可配置的注意力实现，支持多种注意力模式 |
| torchcomms | 正在集成进 PyTorch Distributed 的通信组件 |

## 口语表达

| Phrase | 释义 |
|---|---|
| That sounds huge, but the wording matters. | 听起来很大，但措辞很重要。 |
| Can you give a plain-language analogy? | 你能用通俗的类比解释吗？ |
| So the graph can contain the decision itself. | 所以图本身可以包含这个判断。 |
| That unblocks the full workflow. | 这让完整流程终于可以跑通。 |
| Let's talk distributed training. | 我们来聊聊分布式训练。 |
| What should engineers watch for? | 工程师应该注意什么？ |
| Let's do the three-point recap. | 我们做一个三点回顾。 |
| Thanks for listening. | 感谢收听。 |

## 中文内容总结

PyTorch 2.12 汇集了自 2.11 以来 457 名贡献者提交的 2,926 次修改，重点改进 CUDA 计算性能、跨设备图捕获、量化模型导出和分布式训练工具。部分功能有明确的硬件或版本条件，未来的分布式接口变更也尚未在本版默认启用。

### 计算性能

CUDA 上的批量对称矩阵和厄米矩阵特征分解改用 cuSolver 的 syevj_batched 内核。官方报告，相比上一版，torch.linalg.eigh 在适用负载中最高快 100 倍，原先耗时数分钟的批量求解可缩短到数秒。Adagrad 新增 fused=True，用单个 CUDA 内核完成优化器步骤，以减少内核启动和内存访问开销；原文未给出其加速幅度。

### 图捕获与模型导出

新的 torch.accelerator.Graph 为 CUDA、XPU 和外部后端提供统一的图捕获与重放接口，后端可注册自己的实现；流接口也新增 is_capturing()。torch.export.save 和 load 现在能处理 MXFP4、MXFP6、MXFP8 所用的 float8_e8m0fnu 块缩放指数，使这类量化模型能够完整导出和加载。借助 CUDA 12.4 的条件 IF 节点，torch.cond 分支可在 CUDA Graph 内由 GPU 判断并重放；目前适用于 eager 和 cudagraphs 后端，Inductor 支持仍待后续版本。

### 数值与分布式调试

Inductor 对 addcdiv 使用融合乘加指令，官方称这使 CUDA 和 XPU 上编译执行的结果与 eager CUDA 达到逐位一致，便于核对优化器训练过程。自定义算子和 c10d 函数式集合通信现在可直接接收 ProcessGroup 对象，也兼容字符串组名。Profiler Events API 增加事件和关联信息；同一进程组内参与同一次 NCCL 集合通信的各 rank 可用 seq_num 对齐轨迹。FlightRecorder 扩展到 ncclx、gloo 和更多 torchcomms 操作，并修复并发访问可能造成的死循环。

### 设备平台更新

CUDA Graph 可为捕获的内核加入注释，经过配套脚本处理后在性能轨迹中显示；实验性的 CUDA Green Context 工作队列上限可控制并发提交。ROCm 7.02 及以上支持可扩展内存段，AMD GPU 新增 rocSHMEM 对称内存通信。ROCm 7.12 及以上默认启用 hipSPARSELt；MI350X 支持 FP8 输入的 2:4 半结构化稀疏计算，输出为 FP32。官方报告，MI350X 上采用两阶段流水线的 FlexAttention 在所测注意力模式和形状下快 5% 至 26%。Apple Silicon 安装包提供在 macOS 26 上预编译的 Metal-4 着色器，以减少首次运行的编译开销。

### 迁移与版本边界

torchcomms 在 2.12 中已可选择启用，但官方计划在 2.13 或之后才默认采用。届时预计要求初始化 ProcessGroup 时指定设备并完成通信器初始化，同组同流的点对点操作也不再保证并发；官方还计划将 torchcomms 设为 PyTorch Distributed 的必需依赖，并弃用现有的 c10d::Backends。这些计划中的变更尚未在 2.12 生效，部分模型未来可能需要调整。TorchScript 已在 2.10 弃用，官方建议分别用 torch.export 和 ExecuTorch 替代相关导出接口与嵌入式运行时。2.12 不再发布标准 CUDA 12.8 安装包；默认版本为 CUDA 13.0，CUDA 13.2 为实验性版本。较旧的 Pascal、Volta 架构可用仍受支持的 CUDA 12.6 安装包；使用 CUDA 13.0 及以上版本的新 GPU 需要相应升级 NVIDIA 驱动。
