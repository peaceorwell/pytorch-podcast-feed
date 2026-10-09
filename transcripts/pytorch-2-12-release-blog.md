# PyTorch 2.12: Faster Math, Unified Graphs, and Wider Hardware Support

原文：[PyTorch 2.12 Release Blog](https://pytorch.org/blog/pytorch-2-12-release-blog/)

## 摘要

本期介绍 PyTorch 2.12 的主要更新：CUDA 上批量特征值分解最高提速一百倍，新增跨硬件的 torch.accelerator.Graph，并让 torch.export 支持 Microscaling 量化。我们还讨论了融合版 Adagrad、torch.cond 在 CUDA Graph 中的捕获、分布式训练分析工具，以及 ROCm、Apple MPS 等平台更新。最后，两位主持人解释了 torchcomms 和 TorchScript 的后续变化，以及 CUDA wheel 的弃用安排。

## 对话

Ava: If your training job spends minutes solving lots of small eigenvalue problems, PyTorch 2.

Ava：如果训练任务要花几分钟求解大量小型特征值问题，PyTorch 2.

12 may change that to seconds.

12 版可能把耗时缩短到几秒。

And the headline number is up to one hundred times faster on CUDA.

最亮眼的数字是：在 CUDA 上最高提速 100 倍。

Brian: Right. This release is also about making PyTorch feel more consistent across hardware.

Brian：对。这次发布也让 PyTorch 在不同硬件上的体验更一致。

We get a device-agnostic graph API, wider export support, and several distributed and ROCm improvements.

新增了设备无关的图 API，扩大了导出支持，还改进了分布式功能和 ROCm。

Ava: So let's start with the speed story. What exactly changed for batched linalg. eigh?

Ava：先说性能。批量 linalg.eigh 究竟改了什么？

Brian: linalg. eigh computes eigenvalues and eigenvectors for symmetric or Hermitian matrices.

Brian：linalg.eigh 用于计算对称矩阵或厄米矩阵的特征值和特征向量。

In earlier releases, the CUDA path could dispatch each matrix solve separately.

此前的版本中，CUDA 路径可能会逐个分派矩阵求解任务。

That created a big performance gap, especially when you had many small or medium matrices.

这造成了很大的性能差距，处理大量小型或中型矩阵时尤其明显。

Ava: So the GPU was doing lots of tiny jobs instead of one big batch?

Ava：所以 GPU 当时在执行许多小任务，而不是一次处理整个批次？

Brian: Exactly. PyTorch has overhauled backend selection.

Brian：正是如此。PyTorch 全面调整了后端选择方式。

The old MAGMA backend was deprecated in favor of cuSolver, and the dispatch heuristics now use cuSolver's syevj_batched kernel unconditionally for this case.

旧的 MAGMA 后端已弃用，改用 cuSolver；现在对此类任务，调度逻辑始终使用 cuSolver 的 syevj_batched 内核。

Ava: And that kernel is designed for many matrices at once.

Ava：这个内核就是为同时处理多个矩阵设计的。

Brian: Yes. It processes the batch as a single GPU operation.

Brian：对。它把整个批次作为一次 GPU 操作来处理。

The article says workloads that used to take minutes can now run in seconds, with speedups of up to one hundred times over the previous release.

文章说，原本要跑几分钟的任务现在几秒就能完成，比上一版本最高快 100 倍。

Ava: That sounds huge, but the wording matters: up to one hundred times, not every workload.

Ava：听起来提升很大，但要注意是“最高 100 倍”，并非所有任务都如此。

Brian: Correct. The gain depends on the workload.

Brian：没错，提升幅度取决于具体任务。

The release calls out scientific computing and machine learning jobs that rely on batched eigendecompositions.

发布说明特别提到了依赖批量特征分解的科学计算和机器学习任务。

It also says this helps close longstanding performance gaps with CuPy.

它还说，这有助于缩小与 CuPy 长期存在的性能差距。

Ava: Let's move to optimizers. Adagrad now has fused equals true. What does fused mean here?

Ava：再说优化器。Adagrad 现在支持 fused=True。这里的 fused 是什么意思？

Brian: A fused optimizer performs the whole optimizer step in one CUDA kernel.

Brian：融合优化器用一个 CUDA 内核完成整个优化步骤。

Without fusion, separate operations launch separate kernels.

如果不融合，各项操作就要分别启动内核。

Fusion reduces kernel launch overhead and memory traffic.

融合能减少内核启动开销和内存数据传输。

Ava: So Adagrad joins Adam, AdamW, and SGD in having a fused variant.

Ava：这样 Adagrad 也和 Adam、AdamW、SGD 一样，有了融合版本。

Brian: Exactly. The CUDA kernel was contributed during the 2.

Brian：没错。CUDA 内核是在 2.

11 cycle, and the Python frontend is now exposed in 2. 12.

11 开发周期贡献的，Python 前端现已在 2.12 中开放。

The practical idea is simple: fewer launches and less movement of intermediate data.

实际思路很简单：减少内核启动次数和中间数据搬运。

Ava: Now, the bigger architectural change seems to be torch. accelerator. Graph.

Ava：接下来，更大的架构变化似乎是 torch.accelerator.Graph。

Brian: Yes. torch. accelerator.

Brian：对，torch.accelerator.

Graph is a new device-agnostic API for graph capture and replay.

Graph 是用于图捕获和重放的新设备无关 API。

Instead of users learning a separate graph interface for every backend, they get one common abstraction.

用户不必为每种后端学习不同的图接口，可以使用统一的抽象接口。

Ava: Can you give a plain-language analogy?

Ava：能用通俗的话打个比方吗？

Brian: Think of recording a fixed sequence of GPU work and replaying that recording many times.

Brian：就像录下一套固定的 GPU 操作，再反复播放。

The API is like a universal remote.

这个 API 就像一只万能遥控器。

CUDA, XPU, and out-of-tree backends can keep their own internal implementations, but users interact with a consistent control surface.

CUDA、XPU 和源码树外后端仍可保留各自的内部实现，但用户使用的是一致的接口。

Ava: How do backends plug into it?

Ava：后端怎么接入？

Brian: Each backend can register an implementation through GraphImplInterface.

Brian：各后端可以通过 GraphImplInterface 注册自己的实现。

That keeps backend autonomy while giving applications a shared API.

这样后端仍能自主实现，应用则可以使用统一的 API。

The release has initial XPU support and an extension path for out-of-tree backends through PrivateUse1.

此版本初步支持 XPU，也为源码树外后端提供了通过 PrivateUse1 扩展的途径。

Ava: There is also a stream change, right?

Ava：流方面也有变化，对吧？

Brian: Right. c10d::Stream and torch.

Brian：对。c10d::Stream 和 torch.

Stream now expose is_capturing(), which replaces the device-specific is_current_stream_capturing.

Stream 现在提供 is_capturing()，取代设备专用的 is_current_stream_capturing。

Stream context manager reentrance was fixed too.

Stream 上下文管理器的重入问题也修复了。

Together, these changes improve parity across backends.

这些改动共同提升了各后端的一致性。

Ava: What about export and quantization?

Ava：导出和量化方面呢？

Brian: torch. export. save and torch. export.

Brian：torch.export.save 和 torch.export.

load now support Microscaling, or MX, quantization formats. Before 2.

load 现在支持微缩放（MX）量化格式。在 2.

12, export could not handle the float8_e8m0fnu dtype used as the shared block-scale exponent in MXFP4, MXFP6, and MXFP8.

12 之前，导出功能无法处理 float8_e8m0fnu 数据类型；它在 MXFP4、MXFP6 和 MXFP8 中用作共享的块缩放指数。

Ava: So a model could use MX quantization, but the export-to-deployment path was blocked.

Ava：也就是说，模型可以使用 MX 量化，却无法顺利导出并部署。

Brian: Exactly. Now tensors with that dtype can be serialized and deserialized correctly.

Brian：没错。现在这种数据类型的张量可以正确序列化和反序列化了。

That unblocks the full workflow for aggressively compressed models, which matters for large language models deployed in cost-constrained or edge environments.

这打通了高度压缩模型的完整工作流程，对部署在成本受限或边缘环境中的大语言模型尤其重要。

Ava: That is a nice example of a small missing dtype support causing a much larger production problem.

Ava：这很能说明问题：只缺少一种 dtype 支持，就可能引发严重得多的生产问题。

Brian: Yes. The quantization method existed, but the deployment handoff was incomplete.

Brian：是的。量化方法已经有了，但部署环节还没打通。

This release closes that gap.

这个版本补上了这一环。

Ava: The release also mentions torch. cond inside CUDA Graphs. Why was that difficult?

Ava：这个版本还提到了 CUDA Graphs 中的 torch.cond。之前难在哪里？

Brian: torch. cond represents data-dependent control flow.

Brian：torch.cond 表示依赖数据的控制流。

Previously, branches were evaluated on the CPU, so CUDA graph capture had to fall back to CUDA graph trees.

以前分支在 CPU 上求值，因此 CUDA 图捕获必须回退到 CUDA Graph Trees。

With CUDA 12.

有了 CUDA 12.

4 conditional IF nodes, both branches can now be evaluated on the GPU inside one graph capture.

4 的条件 IF 节点，现在可以在 GPU 上的一次图捕获中对两个分支求值。

Ava: So the graph can contain the decision itself.

Ava：也就是说，图里可以包含决策本身。

Brian: Exactly. It can capture and replay the control-flow region.

Brian：没错。它能捕获并重放这段控制流。

The current support works with the eager and cudagraphs backends.

目前 eager 和 cudagraphs 后端支持这一功能。

Inductor support is planned for a future release, so that is one limitation to remember.

Inductor 支持计划在未来版本加入，这是一个需要留意的限制。

Ava: There is also a numerical correctness update for XPU involving addcdiv.

Ava：XPU 的 addcdiv 还有一项数值正确性更新。

Brian: Yes.

Brian：是的。

addcdiv is a fused arithmetic operation: input plus value times tensor one divided by tensor two.

addcdiv 是一种融合算术运算：输入加上 value 乘以张量一，再除以张量二。

It appears in optimizer update rules such as Adam, AdamW, and RMSprop.

它用于 Adam、AdamW 和 RMSprop 等优化器的更新规则。

Ava: What was going wrong?

Ava：之前出了什么问题？

Brian: Inductor previously lowered the operation with separate multiply and divide instructions.

Brian：以前 Inductor 会把这个运算拆成单独的乘法和除法指令。

That could create small floating-point rounding differences compared with eager CUDA execution.

与 CUDA 的 eager 执行相比，这可能产生微小的浮点舍入差异。

Over thousands of steps, those differences can accumulate.

经过数千步，这些差异可能累积。

Ava: And now it uses fused multiply-add instructions?

Ava：现在改用融合乘加指令了？

Brian: For CUDA first, and now for XPU as well.

Brian：先支持了 CUDA，现在 XPU 也支持了。

The goal is bitwise numerical parity with eager CUDA execution while preserving Triton kernel fusion benefits.

目标是在保留 Triton 内核融合优势的同时，让数值结果与 CUDA 的 eager 执行逐位一致。

That helps teams validate compiled optimizer-heavy training loops on NVIDIA and Intel hardware.

这有助于团队在 NVIDIA 和 Intel 硬件上验证编译后的优化器密集型训练循环。

Ava: Let's talk distributed training. What changed for custom operators?

Ava：再谈谈分布式训练。自定义算子有什么变化？

Brian: Custom operators can now accept ProcessGroup objects directly.

Brian：自定义算子现在可以直接接收 ProcessGroup 对象。

Callers no longer have to turn a group into a string name and look it up in a global registry.

调用方不必再把进程组转成字符串名称，再从全局注册表中查找。

The c10d functional collective operations, including all_reduce and reduce_scatter, accept both objects and string names.

c10d 的函数式集合通信操作，包括 all_reduce 和 reduce_scatter，现在同时接受对象和字符串名称。

Ava: That sounds like a cleaner interface.

Ava：这个接口更简洁了。

Brian: It is, and profiling got richer too.

Brian：是的，性能分析信息也更丰富了。

The PyTorch Profiler Events API now exposes flow IDs, flow types, activity types, unfinished events, and Python function events.

PyTorch Profiler Events API 现在会提供流 ID、流类型、活动类型、未完成事件和 Python 函数事件。

Its events() output is closer to the Chrome trace JSON output.

它的 events() 输出也更接近 Chrome 跟踪记录的 JSON 输出。

Ava: How does that help with multi-node debugging?

Ava：这对多节点调试有什么帮助？

Brian: NCCL collective traces can now be correlated across ranks with a seq_num field.

Brian：现在可以通过 seq_num 字段关联不同 rank 上的 NCCL 集合通信跟踪记录。

All ranks participating in the same collective share the same sequence number within a process group.

同一进程组内参与同一次集合通信的所有 rank 都共享同一个序列号。

That makes post-hoc analysis much more practical.

这让事后分析实用得多。

Ava: And FlightRecorder covers more communication backends.

Ava：FlightRecorder 也覆盖了更多通信后端。

Brian: Yes. Its trace analyzer now supports ncclx and gloo alongside nccl and xccl.

Brian：是的。除了 nccl 和 xccl，它的跟踪分析器现在还支持 ncclx 和 gloo。

It also recognizes torchcomms operations such as all_gather_single, reduce_scatter_v, and barrier.

它也能识别 all_gather_single、reduce_scatter_v 和 barrier 等 torchcomms 操作。

A race condition involving multiple process groups was fixed as well.

此外还修复了涉及多个进程组的竞态问题。

Ava: What platform updates should ROCm users notice?

Ava：ROCm 用户该关注哪些平台更新？

Brian: Several. On AMD GPUs with ROCm 7.

Brian：有几项。在使用 ROCm 7. 的 AMD GPU 上，

02 or newer, the caching allocator supports expandable memory segments.

02 或更高版本的缓存分配器支持可扩展内存段。

That dynamically grows allocations through virtual memory APIs and reduces fragmentation, matching the CUDA feature.

它通过虚拟内存 API 动态扩展内存分配，减少碎片化，与 CUDA 的这项功能一致。

Ava: There is also rocSHMEM.

Ava：还有 rocSHMEM。

Brian: Right. rocSHMEM brings symmetric memory collective operations to AMD GPUs through torch.

Brian：对。rocSHMEM 通过 torch.

ops. symm_mem.

ops.symm_mem 为 AMD GPU 提供对称内存集合通信操作。

It ports on-GPU primitives such as point-to-point, broadcast, all-to-all, and MoE-oriented two-dimensional AllToAllv.

它移植了 GPU 上的点对点、广播、全对全，以及面向 MoE 的二维 AllToAllv 等原语。

Ava: And sparse computation?

Ava：稀疏计算呢？

Brian: hipSPARSELt is enabled by default in ROCm builds at 7. 12 or newer.

Brian：ROCm 7.12 及更新版本的构建默认启用 hipSPARSELt。

It adds semi-structured two-to-four sparsity support, and FP8 inputs are supported on MI350X with FP32 output.

它支持 2:4 半结构化稀疏；在 MI350X 上支持 FP8 输入和 FP32 输出。

That enables the torch. _cslt_sparse_mm acceleration path on AMD.

这让 AMD GPU 能使用 torch._cslt_sparse_mm 加速路径。

Ava: FlexAttention also gets a pipeline change.

Ava：FlexAttention 的流水线也有变化。

Brian: Yes. On AMD GPUs, the Triton backend now uses two-stage pipelining for FlexAttention.

Brian：对。AMD GPU 上的 Triton 后端现在为 FlexAttention 使用两阶段流水线。

The article reports five to twenty-six percent speedups across causal, alibi, and sliding-window attention patterns on MI350X.

文章称，在 MI350X 上，因果、ALiBi 和滑动窗口注意力模式提速 5% 至 26%。

The change was just a one-line configuration adjustment from one stage to two.

这次改动只用一行配置，就把阶段数从一改成了二。

Ava: What about Apple users?

Ava：苹果用户呢？

Brian: Apple Silicon binary wheels now ship with ahead-of-time-compiled Metal-4 shaders.

Brian：Apple Silicon 的二进制 wheel 现在附带预编译的 Metal 4 着色器。

They were built on macOS 26 with the Metal-4 standard, so MPS workloads avoid runtime shader compilation on first run and get lower startup latency.

它们基于 macOS 26 和 Metal 4 标准构建，让 MPS 工作负载首次运行时无需现场编译着色器，启动延迟也更低。

Ava: Before we wrap up, we need the migration warnings.

Ava：结束前，还得说说迁移注意事项。

Brian: The biggest future change is torchcomms. In an upcoming release, 2.

Brian：未来最大的变化是 torchcomms。PyTorch 计划在即将发布的 2.

13 or later, PyTorch plans to use torchcomms by default in Distributed.

13 或更新版本中，让分布式模块默认使用 torchcomms。

That brings breaking changes to ProcessGroup behavior, even though the team aims to make most migrations automatic.

这会对 ProcessGroup 行为带来不兼容变更，尽管团队希望让大部分迁移自动完成。

Ava: What should engineers watch for?

Ava：工程师要注意什么？

Brian: ProcessGroups and communicators will need eager initialization during dist.

Brian：ProcessGroup 和通信器需要在调用 dist.

init_process_group, with one backend device.

init_process_group 时提前初始化，并指定一个后端设备。

P2P operations on the same group and stream will not be guaranteed to run concurrently; concurrent operations will need batch APIs or a separate group.

同一组和流上的 P2P 操作不保证并发执行；并发操作需要使用批量 API 或单独的组。

PyTorch also plans to make torchcomms a required package and deprecate the existing c10d backends.

PyTorch 还计划将 torchcomms 设为必需依赖，并弃用现有的 c10d 后端。

Ava: And TorchScript?

Ava：那 TorchScript 呢？

Brian: TorchScript is now deprecated. It was deprecated in 2. 10.

Brian：TorchScript 已弃用，从 2.10 版开始就是如此。

The recommended direction is torch.

推荐改用 torch.

export instead of the jit trace and script APIs, and Executorch instead of the embedded runtime.

export 替代 jit 的 trace 和 script API，并用 ExecuTorch 替代嵌入式运行时。

Ava: There is one more packaging detail: CUDA 12. 8 wheels.

Ava：还有一个打包细节：CUDA 12.8 wheel。

Brian: Starting with 2. 12, the CUDA 12.

Brian：从 2.12 版开始，CUDA 12.

8 binary wheel is deprecated and will no longer be in the standard release matrix.

8 二进制 wheel 将被弃用，不再列入标准发布矩阵。

The default wheel is CUDA 13. 0, CUDA 13. 2 is experimental, and CUDA 12.

默认 wheel 使用 CUDA 13.0，CUDA 13.2 尚处实验阶段，而 CUDA 12.

6 remains supported for older architectures. Newer GPUs need CUDA 13.

6 仍支持较旧的架构。新款 GPU 需要 CUDA 13.

0 or newer and an NVIDIA driver upgrade to the versions listed in the release notes.

0 或更新版本，还需将 NVIDIA 驱动升级至发行说明列出的版本。

Ava: Let's do the three-point recap. First?

Ava：最后用三点回顾。第一点？

Brian: First, performance: batched CUDA linalg.

Brian：首先是性能：批量 CUDA linalg.eigh 运算。

eigh can be up to one hundred times faster, and Adagrad gets a fused optimizer step.

其速度最高可提升 100 倍，Adagrad 也增加了融合优化器步骤。

Ava: Second: portability and deployment.

Ava：第二点：可移植性和部署。

Brian: Right. torch. accelerator. Graph unifies graph capture and replay, torch.

Brian：没错。torch.accelerator.Graph 统一了计算图捕获与重放，torch.

export supports MX quantization, and torch. cond can run inside CUDA Graph capture.

export 支持 MX 量化，torch.cond 也能在 CUDA Graph 捕获期间运行。

Ava: Third: production tooling and platform reach.

Ava：第三点：生产工具和平台支持。

Brian: That includes richer distributed profiling, broader ROCm support, Metal-4 offline shaders, and migration work toward torchcomms.

Brian：包括更丰富的分布式性能分析、更广泛的 ROCm 支持、Metal 4 离线着色器，以及向 torchcomms 迁移的工作。

The release has 2,926 commits from 457 contributors, so there is a lot packed in.

这个版本有 457 位贡献者提交的 2,926 次代码提交，内容相当丰富。

Ava: PyTorch 2.

Ava：说到 PyTorch 2.

12 is faster, more hardware-agnostic, and more ready for deployment, while some migration work is still ahead.

12，它更快、更不依赖特定硬件，也更适合部署，不过后续仍有迁移工作要做。

Brian: Exactly.

Brian：正是如此。

Try the features, check the limits for your backend, and report issues as the 2.

试试这些功能，核对所用后端的限制，并随着 2.

x series keeps moving.

x 系列持续更新，及时反馈问题。

Ava: Thanks for listening. We'll be back with another PyTorch release explained.

Ava：感谢收听。下次我们再来解读 PyTorch 的新版本。

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
