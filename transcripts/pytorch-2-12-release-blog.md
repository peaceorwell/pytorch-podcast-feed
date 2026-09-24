# PyTorch 2.12: Faster Math, Unified Graphs, and Wider Hardware Support

原文：[PyTorch 2.12 Release Blog](https://pytorch.org/blog/pytorch-2-12-release-blog/)

## 摘要

本期介绍 PyTorch 2.12 的主要更新：CUDA 上批量特征值分解最高提速一百倍，新增跨硬件的 torch.accelerator.Graph，并让 torch.export 支持 Microscaling 量化。我们还讨论了融合版 Adagrad、torch.cond 在 CUDA Graph 中的捕获、分布式训练分析工具，以及 ROCm、Apple MPS 等平台更新。最后，两位主持人解释了 torchcomms 和 TorchScript 的后续变化，以及 CUDA wheel 的弃用安排。

## 对话

**Ava:** If your training job spends minutes solving lots of small eigenvalue problems, PyTorch 2.12 may change that to seconds. And the headline number is up to one hundred times faster on CUDA.

**Brian:** Right. This release is also about making PyTorch feel more consistent across hardware. We get a device-agnostic graph API, wider export support, and several distributed and ROCm improvements.

**Ava:** So let's start with the speed story. What exactly changed for batched linalg.eigh?

**Brian:** linalg.eigh computes eigenvalues and eigenvectors for symmetric or Hermitian matrices. In earlier releases, the CUDA path could dispatch each matrix solve separately. That created a big performance gap, especially when you had many small or medium matrices.

**Ava:** So the GPU was doing lots of tiny jobs instead of one big batch?

**Brian:** Exactly. PyTorch has overhauled backend selection. The old MAGMA backend was deprecated in favor of cuSolver, and the dispatch heuristics now use cuSolver's syevj_batched kernel unconditionally for this case.

**Ava:** And that kernel is designed for many matrices at once.

**Brian:** Yes. It processes the batch as a single GPU operation. The article says workloads that used to take minutes can now run in seconds, with speedups of up to one hundred times over the previous release.

**Ava:** That sounds huge, but the wording matters: up to one hundred times, not every workload.

**Brian:** Correct. The gain depends on the workload. The release calls out scientific computing and machine learning jobs that rely on batched eigendecompositions. It also says this helps close longstanding performance gaps with CuPy.

**Ava:** Let's move to optimizers. Adagrad now has fused equals true. What does fused mean here?

**Brian:** A fused optimizer performs the whole optimizer step in one CUDA kernel. Without fusion, separate operations launch separate kernels. Fusion reduces kernel launch overhead and memory traffic.

**Ava:** So Adagrad joins Adam, AdamW, and SGD in having a fused variant.

**Brian:** Exactly. The CUDA kernel was contributed during the 2.11 cycle, and the Python frontend is now exposed in 2.12. The practical idea is simple: fewer launches and less movement of intermediate data.

**Ava:** Now, the bigger architectural change seems to be torch.accelerator.Graph.

**Brian:** Yes. torch.accelerator.Graph is a new device-agnostic API for graph capture and replay. Instead of users learning a separate graph interface for every backend, they get one common abstraction.

**Ava:** Can you give a plain-language analogy?

**Brian:** Think of recording a fixed sequence of GPU work and replaying that recording many times. The API is like a universal remote. CUDA, XPU, and out-of-tree backends can keep their own internal implementations, but users interact with a consistent control surface.

**Ava:** How do backends plug into it?

**Brian:** Each backend can register an implementation through GraphImplInterface. That keeps backend autonomy while giving applications a shared API. The release has initial XPU support and an extension path for out-of-tree backends through PrivateUse1.

**Ava:** There is also a stream change, right?

**Brian:** Right. c10d::Stream and torch.Stream now expose is_capturing(), which replaces the device-specific is_current_stream_capturing. Stream context manager reentrance was fixed too. Together, these changes improve parity across backends.

**Ava:** What about export and quantization?

**Brian:** torch.export.save and torch.export.load now support Microscaling, or MX, quantization formats. Before 2.12, export could not handle the float8_e8m0fnu dtype used as the shared block-scale exponent in MXFP4, MXFP6, and MXFP8.

**Ava:** So a model could use MX quantization, but the export-to-deployment path was blocked.

**Brian:** Exactly. Now tensors with that dtype can be serialized and deserialized correctly. That unblocks the full workflow for aggressively compressed models, which matters for large language models deployed in cost-constrained or edge environments.

**Ava:** That is a nice example of a small missing dtype support causing a much larger production problem.

**Brian:** Yes. The quantization method existed, but the deployment handoff was incomplete. This release closes that gap.

**Ava:** The release also mentions torch.cond inside CUDA Graphs. Why was that difficult?

**Brian:** torch.cond represents data-dependent control flow. Previously, branches were evaluated on the CPU, so CUDA graph capture had to fall back to CUDA graph trees. With CUDA 12.4 conditional IF nodes, both branches can now be evaluated on the GPU inside one graph capture.

**Ava:** So the graph can contain the decision itself.

**Brian:** Exactly. It can capture and replay the control-flow region. The current support works with the eager and cudagraphs backends. Inductor support is planned for a future release, so that is one limitation to remember.

**Ava:** There is also a numerical correctness update for XPU involving addcdiv.

**Brian:** Yes. addcdiv is a fused arithmetic operation: input plus value times tensor one divided by tensor two. It appears in optimizer update rules such as Adam, AdamW, and RMSprop.

**Ava:** What was going wrong?

**Brian:** Inductor previously lowered the operation with separate multiply and divide instructions. That could create small floating-point rounding differences compared with eager CUDA execution. Over thousands of steps, those differences can accumulate.

**Ava:** And now it uses fused multiply-add instructions?

**Brian:** For CUDA first, and now for XPU as well. The goal is bitwise numerical parity with eager CUDA execution while preserving Triton kernel fusion benefits. That helps teams validate compiled optimizer-heavy training loops on NVIDIA and Intel hardware.

**Ava:** Let's talk distributed training. What changed for custom operators?

**Brian:** Custom operators can now accept ProcessGroup objects directly. Callers no longer have to turn a group into a string name and look it up in a global registry. The c10d functional collective operations, including all_reduce and reduce_scatter, accept both objects and string names.

**Ava:** That sounds like a cleaner interface.

**Brian:** It is, and profiling got richer too. The PyTorch Profiler Events API now exposes flow IDs, flow types, activity types, unfinished events, and Python function events. Its events() output is closer to the Chrome trace JSON output.

**Ava:** How does that help with multi-node debugging?

**Brian:** NCCL collective traces can now be correlated across ranks with a seq_num field. All ranks participating in the same collective share the same sequence number within a process group. That makes post-hoc analysis much more practical.

**Ava:** And FlightRecorder covers more communication backends.

**Brian:** Yes. Its trace analyzer now supports ncclx and gloo alongside nccl and xccl. It also recognizes torchcomms operations such as all_gather_single, reduce_scatter_v, and barrier. A race condition involving multiple process groups was fixed as well.

**Ava:** What platform updates should ROCm users notice?

**Brian:** Several. On AMD GPUs with ROCm 7.02 or newer, the caching allocator supports expandable memory segments. That dynamically grows allocations through virtual memory APIs and reduces fragmentation, matching the CUDA feature.

**Ava:** There is also rocSHMEM.

**Brian:** Right. rocSHMEM brings symmetric memory collective operations to AMD GPUs through torch.ops.symm_mem. It ports on-GPU primitives such as point-to-point, broadcast, all-to-all, and MoE-oriented two-dimensional AllToAllv.

**Ava:** And sparse computation?

**Brian:** hipSPARSELt is enabled by default in ROCm builds at 7.12 or newer. It adds semi-structured two-to-four sparsity support, and FP8 inputs are supported on MI350X with FP32 output. That enables the torch._cslt_sparse_mm acceleration path on AMD.

**Ava:** FlexAttention also gets a pipeline change.

**Brian:** Yes. On AMD GPUs, the Triton backend now uses two-stage pipelining for FlexAttention. The article reports five to twenty-six percent speedups across causal, alibi, and sliding-window attention patterns on MI350X. The change was just a one-line configuration adjustment from one stage to two.

**Ava:** What about Apple users?

**Brian:** Apple Silicon binary wheels now ship with ahead-of-time-compiled Metal-4 shaders. They were built on macOS 26 with the Metal-4 standard, so MPS workloads avoid runtime shader compilation on first run and get lower startup latency.

**Ava:** Before we wrap up, we need the migration warnings.

**Brian:** The biggest future change is torchcomms. In an upcoming release, 2.13 or later, PyTorch plans to use torchcomms by default in Distributed. That brings breaking changes to ProcessGroup behavior, even though the team aims to make most migrations automatic.

**Ava:** What should engineers watch for?

**Brian:** ProcessGroups and communicators will need eager initialization during dist.init_process_group, with one backend device. P2P operations on the same group and stream will not be guaranteed to run concurrently; concurrent operations will need batch APIs or a separate group. PyTorch also plans to make torchcomms a required package and deprecate the existing c10d backends.

**Ava:** And TorchScript?

**Brian:** TorchScript is now deprecated. It was deprecated in 2.10. The recommended direction is torch.export instead of the jit trace and script APIs, and Executorch instead of the embedded runtime.

**Ava:** There is one more packaging detail: CUDA 12.8 wheels.

**Brian:** Starting with 2.12, the CUDA 12.8 binary wheel is deprecated and will no longer be in the standard release matrix. The default wheel is CUDA 13.0, CUDA 13.2 is experimental, and CUDA 12.6 remains supported for older architectures. Newer GPUs need CUDA 13.0 or newer and an NVIDIA driver upgrade to the versions listed in the release notes.

**Ava:** Let's do the three-point recap. First?

**Brian:** First, performance: batched CUDA linalg.eigh can be up to one hundred times faster, and Adagrad gets a fused optimizer step.

**Ava:** Second: portability and deployment.

**Brian:** Right. torch.accelerator.Graph unifies graph capture and replay, torch.export supports MX quantization, and torch.cond can run inside CUDA Graph capture.

**Ava:** Third: production tooling and platform reach.

**Brian:** That includes richer distributed profiling, broader ROCm support, Metal-4 offline shaders, and migration work toward torchcomms. The release has 2,926 commits from 457 contributors, so there is a lot packed in.

**Ava:** PyTorch 2.12 is faster, more hardware-agnostic, and more ready for deployment, while some migration work is still ahead.

**Brian:** Exactly. Try the features, check the limits for your backend, and report issues as the 2.x series keeps moving.

**Ava:** Thanks for listening. We'll be back with another PyTorch release explained.

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
