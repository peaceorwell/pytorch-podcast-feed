# PyTorch 2.11: Differentiable Collectives, FlashAttention-4, and More

原文：[PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)

## 摘要

本期介绍 PyTorch 2.11 的主要更新，包括可微分集合通信、FlexAttention 的 FlashAttention-4 后端、Apple Silicon MPS 算子扩展，以及 RNN/LSTM GPU 导出支持。对 Hopper 和 Blackwell GPU，FlashAttention-4 在计算受限工作负载上相比现有 Triton 实现可获得一点二到三点二倍加速，但功能仍在积极开发中。节目还讨论了 ROCm 调试与 TopK 优化、Intel XPU Graph、CPU 上通过 OpenBLAS 的 FP16 GEMM，以及 CUDA 13 默认版本。最后总结 TorchScript 弃用、2026 年改为每两个月发布一次，以及使用这些 API-UNSTABLE 功能时需要注意的限制。

## 对话

**Ava:** If your distributed training code needs gradients through communication, PyTorch 2.11 could change what you can build. And if you're on Hopper or Blackwell, attention kernels may get much faster. Brian, what landed in this release?

**Brian:** A lot, actually. PyTorch 2.11 has differentiable collectives, a FlashAttention-4 backend for FlexAttention, a much broader MPS story, GPU export for RNNs and LSTMs, and several device and inference improvements.

**Ava:** Let's start with the distributed training change. The phrase “differentiable collectives” sounds important, but also a little abstract.

**Brian:** Sure. A collective is a communication operation used by distributed workers. In 2.11, functional collectives can support differentiation. That means a training workflow can backpropagate through the collective operation itself.

**Ava:** Wait, so communication can now sit inside the gradient path?

**Brian:** Exactly. That's the key idea. Before this support, researchers might need custom autograd functions for some advanced workflows. With differentiable collectives, those workflows may be implemented without writing those custom functions.

**Ava:** The article calls this a significant step for distributed deep learning research and advanced training techniques. Does it promise a specific new algorithm?

**Brian:** No. It doesn't name a benchmark or one particular algorithm. The claim is about enabling a class of training workflows. The feature is also listed under API-UNSTABLE, so users should expect the interface to keep changing.

**Ava:** Got it. So the practical message is: gradients through communication are now possible, but the API isn't frozen yet.

**Brian:** Right. That's a good way to put it.

**Ava:** Now let's talk about attention. What does the new FlashAttention-4 backend actually do inside FlexAttention?

**Brian:** On Hopper and Blackwell GPUs, FlexAttention can use a FlashAttention-4 backend. PyTorch can automatically generate CuTeDSL score and mask modification functions, then just-in-time instantiate FlashAttention-4 kernels.

**Ava:** Can you unpack that? What should an engineer picture?

**Brian:** Picture FlexAttention as the flexible front end. You describe how scores or masks should be modified. The backend then creates specialized kernels for that description. Here, those kernels come from FlashAttention-4, using CuTeDSL, and they're instantiated just in time.

**Ava:** And the result is faster execution?

**Brian:** The article reports one point two to three point two times speedups over the existing Triton implementation on compute-bound workloads.

**Ava:** That's a wide range. So it depends heavily on the workload.

**Brian:** Yes. The statement is specifically for compute-bound workloads. We shouldn't turn it into a universal speed claim. Also, this backend is still under active development, and its setup details and limitations are described in a separate FlexAttention and FlashAttention-4 post.

**Ava:** So: promising numbers, specific hardware, specific workload, and a moving API.

**Brian:** Exactly. Hopper and Blackwell matter here, and the feature may change as it stabilizes.

**Ava:** Let's move to Apple Silicon. MPS has been expanding for a while. What's new in 2.11?

**Brian:** MPS gets error reporting support and more operator coverage. New distribution functions include log_normal, cauchy, and geometric. There are also operator migrations, such as erfcx, and grid_sampler_2d now supports all operation modes. Finally, baddbmm and addbmm are extended to integer and complex types.

**Ava:** The error reporting sounds especially useful. What problem does it catch?

**Brian:** Asynchronous error reporting can detect out-of-bounds access during GPU indexing operations. The article gives an example where an MPS tensor is indexed with an invalid index, and torch.mps.synchronize raises an index-out-of-bounds error.

**Ava:** So the failure might surface when you synchronize, rather than exactly where the indexing line runs.

**Brian:** Right. That's the important debugging detail. GPU work can be asynchronous, and synchronization is where the error is reported in that example.

**Ava:** Next, export support. The release says RNN modules, including LSTM and GRU, can now be exported on GPUs. Why does that matter?

**Brian:** It broadens the model types that can be deployed with torch.export for production inference. In addition, tracing LSTM with dynamic shapes is now supported.

**Ava:** And the GRU API stays the same?

**Brian:** Yes. The article says the GRU API is unchanged, while the new API is LSTM. So teams working with recurrent models get a larger export path without an API change for GRU.

**Ava:** That sounds practical for production teams. What else is happening on AMD GPUs?

**Brian:** ROCm gets device-side assertions, which should improve debugging. It also gets significant TopK optimizations and radix-select improvements. Those improvements cache data in shared memory, helping both developer experience and performance on AMD GPUs.

**Ava:** Let's switch vendors again. What's XPU Graph?

**Brian:** XPU Graph is for Intel GPUs. It captures a sequence of XPU operations into a runtime execution graph. You can replay that graph multiple times.

**Ava:** And replaying avoids repeated overhead?

**Brian:** Yes. It reduces CPU overhead, including kernel-launch overhead and Python-runtime overhead. The goal is better workload performance on Intel GPUs. The article points users to the API documentation for usage details.

**Ava:** There's also FP16 GEMM on CPUs through OpenBLAS. Is that mainly for servers?

**Brian:** The article frames it as useful for CPU-based deployments, especially edge devices and CPU-only inference scenarios. OpenBLAS now provides FP16 half-precision GEMM support there, which can make FP16 inference faster.

**Ava:** Now, a release detail that can break build environments: CUDA.

**Brian:** Starting with 2.11, CUDA 13 is the default installed version for both x86_64 and ARM platforms. Users who need another build can still use CPU-only packages or CUDA 12.8 builds from the relevant wheel subfolders.

**Ava:** So upgrading may mean checking which CUDA wheel your deployment expects.

**Brian:** Yes, especially if your environment is pinned to CUDA 12.8.

**Ava:** And TorchScript?

**Brian:** TorchScript was deprecated in 2.10. The guidance is to use torch.export instead of the jit trace and script APIs, and to use ExecuTorch instead of the embedded runtime.

**Ava:** So 2.11 reinforces a migration that's already underway.

**Brian:** That's right. The release blog points readers to a PTC talk for more detail, but the main direction is clear: torch.export for export workflows, and ExecuTorch for the embedded runtime.

**Ava:** How large was this release effort?

**Brian:** The release includes 2,723 commits from 432 contributors since PyTorch 2.10. The project thanks the community and encourages users to try the changes and report issues.

**Ava:** And the release rhythm is changing too.

**Brian:** For 2026, PyTorch moves from quarterly releases to one release every two months. That means users may see features arrive more often, but they'll also need to track compatibility more closely.

**Ava:** Before we close, give me the three points I should remember.

**Brian:** First, differentiable functional collectives let gradients pass through collective operations, opening advanced distributed-training workflows without custom autograd functions. Second, FlexAttention can use FlashAttention-4 on Hopper and Blackwell, with reported one point two to three point two times speedups on compute-bound workloads, while the backend remains under active development. Third, 2.11 expands deployment and hardware support: MPS operators and error reporting, GPU export for RNNs and LSTMs, ROCm debugging and TopK work, Intel XPU Graph, and CPU FP16 GEMM.

**Ava:** And the migration notes are CUDA 13 by default, TorchScript deprecated, and releases every two months in 2026.

**Brian:** Exactly. Check the current documentation before relying on the API-UNSTABLE features, and report issues as you try them.

**Ava:** That's PyTorch 2.11. Thanks for listening, and we'll be back with another paper-to-practice conversation.

## 术语

| Term | 释义 |
|---|---|
| Differentiable Collectives | 可微分集合通信；允许梯度通过集合通信操作反向传播。 |
| Functional Collectives | 函数式集合通信 API，用于分布式工作进程之间的通信。 |
| Autograd | 自动微分系统，用于构建和计算反向传播梯度。 |
| FlexAttention | PyTorch 中可灵活定义分数和掩码修改的注意力接口。 |
| FlashAttention-4 | 用于 Hopper 和 Blackwell GPU 的注意力后端。 |
| CuTeDSL | 用于生成分数或掩码修改函数及专用 GPU 内核的领域特定语言。 |
| JIT instantiation | 即时实例化，在运行时生成或专门化内核。 |
| MPS | Apple Silicon GPU 后端，Metal Performance Shaders。 |
| torch.export | 用于导出模型并支持生产推理部署的 PyTorch API。 |
| Device-side assertions | 在 GPU 设备端触发断言，帮助定位运行时错误。 |
| TopK | 选取张量中最大或最小 K 个元素的算子。 |
| XPU Graph | 在 Intel GPU 上捕获并重复执行 XPU 操作序列的运行时图。 |
| GEMM | 通用矩阵乘法，General Matrix-Matrix Multiplication。 |
| OpenBLAS | 提供高性能 BLAS 运算实现的数学库。 |
| ExecuTorch | 用于嵌入式运行时的 PyTorch 部署方案。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| That's the key idea. | 这就是关键点。 |
| Can you unpack that? | 你能把这个讲得具体一点吗？ |
| It depends heavily on the workload. | 这很大程度取决于工作负载。 |
| Let's move to... | 我们转到…… |
| The practical message is... | 实际要点是…… |
| So the failure might surface when... | 所以错误可能会在……时暴露。 |
| Before we close... | 结束之前…… |
| That's a good way to put it. | 这样说很准确。 |
