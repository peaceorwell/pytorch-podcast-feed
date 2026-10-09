# PyTorch 2.11: Differentiable Collectives, FlashAttention-4, and More

原文：[PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)

## 摘要

本期介绍 PyTorch 2.11 的主要更新，包括可微分集合通信、FlexAttention 的 FlashAttention-4 后端、Apple Silicon MPS 算子扩展，以及 RNN/LSTM GPU 导出支持。对 Hopper 和 Blackwell GPU，FlashAttention-4 在计算受限工作负载上相比现有 Triton 实现可获得一点二到三点二倍加速，但功能仍在积极开发中。节目还讨论了 ROCm 调试与 TopK 优化、Intel XPU Graph、CPU 上通过 OpenBLAS 的 FP16 GEMM，以及 CUDA 13 默认版本。最后总结 TorchScript 弃用、2026 年改为每两个月发布一次，以及使用这些 API-UNSTABLE 功能时需要注意的限制。

## 对话

Ava: If your distributed training code needs gradients through communication, PyTorch 2.

Ava：如果分布式训练代码需要让梯度穿过通信操作，PyTorch 2.

11 could change what you can build.

11 可能会带来新的实现方式。

And if you're on Hopper or Blackwell, attention kernels may get much faster.

如果你用的是 Hopper 或 Blackwell，注意力内核也可能快得多。

Brian, what landed in this release?

Brian，这次发布了什么？

Brian: A lot, actually. PyTorch 2.

Brian：其实有很多。PyTorch 2.

11 has differentiable collectives, a FlashAttention-4 backend for FlexAttention, a much broader MPS story, GPU export for RNNs and LSTMs, and several device and inference improvements.

11 加入了可微集合通信、FlexAttention 的 FlashAttention-4 后端、更完善的 MPS 支持、RNN 和 LSTM 的 GPU 导出，以及多项设备和推理改进。

Ava: Let's start with the distributed training change.

Ava：先说说分布式训练的变化。

The phrase “differentiable collectives” sounds important, but also a little abstract.

“可微集合通信”听起来很重要，但也有点抽象。

Brian: Sure. A collective is a communication operation used by distributed workers. In 2.

Brian：集合通信是分布式工作进程使用的一类通信操作。在 2.

11, functional collectives can support differentiation.

11 中，函数式集合通信操作支持求导。

That means a training workflow can backpropagate through the collective operation itself.

也就是说，训练过程可以穿过集合通信操作本身进行反向传播。

Ava: Wait, so communication can now sit inside the gradient path?

Ava：等等，通信操作现在也能处在梯度传播路径上？

Brian: Exactly. That's the key idea.

Brian：没错，这就是关键。

Before this support, researchers might need custom autograd functions for some advanced workflows.

以前，一些高级训练流程可能需要研究人员编写自定义 autograd 函数。

With differentiable collectives, those workflows may be implemented without writing those custom functions.

有了可微集合通信，这些流程或许无需自定义函数就能实现。

Ava: The article calls this a significant step for distributed deep learning research and advanced training techniques.

Ava：文章说，这是分布式深度学习研究和高级训练技术的一大进步。

Does it promise a specific new algorithm?

它有承诺支持某种具体的新算法吗？

Brian: No. It doesn't name a benchmark or one particular algorithm.

Brian：没有。文章没有提具体算法或基准测试。

The claim is about enabling a class of training workflows.

它强调的是，这项功能让一类训练流程成为可能。

The feature is also listed under API-UNSTABLE, so users should expect the interface to keep changing.

这项功能还被标为 API-UNSTABLE，所以接口今后可能继续变化。

Ava: Got it.

Ava：明白了。

So the practical message is: gradients through communication are now possible, but the API isn't frozen yet.

所以实际意义是：梯度现在能穿过通信操作，但 API 还没定型。

Brian: Right. That's a good way to put it.

Brian：对，可以这么理解。

Ava: Now let's talk about attention.

Ava：接着聊注意力机制。

What does the new FlashAttention-4 backend actually do inside FlexAttention?

新的 FlashAttention-4 后端在 FlexAttention 中具体做什么？

Brian: On Hopper and Blackwell GPUs, FlexAttention can use a FlashAttention-4 backend.

Brian：在 Hopper 和 Blackwell GPU 上，FlexAttention 可以使用 FlashAttention-4 后端。

PyTorch can automatically generate CuTeDSL score and mask modification functions, then just-in-time instantiate FlashAttention-4 kernels.

PyTorch 可以自动生成用于修改分数和掩码的 CuTeDSL 函数，再即时实例化 FlashAttention-4 内核。

Ava: Can you unpack that? What should an engineer picture?

Ava：能说得更具体些吗？工程师该怎么理解？

Brian: Picture FlexAttention as the flexible front end.

Brian：可以把 FlexAttention 看成灵活的前端。

You describe how scores or masks should be modified.

你描述分数或掩码该如何修改。

The backend then creates specialized kernels for that description.

后端再根据这套描述生成专用内核。

Here, those kernels come from FlashAttention-4, using CuTeDSL, and they're instantiated just in time.

这里的内核基于 FlashAttention-4，使用 CuTeDSL，并按需即时实例化。

Ava: And the result is faster execution?

Ava：所以运行会更快？

Brian: The article reports one point two to three point two times speedups over the existing Triton implementation on compute-bound workloads.

Brian：文章称，在计算密集型工作负载上，相比现有的 Triton 实现，速度提升了 1.2 到 3.2 倍。

Ava: That's a wide range. So it depends heavily on the workload.

Ava：这个范围挺大，看来很依赖具体工作负载。

Brian: Yes. The statement is specifically for compute-bound workloads.

Brian：对，这个数据专指计算密集型工作负载。

We shouldn't turn it into a universal speed claim.

不能说所有场景都能达到这个速度提升。

Also, this backend is still under active development, and its setup details and limitations are described in a separate FlexAttention and FlashAttention-4 post.

而且后端仍在积极开发中，配置细节和限制另有一篇关于 FlexAttention 和 FlashAttention-4 的文章说明。

Ava: So: promising numbers, specific hardware, specific workload, and a moving API.

Ava：也就是说，数据很有前景，但限定了硬件和工作负载，API 也还在变化。

Brian: Exactly. Hopper and Blackwell matter here, and the feature may change as it stabilizes.

Brian：没错。这里需要 Hopper 或 Blackwell，功能在稳定下来之前也可能变化。

Ava: Let's move to Apple Silicon. MPS has been expanding for a while. What's new in 2. 11?

Ava：再聊聊 Apple Silicon。MPS 一直在扩展，2.11 有什么新变化？

Brian: MPS gets error reporting support and more operator coverage.

Brian：MPS 增加了错误报告支持，也支持了更多算子。

New distribution functions include log_normal, cauchy, and geometric.

新增的分布函数包括 log_normal、cauchy 和 geometric。

There are also operator migrations, such as erfcx, and grid_sampler_2d now supports all operation modes.

还有 erfcx 等算子迁移到了 MPS；grid_sampler_2d 现在支持所有操作模式。

Finally, baddbmm and addbmm are extended to integer and complex types.

最后，baddbmm 和 addbmm 扩展了对整数和复数类型的支持。

Ava: The error reporting sounds especially useful. What problem does it catch?

Ava：错误报告听起来特别实用。它能发现什么问题？

Brian: Asynchronous error reporting can detect out-of-bounds access during GPU indexing operations.

Brian：异步错误报告能检测 GPU 索引操作中的越界访问。

The article gives an example where an MPS tensor is indexed with an invalid index, and torch.

文章举了个例子：用无效索引访问 MPS 张量时，torch.

mps. synchronize raises an index-out-of-bounds error.

mps.synchronize 会抛出索引越界错误。

Ava: So the failure might surface when you synchronize, rather than exactly where the indexing line runs.

Ava：所以错误可能在同步时才显现，而不是恰好在执行索引那一行时。

Brian: Right. That's the important debugging detail.

Brian：对，这正是调试时的关键细节。

GPU work can be asynchronous, and synchronization is where the error is reported in that example.

GPU 操作可能是异步的；在这个例子中，错误会在同步时报告。

Ava: Next, export support.

Ava：接下来聊导出支持。

The release says RNN modules, including LSTM and GRU, can now be exported on GPUs.

发布说明称，LSTM 和 GRU 等 RNN 模块现在可以在 GPU 上导出。

Why does that matter?

这有什么意义？

Brian: It broadens the model types that can be deployed with torch.

Brian：这让更多类型的模型可以通过 torch.

export for production inference.

export 导出，用于生产环境推理。

In addition, tracing LSTM with dynamic shapes is now supported.

此外，现在也支持以动态形状追踪 LSTM。

Ava: And the GRU API stays the same?

Ava：GRU API 还是一样的吗？

Brian: Yes. The article says the GRU API is unchanged, while the new API is LSTM.

Brian：是的。文章说 GRU API 没有变化，新 API 针对的是 LSTM。

So teams working with recurrent models get a larger export path without an API change for GRU.

因此，使用循环模型的团队有了更广的导出选择，而无需更改 GRU API。

Ava: That sounds practical for production teams. What else is happening on AMD GPUs?

Ava：这对生产团队很实用。AMD GPU 还有什么新变化？

Brian: ROCm gets device-side assertions, which should improve debugging.

Brian：ROCm 增加了设备端断言，应该能让调试更方便。

It also gets significant TopK optimizations and radix-select improvements.

它还大幅优化了 TopK，并改进了基数选择。

Those improvements cache data in shared memory, helping both developer experience and performance on AMD GPUs.

这些改进将数据缓存在共享内存中，有助于改善开发体验并提升 AMD GPU 的性能。

Ava: Let's switch vendors again. What's XPU Graph?

Ava：再换一家厂商。XPU Graph 是什么？

Brian: XPU Graph is for Intel GPUs.

Brian：XPU Graph 面向 Intel GPU。

It captures a sequence of XPU operations into a runtime execution graph.

它把一系列 XPU 操作捕获为运行时执行图。

You can replay that graph multiple times.

这个图可以重复执行多次。

Ava: And replaying avoids repeated overhead?

Ava：重复执行就能避免反复产生开销？

Brian: Yes.

Brian：对。

It reduces CPU overhead, including kernel-launch overhead and Python-runtime overhead.

它能降低 CPU 开销，包括内核启动和 Python 运行时的开销。

The goal is better workload performance on Intel GPUs.

目标是提升 Intel GPU 上的工作负载性能。

The article points users to the API documentation for usage details.

文章建议用户查阅 API 文档，了解具体用法。

Ava: There's also FP16 GEMM on CPUs through OpenBLAS. Is that mainly for servers?

Ava：CPU 上也能通过 OpenBLAS 使用 FP16 GEMM。这主要面向服务器吗？

Brian: The article frames it as useful for CPU-based deployments, especially edge devices and CPU-only inference scenarios.

Brian：文章认为它适用于基于 CPU 的部署，尤其是边缘设备和纯 CPU 推理场景。

OpenBLAS now provides FP16 half-precision GEMM support there, which can make FP16 inference faster.

OpenBLAS 现在为这些场景提供 FP16 半精度 GEMM 支持，可加快 FP16 推理。

Ava: Now, a release detail that can break build environments: CUDA.

Ava：接下来是一个可能影响构建环境的发布变更：CUDA。

Brian: Starting with 2.

Brian：从 2.

11, CUDA 13 is the default installed version for both x86_64 and ARM platforms.

11 版起，x86_64 和 ARM 平台默认安装 CUDA 13。

Users who need another build can still use CPU-only packages or CUDA 12.

需要其他构建版本的用户，仍可使用纯 CPU 软件包或 CUDA 12.

8 builds from the relevant wheel subfolders.

8 构建版本，可从相应的 wheel 子目录获取。

Ava: So upgrading may mean checking which CUDA wheel your deployment expects.

Ava：所以升级时可能需要检查部署环境要求哪个 CUDA wheel。

Brian: Yes, especially if your environment is pinned to CUDA 12. 8.

Brian：对，尤其是环境固定使用 CUDA 12.8 的情况。

Ava: And TorchScript?

Ava：TorchScript 呢？

Brian: TorchScript was deprecated in 2. 10. The guidance is to use torch.

Brian：TorchScript 已在 2.10 版弃用。建议使用 torch.

export instead of the jit trace and script APIs, and to use ExecuTorch instead of the embedded runtime.

export 取代 jit 的 trace 和 script API，并用 ExecuTorch 取代嵌入式运行时。

Ava: So 2. 11 reinforces a migration that's already underway.

Ava：所以 2.11 进一步推动了已经开始的迁移。

Brian: That's right.

Brian：没错。

The release blog points readers to a PTC talk for more detail, but the main direction is clear: torch.

发布博客推荐读者观看 PTC 演讲了解详情，但主要方向很明确：用 torch.

export for export workflows, and ExecuTorch for the embedded runtime.

export 处理导出工作流，用 ExecuTorch 作为嵌入式运行时。

Ava: How large was this release effort?

Ava：这次发布投入了多少工作？

Brian: The release includes 2,723 commits from 432 contributors since PyTorch 2. 10.

Brian：自 PyTorch 2.10 以来，这次发布汇集了 432 位贡献者的 2,723 次提交。

The project thanks the community and encourages users to try the changes and report issues.

项目组感谢社区，并鼓励用户试用这些更新、报告问题。

Ava: And the release rhythm is changing too.

Ava：发布节奏也在变化。

Brian: For 2026, PyTorch moves from quarterly releases to one release every two months.

Brian：2026 年，PyTorch 将从每季度发布一次改为每两个月发布一次。

That means users may see features arrive more often, but they'll also need to track compatibility more closely.

这意味着新功能可能更频繁地推出，用户也需要更密切地关注兼容性。

Ava: Before we close, give me the three points I should remember.

Ava：结束前，说说我该记住的三个要点。

Brian: First, differentiable functional collectives let gradients pass through collective operations, opening advanced distributed-training workflows without custom autograd functions.

Brian：第一，可微的函数式集合通信让梯度能够穿过集合通信操作，无需自定义 autograd 函数即可支持更高级的分布式训练工作流。

Second, FlexAttention can use FlashAttention-4 on Hopper and Blackwell, with reported one point two to three point two times speedups on compute-bound workloads, while the backend remains under active development.

第二，FlexAttention 可以在 Hopper 和 Blackwell 上使用 FlashAttention-4。据报告，计算受限的工作负载可提速 1.2 到 3.2 倍，但该后端仍在持续开发中。

Third, 2.

第三，2。

11 expands deployment and hardware support: MPS operators and error reporting, GPU export for RNNs and LSTMs, ROCm debugging and TopK work, Intel XPU Graph, and CPU FP16 GEMM.

11 扩展了部署和硬件支持：MPS 算子及错误报告、RNN 和 LSTM 的 GPU 导出、ROCm 调试和 TopK 改进、Intel XPU Graph，以及 CPU FP16 GEMM。

Ava: And the migration notes are CUDA 13 by default, TorchScript deprecated, and releases every two months in 2026.

Ava：迁移时还要注意：默认使用 CUDA 13，TorchScript 已弃用，2026 年每两个月发布一个版本。

Brian: Exactly.

Brian：没错。

Check the current documentation before relying on the API-UNSTABLE features, and report issues as you try them.

使用标记为 API-UNSTABLE 的功能前，请先查看最新文档；试用时遇到问题也请反馈。

Ava: That's PyTorch 2. 11.

Ava：以上就是 PyTorch 2.11。

Thanks for listening, and we'll be back with another paper-to-practice conversation.

感谢收听，我们下期再聊如何把论文成果用于实践。

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
