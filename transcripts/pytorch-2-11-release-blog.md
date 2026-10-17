# PyTorch 2.11: Differentiable Collectives, FlashAttention-4, and More

原文：[PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)

## 摘要

本期介绍 PyTorch 2.11 的主要更新，包括可微分集合通信、FlexAttention 的 FlashAttention-4 后端、Apple Silicon MPS 算子扩展，以及 RNN/LSTM GPU 导出支持。对 Hopper 和 Blackwell GPU，FlashAttention-4 在计算受限工作负载上相比现有 Triton 实现可获得一点二到三点二倍加速，但功能仍在积极开发中。节目还讨论了 ROCm 调试与 TopK 优化、Intel XPU Graph、CPU 上通过 OpenBLAS 的 FP16 GEMM，以及 CUDA 13 默认版本。最后总结 TorchScript 弃用、2026 年改为每两个月发布一次，以及使用这些 API-UNSTABLE 功能时需要注意的限制。

## 对话

Ava: If your distributed training code needs gradients through communication, PyTorch 2.11 could change what you can build. And if you're on Hopper or Blackwell, attention kernels may get much faster. Brian, what landed in this release?

Ava：如果你的分布式训练代码需要让梯度穿过通信操作，PyTorch 2.11 可能会改变你能构建的东西。如果你使用的是 Hopper 或 Blackwell，注意力核也可能快得多。Brian，这次版本发布了哪些功能？

Brian: A lot, actually. PyTorch 2.11 has differentiable collectives, a FlashAttention-4 backend for FlexAttention, a much broader MPS story, GPU export for RNNs and LSTMs, and several device and inference improvements.

Brian：其实有很多。PyTorch 2.11 带来了可微分集合通信、供 FlexAttention 使用的 FlashAttention-4 后端、更广泛的 MPS 支持、RNN 和 LSTM 的 GPU 导出支持，以及多项设备和推理方面的改进。

Ava: Let's start with the distributed training change. The phrase “differentiable collectives” sounds important, but also a little abstract.

Ava：先说分布式训练的变化。“可微分集合通信”听起来很重要，但也有点抽象。

Brian: Sure. A collective is a communication operation used by distributed workers. In 2.11, functional collectives can support differentiation. That means a training workflow can backpropagate through the collective operation itself.

Brian：当然。集合通信是分布式工作节点使用的一种通信操作。在 2.11 中，函数式集合通信可以支持求导。这意味着训练流程可以通过集合通信操作本身进行反向传播。

Ava: Wait, so communication can now sit inside the gradient path?

Ava：等等，也就是说，通信操作现在可以处在梯度传播路径中？

Brian: Exactly. That's the key idea. Before this support, researchers might need custom autograd functions for some advanced workflows. With differentiable collectives, those workflows may be implemented without writing those custom functions.

Brian：没错。这就是核心。此前，对某些高级训练流程，研究人员可能需要编写自定义 autograd 函数。有了可微分集合通信，这些流程或许就不需要编写自定义函数了。

Ava: The article calls this a significant step for distributed deep learning research and advanced training techniques. Does it promise a specific new algorithm?

Ava：文章称，这是分布式深度学习研究和高级训练技术的重要一步。它有承诺支持某种具体的新算法吗？

Brian: No. It doesn't name a benchmark or one particular algorithm. The claim is about enabling a class of training workflows. The feature is also listed under API-UNSTABLE, so users should expect the interface to keep changing.

Brian：没有。文章没有提到基准测试，也没有点名某个特定算法。它说的是，这项功能让一类训练流程成为可能。该功能还被列在 API-UNSTABLE 类别下，因此用户应该预期接口会继续变化。

Ava: Got it. So the practical message is: gradients through communication are now possible, but the API isn't frozen yet.

Ava：明白了。所以实际要点是：梯度现在可以穿过通信操作，但 API 还没有定型。

Brian: Right. That's a good way to put it.

Brian：对，这样概括很准确。

Ava: Now let's talk about attention. What does the new FlashAttention-4 backend actually do inside FlexAttention?

Ava：现在谈谈注意力机制。新的 FlashAttention-4 后端在 FlexAttention 中究竟做了什么？

Brian: On Hopper and Blackwell GPUs, FlexAttention can use a FlashAttention-4 backend. PyTorch can automatically generate CuTeDSL score and mask modification functions, then just-in-time instantiate FlashAttention-4 kernels.

Brian：在 Hopper 和 Blackwell GPU 上，FlexAttention 可以使用 FlashAttention-4 后端。PyTorch 可以自动生成用于修改分数和掩码的 CuTeDSL 函数，然后即时实例化 FlashAttention-4 核。

Ava: Can you unpack that? What should an engineer picture?

Ava：能展开讲讲吗？工程师该如何理解这个过程？

Brian: Picture FlexAttention as the flexible front end. You describe how scores or masks should be modified. The backend then creates specialized kernels for that description. Here, those kernels come from FlashAttention-4, using CuTeDSL, and they're instantiated just in time.

Brian：可以把 FlexAttention 想成灵活的前端。你描述分数或掩码该怎样修改，后端就会根据这个描述创建专用的核。在这里，这些核来自 FlashAttention-4，使用 CuTeDSL，并且会在需要时即时实例化。

Ava: And the result is faster execution?

Ava：结果就是执行速度更快？

Brian: The article reports one point two to three point two times speedups over the existing Triton implementation on compute-bound workloads.

Brian：文章报告称，在计算受限的工作负载上，相比现有的 Triton 实现，速度提高了 1.2 到 3.2 倍。

Ava: That's a wide range. So it depends heavily on the workload.

Ava：这个范围挺宽的。所以具体提升很大程度上取决于工作负载。

Brian: Yes. The statement is specifically for compute-bound workloads. We shouldn't turn it into a universal speed claim. Also, this backend is still under active development, and its setup details and limitations are described in a separate FlexAttention and FlashAttention-4 post.

Brian：是的。这个说法专门针对计算受限的工作负载。我们不该把它当成适用于所有情况的速度承诺。另外，这个后端仍在积极开发中，设置方法和限制在另一篇关于 FlexAttention 和 FlashAttention-4 的文章中有说明。

Ava: So: promising numbers, specific hardware, specific workload, and a moving API.

Ava：所以是：数据很有前景，但针对特定硬件、特定工作负载，而且 API 还在变化。

Brian: Exactly. Hopper and Blackwell matter here, and the feature may change as it stabilizes.

Brian：没错。这里 Hopper 和 Blackwell 很关键，而且这项功能在趋于稳定的过程中可能会变化。

Ava: Let's move to Apple Silicon. MPS has been expanding for a while. What's new in 2.11?

Ava：再来看看 Apple Silicon。MPS 已经扩展了一段时间。2.11 有什么新变化？

Brian: MPS gets error reporting support and more operator coverage. New distribution functions include log_normal, cauchy, and geometric. There are also operator migrations, such as erfcx, and grid_sampler_2d now supports all operation modes. Finally, baddbmm and addbmm are extended to integer and complex types.

Brian：MPS 增加了错误报告支持，并覆盖了更多算子。新增的分布函数包括 log_normal、cauchy 和 geometric。还有一些算子迁移，例如 erfcx；grid_sampler_2d 现在支持所有操作模式。最后，baddbmm 和 addbmm 的支持范围扩展到了整数类型和复数类型。

Ava: The error reporting sounds especially useful. What problem does it catch?

Ava：错误报告听起来特别有用。它能发现什么问题？

Brian: Asynchronous error reporting can detect out-of-bounds access during GPU indexing operations. The article gives an example where an MPS tensor is indexed with an invalid index, and torch.mps.synchronize raises an index-out-of-bounds error.

Brian：异步错误报告可以检测 GPU 索引操作中的越界访问。文章举了一个例子：用无效索引访问 MPS 张量时，torch.mps.synchronize 会抛出索引越界错误。

Ava: So the failure might surface when you synchronize, rather than exactly where the indexing line runs.

Ava：所以错误可能是在同步时才显现，而不是恰好在执行索引操作的那一行显现。

Brian: Right. That's the important debugging detail. GPU work can be asynchronous, and synchronization is where the error is reported in that example.

Brian：对。这是调试时要注意的一点。GPU 工作可能是异步的，在那个例子中，错误是在同步时报告的。

Ava: Next, export support. The release says RNN modules, including LSTM and GRU, can now be exported on GPUs. Why does that matter?

Ava：接下来说导出支持。这次发布称，RNN 模块，包括 LSTM 和 GRU，现在都可以在 GPU 上导出。这为什么重要？

Brian: It broadens the model types that can be deployed with torch.export for production inference. In addition, tracing LSTM with dynamic shapes is now supported.

Brian：这扩大了能够通过 torch.export 部署到生产推理环境的模型类型范围。此外，现在还支持以动态形状追踪 LSTM。

Ava: And the GRU API stays the same?

Ava：GRU API 还是一样的吗？

Brian: Yes. The article says the GRU API is unchanged, while the new API is LSTM. So teams working with recurrent models get a larger export path without an API change for GRU.

Brian：是的。文章说 GRU API 没有变化，而新 API 是 LSTM。因此，使用循环模型的团队有了更广的导出途径，同时不需要为 GRU 适配 API 变化。

Ava: That sounds practical for production teams. What else is happening on AMD GPUs?

Ava：这对生产团队听起来很实用。AMD GPU 方面还有什么变化？

Brian: ROCm gets device-side assertions, which should improve debugging. It also gets significant TopK optimizations and radix-select improvements. Those improvements cache data in shared memory, helping both developer experience and performance on AMD GPUs.

Brian：ROCm 增加了设备端断言，应该能改善调试体验。它还带来了显著的 TopK 优化和 radix-select 改进。这些改进把数据缓存在共享内存中，有助于提升 AMD GPU 上的开发体验和性能。

Ava: Let's switch vendors again. What's XPU Graph?

Ava：再换一家厂商。XPU Graph 是什么？

Brian: XPU Graph is for Intel GPUs. It captures a sequence of XPU operations into a runtime execution graph. You can replay that graph multiple times.

Brian：XPU Graph 面向 Intel GPU。它把一系列 XPU 操作捕获为运行时执行图，而你可以多次重放这个图。

Ava: And replaying avoids repeated overhead?

Ava：重放就能避免重复的开销？

Brian: Yes. It reduces CPU overhead, including kernel-launch overhead and Python-runtime overhead. The goal is better workload performance on Intel GPUs. The article points users to the API documentation for usage details.

Brian：是的。它能减少 CPU 开销，包括内核启动开销和 Python 运行时开销。目标是提升 Intel GPU 上的工作负载性能。文章建议用户查看 API 文档了解用法细节。

Ava: There's also FP16 GEMM on CPUs through OpenBLAS. Is that mainly for servers?

Ava：还有通过 OpenBLAS 在 CPU 上运行的 FP16 GEMM。这主要是给服务器用的吗？

Brian: The article frames it as useful for CPU-based deployments, especially edge devices and CPU-only inference scenarios. OpenBLAS now provides FP16 half-precision GEMM support there, which can make FP16 inference faster.

Brian：文章认为它适用于基于 CPU 的部署，尤其是边缘设备和纯 CPU 推理场景。OpenBLAS 现在为这些场景提供 FP16 半精度 GEMM 支持，可以加快 FP16 推理。

Ava: Now, a release detail that can break build environments: CUDA.

Ava：现在说一个可能让构建环境出问题的发布细节：CUDA。

Brian: Starting with 2.11, CUDA 13 is the default installed version for both x86_64 and ARM platforms. Users who need another build can still use CPU-only packages or CUDA 12.8 builds from the relevant wheel subfolders.

Brian：从 2.11 开始，x86_64 和 ARM 平台默认安装的 CUDA 版本都是 CUDA 13。需要其他构建版本的用户仍可从相应的 wheel 子目录获取纯 CPU 软件包或 CUDA 12.8 构建版本。

Ava: So upgrading may mean checking which CUDA wheel your deployment expects.

Ava：所以升级时可能需要检查部署环境预期使用哪个 CUDA wheel。

Brian: Yes, especially if your environment is pinned to CUDA 12.8.

Brian：是的，尤其是当你的环境固定使用 CUDA 12.8 时。

Ava: And TorchScript?

Ava：那 TorchScript 呢？

Brian: TorchScript was deprecated in 2.10. The guidance is to use torch.export instead of the jit trace and script APIs, and to use ExecuTorch instead of the embedded runtime.

Brian：TorchScript 在 2.10 中已被弃用。建议用 torch.export 代替 jit trace 和 script API，并用 ExecuTorch 代替嵌入式运行时。

Ava: So 2.11 reinforces a migration that's already underway.

Ava：所以 2.11 再次强调了已经开始的迁移方向。

Brian: That's right. The release blog points readers to a PTC talk for more detail, but the main direction is clear: torch.export for export workflows, and ExecuTorch for the embedded runtime.

Brian：没错。发布博客建议读者观看一场 PTC 演讲了解更多细节，但主要方向已经很明确：导出流程使用 torch.export，嵌入式运行时使用 ExecuTorch。

Ava: How large was this release effort?

Ava：这次版本发布的工作规模有多大？

Brian: The release includes 2,723 commits from 432 contributors since PyTorch 2.10. The project thanks the community and encourages users to try the changes and report issues.

Brian：自 PyTorch 2.10 以来，这次发布汇集了 432 位贡献者的 2,723 次提交。项目向社区表示感谢，并鼓励用户试用这些改动、报告问题。

Ava: And the release rhythm is changing too.

Ava：发布节奏也在变化。

Brian: For 2026, PyTorch moves from quarterly releases to one release every two months. That means users may see features arrive more often, but they'll also need to track compatibility more closely.

Brian：在 2026 年，PyTorch 将从每季度发布一次改为每两个月发布一次。这意味着用户可能会更频繁地看到新功能，但也需要更密切地关注兼容性。

Ava: Before we close, give me the three points I should remember.

Ava：结束前，告诉我应该记住哪三点。

Brian: First, differentiable functional collectives let gradients pass through collective operations, opening advanced distributed-training workflows without custom autograd functions. Second, FlexAttention can use FlashAttention-4 on Hopper and Blackwell, with reported one point two to three point two times speedups on compute-bound workloads, while the backend remains under active development. Third, 2.11 expands deployment and hardware support: MPS operators and error reporting, GPU export for RNNs and LSTMs, ROCm debugging and TopK work, Intel XPU Graph, and CPU FP16 GEMM.

Brian：首先，可微分的函数式集合通信操作允许梯度通过集合通信操作，从而无需自定义 autograd 函数即可实现高级分布式训练流程。其次，FlexAttention 可以在 Hopper 和 Blackwell 上使用 FlashAttention-4；据报告，在受计算性能限制的工作负载上可实现 1.2 到 3.2 倍的加速，但该后端仍在积极开发中。第三，2.11 扩展了部署和硬件支持，包括 MPS 算子和错误报告、RNN 和 LSTM 的 GPU 导出、ROCm 调试和 TopK 相关工作、Intel XPU Graph，以及 CPU FP16 GEMM。

Ava: And the migration notes are CUDA 13 by default, TorchScript deprecated, and releases every two months in 2026.

Ava：迁移注意事项则是默认使用 CUDA 13、TorchScript 已弃用，以及 2026 年每两个月发布一个版本。

Brian: Exactly. Check the current documentation before relying on the API-UNSTABLE features, and report issues as you try them.

Brian：没错。在依赖 API-UNSTABLE 功能之前，请查看最新文档；试用时如果遇到问题，也请报告。

Ava: That's PyTorch 2.11. Thanks for listening, and we'll be back with another paper-to-practice conversation.

Ava：以上就是 PyTorch 2.11。感谢收听，我们下次再聊另一场从论文到实践的对话。

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
