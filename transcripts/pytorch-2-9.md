# PyTorch 2.9: More Portable Kernels, Smarter Compilation, and Wider Hardware Support

原文：[PyTorch 2.9 Release Blog](https://pytorch.org/blog/pytorch-2-9/)

## 摘要

本期介绍 PyTorch 2.9 的主要更新，包括稳定的 libtorch ABI、Symmetric Memory、多 GPU 内核编程，以及 torch.compile 图断点控制。我们还讨论了 ROCm、XPU、CUDA 13 的 wheel variant 支持、Intel GPU 上的 FlexAttention，以及基于 FlexAttention 的 X86 CPU flash decoding。最后回顾 Arm 平台优化、当前 API 仍处于不稳定或预览阶段的限制，以及后续工作方向。

## 对话

Ava: If you build C++ or CUDA extensions, run models across different GPUs, or care about long-context inference, PyTorch 2.9 has something aimed directly at you. Brian, what makes this release worth a close listen?

Ava：如果你开发 C++ 或 CUDA 扩展、在不同 GPU 上运行模型，或者关注长上下文推理，PyTorch 2.9 都有直接面向你的新内容。Brian，为什么这次发布值得仔细听听？

Brian: It’s a broad release rather than one single headline feature. PyTorch 2.9 brings a more stable libtorch ABI, easier multi-GPU kernel programming through Symmetric Memory, finer control over graph breaks in torch.compile, and wider hardware packaging. It also has attention optimizations for Intel GPUs and X86 CPUs.

Brian：这次发布的内容很广，不是只有一个最突出的功能。PyTorch 2.9 带来了更稳定的 libtorch ABI、通过 Symmetric Memory 更轻松地编写多 GPU 内核、对 torch.compile 中的图中断进行更精细的控制，以及覆盖更多硬件的安装包。它还针对 Intel GPU 和 X86 CPU 优化了注意力计算。

Ava: Let’s start with the ABI piece. I’ve heard extension authors complain that a compiled extension can be tied too closely to one PyTorch version. Does 2.9 address that?

Ava：先说 ABI。我听扩展开发者抱怨过，编译好的扩展可能与某个 PyTorch 版本绑定得太紧。2.9 解决了这个问题吗？

Brian: That’s exactly the motivation. ABI means application binary interface. PyTorch is building a stable ABI with C++ convenience wrappers, so you can build an extension with one torch version and run it with another. In 2.9, that surface gets several new APIs.

Brian：这正是开发它的动机。ABI 指应用程序二进制接口。PyTorch 正在构建一个带有 C++ 便捷封装的稳定 ABI，让你能用一个 torch 版本构建扩展，再用另一个版本运行。到了 2.9，这个接口又增加了几个新 API。

Ava: What kind of APIs are we talking about?

Ava：具体是哪些 API？

Brian: There are device utilities, such as Device Guard and Stream, available through the stable accelerator header. The stable torch::stable::Tensor also gets a default constructor, is_cpu, scalar_type, and get_device_index. And more stable ATen operations are exposed, including amax, narrow, dtype variants of new_empty and new_zeros, and pad.

Brian：稳定的 accelerator 头文件提供了 Device Guard 和 Stream 等设备工具。稳定版 torch::stable::Tensor 也增加了默认构造函数，以及 is_cpu、scalar_type 和 get_device_index。此外，更多稳定的 ATen 操作已开放，包括 amax、narrow、new_empty 和 new_zeros 的 dtype 变体，以及 pad。

Ava: So this is useful for people maintaining custom C++ or CUDA extensions, but it’s not a finished contract yet?

Ava：所以这对维护自定义 C++ 或 CUDA 扩展的人很有用，但这套接口约定还没有最终定型？

Brian: Right. The team enabled a libtorch-ABI wheel for Flash-Attention 3, which shows the direction. But the high-level C++ APIs are still in preview. PyTorch says it’s continuing to expand the ABI surface, establish versioning, write more documentation, and make more custom kernels ABI stable.

Brian：对。团队为 Flash-Attention 3 启用了 libtorch-ABI wheel，这展示了他们的发展方向。不过，高层 C++ API 仍处于预览阶段。PyTorch 表示会继续扩展 ABI 接口范围、建立版本管理机制、编写更多文档，并让更多自定义内核实现 ABI 稳定。

Ava: That sounds like a compatibility layer being built piece by piece. Now, Symmetric Memory sounds more fundamental. What problem does it solve?

Ava：听起来，这个兼容层正在逐步建成。接下来，Symmetric Memory 听起来更基础。它解决什么问题？

Brian: It makes multi-GPU kernels easier to program across NVLinks and RDMA networks. The key idea is that a GPU kernel can communicate while it computes. A kernel can issue puts and gets interleaved with computation instructions, so communication can be fused at a very small granularity.

Brian：它让开发者更容易编写通过 NVLinks 和 RDMA 网络协作的多 GPU 内核。核心思路是让 GPU 内核一边计算一边通信。内核可以在计算指令之间穿插发出 puts 和 gets，使通信与计算能以很细的粒度融合。

Ava: So instead of launching separate communication work and computation work, the kernel can coordinate both?

Ava：也就是说，不必分别启动通信任务和计算任务，内核自己就能协调两者？

Brian: Exactly. There are three programming opportunities. First, in-kernel communication. Second, ultralow-latency remote access, where access can be one-sided and doesn’t wait for the remote GPU to issue a matching command. Protocols such as RDMA and IB-GDA can provide direct memory access over the network.

Brian：没错。这里有三个编程方面的机会。第一，在内核内部通信。第二，超低延迟的远程访问：访问可以是单边的，无须等待远端 GPU 发出配对命令。RDMA 和 IB-GDA 等协议可以提供跨网络的直接内存访问。

Ava: And the third opportunity is customization?

Ava：第三个机会是定制？

Brian: Yes, customized communication patterns. Developers can write kernels tailored to an application’s communication needs. The release also says it should be straightforward to add support for new data layouts, even when those layouts aren’t in standard libraries yet.

Brian：对，定制通信模式。开发者可以根据应用的通信需求编写内核。发布说明还提到，即使标准库尚未支持某种新数据布局，增加对它的支持也应该比较直接。

Ava: What is actually included in 2.9?

Ava：2.9 实际包含了什么？

Brian: Symmetric tensors can be allocated for remote direct access. The currently supported backends are CUDA and NVSHMEM. There are also accelerated collectives using direct access, such as one-shot all-reduce, two-shot all-reduce, and multimem all-gather out.

Brian：现在可以分配供远程直接访问的对称张量。目前支持的后端是 CUDA 和 NVSHMEM。此外，还有利用直接访问加速的集合通信操作，例如 one-shot all-reduce、two-shot all-reduce 和 multimem all-gather out。

Ava: Those names are pretty specialized. Is there support for mixture-of-experts models too?

Ava：这些名称相当专业。混合专家模型也有相关支持吗？

Brian: There is a nontraditional all-to-all-v path for MoE models, meaning mixture-of-experts models. The listed operations include all-to-all-v-dev, all-to-all-v-dev-2D, and an offset form for token combination. For custom multi-GPU kernels, there’s an NVSHMEM plugin for Triton, plus continued support for Async TP and generalization to other patterns.

Brian：有一条供 MoE 模型使用的非传统 all-to-all-v 路径；MoE 指混合专家模型。列出的操作包括 all-to-all-v-dev、all-to-all-v-dev-2D，以及用于合并 token 的 offset 形式。对于自定义多 GPU 内核，还有供 Triton 使用的 NVSHMEM 插件；Async TP 也会继续得到支持，并推广到其他模式。

Ava: Where do developers access these operations?

Ava：开发者从哪里访问这些操作？

Brian: The Symmetric Memory operations are available under torch.ops.symm_mem. The article points users to the Symmetric Memory API documentation for details. It’s labeled API-Unstable, so users should expect the interface to evolve.

Brian：Symmetric Memory 操作位于 torch.ops.symm_mem 下。文章建议用户查阅 Symmetric Memory API 文档了解详情。它被标记为 API-Unstable，因此用户应预期接口还会变化。

Ava: Let’s switch to torch.compile. Graph breaks can be useful during development, but sometimes I want them to fail loudly. What changed?

Ava：我们转到 torch.compile。图中断在开发过程中可能有用，但有时我希望遇到它就明确报错。这次有什么变化？

Brian: PyTorch 2.9 adds torch._dynamo.error_on_graph_break(). It’s a context manager and decorator. You can mark regions of compiled code where torch.compile should error when it encounters a graph break, or resume instead.

Brian：PyTorch 2.9 增加了 torch._dynamo.error_on_graph_break()。它既是上下文管理器，也是装饰器。你可以标记已编译代码中的某些区域，指定 torch.compile 遇到图中断时应报错，还是继续执行。

Ava: How is that different from fullgraph?

Ava：这和 fullgraph 有什么区别？

Brian: The important difference is that error_on_graph_break can be toggled arbitrarily. With fullgraph, once you set it to true, you can’t set it back to false. So the new option lets you choose stricter behavior only around the regions where you need it.

Brian：关键区别在于，error_on_graph_break 可以随意切换。而 fullgraph 一旦设为 true，就不能再设回 false。因此，新选项允许你只在需要的区域采用更严格的行为。

Ava: That sounds useful for narrowing down a troublesome part of a model without forcing the whole program into one graph.

Ava：这样就能缩小模型中问题部分的范围，而不必强制整个程序都放进一张图里。

Brian: That’s the practical reading. The article describes it as expanding torch.compile’s graph-break options. It also points to a tutorial for more details, so the exact usage patterns are best learned there.

Brian：从实际使用角度可以这么理解。文章称它扩展了 torch.compile 处理图中断的选项。文章还指向一篇教程；具体用法最好在那里学习。

Ava: Packaging is another big theme. What does expanded wheel variant support mean for users?

Ava：安装包也是一大主题。扩展 wheel 变体支持对用户意味着什么？

Brian: PyTorch is participating in the WheelNext initiative to improve the Python packaging ecosystem. In 2.9.0, the wheel variant support matrix adds AMD ROCm, Intel XPU, and NVIDIA CUDA 13. These platforms get provider plugins that can detect platform attributes, including supported software and installed hardware.

Brian：PyTorch 正在参与 WheelNext 计划，以改进 Python 打包生态。在 2.9.0 中，wheel 变体支持矩阵增加了 AMD ROCm、Intel XPU 和 NVIDIA CUDA 13。这些平台都有提供方插件，可检测平台属性，包括受支持的软件和已安装的硬件。

Ava: Is the support equally available on every operating system?

Ava：每种操作系统都能获得同样的支持吗？

Brian: No. NVIDIA CUDA wheels support Windows and Linux. ROCm and XPU currently support Linux only. The article also stresses that this feature is experimental and based on a work-in-progress wheel variants proposal.

Brian：不能。NVIDIA CUDA wheels 支持 Windows 和 Linux。ROCm 和 XPU 目前只支持 Linux。文章还强调，这项功能是实验性的，基于一份仍在制定中的 wheel variants 提案。

Ava: So users should treat it as an evolving installation path.

Ava：所以用户应该把它看作一条仍在发展的安装途径。

Brian: Yes. The article gives uv-based installation instructions for Linux x86, Linux aarch64, and macOS, and separate PowerShell instructions for Windows x86. The main point is that uv can select the appropriate torch wheel through the WheelNext installer URL.

Brian：对。文章提供了适用于 Linux x86、Linux aarch64 和 macOS 的 uv 安装说明，以及单独适用于 Windows x86 的 PowerShell 安装说明。核心是，uv 可以通过 WheelNext 安装程序 URL 选择合适的 torch wheel。

Ava: Now, FlexAttention appears twice: once for Intel GPUs and once for X86 CPUs. Let’s separate those.

Ava：FlexAttention 出现了两次：一次针对 Intel GPU，一次针对 X86 CPU。我们分别说说。

Brian: On Intel GPUs, PyTorch 2.9 adds FlexAttention forward and backward support, aligned with common PyTorch GPU behavior. That gives developers more consistent and portable performance across different GPUs. The article says code can be written once, and Hugging Face or Transformers workloads can use FlexAttention without code changes.

Brian：在 Intel GPU 上，PyTorch 2.9 增加了 FlexAttention 前向和反向计算支持，与 PyTorch 在 GPU 上的常见行为保持一致。这让开发者能在不同 GPU 上获得更一致、可移植的性能。文章说，代码只需编写一次，Hugging Face 或 Transformers 工作负载无须修改代码就能使用 FlexAttention。

Ava: And the CPU feature is flash decoding?

Ava：CPU 方面的功能是 flash decoding？

Brian: Right. Flash decoding is a common LLM inference technique for speeding up attention and generation with long sequences. Before this release, the article says it existed only on the CUDA path. PyTorch 2.9 adds a FlexAttention-based optimization in the X86 CPU Inductor backend.

Brian：对。Flash decoding 是一种常见的 LLM 推理技术，用于加快长序列的注意力计算和生成。文章说，在这次发布之前，它只存在于 CUDA 路径中。PyTorch 2.9 在 X86 CPU Inductor 后端增加了一项基于 FlexAttention 的优化。

Ava: How does that optimization create more parallel work?

Ava：这项优化怎样产生更多可并行执行的工作？

Brian: It parallelizes the key-value sequence by partition and reduction. That can improve CPU utilization when the original parallelism is insufficient. The examples given are small batch size, small head number, or short query sequence length combined with a long key-value sequence length.

Brian：它通过分区和归约，让键值序列并行处理。当原有的并行度不足时，这可以提高 CPU 利用率。文中给出的例子是批量大小较小、注意力头数较少，或者查询序列较短而键值序列较长。

Ava: So the target is especially long-context decoding on CPUs?

Ava：所以它尤其针对 CPU 上的长上下文解码？

Brian: Yes. The article says it’s expected to help during the LLM decoding phase, especially with long context length. It doesn’t give a benchmark number, so we should describe the benefit as an intended optimization rather than quote a measured speedup.

Brian：对。文章说，预计它能在 LLM 解码阶段发挥作用，尤其是在上下文很长时。文章没有给出基准测试数值，所以我们应该把这一收益描述为预期的优化效果，而不是引用实测加速幅度。

Ava: Good distinction. What about Arm?

Ava：这个区别说得好。Arm 方面呢？

Brian: PyTorch 2.9 delivers backend improvements and better test coverage on Arm. In torch.compile mode, the TorchBench, HuggingFace, and TIMM test suites are faster than Eager mode, with results available on the PyTorch HUD Dashboard.

Brian：PyTorch 2.9 改进了 Arm 后端，也提高了测试覆盖率。在 torch.compile 模式下，TorchBench、HuggingFace 和 TIMM 测试套件的运行速度比 Eager 模式更快；相关结果可在 PyTorch HUD Dashboard 查看。

Ava: Are there operator-level changes too?

Ava：算子层面也有变化吗？

Brian: Yes. Convolution, activation, and quantized operations on AArch64 were expanded and optimized for faster model execution. Arm continuous-integration coverage was also broadened by adding AWS Graviton 4 instances based on Arm Neoverse V2.

Brian：有。AArch64 上的卷积、激活和量化操作得到扩展和优化，让模型执行得更快。Arm 的持续集成覆盖范围也扩大了，新增了基于 Arm Neoverse V2 的 AWS Graviton 4 实例。

Ava: And Linux aarch64 wheels are part of the release story as well?

Ava：Linux aarch64 wheels 也是这次发布的一部分？

Brian: Yes. The release highlights enablement of Linux aarch64 binary wheel builds across all supported CUDA versions. That should make installation easier for users on those Arm systems, although the article doesn’t enumerate the versions.

Brian：是的。发布说明重点提到，所有受支持的 CUDA 版本都启用了 Linux aarch64 二进制 wheel 构建。这应该会让使用这些 Arm 系统的用户更容易安装，不过文章没有逐一列出版本。

Ava: Before we wrap up, how large was this release effort?

Ava：结束前再问一句，这次发布投入了多大工作量？

Brian: Since PyTorch 2.8, the release includes three thousand two hundred sixteen commits from four hundred fifty-two contributors. The team thanks the community and encourages users to try the features and report issues as 2.9 improves.

Brian：自 PyTorch 2.8 以来，这次发布包含 452 位贡献者提交的 3216 次提交。团队感谢社区，并鼓励用户试用这些功能、报告问题，帮助 2.9 继续改进。

Ava: Let me recap in three points. First, the stable libtorch ABI and Symmetric Memory target extension authors and multi-GPU kernel developers. Second, torch.compile gets selective error-or-resume behavior for graph breaks. Third, hardware coverage expands through ROCm, XPU, CUDA 13, Intel GPUs, X86 CPUs, and Arm.

Ava：我用三点回顾一下。第一，稳定的 libtorch ABI 和 Symmetric Memory 面向扩展开发者及多 GPU 内核开发者。第二，torch.compile 增加了针对图中断选择报错或继续执行的行为。第三，通过 ROCm、XPU、CUDA 13、Intel GPU、X86 CPU 和 Arm，硬件覆盖范围进一步扩大。

Brian: That’s the release in a nutshell. The main limitation is maturity: several features are API-Unstable, the high-level C++ APIs are still in preview, and wheel variants are experimental. So teams should test carefully and follow the documentation as these interfaces develop.

Brian：这就是这次发布的概况。主要限制在于成熟度：有几项功能仍标记为 API-Unstable，高层 C++ API 仍处于预览阶段，wheel 变体也还是实验性功能。因此，团队应仔细测试，并随着这些接口的发展持续查阅文档。

Ava: Thanks, Brian. For listeners, the three things to remember are: more stable extension compatibility, more flexible communication and compilation control, and broader hardware-specific performance work.

Ava：谢谢，Brian。听众朋友们，要记住三点：扩展兼容性更稳定，通信和编译控制更灵活，以及针对特定硬件开展了更广泛的性能优化工作。

Brian: And if you try PyTorch 2.9, report what breaks or improves. That feedback is part of how these APIs become dependable.

Brian：如果你试用了 PyTorch 2.9，请反馈哪些地方出了问题，哪些地方有所改进。这些反馈也是让这些 API 变得可靠的过程的一部分。

Ava: That’s all for today. Keep your kernels fast and your graph breaks intentional.

Ava：今天就聊到这里。愿你的内核始终快速运行，每次图中断都是有意为之。

## 术语

| Term | 释义 |
|---|---|
| libtorch ABI | libtorch 应用二进制接口，用于让扩展跨不同 PyTorch 版本运行。 |
| C++/CUDA extension | 用 C++ 或 CUDA 编写的 PyTorch 自定义扩展。 |
| Device Guard | 用于管理设备上下文的设备工具。 |
| Symmetric Memory | 支持多 GPU 远程直接访问和内核内通信的内存编程模型。 |
| NVLink | GPU 之间的高速互连。 |
| RDMA | 远程直接内存访问网络技术。 |
| NVSHMEM | 支持 GPU 间对称内存和通信的后端。 |
| graph break | torch.compile 无法继续捕获计算图而中断的点。 |
| FlexAttention | 用于实现灵活高效注意力计算的 PyTorch 机制。 |
| Flash decoding | 面向长序列生成、加速注意力推理的解码技术。 |
| Inductor backend | PyTorch 编译器用于生成优化代码的后端。 |
| wheel variant | 根据硬件和软件平台选择不同 Python wheel 的打包机制。 |
| ROCm | AMD GPU 的软件平台。 |
| XPU | Intel 加速器设备后端。 |
| AArch64 | Arm 64 位指令集架构。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What makes this release worth a close listen? | 是什么让这个版本值得仔细关注？ |
| That’s exactly the motivation. | 这正是它的动机。 |
| Let’s switch to... | 我们转到…… |
| How is that different from...? | 这和……有什么不同？ |
| The practical reading is... | 从实际角度看，它意味着…… |
| Good distinction. | 区分得很好。 |
| Before we wrap up... | 在结束之前…… |
| Let me recap in three points. | 我用三点来回顾一下。 |
| That’s the release in a nutshell. | 这就是这个版本的简要概括。 |
