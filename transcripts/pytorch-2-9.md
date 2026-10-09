# PyTorch 2.9: More Portable Kernels, Smarter Compilation, and Wider Hardware Support

原文：[PyTorch 2.9 Release Blog](https://pytorch.org/blog/pytorch-2-9/)

## 摘要

本期介绍 PyTorch 2.9 的主要更新，包括稳定的 libtorch ABI、Symmetric Memory、多 GPU 内核编程，以及 torch.compile 图断点控制。我们还讨论了 ROCm、XPU、CUDA 13 的 wheel variant 支持、Intel GPU 上的 FlexAttention，以及基于 FlexAttention 的 X86 CPU flash decoding。最后回顾 Arm 平台优化、当前 API 仍处于不稳定或预览阶段的限制，以及后续工作方向。

## 对话

Ava: If you build C++ or CUDA extensions, run models across different GPUs, or care about long-context inference, PyTorch 2.

Ava：如果你开发 C++ 或 CUDA 扩展、让模型跨 GPU 运行，或关注长上下文推理，PyTorch 2.9

9 has something aimed directly at you.

就有专门面向你的新功能。

Brian, what makes this release worth a close listen?

Brian，这次发布为什么值得仔细听？

Brian: It’s a broad release rather than one single headline feature. PyTorch 2.

Brian：这次更新涉及面很广，没有单一的主打功能。PyTorch 2.9

9 brings a more stable libtorch ABI, easier multi-GPU kernel programming through Symmetric Memory, finer control over graph breaks in torch.

带来了更稳定的 libtorch ABI、让多 GPU 内核编程更容易的 Symmetric Memory，以及对 torch.compile 图中断更细的控制，

compile, and wider hardware packaging.

还扩大了硬件包的覆盖范围。

It also has attention optimizations for Intel GPUs and X86 CPUs.

它还优化了 Intel GPU 和 X86 CPU 上的注意力计算。

Ava: Let’s start with the ABI piece.

Ava：先说说 ABI。

I’ve heard extension authors complain that a compiled extension can be tied too closely to one PyTorch version.

我听扩展开发者抱怨过，编译好的扩展可能与某个 PyTorch 版本绑得太紧。

Does 2. 9 address that?

2.9 解决了这个问题吗？

Brian: That’s exactly the motivation. ABI means application binary interface.

Brian：这正是改进的初衷。ABI 指应用程序二进制接口。

PyTorch is building a stable ABI with C++ convenience wrappers, so you can build an extension with one torch version and run it with another.

PyTorch 正在构建稳定的 ABI，并提供方便使用的 C++ 封装，让扩展可以用一个 torch 版本构建，再用另一个版本运行。

In 2. 9, that surface gets several new APIs.

在 2.9 中，这套接口新增了多个 API。

Ava: What kind of APIs are we talking about?

Ava：具体有哪些 API？

Brian: There are device utilities, such as Device Guard and Stream, available through the stable accelerator header.

Brian：稳定版加速器头文件提供了 Device Guard 和 Stream 等设备工具。

The stable torch::stable::Tensor also gets a default constructor, is_cpu, scalar_type, and get_device_index.

稳定的 torch::stable::Tensor 也新增了默认构造函数、is_cpu、scalar_type 和 get_device_index。

And more stable ATen operations are exposed, including amax, narrow, dtype variants of new_empty and new_zeros, and pad.

此外还开放了更多稳定的 ATen 操作，包括 amax、narrow、支持 dtype 的 new_empty 和 new_zeros 变体，以及 pad。

Ava: So this is useful for people maintaining custom C++ or CUDA extensions, but it’s not a finished contract yet?

Ava：所以这对维护自定义 C++ 或 CUDA 扩展的人很有用，但接口规范还没最终定下来？

Brian: Right.

Brian：对。

The team enabled a libtorch-ABI wheel for Flash-Attention 3, which shows the direction.

团队为 Flash-Attention 3 提供了 libtorch-ABI wheel，展示了发展方向。

But the high-level C++ APIs are still in preview.

但高级 C++ API 仍处于预览阶段。

PyTorch says it’s continuing to expand the ABI surface, establish versioning, write more documentation, and make more custom kernels ABI stable.

PyTorch 表示会继续扩展 ABI 接口、建立版本机制、完善文档，并让更多自定义内核支持稳定 ABI。

Ava: That sounds like a compatibility layer being built piece by piece.

Ava：听起来，这个兼容层正在逐步建成。

Now, Symmetric Memory sounds more fundamental. What problem does it solve?

接下来，Symmetric Memory 听起来更底层。它解决什么问题？

Brian: It makes multi-GPU kernels easier to program across NVLinks and RDMA networks.

Brian：它让开发者更容易编写跨 NVLink 和 RDMA 网络的多 GPU 内核。

The key idea is that a GPU kernel can communicate while it computes.

核心思路是让 GPU 内核在计算时也能通信。

A kernel can issue puts and gets interleaved with computation instructions, so communication can be fused at a very small granularity.

内核可以在计算指令之间穿插执行 put 和 get 操作，从而在很细的粒度上融合通信与计算。

Ava: So instead of launching separate communication work and computation work, the kernel can coordinate both?

Ava：也就是说，内核可以同时协调通信和计算，不必分别启动这两类任务？

Brian: Exactly. There are three programming opportunities. First, in-kernel communication.

Brian：没错。它提供了三种编程方式。第一种是内核内通信。

Second, ultralow-latency remote access, where access can be one-sided and doesn’t wait for the remote GPU to issue a matching command.

第二种是超低延迟远程访问：可以单边访问，无须等待远端 GPU 发出对应命令。

Protocols such as RDMA and IB-GDA can provide direct memory access over the network.

RDMA 和 IB-GDA 等协议可以通过网络提供直接内存访问。

Ava: And the third opportunity is customization?

Ava：第三种是自定义通信？

Brian: Yes, customized communication patterns.

Brian：对，自定义通信模式。

Developers can write kernels tailored to an application’s communication needs.

开发者可以根据应用的通信需求编写内核。

The release also says it should be straightforward to add support for new data layouts, even when those layouts aren’t in standard libraries yet.

发布说明还提到，即使标准库尚未支持某种数据布局，新增对它的支持也应该很直接。

Ava: What is actually included in 2. 9?

Ava：2.9 具体包含哪些功能？

Brian: Symmetric tensors can be allocated for remote direct access.

Brian：可以分配对称张量，供远程直接访问。

The currently supported backends are CUDA and NVSHMEM.

目前支持的后端是 CUDA 和 NVSHMEM。

There are also accelerated collectives using direct access, such as one-shot all-reduce, two-shot all-reduce, and multimem all-gather out.

还有利用直接访问加速的集合通信操作，例如 one-shot all-reduce、two-shot all-reduce 和 multimem all-gather out。

Ava: Those names are pretty specialized. Is there support for mixture-of-experts models too?

Ava：这些名称很专业。混合专家模型也有相应支持吗？

Brian: There is a nontraditional all-to-all-v path for MoE models, meaning mixture-of-experts models.

Brian：有。MoE，也就是混合专家模型，可以使用一种非传统的 all-to-all-v 路径。

The listed operations include all-to-all-v-dev, all-to-all-v-dev-2D, and an offset form for token combination.

列出的操作包括 all-to-all-v-dev、all-to-all-v-dev-2D，以及用于合并 token 的 offset 形式。

For custom multi-GPU kernels, there’s an NVSHMEM plugin for Triton, plus continued support for Async TP and generalization to other patterns.

自定义多 GPU 内核可以使用面向 Triton 的 NVSHMEM 插件；Async TP 也继续得到支持，并扩展到其他模式。

Ava: Where do developers access these operations?

Ava：开发者从哪里调用这些操作？

Brian: The Symmetric Memory operations are available under torch. ops. symm_mem.

Brian：Symmetric Memory 操作位于 torch.ops.symm_mem 下。

The article points users to the Symmetric Memory API documentation for details.

文章建议用户查阅 Symmetric Memory API 文档，了解详细用法。

It’s labeled API-Unstable, so users should expect the interface to evolve.

它被标为 API-Unstable，接口预计还会变化。

Ava: Let’s switch to torch. compile.

Ava：我们再聊聊 torch.compile。

Graph breaks can be useful during development, but sometimes I want them to fail loudly.

开发时，图中断可能有用，但有时我希望它直接报错。

What changed?

这次有什么变化？

Brian: PyTorch 2. 9 adds torch. _dynamo. error_on_graph_break().

Brian：PyTorch 2.9 新增 torch._dynamo.error_on_graph_break()。

It’s a context manager and decorator. You can mark regions of compiled code where torch.

它既是上下文管理器，也是装饰器。你可以指定编译代码区域，让 torch.compile

compile should error when it encounters a graph break, or resume instead.

在遇到图中断时选择报错或继续执行。

Ava: How is that different from fullgraph?

Ava：这和 fullgraph 有什么区别？

Brian: The important difference is that error_on_graph_break can be toggled arbitrarily.

Brian：关键区别是 error_on_graph_break 可以随时切换。

With fullgraph, once you set it to true, you can’t set it back to false.

而 fullgraph 一旦设为 true，就不能再改回 false。

So the new option lets you choose stricter behavior only around the regions where you need it.

因此，新选项让你只在需要的区域启用更严格的行为。

Ava: That sounds useful for narrowing down a troublesome part of a model without forcing the whole program into one graph.

Ava：这样就能定位模型中出问题的部分，而不用强制整个程序只用一张图。

Brian: That’s the practical reading. The article describes it as expanding torch.

Brian：可以这么理解。文章称，这扩展了 torch.compile

compile’s graph-break options.

处理图中断的选项。

It also points to a tutorial for more details, so the exact usage patterns are best learned there.

文章还提供了教程链接；具体用法最好参照教程。

Ava: Packaging is another big theme. What does expanded wheel variant support mean for users?

Ava：打包也是一大主题。扩展 wheel 变体支持对用户意味着什么？

Brian: PyTorch is participating in the WheelNext initiative to improve the Python packaging ecosystem.

Brian：PyTorch 正参与 WheelNext 计划，以改进 Python 打包生态。

In 2. 9.

在 2.9.0 版本中，

0, the wheel variant support matrix adds AMD ROCm, Intel XPU, and NVIDIA CUDA 13.

wheel 变体支持矩阵新增 AMD ROCm、Intel XPU 和 NVIDIA CUDA 13。

These platforms get provider plugins that can detect platform attributes, including supported software and installed hardware.

这些平台有相应的提供方插件，可检测支持的软件和已安装的硬件等平台属性。

Ava: Is the support equally available on every operating system?

Ava：所有操作系统都同样支持吗？

Brian: No. NVIDIA CUDA wheels support Windows and Linux.

Brian：不是。NVIDIA CUDA 的 wheel 支持 Windows 和 Linux。

ROCm and XPU currently support Linux only.

ROCm 和 XPU 目前仅支持 Linux。

The article also stresses that this feature is experimental and based on a work-in-progress wheel variants proposal.

文章还强调，这项功能仍处于实验阶段，依据的是尚未定稿的 wheel 变体提案。

Ava: So users should treat it as an evolving installation path.

Ava：所以用户应把它视为仍在演进的安装方式。

Brian: Yes.

Brian：对。

The article gives uv-based installation instructions for Linux x86, Linux aarch64, and macOS, and separate PowerShell instructions for Windows x86.

文章提供了适用于 Linux x86、Linux aarch64 和 macOS 的 uv 安装说明，以及单独适用于 Windows x86 的 PowerShell 说明。

The main point is that uv can select the appropriate torch wheel through the WheelNext installer URL.

关键是 uv 可以通过 WheelNext 安装器地址选出合适的 torch wheel。

Ava: Now, FlexAttention appears twice: once for Intel GPUs and once for X86 CPUs.

Ava：FlexAttention 出现了两次：一次针对 Intel GPU，一次针对 x86 CPU。

Let’s separate those.

我们分别来说。

Brian: On Intel GPUs, PyTorch 2.

Brian：在 Intel GPU 上，PyTorch 2.9

9 adds FlexAttention forward and backward support, aligned with common PyTorch GPU behavior.

新增 FlexAttention 前向和反向计算支持，与 PyTorch GPU 的常见行为保持一致。

That gives developers more consistent and portable performance across different GPUs.

这让开发者在不同 GPU 上获得更一致、可移植的性能。

The article says code can be written once, and Hugging Face or Transformers workloads can use FlexAttention without code changes.

文章称，代码只需编写一次，Hugging Face 或 Transformers 工作负载无需修改代码即可使用 FlexAttention。

Ava: And the CPU feature is flash decoding?

Ava：CPU 方面的新功能是 Flash Decoding 吗？

Brian: Right.

Brian：没错。

Flash decoding is a common LLM inference technique for speeding up attention and generation with long sequences.

Flash Decoding 是一种常见的大语言模型推理技术，可加快长序列的注意力计算和生成。

Before this release, the article says it existed only on the CUDA path. PyTorch 2.

文章称，此前它只用于 CUDA 路径。PyTorch 2.9

9 adds a FlexAttention-based optimization in the X86 CPU Inductor backend.

在 x86 CPU 的 Inductor 后端新增了基于 FlexAttention 的优化。

Ava: How does that optimization create more parallel work?

Ava：这种优化如何增加并行任务？

Brian: It parallelizes the key-value sequence by partition and reduction.

Brian：它通过分区和归约，对键值序列进行并行处理。

That can improve CPU utilization when the original parallelism is insufficient.

当原有并行度不足时，这可以提高 CPU 利用率。

The examples given are small batch size, small head number, or short query sequence length combined with a long key-value sequence length.

例如批量小、注意力头少，或查询序列短而键值序列长的情况。

Ava: So the target is especially long-context decoding on CPUs?

Ava：所以它尤其针对 CPU 上的长上下文解码？

Brian: Yes.

Brian：对。

The article says it’s expected to help during the LLM decoding phase, especially with long context length.

文章预计它能改善大语言模型的解码阶段，尤其是上下文较长时。

It doesn’t give a benchmark number, so we should describe the benefit as an intended optimization rather than quote a measured speedup.

文章没有给出基准测试数据，因此应将其描述为预期的优化效果，而非实测加速。

Ava: Good distinction. What about Arm?

Ava：这个区别很重要。Arm 方面呢？

Brian: PyTorch 2. 9 delivers backend improvements and better test coverage on Arm. In torch.

Brian：PyTorch 2.9 改进了 Arm 后端，并扩大了测试覆盖。在 torch.compile

compile mode, the TorchBench, HuggingFace, and TIMM test suites are faster than Eager mode, with results available on the PyTorch HUD Dashboard.

模式下，TorchBench、HuggingFace 和 TIMM 测试套件的运行速度快于 Eager 模式，结果可在 PyTorch HUD Dashboard 查看。

Ava: Are there operator-level changes too?

Ava：算子层面也有变化吗？

Brian: Yes.

Brian：有。

Convolution, activation, and quantized operations on AArch64 were expanded and optimized for faster model execution.

AArch64 上的卷积、激活和量化运算得到扩展与优化，可加快模型运行。

Arm continuous-integration coverage was also broadened by adding AWS Graviton 4 instances based on Arm Neoverse V2.

Arm 的持续集成测试也扩展到基于 Arm Neoverse V2 的 AWS Graviton 4 实例。

Ava: And Linux aarch64 wheels are part of the release story as well?

Ava：Linux aarch64 的 wheel 包也是本次发布的一部分吗？

Brian: Yes.

Brian：是的。

The release highlights enablement of Linux aarch64 binary wheel builds across all supported CUDA versions.

本次发布支持为所有受支持的 CUDA 版本构建 Linux aarch64 二进制 wheel 包。

That should make installation easier for users on those Arm systems, although the article doesn’t enumerate the versions.

这会让 Arm 系统用户更容易安装，不过文章没有列出具体版本。

Ava: Before we wrap up, how large was this release effort?

Ava：结束前再问一句，这次发布的规模有多大？

Brian: Since PyTorch 2.

Brian：自 PyTorch 2.

8, the release includes three thousand two hundred sixteen commits from four hundred fifty-two contributors.

8 以来，本次发布汇集了 452 位贡献者的 3216 次提交。

The team thanks the community and encourages users to try the features and report issues as 2.

团队感谢社区，也鼓励用户试用新功能并反馈问题，帮助 2.

9 improves.

9 版本持续改进。

Ava: Let me recap in three points.

Ava：我用三点总结一下。

First, the stable libtorch ABI and Symmetric Memory target extension authors and multi-GPU kernel developers.

第一，稳定的 libtorch ABI 和 Symmetric Memory 面向扩展开发者及多 GPU 内核开发者。

Second, torch. compile gets selective error-or-resume behavior for graph breaks.

第二，torch.compile 遇到计算图中断时，可以选择报错或继续执行。

Third, hardware coverage expands through ROCm, XPU, CUDA 13, Intel GPUs, X86 CPUs, and Arm.

第三，硬件支持扩展到 ROCm、XPU、CUDA 13、Intel GPU、X86 CPU 和 Arm。

Brian: That’s the release in a nutshell.

Brian：这就是本次发布的概要。

The main limitation is maturity: several features are API-Unstable, the high-level C++ APIs are still in preview, and wheel variants are experimental.

主要限制在于成熟度：多项功能仍标为 API-Unstable，高层 C++ API 仍处于预览阶段，wheel 变体也仍是实验性功能。

So teams should test carefully and follow the documentation as these interfaces develop.

因此，团队应仔细测试，并随着这些接口的发展关注文档。

Ava: Thanks, Brian.

Ava：谢谢你，Brian。

For listeners, the three things to remember are: more stable extension compatibility, more flexible communication and compilation control, and broader hardware-specific performance work.

听众记住三点就好：扩展兼容性更稳定，通信和编译控制更灵活，针对更多硬件开展了性能优化。

Brian: And if you try PyTorch 2. 9, report what breaks or improves.

Brian：如果你试用 PyTorch 2.9，请反馈哪些地方出了问题，哪些地方有了改进。

That feedback is part of how these APIs become dependable.

这些反馈有助于让 API 更可靠。

Ava: That’s all for today. Keep your kernels fast and your graph breaks intentional.

Ava：今天就到这里。祝大家的内核运行飞快，计算图中断都在意料之中。

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
