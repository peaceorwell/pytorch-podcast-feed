# PyTorch 2.9: More Portable Kernels, Smarter Compilation, and Wider Hardware Support

原文：[PyTorch 2.9 Release Blog](https://pytorch.org/blog/pytorch-2-9/)

## 摘要

本期介绍 PyTorch 2.9 的主要更新，包括稳定的 libtorch ABI、Symmetric Memory、多 GPU 内核编程，以及 torch.compile 图断点控制。我们还讨论了 ROCm、XPU、CUDA 13 的 wheel variant 支持、Intel GPU 上的 FlexAttention，以及基于 FlexAttention 的 X86 CPU flash decoding。最后回顾 Arm 平台优化、当前 API 仍处于不稳定或预览阶段的限制，以及后续工作方向。

## 对话

**Ava:** If you build C++ or CUDA extensions, run models across different GPUs, or care about long-context inference, PyTorch 2.9 has something aimed directly at you. Brian, what makes this release worth a close listen?

**Brian:** It’s a broad release rather than one single headline feature. PyTorch 2.9 brings a more stable libtorch ABI, easier multi-GPU kernel programming through Symmetric Memory, finer control over graph breaks in torch.compile, and wider hardware packaging. It also has attention optimizations for Intel GPUs and X86 CPUs.

**Ava:** Let’s start with the ABI piece. I’ve heard extension authors complain that a compiled extension can be tied too closely to one PyTorch version. Does 2.9 address that?

**Brian:** That’s exactly the motivation. ABI means application binary interface. PyTorch is building a stable ABI with C++ convenience wrappers, so you can build an extension with one torch version and run it with another. In 2.9, that surface gets several new APIs.

**Ava:** What kind of APIs are we talking about?

**Brian:** There are device utilities, such as Device Guard and Stream, available through the stable accelerator header. The stable torch::stable::Tensor also gets a default constructor, is_cpu, scalar_type, and get_device_index. And more stable ATen operations are exposed, including amax, narrow, dtype variants of new_empty and new_zeros, and pad.

**Ava:** So this is useful for people maintaining custom C++ or CUDA extensions, but it’s not a finished contract yet?

**Brian:** Right. The team enabled a libtorch-ABI wheel for Flash-Attention 3, which shows the direction. But the high-level C++ APIs are still in preview. PyTorch says it’s continuing to expand the ABI surface, establish versioning, write more documentation, and make more custom kernels ABI stable.

**Ava:** That sounds like a compatibility layer being built piece by piece. Now, Symmetric Memory sounds more fundamental. What problem does it solve?

**Brian:** It makes multi-GPU kernels easier to program across NVLinks and RDMA networks. The key idea is that a GPU kernel can communicate while it computes. A kernel can issue puts and gets interleaved with computation instructions, so communication can be fused at a very small granularity.

**Ava:** So instead of launching separate communication work and computation work, the kernel can coordinate both?

**Brian:** Exactly. There are three programming opportunities. First, in-kernel communication. Second, ultralow-latency remote access, where access can be one-sided and doesn’t wait for the remote GPU to issue a matching command. Protocols such as RDMA and IB-GDA can provide direct memory access over the network.

**Ava:** And the third opportunity is customization?

**Brian:** Yes, customized communication patterns. Developers can write kernels tailored to an application’s communication needs. The release also says it should be straightforward to add support for new data layouts, even when those layouts aren’t in standard libraries yet.

**Ava:** What is actually included in 2.9?

**Brian:** Symmetric tensors can be allocated for remote direct access. The currently supported backends are CUDA and NVSHMEM. There are also accelerated collectives using direct access, such as one-shot all-reduce, two-shot all-reduce, and multimem all-gather out.

**Ava:** Those names are pretty specialized. Is there support for mixture-of-experts models too?

**Brian:** There is a nontraditional all-to-all-v path for MoE models, meaning mixture-of-experts models. The listed operations include all-to-all-v-dev, all-to-all-v-dev-2D, and an offset form for token combination. For custom multi-GPU kernels, there’s an NVSHMEM plugin for Triton, plus continued support for Async TP and generalization to other patterns.

**Ava:** Where do developers access these operations?

**Brian:** The Symmetric Memory operations are available under torch.ops.symm_mem. The article points users to the Symmetric Memory API documentation for details. It’s labeled API-Unstable, so users should expect the interface to evolve.

**Ava:** Let’s switch to torch.compile. Graph breaks can be useful during development, but sometimes I want them to fail loudly. What changed?

**Brian:** PyTorch 2.9 adds torch._dynamo.error_on_graph_break(). It’s a context manager and decorator. You can mark regions of compiled code where torch.compile should error when it encounters a graph break, or resume instead.

**Ava:** How is that different from fullgraph?

**Brian:** The important difference is that error_on_graph_break can be toggled arbitrarily. With fullgraph, once you set it to true, you can’t set it back to false. So the new option lets you choose stricter behavior only around the regions where you need it.

**Ava:** That sounds useful for narrowing down a troublesome part of a model without forcing the whole program into one graph.

**Brian:** That’s the practical reading. The article describes it as expanding torch.compile’s graph-break options. It also points to a tutorial for more details, so the exact usage patterns are best learned there.

**Ava:** Packaging is another big theme. What does expanded wheel variant support mean for users?

**Brian:** PyTorch is participating in the WheelNext initiative to improve the Python packaging ecosystem. In 2.9.0, the wheel variant support matrix adds AMD ROCm, Intel XPU, and NVIDIA CUDA 13. These platforms get provider plugins that can detect platform attributes, including supported software and installed hardware.

**Ava:** Is the support equally available on every operating system?

**Brian:** No. NVIDIA CUDA wheels support Windows and Linux. ROCm and XPU currently support Linux only. The article also stresses that this feature is experimental and based on a work-in-progress wheel variants proposal.

**Ava:** So users should treat it as an evolving installation path.

**Brian:** Yes. The article gives uv-based installation instructions for Linux x86, Linux aarch64, and macOS, and separate PowerShell instructions for Windows x86. The main point is that uv can select the appropriate torch wheel through the WheelNext installer URL.

**Ava:** Now, FlexAttention appears twice: once for Intel GPUs and once for X86 CPUs. Let’s separate those.

**Brian:** On Intel GPUs, PyTorch 2.9 adds FlexAttention forward and backward support, aligned with common PyTorch GPU behavior. That gives developers more consistent and portable performance across different GPUs. The article says code can be written once, and Hugging Face or Transformers workloads can use FlexAttention without code changes.

**Ava:** And the CPU feature is flash decoding?

**Brian:** Right. Flash decoding is a common LLM inference technique for speeding up attention and generation with long sequences. Before this release, the article says it existed only on the CUDA path. PyTorch 2.9 adds a FlexAttention-based optimization in the X86 CPU Inductor backend.

**Ava:** How does that optimization create more parallel work?

**Brian:** It parallelizes the key-value sequence by partition and reduction. That can improve CPU utilization when the original parallelism is insufficient. The examples given are small batch size, small head number, or short query sequence length combined with a long key-value sequence length.

**Ava:** So the target is especially long-context decoding on CPUs?

**Brian:** Yes. The article says it’s expected to help during the LLM decoding phase, especially with long context length. It doesn’t give a benchmark number, so we should describe the benefit as an intended optimization rather than quote a measured speedup.

**Ava:** Good distinction. What about Arm?

**Brian:** PyTorch 2.9 delivers backend improvements and better test coverage on Arm. In torch.compile mode, the TorchBench, HuggingFace, and TIMM test suites are faster than Eager mode, with results available on the PyTorch HUD Dashboard.

**Ava:** Are there operator-level changes too?

**Brian:** Yes. Convolution, activation, and quantized operations on AArch64 were expanded and optimized for faster model execution. Arm continuous-integration coverage was also broadened by adding AWS Graviton 4 instances based on Arm Neoverse V2.

**Ava:** And Linux aarch64 wheels are part of the release story as well?

**Brian:** Yes. The release highlights enablement of Linux aarch64 binary wheel builds across all supported CUDA versions. That should make installation easier for users on those Arm systems, although the article doesn’t enumerate the versions.

**Ava:** Before we wrap up, how large was this release effort?

**Brian:** Since PyTorch 2.8, the release includes three thousand two hundred sixteen commits from four hundred fifty-two contributors. The team thanks the community and encourages users to try the features and report issues as 2.9 improves.

**Ava:** Let me recap in three points. First, the stable libtorch ABI and Symmetric Memory target extension authors and multi-GPU kernel developers. Second, torch.compile gets selective error-or-resume behavior for graph breaks. Third, hardware coverage expands through ROCm, XPU, CUDA 13, Intel GPUs, X86 CPUs, and Arm.

**Brian:** That’s the release in a nutshell. The main limitation is maturity: several features are API-Unstable, the high-level C++ APIs are still in preview, and wheel variants are experimental. So teams should test carefully and follow the documentation as these interfaces develop.

**Ava:** Thanks, Brian. For listeners, the three things to remember are: more stable extension compatibility, more flexible communication and compilation control, and broader hardware-specific performance work.

**Brian:** And if you try PyTorch 2.9, report what breaks or improves. That feedback is part of how these APIs become dependable.

**Ava:** That’s all for today. Keep your kernels fast and your graph breaks intentional.

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
