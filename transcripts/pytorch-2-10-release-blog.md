# PyTorch 2.10: Faster Kernels, Safer Numbers, and a New Release Rhythm

原文：[PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/)

## 摘要

本期对话介绍 PyTorch 2.10 的主要更新，重点包括性能优化、变长注意力和数值调试。Ava 与 Brian 解释了 Combo Kernels 如何减少 GPU kernel launch overhead，以及 varlen_attn() 对 ragged 和 packed sequences 的支持。节目还讨论了确定性执行、DebugMode、Tensor hashing、Intel GPU 增强，以及 TorchScript 弃用和新的发布节奏。最后，两位主持人总结了升级时需要关注的功能限制与后续方向。

## 对话

Ava: If you're training large models, a tiny numerical difference can turn into a very long debugging session. And a little kernel overhead can quietly slow down every step. PyTorch 2.10 is aimed at both problems.

Ava：如果你在训练大型模型，微小的数值差异就可能带来漫长的调试过程。而一点点内核开销，也可能在不知不觉中拖慢每一步。PyTorch 2.10 旨在解决这两个问题。

Brian: Exactly. This release pushes performance, but it also makes numerical behavior easier to inspect. It adds Python 3.14 support for torch.compile(), new attention and fusion features, and tools for finding where two runs start to disagree.

Brian：没错。这个版本着力提升性能，也让数值行为更容易检查。它为 torch.compile() 增加了 Python 3.14 支持，带来了新的注意力机制和融合功能，以及用于找出两次运行从哪里开始出现分歧的工具。

Ava: Let's start with the big picture. What changed in this release?

Ava：先从整体来看。这个版本有哪些变化？

Brian: PyTorch 2.10 contains four thousand one hundred sixty commits from five hundred thirty-six contributors since 2.9. The article groups the changes into performance, numerical debugging, and some non-feature updates.

Brian：自 2.9 以来，PyTorch 2.10 汇集了来自 536 位贡献者的 4160 次提交。文章将这些变化分为性能改进、数值调试，以及一些非功能性更新。

Ava: And the release is part of the larger two-series compiler story, right?

Ava：这次发布也是 2.x 系列更大范围的编译器发展历程的一部分，对吧？

Brian: Yes. Performance has been a focus throughout the two-x releases, building on the PyTorch compiler stack introduced in 2.0. In 2.10, that means more work in torchinductor, more compiler support, and better ways to test numerical behavior.

Brian：是的。自 PyTorch 2.0 引入编译器技术栈以来，性能一直是整个 2.x 系列版本的重点。到了 2.10，这体现在对 torchinductor 的进一步改进、更多编译器支持，以及更好的数值行为测试方法上。

Ava: Let's talk about the kernel work first. I saw the phrase Combo Kernels. What does that actually mean?

Ava：先聊内核方面的工作。我看到了 Combo Kernels 这个词。它到底是什么意思？

Brian: Combo Kernels are a horizontal fusion optimization in torchinductor. They combine multiple independent operations, with no data dependencies, into one unified GPU kernel.

Brian：Combo Kernels 是 torchinductor 中的一种横向融合优化。它把多个互不依赖、没有数据依赖关系的操作合并到一个统一的 GPU 内核中。

Ava: So this is different from the usual producer-consumer fusion?

Ava：这和通常的生产者—消费者融合不一样？

Brian: Right. Vertical fusion combines sequential operations, where one operation produces data for the next. Combo Kernels fuse parallel operations instead. Imagine several small checkout lines merging into one larger line, even though none of the customers depend on one another.

Brian：对。纵向融合会合并顺序执行的操作，其中一个操作为下一个操作生成数据。Combo Kernels 融合的则是并行操作。可以想象几条短短的结账队伍汇成一条更长的队伍，尽管顾客之间互不依赖。

Ava: The benefit is fewer kernel launches?

Ava：好处是减少内核启动次数？

Brian: That's the goal: reduced kernel launch overhead. The article doesn't give a benchmark number, but it explains that independent operations can now share one launch. The feature is marked API-UNSTABLE, so users should expect it to evolve.

Brian：目标就是减少内核启动开销。文章没有给出基准测试数据，但解释了独立操作现在可以共用一次启动。这个功能被标记为 API-UNSTABLE，因此用户应当预期它还会演变。

Ava: The next item is varlen_attn(). The name sounds like variable-length attention.

Ava：下一项是 varlen_attn()。这个名字听起来是可变长度注意力机制。

Brian: That's exactly what it is. varlen_attn() is a new torch.nn.attention operation for ragged and packed sequences. Ragged means the sequences can have different lengths, and packed means they can be stored together without padding every item to the same size.

Brian：正是如此。varlen_attn() 是 torch.nn.attention 中一项用于不规则序列和打包序列的新操作。不规则意味着序列可以有不同长度；打包意味着可以将它们一起存储，而不必用填充把每一项补到相同大小。

Ava: Does it support training, or only inference?

Ava：它支持训练，还是只支持推理？

Brian: It supports both forward and backward, and it's torch.compile-able. Right now, FlashAttention 2, or FA2, supports it. The article says there are plans to add cuDNN and FlashAttention 4 support.

Brian：它同时支持前向和反向计算，也可以通过 torch.compile 编译。目前，FlashAttention 2，也就是 FA2，支持它。文章说，后续计划增加对 cuDNN 和 FlashAttention 4 的支持。

Ava: What hardware and data types are supported?

Ava：支持哪些硬件和数据类型？

Brian: For now, it runs on NVIDIA CUDA with an A100 GPU or newer. It supports BF16 and FP16 data types. Again, this is API-UNSTABLE, so those boundaries may change later.

Brian：目前它可在配备 A100 或更新型号 GPU 的 NVIDIA CUDA 环境中运行，支持 BF16 和 FP16 数据类型。同样，这项功能被标记为 API-UNSTABLE，因此这些支持范围以后可能会变化。

Ava: Variable-length attention can save work because you don't calculate attention on padding, right?

Ava：可变长度注意力机制能省下计算量，因为不用对填充部分计算注意力，对吧？

Brian: That is the practical intuition, although the article focuses on the API and its current support rather than giving a measured speedup. The important point is that ragged and packed sequences now have a dedicated operation that works with compilation and gradients.

Brian：从实际使用角度看，可以这么理解，不过文章侧重介绍 API 及其当前支持情况，没有给出实测的加速幅度。关键是，不规则序列和打包序列现在有了专门的操作，而且支持编译和梯度计算。

Ava: There is also an eigenvalue update. That sounds more specialized.

Ava：还有一项特征值方面的更新。听起来更专门一些。

Brian: It is more specialized, but useful for numerical workloads. PyTorch linalg can now use cuSOLVER's DnXgeev to provide highly efficient general eigenvalue decomposition on NVIDIA GPUs.

Brian：确实更专门，但对数值计算任务很有用。PyTorch linalg 现在可以使用 cuSOLVER 的 DnXgeev，在 NVIDIA GPU 上高效地进行一般特征值分解。

Ava: So PyTorch is connecting its linear algebra path to a more efficient cuSOLVER routine?

Ava：也就是说，PyTorch 把线性代数计算路径接入了更高效的 cuSOLVER 例程？

Brian: Yes. That's the whole claim in the release note: use DnXgeev through cuSOLVER for efficient general eigenvalue decompositions on NVIDIA GPUs. The article doesn't provide a performance figure.

Brian：是的。这就是发布说明中的完整主张：通过 cuSOLVER 使用 DnXgeev，在 NVIDIA GPU 上高效地进行一般特征值分解。文章没有提供性能数据。

Ava: What about Intel GPUs? The release notes list several changes there.

Ava：Intel GPU 方面呢？发布说明列出了几项变化。

Brian: The support expands to the latest Intel Core Ultra Series 3 with Intel Arc Graphics, on both Windows and Linux. PyTorch also adds FP8 support on Intel GPUs.

Brian：支持范围扩展到了搭载 Intel Arc Graphics 的最新 Intel Core Ultra Series 3，涵盖 Windows 和 Linux。PyTorch 还为 Intel GPU 增加了 FP8 支持。

Ava: FP8 support can mean a lot of different things. What does this release include?

Ava：FP8 支持可能包含很多不同的内容。这个版本具体包含什么？

Brian: It adds commonly used basic operators, such as type promotion and shape operators, plus scaled matrix multiplication using tensor-wise and channel-wise scaling factors.

Brian：它增加了常用的基础算子，例如类型提升和形状算子，还增加了使用逐张量及逐通道缩放因子的缩放矩阵乘法。

Ava: Anything beyond FP8?

Ava：除了 FP8，还有别的吗？

Brian: Yes. Aten operator MatMul gets complex data type support on Intel GPUs. The PyTorch C++ Extension API also extends SYCL support, so users can implement new custom operators on Windows. And Intel GPU unit-test coverage is broadened.

Brian：有。Intel GPU 上的 Aten 算子 MatMul 增加了对复数数据类型的支持。PyTorch C++ Extension API 也扩展了 SYCL 支持，让用户可以在 Windows 上实现新的自定义算子。此外，Intel GPU 的单元测试覆盖范围也扩大了。

Ava: Now let's move to determinism. Why is this becoming such a big issue?

Ava：现在来谈谈确定性。为什么这个问题变得这么重要？

Brian: The article connects it to post-training with distributed reinforcement learning workflows. At that scale, run-to-run determinism makes debugging training runs easier. It also helps people test code that uses torch.compile reliably.

Brian：文章将其与使用分布式强化学习工作流的后训练联系起来。在这种规模下，不同运行之间保持确定性，能让训练过程更容易调试，也有助于可靠地测试使用 torch.compile 的代码。

Ava: What changed in 2.10?

Ava：2.10 中有什么变化？

Brian: torch.compile() now respects use_deterministic_mode. Users can turn deterministic algorithms on with torch.use_deterministic_algorithms(True). The article says this makes two invocations of torch.compile perform the same operations exactly.

Brian：torch.compile() 现在会遵循 use_deterministic_mode。用户可以通过 torch.use_deterministic_algorithms(True) 开启确定性算法。文章说，这会让两次调用 torch.compile 执行完全相同的操作。

Ava: That sounds stronger than just getting similar outputs.

Ava：这听起来比只是得到相近的输出要求更高。

Brian: Yes, the wording is about performing the same operations exactly. For someone chasing a rare divergence, that gives a much steadier baseline.

Brian：是的，原文说的是执行完全相同的操作。对于追查偶发分歧的人来说，这提供了稳定得多的基准。

Ava: And then there's DebugMode. Is it a profiler?

Ava：接下来是 DebugMode。它是性能分析器吗？

Brian: It's a custom TorchDispatchMode that provides profiling-style runtime dumps. Its purpose here is numerical debugging: tracking dispatched calls and isolating numerical divergence.

Brian：它是一种自定义的 TorchDispatchMode，提供类似性能分析的运行时信息转储。这里的用途是数值调试：跟踪已分派的调用，并定位数值分歧。

Ava: How does it find the first bad operation?

Ava：它怎么找到第一个出问题的操作？

Brian: The key feature is tensor hashing. You run two versions of a model with the same input, and you expect corresponding tensors to have the same deterministic hash. You then look for the point where the hashes stop matching. That operation is often where the behavior starts to differ.

Brian：关键功能是张量哈希。你用相同的输入运行模型的两个版本，并预期相应的张量具有相同的确定性哈希值。然后查看哈希值从哪里开始不匹配。行为往往就是从那个操作开始出现差异的。

Ava: So instead of comparing only the final loss, you compare the whole trail of tensors.

Ava：所以你比较的是沿途所有张量，而不只是最终的损失值。

Brian: Exactly. It's like checking every station on a train route instead of discovering at the final station that something went wrong. DebugMode also records dispatched operations and TorchInductor-compiled Triton kernels.

Brian：没错。这就像沿着火车线路逐站检查，而不是到了终点站才发现出了问题。DebugMode 还会记录已分派的操作以及由 TorchInductor 编译的 Triton 内核。

Ava: Can engineers add their own information to those records?

Ava：工程师能在这些记录里添加自己的信息吗？

Brian: Yes. Dispatch hooks let users register custom hooks to annotate calls. The article presents three main capabilities: runtime logging, tensor hashing, and dispatch hooks.

Brian：可以。分派钩子允许用户注册自定义钩子，为调用添加注释。文章介绍了三项主要能力：运行时日志记录、张量哈希和分派钩子。

Ava: Are these production-ready APIs?

Ava：这些 API 已经可以用于生产环境了吗？

Brian: The numerical debugging features are also labeled API-UNSTABLE. The article points readers to a tutorial for details, so teams should read that before building a long-term workflow around them.

Brian：这些数值调试功能也被标记为 API-UNSTABLE。文章引导读者阅读教程了解详情，因此团队在围绕它们建立长期工作流程前，应先阅读那份教程。

Ava: Let's cover the non-feature updates. TorchScript is now deprecated.

Ava：再来说说非功能性更新。TorchScript 现在已被弃用。

Brian: Correct. In PyTorch 2.10, TorchScript is deprecated, and the article says torch.export should be used instead. It points to a talk from the PyTorch Technical Committee for more detail.

Brian：没错。在 PyTorch 2.10 中，TorchScript 已被弃用，文章说应改用 torch.export。它还提供了 PyTorch Technical Committee 一场演讲的链接，供读者了解更多细节。

Ava: That could affect older deployment pipelines.

Ava：这可能会影响较旧的部署流程。

Brian: It could, especially if they still depend on TorchScript. The release blog doesn't provide a migration timeline, so the safe takeaway is to start evaluating torch.export and consult the linked guidance.

Brian：有可能，特别是那些仍然依赖 TorchScript 的流程。发布博客没有提供迁移时间表，因此稳妥的结论是开始评估 torch.export，并查阅文中链接的指导资料。

Ava: There is also tlparse and TORCH_TRACE for compiler bug reports. What problem do they solve?

Ava：还有用于编译器错误报告的 tlparse 和 TORCH_TRACE。它们解决什么问题？

Brian: Sometimes a compiler issue is too complex for a small standalone reproduction. tlparse results provide a log format that's easier to upload and share on GitHub. Even people outside the PyTorch project can extract useful information from those logs.

Brian：有时，编译器问题太复杂，没法用一个小型独立复现案例来说明。tlparse 的结果采用一种更容易上传到 GitHub 并分享的日志格式。即使不是 PyTorch 项目成员，也能从这些日志中提取有用信息。

Ava: So attaching a tlparse artifact can give maintainers more context than a short description?

Ava：所以，附上 tlparse 产物能比简短描述给维护者提供更多背景信息？

Brian: That's the idea. The article encourages users to attach tlparse log artifacts when reporting compiler bugs. It also mentions TORCH_TRACE as part of this clearer reporting workflow.

Brian：正是如此。文章鼓励用户在报告编译器错误时附上 tlparse 日志产物，还提到 TORCH_TRACE 也是这一更清晰的报告流程的一部分。

Ava: One final change is the release cadence for 2026.

Ava：最后一项变化是 2026 的发布频率。

Brian: Yes. The expected cadence increases from quarterly releases to one release every two months. The article refers readers to the published release schedule.

Brian：对。预计发布频率会从每季度一次提高到每两个月一次。文章请读者参阅已公布的发布计划。

Ava: More frequent releases mean faster access to improvements, but also more upgrade decisions.

Ava：发布更频繁意味着能更快用上改进，但也意味着要更频繁地决定是否升级。

Brian: That's a reasonable implication. The release itself simply states the new schedule and encourages users to try 2.10 and report issues.

Brian：这样理解很合理。此次发布本身只是说明了新计划，并鼓励用户试用 2.10、报告问题。

Ava: Let's make the practical advice concrete. If I'm a framework engineer, what should I test first?

Ava：我们把实用建议说具体些。如果我是一名框架工程师，应该先测试什么？

Brian: First, check your torch.compile workflow on Python 3.14. Python 3.14t, the freethreaded build, is experimentally supported as well. Second, try Combo Kernels and varlen_attn() in isolated experiments because those APIs are unstable and have specific hardware or backend limits.

Brian：首先，检查你在 Python 3.14 上的 torch.compile 工作流程。Python 3.14t 这个自由线程构建版本也获得了实验性支持。其次，在独立实验中试用 Combo Kernels 和 varlen_attn()，因为这些 API 尚不稳定，而且对硬件或后端有特定限制。

Ava: And for numerical issues?

Ava：那数值问题呢？

Brian: Enable deterministic algorithms, then use DebugMode with tensor hashing to compare two model versions. Log the dispatched calls and compiled Triton kernels, and add dispatch hooks if you need custom annotations.

Brian：启用确定性算法，然后使用带张量哈希功能的 DebugMode 比较两个模型版本。记录分派的调用和编译后的 Triton 内核；如果需要自定义注释，就添加分派钩子。

Ava: Before we close, give us the three-point recap.

Ava：结束之前，请用三点为我们回顾一下。

Brian: First, performance: Combo Kernels reduce kernel launch overhead, varlen_attn() handles ragged and packed sequences, DnXgeev improves the eigenvalue decomposition path, and Intel GPU support expands. Second, numerical confidence: torch.compile respects deterministic mode, and DebugMode helps locate divergence with runtime logs and tensor hashes. Third, ecosystem direction: TorchScript is deprecated, tlparse and TORCH_TRACE improve compiler bug reports, and 2026 moves to a two-month release cadence.

Brian：第一，性能方面：Combo Kernels 减少内核启动开销，varlen_attn() 处理不规则序列和打包序列，DnXgeev 改进特征值分解路径，Intel GPU 支持也得到扩展。第二，数值结果的可信度方面：torch.compile 遵循确定性模式，DebugMode 借助运行时日志和张量哈希帮助定位结果差异。第三，生态系统的发展方向：TorchScript 已弃用，tlparse 和 TORCH_TRACE 改进编译器错误报告，而 2026 将转为每两个月发布一次。

Ava: So the message is faster execution, clearer numerical debugging, and a faster-moving project.

Ava：所以核心信息是：执行更快，数值调试更清晰，项目的发展节奏也更快。

Brian: Exactly. Try the new features, check their current limitations, and report issues as PyTorch 2.10 evolves.

Brian：没错。试用新功能，了解它们目前的限制，并在 PyTorch 2.10 持续发展过程中报告问题。

Ava: Thanks for listening. We'll be back with another PyTorch release explained in plain English.

Ava：感谢收听。我们下次会再用通俗易懂的英语解读另一个 PyTorch 版本。

## 术语

| Term | 释义 |
|---|---|
| torch.compile() | PyTorch 的编译接口，用于将模型转换为更高效的执行形式 |
| torchinductor | PyTorch 编译器栈中的后端组件，负责生成和优化内核 |
| Combo Kernels | 水平融合优化，把多个相互独立的操作合并到一个 GPU kernel 中 |
| horizontal fusion | 水平融合，合并并行且无数据依赖的操作 |
| vertical fusion | 垂直融合，合并生产者和消费者这样的连续操作 |
| varlen_attn() | 支持变长、ragged 和 packed sequences 的注意力算子 |
| ragged sequence | 长度不一致的序列 |
| DnXgeev | cuSOLVER 提供的通用特征值分解例程 |
| use_deterministic_mode | 控制确定性执行模式的设置 |
| DebugMode | 用于记录运行时调用并调试数值分歧的 TorchDispatchMode |
| tensor hashing | 为张量生成确定性哈希，用于定位两次运行开始分歧的位置 |
| dispatch hooks | 用于给调用添加自定义注释的钩子 |
| TorchScript | PyTorch 的脚本化技术，在 2.10 中被弃用 |
| torch.export | 官方建议用于替代 TorchScript 的导出方式 |
| tlparse | 用于解析和分享编译器日志、辅助提交 bug 报告的工具 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's start with the big picture. | 我们先从整体情况讲起。 |
| What does that actually mean? | 这实际上是什么意思？ |
| That's the practical intuition. | 这就是实际上的直观理解。 |
| The important point is that... | 关键点是…… |
| That sounds stronger than... | 这听起来比……更强。 |
| Let's make the practical advice concrete. | 我们把实用建议说具体一点。 |
| Before we close, give us the three-point recap. | 结束前，请给我们做一个三点总结。 |
| That's the idea. | 大致就是这个意思。 |
