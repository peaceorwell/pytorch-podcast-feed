# PyTorch 2.10: Faster Kernels, Safer Numbers, and a New Release Rhythm

原文：[PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/)

## 摘要

本期对话介绍 PyTorch 2.10 的主要更新，重点包括性能优化、变长注意力和数值调试。Ava 与 Brian 解释了 Combo Kernels 如何减少 GPU kernel launch overhead，以及 varlen_attn() 对 ragged 和 packed sequences 的支持。节目还讨论了确定性执行、DebugMode、Tensor hashing、Intel GPU 增强，以及 TorchScript 弃用和新的发布节奏。最后，两位主持人总结了升级时需要关注的功能限制与后续方向。

## 对话

Ava: If you're training large models, a tiny numerical difference can turn into a very long debugging session.

Ava：训练大模型时，一点数值差异就可能让你调试很久。

And a little kernel overhead can quietly slow down every step. PyTorch 2.

而一点内核开销也会悄悄拖慢每一步。PyTorch 2.

10 is aimed at both problems.

10 正是为解决这两个问题而来。

Brian: Exactly.

Brian：没错。

This release pushes performance, but it also makes numerical behavior easier to inspect.

这个版本提升了性能，也让数值行为更容易检查。

It adds Python 3. 14 support for torch.

它新增了 Python 3.14 对 torch.

compile(), new attention and fusion features, and tools for finding where two runs start to disagree.

compile() 的支持，还带来新的注意力与融合功能，以及定位两次运行从哪里开始出现差异的工具。

Ava: Let's start with the big picture. What changed in this release?

Ava：先看整体。这次发布有哪些变化？

Brian: PyTorch 2.

Brian：再看 PyTorch 2.

10 contains four thousand one hundred sixty commits from five hundred thirty-six contributors since 2.

10 版收录了 536 位贡献者的 4160 次提交，从 2.

9.

9 版算起。

The article groups the changes into performance, numerical debugging, and some non-feature updates.

文章将这些变化分为性能、数值调试和其他非功能更新。

Ava: And the release is part of the larger two-series compiler story, right?

Ava：这次发布也是整个 2.x 系列编译器发展的一部分，对吧？

Brian: Yes.

Brian：是的。

Performance has been a focus throughout the two-x releases, building on the PyTorch compiler stack introduced in 2.

整个 2.x 系列都在持续提升性能，基础是 PyTorch 在 2.

0. In 2.

0 版引入的编译器体系。在 2.

10, that means more work in torchinductor, more compiler support, and better ways to test numerical behavior.

10 版中，torchinductor 得到更多改进，编译器支持更广，数值行为也更容易测试。

Ava: Let's talk about the kernel work first. I saw the phrase Combo Kernels.

Ava：先聊内核方面的改进。我看到了“Combo Kernels”这个词。

What does that actually mean?

它具体是什么意思？

Brian: Combo Kernels are a horizontal fusion optimization in torchinductor.

Brian：Combo Kernels 是 torchinductor 中的一项横向融合优化。

They combine multiple independent operations, with no data dependencies, into one unified GPU kernel.

它把多个没有数据依赖关系的独立操作合并成一个 GPU 内核。

Ava: So this is different from the usual producer-consumer fusion?

Ava：这和常见的生产者与消费者融合不同？

Brian: Right.

Brian：对。

Vertical fusion combines sequential operations, where one operation produces data for the next.

纵向融合会合并前后相继的操作，前一个操作为后一个提供数据。

Combo Kernels fuse parallel operations instead.

Combo Kernels 则融合并行操作。

Imagine several small checkout lines merging into one larger line, even though none of the customers depend on one another.

想象几条短的结账队伍汇成一条长队，尽管顾客之间互不依赖。

Ava: The benefit is fewer kernel launches?

Ava：好处是减少内核启动次数？

Brian: That's the goal: reduced kernel launch overhead.

Brian：对，目的是降低内核启动开销。

The article doesn't give a benchmark number, but it explains that independent operations can now share one launch.

文章没有给出基准测试数据，但解释了独立操作现在可以共用一次启动。

The feature is marked API-UNSTABLE, so users should expect it to evolve.

这项功能标记为 API-UNSTABLE，用户应预期它还会变化。

Ava: The next item is varlen_attn(). The name sounds like variable-length attention.

Ava：下一项是 varlen_attn()。听名字像是变长注意力。

Brian: That's exactly what it is. varlen_attn() is a new torch. nn.

Brian：正是。varlen_attn() 是 torch.nn 中的新型

attention operation for ragged and packed sequences.

注意力算子，适用于不规则序列和打包序列。

Ragged means the sequences can have different lengths, and packed means they can be stored together without padding every item to the same size.

不规则序列的长度可以不同；打包序列则能存放在一起，无须把每个序列填充到相同长度。

Ava: Does it support training, or only inference?

Ava：它支持训练，还是只支持推理？

Brian: It supports both forward and backward, and it's torch. compile-able.

Brian：前向和反向计算都支持，也可以用 torch.compile 编译。

Right now, FlashAttention 2, or FA2, supports it.

目前，FlashAttention 2（FA2）支持它。

The article says there are plans to add cuDNN and FlashAttention 4 support.

文章说，未来计划加入 cuDNN 和 FlashAttention 4 支持。

Ava: What hardware and data types are supported?

Ava：支持哪些硬件和数据类型？

Brian: For now, it runs on NVIDIA CUDA with an A100 GPU or newer.

Brian：目前支持 NVIDIA CUDA，GPU 需为 A100 或更新型号。

It supports BF16 and FP16 data types.

支持 BF16 和 FP16 数据类型。

Again, this is API-UNSTABLE, so those boundaries may change later.

同样，它标记为 API-UNSTABLE，这些支持范围以后可能变化。

Ava: Variable-length attention can save work because you don't calculate attention on padding, right?

Ava：变长注意力能省去填充位置上的计算，对吧？

Brian: That is the practical intuition, although the article focuses on the API and its current support rather than giving a measured speedup.

Brian：实际意义可以这么理解。不过文章重点介绍的是 API 和当前支持情况，没有给出实测加速数据。

The important point is that ragged and packed sequences now have a dedicated operation that works with compilation and gradients.

关键是，不规则序列和打包序列现在有了专用算子，而且支持编译和梯度计算。

Ava: There is also an eigenvalue update. That sounds more specialized.

Ava：特征值计算也有更新。听起来更专门一些。

Brian: It is more specialized, but useful for numerical workloads.

Brian：是更专门，但对数值计算很有用。

PyTorch linalg can now use cuSOLVER's DnXgeev to provide highly efficient general eigenvalue decomposition on NVIDIA GPUs.

PyTorch linalg 现在可以调用 cuSOLVER 的 DnXgeev，在 NVIDIA GPU 上高效完成一般特征值分解。

Ava: So PyTorch is connecting its linear algebra path to a more efficient cuSOLVER routine?

Ava：所以 PyTorch 的线性代数功能接入了效率更高的 cuSOLVER 例程？

Brian: Yes.

Brian：对。

That's the whole claim in the release note: use DnXgeev through cuSOLVER for efficient general eigenvalue decompositions on NVIDIA GPUs.

发布说明的主张就是：在 NVIDIA GPU 上通过 cuSOLVER 使用 DnXgeev，高效完成一般特征值分解。

The article doesn't provide a performance figure.

文章没有提供性能数据。

Ava: What about Intel GPUs? The release notes list several changes there.

Ava：Intel GPU 呢？发布说明列出了几项相关更新。

Brian: The support expands to the latest Intel Core Ultra Series 3 with Intel Arc Graphics, on both Windows and Linux.

Brian：支持范围扩展到搭载 Intel Arc Graphics 的最新 Intel Core Ultra Series 3，覆盖 Windows 和 Linux。

PyTorch also adds FP8 support on Intel GPUs.

PyTorch 还为 Intel GPU 增加了 FP8 支持。

Ava: FP8 support can mean a lot of different things. What does this release include?

Ava：FP8 支持可能涵盖很多内容。这次具体包含什么？

Brian: It adds commonly used basic operators, such as type promotion and shape operators, plus scaled matrix multiplication using tensor-wise and channel-wise scaling factors.

Brian：包括类型提升、形状算子等常用基础算子，以及使用按张量和按通道缩放因子的缩放矩阵乘法。

Ava: Anything beyond FP8?

Ava：除了 FP8，还有别的吗？

Brian: Yes. Aten operator MatMul gets complex data type support on Intel GPUs.

Brian：有。Intel GPU 上的 Aten MatMul 算子现在支持复数数据类型。

The PyTorch C++ Extension API also extends SYCL support, so users can implement new custom operators on Windows.

PyTorch C++ Extension API 也扩展了 SYCL 支持，让用户能在 Windows 上实现新的自定义算子。

And Intel GPU unit-test coverage is broadened.

Intel GPU 的单元测试覆盖范围也扩大了。

Ava: Now let's move to determinism. Why is this becoming such a big issue?

Ava：接下来谈确定性。为什么它变得这么重要？

Brian: The article connects it to post-training with distributed reinforcement learning workflows.

Brian：文章将它与分布式强化学习流程中的后训练联系起来。

At that scale, run-to-run determinism makes debugging training runs easier.

在这种规模下，多次运行结果保持确定性，能让训练调试更容易。

It also helps people test code that uses torch. compile reliably.

这也有助于可靠地测试使用 torch.compile 的代码。

Ava: What changed in 2. 10?

Ava：2.10 有什么变化？

Brian: torch. compile() now respects use_deterministic_mode.

Brian：torch.compile() 现在会遵循 use_deterministic_mode。

Users can turn deterministic algorithms on with torch.

用户可以通过以下调用开启确定性算法：

use_deterministic_algorithms(True).

调用 torch.use_deterministic_algorithms(True) 即可。

The article says this makes two invocations of torch.

文章说，两次调用 torch.compile

compile perform the same operations exactly.

会执行完全相同的操作。

Ava: That sounds stronger than just getting similar outputs.

Ava：这听起来比输出结果相近更进一步。

Brian: Yes, the wording is about performing the same operations exactly.

Brian：对，原文强调的是执行完全相同的操作。

For someone chasing a rare divergence, that gives a much steadier baseline.

追查偶发的结果偏差时，这能提供更稳定的基准。

Ava: And then there's DebugMode. Is it a profiler?

Ava：还有 DebugMode。它是性能分析器吗？

Brian: It's a custom TorchDispatchMode that provides profiling-style runtime dumps.

Brian：它是一种自定义 TorchDispatchMode，可生成类似性能分析记录的运行时转储。

Its purpose here is numerical debugging: tracking dispatched calls and isolating numerical divergence.

它在这里用于数值调试：追踪派发调用，定位数值偏差。

Ava: How does it find the first bad operation?

Ava：它怎么找到第一个出问题的操作？

Brian: The key feature is tensor hashing.

Brian：关键功能是张量哈希。

You run two versions of a model with the same input, and you expect corresponding tensors to have the same deterministic hash.

用相同输入运行模型的两个版本，预期对应张量会得到相同的确定性哈希值。

You then look for the point where the hashes stop matching.

然后找出哈希值从哪里开始不一致。

That operation is often where the behavior starts to differ.

行为往往就是从那个操作开始出现差异的。

Ava: So instead of comparing only the final loss, you compare the whole trail of tensors.

Ava：所以不只比较最终的损失值，还要比较沿途的所有张量。

Brian: Exactly.

Brian：正是这样。

It's like checking every station on a train route instead of discovering at the final station that something went wrong.

就像沿途逐站检查，而不是到了终点才发现出了问题。

DebugMode also records dispatched operations and TorchInductor-compiled Triton kernels.

DebugMode 还会记录派发的操作和由 TorchInductor 编译的 Triton 内核。

Ava: Can engineers add their own information to those records?

Ava：工程师能在这些记录中加入自己的信息吗？

Brian: Yes. Dispatch hooks let users register custom hooks to annotate calls.

Brian：能。派发钩子允许用户注册自定义钩子，为调用添加注释。

The article presents three main capabilities: runtime logging, tensor hashing, and dispatch hooks.

文章介绍了三项主要能力：运行时日志、张量哈希和派发钩子。

Ava: Are these production-ready APIs?

Ava：这些 API 可以用于生产环境了吗？

Brian: The numerical debugging features are also labeled API-UNSTABLE.

Brian：这些数值调试功能也被标记为 API-UNSTABLE。

The article points readers to a tutorial for details, so teams should read that before building a long-term workflow around them.

文章提供了教程链接，团队若要长期使用这些功能，应先读教程。

Ava: Let's cover the non-feature updates. TorchScript is now deprecated.

Ava：再谈谈功能之外的更新。TorchScript 现在已弃用。

Brian: Correct. In PyTorch 2. 10, TorchScript is deprecated, and the article says torch.

Brian：对。在 PyTorch 2.10 中，TorchScript 已弃用，文章建议

export should be used instead.

改用 torch.export。

It points to a talk from the PyTorch Technical Committee for more detail.

文章还推荐了一场 PyTorch 技术委员会的演讲，供读者了解详情。

Ava: That could affect older deployment pipelines.

Ava：这可能影响较早的部署流程。

Brian: It could, especially if they still depend on TorchScript.

Brian：确实，尤其是仍依赖 TorchScript 的流程。

The release blog doesn't provide a migration timeline, so the safe takeaway is to start evaluating torch.

发布博客没有给出迁移时间表，所以稳妥的做法是开始评估

export and consult the linked guidance.

torch.export，并参考文中链接的指南。

Ava: There is also tlparse and TORCH_TRACE for compiler bug reports.

Ava：编译器问题报告还可以用 tlparse 和 TORCH_TRACE。

What problem do they solve?

它们解决什么问题？

Brian: Sometimes a compiler issue is too complex for a small standalone reproduction.

Brian：有些编译器问题太复杂，难以写成简短的独立复现案例。

tlparse results provide a log format that's easier to upload and share on GitHub.

tlparse 的输出是更方便上传到 GitHub 分享的日志。

Even people outside the PyTorch project can extract useful information from those logs.

即使不是 PyTorch 项目成员，也能从这些日志中提取有用信息。

Ava: So attaching a tlparse artifact can give maintainers more context than a short description?

Ava：所以附上 tlparse 产物，比简短描述更能帮助维护者了解问题？

Brian: That's the idea.

Brian：正是如此。

The article encourages users to attach tlparse log artifacts when reporting compiler bugs.

文章建议用户报告编译器错误时附上 tlparse 日志。

It also mentions TORCH_TRACE as part of this clearer reporting workflow.

文章还提到用 TORCH_TRACE 帮助把问题报告得更清楚。

Ava: One final change is the release cadence for 2026.

Ava：最后一项变化是 2026 年的发布频率。

Brian: Yes.

Brian：对。

The expected cadence increases from quarterly releases to one release every two months.

预计将从每季度发布一次，改为每两个月发布一次。

The article refers readers to the published release schedule.

文章请读者查看已公布的发布日程。

Ava: More frequent releases mean faster access to improvements, but also more upgrade decisions.

Ava：发布更频繁，用户能更快用上改进，但也得更频繁地决定是否升级。

Brian: That's a reasonable implication.

Brian：这样理解很合理。

The release itself simply states the new schedule and encourages users to try 2.

此次发布只说明了新日程，并鼓励用户试用 2.

10 and report issues.

10 版并反馈问题。

Ava: Let's make the practical advice concrete.

Ava：说说具体该怎么做吧。

If I'm a framework engineer, what should I test first?

如果我是框架工程师，应该先测什么？

Brian: First, check your torch. compile workflow on Python 3. 14. Python 3.

Brian：首先，在 Python 3.14 上检查你的 torch.compile 工作流；Python 3.

14t, the freethreaded build, is experimentally supported as well.

14t 这个自由线程构建版也获得了实验性支持。

Second, try Combo Kernels and varlen_attn() in isolated experiments because those APIs are unstable and have specific hardware or backend limits.

其次，单独试验 Combo Kernels 和 varlen_attn()，因为这些 API 尚不稳定，而且对硬件或后端有特定限制。

Ava: And for numerical issues?

Ava：数值问题呢？

Brian: Enable deterministic algorithms, then use DebugMode with tensor hashing to compare two model versions.

Brian：开启确定性算法，再用 DebugMode 和张量哈希比较两个模型版本。

Log the dispatched calls and compiled Triton kernels, and add dispatch hooks if you need custom annotations.

记录实际调用和编译后的 Triton 内核；需要自定义标注时，再添加分发钩子。

Ava: Before we close, give us the three-point recap.

Ava：结束前，请用三点总结一下。

Brian: First, performance: Combo Kernels reduce kernel launch overhead, varlen_attn() handles ragged and packed sequences, DnXgeev improves the eigenvalue decomposition path, and Intel GPU support expands.

Brian：第一，性能：Combo Kernels 减少内核启动开销，varlen_attn() 处理长度不一和打包的序列，DnXgeev 改进特征值分解流程，Intel GPU 支持也得到扩展。

Second, numerical confidence: torch.

第二，数值可靠性：torch.compile

compile respects deterministic mode, and DebugMode helps locate divergence with runtime logs and tensor hashes.

遵循确定性模式；DebugMode 借助运行时日志和张量哈希定位差异。

Third, ecosystem direction: TorchScript is deprecated, tlparse and TORCH_TRACE improve compiler bug reports, and 2026 moves to a two-month release cadence.

第三，生态方向：TorchScript 已弃用，tlparse 和 TORCH_TRACE 改进编译器问题报告，2026 年起改为每两个月发布一次。

Ava: So the message is faster execution, clearer numerical debugging, and a faster-moving project.

Ava：也就是说，运行更快，数值问题更容易排查，项目更新也更频繁。

Brian: Exactly.

Brian：没错。

Try the new features, check their current limitations, and report issues as PyTorch 2.

试用新功能，了解当前限制，并随着 PyTorch 2.

10 evolves.

10 版持续发展，及时反馈问题。

Ava: Thanks for listening.

Ava：感谢收听。

We'll be back with another PyTorch release explained in plain English.

下次我们会继续用通俗的话解读 PyTorch 的新版本。

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
