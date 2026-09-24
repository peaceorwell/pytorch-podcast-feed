# PyTorch 2.10: Faster Kernels, Safer Numbers, and a New Release Rhythm

原文：[PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/)

## 摘要

本期对话介绍 PyTorch 2.10 的主要更新，重点包括性能优化、变长注意力和数值调试。Ava 与 Brian 解释了 Combo Kernels 如何减少 GPU kernel launch overhead，以及 varlen_attn() 对 ragged 和 packed sequences 的支持。节目还讨论了确定性执行、DebugMode、Tensor hashing、Intel GPU 增强，以及 TorchScript 弃用和新的发布节奏。最后，两位主持人总结了升级时需要关注的功能限制与后续方向。

## 对话

**Ava:** If you're training large models, a tiny numerical difference can turn into a very long debugging session. And a little kernel overhead can quietly slow down every step. PyTorch 2.10 is aimed at both problems.

**Brian:** Exactly. This release pushes performance, but it also makes numerical behavior easier to inspect. It adds Python 3.14 support for torch.compile(), new attention and fusion features, and tools for finding where two runs start to disagree.

**Ava:** Let's start with the big picture. What changed in this release?

**Brian:** PyTorch 2.10 contains four thousand one hundred sixty commits from five hundred thirty-six contributors since 2.9. The article groups the changes into performance, numerical debugging, and some non-feature updates.

**Ava:** And the release is part of the larger two-series compiler story, right?

**Brian:** Yes. Performance has been a focus throughout the two-x releases, building on the PyTorch compiler stack introduced in 2.0. In 2.10, that means more work in torchinductor, more compiler support, and better ways to test numerical behavior.

**Ava:** Let's talk about the kernel work first. I saw the phrase Combo Kernels. What does that actually mean?

**Brian:** Combo Kernels are a horizontal fusion optimization in torchinductor. They combine multiple independent operations, with no data dependencies, into one unified GPU kernel.

**Ava:** So this is different from the usual producer-consumer fusion?

**Brian:** Right. Vertical fusion combines sequential operations, where one operation produces data for the next. Combo Kernels fuse parallel operations instead. Imagine several small checkout lines merging into one larger line, even though none of the customers depend on one another.

**Ava:** The benefit is fewer kernel launches?

**Brian:** That's the goal: reduced kernel launch overhead. The article doesn't give a benchmark number, but it explains that independent operations can now share one launch. The feature is marked API-UNSTABLE, so users should expect it to evolve.

**Ava:** The next item is varlen_attn(). The name sounds like variable-length attention.

**Brian:** That's exactly what it is. varlen_attn() is a new torch.nn.attention operation for ragged and packed sequences. Ragged means the sequences can have different lengths, and packed means they can be stored together without padding every item to the same size.

**Ava:** Does it support training, or only inference?

**Brian:** It supports both forward and backward, and it's torch.compile-able. Right now, FlashAttention 2, or FA2, supports it. The article says there are plans to add cuDNN and FlashAttention 4 support.

**Ava:** What hardware and data types are supported?

**Brian:** For now, it runs on NVIDIA CUDA with an A100 GPU or newer. It supports BF16 and FP16 data types. Again, this is API-UNSTABLE, so those boundaries may change later.

**Ava:** Variable-length attention can save work because you don't calculate attention on padding, right?

**Brian:** That is the practical intuition, although the article focuses on the API and its current support rather than giving a measured speedup. The important point is that ragged and packed sequences now have a dedicated operation that works with compilation and gradients.

**Ava:** There is also an eigenvalue update. That sounds more specialized.

**Brian:** It is more specialized, but useful for numerical workloads. PyTorch linalg can now use cuSOLVER's DnXgeev to provide highly efficient general eigenvalue decomposition on NVIDIA GPUs.

**Ava:** So PyTorch is connecting its linear algebra path to a more efficient cuSOLVER routine?

**Brian:** Yes. That's the whole claim in the release note: use DnXgeev through cuSOLVER for efficient general eigenvalue decompositions on NVIDIA GPUs. The article doesn't provide a performance figure.

**Ava:** What about Intel GPUs? The release notes list several changes there.

**Brian:** The support expands to the latest Intel Core Ultra Series 3 with Intel Arc Graphics, on both Windows and Linux. PyTorch also adds FP8 support on Intel GPUs.

**Ava:** FP8 support can mean a lot of different things. What does this release include?

**Brian:** It adds commonly used basic operators, such as type promotion and shape operators, plus scaled matrix multiplication using tensor-wise and channel-wise scaling factors.

**Ava:** Anything beyond FP8?

**Brian:** Yes. Aten operator MatMul gets complex data type support on Intel GPUs. The PyTorch C++ Extension API also extends SYCL support, so users can implement new custom operators on Windows. And Intel GPU unit-test coverage is broadened.

**Ava:** Now let's move to determinism. Why is this becoming such a big issue?

**Brian:** The article connects it to post-training with distributed reinforcement learning workflows. At that scale, run-to-run determinism makes debugging training runs easier. It also helps people test code that uses torch.compile reliably.

**Ava:** What changed in 2.10?

**Brian:** torch.compile() now respects use_deterministic_mode. Users can turn deterministic algorithms on with torch.use_deterministic_algorithms(True). The article says this makes two invocations of torch.compile perform the same operations exactly.

**Ava:** That sounds stronger than just getting similar outputs.

**Brian:** Yes, the wording is about performing the same operations exactly. For someone chasing a rare divergence, that gives a much steadier baseline.

**Ava:** And then there's DebugMode. Is it a profiler?

**Brian:** It's a custom TorchDispatchMode that provides profiling-style runtime dumps. Its purpose here is numerical debugging: tracking dispatched calls and isolating numerical divergence.

**Ava:** How does it find the first bad operation?

**Brian:** The key feature is tensor hashing. You run two versions of a model with the same input, and you expect corresponding tensors to have the same deterministic hash. You then look for the point where the hashes stop matching. That operation is often where the behavior starts to differ.

**Ava:** So instead of comparing only the final loss, you compare the whole trail of tensors.

**Brian:** Exactly. It's like checking every station on a train route instead of discovering at the final station that something went wrong. DebugMode also records dispatched operations and TorchInductor-compiled Triton kernels.

**Ava:** Can engineers add their own information to those records?

**Brian:** Yes. Dispatch hooks let users register custom hooks to annotate calls. The article presents three main capabilities: runtime logging, tensor hashing, and dispatch hooks.

**Ava:** Are these production-ready APIs?

**Brian:** The numerical debugging features are also labeled API-UNSTABLE. The article points readers to a tutorial for details, so teams should read that before building a long-term workflow around them.

**Ava:** Let's cover the non-feature updates. TorchScript is now deprecated.

**Brian:** Correct. In PyTorch 2.10, TorchScript is deprecated, and the article says torch.export should be used instead. It points to a talk from the PyTorch Technical Committee for more detail.

**Ava:** That could affect older deployment pipelines.

**Brian:** It could, especially if they still depend on TorchScript. The release blog doesn't provide a migration timeline, so the safe takeaway is to start evaluating torch.export and consult the linked guidance.

**Ava:** There is also tlparse and TORCH_TRACE for compiler bug reports. What problem do they solve?

**Brian:** Sometimes a compiler issue is too complex for a small standalone reproduction. tlparse results provide a log format that's easier to upload and share on GitHub. Even people outside the PyTorch project can extract useful information from those logs.

**Ava:** So attaching a tlparse artifact can give maintainers more context than a short description?

**Brian:** That's the idea. The article encourages users to attach tlparse log artifacts when reporting compiler bugs. It also mentions TORCH_TRACE as part of this clearer reporting workflow.

**Ava:** One final change is the release cadence for 2026.

**Brian:** Yes. The expected cadence increases from quarterly releases to one release every two months. The article refers readers to the published release schedule.

**Ava:** More frequent releases mean faster access to improvements, but also more upgrade decisions.

**Brian:** That's a reasonable implication. The release itself simply states the new schedule and encourages users to try 2.10 and report issues.

**Ava:** Let's make the practical advice concrete. If I'm a framework engineer, what should I test first?

**Brian:** First, check your torch.compile workflow on Python 3.14. Python 3.14t, the freethreaded build, is experimentally supported as well. Second, try Combo Kernels and varlen_attn() in isolated experiments because those APIs are unstable and have specific hardware or backend limits.

**Ava:** And for numerical issues?

**Brian:** Enable deterministic algorithms, then use DebugMode with tensor hashing to compare two model versions. Log the dispatched calls and compiled Triton kernels, and add dispatch hooks if you need custom annotations.

**Ava:** Before we close, give us the three-point recap.

**Brian:** First, performance: Combo Kernels reduce kernel launch overhead, varlen_attn() handles ragged and packed sequences, DnXgeev improves the eigenvalue decomposition path, and Intel GPU support expands. Second, numerical confidence: torch.compile respects deterministic mode, and DebugMode helps locate divergence with runtime logs and tensor hashes. Third, ecosystem direction: TorchScript is deprecated, tlparse and TORCH_TRACE improve compiler bug reports, and 2026 moves to a two-month release cadence.

**Ava:** So the message is faster execution, clearer numerical debugging, and a faster-moving project.

**Brian:** Exactly. Try the new features, check their current limitations, and report issues as PyTorch 2.10 evolves.

**Ava:** Thanks for listening. We'll be back with another PyTorch release explained in plain English.

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
