# Muse Spark Open: The Model and the PyTorch Tools Behind It

原文：[Muse Spark Is Now an Open-Weights Model - PyTorch](https://pytorch.org/blog/muse-spark-open/)

## 摘要

Meta 发布了 Muse Spark Open 的模型权重，面向编程和长时间运行的智能体任务，并支持图像、音频、视频及文档理解。文章介绍了 Helion 内核在不同硬件上的性能，以及 TorchTitan 对模型微调的支持。节目区分了内核级加速与端到端吞吐提升，也指出文章没有提供编程能力的量化测试结果。

## 对话

Ava: A model with eight hundred and sixteen billion parameters is opening its weights. Why should a framework engineer care?

Ava：一个拥有8160亿参数的模型正在开放权重。框架工程师为什么应该关注？

Brian: Because Meta is releasing Muse Spark Open alongside work on inference kernels and fine-tuning tools. This announcement covers more than a download.

Brian：因为 Meta 在发布 Muse Spark Open 的同时，还在推进推理内核和微调工具的工作。这次发布不只是提供模型下载。

Ava: Okay, let's start with the download. What exactly is available?

Ava：好，那就先说下载。具体有哪些东西可以获取？

Brian: The weights are available on Hugging Face under the Muse Model License. The article calls Muse Spark Open a coding-optimized, open-weights model.

Brian：模型权重已在 Hugging Face 上提供，采用 Muse Model License。文章称 Muse Spark Open 是一个针对编程优化的开放权重模型。

Ava: And the huge parameter count? I also saw a much smaller active count.

Ava：那个庞大的参数量是怎么回事？我还看到一个小得多的活跃参数量。

Brian: Right. It has eight hundred and sixteen billion parameters in total, with forty-two billion active. Think of a large library with only some books open on the desk.

Brian：对。它总共有8160亿参数，其中420亿参数处于激活状态。可以把它想成一座大型图书馆，只把部分书摊开放在桌上。

Ava: That helps. How much material can it keep in view?

Ava：这样就好理解了。它一次能处理多少内容？

Brian: Its context window is one million tokens. Meta says that's useful for project-scale work, including complex dependencies, legacy code, and many files.

Brian：它的上下文窗口可容纳100万个词元。Meta表示，这对项目规模的工作很有用，包括处理复杂的依赖关系、遗留代码和大量文件。

Ava: So the motivation is working across a real codebase, where a fix may touch several files?

Ava：所以，它的出发点是在真实代码库中工作，因为一次修复可能涉及多个文件？

Brian: Exactly. The article describes training on multi-file refactors, debugging sessions, and long-running tasks. It also claims stronger first-attempt accuracy and reliable tool calling.

Brian：正是如此。文章介绍说，训练内容包括多文件重构、调试过程和长时间运行的任务。文章还声称，它首次尝试的准确率更高，工具调用也更可靠。

Ava: Can we put a number on that first-attempt claim?

Ava：首次尝试的准确率提升能用数字说明吗？

Brian: No. This article doesn't give a coding benchmark number for it. We can say what Meta claims, but we can't measure the size of the gain here.

Brian：不能。这篇文章没有给出相关的编程基准测试数据。我们可以转述 Meta 的说法，但无法从这里衡量提升有多大。

Ava: Fair. It also calls the model agentic. What does that mean in this case?

Ava：有道理。文章还说这个模型具有智能体能力。在这里是什么意思？

Brian: It means the model is trained for work that unfolds over many steps: plan, act, observe, then correct its course.

Brian：意思是，它接受过针对多步骤工作的训练：先规划，再行动、观察，然后调整方向。

Ava: Like debugging when the first change fails a test, and you have to inspect what happened?

Ava：就像调试时，第一次修改没通过测试，还得检查出了什么问题？

Brian: That's a useful analogy. The article says it's trained to browse, coordinate subagents for work that can be divided, and use computers across desktop, mobile, and browser.

Brian：这个类比很贴切。文章说，它接受过浏览网页、协调子智能体处理可拆分工作，以及在桌面端、移动端和浏览器中使用计算机的训练。

Ava: Does it tell people what it's about to do?

Ava：它会告诉人们自己接下来要做什么吗？

Brian: The article says it generates separate commentary to state its intent before acting. That's meant to make a long task easier to follow.

Brian：文章说，它会在行动前单独生成说明，表明自己的意图。这样是为了让人更容易跟上一个长任务的进展。

Ava: There's also a multimodal claim. What can it take in?

Ava：文章还提到多模态。它能接收哪些输入？

Brian: The article names image, audio, and text inputs. It also describes audio-video perception and document understanding.

Brian：文章列出了图像、音频和文本输入，还介绍了音视频感知和文档理解能力。

Ava: So, quick recap: this is a coding-focused model built for long tasks, with a large context window and several input types.

Ava：简单回顾一下：这是一个面向编程、为长任务设计的模型，拥有很大的上下文窗口，能接收多种输入。

Brian: Yes. It shares the same base model as Muse Spark one point three. Meta says this open-weights version received extra safety post-training for standalone use.

Brian：是的。它与Muse Spark 1.3使用相同的基础模型。Meta表示，这个开放权重版本接受了额外的安全后训练，以便独立使用。

Ava: Standalone use meaning someone might run it without system guardrails?

Ava：独立使用，是指有人可能在没有系统级安全防护的情况下运行它？

Brian: That's the situation the article describes. It calls the result a strong open-source choice for safety, though it doesn't give a safety score here.

Brian：文章描述的就是这种情况。它称这个模型在安全性方面是一个很有竞争力的开源选择，不过这里没有给出安全评分。

Ava: Now to the part framework engineers love: why bring Helion into this release?

Ava：现在说说框架工程师喜欢的部分：为什么这次发布要引入 Helion？

Brian: Helion is a PyTorch-native language for writing high-performance kernels across different hardware. Muse Spark Open ships with Helion versions of several operations.

Brian：Helion 是一种基于 PyTorch 的语言，用于编写可在不同硬件上运行的高性能内核。Muse Spark Open 附带了若干操作的 Helion 实现。

Ava: What's the practical problem those kernels solve?

Ava：这些内核解决了什么实际问题？

Brian: Different accelerators often call for specialized implementations. Helion aims to let developers tune kernels for hardware and workload choices without maintaining a separate hand-tuned version for every combination.

Brian：不同加速器通常需要专门的实现。Helion 的目标是让开发者能针对硬件和工作负载调整内核，而不必为每种组合分别维护一个手工调优的版本。

Ava: What choices can users tune for?

Ava：用户可以针对哪些方面进行调优？

Brian: The article lists accelerator architectures, quantization formats, tensor shapes, and parallelism strategies. That's a wide range of shapes for one kernel author to think about.

Brian：文章列出了加速器架构、量化格式、张量形状和并行策略。对编写一个内核的人来说，要考虑的组合相当多。

Ava: And the speed results? Give me the careful version.

Ava：那速度结果呢？请谨慎地说说。

Brian: For individual operations on NVIDIA G B three hundred, it reports speedups from one point zero three to three point nine five times over custom kernels in hardware-specific languages.

Brian：对于在NVIDIA GB300上执行的单项操作，据其报告，与用硬件专用语言编写的定制内核相比，加速幅度为1.03倍至3.95倍。

Ava: Wait, so that's a kernel comparison. It doesn't mean the whole model runs nearly four times faster?

Ava：等等，所以这比较的是内核，并不意味着整个模型的运行速度提高了近4倍，对吗？

Brian: Right. The article says those kernels ported directly to AMD M I three hundred and fifty, with similar performance improvements. Its reported overall end-to-end throughput gain there is three percent.

Brian：对。文章称，这些内核可以直接移植到AMD MI350，并获得类似的性能提升。据其报告，在该硬件上，整体端到端吞吐量提升了3%。

Ava: That's a useful distinction. Small operations can win a lot while the complete serving path gains less.

Ava：这个区别很有用。单项操作可能快很多，但完整服务流程的收益要小一些。

Brian: Exactly. The article also introduces Helion megakernels in SGLang. They combine multiple operations in one persistent kernel to improve low-latency inference.

Brian：没错。文章还介绍了 SGLang 中的 Helion 巨型内核。它们将多个操作合并到一个常驻内核中，以改善低延迟推理。

Ava: Persistent sounds like keeping a kitchen station running while several dishes move through it.

Ava：常驻听起来就像让厨房里的一个工作台一直运转，让几道菜依次经过。

Brian: That's the idea to keep in mind. The article says these megakernels can also be autotuned for different shapes and hardware.

Brian：可以这样理解。文章说，这些巨型内核也能针对不同的形状和硬件自动调优。

Ava: Where can people run the model?

Ava：人们可以在哪里运行这个模型？

Brian: The article says Muse Spark Open is optimized for the SGLang and vLLM inference engines, with optimization for NVIDIA, AMD, and Intel.

Brian：文章说，Muse Spark Open 针对 SGLang 和 vLLM 推理引擎做了优化，也针对 NVIDIA、AMD 和 Intel 做了优化。

Ava: And if I want to adapt it to my own workload?

Ava：如果我想让它适应自己的工作负载呢？

Brian: TorchTitan now includes a model definition for Muse Spark Open. It's PyTorch's platform for experimenting with and training generative models at scale.

Brian：TorchTitan 现在包含 Muse Spark Open 的模型定义。它是 PyTorch 用于大规模试验和训练生成式模型的平台。

Ava: What kind of tuning does TorchTitan support here?

Ava：TorchTitan 在这里支持哪些调优方式？

Brian: It supports multi-turn supervised fine-tuning and LoRA, short for low-rank adaptation. Its TitanRL component supports reinforcement learning.

Brian：它支持多轮监督微调和 LoRA，后者是低秩适配的缩写。它的 TitanRL 组件支持强化学习。

Ava: What's still unclear after reading this announcement?

Ava：读完这份公告，还有哪些地方不清楚？

Brian: We don't get coding benchmark figures or detailed safety results here. And the stated kernel speedups alone can't tell us end-to-end performance on every setup.

Brian：这里没有编程基准测试数据，也没有详细的安全性结果。仅凭公布的内核加速倍数，也无法判断各种配置下的端到端性能。

Ava: Let's close with three points. First, the weights are available under the Muse Model License. Second, Helion supplies portable, tunable kernels.

Ava：最后我们说3点。第一，权重依据Muse Model License提供。第二，Helion提供可移植、可调优的内核。

Brian: Third, TorchTitan supports further tuning. That's Muse Spark Open from model to serving to training. Thanks for listening.

Brian：第三，TorchTitan 支持进一步调优。这就是 Muse Spark Open 从模型、服务到训练的情况。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| open-weights model | 开放模型权重、可按相应许可获取权重的模型 |
| active parameters | 一次处理过程中参与计算的参数 |
| context window | 模型一次可纳入上下文的文本容量 |
| agentic | 能够围绕多步骤任务规划、行动、观察和调整的 |
| multimodal | 可处理多种输入模态的 |
| Helion | 用于编写高性能、可移植内核的 PyTorch 原生语言 |
| kernel | 在加速硬件上执行特定计算操作的程序 |
| autotuned | 针对不同硬件或计算形状自动调优的 |
| megakernels | 将多个操作融合到单个持久化内核中的实现 |
| end-to-end throughput | 完整推理流程的整体吞吐量 |
| TorchTitan | 用于生成式模型实验和大规模训练的 PyTorch 原生平台 |
| supervised fine-tuning | 使用有监督数据继续调整模型 |
| LoRA | 低秩适配，一种模型微调方法 |
| reinforcement learning | 强化学习 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's start with the download. | 我们先从如何获取它说起。 |
| That helps. | 这样就好理解了。 |
| Can we put a number on that? | 这个能用数字说明吗？ |
| Fair. | 有道理。 |
| What's the practical problem? | 实际要解决的问题是什么？ |
| Give me the careful version. | 请说一个严谨的版本。 |
| Wait, so that's a kernel comparison. | 等等，也就是说这是内核层面的比较。 |
| That's a useful distinction. | 这个区别很关键。 |

## 中文内容总结

Meta 于 2026 年 10 月 15 日发布 Muse Spark Open 权重，开发者可在 Hugging Face 按 Muse Model License 获取。模型总参数量为 816B、每次激活 42B，上下文窗口为 100 万 token，与 Muse Spark 1.3 共用基础模型。文章重点介绍其编程代理能力、Helion 推理算子和 TorchTitan 微调支持。

### 模型能力

Meta 称，Muse Spark Open 针对多文件重构、调试和长期编程任务训练，旨在提高首次尝试的准确率、工具调用可靠性和代码库理解能力。模型可处理文本、图像、文档、音频及音视频输入；其代理训练涵盖网页浏览、多模态浏览、跨站点任务，以及在代理框架中协调子代理、操作桌面、移动端和浏览器。模型还能在执行动作前单独说明意图。开放权重版本增加了安全后训练，以适应脱离系统防护措施的独立使用。上述能力和安全性主要是发布方的描述，文章没有提供相应的基准测试数据。

### Helion 推理优化

Muse Spark Open 支持在 SGLang 和 vLLM 上优化推理，面向 NVIDIA、AMD 和 Intel 硬件。发布方用 PyTorch 原生内核语言 Helion 实现 RoPE、RMSNorm、量化等操作。在 NVIDIA GB300 上，这些算子相对以 CUDA、CuTe DSL 等硬件专用语言编写的定制内核，测得 1.03 至 3.95 倍加速；移植到 AMD MI350 后，发布方称算子获得相近改进，端到端吞吐量整体提高 3%。文章还介绍了 SGLang 中用 Helion 编写的 megakernel：它将多个操作融合进一个常驻内核，以改善低延迟推理，并可按形状和硬件自动调优。算子级加速范围不能直接视为整个模型的推理加速范围。

### 微调与实际使用

TorchTitan 新增 Muse Spark Open 的模型定义，支持多轮监督微调、LoRA，以及通过 TitanRL 进行强化学习微调。开发者可获取权重，并按任务调整模型；Helion 内核也可针对加速器架构、量化格式、张量形状和并行策略继续调优。文章没有给出微调效果、资源需求或各类硬件配置下的完整性能数据，因此实际部署效果仍取决于具体工作负载和配置。
