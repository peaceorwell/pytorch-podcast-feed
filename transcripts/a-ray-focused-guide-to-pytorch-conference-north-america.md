# Ray Across the PyTorch Stack: Data, Training, Serving, and RL

原文：[A Ray-Focused Guide to PyTorch Conference North America](https://pytorch.org/blog/a-ray-focused-guide-to-pytorch-conference-north-america/)

## 摘要

本文以 PyTorch Conference North America 2026 的 Ray 相关议程为线索，介绍 Ray 如何贯穿数据处理、分布式训练、部署和后训练。对话重点讨论 LinkedIn 的三层训练架构、Uber Eats 的数据与训练改造，以及 Pinterest 将平台职责与训练内循环分离的设计。SkyRL 和 Ray Core 的议题进一步展示了强化学习对异步执行、资源调度和权重传输的需求。节目也强调，这是一篇会议导览，文中的性能数据属于特定案例，文章未提供足以复现或全面比较的实验细节。

## 对话

Ava: Your model needs more machines, but suddenly you're debugging data loading, storage, and scheduling.

Ava：模型需要更多机器，你却突然要排查数据加载、存储和调度问题。

When did training become an infrastructure project?

训练什么时候变成基础设施项目了？

Brian: That's why this conference guide matters.

Brian：这就是这份会议指南的价值。

It follows Ray across those problems, from preparing data to training and production.

它介绍了 Ray 如何贯穿这些环节，从数据准备到训练和生产部署。

The interesting question is how those pieces fit together.

值得探讨的是，这些环节如何衔接。

Ava: I'm Ava, a software engineer outside the PyTorch core team.

Ava：我是 Ava，一名不在 PyTorch 核心团队的软件工程师。

Brian, you're our PyTorch guide. Give us the starting point: what's Ray?

Brian，你是我们的 PyTorch 向导。先说说 Ray 是什么？

Brian: Ray is a distributed compute framework.

Brian：Ray 是一个分布式计算框架。

It helps teams scale AI frameworks, libraries, and models across machines.

它帮助团队将 AI 框架、库和模型扩展到多台机器。

The article describes a consistent developer surface across the AI lifecycle.

文章介绍了贯穿 AI 全生命周期的一致开发接口。

Ava: So it covers the journey from getting data ready to running a model for users?

Ava：也就是从准备数据，到把模型提供给用户使用？

Brian: Right.

Brian：对。

That includes curating multimodal data, meaning data with different types of content, training across multiple nodes, and deploying on virtual machines or clusters managed by Kubernetes.

这包括整理多模态数据，也就是包含不同类型内容的数据；在多个节点上训练；以及部署到虚拟机或 Kubernetes 管理的集群。

Ava: And this is a guide to upcoming talks, so we're getting previews?

Ava：这是一份即将举行的演讲指南，所以我们看到的是预告？

Brian: Exactly.

Brian：没错。

It's a Ray-focused guide to PyTorch Conference North America, twenty twenty-six, in San Jose.

这是一份关于 2026 年圣何塞 PyTorch 北美大会的 Ray 专题指南。

It gives us engineering examples, but it isn't a detailed benchmark report.

它提供了工程实例，但不是详细的基准测试报告。

Ava: The first theme is Ray and Kubernetes evolving together.

Ava：第一个主题是 Ray 与 Kubernetes 的协同演进。

Why does that deserve attention?

为什么值得关注？

Brian: The article describes two communities users depend on.

Brian：文章介绍了用户所依赖的两个社区。

The PyTorch Foundation includes PyTorch and Ray.

PyTorch 基金会涵盖 PyTorch 和 Ray。

The Cloud Native Computing Foundation, or CNCF, includes Kubernetes and other infrastructure projects.

云原生计算基金会（CNCF）涵盖 Kubernetes 和其他基础设施项目。

Ava: Separate communities, but the engineer building a system needs both.

Ava：社区虽不同，但构建系统的工程师两边都需要。

The organization chart doesn't make the integration work disappear.

组织架构图不会让集成工作凭空消失。

Brian: Exactly. The talks explore a unified open source AI stack.

Brian：没错。这些演讲探讨如何构建统一的开源 AI 技术栈。

Google, Anyscale, and the broader community are collaborating to connect those ecosystems and prevent fragmentation.

Google、Anyscale 和更广泛的社区正合作连接这两个生态，避免碎片化。

Ava: Okay, that's the motivation. Let's get closer to the training job.

Ava：好，这就是背景。我们来看看训练任务。

What's LinkedIn building?

LinkedIn 在构建什么？

Brian: A three-layer Ray-based training stack.

Brian：一个基于 Ray 的三层训练技术栈。

First, a data loading library built with Ray actors, which are workers that can keep state.

第一层是用 Ray actor 构建的数据加载库；actor 是能保留状态的工作进程。

It streams Avro, Parquet, and Iceberg data.

它以流式方式读取 Avro、Parquet 和 Iceberg 数据。

Ava: That's the data layer. What sits above it?

Ava：这是数据层。上面一层是什么？

Brian: Ray Train, with Fully Sharded Data Parallel, or FSDP, and Hybrid Sharded Data Parallel, or HSDP.

Brian：Ray Train，配合全分片数据并行（FSDP）和混合分片数据并行（HSDP）。

Those are distributed training strategies.

这些都是分布式训练策略。

The layer also supports elastic scaling and asynchronous checkpointing.

这一层还支持弹性扩缩容和异步保存检查点。

Ava: Elastic scaling means adjusting resources, and asynchronous checkpointing means saving training state without making everything wait for the whole save?

Ava：弹性扩缩容是调整资源；异步保存检查点是保存训练状态时，不必让所有操作都等它完成？

Brian: That's the basic idea. The third layer provisions clusters on demand.

Brian：基本就是这样。第三层按需配置集群。

But the central lesson here is separating data loading from the machines doing the GPU work.

但这里的关键经验是，让数据加载与执行 GPU 计算的机器分开。

Ava: GPU meaning graphics processing unit. So where does the loading happen?

Ava：GPU 就是图形处理器。那么数据在哪里加载？

Brian: On dedicated CPU nodes, using central processing units.

Brian：在专用 CPU 节点上，也就是使用中央处理器的节点。

Data travels through a zero-copy bridge, meaning the transfer avoids an extra data copy.

数据通过零拷贝桥接传输，避免额外复制。

Think of a separate preparation area feeding the kitchen.

就像单独的备料区为厨房供料。

Ava: Let the cooks cook. What did that change?

Ava：让厨师专心做菜。这样带来了什么变化？

Brian: The guide reports a fifty to seventy percent reduction in dataloader memory.

Brian：指南称，数据加载器的内存占用降低了 50% 至 70%。

It also mentions a situation where LinkedIn had to work around Ray to reclaim twenty-four percent of step time.

它还提到，LinkedIn 曾通过绕开 Ray，让每步训练耗时降低了 24%。

Ava: Wait, so a Ray success story includes working around Ray?

Ava：等等，一个 Ray 的成功案例还包括绕开 Ray？

Brian: Yes, and that's useful.

Brian：是的，这点很有参考价值。

An infrastructure choice can help overall while still creating a specific problem.

一种基础设施方案整体上可能有帮助，同时仍会带来某个具体问题。

The article doesn't explain that workaround, so we can't reconstruct it.

文章没有解释具体的绕开方法，所以我们无法还原。

Ava: The takeaway is to separate the work, then measure what still slows it down.

Ava：结论是先把工作分开，再测量还有什么在拖慢速度。

How does Uber's example compare?

Uber 的案例怎么样？

Brian: Uber Eats is migrating its recommendation stack from TensorFlow and Horovod to PyTorch.

Brian：Uber Eats 正在将推荐系统从 TensorFlow 和 Horovod 迁移到 PyTorch。

Ray Data replaces legacy Spark-based preprocessing with distributed feature-statistics computation and zero-copy batch transformations.

Ray Data 用分布式特征统计计算和零拷贝批量转换，取代了原先基于 Spark 的预处理。

Ava: Feature statistics being information calculated about the model's input features.

Ava：特征统计就是针对模型输入特征计算出的信息。

What results does the guide report?

指南报告了什么成果？

Brian: Data transformation is five times faster and uses ninety percent less memory.

Brian：数据转换速度提高了 5 倍，内存占用减少了 90%。

Training throughput improves twenty times.

训练吞吐量提高了 20 倍。

Those are reported results for Uber's redesign.

这些是 Uber 此次重新设计所报告的结果。

Ava: Twenty times is a big number. Can we tell how much came from Ray Data alone?

Ava：20 倍可不小。能看出其中有多少归功于 Ray Data 吗？

Brian: No.

Brian：不能。

The article doesn't isolate each change's contribution or provide the full measurement setup.

文章没有单独量化每项改动的贡献，也没有提供完整的测试条件。

It's an impressive case to investigate, but it doesn't predict another team's improvement.

这是个值得研究的出色案例，但不能据此预测其他团队能提升多少。

Ava: We've covered loading and transformation.

Ava：我们聊过数据加载和转换了。

What if the data simply doesn't arrive from storage fast enough?

如果数据从存储系统传来的速度就是不够快呢？

Brian: That's the storage talk's focus.

Brian：这正是那场存储主题演讲的重点。

It describes GPUs waiting on legacy REST-based access, a web request style.

演讲提到，传统的 REST 式访问会让 GPU 等待；REST 是一种网络请求方式。

Rapid Storage proposes a high-throughput protocol based on gRPC, a remote procedure call framework.

Rapid Storage 提出采用基于 gRPC 的高吞吐量协议；gRPC 是一种远程过程调用框架。

Ava: How does that reach PyTorch?

Ava：它怎么接入 PyTorch？

Brian: Through fsspec, a filesystem interface layer.

Brian：通过文件系统接口层 fsspec。

The team says the benefits extend across its ecosystem, including Ray.

团队表示，收益也覆盖包括 Ray 在内的生态系统。

For a storage-bound pipeline, storage is the delivery truck that keeps arriving late.

对受存储速度制约的处理流程来说，存储就像总是迟到的送货车。

Ava: And a faster kitchen still waits for that truck.

Ava：厨房再快，也得等那辆车。

There's also a climate model talk in the serving section. What's Ray doing there?

服务部署部分还有一场气候模型演讲。Ray 在其中做什么？

Brian: Ray helps scale training for a physics-constrained generative model used in climate downscaling.

Brian：Ray 帮助扩展用于气候降尺度的物理约束生成模型的训练规模。

The talk follows that model into production, including export to ONNX, the Open Neural Network Exchange format.

演讲还介绍了模型如何投入生产，包括导出为 ONNX，即开放神经网络交换格式。

Ava: Does the guide say Ray handles the entire serving path?

Ava：指南说整个服务流程都由 Ray 负责吗？

Brian: It places Ray in the training recipe.

Brian：它将 Ray 用于训练流程。

The serving path includes ONNX Runtime, Temporal, and a vLLM agent.

服务流程则包括 ONNX Runtime、Temporal 和一个 vLLM 智能体。

It also covers keeping the sampling process outside the exported graph and enforcing physics constraints during inference.

演讲还讨论了如何将采样过程留在导出的计算图之外，以及如何在推理时施加物理约束。

Ava: So we should pay attention to each tool's actual role.

Ava：所以我们得看清每个工具的实际作用。

Now, why does post-training get so much space?

那么，为什么后训练占了这么多篇幅？

Brian: Post-training means further training after a model's initial training.

Brian：后训练是指模型完成初始训练后继续训练。

At Pinterest, teams adopted different frameworks, data formats, and distributed strategies.

在 Pinterest，各团队采用了不同的框架、数据格式和分布式策略。

They spent weeks connecting open source frameworks to internal infrastructure before training even started.

训练还没开始，他们就得花几周时间把开源框架接入内部基础设施。

Ava: Weeks before the first step. That's a painful progress bar.

Ava：第一步还没开始就得等几周，这进度条看着真难受。

Brian: Their earlier platform owned the PyTorch training loop and reached ninety-five percent adoption.

Brian：他们之前的平台负责 PyTorch 训练循环，采用率达到了 95%。

But post-training loops lived inside fast-moving external frameworks.

但后训练循环存在于快速迭代的外部框架中。

Keeping platform ownership of that loop became a problem.

平台继续掌控训练循环就成了问题。

Ava: What's their new boundary?

Ava：他们现在怎么划分职责？

Brian: PTEnv is a lifecycle harness, a support structure around training.

Brian：PTEnv 是一个训练生命周期框架，为训练提供外围支持。

It owns data, orchestration, scaling, evaluation, and export.

它负责数据、编排、扩展、评估和导出。

The open source framework owns the inner training loop.

开源框架负责内部训练循环。

Ava: Like managing the theater while each production brings its own performance?

Ava：就像剧院负责运营，每个剧组负责自己的演出？

Brian: Yes.

Brian：对。

One interface supports separate-process execution with Ray for reinforcement learning, or RL.

同一个接口支持通过 Ray 以独立进程运行强化学习，即 RL。

It also supports in-process execution with MS-Swift for supervised fine-tuning, meaning training on supplied examples.

也支持通过 MS-Swift 在进程内运行监督微调，也就是用给定样本训练。

Framework selection becomes a configuration choice.

选择框架成了一项配置选择。

Ava: That sounds flexible. What difficulties remain?

Ava：听起来很灵活。还有什么难题？

Brian: The talk includes bottlenecks in synchronizing model weights and workarounds for mixture-of-experts models, or MoE models, which contain multiple expert components.

Brian：演讲谈到了同步模型权重时的瓶颈，以及针对混合专家模型（MoE）的变通办法；这种模型包含多个专家组件。

A shared interface doesn't remove those engineering problems.

统一接口并不能消除这些工程难题。

Ava: And SkyRL tackles reinforcement learning itself?

Ava：SkyRL 也解决强化学习本身的问题吗？

Brian: Right. It's a modular RL library.

Brian：对，它是一个模块化的强化学习库。

It separates training, inference, and environments, with Ray coordinating them.

它把训练、推理和环境分开，由 Ray 统筹协调。

The motivation includes agents handling longer tasks and interactions with multiple turns.

其动机包括让智能体处理更长的任务和多轮交互。

Ava: What scale does the session discuss?

Ava：这场分享讨论了多大规模？

Brian: Fully asynchronous RL training on mixture-of-experts models with more than three hundred fifty billion parameters, using Megatron and vLLM.

Brian：借助 Megatron 和 vLLM，在超过3500亿参数的混合专家模型上进行完全异步的强化学习训练。

It also covers a shared Tinker Engine for researchers using their own hardware.

还介绍了供研究人员在自有硬件上使用的共享 Tinker Engine。

Ava: Underneath that, what does Ray Core need to do?

Ava：底层的 Ray Core 需要做什么？

Brian: Schedule trainers and inference engines, then support moving fresh weights between them.

Brian：调度训练器和推理引擎，并支持在两者之间传输最新权重。

Its demo discusses scheduling across ten thousand nodes and Ray Direct Transport, which moves PyTorch tensors directly between GPUs.

演示还谈到跨一万个节点的调度，以及让 PyTorch 张量在 GPU 之间直接传输的 Ray Direct Transport。

Ava: Tensors being the arrays holding model data.

Ava：张量就是存放模型数据的数组。

Does scheduling at that scale establish end-to-end training performance?

这种规模的调度能证明端到端训练性能吗？

Brian: The guide doesn't establish that connection.

Brian：这篇指南没有证明两者之间的联系。

It presents scheduling scale and transport improvements as separate achievements.

它将调度规模和传输改进作为两项独立成果介绍。

The demo also promises a future roadmap, but the article doesn't give its details.

演示还预告了未来路线图，但文章没有给出细节。

Ava: Let's finish with three points.

Ava：最后总结三点。

First, Ray spans the AI lifecycle, and its integration with Kubernetes is a major theme.

第一，Ray 覆盖 AI 的整个生命周期，与 Kubernetes 的集成是一大主题。

Brian: Second, data preparation, storage, and weight movement deserve attention alongside model computation.

Brian：第二，除了模型计算，数据准备、存储和权重传输也值得关注。

The talks show concrete gains, plus bottlenecks that still need work.

这些演讲展示了具体进展，也指出了仍需解决的瓶颈。

Ava: Third, flexible platform boundaries matter as training frameworks change.

Ava：第三，随着训练框架变化，灵活的平台边界很重要。

Treat these conference results as specific cases, and bring questions about their measurement conditions.

应把这些会议成果视为具体案例，并追问其测量条件。

Brian: Thanks for listening, and we'll see you next time.

Brian：感谢收听，我们下次再见。

## 术语

| Term | 释义 |
|---|---|
| Ray | 分布式计算框架，用于扩展 AI 数据处理、训练和部署等工作负载。 |
| Kubernetes | 用于管理容器化应用和集群资源的编排系统。 |
| Ray actors | Ray 中能够保存状态的执行单元；文中的 LinkedIn 数据加载库基于这一机制。 |
| FSDP | Fully Sharded Data Parallel，全分片数据并行，一种分布式训练策略。 |
| HSDP | Hybrid Sharded Data Parallel，混合分片数据并行，一种分布式训练策略。 |
| async checkpointing | 异步检查点保存，让训练状态保存与其他训练工作尽可能重叠。 |
| zero-copy | 零拷贝，在相关数据传输或转换环节避免额外的数据复制。 |
| Ray Data | Ray 的数据处理组件；文中用于分布式特征统计和批次转换。 |
| fsspec | 文件系统接口层；文中的 Rapid Storage 通过它接入 PyTorch 及相关生态。 |
| ONNX | Open Neural Network Exchange，开放神经网络交换格式，用于模型表示与交换。 |
| post-training | 后训练，模型完成初始训练之后进行的进一步训练。 |
| reinforcement learning | 强化学习，简称 RL；文中重点涉及其分布式训练基础设施。 |
| MoE | Mixture of Experts，混合专家，包含多个专家组件的模型架构。 |
| PTEnv | Pinterest 的后训练生命周期支撑平台，管理数据、编排、扩缩容、评估和导出，将训练内循环交给外部框架。 |
| Ray Direct Transport | Ray 的直接传输机制，用于在 GPU 之间直接移动 PyTorch 张量。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Give us the starting point. | 先给我们讲讲最基础的概念。 |
| What sits above it? | 它的上一层是什么？ |
| That's the basic idea. | 基本思路就是这样。 |
| What did that change? | 那带来了什么变化？ |
| How does Uber's example compare? | Uber 的例子相比之下是什么情况？ |
| Can we tell how much came from Ray Data alone? | 我们能判断其中有多少收益单独来自 Ray Data 吗？ |
| What difficulties remain? | 还存在哪些困难？ |
| Let's finish with three points. | 最后用三点来回顾。 |
