# Ray Across the PyTorch Stack: Data, Training, Serving, and RL

原文：[A Ray-Focused Guide to PyTorch Conference North America](https://pytorch.org/blog/a-ray-focused-guide-to-pytorch-conference-north-america/)

## 摘要

本文以 PyTorch Conference North America 2026 的 Ray 相关议程为线索，介绍 Ray 如何贯穿数据处理、分布式训练、部署和后训练。对话重点讨论 LinkedIn 的三层训练架构、Uber Eats 的数据与训练改造，以及 Pinterest 将平台职责与训练内循环分离的设计。SkyRL 和 Ray Core 的议题进一步展示了强化学习对异步执行、资源调度和权重传输的需求。节目也强调，这是一篇会议导览，文中的性能数据属于特定案例，文章未提供足以复现或全面比较的实验细节。

## 对话

Ava: Your model needs more machines, but suddenly you're debugging data loading, storage, and scheduling. When did training become an infrastructure project?

Ava: 你的模型需要更多机器，可你突然开始排查数据加载、存储和调度的问题。训练是什么时候变成基础设施项目的？

Brian: That's why this conference guide matters. It follows Ray across those problems, from preparing data to training and production. The interesting question is how those pieces fit together.

Brian: 这就是这份会议指南有价值的原因。它沿着这些问题介绍 Ray，从准备数据一直讲到训练和生产环境。值得探讨的是，这些环节如何衔接起来。

Ava: I'm Ava, a software engineer outside the PyTorch core team. Brian, you're our PyTorch guide. Give us the starting point: what's Ray?

Ava: 我是 Ava，一名不在 PyTorch 核心团队的软件工程师。Brian，你是我们的 PyTorch 向导。先从基础讲起：Ray 是什么？

Brian: Ray is a distributed compute framework. It helps teams scale AI frameworks, libraries, and models across machines. The article describes a consistent developer surface across the AI lifecycle.

Brian: Ray 是一个分布式计算框架。它帮助团队把 AI 框架、库和模型扩展到多台机器上。文章说，它在整个 AI 生命周期中为开发者提供一致的操作方式。

Ava: So it covers the journey from getting data ready to running a model for users?

Ava: 所以，从准备数据到把模型提供给用户使用，整个过程它都涵盖？

Brian: Right. That includes curating multimodal data, meaning data with different types of content, training across multiple nodes, and deploying on virtual machines or clusters managed by Kubernetes.

Brian: 对。包括整理多模态数据，也就是包含不同类型内容的数据；在多个节点上训练；以及在虚拟机或 Kubernetes 管理的集群上部署。

Ava: And this is a guide to upcoming talks, so we're getting previews?

Ava: 这是一份即将举行的演讲指南，所以我们看到的是内容预告？

Brian: Exactly. It's a Ray-focused guide to PyTorch Conference North America, twenty twenty-six, in San Jose. It gives us engineering examples, but it isn't a detailed benchmark report.

Brian：没错。这是一份聚焦 Ray 的 PyTorch Conference North America 2026 指南，会议在圣何塞举行。它提供了一些工程实例，但不是详细的基准测试报告。

Ava: The first theme is Ray and Kubernetes evolving together. Why does that deserve attention?

Ava: 第一个主题是 Ray 和 Kubernetes 共同演进。为什么这值得关注？

Brian: The article describes two communities users depend on. The PyTorch Foundation includes PyTorch and Ray. The Cloud Native Computing Foundation, or CNCF, includes Kubernetes and other infrastructure projects.

Brian: 文章介绍了用户依赖的两个社区。PyTorch Foundation 包括 PyTorch 和 Ray。Cloud Native Computing Foundation，简称 CNCF，包括 Kubernetes 和其他基础设施项目。

Ava: Separate communities, but the engineer building a system needs both. The organization chart doesn't make the integration work disappear.

Ava: 社区虽然不同，但构建系统的工程师两边都需要。组织架构图不会让集成工作凭空消失。

Brian: Exactly. The talks explore a unified open source AI stack. Google, Anyscale, and the broader community are collaborating to connect those ecosystems and prevent fragmentation.

Brian: 没错。这些演讲会探讨一个统一的开源 AI 技术栈。Google、Anyscale 和更广泛的社区正在合作，连接这两个生态系统，避免碎片化。

Ava: Okay, that's the motivation. Let's get closer to the training job. What's LinkedIn building?

Ava: 好，动机清楚了。我们再靠近训练任务本身。LinkedIn 在构建什么？

Brian: A three-layer Ray-based training stack. First, a data loading library built with Ray actors, which are workers that can keep state. It streams Avro, Parquet, and Iceberg data.

Brian：这是一个基于 Ray 的三层训练技术栈。第一层是用 Ray actors 构建的数据加载库；这些 actor 是能够保持状态的工作进程。它以流式方式处理 Avro、Parquet 和 Iceberg 数据。

Ava: That's the data layer. What sits above it?

Ava: 这是数据层。它上面是什么？

Brian: Ray Train, with Fully Sharded Data Parallel, or FSDP, and Hybrid Sharded Data Parallel, or HSDP. Those are distributed training strategies. The layer also supports elastic scaling and asynchronous checkpointing.

Brian: 是 Ray Train，配合 Fully Sharded Data Parallel（FSDP）和 Hybrid Sharded Data Parallel（HSDP）。它们都是分布式训练策略。这一层还支持弹性扩缩容和异步保存检查点。

Ava: Elastic scaling means adjusting resources, and asynchronous checkpointing means saving training state without making everything wait for the whole save?

Ava: 弹性扩缩容是指调整资源；异步保存检查点是指保存训练状态时，不用让所有工作都等着整个保存过程完成？

Brian: That's the basic idea. The third layer provisions clusters on demand. But the central lesson here is separating data loading from the machines doing the GPU work.

Brian: 基本就是这个意思。third layer 负责按需创建集群。但这里的核心经验是，把数据加载与执行 GPU 工作的机器分开。

Ava: GPU meaning graphics processing unit. So where does the loading happen?

Ava: GPU 指图形处理器。那么数据加载在哪里进行？

Brian: On dedicated CPU nodes, using central processing units. Data travels through a zero-copy bridge, meaning the transfer avoids an extra data copy. Think of a separate preparation area feeding the kitchen.

Brian：这些工作在专用 CPU 节点上运行，使用的是中央处理器。数据通过零拷贝桥接机制传输，也就是说，传输过程中避免了一次额外的数据复制。可以把它想成一个独立的备料区，持续给厨房供料。

Ava: Let the cooks cook. What did that change?

Ava: 让厨师专心做菜。那么这样做带来了什么变化？

Brian: The guide reports a fifty to seventy percent reduction in dataloader memory. It also mentions a situation where LinkedIn had to work around Ray to reclaim twenty-four percent of step time.

Brian：指南报告称，数据加载器的内存占用降低了 50% 至 70%。它还提到一种情况：LinkedIn 必须绕开 Ray，才能挽回 24% 的单步训练时间。

Ava: Wait, so a Ray success story includes working around Ray?

Ava: 等等，所以这个 Ray 的成功案例里还包括绕过 Ray？

Brian: Yes, and that's useful. An infrastructure choice can help overall while still creating a specific problem. The article doesn't explain that workaround, so we can't reconstruct it.

Brian: 是的，这一点很有启发。选用某种基础设施，整体上可能有帮助，同时仍会带来某个具体问题。文章没有解释那种绕过方法，所以我们无法还原其具体做法。

Ava: The takeaway is to separate the work, then measure what still slows it down. How does Uber's example compare?

Ava: 要点就是把不同工作分开，再测量还有什么在拖慢速度。Uber 的例子相比之下怎么样？

Brian: Uber Eats is migrating its recommendation stack from TensorFlow and Horovod to PyTorch. Ray Data replaces legacy Spark-based preprocessing with distributed feature-statistics computation and zero-copy batch transformations.

Brian：Uber Eats 正在把推荐系统技术栈从 TensorFlow 和 Horovod 迁移到 PyTorch。Ray Data 取代了原先基于 Spark 的预处理方式，用于分布式计算特征统计量和进行零拷贝批量转换。

Ava: Feature statistics being information calculated about the model's input features. What results does the guide report?

Ava: 特征统计就是针对模型输入特征计算出的信息。指南报告了什么结果？

Brian: Data transformation is five times faster and uses ninety percent less memory. Training throughput improves twenty times. Those are reported results for Uber's redesign.

Brian：数据转换速度提高了 5 倍，内存用量减少了 90%。训练吞吐量提高了 20 倍。这些是 Uber 重新设计系统后报告的结果。

Ava: Twenty times is a big number. Can we tell how much came from Ray Data alone?

Ava：20 倍是个很大的数字。我们能判断其中有多少提升单独来自 Ray Data 吗？

Brian: No. The article doesn't isolate each change's contribution or provide the full measurement setup. It's an impressive case to investigate, but it doesn't predict another team's improvement.

Brian: 不能。文章没有单独说明每项改动的贡献，也没有给出完整的测量设置。这是一个值得进一步研究的出色案例，但不能据此预测其他团队能获得多大提升。

Ava: We've covered loading and transformation. What if the data simply doesn't arrive from storage fast enough?

Ava: 我们已经谈了加载和转换。如果数据从存储系统传来的速度就是不够快呢？

Brian: That's the storage talk's focus. It describes GPUs waiting on legacy REST-based access, a web request style. Rapid Storage proposes a high-throughput protocol based on gRPC, a remote procedure call framework.

Brian: 这正是存储主题演讲关注的问题。它描述了 GPU 因旧式的 REST-based 访问方式而等待数据；REST 是一种 Web 请求方式。Rapid Storage 提出一种基于 gRPC 的高吞吐量协议；gRPC 是一个远程过程调用框架。

Ava: How does that reach PyTorch?

Ava: 它怎么与 PyTorch 对接？

Brian: Through fsspec, a filesystem interface layer. The team says the benefits extend across its ecosystem, including Ray. For a storage-bound pipeline, storage is the delivery truck that keeps arriving late.

Brian: 通过 fsspec，也就是一个文件系统接口层。团队表示，其带来的好处会扩展到 fsspec 的整个生态系统，包括 Ray。对于受存储速度限制的流水线，存储系统就像一辆总是迟到的送货车。

Ava: And a faster kitchen still waits for that truck. There's also a climate model talk in the serving section. What's Ray doing there?

Ava: 厨房再快，也得等那辆车。服务部分还有一个气候模型的演讲。Ray 在那里做什么？

Brian: Ray helps scale training for a physics-constrained generative model used in climate downscaling. The talk follows that model into production, including export to ONNX, the Open Neural Network Exchange format.

Brian: Ray 帮助扩展一个用于气候降尺度的物理约束生成模型的训练。演讲还会讲这个模型如何进入生产环境，包括导出为 ONNX，也就是 Open Neural Network Exchange 格式。

Ava: Does the guide say Ray handles the entire serving path?

Ava: 指南说 Ray 负责整个服务流程吗？

Brian: It places Ray in the training recipe. The serving path includes ONNX Runtime, Temporal, and a vLLM agent. It also covers keeping the sampling process outside the exported graph and enforcing physics constraints during inference.

Brian: 它把 Ray 放在训练方案中。服务流程包括 ONNX Runtime、Temporal 和一个 vLLM agent。演讲还会介绍如何让采样过程留在导出的计算图之外，以及如何在推理时施加物理约束。

Ava: So we should pay attention to each tool's actual role. Now, why does post-training get so much space?

Ava: 所以我们得留意每种工具实际负责什么。现在，为什么后训练占了这么大篇幅？

Brian: Post-training means further training after a model's initial training. At Pinterest, teams adopted different frameworks, data formats, and distributed strategies. They spent weeks connecting open source frameworks to internal infrastructure before training even started.

Brian: 后训练指模型完成初始训练后继续进行的训练。在 Pinterest，各团队采用了不同的框架、数据格式和分布式策略。他们甚至在开始训练之前，就要花上数周把开源框架接入内部基础设施。

Ava: Weeks before the first step. That's a painful progress bar.

Ava: 第一步训练还没开始，就已经过去了好几个星期。这进度条看着真难受。

Brian: Their earlier platform owned the PyTorch training loop and reached ninety-five percent adoption. But post-training loops lived inside fast-moving external frameworks. Keeping platform ownership of that loop became a problem.

Brian：他们之前的平台负责 PyTorch 训练循环，采用率达到了 95%。但后训练循环是在快速迭代的外部框架中运行的。平台要继续掌控这个循环就成了一个问题。

Ava: What's their new boundary?

Ava: 他们现在怎么划分职责？

Brian: PTEnv is a lifecycle harness, a support structure around training. It owns data, orchestration, scaling, evaluation, and export. The open source framework owns the inner training loop.

Brian: PTEnv 是一个生命周期管理框架，也就是围绕训练提供支持的结构。它负责数据、编排、扩缩容、评估和导出。开源框架负责内部的训练循环。

Ava: Like managing the theater while each production brings its own performance?

Ava: 就像管理剧院，而每个剧组带来自己的演出？

Brian: Yes. One interface supports separate-process execution with Ray for reinforcement learning, or RL. It also supports in-process execution with MS-Swift for supervised fine-tuning, meaning training on supplied examples. Framework selection becomes a configuration choice.

Brian: 对。同一个接口支持通过 Ray 以独立进程运行 reinforcement learning（RL），也支持通过 MS-Swift 在进程内运行 supervised fine-tuning，也就是用给定样例进行训练。选择框架变成了一项配置。

Ava: That sounds flexible. What difficulties remain?

Ava: 听起来很灵活。还有哪些难题？

Brian: The talk includes bottlenecks in synchronizing model weights and workarounds for mixture-of-experts models, or MoE models, which contain multiple expert components. A shared interface doesn't remove those engineering problems.

Brian: 演讲会谈到同步模型权重时的瓶颈，以及针对 mixture-of-experts models（MoE models）的变通办法；这类模型包含多个专家组件。共用一个接口并不会消除这些工程问题。

Ava: And SkyRL tackles reinforcement learning itself?

Ava: SkyRL 则是直接处理强化学习本身？

Brian: Right. It's a modular RL library. It separates training, inference, and environments, with Ray coordinating them. The motivation includes agents handling longer tasks and interactions with multiple turns.

Brian: 对。它是一个模块化的 RL 库，把训练、推理和环境分开，由 Ray 负责协调。开发它的动机之一，是让智能体能够处理更长的任务和多轮交互。

Ava: What scale does the session discuss?

Ava：这场会议讨论的是多大的规模？

Brian: Fully asynchronous RL training on mixture-of-experts models with more than three hundred fifty billion parameters, using Megatron and vLLM. It also covers a shared Tinker Engine for researchers using their own hardware.

Brian：还有基于 Megatron 和 vLLM、针对参数量超过 3500 亿的混合专家模型开展的全异步强化学习训练。指南也介绍了供使用自有硬件的研究人员使用的共享 Tinker Engine。

Ava: Underneath that, what does Ray Core need to do?

Ava：在这背后，Ray Core 需要做什么？

Brian: Schedule trainers and inference engines, then support moving fresh weights between them. Its demo discusses scheduling across ten thousand nodes and Ray Direct Transport, which moves PyTorch tensors directly between GPUs.

Brian：调度训练器和推理引擎，然后支持在两者之间传输最新权重。演示中讨论了跨 1 万个节点进行调度，以及 Ray Direct Transport；后者能在 GPU 之间直接传输 PyTorch 张量。

Ava: Tensors being the arrays holding model data. Does scheduling at that scale establish end-to-end training performance?

Ava：张量就是存放模型数据的数组。在这种规模上实现调度，能证明端到端的训练性能吗？

Brian: The guide doesn't establish that connection. It presents scheduling scale and transport improvements as separate achievements. The demo also promises a future roadmap, but the article doesn't give its details.

Brian：这份指南没有证实两者之间的联系。它将调度规模和传输方面的改进作为两项独立成果来介绍。演示还提到了未来的路线图，但文章没有给出具体内容。

Ava: Let's finish with three points. First, Ray spans the AI lifecycle, and its integration with Kubernetes is a major theme.

Ava：最后总结三点。第一，Ray 贯穿 AI 生命周期，而它与 Kubernetes 的集成是一个重要主题。

Brian: Second, data preparation, storage, and weight movement deserve attention alongside model computation. The talks show concrete gains, plus bottlenecks that still need work.

Brian：第二，除了模型计算，数据准备、存储和权重传输也值得关注。这些演讲展示了具体的性能提升，也指出了仍需解决的瓶颈。

Ava: Third, flexible platform boundaries matter as training frameworks change. Treat these conference results as specific cases, and bring questions about their measurement conditions.

Ava：第三，随着训练框架不断变化，灵活的平台边界很重要。应把这些会议成果视为具体案例，并追问其测量条件。

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

## 中文内容总结

这篇活动导览面向使用 PyTorch 构建 AI 基础设施的开发者，介绍 2026 年 10 月 20 日至 21 日在圣何塞举行的 PyTorch 北美大会中与 Ray 相关的报告和交流活动。议程覆盖分布式计算、数据处理、模型训练、部署及后训练。

### 开源计算栈

10 月 20 日，Ray 联合创始人 Ion Stoica 将介绍 Ray 从早期支持强化学习，到编排当前强化学习框架的演进。同日另一场报告将讨论 Kubernetes 与 Ray 如何协同，让云原生基础设施与 PyTorch 生态中的项目共同支撑 AI 应用。

### 数据处理与训练

LinkedIn 将展示三层 Ray 训练栈：在专用 CPU 节点上读取 Avro、Parquet 和 Iceberg 数据，以 Ray Train 结合 FSDP、HSDP、弹性扩缩容和异步检查点，再按需提供集群。议程称，其数据加载器内存用量降低了 50% 至 70%，团队还通过绕开 Ray 的一处开销追回了 24% 的单步训练时间。10 月 21 日，Uber 将介绍把 Eats 推荐系统从 TensorFlow/Horovod 迁移到 PyTorch，并以 Ray Data 取代基于 Spark 的预处理。议程报告，这次改造使数据转换速度提高 5 倍、内存用量降低 90%，训练吞吐量提高 20 倍。

### 存储与模型部署

一场报告将介绍通过 fsspec 接入基于 gRPC 的 Rapid Storage，以缓解 GPU 等待数据的问题，并讨论其对 Ray 等工具的适用性。另一场报告以气候降尺度模型 PC-RF 为例，讲解 ONNX 导出、图外采样和推理时的物理约束。Ray 用于该模型的扩展训练，部署流程则涉及 ONNX Runtime、Temporal 和 vLLM 智能体。

### 后训练与强化学习

Pinterest 将介绍 PTEnv：平台负责数据、编排、扩展、评估和导出，把训练循环交给不同的开源框架；报告会涉及使用 Ray 执行强化学习任务，以及视觉语言模型训练中的权重同步问题。SkyRL 报告将介绍由 Ray 编排的训练、推理与环境模块，并展示面向超过 3500 亿参数 MoE 模型的全异步强化学习、让研究者用自有硬件训练的多租户 Tinker Engine，以及基于 HTTP 的推理接口。

### 演示与交流

10 月 21 日的演示将用后训练流程说明 Ray Core 的调度与权重传输，并介绍议程所称的万节点集群调度能力及 GPU 间直接传输 PyTorch 张量的 Ray Direct Transport。10 月 20 日下午还有与 Ray 开发者的小组交流，供参会者提问。
