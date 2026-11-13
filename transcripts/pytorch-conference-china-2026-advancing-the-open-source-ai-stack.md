# PyTorch Conference China 2026: Connecting the Open AI Stack

原文：[PyTorch Conference China 2026: Advancing the Open Source AI Stack - PyTorch](https://pytorch.org/blog/pytorch-conference-china-2026-advancing-the-open-source-ai-stack/)

## 摘要

本期围绕 PyTorch Conference China 2026，讨论模型、框架、算力和基础设施如何组成开放的 AI 技术栈。两位主持人解释了 PyTorch 基金会的新成员、异构硬件适配工作组，以及中国开发者社区的参与情况。他们还审慎解读了大会展示的 DeepSeek-R1 优化案例和开源权重模型流量数据，并指出这些案例不能代表所有工作负载的表现。

## 对话

Ava: If a training job has to move to another chip, how much of the software stack has to move with it?

Ava: 如果一项训练任务必须迁移到另一种芯片，软件栈需要跟着迁移多少？

Brian: That's the practical question behind PyTorch Conference China twenty twenty-six. The conference looked at how an open AI stack can span different hardware.

Brian：这正是 2026 年 PyTorch Conference China 背后的实际问题。大会探讨了开放式 AI 技术栈如何适配不同的硬件。

Ava: So this wasn't just a conference about new PyTorch features?

Ava: 所以这场大会讨论的不只是 PyTorch 的新功能？

Brian: Right. It brought the community together in Shanghai on September eighth and ninth, after two related events on September seventh.

Brian: 对。继九月七日的两场相关活动之后，社区在九月八日和九日齐聚上海。

Ava: And the theme was Open Source for the AI Era. What does that mean in practice?

Ava: 大会的主题是 Open Source for the AI Era。这在实践中是什么意思？

Brian: It means looking beyond the framework alone. The discussions covered models, training, inference, hardware, cloud infrastructure, and agents.

Brian: 意思是不能只看框架本身。讨论涵盖了模型、训练、推理、硬件、云基础设施和智能体。

Ava: That's a lot of ground. Is there one idea that connects it?

Ava: 涉及的范围真广。有没有一个理念把这些内容串起来？

Brian: The article calls it one open source AI stack: software and hardware layers that different communities build together.

Brian: 文章把它称为一套开源 AI 技术栈：由不同社区共同构建的软件层和硬件层。

Ava: Let's slow down. When you say stack, what sits on top of what?

Ava: 我们说慢一点。你说的技术栈，各层是怎么排列的？

Brian: At the top are applications and agents. Beneath them are tools for building and delivering intelligence, then tools for running and scaling workloads.

Brian: 最上层是应用和智能体。下面是构建和交付智能的工具，再往下是运行工作负载并将其扩展到更大规模的工具。

Ava: And below those?

Ava: 这些层下面呢？

Brian: Infrastructure and isolation, then heterogeneous compute. Heterogeneous just means different kinds of processors, including CPUs, GPUs, NPUs, and other accelerators.

Brian: 再下面是基础设施和隔离层，然后是异构计算。异构指的就是不同种类的处理器，包括 CPUs、GPUs、NPUs 和其他加速器。

Ava: So a developer working on an agent still depends on choices several layers below it.

Ava: 这么说，开发智能体的人也会受到下方好几层技术选择的影响。

Brian: Exactly. Imagine changing the engine in a vehicle. You'd want the controls above it to keep making sense.

Brian: 正是这样。想象一下给一辆车换发动机。你会希望上面的操控系统仍然能正常发挥作用。

Ava: That's the appeal. Does the article say every model already runs on every chip?

Ava: 这就是它吸引人的地方。文章有没有说每个模型现在都能在每种芯片上运行？

Brian: No. Any Model, Any Chip, Any Cloud, Any Agent is the Foundation's framing. The article also describes the adaptation work still needed.

Brian: 没有。Any Model, Any Chip, Any Cloud, Any Agent 是基金会提出的愿景。文章也描述了仍需完成的适配工作。

Ava: What projects did the Foundation put in those middle layers?

Ava: 基金会把哪些项目放在中间那些层？

Brian: It placed PyTorch, vLLM, and Ray in the building and delivering layer. vLLM comes up elsewhere in the program in connection with serving.

Brian: 它把 PyTorch、vLLM 和 Ray 放在构建与交付层。vLLM 在大会其他议程中也被提到，涉及模型服务。

Ava: And running workloads at scale?

Ava: 那大规模运行工作负载这一层呢？

Brian: That layer included Kubernetes, KServe, Kueue, OpenTelemetry, and llm-d. The article presents them as examples within the larger stack.

Brian: 这一层包括 Kubernetes、KServe、Kueue、OpenTelemetry 和 llm-d。文章把它们作为整个技术栈中的例子。

Ava: Then OpenStack and Kata Containers are below that?

Ava: 那 OpenStack 和 Kata Containers 就在更下面？

Brian: Yes, in infrastructure and isolation. The key point is that these layers meet at real workload boundaries.

Brian: 对，在基础设施和隔离层。关键是，这些层会在实际工作负载的衔接处相遇。

Ava: So who takes responsibility when a boundary doesn't work?

Ava: 那如果层与层的衔接出了问题，谁来负责？

Brian: That's where collaboration matters. The Foundation says no single organization builds the whole system; its members work across different parts of it.

Brian: 这就是协作的重要之处。基金会说，没有任何一家组织能构建整个系统；其成员在系统的不同部分开展工作。

Ava: The article also announces new members. Who joined?

Ava: 文章还宣布了新成员。有哪些机构加入？

Brian: Alibaba Cloud and Cambricon joined as Platinum Members. Ant Group joined as a Gold Member. The announcement happened in Shanghai on September eighth.

Brian: Alibaba Cloud 和 Cambricon 作为 Platinum Members 加入。Ant Group 作为 Gold Member 加入。这项公告是在九月八日于上海发布的。

Ava: Do their backgrounds show why the Foundation talks about a whole stack?

Ava: 它们各自的背景能说明基金会为什么强调完整的技术栈吗？

Brian: They do. Alibaba Cloud's work spans cloud infrastructure and the Qwen model family. Ant Group brings production AI experience at financial-services scale.

Brian: 能。Alibaba Cloud 的工作横跨云基础设施和 Qwen 模型家族。Ant Group 带来了金融服务规模的生产环境 AI 经验。

Ava: And Cambricon is the hardware side?

Ava: Cambricon 代表硬件这一侧？

Brian: Yes. It develops MLU accelerators. The article says its PyTorch contributions follow an Upstream First approach.

Brian: 对。它开发 MLU 加速器。文章说，它对 PyTorch 的贡献遵循 Upstream First 方针。

Ava: Upstream First sounds important, but what did it contribute?

Ava: Upstream First 听起来很重要，但它具体贡献了什么？

Brian: The article lists work across torch.compile, Eager Operators, Device Runtime, Distributed Computing, AMP, Dataloader, and Profiler.

Brian: 文章列出的工作涉及 torch.compile、Eager Operators、Device Runtime、Distributed Computing、AMP、Dataloader 和 Profiler。

Ava: That's more than a single device hook.

Ava: 这远不止一个设备接入接口。

Brian: Right. It touches several places a workload passes through. The article doesn't spell out the individual changes, though.

Brian: 对。这涉及工作负载运行时经过的多个环节。不过，文章没有详细说明每一项具体改动。

Ava: Before we get to hardware integration, how large is the China-based contributor community?

Ava: 在谈硬件集成之前，先说说中国的贡献者社区有多大？

Brian: The article reports more than one hundred China-based developers contributing to PyTorch across more than forty affiliated organizations.

Brian: 文章称，来自四十多个所属组织的一百多名中国开发者为 PyTorch 做出了贡献。

Ava: I also saw a figure above two hundred and fifty. Is that the same count?

Ava: 我还看到一个超过二百五十的数字。那是同一项统计吗？

Brian: No. That figure covers organizations across China contributing to PyTorch Foundation projects, including PyTorch, Ray, DeepSpeed, Safetensors, and vLLM.

Brian: 不是。那个数字统计的是中国各地为 PyTorch 基金会项目做出贡献的组织，项目包括 PyTorch、Ray、DeepSpeed、Safetensors 和 vLLM。

Ava: That's a useful distinction. One count is PyTorch developers; the other spans Foundation projects.

Ava: 这个区别很有用。一个统计的是 PyTorch 开发者，另一个涵盖了基金会的多个项目。

Brian: Exactly. Put them together carelessly, and you'd give listeners the wrong picture.

Brian: 正是这样。如果随意把两个数字放在一起，就会让听众产生误解。

Ava: Now, back to chips. What's making hardware support so difficult?

Ava: 现在回到芯片。硬件支持为什么这么难做？

Brian: The article points to hardware heterogeneity, high adaptation costs, and a lack of unified standards. Each new chip can create repeated integration work.

Brian: 文章指出了硬件的异构性、高昂的适配成本，以及缺乏统一标准。每增加一种新芯片，都可能带来重复的集成工作。

Ava: Repeated where? Just in operators?

Ava: 哪些地方会重复？只有算子吗？

Brian: Operators are one part. The September seventh discussions also named compilers, training and inference frameworks, and communications.

Brian: 算子只是其中一部分。九月七日的讨论还提到了编译器、训练和推理框架，以及通信。

Ava: So it's like adding a machine to a lab and discovering every tool needs a new connector.

Ava: 就像给实验室添一台机器，却发现每件工具都需要一个新的连接器。

Brian: Yes. That's a helpful picture. The FlagOS-hosted event discussed open system software for diverse hardware and collaboration across chip architectures.

Brian: 对，这个比喻很贴切。由 FlagOS 主办的活动讨论了适用于多样化硬件的开放系统软件，以及跨芯片架构的协作。

Ava: There was a Huawei-hosted event too, wasn't there?

Ava: 当时还有一场由 Huawei 主办的活动，对吧？

Brian: There was. It focused on PyTorch workloads across GPUs, NPUs, and XPUs, including migration, operators, compiler work, and performance tuning.

Brian: 有。活动重点讨论了 PyTorch 工作负载如何跨 GPUs、NPUs 和 XPUs 运行，包括迁移、算子、编译器工作和性能调优。

Ava: Can you make the Foundation's response more concrete?

Ava: 能把基金会的应对办法说得更具体一些吗？

Brian: The PyTorch Technical Advisory Council has an Accelerator Integration Working Group, co-chaired by Huawei and Intel.

Brian: PyTorch Technical Advisory Council 下设一个 Accelerator Integration Working Group，由 Huawei 和 Intel 共同担任主席。

Ava: What does that group actually deliver?

Ava：那个工作组具体交付什么？

Brian: It provides standardized hardware onboarding guidelines, cross-repo CI testing, generalized device-aware test suites, and platform incubation workflows.

Brian：它提供标准化的硬件接入指南、跨仓库 CI 测试、通用的设备感知测试套件，以及平台孵化流程。

Ava: Cross-repo CI means integration testing across repositories, right? Why does that matter here?

Ava：跨仓库 CI 是指跨多个代码仓库进行集成测试，对吧？这在这里为什么重要？

Brian: That's the idea. Hardware support crosses project boundaries, so a test in just one repository may miss a problem elsewhere.

Brian：就是这个意思。硬件支持涉及多个项目，所以只在一个代码仓库里测试，可能漏掉其他地方的问题。

Ava: And device-aware tests check behavior with the device in mind?

Ava：设备感知测试就是结合设备的特性来检查运行表现？

Brian: Yes. The group also described refined device-agnostic APIs and a broader multi-backend test matrix in its roadmap.

Brian：对。这个工作组还在路线图中提到了改进设备无关的 API，以及扩大多后端测试矩阵。

Ava: Device-agnostic APIs: interfaces intended to work without being tied to one device. Is that fair?

Ava：设备无关的 API，就是不绑定某一种设备、旨在通用的接口。这样理解对吗？

Brian: That's fair. Multi-backend means testing across different hardware implementations. The article presents this as ongoing work.

Brian：对。多后端是指在不同的硬件实现上进行测试。文章说这项工作仍在进行中。

Ava: Let me check the main point. The goal is to make adding hardware less repetitive and easier to test.

Ava：我确认一下重点：目标是减少添加硬件支持时的重复工作，并让测试更容易。

Brian: Yes. And hardware vendors and developers are invited to participate in that working group.

Brian：对。硬件厂商和开发者也受邀参与这个工作组。

Ava: What did the conference say about models and serving? We haven't spent much time there.

Ava：大会在模型和推理服务方面说了些什么？我们还没怎么谈到这部分。

Brian: The keynotes ranged from open-weight models to vLLM deployment, distributed acceleration, cloud infrastructure for Qwen, and agent runtimes.

Brian：主题演讲涵盖了开放权重模型、vLLM 部署、分布式加速、支持 Qwen 的云基础设施，以及智能体运行时。

Ava: Open-weight means the model weights are available. Did the article give a concrete optimization example?

Ava：开放权重是指模型权重可获取。文章给出具体的优化案例了吗？

Brian: It did. One DeepSeek-R1 case study discussed kernels, routing, parallelism, and serving on the same GB three hundred hardware six months later.

Brian：是的。一项 DeepSeek-R1 案例研究讨论了内核、路由、并行化和服务部署，以及 6 个月后在同一款 GB300 硬件上的表现。

Ava: Wait, so what changed in the result?

Ava：等等，最终结果有什么变化？

Brian: In the configuration presented, the optimized system delivered two point seven seven times the throughput and sixty percent lower token cost.

Brian：在展示的配置下，优化后的系统实现了 2.77 倍的吞吐量，token 成本降低了 60%。

Ava: Throughput means the amount of work served over time. Did they separate the effect of each optimization?

Ava：吞吐量是指单位时间内提供服务所完成的工作量。他们分别说明了每项优化的效果吗？

Brian: The article doesn't give that breakdown. It reports the combined result for the presented configuration and cites NVIDIA's twenty twenty-six material.

Brian：文章没有给出那项细分数据。它报告的是所展示配置的综合结果，并引用了 NVIDIA 2026 年的资料。

Ava: So I shouldn't treat those numbers as a promise for my own serving setup.

Ava：所以我不该把这些数字当成自己部署推理服务时也能达到的保证。

Brian: Right. They show what that case study achieved. The article doesn't claim every workload or accelerator will see the same gain.

Brian：对。这些数字说明的是那个案例取得的结果。文章没有声称每种工作负载或加速器都能获得相同的提升。

Ava: Was there another number that stood out?

Ava：还有其他值得注意的数字吗？

Brian: Yes. The article cites Mozilla for a change in OpenRouter token traffic served by open-weight models developed in China.

Brian：有。文章引用 Mozilla 的数据，谈到了中国开发的开放权重模型在 OpenRouter token 流量中所占份额的变化。

Ava: How big a change?

Ava：变化有多大？

Brian: Their share rose from two percent in late twenty twenty-four to forty-five percent in April twenty twenty-six.

Brian：他们的占比从 2024 年末的 2% 上升到 2026 年 4 月的 45%。

Ava: That's a large shift. But it's specifically OpenRouter traffic, correct?

Ava：变化很大。但这说的专门是 OpenRouter 的流量，对吗？

Brian: Correct. It isn't a measure of all AI usage. It's one traffic measure that helps explain the conference's focus on open models.

Brian：对。它不能衡量所有 AI 使用情况，只是一项流量指标，有助于解释大会为什么关注开放模型。

Ava: What about the keynote from Mark Collier? How does it fit with these examples?

Ava：Mark Collier 的主题演讲呢？它和这些案例有什么联系？

Brian: He focused on an open ecosystem where researchers, developers, model builders, and hardware companies can work together from experimentation to production.

Brian：他重点谈了一个开放生态系统，让研究人员、开发者、模型构建者和硬件公司能够从实验阶段到生产阶段协作。

Ava: That sounds broad. Did the conference connect those people in person too?

Ava：这听起来范围很广。大会也让这些人在线下交流了吗？

Brian: Yes. There were technical demonstrations, community meetings, a Foundation press conference, and conversations at the PyTorch booth.

Brian：是的。现场有技术演示、社区会议、基金会新闻发布会，以及在 PyTorch 展位进行的交流。

Ava: And Technical Advisory Council members met in Shanghai on day one.

Ava：Technical Advisory Council 的成员也在首日于上海举行了会议。

Brian: They did. Those meetings fit the same theme: people working on different layers need places to coordinate.

Brian：是的。这些会议也呼应了同一个主题：从事不同技术层面工作的人需要有地方协调合作。

Ava: Here's the catch. A conference article can show direction, but it can't answer every implementation question.

Ava：不过有一点需要注意。大会文章能展示方向，却无法回答每一个实现层面的问题。

Brian: Exactly. It names projects and work areas, but gives limited detail on individual integrations, test outcomes, and benchmark methods.

Brian：没错。文章列出了项目和工作领域，但对各项集成的细节、测试结果和基准测试方法介绍有限。

Ava: What should an engineer watch next?

Ava：工程师接下来应该关注什么？

Brian: The Accelerator Integration Working Group's device-agnostic APIs and expanded multi-backend testing are concrete areas to follow.

Brian：Accelerator Integration Working Group 的设备无关 API 和扩大的多后端测试，是值得关注的具体工作。

Ava: And the article points to another conference?

Ava：文章还提到了另一场大会？

Brian: PyTorch Conference North America is scheduled for San Jose on October twentieth and twenty-first, with maintainers from PyTorch, vLLM, DeepSpeed, and Ray.

Brian：PyTorch Conference North America 定于 10 月 20 日和 21 日在圣何塞举行，PyTorch、vLLM、DeepSpeed 和 Ray 的维护者将出席。

Ava: Let's finish with three things to remember. First?

Ava：最后说三件值得记住的事。第一件是什么？

Brian: First, the open AI stack connects applications all the way down to diverse hardware.

Brian：首先，开放 AI 技术栈把应用一路连接到各种不同的硬件。

Ava: Second, hardware integration needs shared guidelines and tests. Third, promising numbers need their original context.

Ava：其次，硬件集成需要共享的指南和测试。第三，看起来很亮眼的数字也需要结合其原始背景来理解。

Brian: That's it. Thanks for listening, and we'll see you next time.

Brian：就是这样。谢谢收听，我们下次见。

## 术语

| Term | 释义 |
|---|---|
| open source AI stack | 由社区共同建设、涵盖应用到硬件多个层次的开源 AI 技术栈 |
| heterogeneous compute | 由 CPU、GPU、NPU 等不同类型处理器构成的计算环境 |
| accelerator integration | 将 AI 加速器接入 PyTorch 及相关软件体系的工作 |
| hardware onboarding | 让新硬件平台进入并适配软件生态的过程 |
| cross-repo CI | 跨代码仓库的持续集成测试 |
| device-aware test suites | 考虑不同设备特性的成套测试 |
| device-agnostic APIs | 尽量不绑定某一种设备的应用程序接口 |
| multi-backend test matrix | 覆盖多种硬件后端的测试组合 |
| open-weight models | 权重开放的模型 |
| throughput | 单位时间内处理的工作量，即吞吐量 |
| token cost | 处理每个 token 的成本 |
| Upstream First | 优先向上游开源项目贡献的方式 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What does that mean in practice? | 实际做起来是什么意思？ |
| Let's slow down. | 我们放慢一点，先理清楚。 |
| Can you make the Foundation's response more concrete? | 能把基金会的应对方式讲得更具体吗？ |
| That's a useful distinction. | 这个区分很有用。 |
| Let me check the main point. | 我确认一下核心意思。 |
| Wait, so what changed in the result? | 等等，结果具体发生了什么变化？ |
| Here's the catch. | 但这里有个需要注意的地方。 |
| What should an engineer watch next? | 工程师接下来该关注什么？ |

## 中文内容总结

PyTorch 中国大会于 2026 年 9 月 8 日至 9 日在上海举行，与 KubeCon + CloudNativeCon 和 OpenInfra Summit 同期举办；9 月 7 日还有配套技术活动。大会围绕模型、框架、分布式训练、推理、硬件、云原生基础设施和智能体，讨论开放软件如何支持不同芯片上的 AI 工作负载。

### 技术栈与基金会成员

PyTorch 基金会将开放 AI 技术栈描述为多个协作层次：应用与智能体之下，PyTorch、vLLM 和 Ray 支持模型开发与交付；Kubernetes、KServe、Kueue、OpenTelemetry 和 llm-d 支持工作负载运行与扩展；OpenStack 和 Kata Containers 提供基础设施与隔离。这些软件运行在 NPU、CPU、GPU 等计算设备上。9 月 8 日，阿里云和寒武纪以白金会员、蚂蚁集团以金牌会员身份加入基金会。

### 中国社区与大会案例

大会资料称，超过 100 名中国开发者通过 40 多家所属机构为 PyTorch 贡献代码；中国还有超过 250 家机构参与 PyTorch 基金会旗下项目。大会援引 NVIDIA 2026 年的一项 DeepSeek-R1 案例：在同一 GB300 硬件上，经过六个月的内核、路由、并行和服务优化，所展示配置的吞吐量达到原来的 2.77 倍，token 成本降低 60%。另一项援引 Mozilla 的数据称，中国开发的开放权重模型在 OpenRouter token 流量中的占比，从 2024 年末的 2% 升至 2026 年 4 月的 45%。前一组性能数字仅对应所展示的配置，文章没有提供跨硬件的普遍结论。

### 异构硬件协作

9 月 7 日，FlagOS 主办的讨论聚焦不同 AI 芯片带来的重复适配工作，涉及算子、编译器、训练与推理框架及通信；华为主办的活动则讨论 GPU、NPU、XPU 上的 PyTorch 迁移和调优。由华为与英特尔共同主持的 PyTorch TAC 加速器集成工作组，推进硬件接入指南、跨仓库 CI 测试、设备感知测试和平台孵化流程。大会还介绍了设备无关 API 与多后端测试矩阵的后续计划。

### 后续活动

大会期间，技术顾问委员会成员和社区开发者也在上海交流。文章另预告，PyTorch 北美大会定于 2026 年 10 月 20 日至 21 日在圣何塞举行。
