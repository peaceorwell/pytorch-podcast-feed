# Nebius Joins the PyTorch Foundation: What Engineers Should Watch

原文：[Nebius Joins the PyTorch Foundation as a Platinum Member - PyTorch](https://pytorch.org/blog/nebius-joins-the-pytorch-foundation-as-a-platinum-member/)

## 摘要

Nebius 以白金会员身份加入 PyTorch 基金会，获得理事会的一个席位，Ryan Hanrui Wang 也将代表公司参加技术咨询委员会。文章介绍了 Nebius AI Cloud 上的 PyTorch 训练与微调，以及 Nebius 为 vLLM 开发、测试和基准测试提供算力的既有工作。Nebius 计划为最新硬件平台上的 PyTorch 和 vLLM 性能作出贡献，并分享生产规模推理与后训练的经验。对工程师而言，这篇公告说明了已有合作和未来方向；具体贡献仍需结合后续发布的工作来评估。

## 对话

Ava: A team can train a model with PyTorch and still face a lot of work before serving it. That's why this membership announcement caught my attention. Nebius has joined the PyTorch Foundation as a Platinum member, and its work touches both training and inference.

Ava：团队即使用 PyTorch 训练好了模型，在把模型投入服务之前，仍有许多工作要做。这就是这则会员公告引起我注意的原因。Nebius 已成为 PyTorch 基金会的白金会员，其工作涉及训练和推理两个方面。

Brian: The useful question is what Nebius already does and what it plans to do through the Foundation. Startups and enterprises use PyTorch to train and fine-tune models at scale on Nebius AI Cloud. Nebius also supports work on vLLM, a Foundation project concerned with inference.

Brian：值得问的是，Nebius 已经在做什么，又计划通过基金会做什么。创业公司和企业在 Nebius AI Cloud 上使用 PyTorch 大规模训练和微调模型。Nebius 还支持 vLLM 相关工作；vLLM 是基金会旗下一个聚焦推理的项目。

Ava: Let's make that concrete. Imagine a team fine-tuning a model for a product, then preparing to serve it. They'd have questions at more than one stage. The announcement connects Nebius to that broad workflow, though it doesn't say every customer uses every part of its platform.

Ava：我们说得具体一些。设想一个团队为某款产品微调模型，然后准备将其投入服务。他们会在不止一个环节遇到问题。这则公告将 Nebius 与这套较广泛的工作流程联系起来，但并没有说每位客户都会使用其平台的每个部分。

Brian: Nebius describes its AI cloud platform as covering data and model training through production deployment. That's the company's scope. The more specific statement is that customers train and fine-tune with PyTorch on its cloud. Its inference and post-training contributions are another part of the story.

Brian：Nebius 称其 AI 云平台的覆盖范围从数据和模型训练延伸到生产部署。这是公司对自身业务范围的描述。更具体的说法是，客户在其云上使用 PyTorch 训练和微调模型。它在推理和后训练方面的贡献则是另一部分。

Ava: I want to understand why the Foundation is the setting for this announcement. Nebius runs a cloud, but the software involved reaches well beyond one cloud. The Foundation calls itself vendor-neutral. How does that shape the role of a Platinum member?

Ava：我想弄清楚为什么这则公告是在基金会的背景下发布的。Nebius 运营着云平台，但相关软件的应用范围远不止一个云平台。基金会称自己保持厂商中立。这会如何影响白金会员的角色？

Brian: The PyTorch Foundation is a community-driven hub hosted by the Linux Foundation. It supports the core PyTorch framework and projects including vLLM, DeepSpeed, Ray, Helion, and Safetensors. Its stated scope covers training, optimizing, serving, and running models across chips and clouds.

Brian：PyTorch 基金会是由 Linux 基金会托管、由社区驱动的协作中心。它支持 PyTorch 核心框架，以及 vLLM、DeepSpeed、Ray、Helion 和 Safetensors 等项目。其声明的工作范围涵盖在不同芯片和云平台上训练、优化、提供服务和运行模型。

Ava: So membership puts Nebius in a community that works on several parts of our imagined team's path. That doesn't turn a membership into a software feature. I find the existing work more useful for understanding what Nebius brings to the table.

Ava：因此，成为会员让 Nebius 加入了一个在我们设想的团队所经流程的多个环节开展工作的社区。但会员身份本身不会变成一项软件功能。对我来说，Nebius 已开展的工作更有助于理解它能带来什么。

Brian: One example is vLLM. Nebius has long provided compute for its development, testing, and benchmarking. The announcement includes multi-node work on large models such as DeepSeek R1. Providing compute for those activities is a concrete form of support; it isn't itself a benchmark result.

Brian：vLLM 就是一个例子。Nebius 长期为其开发、测试和基准测试提供算力。公告还提到了针对 DeepSeek R1 等大型模型开展的多节点工作。为这些活动提供算力是一种具体的支持，但这本身不是基准测试结果。

Ava: That distinction matters. If I hear that a company helped with benchmarking, I might expect a speed comparison. But the statement we can use here establishes the support and the kind of work involved. It doesn't give our imagined team a number for its own workload.

Ava：这个区别很重要。如果我听说一家公司协助开展了基准测试，可能会期待看到速度对比。但这里的声明能确定的是它提供了支持，以及所涉及的工作类型。它没有给我们设想的团队提供适用于其自身工作负载的数值。

Brian: There's work beyond supplying compute. Nebius's Applied AI team contributes inference and post-training optimizations to open-source projects. The announcement also says Nebius's open-source work has been incorporated into the PyTorch Ecosystem. Those are existing activities, distinct from the performance work Nebius now plans.

Brian：Nebius 的工作不止于提供算力。它的 Applied AI 团队向开源项目贡献推理和后训练优化。公告还称，Nebius 的开源工作已被纳入 PyTorch 生态系统。这些都是已经在开展的活动，有别于 Nebius 现在计划进行的性能方面的工作。

Ava: For a framework engineer, that distinction is practical. Compute support can help a project develop and test. An optimization contribution can change software people use. Here, we know the contribution areas, so we can discuss the engineering direction without assigning a particular mechanism to the work.

Ava：对框架工程师来说，这个区别有实际意义。算力支持可以帮助项目开发和测试。优化方面的贡献则可能改变人们使用的软件。这里，我们知道贡献涉及哪些领域，因此可以讨论其工程方向，而不必断定这些工作采用了某种具体机制。

Brian: It gives us a starting point. Nebius has experience with PyTorch training workloads, vLLM development support, and open-source work around inference and post-training. Those activities make its plans easier to place. They don't mean every stage of a customer's workflow uses the same project.

Brian：这些信息给了我们一个起点。Nebius 有支持 PyTorch 训练工作负载、支持 vLLM 开发，以及参与推理和后训练相关开源工作的经验。这些活动有助于我们理解它的计划处于什么位置，但并不意味着客户工作流程的每个阶段都使用同一个项目。

Ava: The new plan is about performance on the latest hardware platforms. I hear two project names in that plan: PyTorch and vLLM. That seems relevant to teams working from model training toward serving, but the two projects would give an engineer different things to examine.

Ava：新计划关注的是最新硬件平台上的性能。我在计划中听到了两个项目名称：PyTorch 和 vLLM。这似乎与从模型训练走向服务部署的团队有关，不过工程师需要分别考察这两个项目的不同方面。

Brian: Yes. Nebius plans to contribute to PyTorch and vLLM performance on those platforms. It also plans to share what it learns from running inference and post-training at production scale. That's a statement of intended work. A future contribution would let engineers examine the actual change.

Brian：没错。Nebius 计划为 PyTorch 和 vLLM 在这些平台上的性能作出贡献。它还计划分享大规模生产环境中的推理和后训练经验。这说的是计划开展的工作。等未来的贡献出现后，工程师才能查看实际改动。

Ava: The production experience interests me. A team running inference repeatedly can encounter problems that weren't obvious during model development. If Nebius shares what it learns, that could help frame work in an open-source project. I'm treating that as a possibility, not an announced result.

Ava：我对生产环境中的经验很感兴趣。一个反复运行推理的团队可能遇到模型开发期间不明显的问题。如果 Nebius 分享这些经验，就可能有助于确定开源项目中相关工作的方向。我把这看作一种可能性，而不是已经宣布的成果。

Brian: That's a fair reading. The announcement names production-scale inference and post-training as sources of lessons to share. It also names performance as a contribution area. For our imagined team, the next question would be which project receives a change and what problem that change addresses.

Brian：这样理解很合理。公告指出，大规模生产环境中的推理和后训练是它计划分享经验的来源，也将性能列为贡献领域。对于我们设想的团队，接下来要问的是哪个项目会收到改动，以及这个改动要解决什么问题。

Ava: Before we look ahead, there's a second kind of contribution in this membership: governance. A cloud provider can provide compute and code, while a Foundation member can also take part in setting direction. What formal role does Platinum membership give Nebius?

Ava：在展望未来之前，这次入会还有另一种贡献形式：治理。云服务商可以提供算力和代码，而基金会会员也可以参与确定方向。白金会员身份赋予 Nebius 什么正式角色？

Brian: Nebius gets 1 seat on the PyTorch Foundation Governing Board. The Board sets policy through the Foundation's bylaws, mission, and vision statements. Those describe the broad scope of Foundation initiatives, its technical vision, and its direction. That's a broad policy role.

Brian：Nebius 在 PyTorch 基金会理事会获得 1 个席位。理事会依据基金会的章程、使命宣言和愿景声明制定政策。这些文件描述了基金会各项举措的总体范围、技术愿景和发展方向。这是一项涉及总体政策的职责。

Ava: It's useful to keep that role in proportion. A board seat tells us Nebius can participate in Foundation governance. By itself, it doesn't tell us what a future PyTorch or vLLM contribution will contain. There's also a Technical Advisory Council role in the announcement.

Ava：客观看待这个角色很有帮助。理事会席位表明 Nebius 可以参与基金会治理。但仅凭这一点，我们不知道它未来会为 PyTorch 或 vLLM 贡献什么内容。公告中还提到了技术咨询委员会的角色。

Brian: Ryan Hanrui Wang will join the board and represent Nebius on the Technical Advisory Council, or TAC. At Nebius, he leads inference, post-training, and agentic systems. Those responsibilities connect directly to the production experience the company says it plans to share.

Brian：Ryan Hanrui Wang 将加入理事会，并在技术咨询委员会（TAC）代表 Nebius。在 Nebius，他负责推理、后训练和智能体系统。这些职责与公司计划分享的生产环境经验直接相关。

Ava: His technical background adds context. The announcement says he created the SpAtten sparse-attention framework and Hardware-Aware Transformers at MIT. That work sounds relevant to model systems, but it would be a leap to predict a specific project change from his earlier research.

Ava：他的技术背景也提供了更多背景信息。公告称，他在 MIT 创建了 SpAtten 稀疏注意力框架和 Hardware-Aware Transformers。这些工作听起来与模型系统有关，但如果根据他早期的研究预测某个项目会发生什么具体改动，就有些推断过度了。

Brian: He also co-founded Eigen AI, which Nebius acquired in 2026. The announcement says his open-source work has passed 1 million downloads and is part of the PyTorch Ecosystem. Those details explain his connection to the community; the download count isn't a measure of performance.

Brian：他还共同创立了 Eigen AI，Nebius 于 2026 年收购了这家公司。公告称，他的开源作品下载量已超过 100 万次，并且是 PyTorch 生态系统的一部分。这些信息说明了他与社区的联系；下载量并不是性能指标。

Ava: So we have people, governance, and existing technical activity in one announcement. Ryan Hanrui Wang says Nebius wants deeper collaboration and an easier path from experimentation to production. I read that as the company's aim, rather than a claim about any particular customer's transition.

Ava：这样一来，一则公告中就同时包含了人员、治理和已有的技术活动。Ryan Hanrui Wang 表示，Nebius 希望深化合作，让从实验走向生产的过程更顺畅。我将其理解为公司的目标，而不是对某位具体客户如何完成这一转变的断言。

Brian: Mark Collier, the Foundation's Executive Director, frames the collaboration around scaling challenges. He points to PyTorch and vLLM, high-throughput inference performance, and large-scale model training. That explains why the Foundation welcomes Nebius's participation across both training and serving work.

Brian：基金会执行董事 Mark Collier 从规模化挑战的角度介绍了这项合作。他提到了 PyTorch 和 vLLM、高吞吐量推理性能，以及大规模模型训练。这解释了基金会为何欢迎 Nebius 同时参与训练和服务部署方面的工作。

Ava: For our imagined team, the announcement is useful as a map of activity. They can see where Nebius already provides cloud capacity or contributes to open source, and where it intends to work next. It isn't enough on its own to choose a serving setup.

Ava：对于我们设想的团队，这则公告可以当作一张了解相关活动的地图。他们可以看到 Nebius 已经在哪些方面提供云端算力或为开源项目作出贡献，以及它接下来打算在哪些方面开展工作。但仅凭这则公告，还不足以选择推理服务方案。

Brian: Right. A technical choice would depend on the team's model and workload. The compute support for multi-node DeepSeek R1 work tells us Nebius has supported a particular kind of large-model work. It doesn't transfer a performance result to a different model or deployment.

Brian：对。技术选择取决于团队使用的模型和工作负载。Nebius 为 DeepSeek R1 的多节点工作提供算力支持，说明它支持过某种大型模型工作。但这不能把性能结果直接套用到不同的模型或部署方式上。

Ava: What would you look for next? I'd start with a concrete contribution to PyTorch or vLLM and the problem it addresses. Then I'd ask whether it applies to my workload. That feels more actionable than trying to infer an improvement from the membership itself.

Ava：接下来你会关注什么？我会先看它对 PyTorch 或 vLLM 作出的具体贡献，以及这项贡献解决了什么问题。然后我会问，它是否适用于我的工作负载。这比试图从会员身份本身推断性能会有所提升更便于采取行动。

Brian: I'd also watch how Nebius shares lessons from production-scale inference and post-training. The value for other engineers will depend on the details it can share and the changes that follow. That's a question for future work. Today's announcement establishes the plan and the formal roles.

Brian：我也会关注 Nebius 如何分享大规模生产环境中的推理和后训练经验。这些经验对其他工程师有多大价值，取决于它能分享多少细节，以及随后会带来哪些改动。这要看未来的工作。今天的公告确定的是计划和正式角色。

Ava: Here's my recap. Nebius already supports PyTorch training and fine-tuning on its cloud, provides compute for vLLM work, and contributes open-source optimizations. Platinum membership adds a Governing Board seat, with Ryan Hanrui Wang also representing Nebius on the Technical Advisory Council.

Ava：我来回顾一下。Nebius 已经在其云上支持使用 PyTorch 进行训练和微调，为 vLLM 相关工作提供算力，并贡献开源优化。白金会员身份使它获得了理事会席位，Ryan Hanrui Wang 还将在技术咨询委员会代表 Nebius。

Brian: And the next areas to watch are PyTorch and vLLM performance on the latest hardware platforms, plus lessons from production-scale inference and post-training. Specific contributions will show what that means for engineers. Thanks for listening; we'll see you next time.

Brian：接下来值得关注的领域，是 PyTorch 和 vLLM 在最新硬件平台上的性能，以及来自大规模生产环境中推理和后训练工作的经验。具体贡献会让工程师看到这些计划意味着什么。感谢收听，我们下次再见。

## 术语

| Term | 释义 |
|---|---|
| PyTorch Foundation | 支持 PyTorch 及相关开源项目的基金会 |
| vendor-neutral | 不隶属于单一供应商的 |
| Platinum member | 白金级会员 |
| PyTorch Ecosystem | PyTorch 相关项目与社区构成的生态 |
| vLLM | 文章提到的开源模型推理项目 |
| fine-tune | 在已有模型基础上继续训练 |
| inference | 使用模型生成输出的过程 |
| post-training | 初始训练之后的模型处理或优化 |
| benchmarking | 通过基准测试评估系统表现 |
| multi-node | 使用多个计算节点共同运行的 |
| Governing Board | 负责基金会政策和总体方向的理事会 |
| Technical Advisory Council | 技术咨询委员会 |
| sparse-attention | 只计算部分注意力关系的稀疏注意力机制 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's make that concrete. | 我们用一个具体情境来说明。 |
| That distinction matters. | 这个区分很重要。 |
| It gives us a starting point. | 这给了我们一个出发点。 |
| That's a fair reading. | 这样理解是合理的。 |
| What would you look for next? | 接下来你会关注什么？ |
| That's a question for future work. | 这个问题要留待后续工作回答。 |
| Here's my recap. | 我来简要回顾一下。 |

## 中文内容总结

2026 年 10 月 20 日，PyTorch 基金会宣布 AI 云服务商 Nebius 成为白金会员。双方将围绕模型训练、推理和部署继续开展开源协作。

### 已有合作

初创企业和大型企业已在 Nebius AI Cloud 上使用 PyTorch 训练及微调模型。Nebius 长期为 vLLM 的开发、测试和基准测试提供算力，包括 DeepSeek R1 等大模型的多节点工作。其 Applied AI 团队向开源项目贡献推理和训练后优化，部分成果已进入 PyTorch 生态。

### 后续计划

Nebius 计划改进 PyTorch 和 vLLM 在新硬件平台上的性能，并分享其大规模生产环境中运行推理和训练后流程的经验。这些是未来计划；公告没有给出性能数据、硬件型号或完成时间。

### 治理安排

白金会员资格使 Nebius 获得 PyTorch 基金会理事会的一个席位。Nebius 的 Applied AI 高级副总裁 Ryan Hanrui Wang 将加入理事会，并代表公司参加技术咨询委员会。基金会由 Linux Foundation 托管，支持 PyTorch、vLLM 等开源项目。
