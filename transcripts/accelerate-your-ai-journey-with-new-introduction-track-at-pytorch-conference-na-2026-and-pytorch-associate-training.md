# Finding Your Way Into the PyTorch Ecosystem

原文：[Accelerate Your AI Journey with new Introduction Track at PyTorch Conference NA 2026 and PyTorch Associate Training - PyTorch](https://pytorch.org/blog/accelerate-your-ai-journey-with-new-introduction-track-at-pytorch-conference-na-2026-and-pytorch-associate-training/)

## 摘要

本期介绍 PyTorch Conference North America 2026 新设的 Introduction Track，以及面向实践的 PyTorch Associate Training。两位主持人讨论了从张量和自动求导，到 Triton 内核、性能分析、视觉语言模型和注意力机制的课程内容。培训包含讲解、演示、Notebook 实验和理解测验；完成培训的学员可获得 PyTorch Certified Associate 考试代金券。文章介绍的是课程安排，并未提供学习成效或性能提升数据。

## 对话

Ava: AI, or artificial intelligence, models are moving into everyday infrastructure. If I know some PyTorch but have gaps, where do I start?

Ava：AI，也就是人工智能，模型正逐渐融入日常基础设施。如果我懂一些 PyTorch，但还有知识空白，该从哪里开始？

Brian: That's the question behind a new Introduction Track at PyTorch Conference North America and a full-day PyTorch Associate Training session. Both focus on practical skills.

Brian：这正是 PyTorch Conference North America 新设的 Introduction Track 和为期一整天的 PyTorch Associate Training 课程所要回答的问题。两者都侧重实用技能。

Ava: Practical as in building a model, or practical as in debugging one when it's slow?

Ava：这里的“实用”是指构建模型，还是指模型运行缓慢时排查问题？

Brian: Both. The article says teams need experience training, optimizing, and deploying models. The track also welcomes experienced engineers exploring an unfamiliar part of the ecosystem.

Brian：两者都是。文章说，团队需要具备训练、优化和部署模型的经验。这个专题也欢迎想探索生态系统中陌生领域的资深工程师。

Ava: Good. An introduction doesn't have to mean I need an introduction to Python. What's the starting point?

Ava：好。入门并不意味着我连 Python 都需要从头学。起点是什么？

Brian: One session begins with tensors, the containers used for model data, and gradients, which guide learning. It uses visuals and exercises before a neural network and training loop.

Brian：其中一节课从张量和梯度讲起：张量是承载模型数据的容器，梯度用于指导学习。在进入神经网络和训练循环之前，课程会先用图示和练习帮助理解。

Ava: And dynamic computation graphs? I can read that phrase, but I'd like a picture in my head.

Ava：那动态计算图呢？这个词我能看懂，但我想在脑子里形成一幅图。

Brian: Think of a graph as a map of the operations a model performs. That session aims to build that mental picture before asking people to implement anything.

Brian：可以把图看作模型所执行操作的路线图。这节课希望先让大家建立这种直观认识，再让他们动手实现。

Ava: Right. But I'm a framework engineer. Is there a session where I can look under the hood?

Ava：明白。但我是框架工程师。有能让我看看底层机制的课程吗？

Brian: Yes. TinyTorch guides developers through rebuilding tensors, autograd, optimizers, and transformers in pure Python. Autograd means automatic gradient calculation. It's like assembling an engine to understand its parts.

Brian：有。TinyTorch 会引导开发者用纯 Python 重新实现张量、autograd、优化器和 transformers。autograd 指自动计算梯度。这就像亲手组装一台发动机，以弄懂各个零件。

Ava: That's the part I didn't get. Is TinyTorch presented as a replacement for PyTorch?

Ava：这正是我没弄明白的地方。TinyTorch 被当作 PyTorch 的替代品来介绍吗？

Brian: The article calls it a project for learning framework internals. The motivation is understanding complex logic, hardware optimization, and backend contributions. It doesn't claim TinyTorch replaces PyTorch.

Brian：文章称它是一个用于学习框架内部机制的项目。其出发点是理解复杂逻辑、硬件优化和后端贡献。文章并未声称 TinyTorch 会取代 PyTorch。

Ava: What about writing code for a graphics processing unit, or GPU? That sounds like a steep jump.

Ava：那为图形处理器，也就是 GPU，编写代码呢？听起来难度一下子高了很多。

Brian: The Triton session is meant to make that jump approachable. Triton is an open-source programming model embedded in Python for writing high-performance GPU code.

Brian：Triton 课程就是为了让这一步变得容易上手。Triton 是一种嵌入 Python 的开源编程模型，用于编写高性能 GPU 代码。

Ava: Wait, so what do people actually write there?

Ava：等等，那大家在里面具体会写什么？

Brian: Custom GPU kernels, which are small programs run on the GPU. They cover memory movement and parallel work, then write vector addition and matrix multiplication kernels.

Brian：自定义 GPU 内核，也就是在 GPU 上运行的小程序。他们会学习内存数据移动和并行计算，然后编写向量加法和矩阵乘法内核。

Ava: And the article mentions torch.compile. Where does that fit?

Ava：文章还提到了 torch.compile。它在这里起什么作用？

Brian: It's a PyTorch tool developers use to generate GPU kernels. The Triton talk addresses the next question: how do you write a custom kernel yourself?

Brian：它是开发者用来生成 GPU 内核的 PyTorch 工具。Triton 讲座接着要回答的问题是：你自己该怎么编写自定义内核？

Ava: Say I've written one. How do I tell whether it helps?

Ava：假设我写了一个，该怎么判断它有没有帮助？

Brian: There's a talk on reading PyTorch traces. A trace is like a receipt for execution: it helps you see where time went before you try to optimize.

Brian：有一场讲座会讲如何解读 PyTorch 执行轨迹。执行轨迹就像一张记录执行过程的收据：它能帮你在尝试优化之前看清时间花在了哪里。

Ava: What should I listen for in that session?

Ava：听那场讲座时，我该留意哪些内容？

Brian: The path from the central processing unit, or CPU, to the GPU; whether work is limited by computation or overhead; and why kernel timings vary.

Brian：从中央处理器，也就是 CPU，到 GPU 的执行路径；工作是受计算本身还是额外开销限制；以及内核执行时间为什么会波动。

Ava: Does it compare tools, or just explain the charts?

Ava：它会比较工具，还是只解释图表？

Brian: It plans workload comparisons using torch.compile, fused Triton kernels, and Liger kernels. The article gives no winning times, so we'd have to hear the talk for those.

Brian：它计划比较使用 torch.compile、融合的 Triton 内核和 Liger 内核时的工作负载表现。文章没有给出胜出方案的运行时间，所以我们得听讲座才能知道。

Ava: That's a useful distinction. What about models that handle both images and words?

Ava：这个区别很有用。那同时处理图像和文字的模型呢？

Brian: A vision-language models session examines five open-source architectures. It covers turning images into tokens, combining vision and language, multi-GPU fine-tuning, and serving overhead.

Brian：一场视觉语言模型课程探讨了5种开源架构，内容涵盖如何将图像转换为词元、结合视觉与语言、多 GPU 微调，以及模型服务的开销。

Ava: Serving overhead sounds broad. Is there a concrete example?

Ava：模型服务的额外开销听起来很宽泛。有具体例子吗？

Brian: Yes: image-token expansion and KV-cache pressure. The KV-cache holds key and value data used by attention, and the session treats its memory use as a serving challenge.

Brian：有，比如图像 token 数量膨胀和 KV-cache 压力。KV-cache 保存注意力机制所用的键和值数据；这节课会把它占用的内存视为模型服务面临的一项挑战。

Ava: There's also a vLLM session. Is that connected?

Ava：还有一场 vLLM 课程。它和这个有关吗？

Brian: It focuses on attention patterns for long contexts, including hybrid, sliding-window, sparse, and linear attention. It explains how vLLM represents them with backends, connectors, and memory allocators.

Brian：它聚焦长上下文中的注意力模式，包括混合注意力、滑动窗口注意力、稀疏注意力和线性注意力，并解释 vLLM 如何通过后端、连接器和内存分配器来表示这些模式。

Ava: Let me put it this way. The track goes from building blocks to what happens during real execution?

Ava：换句话说，这个专题会从基础构件一直讲到实际执行时发生的事？

Brian: Exactly. It moves across framework internals, GPU programming, performance traces, and model serving. Those are different entry points, depending on what you want to learn.

Brian：正是这样。它涵盖框架内部机制、GPU 编程、性能轨迹和模型服务。你想学什么，就可以从相应的方向入手。

Ava: Now the separate Associate Training: what happens during that day?

Ava：再说说单独开设的 Associate Training：那一天会做些什么？

Brian: It's an in-person, instructor-led workshop with sequential modules. There are live demos, guided Jupyter Notebook labs, quizzes, reviews, and projects based on production scenarios.

Brian：这是一场线下、由讲师带领的工作坊，课程模块按顺序展开。内容包括现场演示、有人指导的 Jupyter Notebook 实操、测验、复习，以及基于生产场景的项目练习。

Ava: So attendees build and train models, then get feedback while they're working?

Ava：所以参加者会构建和训练模型，并在动手过程中得到反馈？

Brian: That's the plan. The labs also cover optimization. The article says participants can get direct feedback and real-time guidance from instructors while learning alongside peers.

Brian：计划是这样。实操还会涵盖优化。文章说，参加者可以在与同伴一起学习的过程中，得到讲师的直接反馈和实时指导。

Ava: What should someone know before arriving?

Ava：参加者来之前应该掌握什么？

Brian: They should be comfortable with Python, know basic machine learning, and be familiar with Jupyter Notebooks and basic Google Colab workflows. They'll need an active Google Account for exercises.

Brian：他们应该熟悉 Python，了解机器学习基础，并熟悉 Jupyter Notebooks 和基本的 Google Colab 操作流程。做练习还需要一个有效的 Google Account。

Ava: And what's the practical outcome? Is there an exam involved?

Ava：那实际收获是什么？会涉及考试吗？

Brian: Anyone who completes the training receives a voucher valued at two hundred fifty dollars for the PyTorch Certified Associate exam, or PTCA. The article doesn't report pass rates.

Brian：完成培训的任何人都可获得一张价值250美元的优惠券，用于参加 PyTorch Certified Associate 考试，即 PTCA。文章没有报告考试通过率。

Ava: Before we move on, when and where is the training?

Ava：继续之前，培训在什么时间、什么地点举行？

Brian: It's on Monday, October nineteenth, twenty twenty-six, from nine in the morning to five in the afternoon in San Jose. Faradawn Yang, an NVIDIA engineer, is the instructor.

Brian：课程于2026年10月19日星期一在圣何塞举行，时间为上午9点至下午5点。讲师是 NVIDIA 工程师 Faradawn Yang。

Ava: Is there anything to watch for when signing up?

Ava：报名时有什么需要注意的吗？

Brian: Pre-registration is required, and space is limited. The article says attendees can add the training to their conference registration through Tuesday, October thirteenth.

Brian：需要预先报名，名额有限。文章说，参加者可以在 Tuesday, October thirteenth 之前把培训加到自己的会议报名项目中。

Ava: Let's recap. First, the Introduction Track offers several ways into PyTorch. Second, the training adds guided practice. Third, the article describes planned learning, not measured results.

Ava：我们回顾一下。第一，Introduction Track 提供了多种学习 PyTorch 的切入点。第二，培训增加了有指导的练习。第三，文章描述的是计划开展的学习内容，而不是已经测得的结果。

Brian: That's it. Thanks for listening, and we'll see you next time.

Brian：就是这样。感谢收听，我们下次再见。

## 术语

| Term | 释义 |
|---|---|
| tensors | 用于表示模型数据的多维数据结构。 |
| gradients | 用于指导模型参数更新的梯度。 |
| dynamic computation graphs | 描述模型运算过程、随执行而构建的计算图。 |
| TinyTorch | 帮助开发者用纯 Python 重建框架核心组件的开源项目。 |
| autograd | 自动计算梯度的机制。 |
| Triton | 嵌入 Python、用于编写高性能 GPU 程序的开源编程模型。 |
| GPU kernels | 在图形处理器上运行的小型计算程序。 |
| torch.compile | 文章提及的、用于生成 GPU 内核的 PyTorch 工具。 |
| PyTorch traces | 用于分析 PyTorch 工作负载执行与耗时的跟踪记录。 |
| vision-language models | 同时处理视觉和语言信息的模型。 |
| KV-cache | 注意力计算使用的键值缓存；其内存占用是文章提及的服务挑战。 |
| PyTorch Certified Associate | 文章提及的 PyTorch 助理级认证考试。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Where do I start? | 我该从哪里开始？ |
| That's the part I didn't get. | 这部分我没弄明白。 |
| Wait, so what do people actually write there? | 等等，所以大家在那里具体写什么？ |
| Where does that fit? | 它在整个流程中起什么作用？ |
| What should I listen for? | 我应该重点听什么？ |
| That's a useful distinction. | 这个区分很有帮助。 |
| Let me put it this way. | 我换个说法。 |
| Before we move on | 在进入下一个话题之前。 |
