# TinyTorch: Don’t Just Import PyTorch. Build It.

原文：[TinyTorch: Don’t Just Import PyTorch. Build It. - PyTorch](https://pytorch.org/blog/tinytorch-dont-just-import-pytorch-build-it/)

## 摘要

本期介绍 TinyTorch：一个用纯 Python、仿照 PyTorch API，从张量一路构建到 Transformer 的开源教学框架。它通过二十个模块、四个层级和六个历史里程碑，让学习者亲手实现内存分析、自动微分、优化器、注意力等系统。TinyTorch 强调低硬件门槛、渐进式揭示和与生产 API 保持一致，也被用于大学课程、企业入职培训和调试工作坊。文章同时承认它仍是预览版，缺少 GPU、分布式训练和受控学习成效数据。

## 对话

Ava: What if learning PyTorch internals meant building a tiny version yourself, on an old laptop, with no GPU? That’s TinyTorch, and it could change how framework engineers learn.

Ava：如果学习 PyTorch 内部原理，意味着在一台没有 GPU 的旧笔记本电脑上，亲手搭建一个微型版本，会怎么样？这就是 TinyTorch，它可能改变框架工程师的学习方式。

Brian: Right. TinyTorch is a free, open-source curriculum. You build a working machine-learning framework from scratch, from tensors through transformers, in pure Python, while using PyTorch’s own API.

Brian：没错。TinyTorch 是一套免费、开源的课程。你用纯 Python，从张量一路做到 Transformer，从零搭建一个能运行的机器学习框架，同时使用 PyTorch 自身的 API。

Ava: So this isn’t just another tutorial where you read about autograd and then import torch.

Ava：所以这并不是那种读一读 autograd，然后导入 torch 就算学完的教程。

Brian: Exactly. You implement the pieces. There are twenty modules, and the whole thing runs on a laptop with four gigabytes of RAM and no GPU.

Brian：没错。你要亲手实现各个部分。整个项目有 20 个模块，只需一台配备 4 GB 内存、没有 GPU 的笔记本电脑就能运行。

Ava: Why did the authors think PyTorch needed a teaching version? PyTorch already has a huge amount of documentation.

Ava：作者为什么觉得 PyTorch 需要一个教学版本？PyTorch 已经有大量文档了。

Brian: Because mature systems become too large to hold in your head. The article compares PyTorch with MINIX, the Tiger compiler, and xv6. Those teaching systems don’t replace production systems. They expose what production systems normally hide.

Brian：因为成熟系统会变得太庞大，很难在脑中把握全貌。文章把 PyTorch 与 MINIX、Tiger 编译器和 xv6 相比。这些教学系统不会取代生产系统，而是让人看见生产系统通常藏起来的东西。

Ava: So TinyTorch is the missing rung below the real framework.

Ava：所以 TinyTorch 就是通往真正框架之前缺少的那一级台阶。

Brian: That’s the idea. Existing writing on PyTorch internals is often excellent, but it assumes you already think like a framework engineer. TinyTorch gives you something you can finish and inspect yourself.

Brian：就是这个意思。现有的 PyTorch 内部原理文章往往写得很好，但它们假定你已经具备框架工程师的思维方式。TinyTorch 则给了你一个能亲自做完、亲自检查的系统。

Ava: And the argument is aimed at PyTorch teams too, not only students.

Ava：而且这套论点也是讲给 PyTorch 团队听的，不只是讲给学生听的。

Brian: Yes. Frameworks depend on a small group of people who can reason from the inside. They notice tensor-cache leaks, understand when gradient checkpointing is worth recomputing, and identify bottlenecks before opening a profiler.

Brian：对。框架依赖一小群能从内部理解系统的人。他们能发现张量缓存泄漏，知道什么时候梯度检查点值得付出重新计算的代价，还能在打开性能分析器之前找出瓶颈。

Ava: But that group doesn’t grow automatically.

Ava：但这样的人不会自动变多。

Brian: Most people reach the internals because something broke badly. They learn under deadline pressure, from the outside in, in whatever order the bug demands. That creates competence with holes in it.

Brian：大多数人是因为出了严重问题才接触内部实现。他们迫于截止期限，从外往里学，顺序完全取决于故障要求他们查什么。这样积累的能力难免有缺口。

Ava: Building the system changes that arrival path.

Ava：亲手搭建系统，改变了他们进入这个领域的路径。

Brian: Once you implement autograd yourself, you can’t unsee the computational graph. Once you profile your own memory allocation, you can’t forget its cost.

Brian：一旦你亲手实现了 autograd，就再也无法忽视计算图。一旦你分析过自己程序的内存分配，就再也忘不了它的开销。

Ava: The article has a nice image there: the abstraction can be a wall or a door.

Ava：文章在这里用了个很好的比喻：抽象层可以是一堵墙，也可以是一扇门。

Brian: Right. A student who writes backward() and allocates Adam’s momentum and variance buffers already has a mental model. When they meet production autograd, it looks like a more sophisticated version of something they understand.

Brian：没错。写过 backward()、分配过 Adam 动量缓冲区和方差缓冲区的学生，脑中已经有了一个模型。等他们接触生产环境中的 autograd，看到的就是自己所理解之物的更复杂版本。

Ava: That sounds useful for onboarding.

Ava：这听起来对新人培训很有用。

Brian: Companies are using it that way. Teams run two-to-three-week intensives, quarter-long internal training, and targeted workshops. Someone might work through Module zero six on autograd or Module twelve on attention.

Brian：企业也在这样使用它。团队会开展为期 2 到 3 周的集中学习、持续一个季度的内部培训，以及针对特定主题的工作坊。有人可能会学习关于自动微分的 Module 06，或者关于注意力机制的 Module 12。

Ava: Okay, how is the curriculum organized?

Ava：好，那课程是怎么安排的？

Brian: Twenty modules in four tiers, driven by a command-line tool called tito. The lessons are Jupyter notebooks with the hard parts removed for learners to fill in.

Brian：它有 4 个层级、共 20 个模块，由一个名叫 tito 的命令行工具驱动。课程内容是 Jupyter 笔记本，其中较难的部分被留空，让学习者自己补上。

Ava: What does a learner need?

Ava：学习者需要具备什么条件？

Brian: Python and comfort with NumPy. No GPU, no cloud account, and no prior machine-learning systems background.

Brian：会用 Python，熟悉 NumPy。无需 GPU、云端账号，也不要求事先了解机器学习系统。

Ava: The first design choice is called systems from day one, right?

Ava：第一个设计选择叫作“从第一天起就学习系统知识”，对吧？

Brian: Yes. Module zero one includes a memory_footprint() method before matrix multiplication. You learn that one batch of thirty-two ImageNet images costs nineteen megabytes by computing it.

Brian：对。Module 01 在矩阵乘法之前就引入了 memory_footprint() 方法。你会通过亲自计算，知道一批 32 张 ImageNet 图像要占用 19 MB 内存。

Ava: Then later you discover Adam’s memory cost by actually allocating its buffers.

Ava：之后，你又通过实际分配 Adam 的缓冲区，发现它的内存开销。

Brian: Exactly. Adam needs roughly three times the optimizer memory of SGD. You measure that yourself, so it’s harder to dismiss as a fact from a slide.

Brian：正是如此。Adam 所需的优化器内存大约是 SGD 的 3 倍。你自己测出这个结果，就更难把它当成幻灯片上随口列出的事实而不予理会。

Ava: The second choice is progressive disclosure.

Ava：第二个选择是渐进式呈现。

Brian: The Tensor class stays clean through Module zero five. In Module zero six, you implement enable_autograd(), adding requires_grad, grad, and backward() to the class you already know.

Brian：直到 Module 05，Tensor 类都保持简洁。到了 Module 06，你要实现 enable_autograd()，给已经熟悉的这个类添加 requires_grad、grad 和 backward()。

Ava: That keeps the early data layout and arithmetic from being buried under gradient machinery.

Ava：这样，早期学习的数据布局和算术运算就不会被梯度机制淹没。

Brian: And the old code keeps working. The implementation uses runtime monkey-patching instead of inheritance.

Brian：而且旧代码仍然能运行。这个实现采用运行时 monkey-patching，而不是继承。

Ava: That choice will make some Python people uncomfortable.

Ava：这个选择会让一些 Python 开发者感到不自在。

Brian: The authors admit that. It won because one Tensor class survives across all twenty modules, and students remember the moment their tensors visibly gain new powers. It also mirrors PyTorch zero point four, when Variable was merged into Tensor.

Brian：作者也承认这一点。这个设计最终胜出，是因为同一个 Tensor 类贯穿全部 20 个模块，而且学生会记得自己的张量明显获得新能力的那一刻。这也呼应了 PyTorch 0.4 将 Variable 合并进 Tensor 的做法。

Ava: What’s the third design choice?

Ava：第三个设计选择是什么？

Brian: Build to validate. Six historical milestones check whether the implementation works: the Perceptron in nineteen fifty-eight, the XOR crisis in nineteen sixty-nine, the backpropagation revival in nineteen eighty-six, a nineteen ninety-eight CNN breakthrough, the Transformer in twenty seventeen, and finally MLPerf-style benchmarking.

Brian：通过构建来验证。6 个历史里程碑用来检验你的实现是否有效：1958 年的 Perceptron、1969 年的 XOR 危机、1986 年反向传播的复兴、1998 年 CNN 的突破、2017 年的 Transformer，最后是 MLPerf 风格的基准测试。

Ava: So history supplies motivation, but task performance supplies evidence.

Ava：所以历史提供动机，而任务表现提供证据。

Brian: Exactly. A unit test can be faked or misunderstood. If your network learns, your autograd has to be doing something right. For the CNN milestone, the network must clear seventy-five percent on CIFAR-10.

Brian：没错。单元测试可能被糊弄过去，也可能被误解。如果你的网络学会了东西，你的自动微分实现至少有些地方是对的。在 CNN 里程碑中，这个网络在 CIFAR-10 上的准确率必须超过 75%。

Ava: And the API deliberately looks like PyTorch.

Ava：而且 API 是有意设计得像 PyTorch 的。

Brian: That’s the transfer mechanism. If you build loss.backward() in TinyTorch, you can recognize the shape of PyTorch’s version later. A PyTorch developer can read the attention module and see how the graph is constructed.

Brian：这正是知识迁移的方式。如果你在 TinyTorch 里实现过 loss.backward()，之后就能认出 PyTorch 版本的基本结构。PyTorch 开发者阅读注意力模块时，也能看出计算图是怎样构建的。

Ava: But the resemblance stops at the API surface.

Ava：但相似之处止于 API 表层。

Brian: Yes. There’s no dispatcher, no C++ or CUDA layer, no JIT, and nothing distributed. It’s also slow: pure Python runs somewhere between one hundred and ten thousand times slower than PyTorch.

Brian：对。它没有分发器，没有 C++ 或 CUDA 层，没有 JIT，也没有任何分布式功能。它还很慢：纯 Python 的运行速度比 PyTorch 慢大约 100 到 10000 倍。

Ava: That range is huge.

Ava：这个范围差得真大。

Brian: The article gives the range without narrowing it. The slowness became part of the lesson, though. A TinyTorch Conv2d can take ninety-seven seconds where PyTorch takes ten milliseconds.

Brian：文章只给出了这个范围，没有进一步缩小。不过，这种慢也成了课程的一部分。TinyTorch 的 Conv2d 可能需要 97 秒，而 PyTorch 只需 10 毫秒。

Ava: That makes vectorization painfully concrete.

Ava：这让向量化的意义变得具体得令人难受。

Brian: Exactly. It stops being advice you read and becomes something that happened to you.

Brian：没错。它不再是你读到的一条建议，而是你亲身经历的事。

Ava: Where did the project come from?

Ava：这个项目是怎么来的？

Brian: It began as a Harvard course, CS two forty-nine r, launched in twenty twenty as a graduate seminar on TinyML. There was no textbook, so the course notes became one.

Brian：它起源于哈佛大学的一门课程 CS 249r。这门关于 TinyML 的研究生研讨课于 2020 年开设。当时没有教材，于是课程讲义就成了教材。

Ava: Then an open repository grew around the notes.

Ava：随后，围绕这些笔记发展出了一个开放仓库。

Brian: Students and educators fixed examples and proposed chapters. By twenty twenty-four, people kept asking about training at scale, so the project split into two volumes.

Brian：学生和教育工作者修正了示例，并提出新章节的建议。到了 2024 年，人们不断询问大规模训练的问题，所以这个项目分成了两卷。

Ava: And eventually a textbook wasn’t enough.

Ava：到后来，光有一本教材已经不够了。

Brian: Reading about autograd and implementing autograd create different kinds of knowledge. The project added build modules, Marimo labs, MLSys dot im for performance modeling, hardware kits for Arduino and Raspberry Pi, StaffML for interview preparation, and an instructor hub.

Brian：阅读 autograd 和亲手实现 autograd，带来的是不同类型的知识。项目于是加入了动手构建模块、Marimo 实验课、用于性能建模的 MLSys dot im、面向 Arduino 和 Raspberry Pi 的硬件套件、用于面试准备的 StaffML，以及教师中心。

Ava: The community numbers also changed quickly.

Ava: 社区数据也迅速发生了变化。

Brian: The article says there were about two thousand stars in August twenty twenty-five. Someone posted the repository on X in October. Two weeks later it reached five thousand, passed ten thousand by December, and is now above twenty-seven thousand, with at least ninety-five contributors and courses at fifty or more universities.

Brian: 文章说，2025 年 8 月时，这个仓库大约有 2000 颗星标。10 月，有人在 X 上分享了这个仓库。两周后，星标数达到 5000，到 12 月突破 10000，如今已超过 27000；贡献者至少有 95 位，50 所或更多大学开设了相关课程。

Ava: They’re clear that this wasn’t a clever growth campaign.

Ava: 他们明确表示，这并不是什么巧妙的增长活动。

Brian: Right. One person with reach shared it, after five years of quiet accumulation. Andrea now maintains it from ETH Zurich, after it was built at Harvard. Moving a curriculum between institutions is a real test of whether it serves people beyond its original classroom.

Brian: 对。在默默积累了五年之后，一位有影响力的人分享了它。它最初在 Harvard 建立，如今由 Andrea 在 ETH Zurich 维护。把一套课程从一所机构带到另一所机构，才能真正检验它是否也能服务原来那间教室以外的人。

Ava: The article gives four lessons for other open curricula. Let’s walk through them.

Ava: 文章给其他开放课程总结了四条经验。我们逐条看看。

Brian: First, match the production API exactly. Every hour learners spend translating between a teaching API and the real one is wasted effort.

Brian: 第一，要与生产环境使用的 API 完全一致。学习者花在教学 API 和真实 API 之间转换的每一个小时，都是白费的力气。

Ava: Second, choose the hardware floor before the feature list.

Ava: 第二，先确定硬件门槛，再列功能清单。

Brian: TinyTorch targets a dual-core two-gigahertz CPU, four gigabytes of RAM, and no network during training. It ships two tiny offline datasets: about one thousand grayscale digits and three hundred fifty conversational question-answer pairs, together under fifty megabytes.

Brian: TinyTorch 的目标硬件是双核 2 GHz CPU、4 GB RAM，而且训练期间不需要联网。它自带两个小型离线数据集：约 1000 张灰度数字图像和 350 组对话式问答，总大小不到 50 MB。

Ava: That choice excludes GPU support, but it enables Chromebook classrooms and old laptops.

Ava: 这个选择排除了 GPU 支持，却让使用 Chromebook 的课堂和旧笔记本电脑也能用上它。

Brian: Yes. Decide whom you’re willing to exclude before the project grows, because revisiting that decision gets expensive.

Brian: 是的。要在项目发展壮大之前决定愿意把哪些人排除在外，因为日后重新考虑这个决定会付出很高的代价。

Ava: Third, slow can be educational.

Ava: 第三，运行得慢也可以有教学价值。

Brian: The ninety-seven-second convolution teaches the need for vectorization better than a warning ever could.

Brian: 耗时 97 秒的卷积，比任何警告都更能让人明白为什么需要向量化。

Ava: And fourth, instructor infrastructure is the real bottleneck.

Ava: 第四，教师所需的配套设施才是真正的瓶颈。

Brian: Good content alone doesn’t get adopted. Gradeable content does. TinyTorch includes NBGrader autograding, locked test cells, point allocations, an INSTRUCTOR.md, milestone scripts, and three integration models for different course sizes.

Brian: 单靠优质内容不足以让课程被采用；内容还得便于评分。TinyTorch 包含 NBGrader 自动评分、锁定的测试单元、分值分配、INSTRUCTOR.md、里程碑脚本，以及适用于不同课程规模的三种整合模式。

Ava: So the instructor has to be supported as carefully as the learner.

Ava: 所以，对教师的支持必须和对学习者的支持一样周到。

Brian: Exactly. The instructor must defend the curriculum to a committee. If that path fails, teaching quality doesn’t matter.

Brian: 正是如此。教师必须向委员会说明并捍卫这套课程。如果这一步过不了关，教学质量再高也无济于事。

Ava: There’s also a succession plan.

Ava: 项目还有接班计划。

Brian: Yes. The project commits to maintenance through twenty twenty-seven, a two-week pull-request review target, and a governance transition during twenty twenty-six and twenty twenty-seven. Sustainability means making yourself replaceable, with a date attached.

Brian: 是的。项目承诺维护至 2027 年，目标是在两周内审查 pull request，并在 2026 年和 2027 年期间完成治理交接。可持续发展意味着让别人能够接替自己，而且要为此定下日期。

Ava: What remains unknown?

Ava: 还有哪些问题没有答案？

Brian: TinyTorch is still in preview, aimed at classroom readiness for Fall twenty twenty-six. The authors have not measured learning outcomes. They have a strong educational rationale, but no controlled data showing that TinyTorch students debug production systems better than students in a conventional course.

Brian: TinyTorch 仍处于预览阶段，目标是在 2026 年秋季达到可用于课堂教学的状态。作者尚未衡量学习成效。他们有充分的教育理念依据，但没有对照数据证明，TinyTorch 的学生比传统课程的学生更擅长调试生产系统。

Ava: And the technical scope has real holes.

Ava: 而且，它的技术覆盖范围确实有缺口。

Brian: It’s single-node and CPU-only. It teaches memory and compute, but not GPU kernels, distributed training, or gradient synchronization. Parallel data loading and GPU memory management were deliberately left out because doing them properly would break the four-gigabyte floor.

Brian: 它只支持单节点和 CPU。它教授内存和计算方面的知识，却不涵盖 GPU 内核、分布式训练或梯度同步。并行数据加载和 GPU 显存管理被有意排除，因为要妥善实现这些内容，就会突破 4 GB 的硬件门槛。

Ava: So what can people do next?

Ava: 那么，大家接下来可以做什么？

Brian: They can review modules against real PyTorch semantics, pilot a tier and report failures, write the missing distributed and GPU modules, localize the material, or adopt it and add their institution to the community map.

Brian: 大家可以对照真实的 PyTorch 语义审查各个模块，试用一个课程层级并报告失败之处，编写缺失的分布式和 GPU 模块，将材料本地化，或者采用这套课程，并把自己的机构添加到社区地图上。

Ava: The code is MIT licensed, and the curriculum is CC BY-SA four point zero.

Ava: 代码采用 MIT 许可证，课程内容采用 CC BY-SA 4.0 许可证。

Brian: Right. Forking and adapting are explicitly allowed. The project also invites people to join the PyTorch Academic OSPO Working Group, which connects researchers, educators, students, and open-source practitioners.

Brian: 对。项目明确允许分叉和改编。它还邀请大家加入 PyTorch Academic OSPO Working Group；这个工作组把研究人员、教育工作者、学生和开源实践者联系起来。

Ava: Let’s close with the three points I’m taking away. First, building a small framework creates a mental model for the production one.

Ava: 最后说说我记住的三点。第一，亲手构建一个小型框架，能帮助人建立对生产环境所用框架的认知模型。

Brian: Second, matching the real API and keeping the hardware floor low makes the curriculum transferable.

Brian: 第二，与真实 API 保持一致，并维持较低的硬件门槛，能让这套课程更容易迁移到别处。

Ava: Third, the project is promising but still needs evidence, instructors, and contributors for the parts it leaves out.

Ava: 第三，这个项目很有前景，但仍需要成效证据、教师，以及帮助补齐遗漏部分的贡献者。

Brian: That’s TinyTorch: don’t just import PyTorch. Build enough of it to understand what happens when the abstraction breaks.

Brian: 这就是 TinyTorch：别只是导入 PyTorch。亲手构建其中足够多的部分，才能理解抽象层失效时会发生什么。

Ava: Thanks for listening. We’ll see you next time.

Ava: 感谢收听。下次见。

## 术语

| Term | 释义 |
|---|---|
| TinyTorch | 用纯 Python 从头构建机器学习框架的开源教学课程 |
| autograd | 自动微分系统，用于计算梯度 |
| computational graph | 计算图，记录运算及其依赖关系 |
| runtime monkey-patching | 运行时猴子补丁，在运行时修改或扩展类 |
| progressive disclosure | 渐进式揭示，逐步引入复杂机制 |
| memory footprint | 内存占用量 |
| gradient checkpointing | 梯度检查点，通过重计算减少内存使用 |
| vectorization | 向量化，用批量运算替代逐元素循环 |
| dispatcher | 算子分发器，负责选择具体实现 |
| JIT | 即时编译（Just-In-Time compilation） |
| NBGrader | 用于 Jupyter Notebook 自动评分的工具 |
| MLPerf | 机器学习性能基准测试体系 |
| gradient synchronization | 分布式训练中的梯度同步 |
| hardware floor | 课程或系统支持的最低硬件配置 |
| constructionism | 通过亲手构建对象来学习的教育理念 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What if learning PyTorch internals meant building a tiny version yourself? | 如果学习 PyTorch 内部机制意味着亲手构建一个小版本呢？ |
| That’s the idea. | 这就是核心想法。 |
| You can’t unsee the computational graph. | 一旦理解计算图，就再也无法忽视它。 |
| Let’s walk through them. | 我们逐条讲一下。 |
| That makes it painfully concrete. | 这让它变得非常具体，甚至有点痛苦。 |
| What remains unknown? | 还有哪些事情尚不清楚？ |
| The scope has real holes. | 它的范围确实存在明显缺口。 |
| What can people do next? | 接下来大家可以做什么？ |
| Let’s close with the three points I’m taking away. | 最后说说我总结出的三点。 |

## 中文内容总结

TinyTorch 是免费开源的教学项目。学习者用纯 Python 完成从张量到 Transformer 的 20 个模块，逐步构建可运行的机器学习框架。它沿用 PyTorch 的 API，只需具备 Python 和 NumPy 基础；一台 4 GB 内存、无 GPU 的笔记本就能运行。

### 要解决的问题

PyTorch 的生产代码功能丰富，却很难作为初学者理解框架内部机制的第一份教材。作者认为，亲手实现自动求导、优化器和训练循环，能帮助学习者形成计算图、内存分配和性能瓶颈的具体认识，也便于日后阅读 PyTorch 代码。这是项目的教学主张，尚未经过学习效果对照研究验证。

### 课程如何组织

课程分为四层、20 个模块，通过 tito 命令行工具和留出关键实现的 Jupyter 笔记本交付。第一模块就要求计算内存占用；第六模块才为已有的 Tensor 类加入 requires_grad、.grad 和 .backward()。学习者还会实现注意力等组件，并用六个历史里程碑检验程序能否完成任务。其中，CNN 里程碑要求模型在 CIFAR-10 上超过 75%，这是通关条件，不是所有学习者已取得的结果。

### 运行条件与性能

课程提供两份合计不足 50 MB 的离线小数据集，训练时无需联网。作者称，纯 Python 实现比 PyTorch 慢约 100 至 10,000 倍；文中的一个 Conv2d 示例处理同一批数据需 97 秒，PyTorch 需 10 毫秒。项目把这种速度差用于讲解向量化，但并未把 TinyTorch 设计成生产框架。

### 采用与教学配套

TinyTorch 社区地图自 2025 年 12 月上线以来，记录了分布在 92 所机构的 682 名成员。文章列出的 Machine Learning Systems 仓库超过 27,000 星，并报告至少 95 名贡献者及至少 50 所大学开设相关课程。课程同时提供 NBGrader 自动评分、里程碑验证脚本和教师指南，可按完整学期、半学期单元或单独的优化专题使用。作者认为，可评分且易于部署的材料是学校采用课程的关键。企业团队也将它用于为期两三周的新员工集中培训、贯穿一个季度的内部培训，以及围绕自动求导或注意力模块的定向调试研讨会。

### 边界与后续工作

TinyTorch 仍处预览阶段，目标是供 2026 年秋季课堂使用。它仅支持单机 CPU，没有 PyTorch 式的算子派发机制、C++ 或 CUDA 层、JIT 和分布式训练，也不教授 GPU 内存管理、并行数据加载或梯度同步。项目尚无对照数据证明其学生更擅长调试生产系统。团队计划维持项目至 2027 年，并在 2026 至 2027 年推进教师顾问委员会和维护交接；他们也邀请教师试教、开发缺失模块及本地化材料。
