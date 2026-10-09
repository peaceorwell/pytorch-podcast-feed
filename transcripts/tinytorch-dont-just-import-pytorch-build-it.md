# TinyTorch: Don’t Just Import PyTorch. Build It.

原文：[TinyTorch: Don’t Just Import PyTorch. Build It.](https://pytorch.org/blog/tinytorch-dont-just-import-pytorch-build-it/)

## 摘要

本期介绍 TinyTorch：一个用纯 Python、仿照 PyTorch API，从张量一路构建到 Transformer 的开源教学框架。它通过二十个模块、四个层级和六个历史里程碑，让学习者亲手实现内存分析、自动微分、优化器、注意力等系统。TinyTorch 强调低硬件门槛、渐进式揭示和与生产 API 保持一致，也被用于大学课程、企业入职培训和调试工作坊。文章同时承认它仍是预览版，缺少 GPU、分布式训练和受控学习成效数据。

## 对话

**Ava:** What if learning PyTorch internals meant building a tiny version yourself, on an old laptop, with no GPU? That’s TinyTorch, and it could change how framework engineers learn.

**Brian:** Right. TinyTorch is a free, open-source curriculum. You build a working machine-learning framework from scratch, from tensors through transformers, in pure Python, while using PyTorch’s own API.

**Ava:** So this isn’t just another tutorial where you read about autograd and then import torch.

**Brian:** Exactly. You implement the pieces. There are twenty modules, and the whole thing runs on a laptop with four gigabytes of RAM and no GPU.

**Ava:** Why did the authors think PyTorch needed a teaching version? PyTorch already has a huge amount of documentation.

**Brian:** Because mature systems become too large to hold in your head. The article compares PyTorch with MINIX, the Tiger compiler, and xv6. Those teaching systems don’t replace production systems. They expose what production systems normally hide.

**Ava:** So TinyTorch is the missing rung below the real framework.

**Brian:** That’s the idea. Existing writing on PyTorch internals is often excellent, but it assumes you already think like a framework engineer. TinyTorch gives you something you can finish and inspect yourself.

**Ava:** And the argument is aimed at PyTorch teams too, not only students.

**Brian:** Yes. Frameworks depend on a small group of people who can reason from the inside. They notice tensor-cache leaks, understand when gradient checkpointing is worth recomputing, and identify bottlenecks before opening a profiler.

**Ava:** But that group doesn’t grow automatically.

**Brian:** Most people reach the internals because something broke badly. They learn under deadline pressure, from the outside in, in whatever order the bug demands. That creates competence with holes in it.

**Ava:** Building the system changes that arrival path.

**Brian:** Once you implement autograd yourself, you can’t unsee the computational graph. Once you profile your own memory allocation, you can’t forget its cost.

**Ava:** The article has a nice image there: the abstraction can be a wall or a door.

**Brian:** Right. A student who writes backward() and allocates Adam’s momentum and variance buffers already has a mental model. When they meet production autograd, it looks like a more sophisticated version of something they understand.

**Ava:** That sounds useful for onboarding.

**Brian:** Companies are using it that way. Teams run two-to-three-week intensives, quarter-long internal training, and targeted workshops. Someone might work through Module zero six on autograd or Module twelve on attention.

**Ava:** Okay, how is the curriculum organized?

**Brian:** Twenty modules in four tiers, driven by a command-line tool called tito. The lessons are Jupyter notebooks with the hard parts removed for learners to fill in.

**Ava:** What does a learner need?

**Brian:** Python and comfort with NumPy. No GPU, no cloud account, and no prior machine-learning systems background.

**Ava:** The first design choice is called systems from day one, right?

**Brian:** Yes. Module zero one includes a memory_footprint() method before matrix multiplication. You learn that one batch of thirty-two ImageNet images costs nineteen megabytes by computing it.

**Ava:** Then later you discover Adam’s memory cost by actually allocating its buffers.

**Brian:** Exactly. Adam needs roughly three times the optimizer memory of SGD. You measure that yourself, so it’s harder to dismiss as a fact from a slide.

**Ava:** The second choice is progressive disclosure.

**Brian:** The Tensor class stays clean through Module zero five. In Module zero six, you implement enable_autograd(), adding requires_grad, grad, and backward() to the class you already know.

**Ava:** That keeps the early data layout and arithmetic from being buried under gradient machinery.

**Brian:** And the old code keeps working. The implementation uses runtime monkey-patching instead of inheritance.

**Ava:** That choice will make some Python people uncomfortable.

**Brian:** The authors admit that. It won because one Tensor class survives across all twenty modules, and students remember the moment their tensors visibly gain new powers. It also mirrors PyTorch zero point four, when Variable was merged into Tensor.

**Ava:** What’s the third design choice?

**Brian:** Build to validate. Six historical milestones check whether the implementation works: the Perceptron in nineteen fifty-eight, the XOR crisis in nineteen sixty-nine, the backpropagation revival in nineteen eighty-six, a nineteen ninety-eight CNN breakthrough, the Transformer in twenty seventeen, and finally MLPerf-style benchmarking.

**Ava:** So history supplies motivation, but task performance supplies evidence.

**Brian:** Exactly. A unit test can be faked or misunderstood. If your network learns, your autograd has to be doing something right. For the CNN milestone, the network must clear seventy-five percent on CIFAR-10.

**Ava:** And the API deliberately looks like PyTorch.

**Brian:** That’s the transfer mechanism. If you build loss.backward() in TinyTorch, you can recognize the shape of PyTorch’s version later. A PyTorch developer can read the attention module and see how the graph is constructed.

**Ava:** But the resemblance stops at the API surface.

**Brian:** Yes. There’s no dispatcher, no C++ or CUDA layer, no JIT, and nothing distributed. It’s also slow: pure Python runs somewhere between one hundred and ten thousand times slower than PyTorch.

**Ava:** That range is huge.

**Brian:** The article gives the range without narrowing it. The slowness became part of the lesson, though. A TinyTorch Conv2d can take ninety-seven seconds where PyTorch takes ten milliseconds.

**Ava:** That makes vectorization painfully concrete.

**Brian:** Exactly. It stops being advice you read and becomes something that happened to you.

**Ava:** Where did the project come from?

**Brian:** It began as a Harvard course, CS two forty-nine r, launched in twenty twenty as a graduate seminar on TinyML. There was no textbook, so the course notes became one.

**Ava:** Then an open repository grew around the notes.

**Brian:** Students and educators fixed examples and proposed chapters. By twenty twenty-four, people kept asking about training at scale, so the project split into two volumes.

**Ava:** And eventually a textbook wasn’t enough.

**Brian:** Reading about autograd and implementing autograd create different kinds of knowledge. The project added build modules, Marimo labs, MLSys dot im for performance modeling, hardware kits for Arduino and Raspberry Pi, StaffML for interview preparation, and an instructor hub.

**Ava:** The community numbers also changed quickly.

**Brian:** The article says there were about two thousand stars in August twenty twenty-five. Someone posted the repository on X in October. Two weeks later it reached five thousand, passed ten thousand by December, and is now above twenty-seven thousand, with at least ninety-five contributors and courses at fifty or more universities.

**Ava:** They’re clear that this wasn’t a clever growth campaign.

**Brian:** Right. One person with reach shared it, after five years of quiet accumulation. Andrea now maintains it from ETH Zurich, after it was built at Harvard. Moving a curriculum between institutions is a real test of whether it serves people beyond its original classroom.

**Ava:** The article gives four lessons for other open curricula. Let’s walk through them.

**Brian:** First, match the production API exactly. Every hour learners spend translating between a teaching API and the real one is wasted effort.

**Ava:** Second, choose the hardware floor before the feature list.

**Brian:** TinyTorch targets a dual-core two-gigahertz CPU, four gigabytes of RAM, and no network during training. It ships two tiny offline datasets: about one thousand grayscale digits and three hundred fifty conversational question-answer pairs, together under fifty megabytes.

**Ava:** That choice excludes GPU support, but it enables Chromebook classrooms and old laptops.

**Brian:** Yes. Decide whom you’re willing to exclude before the project grows, because revisiting that decision gets expensive.

**Ava:** Third, slow can be educational.

**Brian:** The ninety-seven-second convolution teaches the need for vectorization better than a warning ever could.

**Ava:** And fourth, instructor infrastructure is the real bottleneck.

**Brian:** Good content alone doesn’t get adopted. Gradeable content does. TinyTorch includes NBGrader autograding, locked test cells, point allocations, an INSTRUCTOR.md, milestone scripts, and three integration models for different course sizes.

**Ava:** So the instructor has to be supported as carefully as the learner.

**Brian:** Exactly. The instructor must defend the curriculum to a committee. If that path fails, teaching quality doesn’t matter.

**Ava:** There’s also a succession plan.

**Brian:** Yes. The project commits to maintenance through twenty twenty-seven, a two-week pull-request review target, and a governance transition during twenty twenty-six and twenty twenty-seven. Sustainability means making yourself replaceable, with a date attached.

**Ava:** What remains unknown?

**Brian:** TinyTorch is still in preview, aimed at classroom readiness for Fall twenty twenty-six. The authors have not measured learning outcomes. They have a strong educational rationale, but no controlled data showing that TinyTorch students debug production systems better than students in a conventional course.

**Ava:** And the technical scope has real holes.

**Brian:** It’s single-node and CPU-only. It teaches memory and compute, but not GPU kernels, distributed training, or gradient synchronization. Parallel data loading and GPU memory management were deliberately left out because doing them properly would break the four-gigabyte floor.

**Ava:** So what can people do next?

**Brian:** They can review modules against real PyTorch semantics, pilot a tier and report failures, write the missing distributed and GPU modules, localize the material, or adopt it and add their institution to the community map.

**Ava:** The code is MIT licensed, and the curriculum is CC BY-SA four point zero.

**Brian:** Right. Forking and adapting are explicitly allowed. The project also invites people to join the PyTorch Academic OSPO Working Group, which connects researchers, educators, students, and open-source practitioners.

**Ava:** Let’s close with the three points I’m taking away. First, building a small framework creates a mental model for the production one.

**Brian:** Second, matching the real API and keeping the hardware floor low makes the curriculum transferable.

**Ava:** Third, the project is promising but still needs evidence, instructors, and contributors for the parts it leaves out.

**Brian:** That’s TinyTorch: don’t just import PyTorch. Build enough of it to understand what happens when the abstraction breaks.

**Ava:** Thanks for listening. We’ll see you next time.

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
