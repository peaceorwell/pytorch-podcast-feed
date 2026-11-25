# When Upstream Changes: How Torch Spyre Builds Downstream Confidence

原文：[From Upstream Changes to Downstream Confidence: Inside Torch Spyre’s Integration with PyTorch CRCR](https://pytorch.org/blog/from-upstream-changes-to-downstream-confidence-inside-torch-spyres-integration-with-pytorch-crcr/)

## 摘要

本文介绍 Torch Spyre 如何基于 PyTorch 的跨仓库持续集成中继机制 CRCR，完成 L2 级别集成，并管理后端、PyTorch 核心和测试套件三个持续变化的对象。其智能体流水线结合代码分析与真实硬件执行结果筛选测试，将分类依据保存在可审查的配置中，再通过声明式 YAML 配置适配上游测试，无需修改测试源码。工作流通过测试拆分、构建产物复用、分层重试、正确配对回调和保存运行元数据，提高效率、可靠性与可复现性。文章也说明了合并标签存在时序问题、nightly 和 release 不触发 CRCR dispatch 等限制，并计划与社区合作推动通用组件进入上游。

## 对话

Ava: Your accelerator backend passed yesterday. Today, PyTorch changes, and something breaks. How do you catch that early without spending your life sorting through thousands of tests?

Ava：你的加速器后端昨天还通过了测试。今天，PyTorch 变了，某个地方出了问题。怎样才能及早发现，又不用整天在成千上万个测试中翻找？

Brian: That's the problem Torch Spyre tackles here. It's the PyTorch backend for the IBM Spyre Accelerator. The goal is useful feedback when upstream changes, with enough evidence to investigate failures.

Brian：这正是 Torch Spyre 在这里要解决的问题。它是 IBM Spyre Accelerator 的 PyTorch 后端。目标是在上游发生变化时提供有用的反馈，并留下足够的证据来调查失败原因。

Ava: And the starting point is CRCR. Let's unpack that before we collect any more letters.

Ava：起点是 CRCR。在继续收集更多缩写之前，我们先把它讲清楚。

Brian: CRCR means Cross-Repository CI Relay. CI is continuous integration: automated builds and tests. The relay lets PyTorch trigger testing in downstream repositories and show results to upstream reviewers.

Brian：CRCR 是 Cross-Repository CI Relay。CI 指持续集成，也就是自动构建和测试。这个中继机制让 PyTorch 能触发下游仓库的测试，并向上游的审核者展示结果。

Ava: Downstream means a separate project that depends on PyTorch?

Ava：下游是指依赖 PyTorch 的另一个独立项目？

Brian: Right. These are out-of-tree backends, maintained outside PyTorch's repository. CRCR delivers notifications and brings results back to the PyTorch CI CRCR HUD, the shared results dashboard.

Brian：对。这些是 out-of-tree 后端，维护工作在 PyTorch 仓库之外进行。CRCR 负责发送通知，并把结果带回 PyTorch CI CRCR HUD，也就是共享的结果仪表板。

Ava: So we've got a delivery service. Who decides what's inside the test package?

Ava：所以我们有了一个投递服务。谁来决定测试包里放什么？

Brian: The backend does. CRCR offers four integration levels, and Torch Spyre reached level two, called L2. But choosing meaningful tests and defining what counts as passing remain downstream responsibilities.

Brian：由后端决定。CRCR 提供四个集成级别，Torch Spyre 达到了第二级，叫作 L2。但选择有意义的测试，以及定义什么算通过，仍然是下游的责任。

Ava: Why's choosing tests so complicated? Can't we just test the latest backend against the latest PyTorch?

Ava：选测试为什么这么复杂？直接用最新的后端测试最新的 PyTorch 不行吗？

Brian: That's the primary combination, together with the latest test suite. There are three moving targets: backend code, PyTorch core, and tests. A test change can expose a problem too.

Brian：这是主要的组合，还要配上最新的测试套件。有三个不断变化的部分：后端代码、PyTorch 核心和测试。测试本身的变化也可能暴露问题。

Ava: Wait, so a new failure doesn't automatically mean the latest PyTorch change caused it?

Ava：等等，所以出现新的失败，并不自动意味着是最新的 PyTorch 改动造成的？

Brian: Exactly. If only upstream changed since the last passing run, that points toward upstream. If several things moved, you need more investigation. Recording the combination matters.

Brian：没错。如果从上次通过到现在只有上游发生了变化，线索就指向上游。如果有几个部分都变了，就需要进一步调查。所以记录具体的组合很重要。

Ava: Okay. We know which versions we're testing. How do we choose from tens of thousands of tests?

Ava：好。我们知道要测试哪些版本了。那怎么从数以万计的测试里挑选？

Brian: They use an agentic pipeline, meaning agents perform the selection work. It has four stages. First, an agent narrows the search to relevant folders and files.

Brian：他们使用一条智能体流水线，也就是由智能体完成筛选工作。它分为四个阶段。首先，一个智能体把搜索范围缩小到相关的文件夹和文件。

Ava: Relevant according to what?

Ava：依据什么判断相关？

Brian: The backend's extension points, where it connects to PyTorch, plus plain-language scope preferences. If automatic differentiation, or autograd, isn't in scope yet, that whole area can be excluded.

Brian：依据后端与 PyTorch 连接的扩展点，再加上用普通语言描述的范围偏好。如果自动微分，也就是 autograd，暂时不在范围内，就可以排除整个领域。

Ava: That gets us a smaller library. We still need a way to find the right books.

Ava：这样图书馆就小一些了。但我们仍需要找到对的书。

Brian: That's stage two: repository memory. It builds a searchable index with symbols, files, summaries written by a large language model, and embeddings, numerical representations for finding related tests.

Brian：这就是第二阶段：仓库记忆。它建立一个可搜索的索引，包含符号、文件、由大语言模型编写的摘要，以及 embeddings，也就是用于查找相关测试的数值表示。

Ava: So the next agent doesn't need the entire library squeezed into one prompt.

Ava：这样下一个智能体就不用把整个图书馆都塞进一条提示词里了。

Brian: Right. Stage three reads backend code, documentation, and metadata such as supported operators. It selects individual tests and classifies them as required to pass or skipped, recording reasons.

Brian：对。第三阶段读取后端代码、文档，以及受支持的算子等元数据。它选出单个测试，将其归类为必须通过或跳过，并记录原因。

Ava: And stage four checks whether those predictions survive contact with actual hardware?

Ava：第四阶段则检查这些预测能否经得起实际硬件的检验？

Brian: Exactly. Selected tests run on real hardware. The agent reviews execution logs and revises the classifications. Runtime failures and numerical differences can reveal things code analysis missed.

Brian：正是如此。选中的测试会在真实硬件上运行。智能体查看执行日志，并修订分类。运行时失败和数值差异可能揭示代码分析遗漏的问题。

Ava: That's the part I'd want to review. An agent deciding to skip tests could make a dashboard look very peaceful.

Ava：这部分我会想亲自审核。让智能体决定跳过哪些测试，可能会让仪表板看起来一片祥和。

Brian: The reasons become comments in the configuration. The agent proposes; the configuration is the reviewable record. The current output covers thousands of individually named test cases.

Brian：原因会写成配置中的注释。智能体提出建议；配置则是可供审核的记录。目前的输出涵盖了数千个逐一命名的测试用例。

Ava: Let's make that concrete. I've just enabled a new operator. What happens next?

Ava：说个具体的例子。我刚启用了一个新算子，接下来会怎样？

Brian: They build a mapping from tests to operators, plus a reverse index from each operator to its tests. Think of it as asking which recipes use a newly available ingredient.

Brian：他们建立从测试到算子的映射，还建立从每个算子反查其相关测试的索引。可以把它想成在问：哪些菜谱会用到一种刚刚有了的食材？

Ava: But having that ingredient doesn't mean I can cook every recipe.

Ava：但有了这种食材，不代表每道菜我都会做。

Brian: Exactly. A test might need other unsupported features. The selection agent filters those candidates. Operator usage comes from an execution pass using TorchDispatchMode for eager execution and TORCH_LOGS for compiled execution.

Brian：没错。一个测试可能还需要其他尚未支持的功能。筛选智能体会过滤这些候选测试。算子的使用情况来自一次执行过程：即时执行使用 TorchDispatchMode，编译执行使用 TORCH_LOGS。

Ava: Eager execution runs operations directly. Compiled execution uses compilation. What about updating PyTorch itself?

Ava：即时执行直接运行操作。编译执行则要经过编译。那更新 PyTorch 本身时怎么办？

Brian: They update repository memory and select from newly added or modified tests. Their example upgrade from PyTorch two point thirteen to two point fourteen evaluated about four thousand changed tests.

Brian：他们会更新仓库记忆，并从新增或修改过的测试中挑选。他们给出的升级例子是从 PyTorch 二点十三到二点十四，评估了约四千个发生变化的测试。

Ava: Instead of processing tens of thousands again. That's a smaller review pile. But selected tests might still assume hardware features we don't have.

Ava：这样就不用重新处理数以万计的测试了。要审核的那堆测试小多了。不过，选中的测试仍可能假定硬件具备我们没有的功能。

Brian: That's where declarative configuration comes in: describing the adaptations you want. They use YAML, a structured text format, to adapt tests without editing upstream test files.

Brian：这就是声明式配置派上用场的地方：描述你想怎样调整测试。他们用 YAML 这种结构化文本格式来调整测试，无须编辑上游测试文件。

Ava: For example, a dtype, meaning a tensor's data type, isn't supported?

Ava：比如某个 dtype，也就是张量的数据类型，不受支持？

Brian: You can exclude that dtype while keeping the rest of the test. You can also exclude shapes or add dtype coverage. One unsupported parameter doesn't have to cost the entire test.

Brian：你可以排除那个 dtype，同时保留测试的其余部分。也可以排除某些形状，或增加 dtype 的覆盖范围。不必因为一个参数不受支持，就舍弃整个测试。

Ava: And the configuration defines which outcomes are acceptable?

Ava：配置也会定义哪些结果可以接受？

Brian: Yes. There are four categories: must pass, expected failure, strict expected failure, and skip. With strict expected failure, an unexpected pass is itself reported as a failure.

Brian：是的。有四类：must pass（必须通过）、expected failure（预期失败）、strict expected failure（严格预期失败）和 skip（跳过）。在 strict expected failure 下，意外通过本身也会被报告为失败。

Ava: Passing gets me a failure? That's a tough performance review.

Ava：通过了反倒算失败？这绩效考核可真严格。

Brian: It's a useful reminder to promote the test into the must-pass category. There's also an explicit default for unlisted tests, so newly added tests have a defined policy.

Brian：这是个有用的提醒：该把这个测试提升到 must pass 类别了。对于没有列出的测试，也有明确的默认规则，因此新加的测试都有既定的处理方式。

Ava: How does the framework apply all that without changing the source?

Ava：框架怎样在不改源码的情况下应用这些配置？

Brian: During test collection, it patches decorators, the helpers that generate test variations. It adds pytest marks, labels used to select tests by operator or dtype. The upstream test tree stays untouched.

Brian：在收集测试时，它会给 decorators 打补丁；这些辅助工具负责生成测试的不同变体。它会添加 pytest marks，也就是用于按算子或 dtype 选择测试的标签。上游测试目录保持原样。

Ava: Got it. Select tests from evidence, record the reasons, and adapt individual cases through configuration. Now, when does testing actually start?

Ava：明白了。根据证据挑选测试，记录原因，再通过配置调整单个用例。那么，测试究竟什么时候开始？

Brian: A dispatch arrives with information about an upstream pull request, or PR. The workflow checks its action, target branch, labels, and SHA, the identifier for a specific commit.

Brian：工作流会收到一条 dispatch，其中包含上游 pull request，也就是 PR 的信息。工作流检查它的操作类型、目标分支、标签和 SHA；SHA 是某个特定提交的标识符。

Ava: Checking whether a pull request merged sounds straightforward.

Ava：检查一个 pull request 是否已合并，听起来很直接。

Brian: There's a wrinkle. PyTorchBot performs the merge and applies a Merged label. A closed pull request alone isn't enough. That label can arrive late, and manual merges may omit it.

Brian：这里有个麻烦之处。PyTorchBot 会执行合并并加上 Merged 标签。仅仅关闭 pull request 还不够。这个标签可能来得晚，手动合并也可能漏掉它。

Ava: So critical workflows may need polling or a fallback. What about nightly builds and releases?

Ava：所以关键工作流可能需要轮询或备用方案。那每日构建和正式发布怎么办？

Brian: Neither gets a CRCR dispatch. Nightly tests run on a schedule and resolve the source commit. Release tests start manually. HUD accepts nightly results but currently lacks a dedicated release view.

Brian：两者都不会收到 CRCR dispatch。每日测试按计划运行，并确定对应的源代码提交。发布测试则要手动启动。HUD 接受每日测试结果，但目前没有专门的发布视图。

Ava: Once testing starts, how do they avoid one enormous job?

Ava：测试开始后，他们怎样避免把所有东西塞进一个巨大的任务？

Brian: They group tests by feature, split groups to meet a duration limit, then balance individual tests using measured timings. Thirty minutes is an example limit, not a reported speedup.

Brian：他们先按功能给测试分组，再把各组拆分到符合时长限制的大小，然后根据测得的耗时平衡各个测试。三十分钟只是时长限制的例子，不是报告的提速幅度。

Ava: And each split builds PyTorch again?

Ava：每个拆分出来的部分都要重新构建 PyTorch 吗？

Brian: They build PyTorch and backend wheels, the installable packages, once. Every split uses those same artifacts. That saves repeated builds and keeps the tested software consistent.

Brian：他们只构建一次 PyTorch 和后端的 wheels，也就是可安装的软件包。每个拆分部分都使用这些相同的构建产物。这既省去了重复构建，也确保测试的软件版本一致。

Ava: What keeps temporary infrastructure trouble from drowning out real regressions?

Ava：怎样避免临时性的基础设施问题淹没真正的回归问题？

Brian: Isolated jobs, retries at workflow and job levels, and logs that help classify failures. Non-critical steps can continue after errors. The article doesn't provide a complete failure-classification algorithm.

Brian：使用相互隔离的作业，在工作流和作业两个层级进行重试，并保留有助于判断故障类型的日志。非关键步骤出错后可以继续执行。文章没有给出完整的故障分类算法。

Ava: There's also a reporting trap with matrix jobs, right? Those are jobs generated from combinations of settings.

Ava：矩阵作业在报告状态时也有个陷阱，对吧？这些作业是由不同设置的组合生成的。

Brian: Yes. Each matrix job has its own check identifier. Its in-progress and completed callbacks, the status reports back to CRCR, must come from the same job.

Brian：对。每个矩阵作业都有自己的检查标识符。它向 CRCR 报告状态时，进行中和已完成这两种回调必须来自同一个作业。

Ava: What are the options?

Ava：有哪些选择？

Brian: Each matrix job can report separately, which may crowd the dashboard. Or a parent job can dispatch a separate matrix workflow and poll until it finishes.

Brian：可以让每个矩阵作业分别报告，但这可能让仪表板变得拥挤。也可以让一个父作业触发独立的矩阵工作流，然后轮询直到它完成。

Ava: And when someone asks what failed two weeks ago?

Ava：那如果有人问两周前出了什么故障呢？

Brian: Save the dispatch payload, package versions or commits, runtime environment, and individual test results with durations. Keep the build artifacts too. Those records support reproduction and tracking down the responsible change.

Brian：保存触发时的载荷、软件包版本或提交记录、运行环境，以及每项测试的结果和耗时。构建产物也要保留。这些记录有助于复现问题，并追查是哪个变更导致了故障。

Ava: What's the demonstrated result, and what's still ahead?

Ava：已经展示了什么成果，接下来还有什么工作？

Brian: Torch Spyre reached L2 integration. The framework targets privateuse1, PyTorch's generic device extension slot. They intend to work toward upstreaming reusable parts, with OpenReg, an in-tree reference backend, as a natural reference.

Brian：Torch Spyre 已达到 L2 集成。该框架面向 privateuse1，也就是 PyTorch 的通用设备扩展槽位。他们计划逐步将可复用的部分贡献到上游，而 OpenReg 这个源码树内的参考后端是一个自然的参照。

Ava: Three things to remember: CRCR handles coordination and reporting. Evidence and reviewable configuration determine meaningful coverage. Shared artifacts, reliable workflows, and saved metadata make failures easier to investigate.

Ava：有三件事值得记住：CRCR 负责协调和报告；证据和可供审查的配置决定了覆盖范围是否有意义；共享的构建产物、可靠的工作流和保存下来的元数据，让故障更容易调查。

Brian: Thanks for listening, and we'll see you next time.

Brian：感谢收听，我们下次再见。

## 术语

| Term | 释义 |
|---|---|
| Cross-Repository CI Relay (CRCR) | 跨仓库持续集成中继机制，用于触发下游测试并将结果反馈给 PyTorch。 |
| out-of-tree (OOT) | 树外；指在 PyTorch 主仓库之外维护的后端或扩展。 |
| HUD | 文章中集中展示 PyTorch CI 与 CRCR 测试结果的看板。 |
| agentic pipeline | 由智能体执行筛选、分析和结果修正等步骤的流水线。 |
| Repository Memory Generator | 仓库记忆生成器，将测试信息组织为可查询的索引。 |
| TorchDispatchMode | 文中用于在 eager 执行路径中采集测试所涉及算子信息的机制。 |
| TORCH_LOGS | 文中用于在编译执行路径中采集算子相关信息的日志配置。 |
| mandatory_success | 必须通过的测试分类。 |
| xfail_strict | 严格预期失败分类；测试意外通过也会被视为失败，以提示更新分类。 |
| unlisted_test_mode | 配置中未明确列出的测试所采用的默认处理策略。 |
| privateuse1 | PyTorch 提供的通用设备扩展槽位，该测试复用框架以其为目标。 |
| matrix strategy | 矩阵策略，根据变量组合生成多个执行任务。 |
| check_run_id | 检查运行的标识符；CRCR 用它跟踪重试，每个矩阵任务有独立标识符。 |
| workflow artifacts | 工作流保存的产物，如安装包、运行元数据和测试结果。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's unpack that | 我们来把这个说清楚。 |
| Wait, so | 等等，这么说…… |
| Let's make that concrete. | 我们举个具体情境来说明。 |
| That's the part I'd want to review. | 这正是我想仔细审查的部分。 |
| Got it. | 明白了。 |
| There's a wrinkle. | 这里有个需要注意的小问题。 |
| What are the options? | 有哪些可选方案？ |
| What's still ahead? | 接下来还有哪些工作？ |

## 中文内容总结

PyTorch 的跨仓库 CI 中继 CRCR 将上游变更发送给树外加速器仓库，并把下游结果呈现在 CI HUD 中。Torch Spyre 已接入到 L2 级别。文章重点解释后端如何选择有意义的测试、复用上游测试，以及让结果可靠且可追溯。

### 测试范围与版本组合

回归可能来自 Torch Spyre 后端、PyTorch 核心或持续变化的测试套件。主要验证组合是调度发生时三者的最新版本；后端也可以把其他版本组合分别报告给 HUD，以检查兼容性。哪些调度值得构建、哪些测试必须通过，仍由后端决定。

### 自动选择测试

选择流程先根据后端使用的 PyTorch 扩展点和范围要求缩小测试目录，再建立包含符号、文件摘要和单项测试向量的仓库索引。第二个代理结合后端代码、文档及算子元数据，把具体测试归入必须通过或跳过，并把理由写进配置。测试随后在真实硬件上运行，执行日志用于修正静态判断遗漏的运行时和数值问题。当前配置覆盖数千个具名测试用例。启用新算子时，流程用执行期间采集的测试与算子关系寻找候选，再过滤依赖其他未支持功能的用例。作者举例称，从 PyTorch 2.13 升至 2.14 时只需评估约 4000 个变化的测试。

### 不修改上游测试的适配方式

YAML 配置可按测试用例调整数据类型、形状等参数，并指定必须通过、预期失败、严格预期失败或跳过。严格预期失败会把意外通过也标为失败，提醒维护者更新分类。配置还声明未列出测试的默认处理方式及支持的算子和数据类型。框架在收集阶段适配上游装饰器并生成 pytest 标记，因此可以按算子筛选测试，同时保持 PyTorch 测试源码不变。

### 调度与可靠性

工作流用调度载荷中的提交 SHA、PR、动作、目标分支和标签筛选任务；判断 PR 是否落入主分支时需检查“关闭”动作及 PyTorchBot 添加的 Merged 标签。标签可能晚于调度到达。夜间版没有 CRCR 调度，需定时解析源码 SHA；发布版需手动触发，目前 HUD 没有专门的发布版结果视图。大量测试按功能、时长和单项耗时拆成并行任务，共用一次构建的 wheel。隔离任务、分类失败及分层重试有助于减少临时故障干扰。

### 结果追踪与适用范围

CRCR 要求进行中和完成回调来自同一任务；矩阵中的每个任务有独立的 check_run_id，可分别回调，或由父任务触发另一工作流并轮询结果。保存调度载荷、构建版本、运行环境及逐项测试结果，有助于日后复现和定位失败。作者表示这些机制面向通用 privateuse1 设备，并计划与 PyTorch 社区讨论复用和上游合入；文章没有给出加速幅度或回归检出率的量化结果。
