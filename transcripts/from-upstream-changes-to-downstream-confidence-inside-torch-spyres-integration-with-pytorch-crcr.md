# When Upstream Changes: How Torch Spyre Builds Downstream Confidence

原文：[From Upstream Changes to Downstream Confidence: Inside Torch Spyre’s Integration with PyTorch CRCR](https://pytorch.org/blog/from-upstream-changes-to-downstream-confidence-inside-torch-spyres-integration-with-pytorch-crcr/)

## 摘要

本文介绍 Torch Spyre 如何基于 PyTorch 的跨仓库持续集成中继机制 CRCR，完成 L2 级别集成，并管理后端、PyTorch 核心和测试套件三个持续变化的对象。其智能体流水线结合代码分析与真实硬件执行结果筛选测试，将分类依据保存在可审查的配置中，再通过声明式 YAML 配置适配上游测试，无需修改测试源码。工作流通过测试拆分、构建产物复用、分层重试、正确配对回调和保存运行元数据，提高效率、可靠性与可复现性。文章也说明了合并标签存在时序问题、nightly 和 release 不触发 CRCR dispatch 等限制，并计划与社区合作推动通用组件进入上游。

## 对话

Ava: Your accelerator backend passed yesterday. Today, PyTorch changes, and something breaks.

Ava：昨天你的加速器后端还通过了测试。今天 PyTorch 一变，就出问题了。

How do you catch that early without spending your life sorting through thousands of tests?

怎样才能及早发现，又不用整天从成千上万个测试里筛查？

Brian: That's the problem Torch Spyre tackles here.

Brian：这正是 Torch Spyre 要解决的问题。

It's the PyTorch backend for the IBM Spyre Accelerator.

它是 IBM Spyre Accelerator 的 PyTorch 后端。

The goal is useful feedback when upstream changes, with enough evidence to investigate failures.

目标是在上游发生变化时提供有用的反馈，并留下足够的证据排查故障。

Ava: And the starting point is CRCR. Let's unpack that before we collect any more letters.

Ava：先从 CRCR 说起吧，别急着再加缩写。

Brian: CRCR means Cross-Repository CI Relay.

Brian：CRCR 是 Cross-Repository CI Relay，也就是跨仓库持续集成中继。

CI is continuous integration: automated builds and tests.

CI 指持续集成，也就是自动构建和测试。

The relay lets PyTorch trigger testing in downstream repositories and show results to upstream reviewers.

这个中继让 PyTorch 触发下游仓库的测试，并向上游审查者展示结果。

Ava: Downstream means a separate project that depends on PyTorch?

Ava：下游是指依赖 PyTorch 的独立项目？

Brian: Right. These are out-of-tree backends, maintained outside PyTorch's repository.

Brian：对。这些后端独立于 PyTorch 仓库，由各自的项目维护。

CRCR delivers notifications and brings results back to the PyTorch CI CRCR HUD, the shared results dashboard.

CRCR 负责发送通知，并将结果带回共享的 PyTorch CI CRCR HUD 结果看板。

Ava: So we've got a delivery service. Who decides what's inside the test package?

Ava：这么说，配送服务有了。测试包里装什么由谁决定？

Brian: The backend does.

Brian：由后端决定。

CRCR offers four integration levels, and Torch Spyre reached level two, called L2.

CRCR 提供四个集成级别，Torch Spyre 已达到第二级 L2。

But choosing meaningful tests and defining what counts as passing remain downstream responsibilities.

但挑选有意义的测试、定义通过标准，仍是下游项目的责任。

Ava: Why's choosing tests so complicated?

Ava：挑测试为什么这么复杂？

Can't we just test the latest backend against the latest PyTorch?

直接用最新版后端测试最新版 PyTorch 不行吗？

Brian: That's the primary combination, together with the latest test suite.

Brian：这是主要的测试组合，还要配上最新版测试套件。

There are three moving targets: backend code, PyTorch core, and tests.

有三个不断变化的部分：后端代码、PyTorch 核心和测试。

A test change can expose a problem too.

测试本身的改动也可能暴露问题。

Ava: Wait, so a new failure doesn't automatically mean the latest PyTorch change caused it?

Ava：等等，新出现的失败不一定是最新版 PyTorch 的改动造成的？

Brian: Exactly.

Brian：没错。

If only upstream changed since the last passing run, that points toward upstream.

如果从上次通过测试以来只有上游变了，问题就可能出在上游。

If several things moved, you need more investigation. Recording the combination matters.

如果几个部分都变了，就得进一步排查。所以要记录测试时的版本组合。

Ava: Okay. We know which versions we're testing.

Ava：好，我们知道要测哪些版本了。

How do we choose from tens of thousands of tests?

几万个测试里，该怎么选？

Brian: They use an agentic pipeline, meaning agents perform the selection work.

Brian：他们采用智能体流程，由智能体完成筛选。

It has four stages. First, an agent narrows the search to relevant folders and files.

流程分四步。第一步，智能体把搜索范围缩小到相关目录和文件。

Ava: Relevant according to what?

Ava：相关性怎么判断？

Brian: The backend's extension points, where it connects to PyTorch, plus plain-language scope preferences.

Brian：看后端与 PyTorch 对接的扩展点，再结合用自然语言描述的测试范围偏好。

If automatic differentiation, or autograd, isn't in scope yet, that whole area can be excluded.

如果自动微分，也就是 autograd，暂时不在范围内，就可以排除整个相关领域。

Ava: That gets us a smaller library. We still need a way to find the right books.

Ava：这样书库小了，但还得找到对的书。

Brian: That's stage two: repository memory.

Brian：第二步就是建立仓库记忆。

It builds a searchable index with symbols, files, summaries written by a large language model, and embeddings, numerical representations for finding related tests.

它建立可搜索的索引，包含符号、文件、大语言模型写的摘要，以及用于查找相关测试的数值表示，也就是嵌入向量。

Ava: So the next agent doesn't need the entire library squeezed into one prompt.

Ava：这样下一个智能体就不用把整个书库塞进一条提示词了。

Brian: Right.

Brian：对。

Stage three reads backend code, documentation, and metadata such as supported operators.

第三步，读取后端代码、文档，以及支持哪些算子等元数据。

It selects individual tests and classifies them as required to pass or skipped, recording reasons.

然后逐项选出测试，标记为必须通过或跳过，并记录原因。

Ava: And stage four checks whether those predictions survive contact with actual hardware?

Ava：第四步就是上真机，看看这些判断准不准？

Brian: Exactly. Selected tests run on real hardware.

Brian：没错。选中的测试会在真实硬件上运行。

The agent reviews execution logs and revises the classifications.

智能体查看运行日志，再调整分类。

Runtime failures and numerical differences can reveal things code analysis missed.

运行时故障和数值差异，可能揭示代码分析遗漏的问题。

Ava: That's the part I'd want to review.

Ava：这部分我会想亲自审核。

An agent deciding to skip tests could make a dashboard look very peaceful.

让智能体决定跳过哪些测试，看板可能会显得一片太平。

Brian: The reasons become comments in the configuration.

Brian：跳过原因会写成配置文件里的注释。

The agent proposes; the configuration is the reviewable record.

智能体提出建议，配置文件留下可供审核的记录。

The current output covers thousands of individually named test cases.

目前的输出涵盖数千个逐一命名的测试用例。

Ava: Let's make that concrete. I've just enabled a new operator. What happens next?

Ava：举个具体例子。我刚启用了一个新算子，接下来呢？

Brian: They build a mapping from tests to operators, plus a reverse index from each operator to its tests.

Brian：他们会建立测试到算子的映射，以及每个算子对应测试的反向索引。

Think of it as asking which recipes use a newly available ingredient.

可以把它想成：哪些菜谱会用到一种新食材？

Ava: But having that ingredient doesn't mean I can cook every recipe.

Ava：但有了这种食材，不代表每道菜我都能做。

Brian: Exactly. A test might need other unsupported features.

Brian：没错。测试可能还依赖其他不支持的功能。

The selection agent filters those candidates.

筛选智能体会过滤这些候选测试。

Operator usage comes from an execution pass using TorchDispatchMode for eager execution and TORCH_LOGS for compiled execution.

算子使用情况来自一次执行：即时执行用 TorchDispatchMode，编译执行用 TORCH_LOGS。

Ava: Eager execution runs operations directly. Compiled execution uses compilation.

Ava：即时执行直接运行操作，编译执行则先经过编译。

What about updating PyTorch itself?

那更新 PyTorch 本身呢？

Brian: They update repository memory and select from newly added or modified tests.

Brian：他们会更新仓库记忆，并从新增或修改过的测试中筛选。

Their example upgrade from PyTorch two point thirteen to two point fourteen evaluated about four thousand changed tests.

他们以 PyTorch 从 2.13 升级到 2.14 为例，评估了约 4000 个变更过的测试。

Ava: Instead of processing tens of thousands again. That's a smaller review pile.

Ava：不用再处理几万个测试了，待审的少多了。

But selected tests might still assume hardware features we don't have.

但选中的测试仍可能依赖我们没有的硬件功能。

Brian: That's where declarative configuration comes in: describing the adaptations you want.

Brian：这就要用声明式配置，写明需要怎样调整。

They use YAML, a structured text format, to adapt tests without editing upstream test files.

他们用结构化文本格式 YAML 调整测试，无须修改上游测试文件。

Ava: For example, a dtype, meaning a tensor's data type, isn't supported?

Ava：比如不支持某种 dtype，也就是张量的数据类型？

Brian: You can exclude that dtype while keeping the rest of the test.

Brian：可以排除这种 dtype，保留测试的其余部分。

You can also exclude shapes or add dtype coverage.

也可以排除某些形状，或增加 dtype 覆盖范围。

One unsupported parameter doesn't have to cost the entire test.

不支持一个参数，不必放弃整个测试。

Ava: And the configuration defines which outcomes are acceptable?

Ava：配置还会规定哪些结果可以接受？

Brian: Yes.

Brian：对。

There are four categories: must pass, expected failure, strict expected failure, and skip.

有四类：必须通过、预期失败、严格预期失败和跳过。

With strict expected failure, an unexpected pass is itself reported as a failure.

严格预期失败的测试如果意外通过，也会被报告为失败。

Ava: Passing gets me a failure? That's a tough performance review.

Ava：通过了反而算失败？这考核够严的。

Brian: It's a useful reminder to promote the test into the must-pass category.

Brian：这能提醒我们把它升级为“必须通过”的测试。

There's also an explicit default for unlisted tests, so newly added tests have a defined policy.

未列出的测试也有明确的默认规则，所以新增测试同样有章可循。

Ava: How does the framework apply all that without changing the source?

Ava：框架怎么做到这些，又不改源文件？

Brian: During test collection, it patches decorators, the helpers that generate test variations.

Brian：收集测试时，它会给生成测试变体的装饰器打补丁。

It adds pytest marks, labels used to select tests by operator or dtype.

它会添加 pytest 标记，用于按算子或 dtype 筛选测试。

The upstream test tree stays untouched.

上游测试目录保持原样。

Ava: Got it.

Ava：明白了。

Select tests from evidence, record the reasons, and adapt individual cases through configuration.

根据证据选测试、记录原因，再通过配置调整具体测试用例。

Now, when does testing actually start?

那么，测试什么时候真正开始？

Brian: A dispatch arrives with information about an upstream pull request, or PR.

Brian：系统收到一个 dispatch 事件，里面有上游拉取请求，也就是 PR 的信息。

The workflow checks its action, target branch, labels, and SHA, the identifier for a specific commit.

工作流会检查操作类型、目标分支、标签和 SHA；SHA 是特定提交的标识。

Ava: Checking whether a pull request merged sounds straightforward.

Ava：检查拉取请求是否合并，听起来很简单。

Brian: There's a wrinkle. PyTorchBot performs the merge and applies a Merged label.

Brian：这里有个细节：合并由 PyTorchBot 执行，它还会加上 Merged 标签。

A closed pull request alone isn't enough.

光是拉取请求已关闭还不够。

That label can arrive late, and manual merges may omit it.

这个标签可能延迟出现，手动合并也可能没有它。

Ava: So critical workflows may need polling or a fallback.

Ava：所以关键工作流可能需要轮询或备用判断方式。

What about nightly builds and releases?

那每夜构建和正式发布呢？

Brian: Neither gets a CRCR dispatch.

Brian：两者都不会收到 CRCR dispatch 事件。

Nightly tests run on a schedule and resolve the source commit.

每夜测试按计划运行，并确定对应的源代码提交。

Release tests start manually.

发布版本的测试则手动启动。

HUD accepts nightly results but currently lacks a dedicated release view.

HUD 能接收每夜测试结果，但目前没有专门的发布版本视图。

Ava: Once testing starts, how do they avoid one enormous job?

Ava：测试开始后，怎么避免变成一个庞大的任务？

Brian: They group tests by feature, split groups to meet a duration limit, then balance individual tests using measured timings.

Brian：他们先按功能分组，再拆分到规定时长内，最后根据实测耗时平衡各测试。

Thirty minutes is an example limit, not a reported speedup.

30 分钟只是时长上限的例子，不是报告中的提速数据。

Ava: And each split builds PyTorch again?

Ava：每个拆分任务都要重新构建 PyTorch 吗？

Brian: They build PyTorch and backend wheels, the installable packages, once.

Brian：他们只构建一次 PyTorch 和后端的 wheel 安装包。

Every split uses those same artifacts.

每个拆分任务都使用同一批构建产物。

That saves repeated builds and keeps the tested software consistent.

这样既省去重复构建，也保证测试使用的软件一致。

Ava: What keeps temporary infrastructure trouble from drowning out real regressions?

Ava：怎样避免临时的基础设施故障掩盖真正的回归问题？

Brian: Isolated jobs, retries at workflow and job levels, and logs that help classify failures.

Brian：隔离作业，在工作流和作业层面设置重试，并用日志帮助判断故障类型。

Non-critical steps can continue after errors.

非关键步骤出错后可以继续执行。

The article doesn't provide a complete failure-classification algorithm.

文章没有给出完整的故障分类算法。

Ava: There's also a reporting trap with matrix jobs, right?

Ava：矩阵作业的报告也有个陷阱，对吧？

Those are jobs generated from combinations of settings.

矩阵作业是由不同设置组合生成的作业。

Brian: Yes. Each matrix job has its own check identifier.

Brian：对。每个矩阵作业都有自己的检查标识符。

Its in-progress and completed callbacks, the status reports back to CRCR, must come from the same job.

它向 CRCR 报告状态时，进行中和已完成的回调必须来自同一个作业。

Ava: What are the options?

Ava：有哪些办法？

Brian: Each matrix job can report separately, which may crowd the dashboard.

Brian：每个矩阵作业可以分别报告，但可能挤满仪表板。

Or a parent job can dispatch a separate matrix workflow and poll until it finishes.

也可以由父作业启动单独的矩阵工作流，并轮询直到它结束。

Ava: And when someone asks what failed two weeks ago?

Ava：如果有人问两周前哪里出了错呢？

Brian: Save the dispatch payload, package versions or commits, runtime environment, and individual test results with durations.

Brian：保存启动参数、软件包版本或提交记录、运行环境，以及每项测试的结果和耗时。

Keep the build artifacts too.

构建产物也要保留。

Those records support reproduction and tracking down the responsible change.

这些记录有助于复现问题，追查是哪次改动造成的。

Ava: What's the demonstrated result, and what's still ahead?

Ava：目前展示了什么成果，接下来还有什么计划？

Brian: Torch Spyre reached L2 integration.

Brian：Torch Spyre 已达到 L2 集成。

The framework targets privateuse1, PyTorch's generic device extension slot.

该框架面向 privateuse1，也就是 PyTorch 的通用设备扩展槽位。

They intend to work toward upstreaming reusable parts, with OpenReg, an in-tree reference backend, as a natural reference.

他们计划推动可复用部分合入上游，并以代码树内的参考后端 OpenReg 作为参照。

Ava: Three things to remember: CRCR handles coordination and reporting.

Ava：记住三点：CRCR 负责协调和报告。

Evidence and reviewable configuration determine meaningful coverage.

证据和可审查的配置决定了测试覆盖是否有意义。

Shared artifacts, reliable workflows, and saved metadata make failures easier to investigate.

共享产物、可靠的工作流和保存的元数据，让故障更容易排查。

Brian: Thanks for listening, and we'll see you next time.

Brian：感谢收听，我们下次见。

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
