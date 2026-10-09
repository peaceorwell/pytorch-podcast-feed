# How New Hardware Finds Its Way into PyTorch

原文：[PyTorch Hardware Enablement: Updates from the Accelerator Integration Working Group](https://pytorch.org/blog/pytorch-hardware-enablement-updates-from-the-acceleration-integration-working-group/)

## 摘要

本期讨论 PyTorch 加速器集成工作组在二〇二六年上半年的进展，重点是通过统一接口、共享测试和参考实现降低新硬件接入成本。跨仓库持续集成机制 CRCR 将下游测试反馈带回上游，而测试重构让更多测试能够跨后端复用。OpenReg 及其相关工作展示了性能分析、分布式通信和编译器集成的关键路径，但这些参考实现并不以生产性能为目标。下半年的方向包括继续减少未分类测试、扩大覆盖范围、完善自动测试选择，并鼓励更多厂商申请加入官方 Additional Platforms 页面。

## 对话

Ava: You’ve built a new accelerator. It runs your model.

Ava：你造了一款新加速器，模型也跑起来了。

Then PyTorch changes, and you’re back fixing patches. How do we make that less painful?

可 PyTorch 一更新，你又得修补丁。怎么才能省点力？

Brian: That’s the problem behind this article.

Brian：这正是这篇文章要解决的问题。

The Accelerator Integration Working Group wants clearer integration paths, shared tests, and earlier feedback when upstream changes break something.

加速器集成工作组希望明确集成路径、共用测试，并在上游改动造成问题时更早收到反馈。

Ava: I’m Ava.

Ava：我是 Ava。

I’m a software engineer outside the PyTorch core team, and maintaining a pile of patches sounds painfully familiar.

我不是 PyTorch 核心团队的工程师，但维护一堆补丁的痛苦，我太熟悉了。

Brian: And I’m Brian.

Brian：我是 Brian。

Today we’re looking at the group’s progress in the first half of twenty twenty-six, from testing to compiler support.

今天我们来看看工作组在 2026 年上半年的进展，从测试到编译器支持。

Ava: Let’s start with the motivation. Why does hardware integration need a working group?

Ava：先说说初衷。为什么硬件集成需要一个工作组？

Brian: More compute platforms are adopting PyTorch.

Brian：越来越多的计算平台开始采用 PyTorch。

Each backend, meaning the implementation connecting hardware to the framework, needs reliable ways to fit into the same system.

每个后端，也就是连接硬件与框架的实现，都需要可靠的方式接入同一套系统。

Ava: Like different appliances needing a standard socket, instead of everyone opening the wall and rewiring it?

Ava：就像不同电器都该用标准插座，不用每家都拆墙改线？

Brian: Exactly.

Brian：没错。

The group works on vendor-neutral mechanisms, shared tools, and reference implementations.

工作组致力于建立不依赖特定厂商的机制、共用工具和参考实现。

Those help hardware teams avoid maintaining their own framework patches.

这样硬件团队就不用各自维护框架补丁了。

Ava: Where does the feedback problem come in?

Ava：反馈问题又是怎么回事？

Brian: PyTorch’s upstream repository, where framework development happens, has extensive continuous integration, or CI.

Brian：PyTorch 的上游代码库是框架开发的地方，那里有完善的持续集成，也就是 CI。

That’s automated testing of changes. But dependent projects run tests elsewhere.

CI 会自动测试代码改动，但依赖 PyTorch 的项目在别处运行测试。

Ava: Those are the downstream projects.

Ava：这些就是下游项目。

Hardware backends and libraries that depend on PyTorch?

比如依赖 PyTorch 的硬件后端和库？

Brian: Right. They lacked a standard way to receive change notifications and return results.

Brian：对。之前没有标准方式向它们发送变更通知并收回测试结果。

Contributors couldn’t easily see downstream damage before merging a change.

贡献者很难在合并改动前发现对下游的影响。

Ava: So the warning could arrive after the change was already in?

Ava：所以等收到警报，改动可能已经合并了？

Brian: Yes. Cross-Repository CI Relay, or CRCR, connects those separate testing systems.

Brian：是的。跨仓库 CI 中继系统（Cross-Repository CI Relay，简称 CRCR）把这些独立的测试系统连接起来。

A pull request, meaning a proposed code change, or a pushed commit triggers notifications.

拉取请求，也就是提议的代码改动，或推送的提交，都会触发通知。

Ava: And every registered project gets its own chance to test?

Ava：每个已注册的项目都能自己测试？

Brian: They’re notified in parallel.

Brian：它们会同时收到通知。

Each repository runs its own workflow, then sends its status back.

每个代码库运行自己的工作流，再把状态报回来。

Results appear in PyTorch’s shared CI dashboard within seconds.

结果上报后，几秒内就会出现在 PyTorch 的共享 CI 看板上。

Ava: Within seconds of reporting, though.

Ava：是上报后几秒内。

That doesn’t mean a huge test suite finishes in seconds.

这可不是说庞大的测试套件几秒就能跑完。

Brian: Right. The article describes fast result delivery.

Brian：对。文章说的是结果传递得快。

It doesn’t promise that the tests themselves run that quickly.

并没说测试本身也能跑得那么快。

Ava: How does PyTorch know who’s reporting?

Ava：PyTorch 怎么知道是谁在上报？

A green check needs a little more evidence than, trust me.

要给绿色通过标记，总不能只凭一句“相信我”吧。

Brian: GitHub OpenID Connect, or OIDC, tokens verify the calling repository’s identity.

Brian：GitHub OpenID Connect（OIDC）令牌会验证调用方代码库的身份。

There are also authorization checks, rate limits, and checks on valid reporting state.

此外还有权限检查、速率限制，以及对上报状态是否有效的检查。

Ava: Does joining immediately give a downstream project power to block changes?

Ava：下游项目一加入，就能阻止改动合并吗？

Brian: No.

Brian：不能。

Four participation levels move from notifications through dashboard reporting to non-blocking and eventually blocking checks.

参与分为四级：从接收通知、在看板展示结果，到不阻止合并的检查，最后才是能阻止合并的检查。

Trusted information stays separate from self-reported data.

可信信息与项目自行上报的数据也会分开。

Ava: Got it. Shared visibility, with controlled participation.

Ava：明白了。大家都能看到结果，参与权限也受到控制。

But teams still need useful tests to run.

但团队还得有用得上的测试。

Brian: Exactly. PyTorch has over six hundred thousand test cases.

Brian：没错。PyTorch 有超过 60 万个测试用例。

Many were tied to particular hardware through fixed device names or backend-specific operations.

其中很多通过写死的设备名称或特定后端操作，绑定到了某类硬件。

Ava: So a test might check a general feature, but its setup only lets one accelerator through the door?

Ava：也就是说，测试检查的可能是通用功能，但测试配置只允许一种加速器运行？

Brian: That’s the issue. Other vendors had to patch tests.

Brian：问题就在这儿。其他厂商只好给测试打补丁。

The refactoring replaces fixed hardware references with device parameters, so the same logic can serve multiple backends.

这次重构用设备参数替代写死的硬件引用，让同一套测试逻辑适用于多个后端。

Ava: How much progress did they report?

Ava：他们报告了多少进展？

Brian: Contributors migrated over two hundred seventy-six test files in the first half of the year.

Brian：今年上半年，贡献者迁移了超过 276 个测试文件。

They covered areas including profiling, neural network modules, and optimizers.

他们涉及的领域包括性能分析、神经网络模块和优化器。

Ava: But some tests really are hardware-specific. You can’t make every test universal.

Ava：但有些测试确实依赖特定硬件，不可能让所有测试都通用。

Brian: Right.

Brian：对。

Classes fall into three groups: accelerator-unrelated, accelerator-agnostic, and accelerator-specific.

测试类分为三类：与加速器无关、适用于不同加速器，以及特定加速器专用。

In plain English, that’s no accelerator needed, reusable across devices, or tied to one backend.

简单说，就是不需要加速器、可跨设备复用，或绑定某个后端。

Ava: And that classification helps select the right tests?

Ava：这样分类有助于选对测试？

Brian: Yes. The metadata and selection flag work across the test execution paths.

Brian：是的。元数据和选择标志适用于各条测试执行路径。

A linter, an automated rule checker, requires new test classes to declare their classification.

代码检查工具会要求新的测试类声明所属类别。

Ava: What happens to all the old ones?

Ava：那已有的测试类怎么办？

Brian: They’re being handled gradually.

Brian：正在逐步处理。

At launch, one thousand one hundred ninety-one unclassified files were on an allowlist.

推出时，允许名单上有 1191 个尚未分类的文件。

Reducing that list is ongoing work.

缩减这份名单仍在进行。

Ava: What if my backend only supports some features?

Ava：如果我的后端只支持部分功能呢？

Brian: There’s fine-grained skipping by feature, class, test, and operator.

Brian：可以按功能、类、测试和算子精细地跳过测试。

Vendors can select what matches their implementation without changing upstream test code.

厂商无需修改上游测试代码，就能选择符合自身实现的测试。

Ava: So we’ve got reusable tests and clearer selection.

Ava：这样测试能复用，选择方式也更清晰了。

Now, where do developers learn the actual integration patterns?

那么，开发者去哪里学习具体的集成方式？

Brian: That’s OpenReg’s role.

Brian：这正是 OpenReg 的作用。

It’s a minimal reference backend built on PrivateUse1, PyTorch’s mechanism for integrating custom accelerator backends.

它是基于 PrivateUse1 构建的最小参考后端；PrivateUse1 是 PyTorch 集成自定义加速器后端的机制。

Ava: Does OpenReg need special hardware?

Ava：OpenReg 需要专用硬件吗？

Brian: No. It’s backed by the central processing unit, or CPU.

Brian：不需要。它由中央处理器，也就是 CPU，提供支持。

It lives inside PyTorch’s source tree and demonstrates integration mechanics without real accelerator complexity.

它位于 PyTorch 源码树中，展示集成机制，无需涉及真实加速器的复杂性。

Ava: Like a teaching engine with the important parts visible.

Ava：就像一台能看清关键部件的教学用引擎。

You wouldn’t enter it in a race.

但不会拿它去参加比赛。

Brian: Exactly. It isn’t a production backend.

Brian：没错。它不是生产用后端。

It demonstrates device registration, operator dispatch, meaning routing operations to implementations, and runtime behavior such as streams and events.

它展示设备注册、算子分派，即将操作交给相应实现，以及流和事件等运行时行为。

Ava: Streams organize device work, and events help track its progress?

Ava：流负责组织设备上的工作，事件帮助跟踪进度？

Brian: That’s the basic idea.

Brian：基本就是这样。

OpenReg also runs in CI, helping catch breaks in PrivateUse1 integration.

OpenReg 也在持续集成（CI）中运行，帮助发现 PrivateUse1 集成中的问题。

Its implementation is kept aligned with the integration documentation.

它的实现会与集成文档保持一致。

Ava: Let’s talk profiling. Knowing an operation took a long time isn’t always enough.

Ava：说说性能分析。只知道某个操作耗时很长，有时还不够。

Brian: Right.

Brian：对。

Developers also need timelines for kernels, the device computations, plus memory and runtime events.

开发者还需要内核，也就是设备计算的时间线，以及内存和运行时事件的时间线。

They need to connect host activity with device activity.

他们需要把主机活动与设备活动对应起来。

Ava: Like matching an order placed at the counter with the work happening in the kitchen?

Ava：就像把柜台接到的订单和厨房里的工作对应起来？

Brian: Yes. Correlation IDs provide that connection.

Brian：对。关联 ID 就能建立这种联系。

OpenReg’s reference profiling stack demonstrates those IDs, activity types, and the lifecycle of a profiling session.

OpenReg 的参考性能分析组件展示了这些 ID、活动类型，以及一次性能分析会话的生命周期。

Ava: Is it a complete profiler?

Ava：它是完整的性能分析器吗？

Brian: No. It’s a stub, a minimal implementation showing the connections.

Brian：不是。它只是一个桩实现，用最少的代码展示各部分如何连接。

It plugs into Kineto, the profiling system used here, through PyTorch’s registration interface.

它通过 PyTorch 的注册接口接入这里使用的性能分析系统 Kineto。

Ava: Then vendors replace the demonstration pieces with their own device profiling calls?

Ava：然后厂商用自己的设备性能分析调用替换这些演示部分？

Brian: Exactly, without modifying PyTorch or Kineto.

Brian：正是如此，无需修改 PyTorch 或 Kineto。

Tests cover registration, session lifecycle, and fallback behavior.

测试覆盖注册、会话生命周期和回退行为。

The older path remains available for coarse operator-level timing.

旧路径仍可用于粗粒度的算子级计时。

Ava: That’s a clearer starting point. Does the same approach extend to distributed training?

Ava：这样入门就清楚多了。同样的方法也适用于分布式训练吗？

Brian: Yes. The group began building OCCL, the OpenReg Collective Communications Library.

Brian：是的。团队开始构建 OCCL，即 OpenReg 集合通信库。

It’s a minimal reference for a custom backend in c10d, PyTorch’s distributed infrastructure.

它是为 PyTorch 分布式基础设施 c10d 中的自定义后端提供的最小参考实现。

Ava: What does it actually demonstrate?

Ava：它具体展示了什么？

Brian: Registering a ProcessGroup, which represents participating processes, dispatching collectives, meaning coordinated communication operations, and handling Work completion, meaning when an operation is considered finished.

Brian：注册代表参与进程的 ProcessGroup、分派集合通信操作，以及处理 Work 的完成状态，也就是何时将操作视为完成。

Ava: So again, the result is a clearer integration contract.

Ava：所以，这同样让集成约定更清晰了。

We’re not hearing a communication speed claim.

我们听到的并不是通信速度方面的承诺。

Brian: Correct. Production performance isn’t OCCL’s target.

Brian：没错。OCCL 的目标不是生产环境的性能。

Related work also improved support for Python-based distributed backends and device-generic distributed test selection.

相关工作还改进了对基于 Python 的分布式后端，以及设备通用的分布式测试筛选的支持。

Ava: How about compilation? That’s another big surface to connect.

Ava：编译方面呢？这也是需要打通的重要环节。

Brian: The OpenReg example shows integration with torch.

Brian：OpenReg 示例展示了如何接入 torch。

compile, PyTorch’s compilation entry point.

compile，也就是 PyTorch 的编译入口。

Dynamo handles graph capture and device management.

Dynamo 负责图捕获和设备管理。

Graph capture records operations for compilation.

图捕获会记录操作，供编译使用。

Ava: And where does Inductor fit?

Ava：Inductor 在其中负责什么？

Brian: Inductor offers scheduling, fusion, and code generation.

Brian：Inductor 提供调度、融合和代码生成能力。

Fusion combines operations into kernels.

融合会把多个操作合并成内核。

Vendors can optionally use those optimization passes, and OpenReg demonstrates the integration points.

厂商可以选择使用这些优化流程；OpenReg 展示了相应的集成接口。

Ava: Including fused kernel generation without changing upstream PyTorch?

Ava：包括无须修改上游 PyTorch，就能生成融合内核？

Brian: Yes. There’s also a compiler integration guide.

Brian：对。另外还有一份编译器集成指南。

Again, the focus is showing how the pieces connect, rather than reporting production performance.

重点仍是展示各部分如何衔接，而不是报告生产环境的性能。

Ava: What’s still ahead?

Ava：接下来还有哪些工作？

Brian: More test coverage, fewer unclassified tests, and connecting classification to automatic CI scheduling.

Brian：扩大测试覆盖范围，减少未分类测试，并将分类结果用于 CI 自动调度。

They’re also exploring a registry where accelerators declare supported operator data types and precision overrides.

他们也在探索一个注册机制，让加速器声明支持的算子数据类型和精度覆盖设置。

Ava: And once a platform’s integrated, how do users find it?

Ava：平台集成后，用户怎么找到它？

Brian: The official Additional Platforms page links from PyTorch’s install page.

Brian：PyTorch 的安装页面链接到了官方的 Additional Platforms 页面。

Admission requires evidence and review.

平台要获收录，必须提供证明材料并通过审核。

The group’s encouraging more vendors to apply in the second half.

该小组鼓励更多厂商在下半年申请。

Ava: Let’s recap three points.

Ava：我们总结三点。

First, CRCR brings downstream test feedback into a shared upstream view.

第一，CRCR 将下游的测试反馈汇集到上游的共享视图中。

Brian: Second, test refactoring helps backends reuse validation.

Brian：第二，测试重构让后端能够复用验证流程。

Third, OpenReg provides concrete integration references, while production readiness still requires further work.

第三，OpenReg 提供了具体的集成参考，但要达到生产可用仍需进一步工作。

Ava: Thanks for listening. We’ll see you next time.

Ava：感谢收听，我们下期再见。

## 术语

| Term | 释义 |
|---|---|
| Accelerator Integration Working Group | 加速器集成工作组，推动新硬件接入 PyTorch 的标准机制、共享工具和参考实现。 |
| Cross-Repository CI Relay | 跨仓库持续集成中继机制，向下游仓库分发上游变更通知并汇集测试结果。 |
| OIDC | OpenID Connect；文中用于验证回报测试状态的 GitHub 仓库身份。 |
| device-agnostic | 设备无关的；通过设备参数等机制让同一套逻辑适用于不同后端。 |
| hardware classification | 硬件分类，用于标明测试与加速器的关系及所需硬件类型。 |
| PrivateUse1 | PyTorch 为自定义加速器后端提供的集成机制。 |
| OpenReg | PyTorch 源码树内基于 CPU 的最小参考后端，用于展示和验证 PrivateUse1 集成路径。 |
| correlation IDs | 关联标识，用于在性能分析中连接主机活动与设备活动。 |
| Kineto | 文中承接自定义后端性能分析插件的性能分析系统。 |
| OCCL | OpenReg Collective Communications Library，展示分布式后端集成要点的最小参考通信库。 |
| ProcessGroup | 进程组，PyTorch 分布式系统中组织参与通信进程的抽象。 |
| Dynamo | 文中负责图捕获和设备管理等编译器集成环节的组件。 |
| Inductor | 提供调度、融合和代码生成等能力的编译器组件，硬件厂商可选择接入其优化流程。 |
| fusion | 融合，将多个操作组合到内核中的编译优化方式。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let’s start with the motivation. | 我们先谈谈为什么要做这件事。 |
| That’s the issue. | 问题就出在这里。 |
| How much progress did they report? | 他们公布了多少进展？ |
| In plain English | 用通俗的话说。 |
| That’s the basic idea. | 基本思路就是这样。 |
| What does it actually demonstrate? | 它具体展示了什么？ |
| That’s another big surface to connect. | 这又是一大块需要打通的集成范围。 |
| What’s still ahead? | 接下来还有哪些工作？ |
