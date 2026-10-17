# How New Hardware Finds Its Way into PyTorch

原文：[PyTorch Hardware Enablement: Updates from the Accelerator Integration Working Group](https://pytorch.org/blog/pytorch-hardware-enablement-updates-from-the-acceleration-integration-working-group/)

## 摘要

本期讨论 PyTorch 加速器集成工作组在二〇二六年上半年的进展，重点是通过统一接口、共享测试和参考实现降低新硬件接入成本。跨仓库持续集成机制 CRCR 将下游测试反馈带回上游，而测试重构让更多测试能够跨后端复用。OpenReg 及其相关工作展示了性能分析、分布式通信和编译器集成的关键路径，但这些参考实现并不以生产性能为目标。下半年的方向包括继续减少未分类测试、扩大覆盖范围、完善自动测试选择，并鼓励更多厂商申请加入官方 Additional Platforms 页面。

## 对话

Ava: You’ve built a new accelerator. It runs your model. Then PyTorch changes, and you’re back fixing patches. How do we make that less painful?

Ava：你造出了一款新加速器，它能运行你的模型。然后 PyTorch 一更新，你又得回头修补丁。怎样才能让这件事没那么折磨人？

Brian: That’s the problem behind this article. The Accelerator Integration Working Group wants clearer integration paths, shared tests, and earlier feedback when upstream changes break something.

Brian：这就是这篇文章要解决的问题。Accelerator Integration Working Group 希望建立更清晰的集成路径、共享测试，并在上游变更导致故障时更早收到反馈。

Ava: I’m Ava. I’m a software engineer outside the PyTorch core team, and maintaining a pile of patches sounds painfully familiar.

Ava：我是 Ava。我是一名不在 PyTorch 核心团队的软件工程师，维护一大堆补丁这件事，听着真是熟悉得让人头疼。

Brian: And I’m Brian. Today we’re looking at the group’s progress in the first half of twenty twenty-six, from testing to compiler support.

Brian：我是 Brian。今天我们来看看这个小组在二零二六年上半年取得的进展，从测试到编译器支持。

Ava: Let’s start with the motivation. Why does hardware integration need a working group?

Ava：先说说动机吧。为什么硬件集成需要一个工作组？

Brian: More compute platforms are adopting PyTorch. Each backend, meaning the implementation connecting hardware to the framework, needs reliable ways to fit into the same system.

Brian：越来越多的计算平台开始采用 PyTorch。每个后端，也就是连接硬件与框架的实现，都需要可靠的方式接入同一个系统。

Ava: Like different appliances needing a standard socket, instead of everyone opening the wall and rewiring it?

Ava：就像不同的电器需要标准插座，而不是每家都得打开墙壁重新布线？

Brian: Exactly. The group works on vendor-neutral mechanisms, shared tools, and reference implementations. Those help hardware teams avoid maintaining their own framework patches.

Brian：正是这样。小组致力于开发厂商中立的机制、共享工具和参考实现，帮助硬件团队避免维护各自的框架补丁。

Ava: Where does the feedback problem come in?

Ava：反馈方面的问题又出在哪里？

Brian: PyTorch’s upstream repository, where framework development happens, has extensive continuous integration, or CI. That’s automated testing of changes. But dependent projects run tests elsewhere.

Brian：PyTorch 的上游代码仓库，也就是框架开发所在的仓库，有完善的持续集成，简称 CI，也就是对变更进行自动化测试。但依赖 PyTorch 的项目是在别处运行测试的。

Ava: Those are the downstream projects. Hardware backends and libraries that depend on PyTorch?

Ava：这些就是下游项目？比如依赖 PyTorch 的硬件后端和库？

Brian: Right. They lacked a standard way to receive change notifications and return results. Contributors couldn’t easily see downstream damage before merging a change.

Brian：对。它们缺少接收变更通知并返回结果的标准方式。贡献者在合并变更前，很难看出它会给下游造成什么问题。

Ava: So the warning could arrive after the change was already in?

Ava：所以等警报传来时，变更可能已经合进去了？

Brian: Yes. Cross-Repository CI Relay, or CRCR, connects those separate testing systems. A pull request, meaning a proposed code change, or a pushed commit triggers notifications.

Brian：是的。Cross-Repository CI Relay，简称 CRCR，把这些独立的测试系统连接起来。拉取请求，也就是一项拟议的代码变更，或者推送的提交，都会触发通知。

Ava: And every registered project gets its own chance to test?

Ava：每个已注册的项目都有机会运行自己的测试？

Brian: They’re notified in parallel. Each repository runs its own workflow, then sends its status back. Results appear in PyTorch’s shared CI dashboard within seconds.

Brian：它们会同时收到通知。每个仓库运行自己的工作流，然后把状态发回来。结果会在上报后的几秒内出现在 PyTorch 的共享 CI 仪表板上。

Ava: Within seconds of reporting, though. That doesn’t mean a huge test suite finishes in seconds.

Ava：不过，是上报后几秒内显示结果。这可不代表一套庞大的测试能在几秒内跑完。

Brian: Right. The article describes fast result delivery. It doesn’t promise that the tests themselves run that quickly.

Brian：对。文章说的是结果传递很快，并没有承诺测试本身也能跑得那么快。

Ava: How does PyTorch know who’s reporting? A green check needs a little more evidence than, trust me.

Ava：PyTorch 怎么知道是谁在上报？一个绿色勾号总得有比“相信我”更充分的依据吧。

Brian: GitHub OpenID Connect, or OIDC, tokens verify the calling repository’s identity. There are also authorization checks, rate limits, and checks on valid reporting state.

Brian：GitHub OpenID Connect，简称 OIDC，令牌用于验证发起调用的仓库身份。此外还有授权检查、速率限制，以及对上报状态是否有效的检查。

Ava: Does joining immediately give a downstream project power to block changes?

Ava：下游项目一加入，就能阻止变更合并吗？

Brian: No. Four participation levels move from notifications through dashboard reporting to non-blocking and eventually blocking checks. Trusted information stays separate from self-reported data.

Brian：不能。参与分为四个级别，从接收通知，到在仪表板上报告结果，再到非阻断性检查，最终是阻断性检查。可信信息与项目自行上报的数据会分开处理。

Ava: Got it. Shared visibility, with controlled participation. But teams still need useful tests to run.

Ava：明白了。大家能共享信息，但参与权限受到控制。不过，团队还得有真正有用的测试可跑。

Brian: Exactly. PyTorch has over six hundred thousand test cases. Many were tied to particular hardware through fixed device names or backend-specific operations.

Brian：没错。PyTorch 有超过六十万个测试用例。许多测试通过固定的设备名称或特定后端的操作，与某种硬件绑定在了一起。

Ava: So a test might check a general feature, but its setup only lets one accelerator through the door?

Ava：也就是说，一个测试可能是在检查通用功能，但它的准备步骤只让一种加速器进得了门？

Brian: That’s the issue. Other vendors had to patch tests. The refactoring replaces fixed hardware references with device parameters, so the same logic can serve multiple backends.

Brian：问题就在这里。其他厂商不得不修改测试。重构工作用设备参数替换固定的硬件引用，让同一套测试逻辑可以服务于多个后端。

Ava: How much progress did they report?

Ava：他们报告了多少进展？

Brian: Contributors migrated over two hundred seventy-six test files in the first half of the year. They covered areas including profiling, neural network modules, and optimizers.

Brian：今年上半年，贡献者迁移了二百七十六个以上的测试文件，涵盖性能分析、神经网络模块和优化器等领域。

Ava: But some tests really are hardware-specific. You can’t make every test universal.

Ava：但有些测试确实是硬件专用的，不可能让每个测试都通用。

Brian: Right. Classes fall into three groups: accelerator-unrelated, accelerator-agnostic, and accelerator-specific. In plain English, that’s no accelerator needed, reusable across devices, or tied to one backend.

Brian：对。测试类分为三组：与加速器无关、与具体加速器无关，以及特定于某种加速器。说白了，就是不需要加速器、可在不同设备上复用，或者只适用于一个后端。

Ava: And that classification helps select the right tests?

Ava：这种分类有助于选出合适的测试？

Brian: Yes. The metadata and selection flag work across the test execution paths. A linter, an automated rule checker, requires new test classes to declare their classification.

Brian：是的。分类元数据和选择标志适用于各种测试执行路径。linter，也就是自动检查规则的工具，会要求新测试类声明自己的分类。

Ava: What happens to all the old ones?

Ava：那所有旧测试类怎么办？

Brian: They’re being handled gradually. At launch, one thousand one hundred ninety-one unclassified files were on an allowlist. Reducing that list is ongoing work.

Brian：会逐步处理。上线时，有一千一百九十一个未分类文件列在允许名单上。缩短这份名单仍是一项持续进行的工作。

Ava: What if my backend only supports some features?

Ava：如果我的后端只支持部分功能呢？

Brian: There’s fine-grained skipping by feature, class, test, and operator. Vendors can select what matches their implementation without changing upstream test code.

Brian：可以按功能、测试类、单项测试和算子进行细粒度跳过。厂商无需修改上游测试代码，就能选择符合自身实现的测试。

Ava: So we’ve got reusable tests and clearer selection. Now, where do developers learn the actual integration patterns?

Ava：这样我们就有了可复用的测试，也能更清楚地选择测试。那么，开发者去哪里学习实际的集成模式？

Brian: That’s OpenReg’s role. It’s a minimal reference backend built on PrivateUse1, PyTorch’s mechanism for integrating custom accelerator backends.

Brian：这就是 OpenReg 的作用。它是一个基于 PrivateUse1 构建的精简参考后端；PrivateUse1 是 PyTorch 用于集成自定义加速器后端的机制。

Ava: Does OpenReg need special hardware?

Ava：OpenReg 需要专门的硬件吗？

Brian: No. It’s backed by the central processing unit, or CPU. It lives inside PyTorch’s source tree and demonstrates integration mechanics without real accelerator complexity.

Brian：不需要。它以中央处理器，也就是 CPU，作为运行基础。它位于 PyTorch 的源代码树中，用来展示集成机制，而不必引入真实加速器的复杂性。

Ava: Like a teaching engine with the important parts visible. You wouldn’t enter it in a race.

Ava：就像一台把重要部件都露出来的教学用发动机，你不会开着它去比赛。

Brian: Exactly. It isn’t a production backend. It demonstrates device registration, operator dispatch, meaning routing operations to implementations, and runtime behavior such as streams and events.

Brian：正是这样。它不是生产环境用的后端。它展示设备注册、算子分派，也就是把操作路由到相应实现，以及流和事件等运行时行为。

Ava: Streams organize device work, and events help track its progress?

Ava：流负责组织设备上的工作，事件则帮助跟踪工作进度？

Brian: That’s the basic idea. OpenReg also runs in CI, helping catch breaks in PrivateUse1 integration. Its implementation is kept aligned with the integration documentation.

Brian：基本就是这样。OpenReg 还会在 CI 中运行，帮助发现 PrivateUse1 集成出现的故障。它的实现也会与集成文档保持一致。

Ava: Let’s talk profiling. Knowing an operation took a long time isn’t always enough.

Ava：说说性能分析吧。只知道某个操作耗时很长，有时还不够。

Brian: Right. Developers also need timelines for kernels, the device computations, plus memory and runtime events. They need to connect host activity with device activity.

Brian：对。开发者还需要内核，也就是设备端计算的时间线，以及内存和运行时事件的时间线。他们需要把主机端活动与设备端活动关联起来。

Ava: Like matching an order placed at the counter with the work happening in the kitchen?

Ava：就像把柜台接下的订单和厨房里正在做的事对应起来？

Brian: Yes. Correlation IDs provide that connection. OpenReg’s reference profiling stack demonstrates those IDs, activity types, and the lifecycle of a profiling session.

Brian：是的。关联 ID 就是用来建立这种联系的。OpenReg 的参考性能分析组件展示了这些 ID、活动类型，以及一次性能分析会话的生命周期。

Ava: Is it a complete profiler?

Ava：它是一套完整的性能分析器吗？

Brian: No. It’s a stub, a minimal implementation showing the connections. It plugs into Kineto, the profiling system used here, through PyTorch’s registration interface.

Brian：不是。它只是一个桩实现，用最小限度的代码展示各部分如何连接。它通过 PyTorch 的注册接口接入这里使用的性能分析系统 Kineto。

Ava: Then vendors replace the demonstration pieces with their own device profiling calls?

Ava：然后，厂商会用自己的设备性能分析调用替换演示代码吗？

Brian: Exactly, without modifying PyTorch or Kineto. Tests cover registration, session lifecycle, and fallback behavior. The older path remains available for coarse operator-level timing.

Brian：没错，而且不需要修改 PyTorch 或 Kineto。测试覆盖注册、会话生命周期和回退行为。原有路径仍可用于较粗粒度的算子级耗时统计。

Ava: That’s a clearer starting point. Does the same approach extend to distributed training?

Ava：这样作为起点就更清楚了。同样的方法也适用于分布式训练吗？

Brian: Yes. The group began building OCCL, the OpenReg Collective Communications Library. It’s a minimal reference for a custom backend in c10d, PyTorch’s distributed infrastructure.

Brian：是的。团队开始构建 OCCL，即 OpenReg Collective Communications Library。它为 c10d 中的自定义后端提供了一个精简的参考实现；c10d 是 PyTorch 的分布式基础设施。

Ava: What does it actually demonstrate?

Ava：它具体展示了什么？

Brian: Registering a ProcessGroup, which represents participating processes, dispatching collectives, meaning coordinated communication operations, and handling Work completion, meaning when an operation is considered finished.

Brian：包括注册代表参与进程的 ProcessGroup、分发 collectives（也就是协同通信操作），以及处理 Work 的完成状态，也就是确定一项操作何时算作完成。

Ava: So again, the result is a clearer integration contract. We’re not hearing a communication speed claim.

Ava：所以，这次的成果同样是让集成约定更清楚。这里并没有宣称通信速度有所提升。

Brian: Correct. Production performance isn’t OCCL’s target. Related work also improved support for Python-based distributed backends and device-generic distributed test selection.

Brian：对。OCCL 的目标并不是生产环境中的性能。相关工作还改进了对基于 Python 的分布式后端的支持，以及与设备无关的分布式测试选择。

Ava: How about compilation? That’s another big surface to connect.

Ava：编译方面呢？那也是一个需要接入的重要部分。

Brian: The OpenReg example shows integration with torch.compile, PyTorch’s compilation entry point. Dynamo handles graph capture and device management. Graph capture records operations for compilation.

Brian：OpenReg 示例展示了如何与 torch.compile 集成；torch.compile 是 PyTorch 的编译入口。Dynamo 负责图捕获和设备管理。图捕获会记录操作，以供编译使用。

Ava: And where does Inductor fit?

Ava：那 Inductor 在其中起什么作用？

Brian: Inductor offers scheduling, fusion, and code generation. Fusion combines operations into kernels. Vendors can optionally use those optimization passes, and OpenReg demonstrates the integration points.

Brian：Inductor 提供调度、融合和代码生成能力。融合会把多个操作合并成内核。厂商可以选择使用这些优化流程，而 OpenReg 展示了相应的集成点。

Ava: Including fused kernel generation without changing upstream PyTorch?

Ava：这也包括无需修改上游 PyTorch 就能生成融合内核吗？

Brian: Yes. There’s also a compiler integration guide. Again, the focus is showing how the pieces connect, rather than reporting production performance.

Brian：是的。另外还有一份编译器集成指南。这里的重点同样是展示各部分如何衔接，而不是报告生产环境中的性能。

Ava: What’s still ahead?

Ava：接下来还有哪些工作？

Brian: More test coverage, fewer unclassified tests, and connecting classification to automatic CI scheduling. They’re also exploring a registry where accelerators declare supported operator data types and precision overrides.

Brian：扩大测试覆盖范围、减少尚未分类的测试，并将分类结果接入自动 CI 调度。他们也在探索建立一个注册机制，让加速器声明所支持的算子数据类型和精度覆盖设置。

Ava: And once a platform’s integrated, how do users find it?

Ava：平台完成集成后，用户该怎么找到它？

Brian: The official Additional Platforms page links from PyTorch’s install page. Admission requires evidence and review. The group’s encouraging more vendors to apply in the second half.

Brian：PyTorch 的安装页面会链接到官方的 Additional Platforms 页面。要被收录，必须提供证据并接受审核。团队也鼓励更多厂商在下半年申请。

Ava: Let’s recap three points. First, CRCR brings downstream test feedback into a shared upstream view.

Ava：我们来回顾三点。第一，CRCR 将下游测试的反馈汇集到一个共享的上游视图中。

Brian: Second, test refactoring helps backends reuse validation. Third, OpenReg provides concrete integration references, while production readiness still requires further work.

Brian：第二，测试重构有助于后端复用验证流程。第三，OpenReg 提供了具体的集成参考，但要达到生产环境可用的程度，仍需进一步工作。

Ava: Thanks for listening. We’ll see you next time.

Ava：感谢收听。我们下次再见。

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
