# How Spyre Became a Native PyTorch Device

原文：[Building Spyre as a Native PyTorch Device - PyTorch](https://pytorch.org/blog/building-spyre-as-a-native-pytorch-device/)

## 摘要

本期讨论 IBM 如何通过 torch-spyre，把 Spyre 接入 PyTorch 的设备、内存分配器、流和编译接口。节目解释了为什么设备端张量驻留、启动时地址修补，以及跨流事件对 Spyre 的数据流硬件很重要。文章报告，在所测模型配置下，移除运行时图处理显著改善了预填充和解码速度。两位主持人也谈到当前限制，包括首次调用的编译开销，以及尚未开放给用户的事件接口。

## 对话

Ava: If an accelerator runs PyTorch models, why should we care whether PyTorch sees it as a real device?

Ava：如果一款加速器能运行 PyTorch 模型，为什么我们还要在意 PyTorch 是否把它识别为真正的设备？

Brian: Because that decides whether tensors can stay on it between operations. For IBM's Spyre accelerator, native device support also cuts launch work and lets data movement overlap computation.

Brian：因为这决定了张量能否在连续操作之间一直留在加速器上。对于 IBM 的 Spyre 加速器，原生设备支持还能减少程序启动时的工作，并让数据传输与计算重叠进行。

Ava: So this is about more than making one model compile?

Ava：所以这件事不只是让某个模型能够编译？

Brian: Exactly. Spyre is a dataflow AI accelerator optimized for inference. The goal is familiar PyTorch code, with the storage and scheduling behavior the hardware needs.

Brian：正是如此。Spyre 是一款针对推理优化的数据流 AI 加速器。目标是让开发者使用熟悉的 PyTorch 代码，同时具备硬件所需的存储和调度行为。

Ava: Let's start with the hardware. What does dataflow mean here?

Ava：先从硬件说起。这里的“数据流”是什么意思？

Brian: A compiled Spyre kernel contains programs for several kinds of functional unit. Those units make progress when their input data arrives. Think of stations in a workshop: each starts its part when the previous station delivers the pieces.

Brian：一个编译后的 Spyre 内核包含供几种不同功能单元运行的程序。这些单元在输入数据到达时就开始推进工作。可以把它想成车间里的各个工位：前一个工位送来零件，后一个工位就开始自己的那部分工作。

Ava: And the compiler decides how those stations work together?

Ava：这些工位如何协作，是由编译器决定的？

Brian: Right. It builds the device-side schedule, including movement between large device memory and each core's local scratchpad. The runtime submits the compiled program and its tensor arguments.

Brian：对。编译器构建设备端的调度安排，包括大容量设备内存与各核心本地暂存区之间的数据移动。运行时则提交编译后的程序及其张量参数。

Ava: Can the runtime launch two compute jobs at once to keep the card busy?

Ava：运行时能同时启动两个计算任务，让加速卡保持忙碌吗？

Brian: Independently submitted compute jobs share one runtime compute queue. They wait behind each other. But transfers have separate pipelines, so data can move while a program computes.

Brian：独立提交的计算任务共用一条运行时计算队列，会排队依次执行。但数据传输有独立的流水线，因此程序计算时也能移动数据。

Ava: Got it. The useful overlap is transfer with compute. How does PyTorch express that?

Ava：明白了。真正有用的重叠是数据传输与计算同时进行。PyTorch 怎么表达这一点？

Brian: With streams: ordered queues of work. PyTorch also has events, which let one stream wait for work on another. torch-spyre maps those ideas onto Spyre's runtime.

Brian：通过流，也就是按顺序执行工作的队列。PyTorch 还有事件，可以让一条流等待另一条流上的工作。torch-spyre 将这些概念映射到 Spyre 的运行时。

Ava: Before streams, though, PyTorch needs to recognize Spyre. What gives it a device identity?

Ava：不过，在使用流之前，PyTorch 得先识别 Spyre。它是怎样获得设备身份的？

Brian: PrivateUse1, PyTorch's extension slot for an outside device backend. torch-spyre names that slot spyre and registers a device module. With allocator, copy, and dispatcher support, a tensor can move to Spyre and operations can reach Spyre kernels.

Brian：靠 PrivateUse1，它是 PyTorch 留给外部设备后端的扩展槽位。torch-spyre 将这个槽位命名为 spyre，并注册一个设备模块。有了分配器、复制操作和分发器的支持，张量就能转移到 Spyre，操作也能调用 Spyre 内核。

Ava: Does that also explain why ordinary PyTorch streams can work?

Ava：这也解释了为什么普通的 PyTorch 流能够工作吗？

Brian: Yes. The backend implements the device-guard hooks PyTorch asks for. Then native stream creation and stream selection use PyTorch's own interface.

Brian：是的。后端实现了 PyTorch 要求的设备保护钩子。这样，原生流的创建和流的选择就能使用 PyTorch 自身的接口。

Ava: Let's talk about tensors staying put. What went wrong with the earlier approach?

Ava：说说让张量留在设备上的事。之前的做法出了什么问题？

Brian: The earlier custom compiler backend could keep selected model state on Spyre through special runtime paths. But PyTorch still saw ordinary tensors as CPU tensors, even when the runtime had moved their data.

Brian：之前的自定义编译器后端可以通过特殊的运行时路径，把选定的模型状态留在 Spyre 上。但 PyTorch 仍将普通张量视为 CPU 张量，即使运行时已经移动了它们的数据。

Ava: That sounds expensive for eager execution, where operations happen one at a time.

Ava：对于一次执行一个操作的即时执行模式，这听起来成本很高。

Brian: It was a poor fit. Each small operation could become another transfer boundary. A native Spyre tensor instead has device storage managed through PyTorch's allocator and storage lifetime.

Brian：这种方式确实不太适合。每个小操作都可能形成另一个数据传输边界。相比之下，原生 Spyre 张量拥有通过 PyTorch 分配器和存储生命周期管理的设备存储空间。

Ava: So PyTorch knows the tensor is resident, and it knows when to release its memory?

Ava：所以 PyTorch 知道张量驻留在设备上，也知道什么时候释放它的内存？

Brian: Exactly. When the last reference goes away, PyTorch calls the backend allocator to release or recycle the allocation.

Brian：没错。最后一个引用消失时，PyTorch 会调用后端分配器来释放或回收这块内存。

Ava: Is Spyre memory just one big set of pointers?

Ava：Spyre 的内存就是一大组指针吗？

Brian: Underneath the allocator interface, memory comes in regions, each with a handle. The allocator takes a small number of large regions and divides them into aligned blocks. Many tensors can then share a few handles.

Brian：在分配器接口之下，内存以区域的形式提供，每个区域都有一个句柄。分配器取得少量的大区域，再把它们划分为对齐的内存块。这样，许多张量就能共用少数几个句柄。

Ava: Why keep the number of handles low?

Ava：为什么要控制句柄的数量？

Brian: The handle budget is limited, especially when tenants share a card. An allocation is described by a region, an offset, and a length. A tensor spread across memory domains may need several such pieces.

Brian：句柄数量有上限，尤其是在多个租户共用一张卡时。一块分配的内存由区域、偏移量和长度来描述。分布在多个内存域中的张量可能需要好几段这样的内存。

Ava: Memory domains? Are we talking about several cards?

Ava：内存域？我们说的是好几张卡吗？

Brian: No, locality within one device. Some cores reach certain memory domains more efficiently. A tensor can sit near the cores using it or be spread across domains for bandwidth. That placement matters because compiled code may depend on it.

Brian：不是，说的是一台设备内部的局部性。某些核心访问特定内存域的效率更高。张量可以放在使用它的核心附近，也可以分布到多个内存域以获得带宽。这种放置方式很重要，因为编译后的代码可能依赖于它。

Ava: And layout matters too, I assume?

Ava：我猜布局也很重要？

Brian: It does. Compiled kernels expect particular layouts. Model adapters prepare weights and key-value caches, the stored attention state, in those layouts. Compiler passes add legal conversions for intermediate tensors.

Brian：是的。编译后的内核要求特定的布局。模型适配器会按这些布局准备权重和 key-value caches，也就是存储下来的注意力状态。编译器处理阶段则为中间张量加入合法的布局转换。

Ava: Let me recap: PyTorch owns tensor lifetime, while the Spyre allocator describes the real device placement.

Ava：我来总结一下：PyTorch 负责张量的生命周期，而 Spyre 分配器描述它在设备上的实际放置位置。

Brian: That's it. Now the next problem is launching a compiled program with tensors whose addresses weren't known when it was compiled.

Brian：就是这样。接下来的问题是，如何用编译时还不知道地址的张量启动编译后的程序。

Ava: Why weren't they known?

Ava：为什么那时还不知道地址？

Brian: The compiler fixes the program's layout ahead of time. But PyTorch's allocator places caller-owned input and output tensors later. Their addresses must stay symbolic until launch.

Brian：编译器会提前确定程序的布局。但调用方拥有的输入和输出张量，要到后来才由 PyTorch 分配器放置。它们的地址必须一直保持为符号形式，直到启动时才能确定。

Ava: What did the earlier runtime do?

Ava：之前的运行时是怎么处理的？

Brian: One path copied inputs into buffers owned by the compiled job, then copied results out. Another bound a small set of device address windows to tensors. That avoided copies, but the number of windows limited how many tensor inputs a fused kernel could take.

Brian：一种路径会把输入复制到编译后任务拥有的缓冲区，再把结果复制出来。另一种路径会把少量设备地址窗口绑定到张量上。这避免了复制，但窗口的数量限制了一个融合内核能接收多少个张量输入。

Ava: What's the current solution?

Ava：目前的解决办法是什么？

Brian: Launch-time patching. The compiled binary has placeholders for tensor addresses. At launch, a CPU callback writes the actual addresses into a pinned host buffer. A transfer moves that correction data to the program's device allocation. Then the device uses it to patch the operands.

Brian：启动时修补。编译后的二进制程序为张量地址留有占位符。启动时，一个 CPU 回调会把实际地址写入页锁定的主机缓冲区。随后一次传输把这些修正数据移到程序的设备内存分配区，再由设备用它们修补操作数。

Ava: Like filling in delivery addresses after the packages are packed?

Ava：就像包裹打包之后再填写收货地址？

Brian: Pretty much. The package layout is fixed; the destinations arrive later. Those three steps need strict ordering, or the program could read an address before it's ready.

Brian：差不多。包裹的布局已经固定，目的地则是后来才确定的。这三个步骤必须严格按顺序执行，否则程序可能在地址准备好之前就读取它。

Ava: Would putting everything on one stream solve that?

Ava：把所有步骤放在同一条流上能解决吗？

Brian: It would preserve order, but it would also make preparation wait while compute runs. The initial design uses a preparation stream for the CPU callback and transfer, and a device stream for compute. An event makes compute wait for its transfer.

Brian：能保证顺序，但也会让准备工作在计算运行期间一直等待。最初的设计用一条准备流处理 CPU 回调和传输，再用一条设备流处理计算。事件会让计算等到对应的传输完成。

Ava: Then preparation for the next invocation can happen during the current computation?

Ava：这样，下一次调用的准备工作就能在当前计算期间进行？

Brian: Yes. Its tensor addresses are already known at launch. A transfer can overlap too, provided it targets device storage the current computation isn't using.

Brian：对。下一次调用的张量地址在启动时已经确定。传输也可以同时进行，前提是目标设备存储空间没有被当前计算使用。

Ava: What if both invocations use the same device correction area?

Ava：如果两次调用使用同一块设备端修正区域呢？

Brian: Then the next transfer must wait until the earlier compute finishes reading that area. There's also a separate hardware detail: Spyre can fetch a program early, before its patch reaches memory. A barrier in program distribution holds that fetch until the patch has landed.

Brian：那下一次传输就必须等前一次计算读完那块区域。还有一个独立的硬件细节：Spyre 可能在补丁到达内存之前就提前取入程序。程序分发过程中的一道屏障会让这次取入等待补丁写入完成。

Ava: So queue order alone doesn't prevent that early fetch. That's a subtle one.

Ava：所以光靠队列顺序无法阻止提前取入。这点真微妙。

Brian: It is. The host staging buffer has its own reuse rule. One preparation stream can reuse it because each transfer finishes reading it before the next callback overwrites it. More preparation streams would need separate buffers or a pool that waits for completion.

Brian：确实如此。主机端暂存缓冲区也有自己的复用规则。一条准备流可以复用它，因为每次传输都会在下一个回调覆盖缓冲区之前读完其中的数据。如果有多条准备流，就需要各自使用独立缓冲区，或者使用一个会等待操作完成的缓冲区池。

Ava: You mentioned events. Can a PyTorch user record and wait on Spyre events today?

Ava：你提到了事件。现在 PyTorch 用户能记录并等待 Spyre 事件吗？

Brian: The runtime uses software events internally and can derive dependencies when operations' device-memory reads and writes conflict. User-callable event recording and waiting through PyTorch are planned, but aren't implemented yet. Users can synchronize a stream or the accelerator to wait for completion.

Brian：运行时会在内部使用软件事件，并且在操作对设备内存的读写发生冲突时推导出依赖关系。通过 PyTorch 供用户调用的事件记录和等待功能已有规划，但尚未实现。用户可以同步一条流或整个加速器，等待工作完成。

Ava: Let's zoom out. Where does the compiler fit, and why did they remove a runtime graph?

Ava：我们先从整体看。编译器处在什么位置？他们为什么移除了运行时图？

Brian: FX graphs, PyTorch's captured computation graphs, stay in Inductor, PyTorch's compiler pipeline. The earlier integration built a second, backend-specific graph after compilation. Even a single launch had to read and rebuild graph information around its tensor arguments.

Brian：FX 图，也就是 PyTorch 捕获的计算图，保留在 PyTorch 的编译流程 Inductor 中。早期的集成方式会在编译后再构建一个后端专用的图。即使只启动一次，也得围绕张量参数读取并重建图的信息。

Ava: So they prepare the compiled artifact once?

Ava：所以他们只需准备一次编译产物？

Brian: Yes. Preparation parses the artifact, allocates and transfers its program binary, and makes an ordered recipe of typed steps. Each launch then builds and enqueues the needed host work, transfers, and compute operations.

Brian：对。准备阶段会解析编译产物、分配并传输程序二进制文件，然后生成一份按顺序排列、注明步骤类型的执行清单。每次启动时，再构建所需的主机端任务、数据传输和计算操作，并将它们加入队列。

Ava: An ordered recipe is easier to picture than another graph. What does that change for eager operations?

Ava：按顺序执行的清单比另一个图更容易理解。这对即时执行的操作有什么影响？

Brian: torch-spyre can use the same decomposition in compiled and eager execution. Inside a compile trace, PyTorch captures that decomposition in the larger FX graph. Outside a trace, the first eager call compiles and caches it; later calls reuse the compiled entry point.

Brian：torch-spyre 可以在编译执行和即时执行中使用同一套分解步骤。在编译跟踪期间，PyTorch 会把这些分解步骤捕获到更大的 FX 图里。在跟踪之外，首次即时调用会编译并缓存它；后续调用则复用已编译的入口。

Ava: So an eager operation can also test the whole path?

Ava：这么说，单个即时执行的操作也能测试整个流程？

Brian: Right. Even one operation exercises real tensors, device memory, compilation, and launch. If that operation works alone but fails inside a model, the problem is more likely in how operations are combined.

Brian：没错。即使只有一个操作，也会用到真实张量、设备内存、编译和启动。如果这个操作单独运行正常，却在模型里失败，问题就更可能出在操作的组合方式上。

Ava: What results does the article report?

Ava：文章报告了哪些结果？

Brian: On Granite three point three eight B, with batch size one and sequence length one thousand twenty-four, removing graph construction from transfers alone improved prefill by nine point nine percent and decode by thirteen point six percent.

Brian：在 Granite 3.3 8B 上，批量大小为 1、序列长度为 1024 时，仅从数据传输环节移除计算图构建，就使预填充性能提升了 9.9%，解码性能提升了 13.6%。

Ava: And when compute dispatch stopped using that runtime graph too?

Ava：那计算任务分派也不再使用这个运行时图之后呢？

Brian: The reported totals were one point seven times faster prefill and two point four times faster decode. Roughly three quarters of the gain came from the compute path. Those numbers describe that model and setup.

Brian：报告的总体结果是，预填充速度达到原来的 1.7 倍，解码速度达到原来的 2.4 倍。大约四分之三的提升来自计算路径。这些数字针对的是该模型和这套配置。

Ava: What still needs work?

Ava：还有哪些工作要做？

Brian: First-use eager calls still compile. Unsupported operations may fail or fall back, and different tensor sizes may need another compiled artifact. Native device support doesn't automatically deliver operator coverage, profiling, distributed collectives, or great kernels for every model.

Brian：首次即时调用仍然需要编译。不支持的操作可能失败，也可能回退到其他实现；不同的张量尺寸可能需要另一个编译产物。原生设备支持也不会自动带来完整的算子覆盖、性能分析能力、分布式集合通信，或适合每个模型的高性能内核。

Ava: Is there a plan for changing tensor sizes?

Ava：对于张量尺寸的变化，有什么计划吗？

Brian: The article describes future support for passing actual variable dimension sizes through the same launch-time correction mechanism as addresses. Within supported bounds and layout rules, that could let one artifact serve multiple sizes.

Brian：文章提到，未来计划通过与地址相同的启动时修正机制，传入可变维度的实际大小。只要符合支持的尺寸范围和布局规则，这就可能让一个编译产物适用于多种尺寸。

Ava: Does any of this flow back into PyTorch?

Ava：这些工作会有一部分回馈给 PyTorch 吗？

Brian: The team contributes to openreg, PyTorch's reference backend for outside devices. It also tests upstream changes against Spyre through the Cross-Repository CI Relay and plans to contribute reusable tools for adapting upstream tests.

Brian：团队正在为 openreg 贡献代码；它是 PyTorch 为外部设备提供的参考后端。团队还通过 Cross-Repository CI Relay 在 Spyre 上测试上游变更，并计划贡献可复用的工具，用于适配上游测试。

Ava: Okay, three-point recap. First, Spyre becomes a real PyTorch device, so tensors have managed storage and can stay resident.

Ava：好，用三点总结。第一，Spyre 成为真正的 PyTorch 设备，张量因此有了受管理的存储空间，也能常驻设备。

Brian: Second, compiled artifacts become prepared recipes. Launch-time patching supplies tensor addresses, and events order work across streams.

Brian：第二，编译产物会变成预先准备好的执行清单。启动时的修补会填入张量地址，事件则负责安排跨流任务的执行顺序。

Ava: Third, that design removes runtime graph work and opens room for transfer and compute overlap, though coverage and public events still need work.

Ava：第三，这种设计移除了运行时的图处理工作，并为数据传输与计算重叠执行创造了空间，不过操作覆盖范围和公开的事件支持仍需完善。

Brian: That's Spyre's PyTorch story for today. Thanks for listening.

Brian：以上就是今天关于 Spyre 与 PyTorch 的介绍。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| PrivateUse1 | PyTorch 为外部设备后端提供的扩展槽位。 |
| dataflow | 由数据到达驱动功能单元继续执行的计算方式。 |
| scratchpad | 每个核心使用的本地暂存存储器。 |
| stream | 按顺序提交并完成操作的工作队列抽象。 |
| event | 用于建立不同流之间执行顺序的信号。 |
| allocator | 分配、释放或回收设备存储的组件。 |
| region | 带有句柄的连续设备内存块，可再分配给多个张量。 |
| memory domain | 设备内部具有不同访问局部性的内存区域。 |
| launch-time patching | 启动时把实际张量地址填入已编译程序的机制。 |
| pinned host buffer | 用于暂存修补数据、供传输读取的固定主机内存缓冲区。 |
| FX graph | PyTorch 捕获的计算图，供编译流程处理。 |
| Inductor | 文中用于处理 FX 图的 PyTorch 编译器路径。 |
| eager execution | 在编译完整模型之外，逐次调用操作的执行方式。 |
| prefill | 语言模型处理已有输入序列的阶段。 |
| decode | 语言模型逐步生成输出的阶段。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's start with the hardware. | 我们先从硬件说起。 |
| Got it. | 明白了。 |
| Let's talk about tensors staying put. | 我们来谈谈张量如何留在设备上。 |
| Let me recap: | 我来总结一下： |
| What's the current solution? | 目前的解决办法是什么？ |
| That's a subtle one. | 这一点挺容易忽略。 |
| Let's zoom out. | 我们退一步看整体。 |
| What still needs work? | 还有哪些方面需要完善？ |
