# How Spyre Became a Native PyTorch Device

原文：[Building Spyre as a Native PyTorch Device - PyTorch](https://pytorch.org/blog/building-spyre-as-a-native-pytorch-device/)

## 摘要

本期讨论 IBM 如何通过 torch-spyre，把 Spyre 接入 PyTorch 的设备、内存分配器、流和编译接口。节目解释了为什么设备端张量驻留、启动时地址修补，以及跨流事件对 Spyre 的数据流硬件很重要。文章报告，在所测模型配置下，移除运行时图处理显著改善了预填充和解码速度。两位主持人也谈到当前限制，包括首次调用的编译开销，以及尚未开放给用户的事件接口。

## 对话

Ava: If an accelerator runs PyTorch models, why should we care whether PyTorch sees it as a real device?

Ava：如果加速器能运行 PyTorch 模型，为什么还要在意 PyTorch 是否把它识别为真正的设备？

Brian: Because that decides whether tensors can stay on it between operations.

Brian：因为这决定了张量能否在操作之间留在加速器上。

For IBM's Spyre accelerator, native device support also cuts launch work and lets data movement overlap computation.

对 IBM 的 Spyre 加速器来说，原生设备支持还能减少启动开销，让数据传输与计算重叠。

Ava: So this is about more than making one model compile?

Ava：所以这不只是让一个模型编译成功？

Brian: Exactly. Spyre is a dataflow AI accelerator optimized for inference.

Brian：没错。Spyre 是专为推理优化的数据流 AI 加速器。

The goal is familiar PyTorch code, with the storage and scheduling behavior the hardware needs.

目标是让开发者照常编写 PyTorch 代码，同时满足硬件对存储和调度的要求。

Ava: Let's start with the hardware. What does dataflow mean here?

Ava：先说硬件。这里的“数据流”是什么意思？

Brian: A compiled Spyre kernel contains programs for several kinds of functional unit.

Brian：编译后的 Spyre 内核包含面向多种功能单元的程序。

Those units make progress when their input data arrives.

这些单元在收到输入数据后就开始工作。

Think of stations in a workshop: each starts its part when the previous station delivers the pieces.

可以把它想成车间里的工位：上一个工位送来零件，下一个工位就开始干活。

Ava: And the compiler decides how those stations work together?

Ava：编译器负责决定这些工位如何配合？

Brian: Right.

Brian：对。

It builds the device-side schedule, including movement between large device memory and each core's local scratchpad.

它会生成设备端的调度安排，包括大容量设备内存与各核心本地暂存区之间的数据传输。

The runtime submits the compiled program and its tensor arguments.

运行时负责提交编译后的程序及其张量参数。

Ava: Can the runtime launch two compute jobs at once to keep the card busy?

Ava：运行时能同时启动两个计算任务，让加速卡保持忙碌吗？

Brian: Independently submitted compute jobs share one runtime compute queue.

Brian：独立提交的计算任务共用一个运行时计算队列。

They wait behind each other.

它们得排队执行。

But transfers have separate pipelines, so data can move while a program computes.

但数据传输有独立的流水线，因此程序计算时也能传输数据。

Ava: Got it. The useful overlap is transfer with compute. How does PyTorch express that?

Ava：明白了，能重叠的是传输和计算。PyTorch 怎么表达这一点？

Brian: With streams: ordered queues of work.

Brian：通过流，也就是按顺序执行任务的队列。

PyTorch also has events, which let one stream wait for work on another.

PyTorch 还有事件，可以让一个流等待另一个流上的任务。

torch-spyre maps those ideas onto Spyre's runtime.

torch-spyre 把这些概念映射到 Spyre 的运行时。

Ava: Before streams, though, PyTorch needs to recognize Spyre.

Ava：不过在用流之前，PyTorch 得先识别 Spyre。

What gives it a device identity?

是什么赋予它设备身份的？

Brian: PrivateUse1, PyTorch's extension slot for an outside device backend.

Brian：PrivateUse1，这是 PyTorch 为外部设备后端预留的扩展槽位。

torch-spyre names that slot spyre and registers a device module.

torch-spyre 把这个槽位命名为 spyre，并注册设备模块。

With allocator, copy, and dispatcher support, a tensor can move to Spyre and operations can reach Spyre kernels.

有了分配器、拷贝和分发器支持，张量就能移到 Spyre 上，操作也能调用 Spyre 内核。

Ava: Does that also explain why ordinary PyTorch streams can work?

Ava：这也解释了为什么普通的 PyTorch 流能用吗？

Brian: Yes. The backend implements the device-guard hooks PyTorch asks for.

Brian：是的。后端实现了 PyTorch 所需的设备守卫钩子。

Then native stream creation and stream selection use PyTorch's own interface.

之后就能通过 PyTorch 自己的接口创建和选择原生流。

Ava: Let's talk about tensors staying put. What went wrong with the earlier approach?

Ava：再说说张量留在设备上。之前的做法有什么问题？

Brian: The earlier custom compiler backend could keep selected model state on Spyre through special runtime paths.

Brian：之前的自定义编译器后端能通过特殊的运行时路径，把部分模型状态保留在 Spyre 上。

But PyTorch still saw ordinary tensors as CPU tensors, even when the runtime had moved their data.

但在 PyTorch 看来，普通张量仍是 CPU 张量，即使运行时已经搬走了它们的数据。

Ava: That sounds expensive for eager execution, where operations happen one at a time.

Ava：对逐个执行操作的即时执行模式来说，听起来开销很大。

Brian: It was a poor fit. Each small operation could become another transfer boundary.

Brian：确实不合适。每个小操作都可能产生一次额外的数据传输。

A native Spyre tensor instead has device storage managed through PyTorch's allocator and storage lifetime.

原生 Spyre 张量则使用设备存储，由 PyTorch 的分配器管理内存及其生命周期。

Ava: So PyTorch knows the tensor is resident, and it knows when to release its memory?

Ava：所以 PyTorch 知道张量留在设备上，也知道何时释放内存？

Brian: Exactly.

Brian：正是。

When the last reference goes away, PyTorch calls the backend allocator to release or recycle the allocation.

最后一个引用消失时，PyTorch 会调用后端分配器，释放或回收那块内存。

Ava: Is Spyre memory just one big set of pointers?

Ava：Spyre 内存就是一大堆指针吗？

Brian: Underneath the allocator interface, memory comes in regions, each with a handle.

Brian：在分配器接口之下，内存按区域划分，每个区域都有一个句柄。

The allocator takes a small number of large regions and divides them into aligned blocks.

分配器取少量大内存区域，将其切分为对齐的内存块。

Many tensors can then share a few handles.

这样，许多张量就能共用少数几个句柄。

Ava: Why keep the number of handles low?

Ava：为什么要尽量少用句柄？

Brian: The handle budget is limited, especially when tenants share a card.

Brian：句柄数量有限，尤其是在多个租户共用一张卡时。

An allocation is described by a region, an offset, and a length.

一块分配的内存由区域、偏移量和长度来描述。

A tensor spread across memory domains may need several such pieces.

如果张量分布在多个内存域，就可能需要好几个这样的片段。

Ava: Memory domains? Are we talking about several cards?

Ava：内存域？是说有好几张卡吗？

Brian: No, locality within one device.

Brian：不是，是同一设备内的局部性。

Some cores reach certain memory domains more efficiently.

有些核心访问特定内存域时效率更高。

A tensor can sit near the cores using it or be spread across domains for bandwidth.

张量可以放在使用它的计算核心附近，也可以分布在多个域中以提高带宽。

That placement matters because compiled code may depend on it.

放在哪里很重要，因为编译后的代码可能依赖这个位置。

Ava: And layout matters too, I assume?

Ava：布局也很重要吧？

Brian: It does. Compiled kernels expect particular layouts.

Brian：对。编译后的计算核要求特定的布局。

Model adapters prepare weights and key-value caches, the stored attention state, in those layouts.

模型适配器会按这些布局准备权重和键值缓存，也就是保存的注意力状态。

Compiler passes add legal conversions for intermediate tensors.

编译器的处理阶段会为中间张量加入合法的布局转换。

Ava: Let me recap: PyTorch owns tensor lifetime, while the Spyre allocator describes the real device placement.

Ava：我总结一下：PyTorch 管理张量的生命周期，Spyre 分配器则描述它在设备上的实际位置。

Brian: That's it.

Brian：没错。

Now the next problem is launching a compiled program with tensors whose addresses weren't known when it was compiled.

接下来的问题是：张量地址在编译时还不知道，怎样用这些张量启动编译好的程序。

Ava: Why weren't they known?

Ava：为什么当时不知道地址？

Brian: The compiler fixes the program's layout ahead of time.

Brian：编译器会提前确定程序布局。

But PyTorch's allocator places caller-owned input and output tensors later.

但 PyTorch 分配器要到之后才会放置调用方持有的输入和输出张量。

Their addresses must stay symbolic until launch.

因此，它们的地址在启动前必须保持为符号值。

Ava: What did the earlier runtime do?

Ava：早期的运行时怎么处理？

Brian: One path copied inputs into buffers owned by the compiled job, then copied results out.

Brian：一种方式是把输入复制到编译任务自己的缓冲区，再把结果复制出来。

Another bound a small set of device address windows to tensors.

另一种方式是把少量设备地址窗口绑定到张量。

That avoided copies, but the number of windows limited how many tensor inputs a fused kernel could take.

这样省去了复制，但窗口数量限制了融合计算核能接收的张量输入数。

Ava: What's the current solution?

Ava：现在怎么解决？

Brian: Launch-time patching. The compiled binary has placeholders for tensor addresses.

Brian：在启动时修补地址。编译后的二进制程序里留有张量地址占位符。

At launch, a CPU callback writes the actual addresses into a pinned host buffer.

启动时，CPU 回调会把实际地址写入锁页主机缓冲区。

A transfer moves that correction data to the program's device allocation.

然后通过传输，把这些修补数据送到程序在设备上分配的内存。

Then the device uses it to patch the operands.

设备再用这些数据修补操作数。

Ava: Like filling in delivery addresses after the packages are packed?

Ava：就像包裹打好后再填收货地址？

Brian: Pretty much. The package layout is fixed; the destinations arrive later.

Brian：差不多。包裹怎么装是固定的，目的地稍后才知道。

Those three steps need strict ordering, or the program could read an address before it's ready.

这三步必须严格按顺序执行，否则程序可能读到尚未准备好的地址。

Ava: Would putting everything on one stream solve that?

Ava：把所有步骤放在同一个流上能解决吗？

Brian: It would preserve order, but it would also make preparation wait while compute runs.

Brian：能保证顺序，但计算运行时，准备工作也得等着。

The initial design uses a preparation stream for the CPU callback and transfer, and a device stream for compute.

初版设计用一个准备流执行 CPU 回调和传输，另一个设备流执行计算。

An event makes compute wait for its transfer.

事件会让计算等待对应的传输完成。

Ava: Then preparation for the next invocation can happen during the current computation?

Ava：那下一次调用的准备工作就能和当前计算同时进行？

Brian: Yes. Its tensor addresses are already known at launch.

Brian：对。启动时，下一次调用的张量地址已经确定了。

A transfer can overlap too, provided it targets device storage the current computation isn't using.

传输也可以重叠进行，前提是它写入的设备存储区未被当前计算使用。

Ava: What if both invocations use the same device correction area?

Ava：如果两次调用都用同一块设备端修补区呢？

Brian: Then the next transfer must wait until the earlier compute finishes reading that area.

Brian：下一次传输就必须等前一次计算读完那块区域。

There's also a separate hardware detail: Spyre can fetch a program early, before its patch reaches memory.

还有一个硬件细节：Spyre 可能在修补数据写入内存前，就提前读取程序。

A barrier in program distribution holds that fetch until the patch has landed.

程序分发环节的屏障会让这次读取等到修补数据写入完成。

Ava: So queue order alone doesn't prevent that early fetch. That's a subtle one.

Ava：所以光靠队列顺序挡不住提前读取。这个细节挺微妙。

Brian: It is. The host staging buffer has its own reuse rule.

Brian：是的。主机暂存缓冲区也有自己的复用规则。

One preparation stream can reuse it because each transfer finishes reading it before the next callback overwrites it.

使用单个准备流时可以复用它，因为每次传输都会在下一次回调覆盖缓冲区前读完。

More preparation streams would need separate buffers or a pool that waits for completion.

如果有更多准备流，就需要各自的缓冲区，或一个等待传输完成后才复用的缓冲池。

Ava: You mentioned events. Can a PyTorch user record and wait on Spyre events today?

Ava：你提到了事件。现在 PyTorch 用户能记录和等待 Spyre 事件吗？

Brian: The runtime uses software events internally and can derive dependencies when operations' device-memory reads and writes conflict.

Brian：运行时内部使用软件事件；当操作对设备内存的读写发生冲突时，也能推导出依赖关系。

User-callable event recording and waiting through PyTorch are planned, but aren't implemented yet.

计划让用户通过 PyTorch 记录和等待事件，但目前还没实现。

Users can synchronize a stream or the accelerator to wait for completion.

用户可以同步单个流或整个加速器，等待操作完成。

Ava: Let's zoom out. Where does the compiler fit, and why did they remove a runtime graph?

Ava：回头看整体流程，编译器处于什么位置？他们为什么移除了运行时图？

Brian: FX graphs, PyTorch's captured computation graphs, stay in Inductor, PyTorch's compiler pipeline.

Brian：FX 图，也就是 PyTorch 捕获的计算图，保留在 PyTorch 的编译流程 Inductor 中。

The earlier integration built a second, backend-specific graph after compilation.

早期集成方案在编译后还会构建一张后端专用图。

Even a single launch had to read and rebuild graph information around its tensor arguments.

即使只启动一次，也得围绕张量参数读取并重建图信息。

Ava: So they prepare the compiled artifact once?

Ava：所以他们只准备一次编译产物？

Brian: Yes.

Brian：对。

Preparation parses the artifact, allocates and transfers its program binary, and makes an ordered recipe of typed steps.

准备阶段解析产物，分配并传输程序二进制文件，再生成一份按顺序排列、标明类型的步骤清单。

Each launch then builds and enqueues the needed host work, transfers, and compute operations.

每次启动时，再构建并提交所需的主机端任务、数据传输和计算操作。

Ava: An ordered recipe is easier to picture than another graph.

Ava：有序的步骤清单比另一张图更容易理解。

What does that change for eager operations?

这对即时执行的操作有什么影响？

Brian: torch-spyre can use the same decomposition in compiled and eager execution.

Brian：torch-spyre 在编译执行和即时执行中可以使用同一套操作拆解方式。

Inside a compile trace, PyTorch captures that decomposition in the larger FX graph.

在编译追踪期间，PyTorch 会把这些拆解后的操作捕获到更大的 FX 图中。

Outside a trace, the first eager call compiles and caches it; later calls reuse the compiled entry point.

不在追踪期间时，首次即时调用会编译并缓存它；后续调用则复用编译后的入口。

Ava: So an eager operation can also test the whole path?

Ava：所以单个即时执行的操作也能测试整条流程？

Brian: Right.

Brian：对。

Even one operation exercises real tensors, device memory, compilation, and launch.

即使只有一个操作，也会用到真实张量、设备内存、编译和启动流程。

If that operation works alone but fails inside a model, the problem is more likely in how operations are combined.

如果它单独运行正常，却在模型中失败，问题更可能出在操作的组合方式上。

Ava: What results does the article report?

Ava：文章报告了哪些结果？

Brian: On Granite three point three eight B, with batch size one and sequence length one thousand twenty-four, removing graph construction from transfers alone improved prefill by nine point nine percent and decode by thirteen point six percent.

Brian：在批大小为 1、序列长度为 1024 的 Granite 3.3 8B 上，仅去掉传输环节的图构建，就让预填充提速 9.9%，解码提速 13.6%。

Ava: And when compute dispatch stopped using that runtime graph too?

Ava：如果计算调度也不再使用运行时图呢？

Brian: The reported totals were one point seven times faster prefill and two point four times faster decode.

Brian：报告的总体结果是预填充速度提高到 1.7 倍，解码速度提高到 2.4 倍。

Roughly three quarters of the gain came from the compute path.

约四分之三的性能提升来自计算路径。

Those numbers describe that model and setup.

这些数字只针对该模型和测试配置。

Ava: What still needs work?

Ava：还有哪些工作要做？

Brian: First-use eager calls still compile.

Brian：首次即时调用仍需编译。

Unsupported operations may fail or fall back, and different tensor sizes may need another compiled artifact.

不支持的操作可能失败或回退，不同的张量尺寸也可能需要另一个编译产物。

Native device support doesn't automatically deliver operator coverage, profiling, distributed collectives, or great kernels for every model.

原生设备支持并不会自动带来全面的算子支持、性能分析、分布式集合通信，或适合每个模型的高效内核。

Ava: Is there a plan for changing tensor sizes?

Ava：张量尺寸变化有应对方案吗？

Brian: The article describes future support for passing actual variable dimension sizes through the same launch-time correction mechanism as addresses.

Brian：文章提出，未来可通过与地址相同的启动时修补机制，传入可变维度的实际大小。

Within supported bounds and layout rules, that could let one artifact serve multiple sizes.

只要符合支持的范围和布局规则，一个产物就可能适用于多种尺寸。

Ava: Does any of this flow back into PyTorch?

Ava：这些工作会回馈给 PyTorch 吗？

Brian: The team contributes to openreg, PyTorch's reference backend for outside devices.

Brian：团队也为 openreg 做贡献。它是 PyTorch 面向外部设备的参考后端。

It also tests upstream changes against Spyre through the Cross-Repository CI Relay and plans to contribute reusable tools for adapting upstream tests.

团队还通过 Cross-Repository CI Relay 在 Spyre 上测试上游改动，并计划贡献可复用的工具，帮助适配上游测试。

Ava: Okay, three-point recap.

Ava：好，用三点总结。

First, Spyre becomes a real PyTorch device, so tensors have managed storage and can stay resident.

第一，Spyre 成为真正的 PyTorch 设备，张量有了受管理的存储空间，也能常驻设备。

Brian: Second, compiled artifacts become prepared recipes.

Brian：第二，编译产物变成预先准备好的步骤清单。

Launch-time patching supplies tensor addresses, and events order work across streams.

启动时修补机制填入张量地址，事件则安排跨流任务的执行顺序。

Ava: Third, that design removes runtime graph work and opens room for transfer and compute overlap, though coverage and public events still need work.

Ava：第三，这种设计省去了运行时的图处理，也为传输与计算重叠执行创造了条件，不过操作支持范围和公开事件机制仍需完善。

Brian: That's Spyre's PyTorch story for today. Thanks for listening.

Brian：以上就是今天 Spyre 与 PyTorch 的故事。感谢收听。

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
