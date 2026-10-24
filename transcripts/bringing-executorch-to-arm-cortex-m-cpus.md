# PyTorch on Tiny Arm CPUs: Inside the ExecuTorch Cortex-M Backend

原文：[Bringing ExecuTorch to Arm Cortex-M CPUs - PyTorch](https://pytorch.org/blog/bringing-executorch-to-arm-cortex-m-cpus/)

## 摘要

本期讨论 ExecuTorch 如何让 PyTorch 模型经过导出、量化和编译后，在 Arm Cortex-M 微控制器的 CPU 上运行，省去转换到其他框架的步骤。Brian 解释了 CMSIS-NN 加速、按芯片能力选择内核，以及编译期优化和静态内存规划。两位主持人也梳理了四个 MLPerf Tiny 模型的测试结果、人员检测演示，以及 CPU 与 Ethos-U NPU 混合执行的方式。最后，他们说明了测试的适用范围和文章列出的后续方向。

## 对话

Ava: What if the PyTorch model you trained could reach a tiny, always-on device without a detour through another framework? That could save a lot of debugging.

Ava：如果你训练好的 PyTorch 模型无需绕道其他框架，就能部署到一台微型、始终运行的设备上，会怎么样？那能省去大量调试工作。

Brian: That's the promise here. ExecuTorch now has a Cortex-M backend for Arm microcontroller CPUs. It keeps export and deployment in the PyTorch ecosystem.

Brian：这正是这里要实现的目标。ExecuTorch 现在为 Arm 微控制器 CPU 提供了 Cortex-M 后端，让模型导出和部署都留在 PyTorch 生态中。

Ava: Tiny devices sound like a tough place for a model. What kind of hardware are we talking about?

Ava：微型设备听起来很难运行模型。我们说的是哪类硬件？

Brian: Microcontroller units, or MCUs, in things like wearables, sensors, and voice interfaces. They often have less than one megabyte of working memory and very limited flash storage.

Brian：是微控制器单元，也就是 MCU，常见于可穿戴设备、传感器和语音交互设备。它们的工作内存通常不到一兆字节，闪存存储空间也非常有限。

Ava: So the old workflow was to train in PyTorch, then convert the model for deployment?

Ava：所以以前的流程是在 PyTorch 中训练，然后转换模型用于部署？

Brian: Right. That meant another format and another toolchain. If the converted model gave different numbers, you had to track down where the mismatch appeared.

Brian：对。这意味着要使用另一种格式和另一套工具链。如果转换后的模型算出了不同的数值，你还得追查差异出现在哪里。

Ava: When you say it runs directly on Cortex-M, you don't mean running ordinary PyTorch Python code on the chip, do you?

Ava：你说它直接在 Cortex-M 上运行，并不是指在芯片上运行普通的 PyTorch Python 代码，对吧？

Brian: No. You export the PyTorch model, quantize it, and compile it for the target. The device runs that prepared graph through the ExecuTorch runtime.

Brian：不是。你要导出 PyTorch 模型、进行量化，再针对目标设备编译。设备通过 ExecuTorch 运行时执行准备好的计算图。

Ava: Good distinction. What does the new backend actually change?

Ava：这个区别很重要。新后端具体改变了什么？

Brian: It replaces supported operators during compilation with versions backed by CMSIS-NN. That's Arm's open-source library of optimized neural network kernels for Cortex-M processors.

Brian：它在编译期间将受支持的算子替换为基于 CMSIS-NN 的实现。CMSIS-NN 是 Arm 为 Cortex-M 处理器提供的开源优化神经网络内核库。

Ava: So if my graph has a convolution, the backend can choose a Cortex-M-friendly implementation before the device ever sees it?

Ava：所以，如果我的计算图里有卷积，后端可以在设备接收模型之前，就选好适合 Cortex-M 的实现？

Brian: Exactly. The article mentions operations such as convolution, linear layers, addition, multiplication, softmax, and activations. The replacements happen ahead of time, during compilation.

Brian：没错。文章提到了卷积、线性层、加法、乘法、softmax 和激活函数等运算。这些替换会提前在编译期间完成。

Ava: And quantization is part of that path. Could you unpack it for someone listening while commuting?

Ava：量化也是这条流程的一部分。能不能用方便通勤路上听懂的方式解释一下？

Brian: Here it means preparing an INT8 graph, using eight-bit integers for the quantized operations. The CortexMQuantizer uses PyTorch's PT2E quantization framework to produce layouts the kernels expect.

Brian：这里的意思是准备一个 INT8 计算图，让量化运算使用八位整数。CortexMQuantizer 使用 PyTorch 的 PT2E 量化框架，生成这些内核所要求的数据布局。

Ava: Does one build work across every Cortex-M chip, or do developers have to describe their exact target?

Ava：一次构建就能适用于所有 Cortex-M 芯片吗？还是开发者必须说明具体的目标芯片？

Brian: They specify the CPU and its instruction set capabilities at compile time. The backend then selects the matching implementation. There's no need to decide between those variants during inference.

Brian：他们需要在编译时指定 CPU 及其指令集能力，后端再选择匹配的实现。推理时就不用再从这些实现中作选择了。

Ava: The article divides those capabilities into three tiers. Can we translate that table into normal speech?

Ava：文章把这些能力分成三个层级。我们能把那张表说得通俗一点吗？

Brian: Sure. Scalar covers cores such as M0 and M3, using standard Arm instructions. DSP adds instructions useful for signal processing. MVE, also called Helium, brings vector operations.

Brian：当然。Scalar 涵盖 M0 和 M3 等核心，使用标准 Arm 指令。DSP 增加了适用于信号处理的指令。MVE 也叫 Helium，带来了向量运算。

Ava: And the M55 in the measurements is in that Helium tier?

Ava：测量中使用的 M55 属于 Helium 这一层级？

Brian: Yes. The M55 and M85 are the MVE examples. CMSIS-NN can use Helium's execution model to move data and compute efficiently, especially for depthwise and pointwise convolutions.

Brian：对。M55 和 M85 是 MVE 的例子。CMSIS-NN 可以利用 Helium 的执行方式高效地搬运数据和计算，尤其是在深度卷积和逐点卷积中。

Ava: But the same developer workflow still targets the smaller cores?

Ava：但同一套开发流程也能面向更小的核心？

Brian: That's the idea. You keep the workflow and change the target configuration. The selected kernels reflect whether that target has Scalar, DSP, or MVE capabilities.

Brian：思路就是这样。开发流程保持一致，只需更改目标配置。选用的内核会反映目标设备具备的是 Scalar、DSP 还是 MVE 能力。

Ava: Let's pause there. The graph is prepared once for a known chip, instead of asking the chip to make those choices each time.

Ava：我们在这里停一下。针对一颗已知芯片，计算图会预先准备好一次，而不是让芯片每次运行时都作这些选择。

Brian: Exactly. Embedded developers usually know the CPU, memory regions, and available instructions in advance. ExecuTorch uses that information while it compiles the model.

Brian：正是如此。嵌入式开发者通常事先就知道 CPU、内存区域和可用指令。ExecuTorch 在编译模型时会利用这些信息。

Ava: What work can it move out of inference?

Ava：它能把哪些工作从推理阶段提前处理？

Brian: For one thing, it removes redundant conversions and transposes from the graph. Think of it as clearing unnecessary turns from a delivery route before the driver leaves.

Brian：其中一项是从计算图中去掉多余的转换和转置。你可以把它想成司机出发前，先从配送路线中删掉不必要的绕行。

Ava: The article also mentions folding quantization parameters. What's being folded?

Ava：文章还提到折叠量化参数。具体折叠的是什么？

Brian: Scale and zero-point arithmetic is prepared as integer multipliers and shifts during export. That lets those quantized kernels stay in fixed-point arithmetic while inference runs.

Brian：导出期间，缩放系数和零点相关的运算会预先处理成整数乘数和移位。这样，量化内核在推理时就能保持使用定点运算。

Ava: And there's a weight layout change too, right?

Ava：权重的布局也会改变，对吧？

Brian: Yes. PyTorch and CMSIS-NN expect convolution weights in different orders. The compiler transposes those weights once during export, so the device doesn't repeat the work.

Brian：对。PyTorch 和 CMSIS-NN 对卷积权重的排列顺序要求不同。编译器在导出期间只转置一次权重，设备就不用重复做这项工作。

Ava: Memory feels like the other big constraint. How does the runtime avoid surprises there?

Ava：内存似乎是另一项重大限制。运行时怎样避免在这方面出现意外？

Brian: It plans tensor and scratch-buffer locations ahead of time. It can place them across memory regions, including fast tightly coupled memory and SRAM. Developers can customize where important buffers go.

Brian：它会提前规划张量和临时缓冲区的位置，可以把它们放在不同的内存区域，包括高速紧耦合内存和 SRAM。开发者也可以自定义重要缓冲区的放置位置。

Ava: So it already knows where the working space will be?

Ava：所以它事先就知道工作空间在哪里？

Brian: Right. Tensor inference doesn't need dynamic allocation. That gives bounded memory use and avoids heap fragmentation during repeated runs.

Brian：对。张量推理不需要动态分配内存，因此内存使用量有明确上限，也能避免反复运行时出现堆碎片。

Ava: Walk me through the workflow without reading Python aloud.

Ava：给我讲讲整个流程，不用照着 Python 代码念。

Brian: Start with an evaluation model and example inputs. Export it, prepare it for quantization, run calibration samples, convert it, and export the quantized version again.

Brian：先准备一个处于评估模式的模型和示例输入。导出模型，为量化做好准备，用校准样本进行校准，完成量化转换，然后再次导出量化后的版本。

Ava: Then you tell the compiler which Cortex-M chip it's aiming for?

Ava：接着告诉编译器目标是哪颗 Cortex-M 芯片？

Brian: Yes. You lower the exported graph to ExecuTorch's edge representation and apply the Cortex-M passes. Finally, you serialize the program for the firmware.

Brian：对。你把导出的计算图转换为 ExecuTorch 的 edge 表示形式，并应用 Cortex-M 编译处理。最后，把程序序列化，供固件使用。

Ava: Where does that serialized program go?

Ava：序列化后的程序放在哪里？

Brian: The resulting PTE binary is linked into firmware alongside the ExecuTorch runtime and CMSIS-NN library, then flashed to the device. Inference runs the precompiled graph.

Brian：生成的 PTE 二进制文件会与 ExecuTorch 运行时和 CMSIS-NN 库一起链接进固件，然后烧录到设备上。推理时执行的就是预先编译好的计算图。

Ava: Wait, some of these chips also have an Ethos-U accelerator. Does the CPU backend compete with that path?

Ava：等等，其中一些芯片还有 Ethos-U 加速器。CPU 后端会和那条部署路径竞争吗？

Brian: It complements it. An Ethos-U NPU, or neural processing unit, can take supported operations. Operations left outside its partition can run on the Cortex-M CPU in the same graph.

Brian：它们是互补的。Ethos-U NPU，也就是神经处理单元，可以处理受支持的运算。同一计算图中，留在它的分区之外的运算可以在 Cortex-M CPU 上运行。

Ava: What decides whether an operation left on the CPU gets CMSIS-NN acceleration?

Ava：留在 CPU 上的运算是否能获得 CMSIS-NN 加速，由什么决定？

Brian: The Cortex-M passes run after Ethos-U partitioning. They can reuse the graph's quantization parameters when the operation meets the backend's quantization, layout, and shape requirements.

Brian：Cortex-M 编译处理在 Ethos-U 分区完成后运行。如果某项运算满足后端对量化、布局和形状的要求，这些处理就能复用计算图中的量化参数。

Ava: And if it doesn't meet those requirements?

Ava：如果不满足这些要求呢？

Brian: It falls back to ExecuTorch's portable kernels. Those work, but the article says they aren't accelerated. That detail matters when you're estimating performance.

Brian：它会回退到 ExecuTorch 的可移植内核。这些内核可以运行，但文章说它们没有加速。估算性能时，这一点很重要。

Ava: So an NPU doesn't have to cover every operation before a mixed CPU and NPU model can run.

Ava：所以，不必让 NPU 覆盖每一项运算，CPU 和 NPU 混合执行的模型也能运行。

Brian: Exactly. That's the practical benefit the article highlights. A few operations outside the NPU partition don't have to block deployment.

Brian：正是如此。这就是文章强调的实际好处。即使有少数运算不在 NPU 分区中，也不会因此挡住部署。

Ava: Let's get to measurements. What did they actually run?

Ava：我们来看看测量结果。他们实际运行了什么？

Brian: Four of the five MLPerf Tiny benchmark models, all quantized to INT8. They used one Cortex-M55 core on an Alif Ensemble E8 at four hundred megahertz.

Brian：他们运行了五个 MLPerf Tiny 基准模型中的四个，全部量化为 INT8。使用的是 Alif Ensemble E8 上的一颗 Cortex-M55 核心，运行频率为四百兆赫兹。

Ava: The Alif chip has Ethos-U units too. Were those helping?

Ava：Alif 芯片也有 Ethos-U 单元。它们起作用了吗？

Brian: No. The NPUs were unused for these measurements. That makes this a useful look at the Cortex-M CPU path on that particular board.

Brian：没有。这些测量没有使用 NPU。因此，这些数据很适合用来了解那块特定开发板上的 Cortex-M CPU 路径。

Ava: How quick were the models?

Ava：这些模型运行得有多快？

Brian: Every one finished within twenty-one point five milliseconds. Keyword spotting took about five point eight milliseconds. Visual wake words took twenty-one point four five milliseconds.

Brian：每个模型都在 21.5 毫秒内完成推理。关键词识别用了约 5.8 毫秒，视觉唤醒词识别用了 21.45 毫秒。

Ava: And the other two?

Ava：另外两个呢？

Brian: Image classification took about fifteen point three milliseconds. The anomaly detection autoencoder took about one point two milliseconds. Those are per-inference latencies.

Brian：图像分类用了约 15.3 毫秒，异常检测自编码器用了约 1.2 毫秒。这些都是单次推理的延迟。

Ava: What about memory? A small latency number won't help if the model doesn't fit.

Ava：内存占用呢？如果模型装不下，延迟再低也没用。

Brian: All four used less than ninety kibibytes of peak RAM. Three used less than sixty. The autoencoder needed just five point two kibibytes.

Brian：四个模型的峰值 RAM 占用都低于 90 KiB，其中三个低于 60 KiB。自编码器只需要 5.2 KiB。

Ava: The article also talks about a selective build. What gets selected?

Ava：文章还提到了选择性构建。具体选择的是什么？

Brian: Only the operators a model needs are linked into its firmware. The ExecuTorch core runtime is under twenty kibibytes of flash, and measured code footprints ranged from about fifteen to thirty-seven.

Brian：只有模型需要的算子才会被链接进固件。ExecuTorch 核心运行时占用的闪存不到 20 KiB，测得的代码占用空间约为 15 到 37 KiB。

Ava: That's code size, though. The model itself also occupies flash.

Ava：但那只是代码大小，模型本身也会占用闪存。

Brian: Right. For example, the visual wake words model used about two hundred fifty-three kibibytes for the model and thirty-six for code. The total was about two hundred eighty-nine.

Brian：对。例如，视觉唤醒词模型本身占用约 253 KiB，代码占用约 36 KiB，总计约 289 KiB。

Ava: Can we compare those timings with the official MLPerf Tiny leaderboard?

Ava：我们能把这些运行时间和官方的 MLPerf Tiny 排行榜比较吗？

Brian: The article says no. These are the team's own measurements, using the Alif evaluation kit as a test harness. They weren't an official MLPerf Tiny submission.

Brian：文章说不能。这些是团队以 Alif 评估套件作为测试平台自行测得的数据，并非正式提交给 MLPerf Tiny 的结果。

Ava: There's also a person detection demo. What did that add beyond a benchmark table?

Ava：还有一个人员检测演示。除了基准测试表格，它展示了什么？

Brian: They deployed a µYOLO person detector on a Renesas board with a Cortex-M85. It takes a one hundred twenty-eight by one hundred twenty-eight RGB camera frame.

Brian：他们在一块搭载 Cortex-M85 的 Renesas 开发板上部署了 µYOLO 人员检测器。它接收来自摄像头的 128 × 128 RGB 画面。

Ava: How big and how fast was that deployment?

Ava：这项部署占用多大空间，运行得多快？

Brian: The INT8 model file was four hundred fifty-two kibibytes. Total flash was six hundred forty kibibytes, peak RAM was two hundred fifty-four, and inference took twenty-four milliseconds per frame.

Brian：INT8 模型文件为 452 KiB，总闪存占用为 640 KiB，峰值 RAM 占用为 254 KiB，每帧推理耗时 24 毫秒。

Ava: But the end-to-end rate was twelve point seven frames per second. That's a different measure from model inference time, isn't it?

Ava：但端到端的帧率是每秒 12.7 帧。这和模型推理时间是不同的指标，对吧？

Brian: Yes. The article reports both numbers separately. The frame rate covers the end-to-end demo, while twenty-four milliseconds describes one model inference.

Brian：对。文章分别报告了这两个数字。帧率衡量的是整个演示的端到端速度，而 24 毫秒指的是单次模型推理。

Ava: I noticed that twenty-four milliseconds came after some hardware-specific tuning.

Ava：我注意到，24 毫秒这个结果是在做了一些针对硬件的调优之后才达到的。

Brian: It did. They put weights for the most expensive operations in fast local memory and enabled the instruction cache. Reported inference time fell from one hundred ninety-one to twenty-four milliseconds.

Brian：没错。他们把计算开销最大的操作所用的权重放进快速本地内存，并启用了指令缓存。据报告，推理时间从 191 毫秒降到了 24 毫秒。

Ava: Nearly eight times faster. That's a strong reminder that buffer placement still matters on these chips.

Ava：快了将近 8 倍。这有力地提醒我们，在这类芯片上，缓冲区放在哪里仍然很重要。

Brian: It does in this demo. The workflow helps with inspection too: intermediate representations remain executable, so developers can compare outputs as the model moves through export and lowering.

Brian：在这个演示中确实如此。这套工作流程也方便检查：中间表示仍然可以执行，因此开发者能在模型经过导出和降低层级的过程中比较输出。

Ava: What can they inspect when performance or behavior changes?

Ava：当性能或行为发生变化时，他们能检查什么？

Brian: The article names Model Explorer for viewing graphs and ETDump for per-operator costs on the target. Those tools help locate a change before treating deployment as a mystery.

Brian：文章提到，可以用 Model Explorer 查看计算图，用 ETDump 查看目标设备上每个算子的开销。这些工具有助于定位变化的原因，免得把部署过程当成一团谜。

Ava: What's available now, and what's still on the roadmap?

Ava：现在已经有哪些功能，哪些还在规划中？

Brian: The backend and examples are available in the ExecuTorch repository. The article lists broader operator coverage, more mixed-precision support, floating-point kernels, more examples, and validation on more hardware as future work.

Brian：后端和示例已经在 ExecuTorch 仓库中提供。文章将覆盖更多算子、支持更多混合精度、加入浮点内核、提供更多示例，以及在更多硬件上验证列为未来工作。

Ava: The fifth MLPerf Tiny model wasn't in these CPU measurements. Why?

Ava：第 5 个 MLPerf Tiny 模型为什么没有纳入这些 CPU 测量？

Brian: It's a streaming wake word benchmark. The article says it still needs test infrastructure for multi-frame inference. So the reported measurements cover four models.

Brian：那是一个流式唤醒词基准测试。文章说，它仍需要支持多帧推理的测试基础设施。因此，报告的测量结果只涵盖 4 个模型。

Ava: Let me try a three-point recap. First, the model stays in the PyTorch-based export workflow, then becomes a compiled program for the microcontroller.

Ava：我试着用 3 点总结一下。第一，模型沿用基于 PyTorch 的导出工作流程，然后变成供微控制器运行的编译后程序。

Brian: Second, compile-time choices handle kernels, layouts, and memory. Third, the CPU path works alone or alongside Ethos-U, with performance depending on the target and supported operations.

Brian：第二，在编译时决定内核、布局和内存安排。第三，CPU 路径既可以独立工作，也可以与 Ethos-U 协同工作，性能取决于目标硬件和受支持的操作。

Ava: That's our tour of PyTorch on Cortex-M. Thanks for listening, and we'll catch you next time.

Ava：这就是我们对 Cortex-M 上运行 PyTorch 的介绍。感谢收听，我们下次再见。

## 术语

| Term | 释义 |
|---|---|
| ExecuTorch | PyTorch 的设备端模型部署与推理框架。 |
| Cortex-M backend | 让 ExecuTorch 模型针对 Arm Cortex-M CPU 编译并执行的后端。 |
| MCU | 微控制器单元，常用于传感器、可穿戴设备等资源受限设备。 |
| CMSIS-NN | Arm 提供的开源 Cortex-M 神经网络优化内核库。 |
| INT8 | 八位整数格式，文中用于量化模型与运算。 |
| CortexMQuantizer | 为 Cortex-M 后端生成符合其内核布局要求的量化图的量化器。 |
| PT2E quantization | 文中用于处理导出模型量化的 PyTorch 框架。 |
| MVE | 文中也称 Helium 的 Cortex-M 向量指令能力。 |
| ahead-of-time compilation | 在设备运行推理前完成图优化、内核选择等编译工作。 |
| static memory planning | 提前确定张量和临时缓冲区的内存位置与用量。 |
| Ethos-U NPU | 可承担受支持模型运算的 Arm 神经处理加速器。 |
| operator selective build | 只链接模型所需算子的构建方式，用于控制代码体积。 |
| ETDump | 用于查看目标设备上各算子执行开销的分析工具。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What kind of hardware are we talking about? | 我们说的是哪一类硬件？ |
| Could you unpack it? | 你能把它解释清楚一点吗？ |
| Let's pause there. | 我们先在这里停一下，梳理刚才的内容。 |
| Walk me through the workflow. | 按步骤带我过一下这个流程。 |
| Wait, some of these chips also have an Ethos-U accelerator. | 等等，其中一些芯片还有 Ethos-U 加速器。 |
| That's the practical benefit. | 这就是实际使用中的好处。 |
| What did they actually run? | 他们实际运行了什么？ |
| That's a different measure from model inference time, isn't it? | 这和模型推理时间是不同的指标，对吧？ |
| Let me try a three-point recap. | 我试着用三点总结一下。 |

## 中文内容总结

ExecuTorch 新增 Cortex-M CPU 后端，使开发者能在 PyTorch 体系内完成模型导出、量化和设备部署，无须转换到其他框架格式。该后端以 CMSIS-NN 加速推理，也可与已有的 Ethos-U NPU 委托执行配合。

### 问题与部署流程

微控制器通常只有不足 1 MB 的 SRAM、数 MB 闪存和毫瓦级功耗预算。过去，将 PyTorch 模型部署到 Cortex-M 往往需要格式转换、另一套工具链，并排查转换引入的数值差异。新流程使用 torch.export 导出模型，通过 PT2E 和 CortexMQuantizer 校准、量化为 INT8，再按目标 CPU 编译成 .pte 文件，与运行时和 CMSIS-NN 一起集成进固件。

### 核心机制

编译阶段的转换会把适用的量化算子替换为 CMSIS-NN 或其他 Cortex-M 优化实现，并按目标指令集选择内核：M0 等使用标量指令，M4 等使用 DSP，M55 和 M85 使用 MVE（Helium）。编译还会消除多余的数据类型转换和量化操作，预计算量化参数，把卷积权重从 OIHW 转为 OHWI，并预先安排张量及临时缓冲区在 TCM、SRAM 等内存中的位置。作者称这可减少运行时开销，并避免张量动态分配。对带 Ethos-U 的芯片，受支持算子可交给 NPU；留在 CPU 上的算子满足量化、布局和形状条件时使用 CMSIS-NN，否则回退到未加速的通用内核。

### 性能测量及条件

作者在 Alif Ensemble E8 的单个 400 MHz Cortex-M55 高性能核心上测试了五个 MLPerf Tiny 模型中的四个，未使用板载 NPU。四者均采用 INT8 量化和算子选择性编译。DS-CNN、MobileNet v1 0.25、ResNet-8 和异常检测自编码器的延迟分别为 5.825、21.450、15.290 和 1.208 毫秒，峰值 RAM 分别为 26.00、85.15、55.14 和 5.20 KiB。包含运行时与内核的代码闪存占用为 14.97 至 36.03 KiB，另需存放模型。编译条件包括 GCC 15.2.1、关闭日志、核心运行时使用 -Oz、内核使用 -O3。这些是作者自己的测量，不是官方 MLPerf Tiny 提交，不能直接与排行榜成绩比较。

### 演示、适用范围与后续工作

在 Cortex-M85 的 Renesas EK-RA8D1 开发板上，µYOLO 人形检测演示使用 128×128 RGB 输入，INT8 .pte 模型为 452 KiB，总闪存占用 640 KiB，峰值 RAM 为 254 KiB。将耗时最高的算子权重放入 DTCM 并启用指令缓存后，480 MHz 下单帧推理延迟从 191 毫秒降至 24 毫秒；端到端吞吐量为 12.7 FPS。仓库内的演示应用面向 Corstone FVP，开发板移植代码在仓库外。作者计划扩展混合精度、浮点内核、注意力等算子和硬件验证；尚未测试的第五个 MLPerf Tiny 流式唤醒词任务还需要多帧推理测试设施。
