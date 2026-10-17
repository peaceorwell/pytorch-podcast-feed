# Helion Meets Hugging Face Kernels: From Tuning to Shipping

原文：[Helion x 🤗 HF Kernels: Building and Shipping Out-of-the-box Performant Kernels - PyTorch](https://pytorch.org/blog/helion-x-%f0%9f%a4%97-hf-kernels-building-and-shipping-out-of-the-box-performant-kernels/)

## 摘要

这篇文章介绍 Helion 如何接入 Hugging Face Kernels，让开发者把内核源码和预调优配置一起发布到 Hub。Helion 用接近 PyTorch 的分块写法描述计算，并自动搜索分块大小、访存方式等实现选择；开发者可针对代表性形状预调优，再生成按输入形状选配置的决策树。文章展示了 H100 上的 attention 内核和 B200 上的 linear-attention 内核相对于指定基线的性能结果，也说明使用者需要安装 Helion，并使用支持该依赖的 Kernels 版本。文中的测试覆盖特定硬件与形状，不能据此推断所有场景的表现。

## 对话

Ava: You find a fast GPU kernel, a small program for a graphics processing unit. Why can sharing it still feel harder than writing it?

Ava：你找到了一个运行很快的 GPU 内核，也就是一段供图形处理器运行的小程序。为什么分享它仍然会让人觉得比编写它还难？

Brian: Packaging and installation can get messy. This article shows how Helion kernels can be tuned, published through Hugging Face Kernels, and loaded by other people.

Brian：打包和安装可能会变得很麻烦。这篇文章介绍了如何调优 Helion 内核、通过 Hugging Face Kernels 发布它们，以及让其他人加载使用。

Ava: So there are two names here. What's Helion, and what's Kernels with a capital K?

Ava：这里有两个名字。Helion 是什么？首字母大写的 Kernels 又是什么？

Brian: Helion is a domain-specific language, or DSL, for writing machine-learning kernels. Kernels is the Hugging Face project for packaging and distributing them.

Brian：Helion 是一种用于编写机器学习内核的领域特定语言，也就是 DSL。Kernels 是 Hugging Face 用来打包和分发这些内核的项目。

Ava: That split helps. Let's start with the developer. What does writing a Helion kernel actually feel like?

Ava：这样区分就清楚了。先从开发者说起。编写 Helion 内核实际上是什么体验？

Brian: The article calls it PyTorch with tiles. You work with PyTorch tensors, and ordinary PyTorch operations describe what happens inside each tile.

Brian：文章把它称为“带分块的 PyTorch”。你使用 PyTorch 张量，并用普通的 PyTorch 操作描述每个数据块内部要做什么。

Ava: A tile is a chunk of the work, right? Like cutting a large sheet into smaller pieces?

Ava：数据块就是把任务切出一部分，对吧？就像把一大张纸裁成小块？

Brian: Exactly. The matrix multiplication example walks over output tiles, accumulates products across another tiled dimension, and writes each result into an output tensor.

Brian：没错。矩阵乘法的例子会逐个处理输出数据块，沿另一个分块维度累加乘积，再把每个结果写入输出张量。

Ava: But a tile needs a size. Does the author choose that while writing the computation?

Ava：但数据块总得有个大小。作者在编写计算过程时就要选定吗？

Brian: They can leave it open. Helion's tile operation says to split the iteration space, while leaving choices such as tile size to its autotuner.

Brian：可以先不定。Helion 的 tile 操作指定要拆分迭代空间，而把数据块大小等选择留给自动调优器。

Ava: Wait, so the same source can be tried with several ways of dividing the work?

Ava：等等，也就是说，同一份源代码可以尝试好几种任务划分方式？

Brian: Yes, and the search goes further. Helion also explores lowering strategies: concrete ways to turn that source into an implementation.

Brian：是的，而且搜索范围还不止于此。Helion 也会探索底层转换策略，也就是把那份源代码变成具体实现的不同方式。

Ava: What changes when a lowering strategy changes? Give me something more tangible than a compiler diagram.

Ava：底层转换策略变了，具体会有什么变化？给我举个比编译器示意图更直观的例子。

Brian: Memory access can change. So can the order of nested loops, how loops are flattened, or whether a reduction persists instead of looping.

Brian：内存访问方式可能会变。嵌套循环的顺序、循环如何展平，以及是否让归约保持驻留而不是通过循环执行，也都可能改变。

Ava: So the written computation stays readable, while Helion experiments with how the machine carries it out.

Ava：所以写出来的计算过程仍然容易读，而 Helion 会试验机器该如何执行它。

Brian: Right. Think of giving a cook the recipe and letting them test different pan sizes and preparation orders for each batch.

Brian：对。可以想成把菜谱交给厨师，让他们针对每一批食材试用不同尺寸的锅和不同的备料顺序。

Ava: Does that search really matter, or is it mainly a convenience for the person writing the kernel?

Ava：这种搜索真的重要吗？还是主要为了让编写内核的人省事？

Brian: The authors say a Helion kernel can often beat a hand-written lower-level kernel across many shapes. They don't claim that happens every time.

Brian：作者说，对于许多不同的形状，Helion 内核往往能胜过手写的更底层内核。他们没有说每次都会如此。

Ava: Here's the catch: trying lots of implementations sounds expensive before the user gets any work done.

Ava：但问题是，用户还没开始处理任务，就尝试大量实现，听起来成本很高。

Brian: It can take a while. That's why the article focuses on tuning ahead of time and shipping the chosen configurations with the kernel.

Brian：这可能要花上一段时间。所以文章着重讲的是提前调优，并把选出的配置随内核一起发布。

Ava: Before we tune it, how does this kernel reach someone else's machine?

Ava：在讨论调优之前，这个内核怎么到达别人的机器上？

Brian: Through the Kernels project. Its developer tool, kernel-builder, gives packages a consistent structure and build process. Its Python library loads published kernels.

Brian：通过 Kernels 项目。它的开发者工具 kernel-builder 为软件包提供统一的结构和构建流程；它的 Python 库负责加载已发布的内核。

Ava: Why did Hugging Face need that shared process?

Ava：Hugging Face 为什么需要这套统一流程？

Brian: The article describes fragmented source layouts, tooling, and compatibility support. Users can face long builds even when a project offers prebuilt packages.

Brian：文章说，各项目的源代码布局、工具和兼容性支持都很分散。即使项目提供预构建包，用户仍可能遇到耗时很长的构建过程。

Ava: What's the user experience supposed to be?

Ava：它希望给用户带来什么样的使用体验？

Brian: You request a named kernel and version from the Hugging Face Hub, then call an exported function. The article uses Flash Attention three as an example.

Brian：从 Hugging Face Hub 请求指定名称和版本的内核，然后调用它导出的函数。文章以 Flash Attention 3 为例。

Ava: Does Kernels only handle source that compiles when somebody calls it?

Ava：Kernels 只能处理在有人调用时才编译的源代码吗？

Brian: No. It supports ahead-of-time and just-in-time kernels. For some ahead-of-time kernels, the project provides prebuilt binaries across a compatibility matrix.

Brian：不是。它支持提前编译和即时编译的内核。对于某些提前编译的内核，该项目会提供覆盖一系列兼容配置的预构建二进制文件。

Ava: And where does Helion fit? Do authors have to publish binaries for every machine?

Ava：那 Helion 在这里扮演什么角色？作者必须为每种机器发布二进制文件吗？

Brian: Helion kernels are Python source. They're described as noarch: authors ship the source, and Helion compiles it on the user's machine at first call.

Brian：Helion 内核是 Python 源代码。它们被描述为 noarch：作者发布源代码，Helion 则在用户首次调用时于用户的机器上编译。

Ava: So what does kernel-builder contribute when there's no binary to compile in advance?

Ava：既然没有要提前编译的二进制文件，kernel-builder 还能提供什么？

Brian: It still provides the package layout, metadata, build, and upload workflow. The author declares Helion as a Python dependency and marks the package noarch.

Brian：它仍然提供软件包布局、元数据、构建和上传流程。作者将 Helion 声明为 Python 依赖项，并把软件包标记为 noarch。

Ava: What does that dependency declaration do for a user?

Ava：这种依赖声明对用户有什么用？

Brian: When the kernel loads, the Kernels library checks whether Helion can be imported. If it can't, the user gets a clear dependency error.

Brian：加载内核时，Kernels 库会检查能否导入 Helion。如果不能，用户会收到明确的依赖项错误提示。

Ava: Can you walk through the article's vector addition example without reading code aloud?

Ava：能不照着代码念，讲讲文章里的向量加法例子吗？

Brian: The developer starts from a scaffold, removes pieces meant for compiled extensions, and keeps a small Python package. Its kernel adds matching tensor tiles.

Brian：开发者从一个项目脚手架开始，移除为编译型扩展准备的部分，保留一个小型 Python 软件包。其中的内核把对应的张量数据块相加。

Ava: The example lists CUDA, ROCm, and XPU backends. Is that the portability part?

Ava：例子列出了 CUDA、ROCm 和 XPU 后端。这就是可移植性的体现吗？

Brian: That's how the example declares its backends. The generated packages carry the source and dependency metadata, with a build directory for each backend.

Brian：例子就是这样声明后端的。生成的软件包包含源代码和依赖项元数据，每个后端各有一个构建目录。

Ava: Then the developer checks the configuration, builds it, and uploads it to the Hub?

Ava：然后开发者检查配置、构建软件包，再把它上传到 Hub？

Brian: Yes. A consumer loads the named package, then calls its vector addition function. The article's loading example also enables trust for remote code.

Brian：对。使用者加载指定名称的软件包，然后调用它的向量加法函数。文章中的加载示例还启用了对远程代码的信任。

Ava: All right. That gets us a working package. Where do the fast settings come from?

Ava：好，这样就有了能用的软件包。那些运行快的配置是怎么来的？

Brian: The simplest example pins one configuration. For multiple input shapes, the article moves to pre-tuning: measure candidate settings before publishing.

Brian：最简单的例子固定使用一套配置。面对多种输入形状时，文章转向预调优：在发布前测量候选配置的表现。

Ava: Let's slow that down. If one configuration works, why bother with several?

Ava：我们放慢一点说。如果一套配置能用，为什么还要准备好几套？

Brian: Different shapes and GPU generations may favor different choices. The workflow aims to ship configurations that perform well across the shapes the author tests.

Brian：不同的形状和不同代的 GPU 可能适合不同的选择。这套流程的目标是发布在作者测试的各种形状上都有良好表现的配置。

Ava: What does the developer change in the kernel?

Ava：开发者需要对内核做什么改动？

Brian: They mark it for Helion's ahead-of-time tuning with static shapes. Then they run a small script that calls it on representative sizes.

Brian：他们将内核标记为使用静态形状进行 Helion 的提前调优，然后运行一个小脚本，用有代表性的尺寸调用它。

Ava: Representative is doing some work there. The developer has to choose the sizes?

Ava：“有代表性”这几个字很关键。尺寸得由开发者来选？

Brian: Yes. In the vector addition walkthrough, the script calls the kernel on four sizes. Those calls become the workload for the tuning runner.

Brian：对。在向量加法的演示中，脚本用四种尺寸调用内核。这些调用就成为调优运行器的工作负载。

Ava: What does that runner actually do? I saw three phases and lost the thread.

Ava：那个运行器具体做什么？我看到三个阶段，就跟不上了。

Brian: First, collect: it autotunes each shape on its own. Then, measure: it tests every discovered configuration again on every shape.

Brian：第一步是收集：分别对每种形状进行自动调优。然后是测量：再次用每一种形状测试找到的每一套配置。

Ava: That second pass sounds useful. A setting discovered for one size might work well for another.

Ava：第二轮检查听起来很有用。为一种尺寸找到的设置，或许也适合另一种尺寸。

Brian: Exactly. Finally, build: the runner chooses a small set of configurations that stays close to each shape's own best result.

Brian：没错。最后是 build：运行程序会选出一小组配置，让每种形状的结果都接近它各自的最佳结果。

Ava: How close is close in the article's example?

Ava：在文章的例子里，“接近”具体是多接近？

Brian: Its threshold is one point zero one, described as within one percent. The example allows up to eight configurations.

Brian：阈值是 1.01，文中将其描述为在 1% 以内。这个例子最多允许 8 种配置。

Ava: Does eight mean everybody downloads eight?

Ava：8 种配置是说每个人都要下载 8 种吗？

Brian: No. That's a limit in the example. If one configuration meets the threshold for every tested shape, one is enough.

Brian：不是。那只是例子中的上限。如果一种配置在所有测试形状上都达到阈值，一种就够了。

Ava: And if the shapes disagree, we get several settings. How does a runtime call choose among them?

Ava：如果不同形状需要不同设置，就会有好几种配置。运行时调用该怎么从中选择？

Brian: Helion builds a decision tree. It's stored as a plain Python file beside the kernel source and maps an input shape to a configuration.

Brian：Helion 会构建一棵决策树。它以普通 Python 文件的形式存放在内核源代码旁边，将输入形状映射到相应的配置。

Ava: Put another way, it's like packing a route guide with the recipe.

Ava：换个说法，就像把路线指南和配方一起打包。

Brian: Yes. When someone calls the shipped kernel, Helion reads that guide and selects a pre-tuned setting for the runtime shape.

Brian：对。有人调用发布的内核时，Helion 会读取这份指南，并为运行时的输入形状选择预调优的设置。

Ava: Does that mean a user waits for another full tuning search?

Ava：这意味着用户还得等一次完整的调优搜索吗？

Brian: The article says the shipped tree avoids tuning on the user's machine. Helion still compiles the Python kernel source on its first call.

Brian：文章说，随内核发布的决策树让用户无需在自己的机器上调优。不过，Helion 在首次调用时仍会编译 Python 内核源代码。

Ava: That makes sense. Shipping decisions saves the search, even though the source still has to become a runnable kernel.

Ava：明白了。把选择结果一起发布，省去了搜索过程，但源代码仍得变成可运行的内核。

Brian: Right. The example file layout shows separate pre-tuned files for H100 and B200 devices beside the same kernel source.

Brian：对。示例文件布局显示，同一份内核源代码旁边分别放着适用于 H100 和 B200 设备的预调优文件。

Ava: We've got the mechanism. Did the article test whether these shipped settings help?

Ava：机制我们弄清楚了。文章有没有测试这些发布的设置是否有帮助？

Brian: It gives two groups of results. The first is a pre-tuned attention kernel on an NVIDIA H100.

Brian：文章给出了两组结果。第一组是在 NVIDIA H100 上运行的预调优注意力内核。

Ava: What's the comparison?

Ava：它拿什么来比较？

Brian: PyTorch's scaled dot product attention, or SDPA, using its FLASH backend. On all nineteen pre-tuned shapes, the shipped kernel was faster.

Brian：对比对象是使用 FLASH 后端的 PyTorch 缩放点积注意力，即 SDPA。在全部 19 个预调优形状上，发布的内核都更快。

Ava: All nineteen? What's the overall speed-up, rather than just the count?

Ava：全部 19 个？除了数量，总体加速比是多少？

Brian: The geomean, or geometric mean of the speed-up ratios, was one point two zero. That's across those tested shapes on the H100.

Brian：加速比的几何平均值是 1.20。这是 H100 上这些测试形状的结果。

Ava: What about shapes the tuning runner hadn't seen?

Ava：调优运行程序没见过的形状呢？

Brian: On ten held-out shapes, the kernel beat that SDPA baseline on nine. Their geomean speed-up was one point one seven.

Brian：在 10 个留出形状中，该内核有 9 个超过了作为基准的 SDPA。它们的几何平均加速比是 1.17。

Ava: Good. Held-out shapes tell us more than repeating the tuning examples. What's the second group?

Ava：不错。留出形状比重复测试调优用的例子更能说明问题。第二组结果是什么？

Brian: Seven linear-attention variants shipped with pre-tuned settings for NVIDIA B200s. The comparison there is flash-linear-attention, called FLA in the article.

Brian：发布的 7 种线性注意力变体都配有针对 NVIDIA B200 的预调优设置。这里的对比对象是 flash-linear-attention，文章中简称 FLA。

Ava: Seven variants is a lot to hear in one breath. Do we need every name?

Ava：一口气听完 7 种变体有点多。需要把每个名字都说出来吗？

Brian: The article names linear attention, simple GLA, retention, GLA, delta rule, gated delta rule, and KDA. Let's focus on their shared results.

Brian：文章列出了 linear attention、simple GLA、retention、GLA、delta rule、gated delta rule 和 KDA。我们重点看它们共同的结果。

Ava: I'm listening. How did they do on the shapes used for tuning?

Ava：我在听。它们在用于调优的形状上表现如何？

Brian: Across six pre-tuned shapes, all seven variants were faster than FLA. The geomean was one point four one for device time.

Brian：在 6 个预调优形状上，全部 7 种变体都比 FLA 快。设备运行时间的几何平均加速比是 1.41。

Ava: And device time means the GPU-side measurement here?

Ava：这里的设备端耗时是指 GPU 侧的测量结果吗？

Brian: Yes. The article measured it with CUDA-graph replay. End to end, including central processing unit, or CPU, dispatch, the geomean was one point three three.

Brian：对。文章使用 CUDA 图重放进行了测量。端到端计算时，如果把中央处理器（CPU）调度也算进去，几何平均加速比是 1.33。

Ava: Does the held-out result tell the same story?

Ava：留出形状的结果也是这样吗？

Brian: For six held-out shapes, the geomean was one point three five on device time and one point three one end to end.

Brian：对于 6 个留出形状，设备运行时间的几何平均加速比是 1.35，端到端的几何平均加速比是 1.31。

Ava: There was also a result combining forward and backward work, wasn't there?

Ava：还有一项把前向和反向计算合在一起的结果，对吧？

Brian: Yes. For forward and backward together on the pre-tuned shapes, the reported geomean speed-up was one point five five.

Brian：对。在预调优形状上，前向和反向计算合计的几何平均加速比报告为 1.55。

Ava: Those results sound promising. What should a listener keep in mind before applying them everywhere?

Ava：这些结果听起来很有希望。听众在把它们推广到所有情况之前，该记住什么？

Brian: The attention comparison is on an H100; the linear-attention comparison is on B200s. The article reports particular shapes and baselines.

Brian：注意力内核的比较是在 H100 上做的；线性注意力的比较是在 B200s 上做的。文章报告的是特定形状和特定基准下的结果。

Ava: So the numbers describe those measurements. They don't tell us the speed-up for every GPU or every input.

Ava：所以这些数字描述的是那些测量结果，不能说明在每一种 GPU 或每一种输入上会有多少加速。

Brian: Correct. The author also has to spend time pre-tuning and choose representative workloads. The post doesn't quantify that tuning time.

Brian：没错。作者还得花时间预调优，并选择有代表性的工作负载。这篇文章没有量化调优所花的时间。

Ava: Any practical loading detail that could trip someone up?

Ava：加载时有没有什么实际细节，可能让人踩坑？

Brian: Users need Helion installed and a Kernels version that recognizes it. The article says older releases reject Helion as an unsupported dependency.

Brian：用户需要安装 Helion，还需要使用能够识别它的 Kernels 版本。文章说，较旧的版本会因为 Helion 是不受支持的依赖项而拒绝加载。

Ava: What if someone's device or shape isn't covered by the shipped tree?

Ava：如果某人的设备或输入形状没有被发布的决策树覆盖呢？

Brian: The post doesn't explain the full fallback behavior for that case. I'd check that before promising the same first-call experience everywhere.

Brian：这篇文章没有解释这种情况下完整的回退行为。在承诺所有场景都有同样的首次调用体验之前，我会先查清楚这一点。

Ava: Let's end with three points. First: Helion lets authors express tiled tensor work while its autotuner explores implementation choices.

Ava：最后用 3 点来收尾。第 1 点：Helion 让开发者能够表达分块张量计算，同时由它的自动调优器探索实现方案。

Brian: Second: Kernels packages and distributes the Python source, dependency information, and pre-tuned decision files through the Hub.

Brian：第二，Kernels 通过 Hub 打包和分发 Python 源代码、依赖信息以及预调优的决策文件。

Ava: Third: the reported speed-ups are strong on the tested H100 and B200 workloads, with clear limits on what they establish.

Ava：第三，在测试过的 H100 和 B200 工作负载上，报告的加速效果很强，但这些结果能证明什么也有明确的界限。

Brian: That's the episode. Thanks for listening, and see you next time.

Brian：这期节目就到这里。谢谢收听，下次见。

## 术语

| Term | 释义 |
|---|---|
| kernel | 在设备上执行特定计算的小程序；本文主要讨论 GPU 计算内核。 |
| Helion | 用于编写高性能、可移植机器学习内核的高级领域专用语言。 |
| Kernels | Hugging Face 的内核打包、分发与加载项目；大写用于区别普通内核。 |
| domain-specific language | 领域专用语言，简称 DSL；为特定类型的任务设计的语言。 |
| tile | 把较大的计算范围分成的小块。 |
| autotuner | 自动测试不同配置与实现选择、寻找性能较好方案的工具。 |
| lowering strategies | 将高层内核描述转成具体实现时采用的策略。 |
| kernel-builder | Kernels 项目中供开发者打包和发布内核的工具。 |
| noarch | 本文中指发布源码，由 Helion 在使用者机器上完成编译的内核形式。 |
| pre-tuned configurations | 发布前针对测试工作负载调好的配置。 |
| decision tree | 根据运行时输入形状选择预调优配置的决策结构。 |
| held-out shapes | 调优阶段没有使用、留待后续测试的输入形状。 |
| geomean | 几何平均值；文中用于汇总多个测试形状的加速比。 |
| CPU dispatch | 在 CPU 侧选择和发起计算的调度工作。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Wait, so | 等等，也就是说……；用于确认刚听到的解释。 |
| That split helps. | 这样区分就清楚了。 |
| Here's the catch: | 但这里有个实际问题： |
| Let's slow that down. | 我们放慢一点，把这部分讲清楚。 |
| Put another way, | 换个说法， |
| That makes sense. | 这样就说得通了。 |
| I'm listening. | 我在听，请继续。 |
| What should a listener keep in mind | 听众应该留意什么；用于引出限制或适用范围。 |
