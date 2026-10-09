# One Kernel, More Choices: Helion Meets vLLM

原文：[Building a High-Performance and Portable vLLM Linear Backend with Helion](https://pytorch.org/blog/building-a-high-performance-and-portable-vllm-linear-backend-with-helion/)

## 摘要

这篇文章介绍了将 Helion 集成到 vLLM 线性层后端的实践：通过一个统一的 GEMM 实现，把 Standard GEMM、Split-K 和 Swap-AB 纳入自动调优空间。方案针对 NVIDIA Hopper 上的三种八位量化格式，逐形状调优，并通过混合分发让小规模输入在 CUDA Graph 回放中使用 Helion，较大输入则使用默认内核。在所评估的模型和工作负载中，内核性能提升转化为了稳定的端到端收益，部分工作负载的吞吐量提升超过百分之十。代价包括离线调优、冷启动及配置维护成本；后续方向包括探索由用户执行工作负载专属调优的上游集成模式，以及扩展硬件和 MoE 模型支持。

## 对话

Ava: What if your model server could get more throughput without someone writing a separate kernel for every awkward matrix shape?

Ava：如果不用为每种棘手的矩阵形状单独写内核，模型服务器也能提高吞吐量呢？

That's the promise we're exploring today.

这就是我们今天要探讨的可能性。

Brian: And there's a practical result behind it.

Brian：而且已有实际成果。

This article reports more than ten percent higher throughput for some workloads by adding Helion to vLLM's linear backend.

这篇文章称，在一些工作负载中，将 Helion 加入 vLLM 的线性后端后，吞吐量提高了 10% 以上。

Ava: Some workloads. I'll keep that word close.

Ava：是“一些”工作负载，这个限定我记住了。

Before we get to the numbers, what's changing inside the server?

谈数字之前，先说说服务器内部有什么变化？

Brian: vLLM is an inference and serving framework for large language models, or LLMs.

Brian：vLLM 是面向大语言模型（LLM）的推理和服务框架。

Its linear backends connect quantized linear layers to optimized kernels from libraries like CUTLASS and DeepGEMM.

它的线性后端将量化线性层接入 CUTLASS、DeepGEMM 等库的优化内核。

Ava: Quantized layers use lower precision representations.

Ava：量化层使用较低精度的表示。

So there's already serious optimization here. What's Helion bringing to the table?

看来这里已经做了不少优化。Helion 又带来了什么？

Brian: Helion is a PyTorch-native domain-specific language, or DSL, for kernels.

Brian：Helion 是一种基于 PyTorch、用于编写内核的领域专用语言（DSL）。

Developers write Python-like code using tiles, which are blocks of data.

开发者用类似 Python 的代码编程，处理的数据块称为 tile。

Helion generates specialized code and searches for good configurations.

Helion 会生成专门优化的代码，并搜索合适的配置。

Ava: So I describe the computation, and Helion explores how to run it efficiently.

Ava：也就是说，我描述计算过程，Helion 探索怎样高效运行。

Does that search happen while users are waiting for answers?

这种搜索会在用户等答案时进行吗？

Brian: The tuning is ahead of time, or AOT, before deployment.

Brian：调优在部署前提前完成，也就是 AOT。

It explores choices ranging from memory layout and scheduling to the algorithm itself.

它会探索内存布局、调度方式，甚至算法本身。

Fine-grained tuning can still take hours.

细粒度调优仍可能耗时数小时。

Ava: Hours upfront, with the hope of faster serving afterward.

Ava：先花几个小时，期望之后提供服务时更快。

Why does the algorithm need to change with the input shape?

为什么算法要随输入形状变化？

Brian: Let's take general matrix multiplication, or GEMM. Matrix A has dimensions M by K.

Brian：以通用矩阵乘法（GEMM）为例。矩阵 A 的维度是 M×K。

Matrix B has dimensions K by N. The output is M by N.

矩阵 B 的维度是 K×N，输出维度是 M×N。

Ava: And K is the dimension we're reducing over. What's the problem when M or N is small?

Ava：K 是进行归约的维度。M 或 N 较小时会有什么问题？

Brian: There may not be enough parallel work.

Brian：可并行的工作可能不够多。

Think of a large kitchen with only a few dishes to prepare.

就像大厨房里只有几道菜要做。

Many cooks could be standing around.

很多厨师可能闲着。

Ava: An expensive kitchen, too. How does Split-K help?

Ava：厨房还很贵。Split-K 怎么帮忙？

Brian: Split-K divides the K dimension across multiple thread blocks, which are groups of GPU threads.

Brian：Split-K 将 K 维分给多个线程块，也就是多组 GPU 线程。

More blocks can work in parallel when M or N is too small.

当 M 或 N 太小时，更多线程块就能并行工作。

Ava: So several groups contribute to the same output.

Ava：这样，多组线程会共同计算同一个输出。

How does this example combine their work?

这个例子怎样合并它们的结果？

Brian: It clears the output first, then uses atomic additions to combine partial results safely.

Brian：它先清零输出，再用原子加法安全地合并部分结果。

With no split, it writes each output tile directly.

不拆分时，每个输出数据块会直接写入。

Ava: Got it. Split the reduction work, then combine it. What's the other option, Swap-AB?

Ava：明白了，拆分归约任务，再合并结果。另一个选项 Swap-AB 呢？

Brian: It reverses the multiplication order using transposes, which exchange rows and columns.

Brian：它通过转置来颠倒乘法顺序；转置就是交换行和列。

Multiply transposed B by transposed A, then transpose the result back.

先用转置后的 B 乘以转置后的 A，再把结果转置回来。

That can improve tiling when M is small.

M 较小时，这样能改善分块效果。

Ava: Same mathematical result, different arrangement of the work.

Ava：数学结果一样，只是安排计算的方式不同。

Like turning a tray so it fits the oven better?

就像转一下烤盘，让它更适合放进烤箱？

Brian: Exactly. Traditionally, authors implement variants separately and write dispatch rules.

Brian：没错。传统做法是分别实现各个变体，再编写选择规则。

The article gives one example: vLLM's current Block_FP8 backend chooses Swap-AB below an M of thirty-two on Hopper.

文章举了个例子：vLLM 当前的 Block_FP8 后端在 Hopper 上遇到 M 小于 32 时，会选择 Swap-AB。

Ava: That's a useful rule, but it doesn't necessarily pick the winner for every shape.

Ava：这条规则有用，但未必对每种形状都能选出最快的方案。

Brian: Right. Helion puts Standard GEMM, Split-K, and Swap-AB into one implementation.

Brian：对。Helion 把标准 GEMM、Split-K 和 Swap-AB 放进同一个实现中。

The split count and whether to swap become tunable choices, alongside lower-level configuration choices.

拆分次数和是否交换都成了可调选项，底层配置也一样。

Ava: How wide is that search in the example?

Ava：这个例子的搜索范围有多大？

Brian: The split count takes powers of two from one through two hundred fifty-six.

Brian：拆分次数取 1 到 256 之间的 2 的幂。

Swapping is either on or off.

交换则只有开启和关闭两种选择。

The autotuner benchmarks combinations and selects a configuration for each workload.

自动调优器会测试各种组合，为每种工作负载选出配置。

Ava: So the key idea is one implementation with several algorithmic choices.

Ava：所以核心是用一个实现涵盖多种算法选择。

We're moving the selection work into tuning.

我们把选择方案的工作交给了调优过程。

What hardware and formats did they actually test?

他们实际测试了哪些硬件和格式？

Brian: The focus is NVIDIA Hopper, using Helion's Triton backend. They target three formats.

Brian：重点是 NVIDIA Hopper，使用 Helion 的 Triton 后端，测试三种格式。

FP8_Dynamic uses eight-bit floating point, with per-token activation scaling and per-channel weight scaling.

FP8_Dynamic 使用 8 位浮点数，激活值按 token 缩放，权重按通道缩放。

Ava: Scaling determines how those low precision values are interpreted.

Ava：缩放决定了如何解读这些低精度数值。

What's the integer version?

整数版本是什么？

Brian: W8A8_INT8 uses eight-bit integer weights and activations, with the same scaling granularity.

Brian：W8A8_INT8 的权重和激活值都使用 8 位整数，缩放粒度相同。

Block_FP8 uses floating point with activation scaling over one by one hundred twenty-eight blocks and weight scaling over one hundred twenty-eight by one hundred twenty-eight blocks.

Block_FP8 使用浮点数，激活值按 1×128 的块缩放，权重按 128×128 的块缩放。

Ava: Now, a faster kernel doesn't automatically mean a faster server.

Ava：内核更快，不代表服务器一定更快。

Where can the benefit disappear?

收益会在哪儿被抵消？

Brian: CPU overhead from dispatching and launching Helion kernels can offset the gains.

Brian：调度和启动 Helion 内核的 CPU 开销可能抵消性能收益。

That's why this integration uses CUDA Graphs, which let it replay captured GPU work.

因此，这项集成使用 CUDA Graphs，重放已捕获的 GPU 工作。

Ava: Wait, so they don't send every linear operation through Helion?

Ava：等等，所以他们没有让每个线性运算都走 Helion？

Brian: Correct. They use hybrid dispatch. Small shapes run Helion under CUDA Graph replay.

Brian：对。他们采用混合调度：小规模运算在 CUDA Graph 重放时使用 Helion。

Larger shapes fall back to the default kernels.

较大规模的运算则回退到默认内核。

The threshold in this work is thirty-two tokens.

这项工作设定的阈值是 32 个 token。

Ava: And the small token counts matter because they dominate decoding workloads.

Ava：少量 token 很重要，因为解码工作负载主要由这类情况构成。

Which counts did they tune?

他们调优了哪些 token 数量？

Brian: One, two, four, eight, sixteen, twenty-four, and thirty-two.

Brian：1、2、4、8、16、24 和 32。

Each count is tuned for the model's corresponding GEMM shapes.

每种数量都针对模型中相应的 GEMM 形状调优。

They also benchmark under CUDA Graphs to match inference execution more closely.

他们还在 CUDA Graphs 下做基准测试，以更贴近实际推理过程。

Ava: That narrows the job. Fewer shapes to tune, and fewer configurations to keep around.

Ava：这样范围就缩小了：要调优的形状更少，要保留的配置也更少。

Brian: Yes.

Brian：没错。

It also avoids Helion's extra runtime dispatch and launch overhead through graph replay.

图重放还能避开 Helion 额外的运行时调度和启动开销。

But startup has another cost: graph capture triggers just-in-time compilation, or JIT compilation.

但启动还有另一项开销：图捕获会触发即时编译，也就是 JIT 编译。

Ava: So a cold start can still be slower?

Ava：所以冷启动仍可能比较慢？

Brian: Yes. Caching compiled artifacts can largely remove that overhead on warm starts.

Brian：对。缓存编译产物后，热启动时基本可以消除这项开销。

The runtime strategy doesn't make every cost disappear.

这种运行时策略并不能消除所有开销。

Ava: The article also mentions an LLM helping with tuning. Is it writing new kernels?

Ava：文章还提到用大语言模型辅助调优。它是在编写新内核吗？

Brian: Here, it identifies promising configuration candidates to seed the search.

Brian：这里，它负责找出有潜力的配置，作为搜索的起点。

The article uses Claude Opus four point eight.

文章使用的是 Claude Opus 4.8。

Better starting candidates help find better configurations and may reduce tuning time.

更好的初始候选配置有助于找到更优配置，也可能缩短调优时间。

Ava: May reduce it. We don't get a measured tuning-time reduction here.

Ava：只是可能缩短；这里没有给出调优时间减少了多少的实测数据。

Let's get to the performance numbers they do report.

来看看他们实际报告的性能数据。

Brian: All benchmarks used an NVIDIA H one hundred GPU with eighty gigabytes of memory.

Brian：所有基准测试都使用一块配备 80 GB 显存的 NVIDIA H100 GPU。

They evaluated dense Qwen models ranging from one point seven billion to thirty-two billion parameters, plus Qwen three point eight, twenty-seven billion.

他们评估了参数量从 17 亿到 320 亿的稠密 Qwen 模型，以及 270 亿参数的 Qwen 3.8。

Ava: First, the kernels on their own. How much faster were they?

Ava：先看内核本身。它们快了多少？

Brian: The geometric mean speedups were one point one one zero times for FP8_Dynamic and one point one seven eight times for W8A8_INT8, both against CUTLASS.

Brian：相较于 CUTLASS，FP8_Dynamic 的几何平均加速比为 1.110 倍，W8A8_INT8 为 1.178 倍。

That's a summary across shapes, not a guarantee for each shape.

这是汇总多个形状的结果，并不意味着每个形状都能达到这个加速比。

Ava: And Block_FP8 had two comparisons, right?

Ava：Block_FP8 做了两组对比，对吧？

Brian: Right.

Brian：对。

One point one four nine times against FlashInfer, and one point one seven seven times against DeepGEMM.

相较于 FlashInfer，速度是 1.149 倍；相较于 DeepGEMM，是 1.177 倍。

Performance varies by shape, which supports the case for individual tuning.

不同形状的性能各异，因此值得分别调优。

Ava: What happened when they tested the whole serving system?

Ava：测试整个服务系统时，结果怎样？

Brian: They used ShareGPT workloads, focused on batch sizes up to thirty-two, and disabled prefix caching to avoid its effect on results.

Brian：他们使用 ShareGPT 工作负载，重点测试不超过 32 的批量大小，并关闭前缀缓存，以免影响结果。

They report consistent gains across the evaluated models and formats.

他们报告称，在评估的模型和格式中，性能均有提升。

Ava: Including more than ten percent higher throughput for some workloads.

Ava：其中一些工作负载的吞吐量提升超过 10%。

We shouldn't stretch that into a promise about every deployment.

但不能据此保证每种部署都有这样的提升。

Brian: Exactly. The evaluation is focused.

Brian：正是如此。评估范围是有限的。

And adoption has a three-way tradeoff: performance, usability, and maintainability.

采用这套方案还要权衡三方面：性能、易用性和可维护性。

Detailed tuning improves performance, but someone must pay the tuning or configuration maintenance cost.

细致调优能提升性能，但调优和维护配置都需要付出成本。

Ava: Who's paying it today, and what might change?

Ava：现在由谁承担这些成本？未来可能有什么变化？

Brian: Their vLLM fork ships pre-tuned configurations.

Brian：他们的 vLLM 分支自带预调优配置。

For upstream adoption, they're exploring shared kernels with a default configuration for testing, while users tune for their own deployments.

为了推动上游采用，他们正考虑共享内核，并提供默认配置用于测试；用户则根据自己的部署环境调优。

That makes initial setup less convenient.

这让初始配置没那么方便。

Ava: Does every kernel need that much individual attention?

Ava：每个内核都需要这么仔细地单独调优吗？

Brian: Apparently not.

Brian：看来不用。

Initial experiments with smaller auxiliary kernels show meaningful gains with as few as six configurations.

对较小辅助内核的初步实验显示，只用六种配置就能取得明显收益。

Future work includes more hardware and a mixture-of-experts, or MoE, backend, where optimization targets differ.

后续工作还包括支持更多硬件，以及优化目标不同的混合专家（MoE）后端。

Ava: Let's finish with three points.

Ava：最后总结三点。

First, one Helion implementation exposes multiple GEMM strategies.

第一，一套 Helion 实现支持多种 GEMM 策略。

Second, per-shape tuning and graph-based hybrid dispatch turn kernel gains into serving gains.

第二，按形状调优和基于计算图的混合分派，把内核性能提升转化为推理服务性能提升。

Brian: Third, those gains come with offline tuning, startup, and maintenance costs.

Brian：第三，这些收益也伴随着离线调优、启动和维护成本。

The practical balance depends on the deployment.

具体怎么权衡，要看部署场景。

Ava: Thanks for listening. We'll see you next time.

Ava：感谢收听，我们下次见。

## 术语

| Term | 释义 |
|---|---|
| linear backend | 线性层后端，负责把线性层计算接入相应的优化内核实现。 |
| DSL | 领域专用语言；文中 Helion 用于编写高性能计算内核。 |
| tile-programming model | 分块编程模型，以数据块为单位表达计算。 |
| GEMM | 通用矩阵乘法，英文为 general matrix multiplication。 |
| Split-K | 将矩阵乘法的 K 维划分给多个线程块，以增加并行度的算法变体。 |
| Swap-AB | 利用转置交换矩阵乘法操作数的顺序，以改善某些形状的分块和硬件利用率。 |
| AOT autotuner | 提前运行的自动调优器，在部署前搜索并选择性能较好的配置。 |
| per-shape tuning | 逐形状调优，为不同输入形状分别寻找合适配置。 |
| hybrid dispatch | 混合分发，根据输入规模及 CUDA Graph 覆盖情况选择 Helion 或默认内核。 |
| CUDA Graph replay | CUDA Graph 回放，执行预先捕获的 GPU 工作，规避文中所述的额外分发与启动开销。 |
| JIT compilation | 即时编译；文中在启动阶段的 CUDA Graph 捕获期间触发。 |
| FP8_Dynamic | 采用逐 token 激活缩放和逐通道权重缩放的八位浮点量化格式。 |
| W8A8_INT8 | 权重和激活均为八位整数，采用逐 token 激活缩放和逐通道权重缩放。 |
| Block_FP8 | 分块八位浮点量化，激活缩放块为 1×128，权重缩放块为 128×128。 |
| geometric mean speedup | 几何平均加速比，用于汇总多个输入形状的相对性能，并不代表每个形状都有相同收益。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What's Helion bringing to the table? | Helion 能带来什么价值？可替换主语来询问某方案的贡献。 |
| I'll keep that word close. | 我会记住这个限定词，提醒自己别把结论扩大。 |
| How does Split-K help? | Split-K 是怎么起作用的？适合追问某项技术的具体作用。 |
| Got it. | 明白了。 |
| Wait, so they don't send every linear operation through Helion? | 等等，也就是说，他们并没有让每个线性运算都走 Helion？用于核对刚形成的理解。 |
| That narrows the job. | 这样就缩小了工作范围。 |
| Let's get to the performance numbers they do report. | 我们来看看文章实际报告的性能数据。 |
| We shouldn't stretch that into a promise about every deployment. | 我们不应该把这个结果扩大成对所有部署的保证。 |
| Who's paying it today, and what might change? | 目前是谁在承担这个成本，今后可能有什么变化？ |
| The practical balance depends on the deployment. | 实际该如何权衡，取决于具体部署。 |
