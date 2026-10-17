# One Kernel, More Choices: Helion Meets vLLM

原文：[Building a High-Performance and Portable vLLM Linear Backend with Helion](https://pytorch.org/blog/building-a-high-performance-and-portable-vllm-linear-backend-with-helion/)

## 摘要

这篇文章介绍了将 Helion 集成到 vLLM 线性层后端的实践：通过一个统一的 GEMM 实现，把 Standard GEMM、Split-K 和 Swap-AB 纳入自动调优空间。方案针对 NVIDIA Hopper 上的三种八位量化格式，逐形状调优，并通过混合分发让小规模输入在 CUDA Graph 回放中使用 Helion，较大输入则使用默认内核。在所评估的模型和工作负载中，内核性能提升转化为了稳定的端到端收益，部分工作负载的吞吐量提升超过百分之十。代价包括离线调优、冷启动及配置维护成本；后续方向包括探索由用户执行工作负载专属调优的上游集成模式，以及扩展硬件和 MoE 模型支持。

## 对话

Ava: What if your model server could get more throughput without someone writing a separate kernel for every awkward matrix shape? That's the promise we're exploring today.

Ava：如果不需要有人为每种棘手的矩阵形状单独编写内核，模型服务器就能获得更高的吞吐量，会怎样？这正是我们今天要探讨的可能性。

Brian: And there's a practical result behind it. This article reports more than ten percent higher throughput for some workloads by adding Helion to vLLM's linear backend.

Brian：这背后还有实际效果。这篇文章报告称，将 Helion 加入 vLLM 的线性后端后，部分工作负载的吞吐量提高了超过 10%。

Ava: Some workloads. I'll keep that word close. Before we get to the numbers, what's changing inside the server?

Ava：某些工作负载。这个限定词我会记住。在看数字之前，服务器内部发生了什么变化？

Brian: vLLM is an inference and serving framework for large language models, or LLMs. Its linear backends connect quantized linear layers to optimized kernels from libraries like CUTLASS and DeepGEMM.

Brian：vLLM 是面向大型语言模型，也就是 LLMs 的推理和服务框架。它的线性层后端将量化线性层接入 CUTLASS 和 DeepGEMM 等库提供的优化内核。

Ava: Quantized layers use lower precision representations. So there's already serious optimization here. What's Helion bringing to the table?

Ava：量化层使用低精度表示。所以这里本来就有不少优化。Helion 又带来了什么？

Brian: Helion is a PyTorch-native domain-specific language, or DSL, for kernels. Developers write Python-like code using tiles, which are blocks of data. Helion generates specialized code and searches for good configurations.

Brian：Helion 是一种用于编写内核的 PyTorch 原生领域专用语言，也就是 DSL。开发者使用 tile，也就是数据块，编写类似 Python 的代码。Helion 会生成针对性的代码，并搜索合适的配置。

Ava: So I describe the computation, and Helion explores how to run it efficiently. Does that search happen while users are waiting for answers?

Ava：也就是说，我描述计算过程，Helion 探索怎样高效运行它。这个搜索会在用户等待答案时进行吗？

Brian: The tuning is ahead of time, or AOT, before deployment. It explores choices ranging from memory layout and scheduling to the algorithm itself. Fine-grained tuning can still take hours.

Brian：调优是在部署前提前进行的，也就是 AOT。它探索的选择从内存布局、调度一直到算法本身。细粒度调优仍可能耗费数小时。

Ava: Hours upfront, with the hope of faster serving afterward. Why does the algorithm need to change with the input shape?

Ava：先花数小时，希望之后的服务运行得更快。为什么算法需要随输入形状变化？

Brian: Let's take general matrix multiplication, or GEMM. Matrix A has dimensions M by K. Matrix B has dimensions K by N. The output is M by N.

Brian：以通用矩阵乘法，也就是 GEMM 为例。矩阵 A 的维度是 M by K，矩阵 B 的维度是 K by N，输出的维度是 M by N。

Ava: And K is the dimension we're reducing over. What's the problem when M or N is small?

Ava：K 是进行归约的维度。当 M 或 N 很小时，会有什么问题？

Brian: There may not be enough parallel work. Think of a large kitchen with only a few dishes to prepare. Many cooks could be standing around.

Brian：可能没有足够的工作可以并行处理。想象一间大厨房，却只有几道菜要做，很多厨师可能只能闲站着。

Ava: An expensive kitchen, too. How does Split-K help?

Ava：而且这间厨房还很贵。Split-K 怎么帮忙？

Brian: Split-K divides the K dimension across multiple thread blocks, which are groups of GPU threads. More blocks can work in parallel when M or N is too small.

Brian：Split-K 把 K 维度分给多个线程块，线程块是一组 GPU 线程。当 M 或 N 太小时，就能让更多线程块并行工作。

Ava: So several groups contribute to the same output. How does this example combine their work?

Ava：这样就有几组线程为同一个输出贡献结果。这个例子怎样合并它们的工作？

Brian: It clears the output first, then uses atomic additions to combine partial results safely. With no split, it writes each output tile directly.

Brian：它先清空输出，再用原子加法安全地合并部分结果。不做拆分时，则直接写入每个输出 tile。

Ava: Got it. Split the reduction work, then combine it. What's the other option, Swap-AB?

Ava：明白了。先拆分归约工作，再合并结果。另一个选项 Swap-AB 是什么？

Brian: It reverses the multiplication order using transposes, which exchange rows and columns. Multiply transposed B by transposed A, then transpose the result back. That can improve tiling when M is small.

Brian：它利用转置来调换乘法顺序，转置就是交换行和列。先用转置后的 B 乘以转置后的 A，再把结果转置回来。当 M 很小时，这可以改善分块方式。

Ava: Same mathematical result, different arrangement of the work. Like turning a tray so it fits the oven better?

Ava：数学结果相同，只是工作安排不同。就像把烤盘转个方向，让它更适合放进烤箱？

Brian: Exactly. Traditionally, authors implement variants separately and write dispatch rules. The article gives one example: vLLM's current Block_FP8 backend chooses Swap-AB below an M of thirty-two on Hopper.

Brian：没错。传统上，作者会分别实现不同变体，再编写分派规则。文章举了一个例子：在 Hopper 上，当 M 小于 32 时，vLLM 当前的 Block_FP8 后端会选择 Swap-AB。

Ava: That's a useful rule, but it doesn't necessarily pick the winner for every shape.

Ava：这是一条有用的规则，但它未必能为每种形状选出最快的方案。

Brian: Right. Helion puts Standard GEMM, Split-K, and Swap-AB into one implementation. The split count and whether to swap become tunable choices, alongside lower-level configuration choices.

Brian：对。Helion 把 Standard GEMM、Split-K 和 Swap-AB 放进同一份实现。拆分次数以及是否交换，都和底层配置选项一起成为可调优的选择。

Ava: How wide is that search in the example?

Ava：这个例子的搜索范围有多大？

Brian: The split count takes powers of two from one through two hundred fifty-six. Swapping is either on or off. The autotuner benchmarks combinations and selects a configuration for each workload.

Brian：拆分数量取 1 到 256 之间的 2 的幂；交换则只有开启和关闭两种选择。自动调优器对各种组合进行基准测试，并为每种工作负载选出配置。

Ava: So the key idea is one implementation with several algorithmic choices. We're moving the selection work into tuning. What hardware and formats did they actually test?

Ava：所以核心思路是在一份实现里提供几种算法选择，把选择工作交给调优。他们实际测试了哪些硬件和格式？

Brian: The focus is NVIDIA Hopper, using Helion's Triton backend. They target three formats. FP8_Dynamic uses eight-bit floating point, with per-token activation scaling and per-channel weight scaling.

Brian：研究重点是 NVIDIA Hopper，使用 Helion 的 Triton 后端。他们针对 3 种格式。FP8_Dynamic 使用 8 位浮点数，激活值按每个 token 缩放，权重按每个通道缩放。

Ava: Scaling determines how those low precision values are interpreted. What's the integer version?

Ava：缩放决定了如何解释这些低精度数值。整数版本是什么？

Brian: W8A8_INT8 uses eight-bit integer weights and activations, with the same scaling granularity. Block_FP8 uses floating point with activation scaling over one by one hundred twenty-eight blocks and weight scaling over one hundred twenty-eight by one hundred twenty-eight blocks.

Brian：W8A8_INT8 使用 8 位整数权重和激活值，缩放粒度相同。Block_FP8 使用浮点数，激活值按 1×128 的块缩放，权重按 128×128 的块缩放。

Ava: Now, a faster kernel doesn't automatically mean a faster server. Where can the benefit disappear?

Ava：内核更快，不代表服务器就一定更快。收益可能在哪儿被抵消？

Brian: CPU overhead from dispatching and launching Helion kernels can offset the gains. That's why this integration uses CUDA Graphs, which let it replay captured GPU work.

Brian：分派和启动 Helion 内核带来的 CPU 开销可能抵消性能收益。所以这项集成使用 CUDA Graphs，重放已经捕获的 GPU 工作。

Ava: Wait, so they don't send every linear operation through Helion?

Ava：等等，所以他们没有让每个线性运算都走 Helion？

Brian: Correct. They use hybrid dispatch. Small shapes run Helion under CUDA Graph replay. Larger shapes fall back to the default kernels. The threshold in this work is thirty-two tokens.

Brian：对。他们采用混合分派：小尺寸的计算在 CUDA Graph 重放下运行 Helion，较大的尺寸则回退到默认内核。这项工作采用的阈值是 32 个 token。

Ava: And the small token counts matter because they dominate decoding workloads. Which counts did they tune?

Ava：小 token 数量很重要，因为它们在解码工作负载中占主导。他们调优了哪些数量？

Brian: One, two, four, eight, sixteen, twenty-four, and thirty-two. Each count is tuned for the model's corresponding GEMM shapes. They also benchmark under CUDA Graphs to match inference execution more closely.

Brian：分别是 1、2、4、8、16、24 和 32 个。每个数量都会针对该模型对应的 GEMM 形状进行调优。他们还在 CUDA Graph 下做基准测试，以更贴近实际推理执行过程。

Ava: That narrows the job. Fewer shapes to tune, and fewer configurations to keep around.

Ava：这样任务范围就缩小了。需要调优的形状更少，需要保留的配置也更少。

Brian: Yes. It also avoids Helion's extra runtime dispatch and launch overhead through graph replay. But startup has another cost: graph capture triggers just-in-time compilation, or JIT compilation.

Brian：是的。通过图重放，还能避开 Helion 额外的运行时分派和启动开销。但启动时还有另一项成本：图捕获会触发即时编译，也就是 JIT 编译。

Ava: So a cold start can still be slower?

Ava：所以冷启动仍可能更慢？

Brian: Yes. Caching compiled artifacts can largely remove that overhead on warm starts. The runtime strategy doesn't make every cost disappear.

Brian：是的。缓存编译产物可以在热启动时基本消除这项开销。运行时策略并不能让所有成本都消失。

Ava: The article also mentions an LLM helping with tuning. Is it writing new kernels?

Ava：文章还提到用 LLM 协助调优。它是在编写新内核吗？

Brian: Here, it identifies promising configuration candidates to seed the search. The article uses Claude Opus four point eight. Better starting candidates help find better configurations and may reduce tuning time.

Brian：这里，它会找出有潜力的配置候选项，作为搜索的起点。文章使用的是 Claude Opus 4.8。更好的初始候选项有助于找到更好的配置，也可能缩短调优时间。

Ava: May reduce it. We don't get a measured tuning-time reduction here. Let's get to the performance numbers they do report.

Ava：只是可能缩短。这里没有给出调优时间缩短了多少的测量结果。我们来看看他们实际报告的性能数字。

Brian: All benchmarks used an NVIDIA H one hundred GPU with eighty gigabytes of memory. They evaluated dense Qwen models ranging from one point seven billion to thirty-two billion parameters, plus Qwen three point eight, twenty-seven billion.

Brian：所有基准测试都使用一块配备 80 GB 显存的 NVIDIA H100 GPU。他们评估了参数量从 17 亿到 320 亿的稠密 Qwen 模型，以及参数量为 270 亿的 Qwen 3.8。

Ava: First, the kernels on their own. How much faster were they?

Ava：先看内核本身。它们快了多少？

Brian: The geometric mean speedups were one point one one zero times for FP8_Dynamic and one point one seven eight times for W8A8_INT8, both against CUTLASS. That's a summary across shapes, not a guarantee for each shape.

Brian：与 CUTLASS 相比，FP8_Dynamic 的几何平均加速比为 1.110 倍，W8A8_INT8 为 1.178 倍。这是对不同形状的汇总结果，并不保证每种形状都能达到这一表现。

Ava: And Block_FP8 had two comparisons, right?

Ava：Block_FP8 有两组对比，对吧？

Brian: Right. One point one four nine times against FlashInfer, and one point one seven seven times against DeepGEMM. Performance varies by shape, which supports the case for individual tuning.

Brian：对。与 FlashInfer 相比是 1.149 倍，与 DeepGEMM 相比是 1.177 倍。性能会随形状变化，这也说明了逐一调优的必要性。

Ava: What happened when they tested the whole serving system?

Ava：他们测试整个服务系统时，结果怎样？

Brian: They used ShareGPT workloads, focused on batch sizes up to thirty-two, and disabled prefix caching to avoid its effect on results. They report consistent gains across the evaluated models and formats.

Brian：他们使用 ShareGPT 工作负载，重点测试不超过 32 的批量大小，并关闭前缀缓存，以免它影响结果。他们报告称，在评估的各个模型和格式上都获得了稳定的提升。

Ava: Including more than ten percent higher throughput for some workloads. We shouldn't stretch that into a promise about every deployment.

Ava：其中一些工作负载的吞吐量提高了超过 10%。但我们不该把这一结果说成适用于每次部署的保证。

Brian: Exactly. The evaluation is focused. And adoption has a three-way tradeoff: performance, usability, and maintainability. Detailed tuning improves performance, but someone must pay the tuning or configuration maintenance cost.

Brian：正是如此。评估范围是明确限定的。采用这项技术还需要在性能、易用性和可维护性三者之间权衡。细致调优能改善性能，但总得有人承担调优或维护配置的成本。

Ava: Who's paying it today, and what might change?

Ava：现在是谁在承担这些成本，未来可能有什么变化？

Brian: Their vLLM fork ships pre-tuned configurations. For upstream adoption, they're exploring shared kernels with a default configuration for testing, while users tune for their own deployments. That makes initial setup less convenient.

Brian：他们的 vLLM 分支自带预先调优的配置。为了让上游项目采用，他们正在探索提供共享内核，并附上供测试使用的默认配置，同时让用户针对自己的部署进行调优。这会让初始设置没那么方便。

Ava: Does every kernel need that much individual attention?

Ava：每个内核都需要这么多单独的调优吗？

Brian: Apparently not. Initial experiments with smaller auxiliary kernels show meaningful gains with as few as six configurations. Future work includes more hardware and a mixture-of-experts, or MoE, backend, where optimization targets differ.

Brian：看来不用。对较小的辅助内核进行的初步实验表明，只用六种配置就能获得显著提升。后续工作还包括支持更多硬件，以及开发 mixture-of-experts（MoE）后端；在这些场景中，优化目标有所不同。

Ava: Let's finish with three points. First, one Helion implementation exposes multiple GEMM strategies. Second, per-shape tuning and graph-based hybrid dispatch turn kernel gains into serving gains.

Ava：最后总结三点。第一，一种 Helion 实现可以提供多种 GEMM 策略。第二，针对不同形状进行调优，再结合基于计算图的混合派发，可以把内核层面的提升转化为服务性能的提升。

Brian: Third, those gains come with offline tuning, startup, and maintenance costs. The practical balance depends on the deployment.

Brian：第三，这些提升也伴随着离线调优、启动和维护方面的成本。实际如何权衡，取决于具体部署。

Ava: Thanks for listening. We'll see you next time.

Ava：感谢收听。我们下次再见。

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
