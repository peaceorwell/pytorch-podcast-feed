# One Search, Two Layers: Tuning Helion Below the Kernel

原文：[One Search, Two Layers: Compiler Controls Inside PyTorch Helion using NVIDIA CompileIQ - PyTorch](https://pytorch.org/blog/one-search-two-layers-compiler-controls-inside-pytorch-helion-using-nvidia-compileiq/)

## 摘要

Helion 原本会搜索 GPU kernel 的配置；这篇文章介绍了另一层可搜索的空间：ptxas 编译器控制项。NVIDIA CompileIQ 可以生成高级控制文件（ACF），Helion 则能把这些文件与 kernel 配置一起测试，并保留默认编译设置作为候选。在一项针对已预调优 NVFP4 kernel 的实验中，不同优化目标带来了约百分之二到百分之七的运行时间收益。文章也强调，ACF 必须匹配硬件和工具链，搜索应在线下进行，实际收益要在自己的工作负载上验证。

## 对话

Ava: If a GPU kernel is already tuned, where can more speed come from? That sounds like trying to squeeze water from a dry sponge.

Ava：如果一个 GPU 内核已经调优过了，还能从哪里获得更快的速度？这听起来像是要从一块干海绵里挤出水来。

Brian: Surprisingly, from the compiler. This article shows how Helion searches settings below the kernel itself. On an already tuned kernel, that produced measurable gains.

Brian：出人意料的是，还能从编译器入手。这篇文章展示了 Helion 如何搜索内核本身以下层级的设置。即使内核已经调优过，这样做仍带来了可测量的性能提升。

Ava: Okay, let's start with Helion. What's already been tuned before the compiler gets its turn?

Ava：好，那我们从 Helion 说起。轮到编译器之前，哪些东西已经调优过了？

Brian: Helion lets developers write GPU kernels in PyTorch-native Python. A kernel is a program for work on the graphics processing unit, or GPU. Helion's autotuner tries many ways to run that work.

Brian：Helion 让开发者用 PyTorch 原生的 Python 编写 GPU 内核。内核是在图形处理单元，也就是 GPU 上执行工作的程序。Helion 的自动调优器会尝试多种执行方式。

Ava: And these weren't weak examples picked to make a speedup look impressive?

Ava：这些例子不是特意挑选的、容易跑出亮眼提速的简单任务吧？

Brian: Right. The article starts from Helion's pretuned kernels. There are roughly twenty-seven, covering work such as quantization, normalization, grouped matrix multiplication, and model decode. Their kernel configurations were tuned offline.

Brian：对。文章从 Helion 预先调优的内核入手。这类内核大约有 27 个，涵盖量化、归一化、分组矩阵乘法和模型解码等工作。它们的内核配置已经离线调优过。

Ava: So someone running one of those kernels doesn't wait for a search?

Ava：所以，运行其中一个内核的人不需要等待搜索完成？

Brian: Exactly. Each ships with an ahead-of-time configuration for its target architecture. Helion selects that configuration at runtime. The user gets the result of earlier tuning without paying the search cost then.

Brian：没错。每个内核都附带针对目标架构预先确定的配置。Helion 在运行时选用该配置。用户可以直接享受先前调优的成果，无须在当时承担搜索开销。

Ava: Good baseline. Now, what does Helion's usual autotuner actually change?

Ava：这个基准很扎实。那么，Helion 通常的自动调优器具体会改变什么？

Brian: Think of a large job split into tiles, like dividing a floor into sections to clean. Helion decides each tile's size and how the GPU's thread groups work on it.

Brian：可以把一项大工作分成多个分块，就像把地板划成几个区域来清扫。Helion 会决定每个分块的大小，以及 GPU 的线程组如何处理它。

Ava: So the developer marks the work as tiled, and Helion picks the details?

Ava：也就是说，开发者指定这项工作要分块，具体细节由 Helion 来选？

Brian: Yes. It can search block sizes, warp counts, pipeline stages, memory indexing, and more. A warp is a group of GPU threads. Helion compiles candidates, times them, and keeps the winner.

Brian：是的。它可以搜索块大小、warp 数量、流水线阶段数、内存索引方式等设置。warp 是一组 GPU 线程。Helion 会编译候选配置、测量运行时间，然后保留表现最好的那个。

Ava: Where does the new layer begin?

Ava：新加的这一层从哪里开始？

Brian: After Helion lowers the kernel through TorchInductor to Triton, the winning candidate reaches ptxas, NVIDIA's PTX assembler. It turns that lower-level program into machine instructions. Traditionally, its controls stayed at their defaults.

Brian：Helion 通过 TorchInductor 把内核降级到 Triton 后，胜出的候选配置会交给 NVIDIA 的 PTX 汇编器 ptxas。它把这个更底层的程序转换成机器指令。以往，它的控制选项一直使用默认值。

Ava: What kind of controls are we talking about? More tile sizes?

Ava：这里说的是什么控制选项？更多的分块大小吗？

Brian: No. These affect choices such as register allocation, instruction scheduling, and code-generation heuristics. Registers are fast storage on the GPU. The kernel's tile can stay the same while the compiler handles its instructions differently.

Brian：不是。这些选项影响寄存器分配、指令调度和代码生成启发式规则等决策。寄存器是 GPU 上的高速存储。内核的分块可以保持不变，而编译器以不同方式处理其中的指令。

Ava: That sounds hard to set by hand.

Ava：听起来很难手动设置。

Brian: It is. One register decision can change occupancy, meaning how much work is active on the GPU. That can change whether another scheduling decision helps. The best settings depend on the workload.

Brian：确实如此。一个寄存器分配决策就可能改变占用率，也就是 GPU 上有多少工作处于活跃状态。这又可能影响另一项调度决策是否有帮助。最佳设置取决于具体工作负载。

Ava: So how does Helion receive a chosen set of compiler controls?

Ava：那么，怎样把选定的一组编译器控制选项交给 Helion？

Brian: Through an Advanced Control File, or ACF. It's a file applied at compile time. Helion offers one setting to pin a chosen ACF and another to give its autotuner several ACFs to test.

Brian：通过 Advanced Control File，简称 ACF。它是在编译时应用的文件。Helion 提供一个设置来固定使用选定的 ACF，另一个设置则允许自动调优器测试多个 ACF。

Ava: If I give it a list, could every file lose to the compiler's normal settings?

Ava：如果我给它一个文件列表，所有文件会不会都输给编译器的常规设置？

Brian: Absolutely. Helion adds a no-ACF candidate automatically. That means the default compiler settings stay in the race. The tuner selects an ACF only if its measured result wins.

Brian：完全可能。Helion 会自动加入一个不使用 ACF 的候选配置。因此，编译器的默认设置仍会参与竞争。只有测得的结果胜出，调优器才会选用某个 ACF。

Ava: Why test kernel settings and compiler settings together? Couldn't I finish one search, then start the other?

Ava：为什么要一起测试内核设置和编译器设置？我不能先完成一轮搜索，再开始另一轮吗？

Brian: You could, but you'd freeze the first choice too early. Imagine choosing a delivery route before learning which vehicle you'll use. The best route for one vehicle might be poor for another.

Brian：可以，但那样会过早固定第一轮的选择。想象一下，还不知道要用哪种车，就先选好了送货路线。适合一种车的最佳路线，对另一种车可能很糟糕。

Ava: What's the GPU version of that problem?

Ava：在 GPU 上，这对应什么问题？

Brian: Block size and pipeline stages affect register pressure and shared-memory use. A compiler control can change how that pressure plays out. If you only test controls on one fixed block size, you may miss a better pairing.

Brian：块大小和流水线阶段数会影响寄存器压力和共享内存用量。编译器控制选项又会改变这些压力产生影响的方式。如果只在一个固定的块大小上测试控制选项，就可能错过更好的组合。

Ava: Right. So the joint search considers pairings that a two-step search might never see.

Ava：明白了。所以联合搜索会考察分两步搜索时可能根本看不到的组合。

Brian: That's the key idea. It doesn't promise the perfect answer, but it avoids that particular ordering problem. The price is more trials: each extra ACF expands the compile-and-measure search.

Brian：这就是关键思路。它不保证找到完美答案，但能避开这种搜索顺序带来的问题。代价是需要更多次试验：每增加一个 ACF，编译和测量的搜索范围就会扩大。

Ava: Where do the ACFs come from? Do users need to learn every ptxas control?

Ava：这些 ACF 从哪里来？用户需要学会 ptxas 的每一个控制选项吗？

Brian: Usually, no. NVIDIA's CompileIQ searches those controls. It proposes candidate settings, compiles them, checks correctness against a reference, times the valid candidates, and carries strong candidates into later rounds.

Brian：通常不需要。NVIDIA 的 CompileIQ 会搜索这些控制选项。它提出候选设置、进行编译、对照参考结果检查正确性、测量有效候选配置的运行时间，并把表现好的候选配置带入后续轮次。

Ava: That sounds like an evolutionary search. What does a typical user start with?

Ava：这听起来像进化式搜索。普通用户一般从哪里开始？

Brian: CompileIQ publishes pre-tuned catalogs called Booster Packs. The article's Helion pack has twenty-four ACFs, tuned across Helion kernels and validated against ptxas from CUDA thirteen point four for B two hundred.

Brian：CompileIQ 发布了名为 Booster Packs 的预调优配置集。文章中的 Helion 配置包包含 24 个 ACF，这些 ACF 在多个 Helion 内核上调优，并针对 B200 使用 CUDA 13.4 中的 ptxas 完成了验证。

Ava: Then I let Helion test that pack, and pin the winner for later runs?

Ava：然后我让 Helion 测试这个配置包，再把胜出的配置固定下来供以后运行使用？

Brian: Exactly. If the pack doesn't cover your workload, the article also describes running a targeted CompileIQ search. Either way, correctness checks matter before timing a candidate.

Brian：没错。如果这个配置包没有覆盖你的工作负载，文章还介绍了如何运行有针对性的 CompileIQ 搜索。无论采用哪种方式，都要先检查候选配置的正确性，再测量它的运行时间。

Ava: Let's get to the evidence. Did their main experiment actually perform the full joint search?

Ava：来说说证据吧。他们的主要实验真的进行了完整的联合搜索吗？

Brian: The detailed case study fixes an existing Helion configuration and searches compiler controls beneath it. That's a useful distinction. It tests whether the second layer has headroom even after Helion's first layer was tuned.

Brian：详细案例研究固定了一个已有的 Helion 配置，只搜索其下层的编译器控制选项。这是个有用的区别：它检验的是，即使 Helion 的第一层已经调优，第二层是否仍有提升空间。

Ava: Which kernel did they use?

Ava：他们用了哪个内核？

Brian: A pretuned NVFP4 decode kernel that multiplies packed weights by an activation vector and scales the output. It already had a checked-in configuration for the shape they tested.

Brian：一个预先调优的 NVFP4 解码内核，它将打包的权重与激活向量相乘，并对输出进行缩放。对于他们测试的形状，这个内核已经有提交到代码库中的配置。

Ava: And where was it measured?

Ava：测量是在哪里进行的？

Brian: On a GB two hundred system with Blackwell hardware. This experiment used ptxas from CUDA thirteen point three, which differs from the Booster Pack's stated validation version. They locked the GPU clock for timing.

Brian：在一台采用 Blackwell 硬件的 GB200 系统上。这项实验使用的是 CUDA 13.3 中的 ptxas，与 Booster Pack 声明的验证版本不同。测量运行时间时，他们锁定了 GPU 时钟频率。

Ava: What happened when CompileIQ optimized for runtime?

Ava：CompileIQ 以运行时间为优化目标时，结果怎么样？

Brian: Its search found an ACF that cut the measured time from about zero point zero nine seven one milliseconds to zero point zero nine two eight milliseconds. That's about four point seven percent faster in the search.

Brian：搜索找到的一个 ACF 将测得的运行时间从约 0.0971 毫秒降至 0.0928 毫秒。在搜索测量中，这相当于快了约 4.7%。

Ava: Search measurements can be lucky. Did they check again?

Ava：搜索时的测量结果可能碰巧偏好。他们重新检查了吗？

Brian: Yes. They independently remeasured the result twice, checked numerical output against a PyTorch reference, and applied statistical tests. Both reruns showed a speedup of about four point eight percent.

Brian：检查了。他们独立重复测量了两次，对照 PyTorch 参考实现检查数值输出，并进行了统计检验。两次重新测量都显示约 4.8% 的提速。

Ava: Could they aim the search at something other than runtime?

Ava：他们能把搜索目标设为运行时间以外的指标吗？

Brian: They did. A profiler suggested the kernel was limited by registers, with achieved occupancy around twenty-one point six percent. It estimated possible headroom from occupancy, but that estimate wasn't a measured speedup.

Brian：他们也这么做了。性能分析器显示，这个内核受到寄存器限制，实际占用率约为 21.6%。它估算了通过提高占用率可能获得的提升空间，但这个估算值不是实测的提速。

Ava: So they searched for higher occupancy directly?

Ava：所以他们直接搜索更高的占用率？

Brian: Yes. The chosen ACF raised achieved occupancy by zero point one six percentage points. Independent timing then showed a smaller runtime gain, roughly one point seven five to two percent.

Brian：是的。选出的 ACF 将实际占用率提高了 0.16 个百分点。随后进行的独立计时显示，运行时间也有所改善，但幅度较小，约为 1.75% 到 2%。

Ava: That's modest. What about optimizing two measures at once?

Ava：这个幅度不大。如果同时优化两个指标呢？

Brian: CompileIQ can score candidates on multiple measures. They paired runtime with L1TEX cache throughput. The search looks for a Pareto frontier: candidates where improving one measure might require giving up another.

Brian：CompileIQ 可以根据多个指标给候选配置评分。他们把运行时间与 L1TEX 缓存吞吐量放在一起考量。搜索会寻找帕累托前沿：在这些候选配置中，改善一项指标可能需要牺牲另一项。

Ava: Did they have to make that trade?

Ava：他们必须做出这种取舍吗？

Brian: In this search, no. One candidate beat the other evaluated candidates on both measures. Fresh validation found a six point nine percent runtime speedup and a four point five percentage-point rise in L1TEX throughput.

Brian：在这次搜索中，不需要。有一个候选配置在两项指标上都优于其他经过评估的候选配置。重新验证发现，运行速度提升了6.9%，L1TEX 吞吐量提高了4.5个百分点。

Ava: I also saw two input shapes. Was that one ACF for both?

Ava：我还看到了两种输入形状。它们用的是同一个 ACF 吗？

Brian: Yes. They searched for a shared ACF across two shapes. One candidate beat the default on both in independent validation: about five percent faster on the down-projection shape and about one percent on the other.

Brian：是的。他们为两种形状搜索了一个共用的 ACF。独立验证显示，有一个候选配置在两种形状上都优于默认配置：在下投影对应的形状上快了约5%，在另一种形状上快了约1%。

Ava: Does that mean an entire model gets six point nine percent faster?

Ava：这意味着整个模型也会快6.9%吗？

Brian: The article doesn't claim that. These are kernel results. For a model, you'd first profile which kernels dominate its work, then tune those. The model-level gain depends on how much time those kernels take.

Brian：文章没有这么说。这些是内核层面的结果。对于一个模型，你得先分析哪些内核占用了最多运行时间，再去调优它们。模型整体能提速多少，取决于这些内核耗时占多大比例。

Ava: What would the practical workflow look like for someone using Helion today?

Ava：对于现在使用 Helion 的人，实际操作流程会是什么样？

Brian: Pick a kernel that matters, search offline with matching ACFs, verify correctness and timing, then pin the winner. The pinned file can travel with the kernel and be reapplied during later builds.

Brian：选一个重要的内核，用与之匹配的 ACF 离线搜索，验证正确性和运行时间，然后固定胜出的配置。保存下来的配置文件可以和内核一起交付，并在以后构建时重新应用。

Ava: You said matching ACFs. What's the risk if they don't match?

Ava：你说 ACF 要匹配。如果不匹配，会有什么风险？

Brian: ACFs are specific to hardware and toolkit versions. The article warns that using one for the wrong architecture can cause wrong results or runtime errors. It's more serious than losing a little speed.

Brian：ACF 适用于特定的硬件和工具包版本。文章警告说，把它用在错误的架构上，可能导致结果错误或运行时出错。这比损失一点速度严重得多。

Ava: And the search itself sounds expensive.

Ava：而且搜索本身听起来也很耗资源。

Brian: It can be. Every ACF adds trials, and gains depend on the shape, workload, architecture, and remaining headroom. The measured result on your own kernel is the one to trust.

Brian：确实可能如此。每增加一个 ACF，试验次数就会增加；能提升多少则取决于输入形状、工作负载、硬件架构，以及还有多少优化空间。最该相信的是在你自己的内核上测得的结果。

Ava: Give me the three-point recap before we go.

Ava：结束前，给我做个三点回顾吧。

Brian: First, Helion tunes the kernel's work plan. Second, ACFs let it explore compiler choices underneath, and joint search can find useful pairings. Third, validate on your target hardware, then pin the measured winner.

Brian：第一，Helion 调优的是内核的工作方案。第二，ACF 让它能够探索更底层的编译器选项，联合搜索可能找到有效的搭配。第三，在目标硬件上验证，然后固定实测表现最好的配置。

Ava: A few percent can hide below a well-tuned kernel. Thanks for listening, and we'll see you next time.

Ava：即使是调优得很好的内核，也可能还藏着几个百分点的提升空间。感谢收听，我们下次再见。

## 术语

| Term | 释义 |
|---|---|
| Helion | 让开发者用 PyTorch 风格的 Python 编写 GPU kernel，并自动搜索执行配置的项目。 |
| autotuner | 自动编译、计时并选择 kernel 配置的程序。 |
| tile | 将计算范围划分后得到的一块工作区域。 |
| warp | GPU 上协同执行的一组线程。 |
| TorchInductor | 文中 Helion 降低到 Triton 时经过的编译组件。 |
| Triton | 文中 Helion 用于生成和编译候选 GPU kernel 的编程及编译层。 |
| ptxas | NVIDIA 的 PTX 汇编器，负责生成更底层的机器指令。 |
| register allocation | 编译器决定如何分配 GPU 寄存器的过程。 |
| occupancy | 衡量 GPU 上有多少工作处于活跃状态的指标。 |
| Advanced Control File (ACF) | 在编译时应用的一组 ptxas 高级控制设置文件。 |
| CompileIQ | NVIDIA 用于搜索编译器控制设置并生成 ACF 的工具。 |
| Booster Pack | CompileIQ 发布的预调优 ACF 文件集合。 |
| Pareto frontier | 多目标搜索中，在所有目标上都无法被另一候选方案同时超越的候选集合。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Where does the new layer begin? | 新的这一层从哪里开始？ |
| That's the key idea. | 这就是核心思路。 |
| Could they aim the search at something other than runtime? | 他们能否把搜索目标设为运行时间之外的指标？ |
| Did they check again? | 他们有没有重新验证？ |
| That's a useful distinction. | 这个区别很重要。 |
| What would the practical workflow look like? | 实际操作流程会是什么样？ |
| The measured result on your own kernel is the one to trust. | 最该相信的是在自己 kernel 上测得的结果。 |

## 中文内容总结

Helion 原本搜索线程块形状、warp 数和流水线阶段等内核配置。这篇文章提出进一步搜索 NVIDIA ptxas 的编译控制项，为已调优的内核寻找额外性能空间。文中的基线是 Helion 仓库里已按架构离线调优、运行时直接选用配置的内核。

### 问题与方法

内核配置决定每个线程块做什么，ptxas 的寄存器分配和指令调度等控制项则影响这些工作如何变成机器指令。ptxas 从 CUDA 13.3 起提供这一高级控制接口。两层参数会相互影响：改变线程块大小可能增加寄存器压力，使原先合适的编译设置失效。Helion 因此允许通过 autotune_search_acf，将多个高级控制文件 ACF 与内核配置一起计时搜索，并自动纳入不使用 ACF 的默认编译结果。搜索完成后，可用 advanced_controls_file 固定胜出的文件。

### ACF 的来源与搜索成本

NVIDIA CompileIQ 把编译控制项选择作为优化问题：生成候选设置，编译并检查输出正确性，再测量目标指标，逐代保留表现较好的候选。现有 helion-booster-pack 含 24 个 ACF，面向 B200，并经过 CUDA 13.4 的 ptxas 验证。用户可用该目录进行一次离线搜索；每增加一个 ACF 都会增加编译和测量次数。

### 固定内核配置的实测结果

文章的实验在 GB200（sm_100）上使用 ptxas 13.3.73、CompileIQ 1.0.3 和 Triton 3.8.0，针对已有 Helion 配置的 NVFP4 解码 GEMV 搜索 ACF。这些结果来自固定内核配置上的编译控制项搜索，不能直接视为联合搜索的实测收益。对 N=8192、K=28672，按运行时间搜索时，搜索过程测得延迟从 0.0971 毫秒降至 0.0928 毫秒，提升 4.7%；两次独立复测约提升 4.8%。所有候选先与 PyTorch 参考实现核对，复测使用锁定的 GPU 时钟，并以 Welch 检验 p<0.05、Cohen’s d>0.2 筛选性能提升。

### 其他优化目标

改以实际占用率为目标时，搜索得到的占用率从 21.60% 升至 21.76%，独立复测的运行时间提升约 1.75% 至 2.0%。Nsight Compute 给出的占用率潜在加速 21.01% 是模型估计，并非实测收益。同时优化运行时间和 L1TEX 吞吐率时，一个候选在独立验证中实现 6.9% 的运行时间提升，吞吐率从 81.51% 升至 85.96%。对两种输入形状搜索共用 ACF 后，所选候选在独立验证中分别约快 1.05 倍和 1.01 倍。

### 适用范围与限制

作者建议先找出模型中耗时的关键内核，再针对具体形状和硬件离线搜索、验证并固定 ACF；文章没有报告端到端模型加速。ACF 与硬件架构和工具链版本相关，错误匹配可能造成错误结果或运行失败，因此正确性检查不可省略。收益也取决于工作负载，文中的百分比不能直接套用到其他内核。Helion 是开源项目；CompileIQ 可免费使用，但采用 NVIDIA 专有许可，生成的 ACF 可以分发。
