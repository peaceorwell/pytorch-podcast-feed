# When Speech Recognition Has to Keep Up

原文：[Optimizing Streaming Speech Recognition on Arm® Edge Devices with ExecuTorch - PyTorch](https://pytorch.org/blog/optimizing-streaming-speech-recognition-on-arm-edge-devices-with-executorch/)

## 摘要

这期讨论如何用 ExecuTorch 在 Arm 边缘设备上运行流式语音识别，并同时评估准确率与响应延迟。文章中的 Nemotron 3.5 在较长音频块下获得更低的最终转写词错误率，但也必须等待更多音频到达。在 Google Pixel 11 的单线程测试中，启用 SME2 明显缩短了整段录音的处理时间；ETDump 和 Streamline 分别帮助定位算子耗时与实际运行的原生内核。选择部署配置时，需要把音频块大小、精度、词错误率和首个 token 的条件性预计延迟放在一起看；该估计假设首次输出会产生文本。

## 对话

Ava: When I speak to a device, I expect words to appear while I'm still talking. Strong batch accuracy doesn't guarantee that experience. Streaming gives the model only a piece of audio at a time, so both accuracy and response delay can change.

Ava：我对设备说话时，希望自己还在说，文字就已经出现。批处理准确率高，并不能保证这种体验。流式处理每次只给模型一段音频，因此准确率和响应延迟都可能变化。

Brian: Exactly. In a cross-model study cited here, Qwen3-ASR-1.7B had the best batch accuracy among the tested models, yet its WER rose by 77% with 2.4-second chunks. In that study's streaming comparison, Nemotron Speech Streaming 0.6B had the lowest average WER with 560 ms chunks.

Brian：没错。这里引用的一项跨模型研究中，在受测模型里，Qwen3-ASR-1.7B 的批处理准确率最高，但使用 2.4 秒的音频块时，它的词错误率（WER）上升了 77%。在该研究的流式处理对比中，Nemotron Speech Streaming 0.6B 使用 560 ms 的音频块时，平均 WER 最低。

Ava: So the chunk cadence has to be part of the accuracy test. The case study then moves to the newer, multilingual Nemotron 3.5. It asks a practical question: which chunk size and precision give useful transcripts soon enough on an edge device?

Ava：所以，音频块的处理节奏必须纳入准确率测试。接着，这个案例研究转向更新的多语言模型 Nemotron 3.5。它提出了一个实际问题：在边缘设备上，哪种音频块大小和数值精度能足够快地产生可用的转录文本？

Brian: The route starts with ExecuTorch, PyTorch's edge inference framework. Export prepares the model for inference and produces a .pte file. The device runtime loads that file. This gives the authors a way to run the model through a lightweight runtime on the phone.

Brian：这条运行路径从 ExecuTorch 开始，它是 PyTorch 的边缘推理框架。导出过程会为推理准备模型，并生成一个 .pte 文件。设备端运行时加载这个文件。这让作者能够通过手机上的轻量级运行时运行模型。

Ava: The exported model still needs CPU implementations for its operations. How does it reach the optimized Arm routines, and what happens when an operation isn't delegated to XNNPACK?

Ava：导出的模型仍需要各项操作的 CPU 实现。它是怎样调用到经过优化的 Arm 例程的？如果某项操作没有委托给 XNNPACK，又会怎样？

Brian: Operations supported for delegation go to XNNPACK. On Android, XNNPACK can select KleidiAI microkernels for supported inner matrix calculations. Operations not delegated to XNNPACK use ExecuTorch's portable CPU kernels. So export gets the model onto the device, while delegation determines which implementations do the work.

Brian：支持委托的操作会交给 XNNPACK。在 Android 上，XNNPACK 可以为受支持的内部矩阵计算选用 KleidiAI 微内核。没有委托给 XNNPACK 的操作则使用 ExecuTorch 的可移植 CPU 内核。所以，导出负责把模型部署到设备上，委托机制则决定实际由哪些实现来执行计算。

Ava: And SME2 is one possible acceleration path for matrix work. But merely enabling it in a build doesn't prove every matrix operation used it. We'd need native samples to see which microkernels actually ran.

Ava：SME2 也是矩阵计算的一条潜在加速路径。但仅仅在构建中启用它，并不能证明每项矩阵操作都用了它。我们需要查看原生代码的采样结果，才能知道实际运行了哪些微内核。

Brian: Right. SME2 extends Arm's instruction set for matrix calculations, including GEMM. In the Arm C1 CPU cluster described here, its matrix unit is shared by cores in that cluster. KleidiAI provides microkernels that can use the unit for supported operations.

Brian：对。SME2 扩展了 Arm 用于矩阵计算的指令集，其中包括 GEMM。在这里描述的 Arm C1 CPU 集群中，它的矩阵单元由该集群内的核心共享。KleidiAI 提供能够在受支持操作中使用这个单元的微内核。

Ava: Let's follow one streaming call. Nemotron 3.5 uses an RNN-T architecture, so the model has to represent the incoming audio and the tokens it has already emitted. Where are those jobs handled?

Ava：我们来跟踪一次流式调用。Nemotron 3.5 采用 RNN-T 架构，因此模型既要表示传入的音频，也要表示它已经输出的 token。这两项工作分别由哪里处理？

Brian: The encoder represents audio using FastConformer attention and convolution layers. The predictor is an LSTM that represents token history. Then the joiner combines those representations and predicts a token. It can also produce a blank, which moves decoding to the next audio position.

Brian：编码器通过 FastConformer 的注意力层和卷积层来表示音频。预测器是一个 LSTM，用来表示 token 历史。然后，联合网络把这两种表示结合起来，预测一个 token。它也可能输出空白符，让解码推进到下一个音频位置。

Ava: Before the encoder, the frontend makes one log-mel feature vector every 10 ms. The encoder reduces the number of time positions by a factor of 8. That means each encoder output position represents 80 ms of audio, with fewer positions to decode.

Ava：在进入编码器之前，前端每 10 ms 生成一个 log-mel 特征向量。编码器将时间位置的数量缩减到原来的 1/8。这意味着编码器的每个输出位置代表 80 ms 音频，需要解码的位置也更少。

Brian: At lookahead 13, a call collects 1,120 ms of audio and produces 14 encoder positions. The predictor and joiner process all 14 valid positions and add their tokens to the transcript. That amount of audio must arrive before this call can produce its update.

Brian：当前瞻量为 13 时，一次调用会收集 1,120 ms 音频，并产生 14 个编码器位置。预测器和联合网络会处理全部 14 个有效位置，把生成的 token 加入转录文本。这段音频必须先到达，这次调用才能产出更新。

Ava: Lookaheads 0, 3, and 6 instead give 80, 320, and 560 ms chunks. Shorter chunks let the system collect fresh audio more often. Longer chunks give the model more future context, but the listener waits longer for the first text and later updates.

Ava：前瞻量为 0、3 和 6 时，音频块分别为 80、320 和 560 ms。音频块越短，系统就能越频繁地收集新音频。音频块越长，模型获得的未来上下文越多，但听者要更久才能看到第一段文字和后续更新。

Brian: Preprocessing and inference add time after collection, too. So the authors measure two different things: transcription quality at each chunk size, and processing time on the device. A good WER result alone won't tell us when text appears.

Brian：收集音频之后，预处理和推理也需要时间。所以作者测量了两件不同的事：每种音频块大小对应的转录质量，以及设备上的处理时间。单看好的 WER 结果，无法知道文字何时出现。

Ava: For quality, they run the ExecuTorch runtime with its XNNPACK CPU backend on a host. A local streaming adapter uses the Open ASR Leaderboard evaluation harness for dataset preparation, transcript normalization, and WER scoring. That's a streaming test rather than a whole-recording batch score.

Ava：为了评估质量，他们在主机上运行采用 XNNPACK CPU 后端的 ExecuTorch 运行时。一个本地流式处理适配器使用 Open ASR Leaderboard 评测框架来准备数据集、规范化转录文本并计算 WER。这是流式测试，而非对整段录音进行批处理后得出的分数。

Brian: Each point covers 14,019 utterances across LibriSpeech Other, AMI Cleaned, Earnings22, and VoxPopuli Cleaned AA. WER is total word errors divided by total reference words across those four corpora. They test FP32, FP16, and INT8 at four chunk sizes.

Brian：每个数据点涵盖 LibriSpeech Other、AMI Cleaned、Earnings22 和 VoxPopuli Cleaned AA 中的 14,019 条语音。WER 的计算方式是这四个语料库中的总词错误数除以参考文本的总词数。他们在四种音频块大小下测试了 FP32、FP16 和 INT8。

Ava: The INT8 choice needs a little care. Here it's QC8W: eligible linear operations use dynamic INT8 activations and per-channel INT8 weights, with FP32 outputs. So a QC8W run isn't a claim that every operation executes entirely in INT8.

Ava：INT8 的选择需要稍作说明。这里使用的是 QC8W：符合条件的线性操作采用动态 INT8 激活值和逐通道 INT8 权重，输出为 FP32。因此，运行 QC8W 并不意味着每项操作都完全以 INT8 执行。

Brian: Across these corpora, longer chunks improve final-transcript accuracy. From 80 to 1,120 ms, FP16 WER falls from 15.2% to 11.5%. QC8W falls from 15.5% to 11.7%. That's a meaningful quality change from chunk size alone.

Brian：在这些语料库中，更长的音频块提高了最终转录文本的准确率。当音频块从 80 ms 增至 1,120 ms 时，FP16 的 WER 从 15.2% 降至 11.5%，QC8W 则从 15.5% 降至 11.7%。仅改变音频块大小，就带来了明显的质量变化。

Ava: FP16 differs from FP32 by less than 0.03 percentage points at every chunk size, so the authors leave FP32 out of the later performance analysis. QC8W adds 0.21 to 0.31 percentage points of WER relative to FP32. That makes the precision trade-off fairly specific.

Ava：在每种音频块大小下，FP16 与 FP32 的差异都小于 0.03 个百分点，因此作者在后续性能分析中不再纳入 FP32。相对于 FP32，QC8W 的 WER 增加了 0.21 至 0.31 个百分点。这样，数值精度的取舍就有了相当具体的量化依据。

Brian: But those host scores don't measure when text first appears on the device. Separately, the authors measure whole-recording processing time on a Google Pixel 11, using a 24.9-second recording and 1,120 ms chunks. One inference thread is pinned to a mid-core CPU, chosen as a representative sustained streaming target.

Brian：但这些主机上的分数并没有测量文字首次出现在设备上的时间。作者另外在 Google Pixel 11 上测量了整段录音的处理时间，使用一段 24.9 秒的录音和 1,120 ms 的音频块。单个推理线程被固定在一个中核 CPU 上，作者将其选作持续流式处理的代表性目标。

Ava: They compare two ExecuTorch 1.3.1 builds, with SME2 support switched on or off. Both retain XNNPACK and KleidiAI, and both use identical model artifacts. Each configuration has two warm-ups and one measured tracing run, so these totals describe that measured run.

Ava：他们比较了两个 ExecuTorch 1.3.1 构建版本，分别开启和关闭 SME2 支持。两者都保留 XNNPACK 和 KleidiAI，并使用完全相同的模型产物。每种配置进行两次预热和一次带跟踪记录的测量运行，因此这些总耗时描述的是那次测量运行。

Brian: With SME2 enabled, FP16 processing falls from 9.79 to 4.61 seconds, a 2.12-times speedup. QC8W falls from 5.20 to 3.24 seconds, a 1.61-times speedup. The totals cover the loaded recording from audio preprocessing through final detokenization.

Brian：启用 SME2 后，FP16 的处理时间从 9.79 秒降至 4.61 秒，加速了 2.12 倍。QC8W 则从 5.20 秒降至 3.24 秒，加速了 1.61 倍。这些总耗时涵盖已加载录音从音频预处理到最终去 token 化的整个过程。

Ava: Those processing times are well below the recording's 24.9-second duration. With SME2 on, the real-time factors are 0.19 for FP16 and 0.13 for QC8W. That leaves room for other CPU work between chunks, even with one inference thread.

Ava：这些处理时间远低于录音的 24.9 秒时长。开启 SME2 后，FP16 和 QC8W 的实时因子分别为 0.19 和 0.13。即使只有一个推理线程，音频块之间也留有余量供其他 CPU 工作使用。

Brian: There's an important limit to that interpretation. Audio still has to arrive before decoding can use it. Faster processing creates headroom and may help manage thermal load during sustained transcription, but it doesn't erase the wait imposed by the chosen chunk size.

Brian：但这种解读有一个重要限制：音频仍必须先到达，解码才能使用它。更快的处理速度带来余量，也可能有助于在持续转录时控制散热压力，但它无法消除所选音频块大小造成的等待。

Ava: To locate the gain, the authors use ETDump operator timings. They group work across the encoder, predictor, and joiner into GEMM, data movement, elementwise work, convolution, normalization, and other operations. Time outside those grouped events appears as residual overhead.

Ava：为了找到性能提升的来源，作者使用了 ETDump 记录的操作耗时。他们把编码器、预测器和联合网络中的工作归为 GEMM、数据搬运、逐元素计算、卷积、归一化及其他操作。没有落入这些分组事件的时间则计为剩余开销。

Brian: GEMM is the main source of the speedup. For FP16, its summed time drops from 8.2 to 3.1 seconds when SME2 is enabled. Its share of total processing time falls from 84% to 67%, though it remains the largest category.

Brian：GEMM 是加速的主要来源。对于 FP16，启用 SME2 后，GEMM 的累计耗时从 8.2 秒降至 3.1 秒。它占总处理时间的比例从 84% 降至 67%，但仍是耗时最多的一类。

Ava: QC8W shows the same shape: GEMM time drops from 4.1 to 2.0 seconds, and its share falls from 78% to 62%. That explains why speeding up matrix work has a large effect, while also showing that matrix work isn't the whole run.

Ava：QC8W 呈现出相同的趋势：GEMM 耗时从 4.1 秒降至 2.0 秒，占比从 78% 降至 62%。这解释了为什么加快矩阵计算会产生很大的效果，同时也表明矩阵计算并非运行过程的全部。

Brian: Data movement takes roughly the same time in both builds. As GEMM gets faster, data movement grows from 8% to 17% of FP16 processing and from 9% to 17% for QC8W. It's a clearer target for further optimization after the matrix gains.

Brian：数据搬运在两个构建版本中花费的时间大致相同。随着 GEMM 加快，数据搬运在 FP16 处理时间中的占比从 8% 升至 17%，在 QC8W 中则从 9% 升至 17%。在矩阵计算获得加速后，数据搬运就成了更明确的后续优化目标。

Ava: ETDump tells us which operator family costs time, but not which native microkernel executed it. How do we know SME2 code ran on the phone, rather than only being available in the build?

Ava：ETDump 能告诉我们哪类操作耗时，却不能告诉我们实际执行了哪个原生微内核。我们怎么知道 SME2 代码确实在手机上运行了，而不是仅仅存在于构建版本中？

Brian: Arm Streamline samples native call paths and hardware counters. The sampled GEMM functions confirm SME2-targeted KleidiAI kernels for both FP16 and QC8W on the Pixel 11. The captures also show other paths: some FP16 GEMM uses NEON, and some QC8W GEMM still runs in FP32.

Brian：Arm Streamline 对原生调用路径和硬件计数器进行采样。采样得到的 GEMM 函数证实，Pixel 11 上的 FP16 和 QC8W 都运行了面向 SME2 的 KleidiAI 内核。采集结果也显示了其他路径：部分 FP16 GEMM 使用 NEON，部分 QC8W GEMM 仍以 FP32 运行。

Ava: Streamline reports 38.4% SME utilization for FP16 and 53.7% for QC8W. That measures how much of each capture the matrix hardware was allocated. It isn't a measure of peak arithmetic throughput, so I wouldn't read those percentages as achieved compute efficiency.

Ava：Streamline 报告的 SME 利用率，FP16 为 38.4%，QC8W 为 53.7%。这个指标衡量的是矩阵硬件在每次采集期间有多少时间处于被分配使用的状态。它不是峰值算术吞吐量指标，所以我不会把这些百分比解读为实际达到的计算效率。

Brian: There's still a tooling gap. ETDump names model operators and their timings; Streamline samples native functions. Neither directly maps an individual model layer to the kernel function observed for it. The authors identify that mapping as future work.

Brian：工具链仍有一个缺口。ETDump 给出模型操作的名称和耗时；Streamline 则对原生函数采样。两者都无法直接把单个模型层映射到实际观察到的内核函数。作者把建立这种映射列为未来工作。

Ava: Now we can choose an operating point. The article pairs host WER with a Pixel 11 component benchmark. Its first-token number is an estimate: audio collection time plus the median encoder–predictor–joiner sequence time. It assumes that the first output actually emits text.

Ava：现在我们可以选择一个运行配置了。文章将主机上的 WER 与 Pixel 11 上的组件基准测试结果放在一起考量。其中的首个 token 延迟是估算值：音频收集时间加上编码器、预测器和联合网络这一处理序列的耗时中位数。它假设首次输出确实产生了文字。

Brian: The component benchmark uses synthetic input, fresh stream state, and 75 measured samples per configuration across three rounds. For an illustrative 13% WER ceiling, both precisions qualify at 320, 560, and 1,120 ms. At 320 ms, WER is 12.67% for FP16 and 12.97% for QC8W.

Brian：这个组件基准测试使用合成输入、全新的流状态，并针对每种配置分三轮收集了 75 个测量样本。以 13% 的 WER 上限为例，两种数值精度在 320、560 和 1,120 ms 时都符合要求。在 320 ms 时，FP16 的 WER 为 12.67%，QC8W 为 12.97%。

Ava: At 1,120 ms, those WERs improve to 11.50% and 11.72%, but audio collection alone takes longer. If the application has an 800 ms latency budget as well as that 13% WER ceiling, the 320 and 560 ms choices fit both conditions.

Ava：在 1,120 ms 时，这两个 WER 分别改善至 11.50% 和 11.72%，但仅收集音频就需要更长时间。如果应用除了 13% 的 WER 上限，还有 800 ms 的延迟预算，那么 320 和 560 ms 两种选择都满足这两个条件。

Brian: At 560 ms, FP16 pairs 12.01% WER with an estimated 691 ms delay; QC8W pairs 12.26% with 660 ms. The 80 ms choices miss the illustrated WER target, while the 1,120 ms choices miss the latency budget. It depends on which limits the application must meet.

Brian：在 560 ms 时，FP16 对应的 WER 为 12.01%，估算延迟为 691 ms；QC8W 对应的数字为 12.26% 和 660 ms。80 ms 的选择达不到示例中的 WER 目标，而 1,120 ms 的选择超出了延迟预算。具体怎么选，取决于应用必须满足哪些限制。

Ava: The useful lesson is to test streaming quality at the chunk cadence you'll deploy, then measure device processing separately. ExecuTorch can reach optimized Arm kernels, and SME2 gives this workload substantial processing headroom. ETDump and Streamline explain where that gain came from.

Ava：有用的经验是：按实际部署时的音频块节奏测试流式转录质量，再单独测量设备处理时间。ExecuTorch 能调用经过优化的 Arm 内核，而 SME2 为这项工作负载提供了可观的处理余量。ETDump 和 Streamline 则解释了这些性能提升来自哪里。

Brian: And choose chunk size together with precision. The accuracy numbers, collection wait, and compute delay all matter to what the user sees. That's it for this episode. Thanks for listening.

Brian：还要把音频块大小和数值精度放在一起选择。准确率、收集音频所需的等待时间，以及计算延迟，都会影响用户看到的结果。这期节目就到这里。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| streaming ASR | 流式自动语音识别；音频持续到达时分块转写。 |
| WER | 词错误率；词错误总数除以参考文本词数。 |
| chunk cadence | 收集新音频并发起一次处理的节奏。 |
| lookahead | 模型在一个音频块内可利用的未来上下文。 |
| RNN-T | 由编码器、预测器和联合网络配合完成转写的架构。 |
| log-mel feature vector | 从音频提取的对数梅尔频谱特征向量。 |
| ExecuTorch | PyTorch 的边缘端推理框架。 |
| XNNPACK | 可执行受支持算子的 CPU 推理库。 |
| KleidiAI microkernel | 针对 Arm CPU 内层计算优化的小型计算例程。 |
| SME2 | Arm Scalable Matrix Extension 2；加速矩阵计算的指令集扩展。 |
| GEMM | 通用矩阵乘法。 |
| QC8W | 针对符合条件的线性运算，使用动态 INT8 激活和逐通道 INT8 权重，并输出 FP32；部分 QC8W GEMM 运算仍以 FP32 运行。 |
| ETDump | 记录 ExecuTorch 算子事件与耗时的工具。 |
| real-time factor | 处理时间与音频时长的比值。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's follow one streaming call. | 我们来追踪一次流式调用。 |
| That means each encoder output position represents | 这意味着编码器的每个输出位置代表…… |
| There's an important limit to that interpretation. | 这种解读有一个重要限制。 |
| How do we know | 我们怎么确认…… |
| That leaves room for | 这为……留出了余量。 |
| It depends on which limits the application must meet. | 这取决于应用必须满足哪些约束。 |
| The useful lesson is to | 值得带走的经验是…… |

## 中文内容总结

这项案例研究将多语言流式语音识别模型 Nemotron 3.5 部署到 Google Pixel 11，联合考察音频分块、数值精度、识别错误率和设备处理时间。核心结论是：流式识别须按实际分块节奏评估；更长的分块提高了最终转写准确率，也推迟了文本出现。

### 问题与运行机制

整段录音的识别成绩不能直接代表流式表现。文中引用的跨模型研究发现，Qwen3-ASR-1.7B 虽在所测模型中取得最佳批量识别准确率，但使用 2.4 秒分块时，词错误率（WER）上升了 77%。Nemotron 3.5 采用 RNN-T：FastConformer 编码器处理音频，LSTM 预测器表示已输出的词元历史，联合网络预测下一个词元或空白。前端每 10 毫秒生成一个特征向量，编码器每 80 毫秒输出一个位置；所测分块长度为 80、320、560 和 1,120 毫秒。

### 从模型到 CPU 内核

ExecuTorch 将 PyTorch 模型导出为设备可加载的 .pte 文件。运行时把适用算子交给 XNNPACK，后者可选择 Arm KleidiAI 的优化微内核；其余算子由 ExecuTorch 的通用 CPU 内核执行。启用 SME2 可提供矩阵计算路径，但是否实际调用这些内核仍需原生性能采样确认。

### 分块与精度对准确率的影响

作者在主机上使用 ExecuTorch 的 XNNPACK 后端评估 FP32、FP16 和 QC8W，四个语料库合计 14,019 条语音。分块从 80 增至 1,120 毫秒时，FP16 的 WER 从 15.2% 降至 11.5%，QC8W 从 15.5% 降至 11.7%。各分块下 FP16 与 FP32 相差不到 0.03 个百分点；QC8W 比 FP32 高 0.21 至 0.31 个百分点。QC8W 对适用的线性运算使用动态 INT8 激活和按通道量化的 INT8 权重，输出为 FP32。这些结果衡量最终转写准确率，不代表设备上的出字时间。

### 设备处理时间与瓶颈

作者在 Pixel 11 上用单个绑定到中核的推理线程处理 24.9 秒录音，分块为 1,120 毫秒；两版 ExecuTorch 1.3.1 仅切换 SME2 支持，模型文件相同。一次计时运行显示，启用 SME2 后，FP16 总处理时间从 9.79 降至 4.61 秒，QC8W 从 5.20 降至 3.24 秒，分别快 2.12 倍和 1.61 倍。对应实时因子为 0.19 和 0.13，但录音仍须先按分块长度到达。ETDump 显示主要收益来自矩阵乘法：FP16 耗时从 8.2 降至 3.1 秒，QC8W 从 4.1 降至 2.0 秒；数据搬运耗时大致不变，因此成为更显著的剩余开销。

### 内核证据与选择条件

Arm Streamline 的采样确认，两种精度均调用了 KleidiAI 的 SME2 矩阵乘法内核；部分 FP16 运算仍使用 NEON，部分 QC8W 运算仍以 FP32 执行。测得的 SME 使用率为 FP16 的 38.4% 和 QC8W 的 53.7%，表示采样期间矩阵硬件的占用情况，并非峰值算力利用率。ETDump 和 Streamline 尚不能直接把每个模型层对应到具体原生内核。

### 部署取舍

作者用主机 WER 与 Pixel 11 的组件基准测试估计首次出字时间。若以 13% WER 和 800 毫秒为示例门槛，两种精度在 320 和 560 毫秒分块下均满足条件；80 毫秒分块未达到准确率门槛，1,120 毫秒分块超过延迟预算。560 毫秒时，FP16 为 12.01% WER、估计 691 毫秒，QC8W 为 12.26%、估计 660 毫秒。延迟值基于合成输入、全新流状态及每种配置三轮共 75 次测量的中位计算时间，并假设首次输出会产生文本；它不是实际交互场景的端到端出字测量。
