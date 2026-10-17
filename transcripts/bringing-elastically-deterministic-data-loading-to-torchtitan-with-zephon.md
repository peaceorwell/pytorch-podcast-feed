# Same Data, Different GPU Counts: Zephon in TorchTitan

原文：[Bringing Elastically Deterministic Data Loading to TorchTitan with Zephon - PyTorch](https://pytorch.org/blog/bringing-elastically-deterministic-data-loading-to-torchtitan-with-zephon/)

## 摘要

Zephon 让 TorchTitan 在改变 GPU 数量并从检查点恢复训练时，仍能向模型提供相同的全局批次，同时在训练过程中完成分词、打包、混合和打乱。它通过累加器、固定数量的逻辑通道，以及刷新后重放输入来保持数据顺序并缩小检查点。文章中的实验显示，使用普通数据加载器时，不同 GPU 数量会带来明显的评测分数差异；使用 Zephon 后，差异大幅缩小。恢复训练时仍需保持数据配方和全局批次大小不变，而且相同的批次不保证浮点计算结果完全一致。

## 对话

Ava: Imagine changing the number of GPUs halfway through training and getting a different result. If you’re testing a new data mixture, how would you know what helped?

Ava：想象一下，训练到一半时改变 GPUs 的数量，结果却变了。如果你正在测试一种新的数据混合方案，怎么知道究竟是什么起了作用？

Brian: That’s the problem Zephon tackles in TorchTitan. It keeps the global batches the same when a run resumes on a different number of GPUs—graphics processing units.

Brian：这正是 Zephon 在 TorchTitan 中要解决的问题。运行在不同数量的 GPUs（图形处理器）上恢复时，它会保持全局批次不变。

Ava: I thought a data loader just handed out samples. Why would the GPU count change which samples the model sees?

Ava：我以为数据加载器只是分发样本。为什么 GPUs 的数量会改变模型看到哪些样本？

Brian: Many loaders divide the stream among the GPUs they have. Change that count, and you can change the order after a checkpoint. The model then trains on a different sequence.

Brian：许多加载器会按现有的 GPUs 数量划分数据流。改变这个数量，就可能改变从检查点恢复后的顺序。模型随后就会按另一种序列训练。

Ava: And that makes a data experiment hard to trust.

Ava：这就让数据实验的结果难以令人信服。

Brian: Exactly. Teams often test a new mixture, filter, tokenizer, or packing method. That’s an ablation experiment: change something, then see whether it helped.

Brian：没错。团队经常测试新的混合方案、过滤器、分词器或打包方法。这就是消融实验：改动一项内容，再看它是否有帮助。

Ava: There’s another cost before the experiment even starts, right?

Ava：实验还没开始，就已经有另一项成本了，对吧？

Brian: Yes. Preparing each version can mean processing the whole corpus offline and saving a new copy. A typical training run reads that copy once. Many experiments don’t improve the model, so that work and storage go to waste.

Brian：对。准备每个版本可能都要离线处理整个语料库，并保存一份新副本。一次典型的训练运行只会读取这份副本一次。许多实验并不能改进模型，因此这些处理工作和存储空间就浪费了。

Ava: Agents can launch lots of experiments now. Does that make the data order problem worse?

Ava：现在智能体可以发起大量实验。这会让数据顺序的问题更严重吗？

Brian: It makes it easier to miss. An agent might attribute a score change to its data edit when the GPU allocation also changed. The article says people can miss that too.

Brian：这会让问题更容易被忽略。GPU 分配也发生变化时，智能体可能把分数变化归因于自己对数据的修改。文章说，人也可能忽略这一点。

Ava: So what does Zephon ask me to define?

Ava：那么 Zephon 需要我定义什么？

Brian: Three things. Datasets describe the sources. A WorkSource plans how they’re mixed and ordered. Then a Pipeline of operators turns stored samples into the tensors the training loop uses.

Brian：三件事。Datasets 描述数据来源。WorkSource 规划它们如何混合及排序。然后，由一系列算子组成的 Pipeline 将存储的样本转换为训练循环使用的张量。

Ava: Can you make that concrete without reading me a screen of code?

Ava：能不能讲得具体些，别给我念满屏代码？

Brian: The article’s example mixes two text datasets. The pipeline tokenizes the text, packs it into fixed-length sequences, shuffles it, and makes batches. That processing happens inside the training job.

Brian：文章中的例子混合了2个文本数据集。数据处理流水线会对文本进行分词，将其打包成固定长度的序列、打乱顺序，再组成批次。这些处理都在训练任务内部完成。

Ava: So it doesn’t need a saved, preprocessed copy for every experiment.

Ava：所以不必为每个实验都保存一份预处理后的副本。

Brian: Right. It can use CPUs available while the model trains. And because the WorkSource is separate, you can change the mixture without rewriting the processing steps.

Brian：对。它可以利用模型训练期间可用的 CPUs。而且，由于 WorkSource 是独立的，你可以改变混合方案，而不必重写处理步骤。

Ava: The example uses equal mixture weights. Equal numbers of documents?

Ava：例子使用了相等的混合权重。意思是文档数量相等吗？

Brian: Equal proportions of tokens, not samples. TokenEstimation checks the token-to-byte ratio in a small sample of each dataset. It uses that estimate to plan the mixture while the stored data is still raw text.

Brian：相等的是 token 所占的比例，不是样本数量。TokenEstimation 会检查每个数据集的一小部分样本中 token 与字节的比率。存储的数据仍是原始文本时，它就用这个估计值来规划混合方案。

Ava: Got it. But packing and shuffling on the fly sound tricky if you want to restart elsewhere.

Ava：明白了。不过，要在处理过程中实时打包和打乱顺序，还能在别处重新启动，听起来挺棘手。

Brian: They are. Picture a packing bin that’s partly full when you stop. The next output depends on what’s already inside. You can’t simply look up the next training sequence at an index on disk.

Brian：确实。想象一下，停止时有个打包容器只装了一部分。下一条输出取决于里面已有的内容。你无法直接按磁盘上的索引查到下一条训练序列。

Ava: That’s the stateful pipeline problem. How does Zephon keep parallel workers from changing the order?

Ava：这就是有状态 Pipeline 的问题。Zephon 怎样防止并行工作进程改变顺序？

Brian: It separates decisions from expensive work. A single-threaded accumulator sees records in order and decides things like which sequences to pack together. Stateless functions do the costly processing across workers.

Brian：它把决策和耗时的工作分开。单线程的累加器按顺序读取记录，决定哪些序列应打包在一起等事项。无状态函数则让多个工作进程并行完成耗时的处理。

Ava: Workers can finish in any order, though.

Ava：不过，工作进程可能以任意顺序完成任务。

Brian: They can. Zephon numbers groups of records before sending them out, then restores their order afterward. That makes worker count a performance setting. More tokenization workers won’t change the output.

Brian：没错。Zephon 在分发前给各组记录编号，之后再恢复它们的顺序。这样一来，工作进程数量就只是性能设置。增加分词工作进程不会改变输出。

Ava: So far, that handles workers. What handles a different number of GPUs?

Ava：到这里解决的是工作进程的问题。不同数量的 GPUs 又怎么处理？

Brian: Lanes. They’re fixed logical streams. Each lane has an ordered sample sequence and its own operator state, including any partly filled packing bins.

Brian：靠 lanes，也就是固定的逻辑数据流。每条 lane 都有按顺序排列的样本序列，以及自己的算子状态，包括那些只装了一部分的打包容器。

Ava: Like numbered conveyor belts that can be assigned to different stations?

Ava：就像有编号的传送带，可以分配给不同的工位？

Brian: That’s a useful picture. The lane produces the same records whichever GPU gets it. When the number of data-parallel ranks changes, Zephon redistributes the lanes. The global batches stay the same.

Brian：这个比喻很贴切。不管分配给哪个 GPU，同一条 lane 都会产生相同的记录。当数据并行 rank 的数量变化时，Zephon 会重新分配这些 lanes。全局批次保持不变。

Ava: Let me check I’ve got it: accumulators keep each lane’s decisions in order, and lanes let the run move across GPU counts.

Ava：我确认一下是否理解对了：累加器让每条 lane 的决策保持顺序，而 lanes 让训练运行可以在不同数量的 GPUs 之间迁移。

Brian: Exactly. There’s one planning choice: you pick the lane count up front. It needs to divide evenly across the data-parallel sizes you expect.

Brian：完全正确。有一项规划选择：你要预先确定 lane 的数量。这个数量必须能被你预计使用的各个数据并行规模整除。

Ava: What happens to those partly filled bins at a checkpoint? Saving every bit of pipeline state sounds large.

Ava：到了检查点，那些只装了一部分的容器怎么办？保存 Pipeline 的每一项状态听起来会占很大空间。

Brian: Zephon doesn’t save the bin contents or shuffle buffers. It periodically sends a flush sentinel through the pipeline. Stateful operators emit what they’re holding, then start from a clean slate.

Brian：Zephon 不会保存容器内容或打乱顺序时使用的缓冲区。它会定期让一个刷新哨兵标记通过 Pipeline。有状态算子输出暂存的内容，然后从干净的状态重新开始。

Ava: And when training resumes?

Ava：那训练恢复时呢？

Brian: The checkpoint records which input chunks were still in progress. Zephon replays input from the latest flush to rebuild the state, then discards output the training loop already consumed.

Brian：检查点会记录哪些输入数据块仍在处理中。Zephon 从最近一次刷新处重新读取输入，重建状态，然后丢弃训练循环已经使用过的输出。

Ava: So replay buys a smaller checkpoint.

Ava：所以重放换来了更小的检查点。

Brian: Yes. In the article’s experiments, those checkpoints were a few MiB—mebibytes—regardless of how far the run had progressed.

Brian：对。在文章的实验中，无论训练运行推进了多远，这些检查点都只有几个 MiB（兆二进制字节）。

Ava: How does this fit into TorchTitan?

Ava：它是怎么集成到 TorchTitan 中的？

Brian: The integration replaces TorchTitan’s default data loader with a Zephon Pipeline built from a data recipe. The recipe includes datasets, mixture weights, and settings such as packing, shuffling, and parallelism.

Brian：这项集成用根据数据配置构建的 Zephon Pipeline 替换了 TorchTitan 的默认数据加载器。配置中包括数据集、混合权重，以及打包、打乱顺序和并行度等设置。

Ava: Does the rest of the training run need to change?

Ava：训练运行的其他部分需要改变吗？

Brian: The article says it stays the same. Zephon’s state goes into the TorchTitan checkpoint with the model. That state covers every lane, so the run can resume on a different GPU count.

Brian：文章说，其他部分保持不变。Zephon 的状态与模型一起写入 TorchTitan 检查点。该状态覆盖每条 lane，因此训练可以在不同数量的 GPUs 上恢复。

Ava: What if I choose a lane count that doesn’t fit the run?

Ava：如果我选择的 lane 数量不适合这次训练，会怎样？

Brian: The integration checks at launch that lanes divide evenly across data-parallel replicas. It also checks that each optimizer step’s batch splits evenly across lanes. A bad configuration fails early.

Brian：集成组件会在启动时检查 lanes 能否在数据并行副本之间平均分配，还会检查每次优化器更新所用的批次能否平均分到各条 lane。配置有误就会及早报错。

Ava: Now for evidence. Did they actually resume a training run with different GPU counts?

Ava：现在说说证据。他们真的用不同数量的 GPUs 恢复过训练运行吗？

Brian: They trained a one-billion-parameter model in TorchTitan on sixteen GPUs, checkpointed at step one hundred fifty, then resumed on eight, thirty-two, and sixty-four GPUs, alongside the original run.

Brian：他们在TorchTitan中用16块GPU训练了一个拥有10亿参数的模型，在第150步保存检查点，然后分别在8块、32块和64块GPU上从检查点恢复训练，同时继续运行原来的训练任务。

Ava: What did the ordinary loader do?

Ava：普通加载器表现如何？

Brian: After the resume, it supplied different sample sequences. Each GPU count ended up with a distinct loss curve.

Brian：恢复后，它提供了不同的样本序列。每种 GPU 数量最终都有一条不同的损失曲线。

Ava: And Zephon?

Ava：Zephon 呢？

Brian: Its loss curves lay on top of each other. The SHA-256 hash—a fingerprint of the data—matched the baseline for every global batch. That checks the batch contents step by step.

Brian：它的损失曲线彼此重合。每个全局批次的 SHA-256 哈希值，也就是数据的指纹，都与基准运行匹配。这逐步核对了每个批次的内容。

Ava: Matching batches is clear. Did the loader affect evaluation scores too?

Ava：批次匹配这一点很清楚。加载器也会影响评估分数吗？

Brian: In another test, they trained the same model from scratch for twenty billion tokens on eight, sixteen, thirty-two, and sixty-four GPUs. With the standard loader, DCLM Core version one scores varied by up to zero point eight two points.

Brian：在另一项测试中，他们分别用8块、16块、32块和64块GPU，从头训练同一个模型，训练量为200亿个token。使用标准数据加载器时，DCLM Core v1的分数差异最大达到0.82分。

Ava: What happened on the FineWeb evaluation suite?

Ava：FineWeb 评估套件上发生了什么？

Brian: Those scores varied by up to zero point six four points. The article says these spreads were roughly four times the relevant thresholds used by the DCLM and FineWeb authors to evaluate data curation methods.

Brian：这些分数的差异最大达到0.64分。文章称，这些差异约为DCLM和FineWeb的作者评估数据筛选方法时所用相关阈值的4倍。

Ava: Wait, so a comparison between data mixtures could pick up the GPU allocation instead?

Ava：等等，那比较不同数据混合方案时，结果可能反而反映了 GPU 的分配情况？

Brian: Yes, if the allocations differ. With Zephon, the score spreads across GPU counts were zero point zero five on DCLM Core version one and zero point zero one one on FineWeb.

Brian：对，如果GPU的分配方式不同。使用Zephon时，不同GPU数量下的分数差异在DCLM Core v1上为0.05，在FineWeb上为0.011。

Ava: Does doing all that processing during training slow the GPUs down?

Ava：训练期间做这么多处理，会拖慢 GPU 吗？

Brian: That’s the practical question. Zephon lets each operator use its own level of parallelism. In the example, tokenization gets four workers because it’s the most expensive step.

Brian：这就是实际要考虑的问题。Zephon允许每个算子使用各自的并行度。在这个例子中，分词使用4个工作线程，因为它是开销最大的步骤。

Ava: And on a more complex pipeline?

Ava：那更复杂的流水线呢？

Brian: On the article’s full vision-language pipeline, which decodes images and then tokenizes, packs, and batches samples, Zephon kept pace with Google Grain. The authors call Grain the fastest loader they tested.

Brian：在文章介绍的完整视觉语言流水线上，系统先解码图像，再对样本进行分词、打包和批处理，Zephon 的速度跟上了 Google Grain。作者称 Grain 是他们测试过的最快的数据加载器。

Ava: Does Grain offer the same behavior when a run changes GPU counts?

Ava：运行时改变 GPU 数量，Grain 也能做到同样的事吗？

Brian: The article says Grain and Mosaic Streaming can preserve global batches across GPU counts when sample order can be worked out ahead of time from an index. That condition rules out packing on the fly.

Brian：文章说，如果能提前根据索引确定样本顺序，Grain 和 Mosaic Streaming 就能在不同 GPU 数量下保持全局批次不变。这个条件排除了即时打包。

Ava: And the other loader they mention, Megatron-Energon?

Ava：那他们提到的另一个加载器 Megatron-Energon 呢？

Brian: It can reproduce batches with a different number of GPUs if the total number of data loader workers stays the same. Zephon’s design removes that worker-count condition.

Brian：如果数据加载器工作进程的总数保持不变，它就能在 GPU 数量不同时复现批次。Zephon 的设计去掉了这个对工作进程数量的要求。

Ava: What should I watch out for when setting up a run?

Ava：设置一次训练运行时，我该注意什么？

Brian: Choose lanes for the data-parallel sizes you expect. The article gives thirty-two lanes as a cap covering eight, sixteen, and thirty-two GPUs. Thousands of lanes for a small run can add restore-time overhead.

Brian：要根据预计使用的数据并行规模选择通道数。文章举的例子是将通道数上限设为32，以覆盖8块、16块和32块GPU的情况。对于小规模运行，设置数千个通道可能增加恢复运行时的开销。

Ava: Can I edit the data recipe when I resume on more GPUs?

Ava：如果恢复训练时用了更多 GPU，我能修改数据配置吗？

Brian: Keep the data, tokenizer, seed, mixture, context length, and global batch size fixed. You can still tune performance settings such as parallelism.

Brian：数据、分词器、随机种子、数据混合方案、上下文长度和全局批次大小都要保持不变。并行度等性能设置仍然可以调整。

Ava: One last distinction: do identical batches mean identical loss values?

Ava：最后一个区别：批次完全相同，损失值也会完全相同吗？

Brian: No. Zephon guarantees the data. Floating-point reductions can happen in a different order when the GPU count changes, so losses should be very close but may differ.

Brian：不会。Zephon 保证数据一致。GPU 数量变化时，浮点数归约的顺序可能不同，所以损失值应该非常接近，但可能有差异。

Ava: What’s next for the project?

Ava：这个项目接下来有什么计划？

Brian: The authors highlight GIL-free Python, meaning Python without the interpreter lock, and dynamic curricula. They report that a thread-based Zephon runner on Python three point fourteen t fetched vision-language samples about three point seven times faster than its process runner on standard Python.

Brian：作者重点提到了无GIL的Python，也就是没有解释器锁的Python，以及动态课程学习。他们报告说，在Python 3.14t上，基于线程的Zephon运行器获取视觉语言样本的速度，约为标准Python上基于进程的运行器的3.7倍。

Ava: And a dynamic curriculum would change the mixture during training?

Ava：动态课程学习会在训练过程中改变数据混合方案吗？

Brian: That’s the idea. Because the WorkSource is separate from the processing Pipeline, it could change mixture weights over a run. They say they’re working on a configurable component that keeps the determinism guarantees.

Brian：就是这个想法。由于 WorkSource 与负责处理的 Pipeline 是分开的，它可以在运行过程中调整数据混合权重。他们说自己正在开发一个可配置的组件，同时保持确定性保证。

Ava: Give me the three-point recap.

Ava：给我做个3点总结。

Brian: First, Zephon prepares data inside the training job. Second, ordered accumulators, lanes, and replay keep global batches stable across GPU counts. Third, keep the recipe fixed when resuming, and remember that matching data doesn’t guarantee identical arithmetic.

Brian：第一，Zephon 在训练作业内部准备数据。第二，有序累加器、lanes 和重放机制让全局批次在不同 GPU 数量下保持稳定。第三，恢复训练时保持数据配置不变，并记住数据一致不保证运算结果完全相同。

Ava: That’s a much cleaner way to test the data. Thanks for listening.

Ava：这样测试数据就清晰多了。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| ablation experiment | 消融实验；改变一项设置并观察效果的实验。 |
| global batch | 全局批次；一个训练步骤中所有数据并行设备合起来处理的数据。 |
| WorkSource | Zephon 中规划数据集混合方式与顺序的组件。 |
| Pipeline | 把存储的样本逐步处理成训练张量的操作流程。 |
| TokenEstimation | 根据少量样本估计词元与字节比例、辅助规划按词元计的数据混合比例的工具。 |
| tokenization | 分词；将文本转换为模型处理的词元。 |
| packing | 打包；将样本组织进固定长度的训练序列。 |
| accumulator | 累加器；按顺序查看记录并作出有状态决策的组件。 |
| lane | 逻辑通道；拥有固定样本顺序和独立操作状态的数据流。 |
| flush sentinel | 刷新标记；让有状态操作输出暂存内容并重新开始的信号。 |
| data-parallel rank | 数据并行进程编号；承担部分数据处理工作的并行训练单元。 |
| checkpoint | 检查点；用于之后恢复训练的状态记录。 |
| elastic determinism | 弹性确定性；改变 GPU 数量后仍保持全局批次一致的性质。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Can you make that concrete? | 能说得具体一点吗？ |
| Got it. | 明白了。 |
| Let me check I’ve got it. | 我确认一下自己是否理解对了。 |
| That’s the practical question. | 这才是实际应用中要问的问题。 |
| Wait, so a comparison between data mixtures could pick up the GPU allocation instead? | 等等，也就是说，比较数据配比时看到的差异可能来自 GPU 分配？ |
| What should I watch out for? | 我需要注意什么？ |
| That’s the idea. | 思路就是这样。 |
| Give me the three-point recap. | 用三点帮我回顾一下。 |
