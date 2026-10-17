# Why DeepEP and MXFP8 Make DeepSeek-V3 Training 41% Faster

原文：[Enabling Up to 41% Faster Pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200 with TorchTitan](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/)

## 摘要

本期讨论 PyTorch 与 Nebius 如何在 256 张 NVIDIA B200 GPU 上，用 TorchTitan 训练 DeepSeek-V3 的 671B 和 16B MoE 模型。实验比较了 BF16 基线、DeepEP 通信优化和 MXFP8 grouped GEMM 计算优化：671B 模型从每秒 651 个 token 提升到 918 个，整体快了 41%。其中 DeepEP 单独带来 32% 提升，说明跨节点 all-to-all 通信是主要瓶颈。对 16B 模型进行 1500 步训练后，MXFP8 与 BF16 的 loss 曲线几乎一致，显示没有明显收敛损失。

## 对话

Ava: What if you could train a 671B-parameter Mixture-of-Experts model 41% faster, using open-source PyTorch tools? That’s the result we’re unpacking today.

Ava：如果能用开源 PyTorch 工具，把一个 671B 参数的 Mixture-of-Experts 模型训练速度提高 41%，会怎样？这就是我们今天要解读的结果。

Brian: And the interesting part is that the gain comes from two different places: MXFP8 speeds up matrix multiplication, while DeepEP speeds up expert communication. Together, they reached 918 tokens per second on 256 NVIDIA B200 GPUs.

Brian：有意思的是，提升来自两个不同的方面：MXFP8 加速矩阵乘法，DeepEP 加速专家之间的通信。两者结合后，在 256 块 NVIDIA B200 GPU 上达到了每秒 918 个 token。

Ava: Let’s start with the setup. This was a joint PyTorch and Nebius experiment, right?

Ava：先说实验配置。这是 PyTorch 和 Nebius 联合开展的实验，对吧？

Brian: Right. They used TorchTitan as the pre-training framework on a Nebius Cloud cluster. The cluster had 32 nodes, eight B200 GPUs per node, so 256 GPUs in total.

Brian：对。他们在 Nebius Cloud 集群上使用 TorchTitan 作为预训练框架。集群有 32 个节点，每个节点配备八块 B200 GPU，总共 256 块 GPU。

Ava: And the model was DeepSeek-V3, with both a 671B version and a 16B MoE version.

Ava：模型是 DeepSeek-V3，包括一个 671B 版本和一个 16B MoE 版本。

Brian: Exactly. The 671B experiment measured throughput. The 16B experiment checked whether MXFP8 changes training convergence.

Brian：没错。671B 实验测量的是吞吐量。16B 实验检验的是 MXFP8 是否会改变训练的收敛情况。

Ava: Why focus on Mixture-of-Experts models? What makes them difficult to train?

Ava：为什么聚焦 Mixture-of-Experts 模型？它们的训练难点是什么？

Brian: They have two major costs. First, the experts do a lot of matrix multiplication. Second, every layer needs two all-to-all exchanges: one to dispatch tokens to experts, and another to combine the results.

Brian：它们有两项主要开销。第一，专家要执行大量矩阵乘法。第二，每一层都需要进行两次 all-to-all 交换：一次把 token 分发给专家，另一次汇总结果。

Ava: So computation and communication can both become bottlenecks.

Ava：所以计算和通信都可能成为瓶颈。

Brian: Yes. And the communication pattern is especially awkward. The router decides dynamically which tokens go to which experts. That means the transfer sizes and destinations change at every step.

Brian：是的。而且通信模式尤其棘手。路由器会动态决定哪些 token 发往哪些专家。这意味着传输的数据量和目的地每一步都会变化。

Ava: Which is a bad match for a generic all-to-all operation designed around fixed transfers.

Ava：这与按固定传输设计的通用 all-to-all 操作很不匹配。

Brian: That’s the idea. As expert parallelism, or EP, grows, this communication cost gets worse. In the 671B run, EP was 32 across 32 nodes, so inter-node traffic mattered a lot.

Brian：就是这样。随着专家并行，也就是 EP，的规模增大，通信开销会进一步上升。在 671B 的运行中，EP 为 32，横跨 32 个节点，所以节点间流量影响很大。

Ava: Okay. Let’s talk about the first optimization: MXFP8. What does the name mean?

Ava：好。来说第一个优化：MXFP8。这个名字是什么意思？

Brian: MXFP8 means Microscaling FP8. It’s a low-precision format defined by the OCP Microscaling Specification. Standard float8 often uses one scale factor per tensor or per row. MXFP8 uses a shared exponent, called an E8M0 scale, for every block of 32 elements.

Brian：MXFP8 意为 Microscaling FP8。它是由 OCP Microscaling Specification 定义的一种低精度格式。标准 float8 通常为每个张量或每一行使用一个缩放因子。MXFP8 则为每个包含 32 个元素的块使用一个共享指数，称为 E8M0 scale。

Ava: So the scaling is more local.

Ava：也就是说，缩放是在更局部的范围内进行的。

Brian: Right. That finer-grained scaling helps preserve numerical fidelity while still using FP8 tensor cores. NVIDIA Blackwell supports MXFP8 natively through its tcgen05.mma tensor core instructions.

Brian：对。这种粒度更细的缩放有助于保持数值保真度，同时仍能使用 FP8 tensor cores。NVIDIA Blackwell 通过其 tcgen05.mma tensor core 指令原生支持 MXFP8。

Ava: That’s why B200 is a natural target. There’s no emulation overhead for MXFP8 GEMMs.

Ava：所以 B200 很适合用来运行它。MXFP8 GEMM 不会产生模拟实现的额外开销。

Brian: Yes. Blackwell can provide up to two times the peak TFLOPS of BF16 for eligible GEMMs. But the article measures end-to-end training, so the real speedup depends on overhead and on which operations are eligible.

Brian：是的。对于适用的 GEMM，Blackwell 能提供最高达 BF16 两倍的峰值 TFLOPS。但文章测量的是端到端训练，因此实际加速效果取决于额外开销，以及哪些操作适用。

Ava: How did TorchTitan apply it?

Ava：TorchTitan 是怎么应用它的？

Brian: Through TorchAO. For linear layers, TorchAO dynamically quantizes inputs to MXFP8 for the forward output, input gradient, and weight gradient GEMMs. The results accumulate back to BF16.

Brian：通过 TorchAO。对于线性层，TorchAO 会动态地把输入量化为 MXFP8，供前向输出、输入梯度和权重梯度这三类 GEMM 使用。结果则以 BF16 累积。

Ava: And for the experts?

Ava：那专家部分呢？

Brian: For routed experts, it converts torch grouped matrix multiplication operations to a TorchAO grouped GEMM function. The same three GEMMs use dynamically quantized MXFP8 inputs.

Brian：对于被路由到的专家，它会把 torch 分组矩阵乘法操作转换为 TorchAO 的 grouped GEMM 函数。同样，这三类 GEMM 都使用动态量化为 MXFP8 的输入。

Ava: The article says grouped GEMMs dominate MoE expert layers.

Ava：文章说，grouped GEMM 是 MoE 专家层中的主要计算。

Brian: Exactly. The recipe quantizes inputs immediately before each grouped GEMM. When the GEMM dimensions are large, the compute cost grows faster than the quantization cost, so the overhead becomes relatively small.

Brian：没错。这套方案会在每次 grouped GEMM 执行前立即量化输入。当 GEMM 的维度较大时，计算成本的增长快于量化成本，因此量化开销所占的比例会变小。

Ava: But for small dimensions, quantization can cancel the benefit.

Ava：但如果维度较小，量化就可能抵消收益。

Brian: Yes. The article is clear about that limitation. In smaller MoE models, the quantization overhead can match or exceed the MXFP8 speedup, leading to neutral or even negative performance.

Brian：是的。文章明确指出了这一局限。在较小的 MoE 模型中，量化开销可能与 MXFP8 带来的加速相当，甚至超过它，导致性能没有提升，乃至下降。

Ava: Now DeepEP sounds like a very different kind of optimization.

Ava：DeepEP 听起来是另一种完全不同的优化。

Brian: It is. DeepEP is a GPU communication library developed by DeepSeek specifically for MoE expert-parallel dispatch and combine operations.

Brian：确实如此。DeepEP 是 DeepSeek 专门为 MoE 专家并行中的分发和汇总操作开发的 GPU 通信库。

Ava: What does it change compared with standard all-to-all?

Ava：与标准 all-to-all 相比，它改变了什么？

Brian: It uses purpose-built NVLink and RDMA kernels. Inside a node, data moves over high-bandwidth NVLink. Between nodes, it uses RDMA over InfiniBand. This hierarchical path matches the hardware’s bandwidth differences.

Brian：它使用专门设计的 NVLink 和 RDMA 内核。在节点内部，数据通过高带宽的 NVLink 传输；在节点之间，则通过 InfiniBand 上的 RDMA 传输。这种分层路径契合了硬件带宽的差异。

Ava: The article mentions around 900 gigabytes per second for intra-node NVLink.

Ava：文章提到，节点内 NVLink 的带宽约为每秒 900 gigabytes。

Brian: Yes, that’s the approximate figure given for the setup. DeepEP also uses NVSHMEM, which provides a Partitioned Global Address Space. Each GPU can map a symmetric memory region that peers can address.

Brian：是的，这是文章给出的该配置下的大致数值。DeepEP 还使用 NVSHMEM，它提供了 Partitioned Global Address Space。每块 GPU 都可以映射一块对等 GPU 能够访问的对称内存区域。

Ava: And with InfiniBand GPUDirect Async, the GPU can drive network operations directly from CUDA kernels.

Ava：借助 InfiniBand GPUDirect Async，GPU 还能直接从 CUDA 内核发起网络操作。

Brian: Right. That removes CPU involvement, which is useful for small, dynamic transfers. DeepEP also fuses metadata such as token embeddings, expert indices, and routing weights into one communication operation.

Brian：对。这省去了 CPU 的参与，对于小规模、动态变化的数据传输很有用。DeepEP 还把 token 嵌入、专家索引和路由权重等元数据融合进一次通信操作。

Ava: Fewer launches and fewer synchronization points.

Ava：这样启动次数和同步点都更少了。

Brian: Exactly. It can also be configured to use a chosen number of streaming multiprocessors, leaving the remaining SMs available for overlapping computation.

Brian：没错。还可以配置 DeepEP 使用指定数量的 streaming multiprocessors，让其余 SMs 可用于与通信重叠执行的计算。

Ava: Before the results, tell me about the actual 671B training configuration.

Ava：看结果之前，先说说 671B 训练的实际配置。

Brian: The sequence length was 8192, with a local batch size of 64. Tensor Parallel, or TP, was 2. Pipeline Parallel, PP, was 2. Data Parallel, DP, was 1. Expert Parallel was 32, and Expert Tensor Parallel was 1.

Brian：序列长度为 8192，本地批量大小为 64。Tensor Parallel，也就是 TP，为 2；Pipeline Parallel，即 PP，为 2；Data Parallel，即 DP，为 1；Expert Parallel 为 32；Expert Tensor Parallel 为 1。

Ava: They also used full activation checkpointing and torch.compile for the model and loss.

Ava：他们还使用了完整的激活检查点，并对模型和损失函数使用 torch.compile。

Brian: Correct. The learning rate was one times ten to the minus four, with 2,000 warmup steps, and the dataset was C4.

Brian：对。学习率是一乘以十的负四次方，预热步数为 2,000，数据集是 C4。

Ava: What were the three configurations?

Ava：他们测试了哪三种配置？

Brian: First, BF16 with standard expert parallel communication. That was the baseline at 651 tokens per second. Second, BF16 with DeepEP. Third, MXFP8 on all grouped GEMMs together with DeepEP.

Brian：第一种是 BF16 搭配标准专家并行通信，作为基线达到了每秒 651 个 token。第二种是 BF16 搭配 DeepEP。第三种是所有 grouped GEMM 都使用 MXFP8，并搭配 DeepEP。

Ava: And DeepEP alone reached 859 tokens per second.

Ava：单独使用 DeepEP 达到了每秒 859 个 token。

Brian: Yes. That’s a 32% improvement over the 651-token baseline. It was the single largest individual gain.

Brian：是的。相对于每秒 651 个 token 的基线，提升了 32%。这是单项优化中最大的收益。

Ava: So communication was the dominant bottleneck at that scale.

Ava：所以在那个规模下，通信是主要瓶颈。

Brian: That’s the interpretation in the article. With EP equal to 32 across 32 nodes, inter-node all-to-all was a major part of step time. DeepEP’s NVLink and RDMA forwarding reduced that cost substantially.

Brian：文章是这样解读的。当 EP 等于 32、横跨 32 个节点时，节点间 all-to-all 占据了单步耗时的很大一部分。DeepEP 的 NVLink 和 RDMA 转发大幅降低了这项开销。

Ava: Then adding MXFP8 grouped GEMMs pushed throughput to 918 tokens per second, for a total gain of 41%.

Ava：随后加入 MXFP8 grouped GEMM，把吞吐量推高到每秒 918 个 token，总提升达到 41%。

Brian: Right. The two optimizations composed well because they target different bottlenecks. DeepEP works on all-to-all communication. MXFP8 works on GEMM computation.

Brian：对。这两项优化结合得很好，因为它们针对不同的瓶颈。DeepEP 优化 all-to-all 通信，MXFP8 优化 GEMM 计算。

Ava: The article says the combined result is close to the sum of the individual improvements.

Ava：文章说，组合后的结果接近各项单独改进效果之和。

Brian: Yes, although the exact individual MXFP8-only number isn’t reported in the summary table. What’s reported is the baseline, the DeepEP result, and the combined result.

Brian：是的，不过汇总表没有报告单独使用 MXFP8 的确切数值。表中报告的是基线、使用 DeepEP 的结果，以及两者结合的结果。

Ava: That’s a useful distinction. We shouldn’t invent a separate MXFP8 speedup.

Ava：这个区分很有用。我们不应该凭空编出一个单独的 MXFP8 加速幅度。

Brian: Exactly. The evidence supports complementarity, not a precise standalone percentage for MXFP8 in this 671B comparison.

Brian：没错。证据表明两者能相互配合，但无法在这次 671B 对比中给出 MXFP8 单独带来的精确百分比。

Ava: Now, speed is only useful if training still behaves correctly. How did they validate that?

Ava：不过，只有训练结果仍然正确，速度提升才有意义。他们是怎么验证这一点的？

Brian: They used the 16B DeepSeek-V3 MoE model and ran BF16 and MXFP8 experiments for 1,500 training steps. These runs used standard EP, without DeepEP.

Brian：他们使用 16B DeepSeek-V3 MoE 模型，分别进行了 BF16 和 MXFP8 实验，各训练了 1,500 步。这些实验使用标准 EP，没有使用 DeepEP。

Ava: What was different in the 16B setup?

Ava：16B 的配置有什么不同？

Brian: The local batch size was 16. TP and PP were both 1, DP was 1, and EP was still 32. They used sequence length 8192, full activation checkpointing, torch.compile for model and loss, the same learning rate of one times ten to the minus four, 2,000 warmup steps, and the C4 dataset.

Brian：本地批量大小为 16。TP 和 PP 都是 1，DP 是 1，EP 仍是 32。他们使用了 8192 的序列长度、完整的激活检查点、用于模型和损失函数的 torch.compile、相同的学习率一乘以十的负四次方、2,000 步预热，以及 C4 数据集。

Ava: And the loss curves?

Ava：损失曲线呢？

Brian: They were virtually identical. The article says MXFP8 showed no meaningful divergence from BF16 over that training window.

Brian：两条曲线几乎完全相同。文章称，在这段训练期间，MXFP8 与 BF16 相比没有出现有意义的偏离。

Ava: So MXFP8 was equivalent to BF16 for convergence in this experiment.

Ava：所以，在这次实验中，MXFP8 与 BF16 的收敛表现相当。

Brian: Yes, over 1,500 steps on the 16B model. That supports the numerical stability of the recipe, while staying within the scope of the test.

Brian：是的，这是在 16B 模型上训练 1,500 步得到的结果。它支持这套训练方案具有数值稳定性，但结论仍限于这次测试的范围。

Ava: Let’s discuss the limitations and what comes next.

Ava：我们来谈谈局限性和接下来的方向。

Brian: The 671B experiment only applied MXFP8 to grouped GEMMs, meaning the routed experts. It couldn’t apply MXFP8 to attention and shared-expert linear layers.

Brian：671B 实验只把 MXFP8 用于分组 GEMM，也就是路由专家。它无法将 MXFP8 用于注意力层和共享专家的线性层。

Ava: Why not?

Ava：为什么不能？

Brian: Because torch.compile didn’t yet support MXFP8 linear layers when combined with tensor parallelism. The 671B configuration required TP equal to 2, so those linear layers were excluded.

Brian：因为当时 torch.compile 还不支持将 MXFP8 线性层与张量并行结合使用。671B 配置要求 TP 等于 2，所以这些线性层没有使用 MXFP8。

Ava: Once TP, torch.compile, and MXFP8 linear-layer support work together in TorchTitan, throughput may improve further.

Ava：等到 TorchTitan 能让 TP、torch.compile 和 MXFP8 线性层支持协同工作，吞吐量可能还会进一步提高。

Brian: That’s the expected next step described by the authors. They also provide open-source recipes and training configurations for reproducibility on a Nebius Cloud B200 cluster.

Brian：这就是作者描述的下一步方向。他们还提供了开源训练方案和配置，供人们在 Nebius Cloud B200 集群上复现实验。

Ava: The infrastructure story is part of the experiment too. Nebius used Soperator, which brings Slurm-style scheduling and multi-node behavior to Kubernetes.

Ava：基础设施也是这次实验的一部分。Nebius 使用了 Soperator，它将 Slurm 风格的调度和多节点运行能力带到了 Kubernetes。

Brian: Yes. It continuously checked GPU and interconnect health, including NCCL all-reduce benchmarks and GPU XID errors. Nodes below performance thresholds could be drained and replaced automatically.

Brian：是的。它持续检查 GPU 和互连的健康状况，包括 NCCL all-reduce 基准测试和 GPU XID 错误。性能低于阈值的节点可以被自动排空并替换。

Ava: And resizing was a single command, with new nodes inheriting the runtime environment.

Ava：而且只需一条命令就能调整规模，新节点会继承运行环境。

Brian: That reduced the operational work around cluster setup, stragglers, and failed jobs, so the team could focus on the training recipe.

Brian：这减少了集群设置、处理拖慢进度的节点和失败任务所需的运维工作，让团队能专注于训练方案。

Ava: Let’s close with the three points I’d write down. First: for the 671B model, DeepEP was the biggest lever, taking throughput from 651 to 859 tokens per second.

Ava：最后说说我会记下的三点。第一：对于 671B 模型，DeepEP 是最重要的提升因素，将吞吐量从每秒 651 个 token 提高到了每秒 859 个 token。

Brian: Second: adding MXFP8 grouped GEMMs to DeepEP reached 918 tokens per second, a 41% total gain, because compute and communication optimizations complemented each other.

Brian：第二：在 DeepEP 的基础上加入 MXFP8 分组 GEMM 后，吞吐量达到每秒 918 个 token，总提升为 41%，因为计算和通信优化相互配合。

Ava: Third: on the 16B model, MXFP8 and BF16 had equivalent loss behavior over 1,500 steps, while broader MXFP8 coverage still depends on TP and torch.compile support.

Ava：第三：在 16B 模型上，MXFP8 和 BF16 在 1,500 步内的损失表现相当，而将 MXFP8 用于更多部分仍取决于 TP 和 torch.compile 的支持。

Brian: That’s the whole story: faster expert communication, faster eligible GEMMs, and convergence that stayed aligned in the tested window.

Brian：事情的全貌就是这样：专家通信更快，适用的 GEMM 运算更快，而在测试期间，收敛表现仍然一致。

Ava: Thanks for listening. We’ll be back with another PyTorch paper or blog post soon.

Ava：感谢收听。我们很快会带来另一篇 PyTorch 论文或博客文章。

## 术语

| Term | 释义 |
|---|---|
| Mixture-of-Experts (MoE) | 混合专家模型 |
| TorchTitan | PyTorch 的大模型预训练框架 |
| MXFP8 | 微缩放 FP8 数值格式 |
| DeepEP | 面向 MoE 专家并行通信优化的库 |
| Grouped GEMM | 分组通用矩阵乘法 |
| Expert Parallelism (EP) | 专家并行 |
| Tensor Parallelism (TP) | 张量并行 |
| All-to-all | 全对全通信操作 |
| NVLink | GPU 节点内高速互连 |
| RDMA | 远程直接内存访问 |
| NVSHMEM | 支持 GPU 间共享寻址和通信的库 |
| GPUDirect Async | 允许 GPU 直接发起网络操作的机制 |
| torch.compile | PyTorch 编译功能 |
| BF16 | Brain Floating Point 16，16 位浮点格式 |
| Activation checkpointing | 激活检查点技术，用计算换显存 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let’s start with the setup. | 我们先从整体设置讲起。 |
| That’s the idea. | 核心就是这个意思。 |
| So computation and communication can both become bottlenecks. | 所以计算和通信都可能成为瓶颈。 |
| That’s a useful distinction. | 这是一个有用的区分。 |
| We shouldn’t invent a separate speedup. | 我们不应该凭空编造一个单独的加速数字。 |
| What comes next? | 接下来会怎样？ |
| Let’s close with the three points. | 最后总结三个要点。 |
| That’s the whole story. | 这就是整个故事。 |
