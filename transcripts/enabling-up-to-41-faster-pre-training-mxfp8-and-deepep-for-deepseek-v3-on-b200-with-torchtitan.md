# Why DeepEP and MXFP8 Make DeepSeek-V3 Training 41% Faster

原文：[Enabling Up to 41% Faster Pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200 with TorchTitan](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/)

## 摘要

本期讨论 PyTorch 与 Nebius 如何在 256 张 NVIDIA B200 GPU 上，用 TorchTitan 训练 DeepSeek-V3 的 671B 和 16B MoE 模型。实验比较了 BF16 基线、DeepEP 通信优化和 MXFP8 grouped GEMM 计算优化：671B 模型从每秒 651 个 token 提升到 918 个，整体快了 41%。其中 DeepEP 单独带来 32% 提升，说明跨节点 all-to-all 通信是主要瓶颈。对 16B 模型进行 1500 步训练后，MXFP8 与 BF16 的 loss 曲线几乎一致，显示没有明显收敛损失。

## 对话

Ava: What if you could train a 671B-parameter Mixture-of-Experts model 41% faster, using open-source PyTorch tools?

Ava：如果用开源 PyTorch 工具，能把一个 671B 参数的专家混合模型训练速度提高 41%，会怎样？

That’s the result we’re unpacking today.

这就是我们今天要聊的成果。

Brian: And the interesting part is that the gain comes from two different places: MXFP8 speeds up matrix multiplication, while DeepEP speeds up expert communication.

Brian：有意思的是，提速来自两个方面：MXFP8 加快矩阵乘法，DeepEP 加快专家间通信。

Together, they reached 918 tokens per second on 256 NVIDIA B200 GPUs.

两者结合后，在 256 块 NVIDIA B200 GPU 上达到了每秒 918 个 token。

Ava: Let’s start with the setup. This was a joint PyTorch and Nebius experiment, right?

Ava：先说实验配置。这是 PyTorch 和 Nebius 联合做的实验，对吧？

Brian: Right. They used TorchTitan as the pre-training framework on a Nebius Cloud cluster.

Brian：对。他们在 Nebius Cloud 集群上，用 TorchTitan 作为预训练框架。

The cluster had 32 nodes, eight B200 GPUs per node, so 256 GPUs in total.

集群有 32 个节点，每个节点 8 块 B200 GPU，总共 256 块。

Ava: And the model was DeepSeek-V3, with both a 671B version and a 16B MoE version.

Ava：模型是 DeepSeek-V3，分别用了 671B 版本和 16B MoE 版本。

Brian: Exactly. The 671B experiment measured throughput.

Brian：没错。671B 实验测的是吞吐量。

The 16B experiment checked whether MXFP8 changes training convergence.

16B 实验检验 MXFP8 是否影响训练收敛。

Ava: Why focus on Mixture-of-Experts models? What makes them difficult to train?

Ava：为什么聚焦专家混合模型？它们难训在哪里？

Brian: They have two major costs. First, the experts do a lot of matrix multiplication.

Brian：主要有两项开销。首先，专家需要做大量矩阵乘法。

Second, every layer needs two all-to-all exchanges: one to dispatch tokens to experts, and another to combine the results.

其次，每一层都要进行两次全对全通信：一次把 token 分发给专家，一次汇总结果。

Ava: So computation and communication can both become bottlenecks.

Ava：所以计算和通信都可能成为瓶颈。

Brian: Yes. And the communication pattern is especially awkward.

Brian：对。而且它的通信模式尤其棘手。

The router decides dynamically which tokens go to which experts.

路由器会动态决定把哪些 token 送给哪些专家。

That means the transfer sizes and destinations change at every step.

这意味着每一步的传输量和目标位置都会变化。

Ava: Which is a bad match for a generic all-to-all operation designed around fixed transfers.

Ava：这就不太适合按固定传输设计的通用全对全操作。

Brian: That’s the idea.

Brian：正是如此。

As expert parallelism, or EP, grows, this communication cost gets worse.

随着专家并行，也就是 EP，的规模扩大，通信开销会更高。

In the 671B run, EP was 32 across 32 nodes, so inter-node traffic mattered a lot.

671B 实验在 32 个节点上采用 EP=32，因此跨节点流量影响很大。

Ava: Okay. Let’s talk about the first optimization: MXFP8. What does the name mean?

Ava：好。先聊第一项优化：MXFP8。这个名字是什么意思？

Brian: MXFP8 means Microscaling FP8.

Brian：MXFP8 是微缩放 FP8。

It’s a low-precision format defined by the OCP Microscaling Specification.

它是 OCP 微缩放规范定义的一种低精度格式。

Standard float8 often uses one scale factor per tensor or per row.

标准 float8 通常每个张量或每一行共用一个缩放因子。

MXFP8 uses a shared exponent, called an E8M0 scale, for every block of 32 elements.

MXFP8 则每 32 个元素共用一个指数，称为 E8M0 缩放值。

Ava: So the scaling is more local.

Ava：也就是说，缩放范围更局部。

Brian: Right.

Brian：对。

That finer-grained scaling helps preserve numerical fidelity while still using FP8 tensor cores.

这种更细粒度的缩放有助于保持数值精度，同时仍能使用 FP8 张量核心。

NVIDIA Blackwell supports MXFP8 natively through its tcgen05.

NVIDIA Blackwell 通过 tcgen05.

mma tensor core instructions.

mma 张量核心指令原生支持 MXFP8。

Ava: That’s why B200 is a natural target. There’s no emulation overhead for MXFP8 GEMMs.

Ava：所以 B200 很适合跑它。MXFP8 GEMM 不需要模拟，因而没有这方面的开销。

Brian: Yes. Blackwell can provide up to two times the peak TFLOPS of BF16 for eligible GEMMs.

Brian：对。对于适用的 GEMM，Blackwell 的峰值 TFLOPS 最高可达 BF16 的两倍。

But the article measures end-to-end training, so the real speedup depends on overhead and on which operations are eligible.

但文章测的是端到端训练，实际提速取决于额外开销，以及哪些操作能用 MXFP8。

Ava: How did TorchTitan apply it?

Ava：TorchTitan 是怎么用上它的？

Brian: Through TorchAO.

Brian：通过 TorchAO。

For linear layers, TorchAO dynamically quantizes inputs to MXFP8 for the forward output, input gradient, and weight gradient GEMMs.

在线性层中，TorchAO 会为前向输出、输入梯度和权重梯度这三类 GEMM，将输入动态量化为 MXFP8。

The results accumulate back to BF16.

计算结果则累加为 BF16。

Ava: And for the experts?

Ava：专家层呢？

Brian: For routed experts, it converts torch grouped matrix multiplication operations to a TorchAO grouped GEMM function.

Brian：对于路由专家，它把 torch 的分组矩阵乘法操作转换为 TorchAO 的分组 GEMM 函数。

The same three GEMMs use dynamically quantized MXFP8 inputs.

这三类 GEMM 同样使用动态量化为 MXFP8 的输入。

Ava: The article says grouped GEMMs dominate MoE expert layers.

Ava：文章说，分组 GEMM 是 MoE 专家层的主要开销。

Brian: Exactly. The recipe quantizes inputs immediately before each grouped GEMM.

Brian：没错。这套方案会在每次分组 GEMM 前立即量化输入。

When the GEMM dimensions are large, the compute cost grows faster than the quantization cost, so the overhead becomes relatively small.

GEMM 维度较大时，计算量比量化开销增长得更快，所以量化的相对开销会变小。

Ava: But for small dimensions, quantization can cancel the benefit.

Ava：但维度较小时，量化开销可能抵消收益。

Brian: Yes. The article is clear about that limitation.

Brian：对。文章明确指出了这个局限。

In smaller MoE models, the quantization overhead can match or exceed the MXFP8 speedup, leading to neutral or even negative performance.

在较小的 MoE 模型中，量化开销可能等于甚至超过 MXFP8 带来的提速，最终性能持平，甚至下降。

Ava: Now DeepEP sounds like a very different kind of optimization.

Ava：DeepEP 听起来是另一种优化。

Brian: It is.

Brian：是的。

DeepEP is a GPU communication library developed by DeepSeek specifically for MoE expert-parallel dispatch and combine operations.

DeepEP 是 DeepSeek 专为 MoE 专家并行中的 token 分发和结果汇总开发的 GPU 通信库。

Ava: What does it change compared with standard all-to-all?

Ava：与标准的全对全通信相比，它改变了什么？

Brian: It uses purpose-built NVLink and RDMA kernels.

Brian：它使用专门设计的 NVLink 和 RDMA 内核。

Inside a node, data moves over high-bandwidth NVLink.

节点内，数据通过高带宽 NVLink 传输。

Between nodes, it uses RDMA over InfiniBand.

节点间，则通过 InfiniBand 使用 RDMA。

This hierarchical path matches the hardware’s bandwidth differences.

这种分层传输方式适应了硬件带宽的差异。

Ava: The article mentions around 900 gigabytes per second for intra-node NVLink.

Ava：文章提到，节点内 NVLink 的带宽约为每秒 900 GB。

Brian: Yes, that’s the approximate figure given for the setup.

Brian：对，这是该配置给出的近似值。

DeepEP also uses NVSHMEM, which provides a Partitioned Global Address Space.

DeepEP 还使用 NVSHMEM，它提供分区全局地址空间。

Each GPU can map a symmetric memory region that peers can address.

每个 GPU 都能映射一块其他 GPU 可寻址的对称内存区域。

Ava: And with InfiniBand GPUDirect Async, the GPU can drive network operations directly from CUDA kernels.

Ava：借助 InfiniBand GPUDirect Async，GPU 还能直接从 CUDA 内核发起网络操作。

Brian: Right. That removes CPU involvement, which is useful for small, dynamic transfers.

Brian：没错。这省去了 CPU 的参与，尤其适合小规模、动态的数据传输。

DeepEP also fuses metadata such as token embeddings, expert indices, and routing weights into one communication operation.

DeepEP 还将词元嵌入、专家索引和路由权重等信息合并到一次通信操作中。

Ava: Fewer launches and fewer synchronization points.

Ava：这样启动次数和同步点都更少。

Brian: Exactly.

Brian：正是如此。

It can also be configured to use a chosen number of streaming multiprocessors, leaving the remaining SMs available for overlapping computation.

它还可以配置为只使用指定数量的流式多处理器，让剩余的 SM 用于并行计算。

Ava: Before the results, tell me about the actual 671B training configuration.

Ava：讲结果之前，先说说 671B 模型的实际训练配置。

Brian: The sequence length was 8192, with a local batch size of 64.

Brian：序列长度为 8192，本地批量大小为 64。

Tensor Parallel, or TP, was 2. Pipeline Parallel, PP, was 2. Data Parallel, DP, was 1.

张量并行 TP 为 2，流水线并行 PP 为 2，数据并行 DP 为 1。

Expert Parallel was 32, and Expert Tensor Parallel was 1.

专家并行 EP 为 32，专家张量并行为 1。

Ava: They also used full activation checkpointing and torch. compile for the model and loss.

Ava：他们还使用了完整激活检查点，并用 torch.compile 编译模型和损失函数。

Brian: Correct.

Brian：没错。

The learning rate was one times ten to the minus four, with 2,000 warmup steps, and the dataset was C4.

学习率为 1×10⁻⁴，预热 2,000 步，数据集为 C4。

Ava: What were the three configurations?

Ava：三种配置分别是什么？

Brian: First, BF16 with standard expert parallel communication.

Brian：第一种是 BF16，使用标准的专家并行通信。

That was the baseline at 651 tokens per second. Second, BF16 with DeepEP.

这是基线，每秒处理 651 个词元。第二种是 BF16 加 DeepEP。

Third, MXFP8 on all grouped GEMMs together with DeepEP.

第三种是所有分组 GEMM 都使用 MXFP8，并搭配 DeepEP。

Ava: And DeepEP alone reached 859 tokens per second.

Ava：单用 DeepEP 就达到了每秒 859 个词元。

Brian: Yes. That’s a 32% improvement over the 651-token baseline.

Brian：对，比每秒 651 个词元的基线提升了 32%。

It was the single largest individual gain.

这是单项优化中最大的提升。

Ava: So communication was the dominant bottleneck at that scale.

Ava：所以在这个规模下，通信是主要瓶颈。

Brian: That’s the interpretation in the article.

Brian：文章是这样解读的。

With EP equal to 32 across 32 nodes, inter-node all-to-all was a major part of step time.

EP 为 32、跨越 32 个节点时，节点间全对全通信占了单步训练时间的很大一部分。

DeepEP’s NVLink and RDMA forwarding reduced that cost substantially.

DeepEP 通过 NVLink 和 RDMA 转发，大幅降低了这部分开销。

Ava: Then adding MXFP8 grouped GEMMs pushed throughput to 918 tokens per second, for a total gain of 41%.

Ava：再加上 MXFP8 分组 GEMM，吞吐量达到每秒 918 个词元，总提升为 41%。

Brian: Right. The two optimizations composed well because they target different bottlenecks.

Brian：对。两项优化配合得很好，因为它们针对不同的瓶颈。

DeepEP works on all-to-all communication. MXFP8 works on GEMM computation.

DeepEP 优化全对全通信，MXFP8 优化 GEMM 计算。

Ava: The article says the combined result is close to the sum of the individual improvements.

Ava：文章说，组合后的提升接近两项单独提升之和。

Brian: Yes, although the exact individual MXFP8-only number isn’t reported in the summary table.

Brian：是的，不过汇总表没有报告单用 MXFP8 的确切结果。

What’s reported is the baseline, the DeepEP result, and the combined result.

表中报告的是基线、使用 DeepEP 的结果，以及组合使用的结果。

Ava: That’s a useful distinction. We shouldn’t invent a separate MXFP8 speedup.

Ava：这个区别很重要。我们不能凭空给出 MXFP8 单独带来的加速幅度。

Brian: Exactly.

Brian：没错。

The evidence supports complementarity, not a precise standalone percentage for MXFP8 in this 671B comparison.

这些数据能说明两项优化互补，却无法得出 MXFP8 在这次 671B 对比中单独提升的准确百分比。

Ava: Now, speed is only useful if training still behaves correctly.

Ava：速度提升的前提是训练仍然正常。

How did they validate that?

他们是怎么验证的？

Brian: They used the 16B DeepSeek-V3 MoE model and ran BF16 and MXFP8 experiments for 1,500 training steps.

Brian：他们用 16B 的 DeepSeek-V3 MoE 模型，分别进行了 1,500 步的 BF16 和 MXFP8 训练实验。

These runs used standard EP, without DeepEP.

这些实验使用标准 EP，没有使用 DeepEP。

Ava: What was different in the 16B setup?

Ava：16B 的配置有什么不同？

Brian: The local batch size was 16. TP and PP were both 1, DP was 1, and EP was still 32.

Brian：本地批量大小为 16。TP 和 PP 都为 1，DP 为 1，EP 仍为 32。

They used sequence length 8192, full activation checkpointing, torch.

序列长度为 8192，使用完整激活检查点，并用 torch.

compile for model and loss, the same learning rate of one times ten to the minus four, 2,000 warmup steps, and the C4 dataset.

compile 编译模型和损失函数；学习率同为 1×10⁻⁴，预热 2,000 步，数据集为 C4。

Ava: And the loss curves?

Ava：损失曲线呢？

Brian: They were virtually identical.

Brian：几乎完全一致。

The article says MXFP8 showed no meaningful divergence from BF16 over that training window.

文章称，在这段训练期间，MXFP8 与 BF16 没有出现明显偏离。

Ava: So MXFP8 was equivalent to BF16 for convergence in this experiment.

Ava：也就是说，在这次实验中，MXFP8 和 BF16 的收敛表现相当。

Brian: Yes, over 1,500 steps on the 16B model.

Brian：对，在 16B 模型上训练了 1,500 步。

That supports the numerical stability of the recipe, while staying within the scope of the test.

这说明该训练方案在测试范围内具有数值稳定性。

Ava: Let’s discuss the limitations and what comes next.

Ava：聊聊局限和下一步吧。

Brian: The 671B experiment only applied MXFP8 to grouped GEMMs, meaning the routed experts.

Brian：671B 模型的实验只对分组 GEMM，也就是路由专家，使用了 MXFP8。

It couldn’t apply MXFP8 to attention and shared-expert linear layers.

注意力层和共享专家的线性层还无法使用 MXFP8。

Ava: Why not?

Ava：为什么？

Brian: Because torch.

Brian：因为 torch。

compile didn’t yet support MXFP8 linear layers when combined with tensor parallelism.

compile 当时还不支持与张量并行结合使用的 MXFP8 线性层。

The 671B configuration required TP equal to 2, so those linear layers were excluded.

671B 模型的配置要求 TP 为 2，因此这些线性层被排除在外。

Ava: Once TP, torch.

Ava：等 TP、torch。

compile, and MXFP8 linear-layer support work together in TorchTitan, throughput may improve further.

compile 和 MXFP8 线性层支持能在 TorchTitan 中协同工作，吞吐量可能进一步提升。

Brian: That’s the expected next step described by the authors.

Brian：这正是作者提出的下一步。

They also provide open-source recipes and training configurations for reproducibility on a Nebius Cloud B200 cluster.

他们还公开了训练方案和配置，方便在 Nebius Cloud 的 B200 集群上复现。

Ava: The infrastructure story is part of the experiment too.

Ava：基础设施也是这次实验的一部分。

Nebius used Soperator, which brings Slurm-style scheduling and multi-node behavior to Kubernetes.

Nebius 使用 Soperator，让 Kubernetes 具备 Slurm 式调度和多节点运行能力。

Brian: Yes.

Brian：没错。

It continuously checked GPU and interconnect health, including NCCL all-reduce benchmarks and GPU XID errors.

它持续检查 GPU 和互连的健康状况，包括 NCCL all-reduce 基准测试和 GPU XID 错误。

Nodes below performance thresholds could be drained and replaced automatically.

性能低于阈值的节点可以自动退出运行并被替换。

Ava: And resizing was a single command, with new nodes inheriting the runtime environment.

Ava：扩容也只需一条命令，新节点会继承运行环境。

Brian: That reduced the operational work around cluster setup, stragglers, and failed jobs, so the team could focus on the training recipe.

Brian：这减少了集群搭建、慢节点和任务失败带来的运维工作，让团队能专注于训练方案。

Ava: Let’s close with the three points I’d write down.

Ava：最后总结我会记下的三点。

First: for the 671B model, DeepEP was the biggest lever, taking throughput from 651 to 859 tokens per second.

第一，对 671B 模型来说，DeepEP 的作用最大，将吞吐量从每秒 651 个 token 提升到 859 个。

Brian: Second: adding MXFP8 grouped GEMMs to DeepEP reached 918 tokens per second, a 41% total gain, because compute and communication optimizations complemented each other.

Brian：第二，在 DeepEP 基础上加入 MXFP8 分组 GEMM 后，吞吐量达到每秒 918 个 token，总体提升 41%，因为计算和通信优化相互配合。

Ava: Third: on the 16B model, MXFP8 and BF16 had equivalent loss behavior over 1,500 steps, while broader MXFP8 coverage still depends on TP and torch.

Ava：第三，在 16B 模型上，MXFP8 和 BF16 训练 1,500 步的损失表现相当；但要将 MXFP8 用于更多层，仍取决于 TP 和 torch。

compile support.

compile 的支持。

Brian: That’s the whole story: faster expert communication, faster eligible GEMMs, and convergence that stayed aligned in the tested window.

Brian：这就是全部：专家通信更快，适用的 GEMM 更快，而且在测试期间收敛表现保持一致。

Ava: Thanks for listening. We’ll be back with another PyTorch paper or blog post soon.

Ava：感谢收听。我们很快会带来另一篇 PyTorch 论文或博文。

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
