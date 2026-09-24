# Why DeepEP and MXFP8 Make DeepSeek-V3 Training 41% Faster

原文：[Enabling Up to 41% Faster Pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200 with TorchTitan](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/)

## 摘要

本期讨论 PyTorch 与 Nebius 如何在 256 张 NVIDIA B200 GPU 上，用 TorchTitan 训练 DeepSeek-V3 的 671B 和 16B MoE 模型。实验比较了 BF16 基线、DeepEP 通信优化和 MXFP8 grouped GEMM 计算优化：671B 模型从每秒 651 个 token 提升到 918 个，整体快了 41%。其中 DeepEP 单独带来 32% 提升，说明跨节点 all-to-all 通信是主要瓶颈。对 16B 模型进行 1500 步训练后，MXFP8 与 BF16 的 loss 曲线几乎一致，显示没有明显收敛损失。

## 对话

**Ava:** What if you could train a 671B-parameter Mixture-of-Experts model 41% faster, using open-source PyTorch tools? That’s the result we’re unpacking today.

**Brian:** And the interesting part is that the gain comes from two different places: MXFP8 speeds up matrix multiplication, while DeepEP speeds up expert communication. Together, they reached 918 tokens per second on 256 NVIDIA B200 GPUs.

**Ava:** Let’s start with the setup. This was a joint PyTorch and Nebius experiment, right?

**Brian:** Right. They used TorchTitan as the pre-training framework on a Nebius Cloud cluster. The cluster had 32 nodes, eight B200 GPUs per node, so 256 GPUs in total.

**Ava:** And the model was DeepSeek-V3, with both a 671B version and a 16B MoE version.

**Brian:** Exactly. The 671B experiment measured throughput. The 16B experiment checked whether MXFP8 changes training convergence.

**Ava:** Why focus on Mixture-of-Experts models? What makes them difficult to train?

**Brian:** They have two major costs. First, the experts do a lot of matrix multiplication. Second, every layer needs two all-to-all exchanges: one to dispatch tokens to experts, and another to combine the results.

**Ava:** So computation and communication can both become bottlenecks.

**Brian:** Yes. And the communication pattern is especially awkward. The router decides dynamically which tokens go to which experts. That means the transfer sizes and destinations change at every step.

**Ava:** Which is a bad match for a generic all-to-all operation designed around fixed transfers.

**Brian:** That’s the idea. As expert parallelism, or EP, grows, this communication cost gets worse. In the 671B run, EP was 32 across 32 nodes, so inter-node traffic mattered a lot.

**Ava:** Okay. Let’s talk about the first optimization: MXFP8. What does the name mean?

**Brian:** MXFP8 means Microscaling FP8. It’s a low-precision format defined by the OCP Microscaling Specification. Standard float8 often uses one scale factor per tensor or per row. MXFP8 uses a shared exponent, called an E8M0 scale, for every block of 32 elements.

**Ava:** So the scaling is more local.

**Brian:** Right. That finer-grained scaling helps preserve numerical fidelity while still using FP8 tensor cores. NVIDIA Blackwell supports MXFP8 natively through its tcgen05.mma tensor core instructions.

**Ava:** That’s why B200 is a natural target. There’s no emulation overhead for MXFP8 GEMMs.

**Brian:** Yes. Blackwell can provide up to two times the peak TFLOPS of BF16 for eligible GEMMs. But the article measures end-to-end training, so the real speedup depends on overhead and on which operations are eligible.

**Ava:** How did TorchTitan apply it?

**Brian:** Through TorchAO. For linear layers, TorchAO dynamically quantizes inputs to MXFP8 for the forward output, input gradient, and weight gradient GEMMs. The results accumulate back to BF16.

**Ava:** And for the experts?

**Brian:** For routed experts, it converts torch grouped matrix multiplication operations to a TorchAO grouped GEMM function. The same three GEMMs use dynamically quantized MXFP8 inputs.

**Ava:** The article says grouped GEMMs dominate MoE expert layers.

**Brian:** Exactly. The recipe quantizes inputs immediately before each grouped GEMM. When the GEMM dimensions are large, the compute cost grows faster than the quantization cost, so the overhead becomes relatively small.

**Ava:** But for small dimensions, quantization can cancel the benefit.

**Brian:** Yes. The article is clear about that limitation. In smaller MoE models, the quantization overhead can match or exceed the MXFP8 speedup, leading to neutral or even negative performance.

**Ava:** Now DeepEP sounds like a very different kind of optimization.

**Brian:** It is. DeepEP is a GPU communication library developed by DeepSeek specifically for MoE expert-parallel dispatch and combine operations.

**Ava:** What does it change compared with standard all-to-all?

**Brian:** It uses purpose-built NVLink and RDMA kernels. Inside a node, data moves over high-bandwidth NVLink. Between nodes, it uses RDMA over InfiniBand. This hierarchical path matches the hardware’s bandwidth differences.

**Ava:** The article mentions around 900 gigabytes per second for intra-node NVLink.

**Brian:** Yes, that’s the approximate figure given for the setup. DeepEP also uses NVSHMEM, which provides a Partitioned Global Address Space. Each GPU can map a symmetric memory region that peers can address.

**Ava:** And with InfiniBand GPUDirect Async, the GPU can drive network operations directly from CUDA kernels.

**Brian:** Right. That removes CPU involvement, which is useful for small, dynamic transfers. DeepEP also fuses metadata such as token embeddings, expert indices, and routing weights into one communication operation.

**Ava:** Fewer launches and fewer synchronization points.

**Brian:** Exactly. It can also be configured to use a chosen number of streaming multiprocessors, leaving the remaining SMs available for overlapping computation.

**Ava:** Before the results, tell me about the actual 671B training configuration.

**Brian:** The sequence length was 8192, with a local batch size of 64. Tensor Parallel, or TP, was 2. Pipeline Parallel, PP, was 2. Data Parallel, DP, was 1. Expert Parallel was 32, and Expert Tensor Parallel was 1.

**Ava:** They also used full activation checkpointing and torch.compile for the model and loss.

**Brian:** Correct. The learning rate was one times ten to the minus four, with 2,000 warmup steps, and the dataset was C4.

**Ava:** What were the three configurations?

**Brian:** First, BF16 with standard expert parallel communication. That was the baseline at 651 tokens per second. Second, BF16 with DeepEP. Third, MXFP8 on all grouped GEMMs together with DeepEP.

**Ava:** And DeepEP alone reached 859 tokens per second.

**Brian:** Yes. That’s a 32% improvement over the 651-token baseline. It was the single largest individual gain.

**Ava:** So communication was the dominant bottleneck at that scale.

**Brian:** That’s the interpretation in the article. With EP equal to 32 across 32 nodes, inter-node all-to-all was a major part of step time. DeepEP’s NVLink and RDMA forwarding reduced that cost substantially.

**Ava:** Then adding MXFP8 grouped GEMMs pushed throughput to 918 tokens per second, for a total gain of 41%.

**Brian:** Right. The two optimizations composed well because they target different bottlenecks. DeepEP works on all-to-all communication. MXFP8 works on GEMM computation.

**Ava:** The article says the combined result is close to the sum of the individual improvements.

**Brian:** Yes, although the exact individual MXFP8-only number isn’t reported in the summary table. What’s reported is the baseline, the DeepEP result, and the combined result.

**Ava:** That’s a useful distinction. We shouldn’t invent a separate MXFP8 speedup.

**Brian:** Exactly. The evidence supports complementarity, not a precise standalone percentage for MXFP8 in this 671B comparison.

**Ava:** Now, speed is only useful if training still behaves correctly. How did they validate that?

**Brian:** They used the 16B DeepSeek-V3 MoE model and ran BF16 and MXFP8 experiments for 1,500 training steps. These runs used standard EP, without DeepEP.

**Ava:** What was different in the 16B setup?

**Brian:** The local batch size was 16. TP and PP were both 1, DP was 1, and EP was still 32. They used sequence length 8192, full activation checkpointing, torch.compile for model and loss, the same learning rate of one times ten to the minus four, 2,000 warmup steps, and the C4 dataset.

**Ava:** And the loss curves?

**Brian:** They were virtually identical. The article says MXFP8 showed no meaningful divergence from BF16 over that training window.

**Ava:** So MXFP8 was equivalent to BF16 for convergence in this experiment.

**Brian:** Yes, over 1,500 steps on the 16B model. That supports the numerical stability of the recipe, while staying within the scope of the test.

**Ava:** Let’s discuss the limitations and what comes next.

**Brian:** The 671B experiment only applied MXFP8 to grouped GEMMs, meaning the routed experts. It couldn’t apply MXFP8 to attention and shared-expert linear layers.

**Ava:** Why not?

**Brian:** Because torch.compile didn’t yet support MXFP8 linear layers when combined with tensor parallelism. The 671B configuration required TP equal to 2, so those linear layers were excluded.

**Ava:** Once TP, torch.compile, and MXFP8 linear-layer support work together in TorchTitan, throughput may improve further.

**Brian:** That’s the expected next step described by the authors. They also provide open-source recipes and training configurations for reproducibility on a Nebius Cloud B200 cluster.

**Ava:** The infrastructure story is part of the experiment too. Nebius used Soperator, which brings Slurm-style scheduling and multi-node behavior to Kubernetes.

**Brian:** Yes. It continuously checked GPU and interconnect health, including NCCL all-reduce benchmarks and GPU XID errors. Nodes below performance thresholds could be drained and replaced automatically.

**Ava:** And resizing was a single command, with new nodes inheriting the runtime environment.

**Brian:** That reduced the operational work around cluster setup, stragglers, and failed jobs, so the team could focus on the training recipe.

**Ava:** Let’s close with the three points I’d write down. First: for the 671B model, DeepEP was the biggest lever, taking throughput from 651 to 859 tokens per second.

**Brian:** Second: adding MXFP8 grouped GEMMs to DeepEP reached 918 tokens per second, a 41% total gain, because compute and communication optimizations complemented each other.

**Ava:** Third: on the 16B model, MXFP8 and BF16 had equivalent loss behavior over 1,500 steps, while broader MXFP8 coverage still depends on TP and torch.compile support.

**Brian:** That’s the whole story: faster expert communication, faster eligible GEMMs, and convergence that stayed aligned in the tested window.

**Ava:** Thanks for listening. We’ll be back with another PyTorch paper or blog post soon.

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
