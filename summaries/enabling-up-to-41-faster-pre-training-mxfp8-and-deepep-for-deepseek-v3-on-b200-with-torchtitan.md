# TorchTitan 上的 DeepSeek-V3：B200 集群预训练提速

依据：原文

来源：https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/

PyTorch 与 Nebius 在 32 个节点、共 256 块 NVIDIA B200 GPU 上，用 TorchTitan 训练 DeepSeek-V3 的 671B 和 16B MoE 模型。实验分别评估 TorchAO 的 MXFP8 计算和 DeepEP 专家通信，并以 BF16 作为比较基线。

### 问题与方法

MoE 每层需要两次 all-to-all 通信，路由器又会动态决定令牌的去向和传输量，跨节点专家并行因此容易受通信限制。DeepEP 用节点内 NVLink、节点间 InfiniBand RDMA 和 GPU 发起的通信优化令牌分发与结果合并。MXFP8 则为每 32 个元素设置共享缩放因子，借助 B200 的原生支持加速专家层的 grouped GEMM。大矩阵更容易摊薄动态量化开销；矩阵较小时，开销可能抵消收益。

### 671B 吞吐结果

在 C4 数据集、序列长度 8192、局部批量 64、张量并行 TP=2、流水线并行 PP=2、专家并行 EP=32 的配置下，BF16 加标准专家通信达到 651 token/s。保持 BF16、改用 DeepEP 后达到 859 token/s，提升 32%；再对全部 grouped GEMM 使用 MXFP8，达到 918 token/s，相对原基线提升 41%。这三个配置显示两种优化组合后继续增益，但实验没有单独报告 671B 使用 MXFP8、保持标准通信时的吞吐量。

### 收敛验证

16B MoE 实验在同一集群上使用 C4、序列长度 8192、局部批量 16 和 EP=32，所有配置都采用标准专家通信。BF16 与仅对 grouped GEMM 使用 MXFP8 的损失曲线在 1500 个训练步骤内基本重合。这个观察支持该训练窗口内收敛行为相近，不能据此推断更长训练或 671B 模型的最终质量。

### 限制与应用

671B 实验只将 MXFP8 用于路由专家的 grouped GEMM；当时 torch.compile 与 TP 结合时尚不支持 MXFP8 线性层，因此注意力层和共享专家的线性层未覆盖。作者预计支持完善后吞吐量还可提高，但未给出测量值。结果表明，在该 EP=32 的跨节点配置中，优化通信带来的已测收益最大；具体收益仍取决于模型矩阵规模和并行配置。训练脚本与配置已公开，复现需要相应的 B200 集群环境。
