# PyTorch 与 TorchTitan 在欧洲公共基础设施上训练一万亿参数 MoE

依据：原文

来源：https://pytorch.org/blog/training-a-1t-parameter-moe-on-eu-public-infrastructure-with-pytorch-and-torchtitan/

EUROPA 联盟在 EuroHPC Frontier AI Grand Challenge 中，使用 PyTorch 和 TorchTitan 于欧洲公共超级计算机 JUPITER 上训练近一万亿参数的 Mixture-of-Experts 模型。团队针对 GH200 节点的内存容量和跨节点带宽限制进行定制优化，使 384 张 GH200 GPU 上的吞吐量从每 GPU 90 tokens/s 提升到约 1,300 tokens/s，并将相关改进回馈到 PyTorch 与 TorchTitan。

### 项目目标与模型

该项目旨在证明共享的欧洲公共计算资源能够支持主权、开源且达到前沿规模的 AI 训练。入选方案要求模型至少包含 400B 参数、支持欧盟 24 种官方语言，并遵守 EU AI Act。实验模型共有 951.9B 参数，每个 token 激活 32.8B 参数，包含 74 层、6,144 隐藏维度、384 个路由专家并采用 top-8 路由，词表大小为 256,000，预训练序列长度为 65,536，优化器使用 Muon 和 AdamW。

### JUPITER 的硬件瓶颈与并行布局

JUPITER 的每个 Booster 节点包含 4 个 NVIDIA GH200，每个 GPU 有 96GB HBM3，并通过 900GB/s 的 NVLink-C2C 连接一颗 72 核 Grace CPU 和 120GB LPDDR5X 内存。节点内 GPU 通过 NVLink 4 互联，节点间通信依赖每 GPU 约 25GB/s 的 InfiniBand，带宽比 GPU 与本地 CPU 的链路低 36 倍。计算节点的主机内存小于部分登录节点，且作业启动时可用容量还可能更低，因此模型、梯度和优化器状态必须共同利用 HBM 与 Grace 内存。团队采用 384 路 FSDP、节点内 4 路专家并行，不使用流水线并行，并以 384 条 65,536 token 序列组成约 2,500 万 token 的大批次。

### 三项关键优化

FlexShard 让每个 rank 按完整专家分片保存参数，使 Muon 的 Newton-Schulz 正交化在专家矩阵上无需跨节点通信；稠密矩阵仍可在传输期间与专家计算重叠。团队还把激活检查点从 Hopper HBM 卸载到 Grace，在反向计算前再传回，从而支持更长序列。针对 FSDP 通信缓冲区被不同 CUDA 流缓存、导致可用 HBM 被碎片化的问题，团队启用 FSDP2 的 unshard_async_op，让 all-gather 设置在计算流上完成，同时让 NCCL 使用内部流。

### 实测收益与运行条件

在 CPU offload 场景中，团队将 Newton-Schulz 从 Grace CPU 移到 GPU，优化器步骤由 41.7 秒降至 2.2 秒，吞吐量从 441 提升到 597 tokens/s/GPU，增幅为 35%。在 32,768 token 序列上，完整激活检查点使吞吐量从 90 提升到 370 tokens/s/GPU；再加入 Grace 激活卸载后达到 508 tokens/s/GPU，增幅为 37%，序列长度随后扩展到 49,152，并在加入 Muon 后达到 65,536。调整 FSDP 流后，384 个 rank 的分配器重试降为零，峰值保留内存从 88.1 GiB 降至 74.6 GiB，吞吐量从 996 提升到 1,210 tokens/s/GPU。最终报告的持续吞吐量约为 1,300 tokens/s/GPU，运行规模为每个 DDP 副本 384 张 GH200、96 个节点，4 个副本合计 1,536 张 GPU。

### 限制、开源贡献与实践意义

这些结果依赖 JUPITER 的 GH200 架构、节点内高速互联、有限的 Grace 内存，以及长序列和大批次带来的计算与通信重叠，不能直接视为所有集群都能达到的性能。MoE token all-to-all 仍难以隐藏，跨节点流量和主机内存容量继续限制扩展；数据与专家负载均衡、InfiniBand 流量削减、节点本地重分片、SonicMoE 动态调度和 NCCL SM 使用限制等优化另有配套说明。团队已向 TorchTitan 加入共享 RoPE 缓存、可配置卸载和种子偏移等功能，并修复 PyTorch 的 CPU 初始化随机状态重置和 DTensor all-gather 问题。文章据此主张，经过针对硬件的细粒度软件优化，公共欧洲基础设施可以承载前沿规模训练。
