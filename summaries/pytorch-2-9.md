# PyTorch 2.9 发布概览

依据：原文

来源：https://pytorch.org/blog/pytorch-2-9/

PyTorch 于 2025 年 10 月 15 日发布 2.9 版。此次更新涉及 C++/CUDA 扩展的 ABI、多 GPU 编程、torch.compile 图中断控制、安装包变体，以及 Intel、x86 CPU 和 Arm 平台支持。该版本自 2.8 以来包含 452 位贡献者提交的 3216 次代码变更。

### 扩展兼容性与编译控制

稳定 libtorch ABI 新增设备与流工具、torch::stable::Tensor 方法，以及更多 ATen 算子，目标是让扩展用一个 PyTorch 版本构建后仍能在其他版本运行。团队已用这些接口为 Flash-Attention 3 构建 libtorch-ABI wheel，但高层 C++ API 仍处于预览阶段，ABI 范围和版本规则还在完善。torch._dynamo.error_on_graph_break() 则允许开发者在已编译代码的指定区域切换图中断时的报错或继续执行行为。

### 多 GPU 对称内存

新引入的对称内存让 GPU 内核通过 NVLink 或 RDMA 网络直接访问远端内存，并在计算中穿插通信操作。它支持单侧访问和按应用需求定制通信模式。2.9 提供对称张量分配、利用直接访问加速的集合通信、面向 MoE 的 all_to_all_v 变体，以及用于编写多 GPU 内核的 Triton NVSHMEM 插件。相关操作位于 torch.ops.symm_mem；目前列出的后端为 CUDA 和 NVSHMEM。

### 安装包与平台条件

实验性的 wheel variant 支持扩展至 AMD ROCm、Intel XPU 和 NVIDIA CUDA 13；配套插件可检测硬件与软件环境。CUDA wheel 支持 Windows 和 Linux，ROCm 与 XPU 目前仅支持 Linux。此次发布还在所有受支持的 CUDA 版本上提供 Linux aarch64 二进制 wheel 构建。

### 注意力计算与 Arm 优化

Intel GPU 新增 FlexAttention 前向和反向支持。x86 CPU 的 Inductor 后端新增基于 FlexAttention 的 flash decoding，通过划分并归约 KV 序列来增加并行度；博客称它适用于批量、头数或查询序列较小而 KV 序列较长、原有并行度不足的解码场景，但未给出加速数值。Arm 方面，2.9 扩展并优化了 AArch64 的卷积、激活和量化算子，且将 AWS Graviton 4 纳入持续集成测试。博客称 Arm 上三个测试套件的 torch.compile 模式快于 Eager 模式，具体结果需查看其测试面板。
