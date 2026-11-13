# Helion 驱动的 vLLM 量化线性后端

依据：原文

来源：https://pytorch.org/blog/building-a-high-performance-and-portable-vllm-linear-backend-with-helion/

作者将 Helion 接入 vLLM 线性后端，用统一内核、按形状预调优和混合分发优化 NVIDIA Hopper GPU 上的量化矩阵乘法。在所测模型中，端到端吞吐量相对默认后端均有提升，部分工作负载提升超过 10%。

### 问题与范围

vLLM 原有量化线性后端依赖 CUTLASS、DeepGEMM 和 FlashInfer 等专用内核。不同输入形状可能需要不同矩阵乘法算法，手写多个内核及分发规则增加了实现和维护成本。本文聚焦 Hopper 上的 FP8_Dynamic、W8A8_INT8 和 Block_FP8，分别采用逐 token 激活与逐通道权重缩放，以及 1×128 激活与 128×128 权重分块缩放。

### 内核与调优机制

Helion 用一份 GEMM 实现覆盖标准算法、将 K 维分给多个线程块的 Split-K，以及通过转置乘法改善小 M 形状利用率的 Swap-AB。实现把 Split-K 系数和是否交换矩阵设为可调参数，由提前运行的自动调优器连同底层配置一起按输入形状选择。此次用 LLMSeededLFBOTreeSearch 搜索，作者使用 Claude Opus 4.8 提供初始候选，并在 CUDA Graph 条件下测量配置。

### 运行时分发

后端将 max_helion_size 设为 32，只对 CUDA Graph 覆盖范围内的 num_tokens 值 1、2、4、8、16、24、32 分别调优。运行时，小形状在 CUDA Graph 重放中调用 Helion；超过阈值则回退到默认内核。这使调优集中于解码常见的小形状，也避开了 Helion 在 Graph 外分发和启动内核带来的额外 CPU 开销。

### 测试结果

测试使用一张 NVIDIA H100 80GB HBM3，涵盖 Qwen3-1.7B 至 32B 的五种规模及 Qwen3.8-27B，并用 ShareGPT 请求测量服务吞吐量，最大并发为 32，关闭前缀缓存。单独测量内核时，Helion 的几何平均加速比分别为 FP8_Dynamic 对 CUTLASS 的 1.110 倍、W8A8_INT8 对 CUTLASS 的 1.178 倍，以及 Block_FP8 对 FlashInfer 和 DeepGEMM 的 1.149 倍、1.177 倍。各形状收益有差异；端到端测试在所评估模型和量化格式上均显示吞吐量提升，部分工作负载超过 10%。

### 限制与落地

细粒度预调优仍可能耗时数小时；首次启动时的 JIT 编译增加延迟，缓存编译产物可缓解再次启动的开销。大量预调优配置也难以长期维护和充分验证。作者称当前实现及配置已在其 vLLM 分支提供；对于上游集成，团队正探索只维护内核和默认测试配置，让部署者针对自身模型与硬件离线调优。辅助内核的初步实验显示，少至六组配置也能取得有意义的改进。Blackwell 上的 CuteDSL 后端、AMD GPU、TPU 和 MoE 后端仍属后续扩展方向，本文的吞吐量结论不涵盖这些场景。
